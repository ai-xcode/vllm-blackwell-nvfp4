#!/usr/bin/env python3
"""
Phase 1: Convert a HuggingFace BF16 model to NVFP4 via NVIDIA modelopt.

Hard-coded to handle Qwen-family models for the first cut. Will generalize
once this works end-to-end on Qwen3-4B.

Pre-flight checks (fail fast, fail loud):
  * source dir exists and has a config.json with torch_dtype != quantized
  * output dir does NOT exist (refuse to overwrite)
  * enough free GPU mem to load the model in BF16
  * enough free disk for output

Pipeline:
  1. Load tokenizer + BF16 model from `--source`
  2. Run calibration (256 samples from c4 by default — small, fast for 4B)
  3. Quantize to NVFP4 via modelopt.torch.quantization
  4. Save quantized model to `--output` in HF format
  5. Print loadable-by-vLLM verification command

Usage:
  python convert_to_nvfp4.py \
      --source ~/vLLM_Servers/models_awq/Qwen3-4B \
      --output ~/vLLM_Servers/models_awq/Qwen3-4B-NVFP4 \
      --calib-samples 256

Environment:
  Must be run inside the dedicated venv at ~/nvfp4_conversion/venv/
  with nvidia-modelopt[torch,hf] installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# ── logging ───────────────────────────────────────────────────────────────────
LOG = logging.getLogger("nvfp4")
LOG.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%H:%M:%S"))
LOG.addHandler(_h)


def fail(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    LOG.error(msg)
    sys.exit(code)


# ── pre-flight ────────────────────────────────────────────────────────────────
def preflight_source(src: Path) -> dict:
    if not src.exists():
        fail(f"source path does not exist: {src}")
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        fail(f"no config.json in source: {src}")
    cfg = json.loads(cfg_path.read_text())

    quant = cfg.get("quantization_config")
    if quant is not None:
        method = quant.get("quant_method", "?")
        fail(f"source is already quantized ({method}). NVFP4 conversion "
             f"requires an unquantized BF16/FP16 source.")

    # Newer transformers configs use "dtype"; older ones use "torch_dtype".
    # Vision-language models nest the LM config inside "text_config".
    dtype = (
        cfg.get("dtype")
        or cfg.get("torch_dtype")
        or (cfg.get("text_config") or {}).get("dtype")
        or (cfg.get("text_config") or {}).get("torch_dtype")
        or "?"
    )
    if dtype not in ("bfloat16", "float16"):
        fail(f"source dtype is {dtype!r}; expected bfloat16/float16.")

    arch_list = cfg.get("architectures", [])
    if not arch_list:
        fail("config.json has no 'architectures' field.")
    arch = arch_list[0]

    # Architectures we've actually tested. Not a hard gate — just a warning.
    # modelopt supports many more; we let it try and bubble up its own error
    # if the layer types aren't supported.
    TESTED = ("Qwen2", "Qwen3", "Qwen2_5", "Llama", "Mistral", "Mixtral",
              "Phi", "Gemma", "DeepseekV2", "DeepseekV3", "OLMoE")
    if not any(arch.startswith(p) for p in TESTED):
        LOG.warning(f"arch {arch!r} is NOT in the tested list {TESTED}.")
        LOG.warning(f"continuing anyway — modelopt will report a clear error "
                    f"if any layer type isn't supported.")
    LOG.info(f"source OK · arch={arch} · dtype={dtype}")
    return cfg


def preflight_output(out: Path) -> None:
    if out.exists():
        # refuse to overwrite — be loud about it. user can rm if they really mean it.
        fail(f"output already exists: {out}\n"
             f"refusing to overwrite. delete it manually if you want to redo.")
    out.parent.mkdir(parents=True, exist_ok=True)


def preflight_disk(out: Path, source_bytes: int) -> None:
    # NVFP4 is ~1/4 of BF16, but we need headroom during write
    free = shutil.disk_usage(out.parent).free
    needed = max(source_bytes // 2, 10 * 1024 ** 3)  # ~half source size, min 10 GB
    if free < needed:
        fail(f"insufficient free disk: have {free/1e9:.1f} GB, "
             f"need ~{needed/1e9:.1f} GB. free up space first.")
    LOG.info(f"disk OK · {free/1e9:.1f} GB free")


def preflight_gpu(model_bytes: int, multi_gpu: bool) -> None:
    import torch
    if not torch.cuda.is_available():
        fail("no CUDA device visible to torch.")
    needed = int(model_bytes * 1.2)
    if multi_gpu:
        n = torch.cuda.device_count()
        total_free = 0
        for i in range(n):
            free, _ = torch.cuda.mem_get_info(i)
            total_free += free
            LOG.info(f"GPU{i} mem · free={free/1e9:.1f} GB")
        LOG.info(f"  ·  total free across {n} GPUs: {total_free/1e9:.1f} GB · "
                 f"model needs ~{needed/1e9:.1f} GB")
        if total_free < needed:
            fail(f"insufficient combined free GPU memory.")
    else:
        free, total = torch.cuda.mem_get_info(0)
        LOG.info(f"GPU0 mem · free={free/1e9:.1f} GB / total={total/1e9:.1f} GB · "
                 f"model needs ~{needed/1e9:.1f} GB")
        if free < needed:
            fail(f"GPU0 has {free/1e9:.1f} GB free, model needs ~{needed/1e9:.1f}. "
                 f"stop other GPU workloads (ai-stop.sh, kill stale vLLM) and retry. "
                 f"For models too big for one GPU, pass --device auto.")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path,
                   help="Path to local BF16/FP16 HF model directory")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the NVFP4 model (must not exist)")
    p.add_argument("--calib-samples", type=int, default=256,
                   help="Calibration samples (default 256; bigger = slower + more accurate)")
    p.add_argument("--calib-dataset", default="cnn_dailymail",
                   help="HF dataset id for calibration (default cnn_dailymail)")
    p.add_argument("--device", default="cuda:0",
                   help="Device for load+calib. Use 'auto' for models too big for one GPU.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Also log to file (defaults to <output>/conversion.log after preflight)")
    args = p.parse_args()

    src: Path = args.source.expanduser().resolve()
    out: Path = args.output.expanduser().resolve()

    # ── pre-flight ───
    LOG.info(f"=== pre-flight ===")
    cfg = preflight_source(src)
    preflight_output(out)
    src_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    preflight_disk(out, src_bytes)

    # second log handler: file (now that out parent exists)
    log_file = args.log_file or (out.parent / f"{out.name}.conversion.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(fh)
    LOG.info(f"log file: {log_file}")

    LOG.info(f"=== imports ===")
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForImageTextToText, AutoConfig)
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    multi_gpu = (args.device == "auto")
    preflight_gpu(src_bytes, multi_gpu=multi_gpu)
    device_map = "auto" if multi_gpu else args.device
    # Inputs need a concrete device. With device_map="auto", accelerate routes
    # them to the right shard; we just need to put them on a real GPU first.
    input_device = "cuda:0" if multi_gpu else args.device

    # ── load model — pick the right AutoModel class ───
    is_vl = "text_config" in cfg or "vision_config" in cfg
    LOG.info(f"=== loading source model from {src} (vision-language={is_vl}) ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if is_vl:
        model = AutoModelForImageTextToText.from_pretrained(
            src, torch_dtype=torch.bfloat16, device_map=device_map,
            trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, device_map=device_map,
            trust_remote_code=True)
    model.eval()
    LOG.info(f"loaded in {time.time()-t0:.1f}s · "
             f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # For VL models, only quantize the language branch; leave the vision
    # encoder at BF16 to avoid degrading visual perception.
    if is_vl:
        target = None
        for path in ("language_model", "model.language_model",
                     "text_model", "model.text_model"):
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                LOG.info(f"VL model: quantizing only `{path}` branch (vision encoder stays BF16)")
                target = obj
                break
        if target is None:
            fail("VL model: couldn't find a 'language_model' / 'text_model' branch. "
                 "Inspect the model with print(model) to find the LM submodule.")
        model_to_quantize = target
    else:
        model_to_quantize = model

    # ── calibration loader ───
    LOG.info(f"=== calibration · {args.calib_samples} samples from {args.calib_dataset} ===")
    from datasets import load_dataset
    ds = load_dataset(args.calib_dataset, "3.0.0" if args.calib_dataset == "cnn_dailymail" else None,
                      split="train", streaming=True)

    def calib_loop():
        n = 0
        # For VL models, use the full multimodal model for the forward pass
        # so attention masks etc. flow correctly; only the LM branch gets
        # quantizers attached.
        forward_target = model
        for sample in ds:
            text = sample.get("article") or sample.get("text") or sample.get("content") or ""
            if not text.strip():
                continue
            ids = tok(text[:4000], return_tensors="pt", truncation=True,
                      max_length=2048).input_ids.to(input_device)
            with torch.no_grad():
                forward_target(ids)
            n += 1
            if n >= args.calib_samples:
                break
            if n % 32 == 0:
                LOG.info(f"  calib step {n}/{args.calib_samples}")
        LOG.info(f"  calib done · {n} samples consumed")

    # ── quantize ───
    LOG.info(f"=== applying NVFP4 quantization ===")
    t0 = time.time()
    cfg = mtq.NVFP4_DEFAULT_CFG
    try:
        mtq.quantize(model_to_quantize, cfg, forward_loop=lambda m: calib_loop())
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        LOG.error(f"")
        LOG.error(f"=== modelopt failed to quantize this architecture ===")
        LOG.error(f"  {type(e).__name__}: {e}")
        LOG.error(f"")
        LOG.error(f"  This usually means a layer type modelopt doesn't yet")
        LOG.error(f"  recognize for NVFP4 — common with very new architectures")
        LOG.error(f"  or custom layer types from `trust_remote_code` models.")
        LOG.error(f"")
        LOG.error(f"  Check your modelopt version (currently 0.43.0). Newer")
        LOG.error(f"  versions add architectures regularly. If your model uses")
        LOG.error(f"  `trust_remote_code`, the custom layers may need a recipe.")
        sys.exit(3)
    LOG.info(f"quantize done in {time.time()-t0:.1f}s")

    # ── export ───
    LOG.info(f"=== exporting to {out} ===")
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=False)
    export_hf_checkpoint(model, export_dir=str(out))
    tok.save_pretrained(out)
    LOG.info(f"export done in {time.time()-t0:.1f}s")

    out_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    LOG.info(f"=== success ===")
    LOG.info(f"output  · {out}")
    LOG.info(f"size    · {out_size/1e9:.1f} GB  (source was {src_bytes/1e9:.1f} GB)")
    LOG.info(f"log     · {log_file}")
    LOG.info(f"")
    LOG.info(f"next: try loading in vLLM:")
    LOG.info(f"  vllm serve {out} --quantization modelopt --dtype bfloat16 --port 8011")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("interrupted by user")
        sys.exit(130)
    except Exception as e:
        LOG.exception(f"unhandled error: {e}")
        sys.exit(2)

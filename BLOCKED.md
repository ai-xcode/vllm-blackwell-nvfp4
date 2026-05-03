# Architectures that don't yet convert cleanly

Honest list of model families I tried and where each one currently breaks
with `convert_to_nvfp4.py` + `nvidia-modelopt 0.43.0`. PRs welcome.

## Qwen3-Next (e.g. Qwen3-Coder-Next 80B)

**Error:** calibration produces `nan` activations after a small number of
samples even with `flash-linear-attention` (FLA 0.5.0) and
`causal-conv1d` 1.6.1 installed.

**Root cause (best understanding):** Qwen3-Next uses a Mamba-hybrid +
sparse-MoE block design. modelopt 0.43 doesn't yet have full quantization
recipes for the linear-attention layers and the sparse-MoE expert gates,
so the calibration forward pass numerically diverges.

**What works instead:** the existing pre-quantized FP8 release of
Qwen3-Coder-Next runs at ~176 tok/s on TP=1 (per the
[tp2-fix repo](https://github.com/ai-xcode/vllm-blackwell-tp2-fix)
benchmarks). Use that until modelopt adds Qwen3-Next NVFP4 support.

## Qwen3.5 / Qwen3.6 35B-A3B vision MoE

**Arch:** `Qwen3_5MoeForConditionalGeneration` and successors.

**Error:** modelopt aborts on `linear_attn` / `shared_expert_gate` /
`.mlp.gate` / `mtp.*` layers — these aren't in its default NVFP4 layer
allow-list.

**What's needed:**
1. Custom exclusion list passed to `mtq.quantize` covering the layer
   names above.
2. Image-aware calibration — the default text-only `cnn_dailymail`
   calibration only exercises the language tower, leaving the vision
   encoder + cross-attention paths unprofiled.

**Workaround for now:** use the existing FP8 / AWQ / GPTQ-Int4 releases
of these models for serving, NVFP4 the dense LMs.

## Qwen2.5-VL (dense vision-language)

**Arch:** `Qwen2_5_VLForConditionalGeneration`.

**Status:** the script *runs* — it loads via `AutoModelForImageTextToText`
and only attaches NVFP4 quantizers to the language branch (the vision
encoder stays BF16). But:

- modelopt's stock NVFP4 recipe is calibrated on text-only data.
- I have not yet end-to-end-validated visual perception on the converted
  model. Real evaluation needs an image-text dataset (e.g. LLaVA-style
  OCR / VQA prompts) and a quality diff vs the BF16 source.

**Recommendation:** treat the converted output as experimental until
someone (you, maybe, with a PR) validates VQA / OCR quality.

---

## How to contribute a fix

1. Pick one of the cases above.
2. Reproduce with the script as-is and capture the full stack trace into
   a log file in `logs/`.
3. Either:
   - Wait for a modelopt release that adds support (track the
     [TensorRT-LLM modelopt repo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/quantization)
     release notes), then re-run, or
   - Patch `convert_to_nvfp4.py` with a custom layer exclusion or
     calibration override and open a PR with the before/after numbers.

If your model converts cleanly *and* benchmarks at acceptable quality
(perplexity within a few % of source on a held-out set), please add a row
to the README's "Conversions confirmed working" table via PR.

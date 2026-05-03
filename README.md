# vLLM NVFP4 Conversion + Serve on RTX PRO 6000 Blackwell

> **Community-documented workflow. Not affiliated with NVIDIA, vLLM, or any
> model author. Use at your own risk. See [`DISCLAIMER.md`](DISCLAIMER.md).**

NVFP4 is NVIDIA's 4-bit float format with native Blackwell tensor-core
support. On sm_120 hardware it shrinks weights to **~1/4 of FP16 memory**
and runs through dedicated FP4 GEMM kernels that older INT4-only
quantization formats can't use — *if* you can actually load it.
(NVFP4 is still lossy 4-bit quantization, not magic. Some perplexity
loss vs BF16 source is expected. Run your own evals before deploying;
this repo measures throughput, not quality.)

This repo is the workflow I use to **convert** unquantized HF models to
NVFP4 via NVIDIA modelopt and **serve** them in vLLM, with all the
CUDA-13 + flashinfer gotchas baked in.

Sibling repo: [vllm-blackwell-tp2-fix](https://github.com/ai-xcode/vllm-blackwell-tp2-fix)
(TP=2 deadlock workaround for the same hardware class).

## TL;DR — convert + serve

```bash
# 1. Convert any BF16/FP16 HF model to NVFP4
python convert_to_nvfp4.py \
    --source ~/models/Qwen3-14B \
    --output ~/models/Qwen3-14B-NVFP4 \
    --calib-samples 256

# 2. Serve it in vLLM with the curand + Blackwell fixes baked in
./start-nvfp4.sh ~/models/Qwen3-14B-NVFP4 --port 8011
```

## The non-obvious gotcha — `curand_kernel.h` not found

Without one of the two fixes below, **no NVFP4 model will load in vLLM
on a default CUDA 13 install.**

flashinfer JIT-compiles its FP4 GEMM kernels for sm_120 the first time
you load an NVFP4 model. The compile fails with:

```
fatal error: curand_kernel.h: No such file or directory
```

The CUDA 13 base install does not ship cuRAND dev headers in
`/usr/local/cuda-13.0/include/` by default — they live in a separate
`libcurand-dev-13-0` package that isn't pulled in as a dependency. Two
ways to fix it:

**Option A — install the apt package (recommended if you have sudo):**
```bash
sudo apt install libcurand-dev-13-0
```
This drops `curand_kernel.h` and friends into
`/usr/local/cuda-13.0/targets/x86_64-linux/include/`, which is on nvcc's
default include path. flashinfer's compile then works with no env-var
trickery.

**Option B — use the pip-wheel headers via env vars (no sudo needed):**
The same headers also ship inside the `nvidia-curand-cu13` pip wheel at:
```
<vllm_venv>/lib/python3.12/site-packages/nvidia/cu13/include/curand_kernel.h
```
[`start-nvfp4.sh`](start-nvfp4.sh) points nvcc at that location with two
env vars before launching vLLM:
```bash
PIP_NV_INC=<venv>/lib/python3.12/site-packages/nvidia/cu13/include
export NVCC_PREPEND_FLAGS="-I$PIP_NV_INC"
export CPATH="$PIP_NV_INC"
```

Either fix works. Once the first JIT compile succeeds, kernels are
cached under `~/.cache/flashinfer/<version>/`. Cold-cache first load on
this hardware takes ~1–2 minutes for the kernel build; warm loads
complete in ~30–40 seconds (measured during the runs in
[`RESULTS.md`](RESULTS.md)).

## Conversions confirmed working

Measured on 1× RTX PRO 6000 Blackwell Workstation Edition, TP=1,
`--max-model-len 4096`, `--gpu-memory-utilization 0.45`. Single-stream
tok/s, 256-token completions at temp=0 with `ignore_eos=True`, 1 warm-up
discarded + 10 timed runs averaged.

| Source model | NVFP4 size | Speed (tok/s) | stdev |
|---|---:|---:|---:|
| Qwen3-4B (dense) | 2.7 GB | **213.62** | 0.07 |
| Qwen3-14B (dense) | 10.6 GB | **100.93** | 0.06 |
| DeepSeek-R1-Distill-14B (Qwen2.5 base) | 9.9 GB | **96.78** | 0.09 |

Reproduce with:
```bash
./start-nvfp4.sh <NVFP4-model-dir> --port 8011 --util 0.45    # in one shell
python bench_tps.py --url http://127.0.0.1:8011/v1 --runs 10  # in another
```

Full per-run log + bench harness in [`RESULTS.md`](RESULTS.md). Source
weights for these models are all on Hugging Face under their original
licenses; this repo does not redistribute them.

## Optional — web UI (`nvfp4_ui.py`)

If you'd rather not type CLI flags, the repo includes a small NiceGUI
dashboard that wraps all three CLIs (convert, serve, bench) with
click-to-run forms, live log streaming, and a live GPU panel.

```bash
./launch_nvfp4_ui.sh        # → http://127.0.0.1:8770
```

Three tabs:
- **Convert** — pick a BF16/FP16 source dir, output name auto-fills,
  choose calibration samples + device, watch the conversion log stream.
- **Serve** — pick an NVFP4 dir, set port + util + max-len, start /
  stop a vLLM endpoint. Includes a hard kill of lingering
  `VLLM::EngineCore` worker processes (vLLM workers don't always exit
  with their parent on SIGTERM).
- **Bench** — point at any OpenAI-compatible URL, set runs / warmup /
  max-tokens, get the same numbers `bench_tps.py` produces.

Override defaults via env vars (sensible fallbacks for the layout in
this repo):
- `NVFP4_MODELS_DIR`   — where to scan for source / NVFP4 dirs
- `NVFP4_CONVERT_VENV` — venv that has `nvidia-modelopt`
- `NVFP4_SERVE_VENV`   — venv that has `vllm`
- `NVFP4_UI_PORT`      — UI port (default 8770)

Requires `pip install nicegui` in whichever python launches the UI.

## Architectures known NOT to work yet

See [`BLOCKED.md`](BLOCKED.md) for the actual error each one fails with.
Short list:

- **Qwen3-Next** (e.g. Qwen3-Coder-Next 80B) — Mamba-hybrid + sparse-MoE
  layers produce NaN during calibration on modelopt 0.43. Use the existing
  FP8 quant for now.
- **Qwen3.5 vision MoEs** (35B-A3B etc.) — needs custom layer exclusion
  list (`linear_attn`, `shared_expert_gate`, `mtp.*`) and image-aware
  calibration.
- **Qwen2.5-VL** (dense text+vision) — modelopt's NVFP4 recipe is
  text-tuned; vision encoder degrades without image-text calibration.
  Script does the right thing (only quantize the LM branch) but
  end-to-end VL quality on the converted model is not yet validated here.

If you make any of these work, PRs welcome.

## Tested on

| Hardware | Status |
|---|---|
| 2× RTX PRO 6000 Blackwell Workstation Edition (sm_120, PCIe x16, no NVLink) | ✅ confirmed |

## Software stack (exact versions used to produce the table)

- **Conversion venv** (`~/nvfp4_conversion/venv`):
  - nvidia-modelopt 0.43.0
  - PyTorch 2.11.0 + cu130
  - transformers 5.8.0.dev0 (needed for Qwen3.5 VL family configs)
- **Serving venv** (`~/vLLM_Servers/vllm_env`):
  - vLLM 0.19.2rc1.dev107 (with `--quantization modelopt_fp4`)
  - PyTorch 2.11.0 + cu130
  - flashinfer-python 0.6.7
  - transformers 5.5.4
- **System:**
  - CUDA 13.0.88 (nvcc), Driver 580.126.18
  - NCCL 2.28.9 (`torch.cuda.nccl.version()`)
  - Python 3.12.3
  - Ubuntu 24.04

The fix is not version-pinned to these — they're just what was on the
box when the RESULTS.md numbers were measured. Newer modelopt releases
add more architectures; newer flashinfer keeps adding sm_120 kernels.

## Likely works on (untested by author — community PRs welcome)

- 2× RTX 5090 (same Blackwell arch, also sm_120)
- Other sm_120 workstation Blackwell variants

## Will NOT help you if

- You're on Hopper (H100/H200), Ada (4090), or Ampere (3090) — those
  don't have FP4 tensor cores. Use FP8 on Hopper, INT4-GPTQ on Ada/Ampere.
- You're trying to convert a **pre-quantized** source — modelopt needs
  unquantized BF16/FP16 weights. Convert from the original FP16 release.
- You're on CUDA 12.x — the `curand_kernel.h` problem is specific to
  CUDA 13. Earlier CUDAs ship the header in `/usr/local/cuda/include/`
  and don't need the workaround.

## Files

- [`convert_to_nvfp4.py`](convert_to_nvfp4.py) — conversion CLI. Pre-flight
  checks (source not pre-quantized, output doesn't exist, enough disk +
  GPU mem). Streams calibration from `cnn_dailymail`. For vision-language
  models, quantizes only the LM branch.
- [`start-nvfp4.sh`](start-nvfp4.sh) — vLLM launcher with curand fix +
  Blackwell-PCIe TP>1 fixes baked in. Defaults: port 8011, TP=1, util 0.30.
- [`bench_tps.py`](bench_tps.py) — single-stream tok/s benchmark for any
  OpenAI-compatible endpoint. Used to produce the table above.
- [`nvfp4_ui.py`](nvfp4_ui.py) — NiceGUI dashboard wrapping convert /
  serve / bench. Default port 8770.
- [`launch_nvfp4_ui.sh`](launch_nvfp4_ui.sh) — UI launcher.
- [`RESULTS.md`](RESULTS.md) — full per-run logs and reproduction recipe.
- [`BLOCKED.md`](BLOCKED.md) — architectures that need more work, with
  the actual error each one currently fails with.
- [`DISCLAIMER.md`](DISCLAIMER.md) — what this repo is and is not.

## Disclaimer

See [`DISCLAIMER.md`](DISCLAIMER.md). Short version: provided as-is, no
warranty, do not rely on this for production without your own validation.

## License

[MIT](LICENSE).

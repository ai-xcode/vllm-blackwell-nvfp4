# vLLM NVFP4 Conversion + Serve on RTX PRO 6000 Blackwell

> **Community-documented workflow. Not affiliated with NVIDIA, vLLM, or any
> model author. Use at your own risk. See [`DISCLAIMER.md`](DISCLAIMER.md).**

NVFP4 is NVIDIA's 4-bit float format with native Blackwell tensor-core
support. On sm_120 hardware it gives **near-FP16 quality** at FP8-tier
memory footprint and INT4-tier speed — *if* you can actually load it.
(Some perplexity loss vs BF16 source is expected — this is still lossy
4-bit quantization, not magic. Run your own evals before deploying.)

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

Without this, **no NVFP4 model will load in vLLM on CUDA 13.**

flashinfer JIT-compiles its FP4 GEMM kernels for sm_120 the first time
you load a model. The compile fails with:

```
fatal error: curand_kernel.h: No such file or directory
```

`/usr/local/cuda-13.0/include/` is missing curand-dev headers — there's
no apt package for cu13 dev as of this writing. The headers *do* exist
inside the pip-installed `nvidia-curand-cu13` wheel:

```
<vllm_venv>/lib/python3.12/site-packages/nvidia/cu13/include/curand_kernel.h
```

[`start-nvfp4.sh`](start-nvfp4.sh) works around it with two env vars:

```bash
PIP_NV_INC=<venv>/lib/python3.12/site-packages/nvidia/cu13/include
export NVCC_PREPEND_FLAGS="-I$PIP_NV_INC"
export CPATH="$PIP_NV_INC"
```

After this the JIT compile succeeds and the kernels are cached for all
subsequent loads. **First load takes ~2 minutes** while kernels build;
subsequent loads are fast.

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

## Software stack

- vLLM 0.19.x (with `modelopt_fp4` quantization support)
- PyTorch 2.11.0 + cu130
- nvidia-modelopt 0.43.0
- transformers 5.x (5.8.0.dev for Qwen3.5 VL families)
- flashinfer (with the curand workaround above)
- CUDA 13.0, NCCL 2.28.9, Driver 580.126.18
- Python 3.12

The fix is not version-pinned to these — they're just what was on the box.

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
- [`RESULTS.md`](RESULTS.md) — full per-run logs and reproduction recipe.
- [`BLOCKED.md`](BLOCKED.md) — architectures that need more work, with
  the actual error each one currently fails with.
- [`DISCLAIMER.md`](DISCLAIMER.md) — what this repo is and is not.

## Disclaimer

See [`DISCLAIMER.md`](DISCLAIMER.md). Short version: provided as-is, no
warranty, do not rely on this for production without your own validation.

## License

[MIT](LICENSE).

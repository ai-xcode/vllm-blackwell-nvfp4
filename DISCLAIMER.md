# Disclaimer

**This is a community-documented workflow, not an official toolchain from
the vLLM project, NVIDIA, or any of the model authors. Use at your own
risk.**

## What this repo is

- A reproduction of the steps needed to convert unquantized HF models to
  NVFP4 via `nvidia-modelopt` on a specific hardware configuration
  (2× RTX PRO 6000 Blackwell Workstation Edition, sm_120, PCIe-only).
- A wrapper around `vllm serve` that bakes in two CUDA-13 / Blackwell
  workarounds (the `curand_kernel.h` JIT-compile fix and the Blackwell
  PCIe TP>1 fixes from the sibling repo).
- A small set of conversion benchmarks measured on that same hardware.

## What this repo is NOT

- ❌ Not an official tool from NVIDIA, the vLLM project, or any model
  author. The conversion script is a thin wrapper over `nvidia-modelopt`;
  the launcher is a thin wrapper over `vllm serve`.
- ❌ Not endorsed by anyone. The maintainer (`@ai-xcode`) is one user
  documenting what happens to work on his box.
- ❌ Not guaranteed to apply to your hardware. Different GPUs, different
  CUDA / vLLM / modelopt versions, different motherboards can all change
  the failure mode.
- ❌ Not safety-tested for production. The `--gpu-memory-utilization 0.30`
  default is conservative for shared-GPU inference; tune for your case.

## Quality of converted models

NVFP4 is a lossy quantization. Expect:
- Perplexity within a few % of the BF16 source for most dense LMs.
- Possible degradation on long-tail / rare-token tasks.
- **Vision-language degradation is not characterized in this repo** —
  see `BLOCKED.md` for the Qwen2.5-VL caveat.

If you depend on the model's output quality for anything important, run
your own evaluation against the BF16 source before deploying.

## Model weights

Nothing in this repo includes model weights. Source-model paths in the
examples are placeholders. You are responsible for licensing of any
model you convert and serve.

## TL;DR

> Try this on a non-production box first. If it breaks, you keep the
> pieces. PRs with confirmed-working configurations on additional
> hardware are very welcome.

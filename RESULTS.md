# Benchmark Results

Single-stream throughput on three NVFP4-converted dense LMs, measured
end-to-end through the OpenAI-compatible vLLM endpoint with the
[`bench_tps.py`](bench_tps.py) harness in this repo.

## Hardware / software

| Component | Version |
|---|---|
| GPU | 1× RTX PRO 6000 Blackwell Workstation Edition (sm_120, PCIe x16) |
| Driver / CUDA | 580.126.18 / CUDA 13.0 |
| vLLM | 0.19.x with `--quantization modelopt_fp4` |
| nvidia-modelopt | 0.43.0 (used for the conversion) |
| PyTorch | 2.11.0 + cu130 |
| flashinfer | bundled with vLLM (FP4 GEMM kernels JIT-compiled with curand workaround — see README) |
| Python | 3.12 |

## Methodology

- **Tensor parallel:** TP=1 (single GPU)
- **`--gpu-memory-utilization`:** 0.45 (≈ 43 GB total — model + KV cache)
- **`--max-model-len`:** 4096
- **`--max-num-seqs`:** 8 (default; we never queue more than 1 request, so this is moot)
- **Generation:** 256 completion tokens at `temperature=0`, `ignore_eos=True`
- **Prompt:** fixed multi-paragraph technical-writing prompt (see `bench_tps.py`)
- **Runs:** 1 warm-up (discarded) + 10 timed runs, single-stream (no concurrency)
- **Metric:** `completion_tokens / wall_clock_seconds` per run, then mean over the 10 timed runs

This measures **output throughput** (tokens generated per second after the
prefill phase completes). Time-to-first-token is **not** captured —
that's a different benchmark and matters more for chatbot UX than for
batch / agent throughput.

## Reproduce

In one shell:

```bash
# any one of the three NVFP4 models
./start-nvfp4.sh ~/path/to/Qwen3-14B-NVFP4 --port 8011 --util 0.45
```

In another shell, after the endpoint is live (`curl http://127.0.0.1:8011/v1/models`):

```bash
python bench_tps.py --url http://127.0.0.1:8011/v1 --runs 10 --max-tokens 256
```

The first launch of any NVFP4 model takes ~30–40 s on this hardware
(flashinfer JIT-compiles FP4 GEMM kernels for sm_120; cached after that).

## Raw runs

### Qwen3-4B-NVFP4 (2.7 GB)

```
warmup: 256 tok in  1.28s = 200.50 tok/s
 run 1: 256 tok in  1.20s = 213.60 tok/s
 run 2: 256 tok in  1.20s = 213.68 tok/s
 run 3: 256 tok in  1.20s = 213.53 tok/s
 run 4: 256 tok in  1.20s = 213.63 tok/s
 run 5: 256 tok in  1.20s = 213.59 tok/s
 run 6: 256 tok in  1.20s = 213.64 tok/s
 run 7: 256 tok in  1.20s = 213.68 tok/s
 run 8: 256 tok in  1.20s = 213.49 tok/s
 run 9: 256 tok in  1.20s = 213.70 tok/s
run 10: 256 tok in  1.20s = 213.64 tok/s

avg    : 213.62 tok/s
median : 213.63 tok/s
min/max: 213.49 / 213.70
stdev  : 0.07
```

### Qwen3-14B-NVFP4 (10.6 GB)

```
warmup: 256 tok in  2.61s =  98.10 tok/s
 run 1: 256 tok in  2.54s = 100.96 tok/s
 run 2: 256 tok in  2.54s = 100.97 tok/s
 run 3: 256 tok in  2.54s = 100.96 tok/s
 run 4: 256 tok in  2.53s = 101.03 tok/s
 run 5: 256 tok in  2.54s = 100.91 tok/s
 run 6: 256 tok in  2.54s = 100.95 tok/s
 run 7: 256 tok in  2.54s = 100.92 tok/s
 run 8: 256 tok in  2.54s = 100.84 tok/s
 run 9: 256 tok in  2.54s = 100.87 tok/s
run 10: 256 tok in  2.54s = 100.87 tok/s

avg    : 100.93 tok/s
median : 100.94 tok/s
min/max: 100.84 / 101.03
stdev  : 0.06
```

### DeepSeek-R1-Distill-14B-NVFP4 (9.9 GB)

```
warmup: 256 tok in  2.71s =  94.44 tok/s
 run 1: 256 tok in  2.64s =  96.90 tok/s
 run 2: 256 tok in  2.64s =  96.92 tok/s
 run 3: 256 tok in  2.64s =  96.92 tok/s
 run 4: 256 tok in  2.65s =  96.77 tok/s
 run 5: 256 tok in  2.65s =  96.75 tok/s
 run 6: 256 tok in  2.65s =  96.70 tok/s
 run 7: 256 tok in  2.65s =  96.71 tok/s
 run 8: 256 tok in  2.65s =  96.72 tok/s
 run 9: 256 tok in  2.65s =  96.73 tok/s
run 10: 256 tok in  2.65s =  96.72 tok/s

avg    : 96.78 tok/s
median : 96.74 tok/s
min/max: 96.70 / 96.92
stdev  : 0.09
```

## Observations

- **Run-to-run variance is tiny** (stdev 0.06–0.09 tok/s) once the warm-up
  is discarded. The Blackwell + NVFP4 path is very deterministic at
  single-stream — no thermal throttling, no kernel-cache thrash.
- **Qwen3-14B vs DeepSeek-R1-Distill-14B** (same parameter count,
  different base): 100.93 vs 96.78 tok/s. The ~4 % gap is consistent with
  the slightly different layer composition between Qwen3-14B and
  DeepSeek's Qwen2.5-base distillation; both are well within the same
  performance class.
- **4 → 14 B scaling is sub-linear**, as expected: 213.62 → 100.93 tok/s
  for a ~3.5 × parameter increase. Single-GPU throughput is dominated by
  memory bandwidth and KV-cache size, not raw FLOPs, so the scaling
  isn't perfectly linear with parameter count.

## What this benchmark is NOT

- **Not a quality measurement.** NVFP4 is lossy. Run your own perplexity
  / MMLU / domain-specific evaluation against the BF16 source before
  deploying. Numbers in this table are throughput only.
- **Not concurrent throughput.** Real workloads with multiple parallel
  requests will show higher *aggregate* tok/s thanks to vLLM's continuous
  batching, but lower *per-request* tok/s under contention.
- **Not a comparison vs FP8 / GPTQ-Int4 / AWQ.** Those would need their
  own runs on the same hardware. PRs welcome.

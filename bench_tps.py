#!/usr/bin/env python3
"""bench_tps.py — minimal single-stream tok/s benchmark for an OpenAI-compatible
endpoint (e.g. vLLM, ollama, llama.cpp's server, sglang).

Sends a fixed prompt, requests --max-tokens of completion at temp=0 with
ignore_eos so we always measure full-throughput rather than stopping early.
Discards --warmup first runs, averages the rest. Reports avg / median / min / max.

Usage:
    python bench_tps.py --url http://127.0.0.1:8011/v1 --runs 10
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time

DEFAULT_PROMPT = (
    "Write a detailed technical essay about the history of computer "
    "compilers. Cover: the transition from machine code to assembly, the "
    "first FORTRAN compiler in 1957, the development of LR/LALR parsing, "
    "Lex and Yacc, the rise of LLVM, JIT compilers like HotSpot and V8, "
    "and modern domain-specific compilers. Include specific names, dates, "
    "and concrete technical details. Aim for several paragraphs of "
    "substantive content."
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8011/v1",
                   help="OpenAI-compatible base URL (default: vLLM on :8011)")
    p.add_argument("--model", default=None,
                   help="Model name (default: first model from /v1/models)")
    p.add_argument("--runs", type=int, default=10,
                   help="Number of timed runs (default 10)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of warmup runs to discard (default 1)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Generation length in tokens (default 256)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--api-key", default="dummy",
                   help="API key (vLLM ignores it; needed for the openai client)")
    args = p.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("error: pip install openai", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.url, api_key=args.api_key)

    if not args.model:
        models = client.models.list().data
        if not models:
            print(f"error: no models listed at {args.url}", file=sys.stderr)
            return 2
        args.model = models[0].id
        print(f"# auto-selected model: {args.model}", file=sys.stderr)

    rates: list[float] = []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=0,
            extra_body={"ignore_eos": True},
        )
        elapsed = time.perf_counter() - t0
        n = (r.usage.completion_tokens if r.usage else args.max_tokens)
        rate = n / elapsed
        tag = "warmup" if i < args.warmup else f"run {i - args.warmup + 1}"
        print(f"  {tag:>8}: {n} tok in {elapsed:5.2f}s = {rate:6.2f} tok/s",
              file=sys.stderr)
        if i >= args.warmup:
            rates.append(rate)

    print()
    print(f"# === {args.model} ===")
    print(f"# runs         : {len(rates)}")
    print(f"# avg  tok/s   : {statistics.mean(rates):.2f}")
    print(f"# median tok/s : {statistics.median(rates):.2f}")
    print(f"# min  / max   : {min(rates):.2f} / {max(rates):.2f}")
    print(f"# stdev        : {statistics.stdev(rates) if len(rates) > 1 else 0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

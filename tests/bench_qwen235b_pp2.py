#!/usr/bin/env python3
"""Benchmark Qwen3-235B-A22B with PP=2 on Krasis (HCS hybrid).

Measures:
  - TTFT (wall clock from generate() start to first token)
  - Prefill throughput (prompt tokens / TTFT)
  - Decode speed (tok/s)
  - Output correctness (quick sanity check)
"""

import gc
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("bench_qwen235b")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from krasis.model import KrasisModel
from krasis.timing import TIMING

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-235B-A22B"
HEATMAP_PATH = "/home/main/Documents/Claude/krasis/tests/qwen235b_heatmap.json"
PROMPT_FILE = "/home/main/Documents/Claude/benchmark_prompt.txt"
PP = [47, 47]
DEVICES = ["cuda:0", "cuda:1"]
NUM_RUNS = 3
DECODE_TOKENS = 50


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def gpu_mem():
    return {i: torch.cuda.memory_allocated(i) / 1e6 for i in range(torch.cuda.device_count())}


def main():
    # Ensure timing is OFF for accurate speed benchmarks
    TIMING.decode = False
    TIMING.prefill = False

    print("=" * 70)
    print("QWEN3-235B-A22B PP=2 BENCHMARK")
    print(f"Model: {MODEL}")
    print(f"Partition: {PP}, Devices: {DEVICES}")
    print(f"Decode tokens: {DECODE_TOKENS}, Runs: {NUM_RUNS}")
    print("=" * 70)

    # Check prerequisites
    if not os.path.exists(HEATMAP_PATH):
        print(f"ERROR: Heatmap not found at {HEATMAP_PATH}")
        sys.exit(1)
    if not os.path.exists(PROMPT_FILE):
        print(f"ERROR: Prompt file not found at {PROMPT_FILE}")
        sys.exit(1)

    clear_gpu()

    # Build and load model
    print("\n--- Loading model ---")
    t_load_start = time.perf_counter()
    model = KrasisModel(
        MODEL,
        pp_partition=PP,
        devices=DEVICES,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=1,
        expert_divisor=-3,  # hot_cached_static
    )
    model.load()
    t_load = time.perf_counter() - t_load_start
    print(f"  Model loaded in {t_load:.1f}s")

    # Initialize HCS
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager._init_hot_cached_static(heatmap_path=HEATMAP_PATH)
        n_experts = sum(manager._hcs_num_pinned.values())
        print(f"  {dev_str}: {n_experts} hot experts pinned")

    mem = gpu_mem()
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")

    # Warmup
    print("\n--- Warmup ---")
    warmup_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}])
    gen = model.generate(warmup_tokens, max_new_tokens=10, temperature=0.6, top_k=50)
    warmup_text = model.tokenizer.decode(gen)
    print(f"  Warmup output: {repr(warmup_text[:100])}")

    # Quick correctness check
    print("\n--- Correctness check ---")
    check_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer with just the number."}])
    gen = model.generate(check_tokens, max_new_tokens=30, temperature=0.0, top_k=1)
    check_text = model.tokenizer.decode(gen)
    print(f"  Correctness output: {repr(check_text[:200])}")
    if "4" in check_text:
        print("  PASS: Output contains '4'")
    else:
        print("  WARNING: Output may not be correct!")

    # Build benchmark prompt
    with open(PROMPT_FILE) as f:
        prompt_text = f.read()
    messages = [{"role": "user", "content": prompt_text}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"\n--- Benchmark: {len(prompt)} tokens prompt, {DECODE_TOKENS} decode ---")

    # Prefill + decode benchmark
    results = []
    for run in range(NUM_RUNS):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        generated = model.generate(
            prompt,
            max_new_tokens=DECODE_TOKENS,
            temperature=0.6,
            top_k=50,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_s = t_end - t_start
        n_gen = len(generated)

        # Use model's measured TTFT if available
        ttft = getattr(model, '_last_ttft', None)
        if ttft is not None:
            prefill_tok_s = len(prompt) / ttft
            decode_s = total_s - ttft
            decode_tok_s = n_gen / decode_s if decode_s > 0 else 0
        else:
            # Fallback: estimate
            decode_est = n_gen / 4.0
            ttft = total_s - decode_est
            prefill_tok_s = len(prompt) / ttft if ttft > 0 else 0
            decode_tok_s = 4.0  # rough estimate

        overall_tok_s = (len(prompt) + n_gen) / total_s

        results.append({
            "run": run + 1,
            "total_s": total_s,
            "ttft": ttft,
            "n_gen": n_gen,
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
            "overall_tok_s": overall_tok_s,
        })

        gen_text = model.tokenizer.decode(generated[:20])
        print(f"\n  Run {run+1}/{NUM_RUNS}:")
        print(f"    Total: {total_s:.1f}s, {n_gen} tokens generated")
        print(f"    TTFT: {ttft:.1f}s → Prefill: {prefill_tok_s:.1f} tok/s")
        print(f"    Decode: {decode_tok_s:.2f} tok/s ({n_gen} tok in {total_s - ttft:.1f}s)")
        print(f"    Overall: {overall_tok_s:.2f} tok/s")
        print(f"    Output: {repr(gen_text[:100])}")

    # Short decode reference (no long prefill overhead)
    print("\n--- Short prompt decode reference (64 tokens) ---")
    short_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a poem about recursion."}])
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    generated = model.generate(short_tokens, max_new_tokens=64, temperature=0.6, top_k=50)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    n_gen = len(generated)
    decode_tok_s = n_gen / (t_end - t_start)
    print(f"  {n_gen} tokens in {t_end - t_start:.1f}s = {decode_tok_s:.2f} tok/s")

    # Summary
    mem = gpu_mem()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: Qwen3-235B-A22B, PP=2 ({PP[0]}+{PP[1]})")
    print(f"  Config: HCS hybrid, INT4 GPU, INT4 CPU, FP8 KV, INT8 attn")
    print(f"  Prompt: {len(prompt)} tokens, Decode: {DECODE_TOKENS}")
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")
    print(f"  RAM: ~{_read_vmrss_gb():.0f} GB (process RSS)")
    for r in results:
        print(f"  Run {r['run']}: {r['total_s']:.1f}s total, "
              f"TTFT={r['ttft']:.1f}s, "
              f"prefill={r['prefill_tok_s']:.1f} tok/s, "
              f"decode={r['decode_tok_s']:.2f} tok/s")
    avg_prefill = sum(r['prefill_tok_s'] for r in results) / len(results)
    avg_decode = sum(r['decode_tok_s'] for r in results) / len(results)
    avg_ttft = sum(r['ttft'] for r in results) / len(results)
    print(f"\n  AVG: prefill={avg_prefill:.1f} tok/s, decode={avg_decode:.2f} tok/s, TTFT={avg_ttft:.1f}s")
    print(f"  Short decode: {decode_tok_s:.2f} tok/s")
    print("=" * 70)


def _read_vmrss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576
    except:
        pass
    return 0


if __name__ == "__main__":
    main()

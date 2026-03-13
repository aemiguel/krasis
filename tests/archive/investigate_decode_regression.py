#!/usr/bin/env python3
"""Investigate decode regression: 6.07 tok/s (previous best) vs 3.71 tok/s.

Phase 1: Speed benchmark (timing OFF) — get real tok/s for each config
Phase 2: Timing data (timing ON) — instrument 5 tokens to see where time goes
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
logger = logging.getLogger("decode_regression")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from krasis.model import KrasisModel
from krasis.timing import TIMING

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
HEATMAP_PATH = "/home/main/Documents/Claude/krasis/tests/coder_next_heatmap.json"
PP = [24, 24]
DEVICES = ["cuda:0", "cuda:1"]


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def gpu_mem():
    return {i: torch.cuda.memory_allocated(i) / 1e6 for i in range(torch.cuda.device_count())}


def bench_decode(model, label, num_tokens=64, num_runs=3):
    """Benchmark pure decode speed (timing OFF)."""
    messages = [{"role": "user", "content": "Write a poem about recursion."}]
    tokens = model.tokenizer.apply_chat_template(messages)

    results = []
    for run in range(num_runs):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        generated = model.generate(tokens, max_new_tokens=num_tokens, temperature=0.6, top_k=50)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        n_gen = len(generated)
        total_s = t_end - t_start
        tok_s = n_gen / total_s if n_gen > 0 else 0
        results.append(tok_s)
        print(f"  [{label}] run {run+1}: {n_gen} tok in {total_s:.1f}s = {tok_s:.2f} tok/s")

    avg = sum(results) / len(results)
    best = max(results)
    print(f"  [{label}] AVG: {avg:.2f} tok/s, BEST: {best:.2f} tok/s")
    return avg, best


def timed_decode(model, label, num_tokens=5):
    """Decode with timing ON — shows per-layer breakdown."""
    TIMING.decode = True
    messages = [{"role": "user", "content": "Write a poem about recursion."}]
    tokens = model.tokenizer.apply_chat_template(messages)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    generated = model.generate(tokens, max_new_tokens=num_tokens, temperature=0.6, top_k=50)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    n_gen = len(generated)
    total_s = t_end - t_start
    print(f"  [{label} TIMED] {n_gen} tok in {total_s:.1f}s (timing overhead included)")
    TIMING.decode = False


def build_model_hcs():
    clear_gpu()
    model = KrasisModel(
        MODEL,
        pp_partition=PP,
        devices=DEVICES,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=1,
        expert_divisor=-3,
    )
    model.load()
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager._init_hot_cached_static(heatmap_path=HEATMAP_PATH)
        n_experts = sum(manager._hcs_num_pinned.values())
        print(f"  {dev_str}: {n_experts} hot experts pinned")
    return model


def build_model_static_pin():
    clear_gpu()
    model = KrasisModel(
        MODEL,
        pp_partition=PP,
        devices=DEVICES,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=1,
        expert_divisor=-1,
    )
    model.load()
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")
    return model


def main():
    print("=" * 70)
    print("DECODE REGRESSION INVESTIGATION")
    print("Previous best: 6.07 tok/s (HCS hybrid)")
    print("=" * 70)

    # Ensure timing is OFF for speed benchmarks
    TIMING.decode = False
    TIMING.prefill = False

    # ═══════════════════════════════════════════════════════════
    # Phase 1: HCS hybrid — speed then timing data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: HCS hybrid (expert_divisor=-3)")
    print("=" * 70)

    model = build_model_hcs()
    mem = gpu_mem()
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")

    # Warmup (timing OFF)
    warmup = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}])
    model.generate(warmup, max_new_tokens=5, temperature=0.6)
    print("  Warmup done")

    # Speed benchmark (timing OFF)
    print("\n  --- Speed benchmark (timing OFF) ---")
    hcs_avg, hcs_best = bench_decode(model, "hcs_hybrid", num_tokens=64, num_runs=3)

    # Timing data (timing ON for 5 tokens)
    print("\n  --- Timing data (5 tokens, timing ON) ---")
    timed_decode(model, "hcs_hybrid", num_tokens=5)

    del model
    clear_gpu()

    # ═══════════════════════════════════════════════════════════
    # Phase 2: static_pin — speed then timing data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: Active-only + static_pin (expert_divisor=-1)")
    print("=" * 70)

    model = build_model_static_pin()
    mem = gpu_mem()
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")

    # Warmup + pinning
    warmup = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}])
    model.generate(warmup, max_new_tokens=5, temperature=0.6)
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.notify_request_done()
        stats = manager.get_ao_stats()
        print(f"  {dev_str}: {stats.get('pinned_experts', 0)} experts pinned")
    print("  Warmup done")

    # Speed benchmark (timing OFF)
    print("\n  --- Speed benchmark (timing OFF) ---")
    sp_avg, sp_best = bench_decode(model, "static_pin", num_tokens=64, num_runs=3)

    # Timing data (timing ON for 5 tokens)
    print("\n  --- Timing data (5 tokens, timing ON) ---")
    timed_decode(model, "static_pin", num_tokens=5)

    del model
    clear_gpu()

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY (timing OFF — real speed)")
    print("=" * 70)
    print(f"  {'Config':<25} {'Avg tok/s':>10} {'Best tok/s':>12}")
    print(f"  {'-'*50}")
    print(f"  {'hcs_hybrid':<25} {hcs_avg:>10.2f} {hcs_best:>12.2f}")
    print(f"  {'static_pin':<25} {sp_avg:>10.2f} {sp_best:>12.2f}")
    print(f"  {'previous best (HCS)':<25} {'':>10} {'6.07':>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()

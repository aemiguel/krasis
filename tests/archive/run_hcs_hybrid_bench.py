#!/usr/bin/env python3
"""Benchmark: HCS hybrid (DMA prefill + HCS decode) vs static_pin.

After discovering that HCS prefill through CPU is 7.7x slower, we now
route M>1 through DMA-based active_only and M=1 through HCS.

Runs sequentially:
  1. static_pin (baseline)
  2. hot_cached_static (hybrid: DMA prefill + HCS decode)

For each: short prompt decode + 8600-token prompt.
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
logger = logging.getLogger("hcs_hybrid")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
HEATMAP_PATH = "/home/main/Documents/Claude/krasis/tests/coder_next_heatmap.json"
PP = [24, 24]
DEVICES = ["cuda:0", "cuda:1"]


def make_quant_cfg():
    return QuantConfig(
        attention="awq", shared_expert="int8",
        dense_mlp="int8", lm_head="int8",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def gpu_mem():
    return {i: torch.cuda.memory_allocated(i) / 1e6 for i in range(torch.cuda.device_count())}


def build_prompt(model, target_tokens=8600):
    """Build a prompt of approximately target_tokens length using Gutenberg text."""
    from prompt_utils import build_prompt_tokens
    return build_prompt_tokens(model, target_tokens)


def benchmark_decode(model, prompt_tokens, num_decode=32, num_runs=3, label=""):
    """Benchmark: prefill + decode, measuring TTFT and decode speed separately."""
    results = []

    for run in range(num_runs):
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=num_decode,
            temperature=0.6,
            top_k=50,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_s = t_end - t_start
        n_gen = len(generated)
        tok_s = n_gen / total_s if n_gen > 0 else 0

        results.append({
            "run": run + 1,
            "prompt_tokens": len(prompt_tokens),
            "generated": n_gen,
            "total_s": total_s,
            "tok_s": tok_s,
        })
        print(f"  [{label}] run {run+1}/{num_runs}: {n_gen} tok in {total_s:.1f}s = {tok_s:.2f} tok/s")

    avg_s = sum(r["total_s"] for r in results) / len(results)
    avg_tok_s = sum(r["tok_s"] for r in results) / len(results)
    best_tok_s = max(r["tok_s"] for r in results)
    print(f"  [{label}] AVG: {avg_s:.1f}s, {avg_tok_s:.2f} tok/s (best: {best_tok_s:.2f})")
    return results


def benchmark_short_decode(model, num_decode=64, num_runs=3, label=""):
    """Short prompt to isolate decode speed."""
    messages = [{"role": "user", "content": "Write a poem about recursion."}]
    tokens = model.tokenizer.apply_chat_template(messages)

    results = []
    for run in range(num_runs):
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        generated = model.generate(
            tokens,
            max_new_tokens=num_decode,
            temperature=0.6,
            top_k=50,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_s = t_end - t_start
        n_gen = len(generated)
        decode_tok_s = n_gen / total_s if n_gen > 1 else 0

        results.append({
            "run": run + 1,
            "generated": n_gen,
            "total_s": total_s,
            "decode_tok_s": decode_tok_s,
        })
        print(f"  [{label}] run {run+1}/{num_runs}: {n_gen} tok in {total_s:.1f}s = {decode_tok_s:.2f} tok/s (short)")

    avg_tok_s = sum(r["decode_tok_s"] for r in results) / len(results)
    best = max(r["decode_tok_s"] for r in results)
    print(f"  [{label}] AVG decode: {avg_tok_s:.2f} tok/s (best: {best:.2f})")
    return results


# ── Step 1: static_pin baseline ──

def step_static_pin():
    print("\n" + "=" * 70)
    print("STEP 1: Benchmark static_pin (baseline)")
    print("=" * 70)

    clear_gpu()
    model = KrasisModel(
        MODEL,
        pp_partition=PP,
        devices=DEVICES,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=1,
        expert_divisor=-1,  # active_only
    )
    model.load()

    # Configure static pinning
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")

    # Warmup (triggers pinning)
    print("  Warmup + pin...")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    tokens = model.tokenizer.apply_chat_template(messages)
    model.generate(tokens, max_new_tokens=10, temperature=0.6)
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.notify_request_done()
        stats = manager.get_ao_stats()
        print(f"    {dev_str}: {stats['pinned_experts']} experts pinned")

    mem = gpu_mem()
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")

    # Short prompt decode
    print("\n  --- Short prompt decode ---")
    short_results = benchmark_short_decode(model, num_decode=64, num_runs=3, label="static_pin")

    # 8600-token prompt
    print("\n  --- 8600 token prompt ---")
    prompt = build_prompt(model, target_tokens=8600)
    print(f"  Prompt: {len(prompt)} tokens")
    long_results = benchmark_decode(model, prompt, num_decode=32, num_runs=2, label="static_pin-8.6K")

    del model
    clear_gpu()
    return {"short": short_results, "long": long_results}


# ── Step 2: HCS hybrid (DMA prefill + HCS decode) ──

def step_hcs_hybrid():
    print("\n" + "=" * 70)
    print("STEP 2: Benchmark HCS hybrid (DMA prefill + HCS decode)")
    print("=" * 70)

    clear_gpu()
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

    # Initialize HCS with heatmap
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager._init_hot_cached_static(heatmap_path=HEATMAP_PATH)
        n_layers = len(manager._hcs_buffers)
        n_experts = sum(manager._hcs_num_pinned.values())
        print(f"    {dev_str}: {n_experts} hot experts across {n_layers} layers")

    mem = gpu_mem()
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")

    # Short prompt decode (pure HCS decode)
    print("\n  --- Short prompt decode ---")
    short_results = benchmark_short_decode(model, num_decode=64, num_runs=3, label="hcs_hybrid")

    # 8600-token prompt (DMA prefill + HCS decode)
    print("\n  --- 8600 token prompt ---")
    prompt = build_prompt(model, target_tokens=8600)
    print(f"  Prompt: {len(prompt)} tokens")
    long_results = benchmark_decode(model, prompt, num_decode=32, num_runs=2, label="hcs_hybrid-8.6K")

    del model
    clear_gpu()
    return {"short": short_results, "long": long_results}


# ── Main ──

def main():
    print("=" * 70)
    print("HCS HYBRID BENCHMARK (DMA prefill + HCS decode)")
    print(f"Model: Qwen3-Coder-Next PP=2 (24+24)")
    print(f"GPUs: cuda:0, cuda:1")
    print("=" * 70)

    all_results = {}

    # Verify heatmap exists
    if not os.path.exists(HEATMAP_PATH):
        print(f"ERROR: Heatmap not found at {HEATMAP_PATH}")
        print("Run run_hcs_benchmark.py first to generate it.")
        sys.exit(1)
    with open(HEATMAP_PATH) as f:
        hm = json.load(f)
    print(f"Heatmap: {len(hm)} entries")

    # Step 1: static_pin baseline
    all_results["static_pin"] = step_static_pin()

    # Step 2: HCS hybrid
    all_results["hcs_hybrid"] = step_hcs_hybrid()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Short decode tok/s':>20} {'8.6K total tok/s':>20}")
    print("-" * 70)

    for name, data in all_results.items():
        short_avg = sum(r["decode_tok_s"] for r in data["short"]) / len(data["short"])
        short_best = max(r["decode_tok_s"] for r in data["short"])
        long_avg = sum(r["tok_s"] for r in data["long"]) / len(data["long"])
        long_best = max(r["tok_s"] for r in data["long"])
        print(f"{name:<25} {short_avg:>8.2f} (best {short_best:.2f}) {long_avg:>8.2f} (best {long_best:.2f})")

    print("=" * 70)

    # Save results
    results_path = "/home/main/Documents/Claude/krasis/tests/hcs_hybrid_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

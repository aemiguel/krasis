#!/usr/bin/env python3
"""Isolated Rust MoE engine benchmark — no attention, no GPU, no Python model.

Loads expert weights into KrasisEngine, then runs forward_moe_direct in a
tight loop across all MoE layers to measure raw CPU MoE throughput.

This lets us iterate on Rust engine optimizations extremely fast without
running the full model (attention, norms, KV cache, etc.).

Usage:
    python tests/bench_engine_isolated.py [--seconds 10] [--threads 48] [--parallel]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the Rust engine directly
import krasis as _krasis
KrasisEngine = _krasis.KrasisEngine


def main():
    parser = argparse.ArgumentParser(description="Isolated MoE engine benchmark")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="How many seconds to run the benchmark (default: 10)")
    parser.add_argument("--threads", type=int, default=48,
                        help="Number of Rayon threads (default: 48)")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Use parallel (multi-threaded) expert matmul")
    parser.add_argument("--model", type=str,
                        default=os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next"),
                        help="Model directory")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    args = parser.parse_args()

    # Read model config
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    moe_intermediate = config.get("moe_intermediate_size", config["intermediate_size"])
    num_experts = config["num_experts"]
    topk = config["num_experts_per_tok"]
    num_layers = config["num_hidden_layers"]

    print(f"Model: {os.path.basename(args.model)}")
    print(f"  hidden={hidden_size}, moe_intermediate={moe_intermediate}")
    print(f"  experts={num_experts}, top-{topk}, layers={num_layers}")
    print(f"  threads={args.threads}, parallel={args.parallel}")
    print()

    # Create engine and load weights
    print("Loading weights...")
    t0 = time.perf_counter()
    engine = KrasisEngine(parallel=args.parallel, num_threads=args.threads)
    engine.load(args.model, cpu_num_bits=4, gpu_num_bits=4)
    t_load = time.perf_counter() - t0
    print(f"Loaded in {t_load:.1f}s")

    n_moe_layers = engine.num_moe_layers()
    print(f"MoE layers: {n_moe_layers}")
    print()

    # Create synthetic data — realistic random activation + expert routing
    rng = np.random.default_rng(42)

    # BF16 activation: random normal, converted to BF16 via float32
    act_f32 = rng.standard_normal(hidden_size).astype(np.float32)
    # Convert to BF16: truncate lower 16 bits of float32
    act_bf16 = (act_f32.view(np.uint32) >> 16).astype(np.uint16)

    # Output buffer
    out_bf16 = np.zeros(hidden_size, dtype=np.uint16)

    # Pre-generate random expert selections for each layer
    # Each layer picks topk random experts with random weights
    layer_expert_ids = []
    layer_expert_weights = []
    for _ in range(n_moe_layers):
        ids = rng.choice(num_experts, size=topk, replace=False).astype(np.int32)
        weights = rng.dirichlet(np.ones(topk)).astype(np.float32)
        layer_expert_ids.append(ids)
        layer_expert_weights.append(weights)

    # Get pointers
    act_ptr = act_bf16.ctypes.data
    out_ptr = out_bf16.ctypes.data
    id_ptrs = [ids.ctypes.data for ids in layer_expert_ids]
    wt_ptrs = [wts.ctypes.data for wts in layer_expert_weights]

    # Warmup
    print(f"Warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        for layer_idx in range(n_moe_layers):
            engine.forward_moe_direct(
                layer_idx,
                act_ptr, id_ptrs[layer_idx], wt_ptrs[layer_idx],
                out_ptr,
                1, topk,
            )
            # Feed output back as input for next layer (realistic data flow)
            act_bf16[:] = out_bf16

    # Reset activation
    act_bf16[:] = (act_f32.view(np.uint32) >> 16).astype(np.uint16)

    # Benchmark
    print(f"Running benchmark for {args.seconds}s...")
    print()

    iterations = 0
    layer_times = np.zeros(n_moe_layers, dtype=np.float64)
    total_start = time.perf_counter()
    deadline = total_start + args.seconds

    while time.perf_counter() < deadline:
        for layer_idx in range(n_moe_layers):
            t_layer = time.perf_counter()
            engine.forward_moe_direct(
                layer_idx,
                act_ptr, id_ptrs[layer_idx], wt_ptrs[layer_idx],
                out_ptr,
                1, topk,
            )
            layer_times[layer_idx] += time.perf_counter() - t_layer
            # Feed forward
            act_bf16[:] = out_bf16
        iterations += 1

    total_elapsed = time.perf_counter() - total_start

    # Results
    ms_per_token = (total_elapsed / iterations) * 1000
    tok_per_sec = iterations / total_elapsed
    avg_layer_ms = layer_times / iterations * 1000

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Iterations (simulated tokens): {iterations}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"MoE time per token: {ms_per_token:.2f}ms")
    print(f"Equivalent tok/s (MoE only): {tok_per_sec:.2f}")
    print()
    print(f"Per-layer average: {avg_layer_ms.mean():.3f}ms")
    print(f"Per-layer min:     {avg_layer_ms.min():.3f}ms")
    print(f"Per-layer max:     {avg_layer_ms.max():.3f}ms")
    print(f"Per-layer std:     {avg_layer_ms.std():.3f}ms")
    print()

    # Show per-layer breakdown if interesting variance
    if avg_layer_ms.std() > 0.05:
        print("Per-layer times (ms):")
        for i, t in enumerate(avg_layer_ms):
            bar = "#" * int(t / avg_layer_ms.max() * 40)
            print(f"  Layer {i:2d}: {t:.3f}ms {bar}")
        print()

    # Context: what this means for full model decode
    # Full decode = MoE + attention + norms
    # From previous measurement: LA ~39ms, GQA ~16ms, norms ~3ms = ~58ms non-MoE
    non_moe_ms = 58.0  # approximate from previous instrumented runs
    estimated_total = ms_per_token + non_moe_ms
    estimated_tok_s = 1000.0 / estimated_total
    print(f"Estimated full decode (with ~{non_moe_ms:.0f}ms attn+norms): "
          f"{estimated_total:.1f}ms = {estimated_tok_s:.2f} tok/s")


if __name__ == "__main__":
    main()

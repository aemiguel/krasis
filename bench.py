#!/usr/bin/env python3
"""Krasis MoE benchmark — measures expert computation throughput.

Usage:
    python bench.py /path/to/hf-model [--group-size 128] [--iters 50] [--layers 5]

Reports:
    - Model loading time (cache vs. quantize)
    - Single expert latency
    - Full MoE forward latency (top-k experts)
    - Per-layer throughput
    - Estimated full-model decode latency
"""

import argparse
import json
import struct
import time
import os
import sys


def f32_to_bf16_bytes(val: float) -> bytes:
    """Convert f32 to raw BF16 bytes (upper 16 bits of IEEE 754)."""
    f32_bytes = struct.pack("f", val)
    return f32_bytes[2:4]


def make_activation(hidden_size: int) -> bytes:
    """Create a synthetic BF16 activation vector."""
    buf = bytearray(hidden_size * 2)
    for i in range(hidden_size):
        val = ((i * 7 + 13) / hidden_size - 0.5) * 0.1
        bf16 = f32_to_bf16_bytes(val)
        buf[i * 2] = bf16[0]
        buf[i * 2 + 1] = bf16[1]
    return bytes(buf)


def main():
    parser = argparse.ArgumentParser(description="Krasis MoE Benchmark")
    parser.add_argument("model_dir", help="Path to HuggingFace model directory")
    parser.add_argument("--group-size", type=int, default=128, help="INT4 group size")
    parser.add_argument("--iters", type=int, default=50, help="Iterations per benchmark")
    parser.add_argument("--layers", type=int, default=5, help="Number of layers to benchmark")
    parser.add_argument("--max-layers", type=int, default=None, help="Max MoE layers to load (None=all)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel matmul")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Error: {args.model_dir} does not exist")
        sys.exit(1)

    # Read model config for display
    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    model_name = os.path.basename(args.model_dir)
    # Handle VL wrappers (e.g., Kimi K2.5 nests under text_config)
    cfg = config.get("text_config", config)
    hidden = cfg["hidden_size"]
    intermediate = cfg["moe_intermediate_size"]
    n_experts = cfg.get("n_routed_experts", cfg.get("num_experts"))
    top_k = cfg["num_experts_per_tok"]
    n_layers = cfg["num_hidden_layers"]
    # first_k_dense_replace (DeepSeek/Kimi) or decoder_sparse_step (Qwen3)
    if "first_k_dense_replace" in cfg:
        first_dense = cfg["first_k_dense_replace"]
    elif "decoder_sparse_step" in cfg:
        step = cfg["decoder_sparse_step"]
        first_dense = 0 if step <= 1 else step - 1
    else:
        first_dense = 0
    moe_layers = n_layers - first_dense

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Krasis MoE Benchmark                           ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Model: {model_name:<40} ║")
    print(f"║  Hidden: {hidden}, Intermediate: {str(intermediate):<18} ║")
    print(f"║  Experts: {n_experts}, Top-{top_k}, MoE layers: {str(moe_layers):<12} ║")
    print(f"║  Group size: {args.group_size}, Parallel: {str(not args.no_parallel):<18} ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Estimate model size
    per_expert_bytes = (
        intermediate * (hidden // 8) * 4  # gate packed
        + intermediate * (hidden // args.group_size) * 2  # gate scales
    ) * 2 + (  # gate + up
        hidden * (intermediate // 8) * 4  # down packed
        + hidden * (intermediate // args.group_size) * 2  # down scales
    )
    total_gb = moe_layers * n_experts * per_expert_bytes / 1e9
    print(f"Estimated INT4 footprint: {total_gb:.1f} GB")
    print(f"Per-expert size: {per_expert_bytes / 1024:.1f} KB")
    print()

    # Load model
    from krasis import KrasisEngine

    engine = KrasisEngine(parallel=not args.no_parallel)

    print("Loading model...")
    t0 = time.time()
    engine.load(args.model_dir, group_size=args.group_size, max_layers=args.max_layers)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  {engine.num_moe_layers()} MoE layers, {engine.num_experts()} experts loaded")
    print()

    # Prepare synthetic data
    activation = make_activation(hidden)
    indices = list(range(top_k))
    weights = [1.0 / top_k] * top_k

    # Benchmark: single expert
    print(f"── Single Expert Forward ──")
    # Warmup
    for _ in range(3):
        engine.moe_forward(0, activation, [0], [1.0])

    times = []
    for _ in range(args.iters):
        t0 = time.time()
        engine.moe_forward(0, activation, [0], [1.0])
        times.append((time.time() - t0) * 1e6)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]
    print(f"  Avg: {avg:.0f} μs, P50: {p50:.0f} μs, P99: {p99:.0f} μs")

    # Bandwidth: how fast we're reading weight data
    expert_bytes = per_expert_bytes
    bw_gb_s = expert_bytes / 1e9 / (avg / 1e6)
    print(f"  Effective bandwidth: {bw_gb_s:.1f} GB/s ({expert_bytes/1024:.0f} KB per expert)")
    print()

    # Benchmark: full MoE forward (top-k experts)
    print(f"── Full MoE Forward (top-{top_k}) ──")
    for _ in range(3):
        engine.moe_forward(0, activation, indices, weights)

    times = []
    for _ in range(args.iters):
        t0 = time.time()
        engine.moe_forward(0, activation, indices, weights)
        times.append((time.time() - t0) * 1e6)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    per_expert = avg / top_k
    print(f"  Avg: {avg:.0f} μs ({avg/1000:.1f} ms), P50: {p50:.0f} μs")
    print(f"  Per expert: {per_expert:.0f} μs")
    print()

    # Benchmark: multiple layers (simulates full model decode)
    num_bench_layers = min(args.layers, engine.num_moe_layers())
    print(f"── Multi-Layer Forward ({num_bench_layers} layers) ──")
    for layer in range(num_bench_layers):
        engine.moe_forward(layer, activation, indices, weights)

    times = []
    for _ in range(args.iters):
        t0 = time.time()
        for layer in range(num_bench_layers):
            engine.moe_forward(layer, activation, indices, weights)
        times.append((time.time() - t0) * 1e6)

    avg = sum(times) / len(times)
    per_layer = avg / num_bench_layers
    print(f"  {num_bench_layers} layers avg: {avg:.0f} μs ({avg/1000:.1f} ms)")
    print(f"  Per layer: {per_layer:.0f} μs ({per_layer/1000:.1f} ms)")
    print()

    # Extrapolate full model
    full_model_ms = per_layer * moe_layers / 1000
    print(f"── Projected Full Model ({moe_layers} MoE layers) ──")
    print(f"  Estimated MoE decode time: {full_model_ms:.1f} ms/token")
    print(f"  Estimated MoE throughput: {1000/full_model_ms:.1f} tok/s (MoE only)")
    print(f"  Note: does not include attention, norms, or GPU communication")
    print()

    # Verify output
    print(f"── Output Sanity Check ──")
    output = engine.moe_forward(0, activation, indices, weights)
    values = struct.unpack(f"{hidden}f", output)
    rms = (sum(v ** 2 for v in values) / hidden) ** 0.5
    nonzero = sum(1 for v in values if abs(v) > 1e-10)
    print(f"  Output RMS: {rms:.6f}")
    print(f"  Non-zero: {nonzero}/{hidden}")
    print(f"  First 3: [{values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f}]")


if __name__ == "__main__":
    main()

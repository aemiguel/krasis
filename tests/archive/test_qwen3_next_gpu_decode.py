#!/usr/bin/env python3
"""Benchmark GPU decode on Qwen3-Coder-Next: CPU vs GPU decode speed.

Runs 4 decode runs each for CPU and GPU paths, reports avg tok/s.
Uses static_pin (-1 divisor + pinning) which is the production config.
"""

import logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("bench_gpu_decode")

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"

NUM_DECODE_TOKENS = 32
NUM_RUNS = 4


def load_model(gpu_prefill_threshold: int):
    """Load Qwen3-Coder-Next PP=2 with active_only + static_pin."""
    qcfg = QuantConfig(gpu_expert_bits=4, cpu_expert_bits=4)
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=gpu_prefill_threshold,
        expert_divisor=-1,  # active_only
    )
    model.load()

    # Configure static pinning
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")

    return model


def benchmark_decode(model, label: str):
    """Run NUM_RUNS decode benchmarks and return results."""
    messages = [{"role": "user", "content": "Explain the theory of relativity in detail, including both special and general relativity, their key equations, experimental confirmations, and modern applications in GPS, gravitational wave detection, and cosmology."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"Prompt: {len(prompt)} tokens, generating {NUM_DECODE_TOKENS} tokens × {NUM_RUNS} runs")
    print(f"{'='*60}")

    results = []
    for run in range(NUM_RUNS):
        # Reset linear attention states for new sequence
        if model.cfg.is_hybrid:
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        seq_states = [
            __import__('krasis.kv_cache', fromlist=['SequenceKVState']).SequenceKVState(c, seq_id=0)
            if c is not None else None
            for c in model.kv_caches
        ]

        device = torch.device(model.ranks[0].device)

        # Prefill
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
        positions = torch.arange(len(prompt), dtype=torch.int32, device=device)

        t_prefill_start = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward(prompt_tensor, positions, seq_states)
        t_prefill_end = time.perf_counter()
        prefill_time = t_prefill_end - t_prefill_start
        prefill_tps = len(prompt) / prefill_time

        # Decode
        from krasis.sampler import sample
        next_logits = logits[-1:, :]
        next_token = sample(next_logits, 0.0, 1, 1.0).item()
        tokens = [next_token]

        t_decode_start = time.perf_counter()
        with torch.inference_mode():
            for step in range(NUM_DECODE_TOKENS - 1):
                pos = len(prompt) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                next_token = sample(logits, 0.0, 1, 1.0).item()
                tokens.append(next_token)
        t_decode_end = time.perf_counter()

        decode_time = t_decode_end - t_decode_start
        decode_tps = (NUM_DECODE_TOKENS - 1) / decode_time  # -1 because first token is from prefill
        ms_per_token = decode_time / (NUM_DECODE_TOKENS - 1) * 1000

        # Free KV
        for s in seq_states:
            if s is not None:
                s.free()

        text = model.tokenizer.decode(tokens)
        results.append({
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "ms_per_token": ms_per_token,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
        })

        print(f"  Run {run+1}: prefill={prefill_tps:.1f} tok/s ({prefill_time:.1f}s), "
              f"decode={decode_tps:.2f} tok/s ({ms_per_token:.0f}ms/tok)")
        if run == 0:
            print(f"  Output: {repr(text[:100])}")

    # Summary
    avg_decode = sum(r["decode_tps"] for r in results) / len(results)
    avg_ms = sum(r["ms_per_token"] for r in results) / len(results)
    avg_prefill = sum(r["prefill_tps"] for r in results) / len(results)
    best_decode = max(r["decode_tps"] for r in results)
    print(f"\n  AVG: decode={avg_decode:.2f} tok/s ({avg_ms:.0f}ms/tok), prefill={avg_prefill:.1f} tok/s")
    print(f"  BEST: decode={best_decode:.2f} tok/s")
    return {"avg_decode": avg_decode, "best_decode": best_decode, "avg_ms": avg_ms,
            "avg_prefill": avg_prefill, "results": results}


def main():
    print("Qwen3-Coder-Next GPU Decode Benchmark")
    print(f"Model: {MODEL}")
    print(f"Config: PP=2, active_only + static_pin, INT4 experts, FP8 KV")
    print()

    # --- GPU decode ---
    # threshold=300 for prefill (short prompts use CPU, long use GPU)
    # M=1 decode always routes through GPU via "or M == 1" in layer.py
    print("Loading model with GPU decode (threshold=300, M=1 → GPU)...")
    t0 = time.time()
    model_gpu = load_model(gpu_prefill_threshold=300)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Warmup run (triggers pinning)
    print("\nWarmup run (triggers static pinning)...")
    warmup_msgs = [{"role": "user", "content": "Hello"}]
    warmup_prompt = model_gpu.tokenizer.apply_chat_template(warmup_msgs)
    model_gpu.generate(warmup_prompt, max_new_tokens=8, temperature=0.0, top_k=1)
    print("Warmup done, experts pinned")

    # Print GPU memory
    for i in range(min(2, torch.cuda.device_count())):
        alloc = torch.cuda.memory_allocated(i) / (1024**2)
        print(f"  GPU{i}: {alloc:.0f} MB")

    gpu_results = benchmark_decode(model_gpu, "GPU Decode (M=1→GPU, prefill threshold=300)")

    # Report pinning stats
    for dev_str, manager in model_gpu.gpu_prefill_managers.items():
        stats = manager.get_ao_stats()
        print(f"  {dev_str}: pinned={len(manager._pinned)}, "
              f"hits={stats['cache_hits']}, misses={stats['cache_misses']}")

    del model_gpu
    torch.cuda.empty_cache()

    # --- CPU decode (baseline) ---
    print("\n\nLoading model with CPU decode (threshold=300)...")
    t0 = time.time()
    model_cpu = load_model(gpu_prefill_threshold=300)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Warmup
    warmup_prompt = model_cpu.tokenizer.apply_chat_template(warmup_msgs)
    model_cpu.generate(warmup_prompt, max_new_tokens=8, temperature=0.0, top_k=1)

    cpu_results = benchmark_decode(model_cpu, "CPU Decode (threshold=300)")
    del model_cpu

    # --- Final comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"  CPU decode: {cpu_results['avg_decode']:.2f} tok/s ({cpu_results['avg_ms']:.0f}ms/tok)")
    print(f"  GPU decode: {gpu_results['avg_decode']:.2f} tok/s ({gpu_results['avg_ms']:.0f}ms/tok)")
    speedup = gpu_results['avg_decode'] / cpu_results['avg_decode'] if cpu_results['avg_decode'] > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()

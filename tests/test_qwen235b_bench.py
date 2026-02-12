#!/usr/bin/env python3
"""Qwen3-235B-A22B benchmark — prefill + decode with VRAM/RAM tracking."""

import logging
import time
import os
import gc
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen235b-bench")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Qwen3-235B-A22B"

from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample


def get_mem_stats():
    """Return VRAM per GPU and system RAM usage."""
    vram = {}
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e6
        vram[f"GPU{i}"] = alloc
    # System RAM from /proc/meminfo
    with open("/proc/meminfo") as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0].rstrip(":")] = int(parts[1])  # KB
    total_gb = meminfo.get("MemTotal", 0) / 1024 / 1024
    avail_gb = meminfo.get("MemAvailable", 0) / 1024 / 1024
    used_gb = total_gb - avail_gb
    return vram, used_gb, total_gb


def print_mem(label):
    vram, used_gb, total_gb = get_mem_stats()
    vram_str = ", ".join(f"{k}={v:.0f}MB" for k, v in sorted(vram.items()))
    print(f"  [{label}] VRAM: {vram_str} | RAM: {used_gb:.1f}/{total_gb:.1f} GB used")


def build_prompt(model, target_tokens):
    base = (
        "Analyze the following code and provide a detailed review:\n\n"
        "```python\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
        "```\n\n"
        "The quick brown fox jumps over the lazy dog. "
        "This is a test of the emergency broadcast system. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    )
    base_tokens = model.tokenizer.tokenizer.encode(base)
    reps = (target_tokens // len(base_tokens)) + 1
    long_text = base * reps
    messages = [{"role": "user", "content": long_text + "\n\nProvide a brief summary."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokens


def profile_prefill(model, prompt_tokens):
    """Single forward call prefill timing."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    token_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model.forward(token_ids, positions, seq_states)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    for s in seq_states:
        s.free()
    return elapsed


def profile_decode(model, prompt_tokens, num_tokens=50):
    """Prefill then decode with per-token timing."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    # Prefill
    token_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    pf_start = time.perf_counter()
    logits = model.forward(token_ids, positions, seq_states)
    torch.cuda.synchronize()
    pf_elapsed = time.perf_counter() - pf_start

    # Decode
    next_logits = logits[-1:, :]
    next_token = sample(next_logits, 0.0, 1, 1.0).item()
    generated = [next_token]
    decode_times = []

    for step in range(num_tokens - 1):
        pos = len(prompt_tokens) + step
        token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(token_tensor, pos_tensor, seq_states)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        decode_times.append(elapsed)

        next_token = sample(logits, 0.0, 1, 1.0).item()
        generated.append(next_token)
        if next_token == model.cfg.eos_token_id:
            break

    for s in seq_states:
        s.free()

    text = model.tokenizer.decode(generated)
    return pf_elapsed, decode_times, text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisor", type=int, default=2,
                        help="expert_divisor: 0=chunked, 2+=layer_grouped (persistent won't fit)")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3-235B-A22B Benchmark")
    print(f"Config: expert_divisor={args.divisor}, threads={args.threads}, kv_dtype={args.kv_dtype}")
    print("=" * 70)

    print_mem("before load")

    kv_dtype = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float8_e4m3fn

    qcfg = QuantConfig(
        attention="int8",
        shared_expert="int8",
        dense_mlp="int8",
        lm_head="int8",
        gpu_expert_bits=4,
        cpu_expert_bits=4,
    )

    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=3,
        gpu_prefill=True,
        gpu_prefill_threshold=10,
        krasis_threads=args.threads,
        quant_cfg=qcfg,
        kv_dtype=kv_dtype,
        expert_divisor=args.divisor,
    )

    t_load_start = time.perf_counter()
    model.load()
    t_load = time.perf_counter() - t_load_start

    print(f"\nModel loaded in {t_load:.1f}s")
    print_mem("after load")

    # ── Correctness check ──
    print("\n=== Correctness ===")
    messages = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    pf_time, dtimes, text = profile_decode(model, tokens, num_tokens=10)
    first_word = text.strip().split()[0] if text.strip() else ""
    passed = "4" in first_word
    print(f"  2+2 = '{text.strip()[:50]}' → {'PASS' if passed else 'FAIL'}")
    print_mem("after first inference")

    if not passed:
        print("  WARNING: Correctness check failed, continuing benchmark anyway")

    # ── Prefill scaling ──
    print("\n=== GPU Prefill Scaling ===")
    for size in [512, 1024, 2048, 4096]:
        tokens = build_prompt(model, size)
        elapsed = profile_prefill(model, tokens)
        tps = len(tokens) / elapsed
        print(f"  {len(tokens):>5} tokens: {elapsed:.3f}s = {tps:.1f} tok/s")
    print_mem("after prefill tests")

    # ── Decode timing ──
    print("\n=== CPU Decode (50 tokens after 2K prefill) ===")
    tokens_2k = build_prompt(model, 2048)
    pf_time, decode_times, gen_text = profile_decode(model, tokens_2k, num_tokens=50)

    if decode_times:
        times_ms = [t * 1000 for t in decode_times]
        avg_ms = sum(times_ms) / len(times_ms)
        p50_ms = sorted(times_ms)[len(times_ms) // 2]
        p90_ms = sorted(times_ms)[int(len(times_ms) * 0.9)]
        min_ms = min(times_ms)
        max_ms = max(times_ms)

        print(f"  Prefill: {len(tokens_2k)} tokens in {pf_time:.3f}s ({len(tokens_2k)/pf_time:.0f} tok/s)")
        print(f"  Decoded: {len(decode_times)} tokens")
        print(f"  Avg:  {avg_ms:.1f}ms ({1000/avg_ms:.2f} tok/s)")
        print(f"  P50:  {p50_ms:.1f}ms")
        print(f"  P90:  {p90_ms:.1f}ms")
        print(f"  Min:  {min_ms:.1f}ms")
        print(f"  Max:  {max_ms:.1f}ms")
        print(f"  First 5: {[f'{t:.0f}ms' for t in times_ms[:5]]}")
        print(f"  Text: {gen_text[:150]}...")

    print_mem("after decode tests")

    # ── Context length scaling ──
    print("\n=== Decode Speed vs Context Length ===")
    for ctx_size in [512, 2048, 4096]:
        ctx_tokens = build_prompt(model, ctx_size)
        _, dtimes, _ = profile_decode(model, ctx_tokens, num_tokens=20)
        if dtimes:
            avg = sum(dtimes) / len(dtimes) * 1000
            print(f"  ctx={len(ctx_tokens):>5}: avg {avg:.1f}ms/tok = {1000/avg:.2f} tok/s")

    print_mem("final")
    print("\n" + "=" * 70)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()

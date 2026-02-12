#!/usr/bin/env python3
"""V2-Lite in-depth timing analysis — per-layer GPU prefill and CPU decode breakdown."""

import logging
import time
import sys
import json
import os
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("timing")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample


def load_model():
    qcfg = QuantConfig(
        attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=MODEL_PATH, num_gpus=1,
        gpu_prefill=True, gpu_prefill_threshold=10,
        krasis_threads=16, quant_cfg=qcfg, kv_dtype=torch.bfloat16,
    )
    model.load()
    return model


def build_prompt(model, target_tokens):
    """Build a prompt of approximately target_tokens length."""
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
    reps_needed = (target_tokens // len(base_tokens)) + 1
    long_text = base * reps_needed
    messages = [{"role": "user", "content": long_text + "\n\nProvide a brief summary."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokens


def benchmark_prefill_sizes(model, sizes):
    """Benchmark GPU prefill at various prompt sizes."""
    results = []
    device = torch.device(model.ranks[0].device)

    for size in sizes:
        tokens = build_prompt(model, size)
        actual_size = len(tokens)

        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]
        prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(actual_size, dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model.forward(prompt_tensor, positions, seq_states)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tps = actual_size / elapsed

        for s in seq_states:
            s.free()

        results.append({
            "target": size,
            "actual": actual_size,
            "time_s": elapsed,
            "tps": tps,
        })
        print(f"  {actual_size:>6} tokens: {elapsed:.3f}s = {tps:.1f} tok/s")

    return results


def benchmark_decode_detailed(model, prompt_tokens, num_decode=100):
    """Detailed decode benchmark with per-token timings."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    # Prefill
    torch.cuda.synchronize()
    pf_start = time.perf_counter()
    logits = model.forward(prompt_tensor, positions, seq_states)
    torch.cuda.synchronize()
    pf_time = time.perf_counter() - pf_start

    # Decode with per-token timing
    next_logits = logits[-1:, :]
    next_token = sample(next_logits, 0.6, 50, 1.0).item()
    generated = [next_token]
    decode_times = []

    for step in range(num_decode - 1):
        pos = len(prompt_tokens) + step
        token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(token_tensor, pos_tensor, seq_states)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        next_token = sample(logits, 0.6, 50, 1.0).item()
        generated.append(next_token)
        if next_token == model.cfg.eos_token_id:
            break

    for s in seq_states:
        s.free()

    text = model.tokenizer.decode(generated)
    return pf_time, decode_times, text


def main():
    print("=" * 70)
    print("V2-Lite In-Depth Timing Analysis")
    print("=" * 70)

    model = load_model()

    # ── GPU Prefill Scaling ──
    print("\n--- GPU Prefill Scaling ---")
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    prefill_results = benchmark_prefill_sizes(model, sizes)

    # ── Decode after short prompt ──
    print("\n--- CPU Decode (after 100-token prompt) ---")
    short_tokens = build_prompt(model, 100)
    pf_short, dec_short, text_short = benchmark_decode_detailed(model, short_tokens, num_decode=50)
    print(f"  Prefill: {len(short_tokens)} tokens in {pf_short:.3f}s = {len(short_tokens)/pf_short:.1f} tok/s")
    if dec_short:
        avg_short = sum(dec_short) / len(dec_short) * 1000
        print(f"  Decode: {len(dec_short)} tokens, avg {avg_short:.1f}ms = {1000/avg_short:.1f} tok/s")
        print(f"  Generated: {text_short[:80]}...")

    # ── Decode after medium prompt ──
    print("\n--- CPU Decode (after 1K-token prompt) ---")
    med_tokens = build_prompt(model, 1000)
    pf_med, dec_med, text_med = benchmark_decode_detailed(model, med_tokens, num_decode=50)
    print(f"  Prefill: {len(med_tokens)} tokens in {pf_med:.3f}s = {len(med_tokens)/pf_med:.1f} tok/s")
    if dec_med:
        avg_med = sum(dec_med) / len(dec_med) * 1000
        print(f"  Decode: {len(dec_med)} tokens, avg {avg_med:.1f}ms = {1000/avg_med:.1f} tok/s")
        print(f"  Generated: {text_med[:80]}...")

    # ── Decode after long prompt ──
    print("\n--- CPU Decode (after 10K-token prompt) ---")
    long_tokens = build_prompt(model, 10000)
    pf_long, dec_long, text_long = benchmark_decode_detailed(model, long_tokens, num_decode=50)
    print(f"  Prefill: {len(long_tokens)} tokens in {pf_long:.3f}s = {len(long_tokens)/pf_long:.1f} tok/s")
    if dec_long:
        avg_long = sum(dec_long) / len(dec_long) * 1000
        print(f"  Decode: {len(dec_long)} tokens, avg {avg_long:.1f}ms = {1000/avg_long:.1f} tok/s")
        print(f"  Generated: {text_long[:80]}...")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("TIMING ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nGPU Prefill Scaling:")
    print(f"  {'Tokens':>8}  {'Time':>8}  {'tok/s':>8}  {'ms/tok':>8}")
    for r in prefill_results:
        ms_per_tok = r["time_s"] * 1000 / r["actual"]
        print(f"  {r['actual']:>8}  {r['time_s']:>7.3f}s  {r['tps']:>7.1f}  {ms_per_tok:>7.2f}")

    print("\nCPU Decode by Context Length:")
    for label, dec_times, prompt_len in [
        ("100 tok prompt", dec_short, len(short_tokens)),
        ("1K tok prompt", dec_med, len(med_tokens)),
        ("10K tok prompt", dec_long, len(long_tokens)),
    ]:
        if dec_times:
            avg = sum(dec_times) / len(dec_times) * 1000
            p50 = sorted(dec_times)[len(dec_times)//2] * 1000
            mn = min(dec_times) * 1000
            mx = max(dec_times) * 1000
            print(f"  {label:>15}: avg={avg:.1f}ms  p50={p50:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms  ({1000/avg:.1f} tok/s)")

    # Write results
    results = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "model": "DeepSeek-V2-Lite",
        "test": "timing_analysis",
        "prefill_scaling": prefill_results,
        "decode_100_tok": {
            "prompt_tokens": len(short_tokens),
            "times_ms": [t * 1000 for t in dec_short] if dec_short else [],
        },
        "decode_1k_tok": {
            "prompt_tokens": len(med_tokens),
            "times_ms": [t * 1000 for t in dec_med] if dec_med else [],
        },
        "decode_10k_tok": {
            "prompt_tokens": len(long_tokens),
            "times_ms": [t * 1000 for t in dec_long] if dec_long else [],
        },
    }

    timing_file = os.path.join(os.path.dirname(__file__), "timing_analysis.json")
    with open(timing_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {timing_file}")


if __name__ == "__main__":
    main()

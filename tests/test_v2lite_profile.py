#!/usr/bin/env python3
"""V2-Lite in-depth timing analysis — per-component breakdown for GPU prefill and CPU decode."""

import logging
import time
import json
import os
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("profile")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample


def load_model(expert_divisor=1):
    qcfg = QuantConfig(
        attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=MODEL_PATH, num_gpus=1,
        gpu_prefill=True, gpu_prefill_threshold=10,
        krasis_threads=16, quant_cfg=qcfg, kv_dtype=torch.bfloat16,
        expert_divisor=expert_divisor,
    )
    model.load()
    return model


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
    reps_needed = (target_tokens // len(base_tokens)) + 1
    long_text = base * reps_needed
    messages = [{"role": "user", "content": long_text + "\n\nProvide a brief summary."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokens


def profile_prefill_single(model, prompt_tokens):
    """Profile GPU prefill as a single forward call (for layer_grouped and persistent)."""
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


def profile_prefill_chunks(model, prompt_tokens, chunk_size=2048):
    """Profile GPU prefill with per-chunk timing (for chunked mode)."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    chunk_times = []
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    for chunk_start in range(0, len(prompt_tokens), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(prompt_tokens))
        chunk_ids = torch.tensor(prompt_tokens[chunk_start:chunk_end], dtype=torch.long, device=device)
        chunk_pos = torch.arange(chunk_start, chunk_end, dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(chunk_ids, chunk_pos, seq_states)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        chunk_times.append({
            "chunk_idx": len(chunk_times),
            "start_pos": chunk_start,
            "num_tokens": chunk_end - chunk_start,
            "time_s": elapsed,
            "tok_per_s": (chunk_end - chunk_start) / elapsed,
        })

    total_elapsed = time.perf_counter() - total_start

    for s in seq_states:
        s.free()

    return total_elapsed, chunk_times


def profile_decode(model, prompt_tokens, num_tokens=100):
    """Profile decode with per-token timing after prefill."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    # Prefill: single forward call (model.forward routes to layer_grouped if needed)
    token_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    logits = model.forward(token_ids, positions, seq_states)
    torch.cuda.synchronize()
    prefill_elapsed = time.perf_counter() - prefill_start

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
    return prefill_elapsed, decode_times, text


def profile_mode(model, mode_name, sizes=None, use_single_call=False):
    """Profile prefill scaling for a given model/mode.

    Args:
        use_single_call: If True, pass all tokens in one forward() call.
            Required for layer_grouped mode (internal chunking).
            Also recommended for persistent mode (no per-chunk DMA overhead).
    """
    if sizes is None:
        sizes = [512, 1024, 2048, 4096, 8192]

    results = []
    for size in sizes:
        tokens = build_prompt(model, size)
        actual_size = len(tokens)

        if use_single_call:
            total_time = profile_prefill_single(model, tokens)
            num_chunks = 1
        else:
            total_time, chunk_times = profile_prefill_chunks(model, tokens)
            num_chunks = len(chunk_times)

        tps = actual_size / total_time
        results.append({
            "tokens": actual_size,
            "total_time_s": total_time,
            "tok_per_s": tps,
            "num_chunks": num_chunks,
        })
        print(f"  {actual_size:>5} tokens: {total_time:.3f}s = {tps:.1f} tok/s ({num_chunks} chunks)")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisor", type=int, default=1,
                        help="expert_divisor: 0=chunked, 1=persistent, 2+=layer_grouped")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 3 modes: chunked (0), layer_grouped (2), persistent (1)")
    args = parser.parse_args()

    print("=" * 70)
    print("V2-Lite In-Depth Timing Analysis")
    print("=" * 70)

    if args.compare:
        # Compare all three modes
        print("\n=== Mode Comparison: Chunked (0) vs Layer-Grouped (2) vs Persistent (1) ===")
        sizes = [512, 1024, 2048, 4096]

        # ── Chunked (divisor=0) ──
        print("\n--- Chunked (divisor=0) ---")
        model_chunked = load_model(expert_divisor=0)
        # Chunked uses single forward call too (fair comparison)
        chunked_results = profile_mode(model_chunked, "chunked", sizes, use_single_call=True)

        tokens_short = build_prompt(model_chunked, 64)
        _, _, text_chunked = profile_decode(model_chunked, tokens_short, num_tokens=30)
        print(f"  Correctness: {text_chunked[:100]}...")
        del model_chunked
        torch.cuda.empty_cache()

        # ── Layer-Grouped (divisor=2) ──
        print("\n--- Layer-Grouped (divisor=2) ---")
        model_grouped = load_model(expert_divisor=2)
        grouped_results = profile_mode(model_grouped, "layer_grouped", sizes, use_single_call=True)

        tokens_short = build_prompt(model_grouped, 64)
        _, _, text_grouped = profile_decode(model_grouped, tokens_short, num_tokens=30)
        print(f"  Correctness: {text_grouped[:100]}...")
        del model_grouped
        torch.cuda.empty_cache()

        # ── Persistent (divisor=1) ──
        print("\n--- Persistent (divisor=1) ---")
        model_persistent = load_model(expert_divisor=1)
        persistent_results = profile_mode(model_persistent, "persistent", sizes, use_single_call=True)

        tokens_short = build_prompt(model_persistent, 64)
        _, _, text_persistent = profile_decode(model_persistent, tokens_short, num_tokens=30)
        print(f"  Correctness: {text_persistent[:100]}...")

        # Summary comparison
        print("\n=== Comparison Summary ===")
        print(f"  {'Tokens':>6} | {'Chunked':>12} | {'LayerGroup':>12} | {'Persistent':>12} | {'LG/Chunk':>8} | {'Pers/Chunk':>10}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}")
        for c, g, p in zip(chunked_results, grouped_results, persistent_results):
            spd_lg = g["tok_per_s"] / c["tok_per_s"] if c["tok_per_s"] > 0 else 0
            spd_p = p["tok_per_s"] / c["tok_per_s"] if c["tok_per_s"] > 0 else 0
            print(f"  {c['tokens']:>6} | {c['tok_per_s']:>9.1f}/s | {g['tok_per_s']:>9.1f}/s | {p['tok_per_s']:>9.1f}/s | {spd_lg:>6.1f}x | {spd_p:>8.1f}x")

        model = model_persistent
        prefill_results = persistent_results
    else:
        model = load_model(expert_divisor=args.divisor)

        # ── Test 1: Prefill scaling across prompt sizes ──
        print(f"\n=== GPU Prefill Scaling (divisor={args.divisor}) ===")
        use_single = args.divisor >= 1  # single call for persistent and layer_grouped
        prefill_results = profile_mode(model, f"divisor={args.divisor}", use_single_call=use_single)

    # ── Test 2: Per-chunk breakdown for 10K prompt ──
    print("\n=== 10K Prefill ===")
    tokens_10k = build_prompt(model, 10000)
    total_time = profile_prefill_single(model, tokens_10k)
    print(f"  TOTAL: {len(tokens_10k)} tokens in {total_time:.3f}s = {len(tokens_10k)/total_time:.1f} tok/s")

    # ── Test 3: Decode token timing distribution ──
    print("\n=== CPU Decode Timing (100 tokens after 2K prefill) ===")
    tokens_2k = build_prompt(model, 2048)
    pf_time, decode_times, gen_text = profile_decode(model, tokens_2k, num_tokens=100)

    if decode_times:
        times_ms = [t * 1000 for t in decode_times]
        avg_ms = sum(times_ms) / len(times_ms)
        p50_ms = sorted(times_ms)[len(times_ms) // 2]
        p90_ms = sorted(times_ms)[int(len(times_ms) * 0.9)]
        p99_ms = sorted(times_ms)[int(len(times_ms) * 0.99)]
        min_ms = min(times_ms)
        max_ms = max(times_ms)

        print(f"  Prefill: {len(tokens_2k)} tokens in {pf_time:.3f}s")
        print(f"  Decoded: {len(decode_times)} tokens")
        print(f"  Avg:  {avg_ms:.1f}ms ({1000/avg_ms:.1f} tok/s)")
        print(f"  P50:  {p50_ms:.1f}ms")
        print(f"  P90:  {p90_ms:.1f}ms")
        print(f"  P99:  {p99_ms:.1f}ms")
        print(f"  Min:  {min_ms:.1f}ms")
        print(f"  Max:  {max_ms:.1f}ms")
        print(f"  First 5 tokens: {[f'{t:.1f}ms' for t in times_ms[:5]]}")
        print(f"  Text: {gen_text[:150]}...")

        warmup_avg = sum(times_ms[:5]) / 5
        steady_avg = sum(times_ms[5:]) / len(times_ms[5:]) if len(times_ms) > 5 else 0
        print(f"\n  Warmup (first 5): {warmup_avg:.1f}ms avg")
        print(f"  Steady (rest):    {steady_avg:.1f}ms avg")

    # ── Test 4: Decode at different context lengths ──
    print("\n=== Decode Speed vs Context Length ===")
    for ctx_size in [512, 2048, 8192]:
        ctx_tokens = build_prompt(model, ctx_size)
        _, dtimes, _ = profile_decode(model, ctx_tokens, num_tokens=30)
        if dtimes:
            avg = sum(dtimes) / len(dtimes) * 1000
            print(f"  ctx={len(ctx_tokens):>5}: avg {avg:.1f}ms/tok = {1000/avg:.1f} tok/s ({len(dtimes)} tokens)")

    # ── Write analysis ──
    print("\n" + "=" * 70)
    print("Writing performance analysis...")

    analysis = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "model": "DeepSeek-V2-Lite",
        "expert_divisor": args.divisor if not args.compare else "compare",
        "prefill_scaling": prefill_results if not args.compare else {
            "chunked": chunked_results,
            "layer_grouped": grouped_results,
            "persistent": persistent_results,
        },
        "decode_distribution": {
            "avg_ms": round(avg_ms, 1) if decode_times else 0,
            "p50_ms": round(p50_ms, 1) if decode_times else 0,
            "p90_ms": round(p90_ms, 1) if decode_times else 0,
            "min_ms": round(min_ms, 1) if decode_times else 0,
            "max_ms": round(max_ms, 1) if decode_times else 0,
        },
    }

    analysis_file = os.path.join(os.path.dirname(__file__), "profile_results.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Raw data written to {analysis_file}")
    print("Done!")


if __name__ == "__main__":
    main()

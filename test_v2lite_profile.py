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


def profile_prefill_chunks(model, prompt_tokens, chunk_size=2048):
    """Profile GPU prefill with per-chunk timing."""
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


def profile_decode(model, prompt_tokens, num_tokens=100, chunk_size=2048):
    """Profile decode with per-token timing after chunked prefill."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    # Chunked prefill first
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    for chunk_start in range(0, len(prompt_tokens), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(prompt_tokens))
        chunk_ids = torch.tensor(prompt_tokens[chunk_start:chunk_end], dtype=torch.long, device=device)
        chunk_pos = torch.arange(chunk_start, chunk_end, dtype=torch.int32, device=device)
        logits = model.forward(chunk_ids, chunk_pos, seq_states)
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


def main():
    print("=" * 70)
    print("V2-Lite In-Depth Timing Analysis")
    print("=" * 70)

    model = load_model()

    # ── Test 1: Prefill scaling across prompt sizes ──
    print("\n=== GPU Prefill Scaling ===")
    prefill_results = []
    for size in [512, 1024, 2048, 4096, 8192]:
        tokens = build_prompt(model, size)
        actual_size = len(tokens)
        total_time, chunk_times = profile_prefill_chunks(model, tokens)
        tps = actual_size / total_time
        prefill_results.append({
            "tokens": actual_size,
            "total_time_s": total_time,
            "tok_per_s": tps,
            "num_chunks": len(chunk_times),
        })
        print(f"  {actual_size:>5} tokens: {total_time:.3f}s = {tps:.1f} tok/s ({len(chunk_times)} chunks)")

    # ── Test 2: Per-chunk breakdown for 10K prompt ──
    print("\n=== 10K Prefill Per-Chunk Breakdown ===")
    tokens_10k = build_prompt(model, 10000)
    total_time, chunk_times = profile_prefill_chunks(model, tokens_10k)
    for ct in chunk_times:
        print(f"  Chunk {ct['chunk_idx']}: pos {ct['start_pos']:>5}..{ct['start_pos']+ct['num_tokens']:>5} "
              f"({ct['num_tokens']:>4} tok) = {ct['time_s']:.3f}s = {ct['tok_per_s']:.1f} tok/s")
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

        # Check for warmup effect
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
        "prefill_scaling": prefill_results,
        "chunk_breakdown_10k": chunk_times,
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

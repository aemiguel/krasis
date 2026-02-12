#!/usr/bin/env python3
"""V2-Lite 10K token prompt benchmark — GPU prefill vs CPU decode."""

import logging
import time
import sys
import json
import os
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_10k")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample


def load_model(gpu_prefill: bool, gpu_prefill_threshold: int = 300):
    qcfg = QuantConfig(
        attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=MODEL_PATH, num_gpus=1,
        gpu_prefill=gpu_prefill, gpu_prefill_threshold=gpu_prefill_threshold,
        krasis_threads=16, quant_cfg=qcfg, kv_dtype=torch.bfloat16,
    )
    model.load()
    return model


def build_long_prompt(model, target_tokens=10000):
    """Build a prompt that's approximately target_tokens long."""
    # Use a repeating pattern to hit target length
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
    # Tokenize base to figure out repetition count
    base_tokens = model.tokenizer.tokenizer.encode(base)
    reps_needed = (target_tokens // len(base_tokens)) + 1

    long_text = base * reps_needed
    messages = [{"role": "user", "content": long_text + "\n\nProvide a brief summary."}]
    tokens = model.tokenizer.apply_chat_template(messages)

    # Trim to approximately target
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]

    return tokens


def benchmark_decode(model, prompt_tokens, num_decode_tokens=50, prefill_chunk=2048):
    """Benchmark chunked GPU prefill + decode, return per-token decode times."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    # Chunked GPU prefill — process prompt in chunks to keep VRAM intermediates manageable
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    num_chunks = 0
    for chunk_start in range(0, len(prompt_tokens), prefill_chunk):
        chunk_end = min(chunk_start + prefill_chunk, len(prompt_tokens))
        chunk_ids = torch.tensor(prompt_tokens[chunk_start:chunk_end], dtype=torch.long, device=device)
        chunk_pos = torch.arange(chunk_start, chunk_end, dtype=torch.int32, device=device)
        logits = model.forward(chunk_ids, chunk_pos, seq_states)
        num_chunks += 1
    torch.cuda.synchronize()
    prefill_elapsed = time.perf_counter() - prefill_start
    print(f"  Chunked prefill: {len(prompt_tokens)} tokens in {num_chunks} chunks of {prefill_chunk}")

    # Decode
    next_logits = logits[-1:, :]
    next_token = sample(next_logits, 0.0, 1, 1.0).item()
    generated = [next_token]
    decode_times = []

    for step in range(num_decode_tokens - 1):
        pos = len(prompt_tokens) + step
        token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(token_tensor, pos_tensor, seq_states)
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t0)

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
    print("V2-Lite 10K Token Benchmark")
    print("=" * 70)

    # Load model with GPU prefill
    print("\nLoading model with GPU prefill...")
    model = load_model(gpu_prefill=True, gpu_prefill_threshold=10)

    # Build 10K prompt
    prompt_tokens = build_long_prompt(model, target_tokens=10000)
    print(f"\nPrompt: {len(prompt_tokens)} tokens")

    # Full generation with decode timing (includes prefill)
    print("\n--- GPU Prefill + CPU Decode ---")
    gpu_pf_time, decode_times, gen_text = benchmark_decode(model, prompt_tokens, num_decode_tokens=50)
    gpu_prefill_time = gpu_pf_time
    gpu_prefill_tps = len(prompt_tokens) / gpu_pf_time

    if decode_times:
        avg_decode_ms = sum(decode_times) / len(decode_times) * 1000
        decode_tps = 1000.0 / avg_decode_ms
        min_decode_ms = min(decode_times) * 1000
        max_decode_ms = max(decode_times) * 1000
        p50_decode_ms = sorted(decode_times)[len(decode_times)//2] * 1000
    else:
        avg_decode_ms = decode_tps = min_decode_ms = max_decode_ms = p50_decode_ms = 0

    print(f"  Prefill: {len(prompt_tokens)} tokens in {gpu_pf_time:.3f}s = {len(prompt_tokens)/gpu_pf_time:.1f} tok/s")
    print(f"  Decode: {len(decode_times)} tokens, avg {avg_decode_ms:.1f}ms, p50 {p50_decode_ms:.1f}ms, min {min_decode_ms:.1f}ms, max {max_decode_ms:.1f}ms = {decode_tps:.1f} tok/s")
    print(f"  Generated: {gen_text[:100]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Prompt size:        {len(prompt_tokens)} tokens")
    print(f"GPU prefill:        {gpu_prefill_tps:.1f} tok/s ({gpu_prefill_time:.3f}s)")
    print(f"CPU decode:         {decode_tps:.1f} tok/s (avg {avg_decode_ms:.1f}ms)")
    print(f"  p50={p50_decode_ms:.1f}ms  min={min_decode_ms:.1f}ms  max={max_decode_ms:.1f}ms")

    # Write results to benchmarks file
    results = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "model": "DeepSeek-V2-Lite",
        "prompt_tokens": len(prompt_tokens),
        "gpu_prefill_tps": round(gpu_prefill_tps, 1),
        "gpu_prefill_time_s": round(gpu_prefill_time, 3),
        "decode_tps": round(decode_tps, 1),
        "decode_avg_ms": round(avg_decode_ms, 1),
        "decode_p50_ms": round(p50_decode_ms, 1),
        "decode_min_ms": round(min_decode_ms, 1),
        "decode_max_ms": round(max_decode_ms, 1),
        "decode_tokens": len(decode_times),
    }

    # Append to JSON lines file
    bench_file = os.path.join(os.path.dirname(__file__), "benchmarks.jsonl")
    with open(bench_file, "a") as f:
        f.write(json.dumps(results) + "\n")
    print(f"\nResults appended to {bench_file}")

    # Free model
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

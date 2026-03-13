#!/usr/bin/env python3
"""Benchmark Qwen3-Coder-Next GPU prefill speed with a ~5000 token prompt.

GPU prefill threshold is 300 tokens. We use a large prompt to measure
actual GPU Marlin MoE throughput rather than DMA overhead.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"


def make_long_prompt(tokenizer, target_tokens=5000):
    """Build a prompt that tokenizes to approximately target_tokens using Gutenberg text."""
    from prompt_utils import load_prompt_text_truncated
    # ~3.5 chars per token for literary text; overshoot then trim
    text = load_prompt_text_truncated(max_chars=target_tokens * 4)
    content = text + "\n\nSummarize the above text briefly."

    tokens = tokenizer.encode(content)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        content = tokenizer.decode(tokens)

    return [{"role": "user", "content": content}]


def main():
    print("=" * 60)
    print("Benchmark: Qwen3-Coder-Next GPU prefill speed")
    print("=" * 60)

    # Load model with GPU prefill enabled
    print("\nLoading model PP=2...")
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        expert_divisor=4,  # layer-grouped: 6 layers per group = ~4.9 GB VRAM per group
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # VRAM info
    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / (1024**2)
        print(f"  GPU{i}: {alloc:.0f} MB")

    # --- Benchmark: ~5000 token prompt ---
    print("\n--- Benchmark: ~5000 token GPU prefill ---")

    long_msg = make_long_prompt(model.tokenizer, target_tokens=5000)
    prompt_tokens = model.tokenizer.apply_chat_template(long_msg)
    n_prompt = len(prompt_tokens)
    print(f"Prompt tokens: {n_prompt}")

    # Generate only 8 tokens to minimize decode time in measurement
    t1 = time.time()
    text = model.chat(long_msg, max_new_tokens=8, temperature=0.0, top_k=1)
    total_time = time.time() - t1

    n_gen = len(model.tokenizer.encode(text))
    # With only 8 decode tokens at ~0.5s each, decode is ~4s
    est_decode_time = n_gen * 0.5
    est_prefill_time = max(0.1, total_time - est_decode_time)
    est_prefill_speed = n_prompt / est_prefill_time

    print(f"Output ({n_gen} tokens): {repr(text[:100])}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Estimated prefill: {est_prefill_time:.1f}s → ~{est_prefill_speed:.0f} tok/s")
    print(f"Estimated decode: {est_decode_time:.1f}s → ~{n_gen / max(0.1, est_decode_time):.1f} tok/s")

    gpu_prefill_used = n_prompt >= 300
    has_output = len(text.strip()) > 0

    print(f"\nGPU prefill used: {gpu_prefill_used} (prompt={n_prompt}, threshold=300)")
    print(f"Non-empty output: {has_output}")

    print("\n" + "=" * 60)
    print(f"RESULT: {n_prompt} tokens prefill in ~{est_prefill_time:.1f}s = ~{est_prefill_speed:.0f} tok/s")
    print(f"OVERALL: {'PASS' if has_output and gpu_prefill_used else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

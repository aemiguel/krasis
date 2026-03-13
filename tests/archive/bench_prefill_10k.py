#!/usr/bin/env python3
"""Quick benchmark: HCS hybrid prefill speed with ~10K token prompt.

Measures:
  - TTFT (wall clock from generate() start to first token)
  - Prefill throughput (prompt tokens / TTFT)
  - Decode speed (short prompt for reference)
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
logger = logging.getLogger("bench_prefill")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
HEATMAP_PATH = "/home/main/Documents/Claude/krasis/tests/coder_next_heatmap.json"
PP = [24, 24]
DEVICES = ["cuda:0", "cuda:1"]
TARGET_TOKENS = 10000


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def gpu_mem():
    return {i: torch.cuda.memory_allocated(i) / 1e6 for i in range(torch.cuda.device_count())}


def build_prompt(model, target_tokens):
    """Build a prompt of approximately target_tokens length using Gutenberg text."""
    from prompt_utils import build_prompt_tokens
    return build_prompt_tokens(model, target_tokens)


def main():
    print("=" * 70)
    print(f"PREFILL BENCHMARK — ~{TARGET_TOKENS} token prompt")
    print(f"Model: Qwen3-Coder-Next PP=2 (24+24), HCS hybrid")
    print(f"GPUs: {DEVICES}")
    print("=" * 70)

    if not os.path.exists(HEATMAP_PATH):
        print(f"ERROR: Heatmap not found at {HEATMAP_PATH}")
        sys.exit(1)

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

    # Initialize HCS
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager._init_hot_cached_static(heatmap_path=HEATMAP_PATH)
        n_experts = sum(manager._hcs_num_pinned.values())
        print(f"  {dev_str}: {n_experts} hot experts pinned")

    mem = gpu_mem()
    print(f"  VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")

    # Warmup with a tiny prompt
    print("\n--- Warmup ---")
    warmup_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}])
    model.generate(warmup_tokens, max_new_tokens=5, temperature=0.6)
    print("  Warmup done")

    # Build 10K prompt
    prompt = build_prompt(model, target_tokens=TARGET_TOKENS)
    print(f"\n--- Prefill benchmark: {len(prompt)} tokens ---")

    for run in range(3):
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        generated = model.generate(
            prompt,
            max_new_tokens=16,
            temperature=0.6,
            top_k=50,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_s = t_end - t_start
        n_gen = len(generated)

        # TTFT estimate: total time minus decode time
        # Rough: assume ~6 tok/s decode, so decode_s ~ n_gen / 6
        decode_est_s = n_gen / 6.0
        ttft_est = total_s - decode_est_s
        prefill_tok_s = len(prompt) / ttft_est if ttft_est > 0 else 0
        overall_tok_s = (len(prompt) + n_gen) / total_s

        print(f"  Run {run+1}/3: {len(prompt)} prompt + {n_gen} gen = {total_s:.1f}s total")
        print(f"    TTFT (est): {ttft_est:.1f}s → prefill: {prefill_tok_s:.1f} tok/s")
        print(f"    Overall: {overall_tok_s:.2f} tok/s")

    # Short decode reference
    print("\n--- Short decode reference (64 tokens) ---")
    short_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a poem about recursion."}])
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    generated = model.generate(short_tokens, max_new_tokens=64, temperature=0.6, top_k=50)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    n_gen = len(generated)
    tok_s = n_gen / (t_end - t_start)
    print(f"  {n_gen} tokens in {t_end - t_start:.1f}s = {tok_s:.2f} tok/s")

    mem = gpu_mem()
    print(f"\n  Final VRAM: GPU0={mem[0]:.0f} MB, GPU1={mem[1]:.0f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()

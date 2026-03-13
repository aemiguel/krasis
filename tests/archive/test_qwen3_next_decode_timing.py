#!/usr/bin/env python3
"""Quick decode timing breakdown for Qwen3-Coder-Next GPU vs CPU decode."""

import logging, sys, time, os
os.environ["KRASIS_DECODE_TIMING"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"


def load_and_time(label, threshold, expert_divisor=-1):
    print(f"\n{'='*60}")
    print(f"{label} (threshold={threshold}, divisor={expert_divisor})")
    print(f"{'='*60}")

    qcfg = QuantConfig(gpu_expert_bits=4, cpu_expert_bits=4)
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=threshold,
        expert_divisor=expert_divisor,
    )
    model.load()

    if expert_divisor == -1:
        for dev_str, manager in model.gpu_prefill_managers.items():
            manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")

    # Warmup (triggers pinning)
    warmup_prompt = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
    model.generate(warmup_prompt, max_new_tokens=4, temperature=0.0, top_k=1)

    # Real test: 1 prefill + 5 decode tokens
    messages = [{"role": "user", "content": "Explain quantum computing briefly."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"\nPrompt: {len(prompt)} tokens, generating 5 decode tokens")
    print("--- Timing for 5 decode tokens (per-layer breakdown) ---")
    out = model.generate(prompt, max_new_tokens=5, temperature=0.0, top_k=1)
    text = model.tokenizer.decode(out)
    print(f"\nOutput: {repr(text)}")

    # Print AO stats
    if expert_divisor < 0:
        for dev_str, manager in model.gpu_prefill_managers.items():
            stats = manager.get_ao_stats()
            print(f"  {dev_str}: pinned={len(manager._pinned)}, "
                  f"DMA={stats['dma_time']:.2f}s ({stats['dma_bytes']/1e6:.1f} MB), "
                  f"hits={stats['cache_hits']}, misses={stats['cache_misses']}")

    del model
    torch.cuda.empty_cache()


def main():
    # GPU decode
    load_and_time("GPU Decode", threshold=1, expert_divisor=-1)

    # CPU decode
    load_and_time("CPU Decode", threshold=300, expert_divisor=-1)


if __name__ == "__main__":
    main()

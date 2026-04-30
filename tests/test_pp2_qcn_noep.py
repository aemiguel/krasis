#!/usr/bin/env python3
"""Test Qwen3-Coder-Next with PP=2 but single-GPU experts (no EP).

Isolates whether the crash is from PP or EP.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig, compute_pp_partition

MODEL = "/home/main/.krasis/models/Qwen3-Coder-Next"

def test():
    print("=" * 60)
    print("QCN PP=2, No EP (single GPU experts)")
    print("=" * 60)

    quant = QuantConfig(
        attention="hqq8", shared_expert="int8",
        dense_mlp="int8", lm_head="int8",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )

    pp_partition = compute_pp_partition(48, 2)  # [24, 24]
    print(f"PP partition: {pp_partition}")

    # Key: num_gpus=1 means only 1 GPU for EP (no expert parallelism)
    # but pp_partition=[24,24] still splits layers across 2 physical GPUs
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=pp_partition,
        num_gpus=1,  # 1 GPU for EP (no EP), but PP still splits layers
        devices=["cuda:0", "cuda:1"],  # 2 devices for PP
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,
        quant_cfg=quant,
        expert_divisor=4,
        kv_cache_mb=1000,
    )
    model.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    for rank in model.ranks:
        print(f"  Rank {rank.rank}: layers {rank.layer_start}-{rank.layer_end-1} on {rank.device}")

    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        print(f"GPU {i}: {alloc:.0f} MB")

    # Test with GPU prefill threshold prompt
    print("\n--- GPU prefill test ---")
    long_content = (
        "Explain distributed consensus algorithms, database transactions, "
        "compiler optimizations, and memory management in detail. "
    ) * 30
    messages = [{"role": "user", "content": long_content}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")

    t0 = time.time()
    out = model.generate(prompt, max_new_tokens=16, temperature=0.0, top_k=1)
    gen_time = time.time() - t0
    text = model.tokenizer.decode(out)
    print(f"Output ({len(out)} tokens, {gen_time:.1f}s): {repr(text[:200])}")
    assert len(out) > 0, "No output!"
    print("PASS")

if __name__ == "__main__":
    test()

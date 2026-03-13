#!/usr/bin/env python3
"""Test Qwen3-235B-A22B with PP=2 (balanced partition across 2 GPUs).

This is the PRIMARY target of the multi-GPU VRAM architecture plan.
Previously OOMs with PP=1 (all 94 attention layers + experts on 1 GPU).
With PP=2: [47, 47] layers, ~5.6 GB fixed on GPU0, ~4.6 GB on GPU1.

94 layers, 128 experts, top-8, MoE intermediate 1536, hidden 4096.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig, compute_pp_partition

MODEL = "/home/main/.krasis/models/Qwen3-235B-A22B"


def test_pp2():
    print("=" * 60)
    print("Qwen3-235B-A22B PP=2 Test")
    print("=" * 60)

    quant = QuantConfig(
        attention="awq", shared_expert="int8",
        dense_mlp="int8", lm_head="int8",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )

    pp_partition = compute_pp_partition(94, 2)  # [47, 47]
    print(f"\nPP partition: {pp_partition}")

    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=pp_partition,
        num_gpus=2,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,
        quant_cfg=quant,
        expert_divisor=4,  # layer_grouped
        kv_cache_mb=1000,
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # Verify layer distribution
    print(f"\nRanks: {len(model.ranks)}")
    for rank in model.ranks:
        print(f"  Rank {rank.rank}: layers {rank.layer_start}-{rank.layer_end-1} on {rank.device}, "
              f"embed={rank.has_embedding}, lm_head={rank.has_lm_head}")

    # Verify layers on correct devices
    for i, layer in enumerate(model.layers):
        expected_rank = model._get_rank_for_layer(i)
        expected_dev = model.ranks[expected_rank].device
        assert str(layer.device) == expected_dev, \
            f"Layer {i} on {layer.device}, expected {expected_dev}"
    print("Layer device assignment: OK")

    # VRAM usage
    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        free = torch.cuda.mem_get_info(i)[0] / 1024**2
        print(f"GPU {i}: {alloc:.0f} MB allocated, {free:.0f} MB free")

    # Test 1: Short generation (CPU decode since below threshold)
    print("\n--- Test 1: Short generation (CPU decode) ---")
    messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")

    t0 = time.time()
    out = model.generate(prompt, max_new_tokens=32, temperature=0.0, top_k=1)
    gen_time = time.time() - t0
    text = model.tokenizer.decode(out)
    print(f"Output ({len(out)} tokens, {gen_time:.1f}s): {repr(text[:200])}")
    assert len(out) > 0, "No output generated!"
    print("PASS")

    # Test 2: Longer prompt to trigger GPU prefill (>300 tokens threshold)
    print("\n--- Test 2: GPU prefill with longer prompt ---")
    long_content = (
        "Please explain the following concepts in detail: "
        "distributed systems, consensus algorithms, database transactions, "
        "compiler optimizations, memory management, "
    ) * 30  # Should produce >300 tokens
    messages2 = [{"role": "user", "content": long_content}]
    prompt2 = model.tokenizer.apply_chat_template(messages2)
    print(f"Prompt: {len(prompt2)} tokens (GPU prefill should activate)")

    t0 = time.time()
    out2 = model.generate(prompt2, max_new_tokens=16, temperature=0.0, top_k=1)
    gen_time2 = time.time() - t0
    text2 = model.tokenizer.decode(out2)
    print(f"Output ({len(out2)} tokens, {gen_time2:.1f}s): {repr(text2[:200])}")
    assert len(out2) > 0, "No output generated!"
    print("PASS")

    print("\n" + "=" * 60)
    print("Qwen3-235B-A22B PP=2: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_pp2()

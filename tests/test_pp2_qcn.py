#!/usr/bin/env python3
"""Test Qwen3-Coder-Next with PP=2 (balanced partition across 2 GPUs).

QCN is a hybrid model: 48 layers (36 linear attn + 12 GQA), 512 experts, top-10.
PP=2 splits [24, 24]. This tests the production target.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig, compute_pp_partition

MODEL = "/home/main/.krasis/models/Qwen3-Coder-Next"

def test_pp2():
    print("=" * 60)
    print("Qwen3-Coder-Next PP=2 Test")
    print("=" * 60)

    quant = QuantConfig(
        attention="hqq8", shared_expert="int8",
        dense_mlp="int8", lm_head="int8",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )

    pp_partition = compute_pp_partition(48, 2)  # [24, 24]
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

    # Verify hybrid model info
    n_linear = sum(1 for lt in model.cfg.layer_types if lt == "linear_attention")
    n_full = sum(1 for lt in model.cfg.layer_types if lt == "full_attention")
    print(f"Hybrid: {n_linear} linear + {n_full} full attention layers")

    # VRAM usage
    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        print(f"GPU {i}: {alloc:.0f} MB allocated")

    # Test 1: Short generation (CPU decode path since below threshold)
    print("\n--- Test 1: Short generation ---")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")

    t0 = time.time()
    out = model.generate(prompt, max_new_tokens=64, temperature=0.0, top_k=1)
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
    out2 = model.generate(prompt2, max_new_tokens=32, temperature=0.0, top_k=1)
    gen_time2 = time.time() - t0
    text2 = model.tokenizer.decode(out2)
    print(f"Output ({len(out2)} tokens, {gen_time2:.1f}s): {repr(text2[:200])}")
    assert len(out2) > 0, "No output generated!"

    # Measure speeds
    ttft = gen_time2 - (len(out2) * 0.1)  # rough estimate
    prefill_speed = len(prompt2) / gen_time2 if gen_time2 > 0 else 0
    print(f"Rough prefill speed: {prefill_speed:.0f} tok/s total time")
    print("PASS")

    print("\n" + "=" * 60)
    print("Qwen3-Coder-Next PP=2: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_pp2()

#!/usr/bin/env python3
"""Test V2-Lite with PP=2 (balanced partition across 2 GPUs).

Validates the multi-GPU VRAM architecture:
- Balanced PP partition: layers split [14, 13] across GPUs
- lm_head on GPU0
- Per-rank weight loading (no replication)
- Forward pass with hidden state transfer at rank boundaries
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/.krasis/models/DeepSeek-V2-Lite"

def test_pp2():
    print("=" * 60)
    print("V2-Lite PP=2 Test")
    print("=" * 60)

    pp_partition = [14, 13]  # 27 layers split across 2 GPUs
    quant = QuantConfig(attention="awq", shared_expert="int8", dense_mlp="int8", lm_head="int8")

    print(f"\nLoading with PP partition: {pp_partition}")
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=pp_partition,
        num_gpus=2,
        kv_dtype=torch.bfloat16,
        krasis_threads=16,
        gpu_prefill=False,  # CPU-only decode for smoke test
        quant_cfg=quant,
        expert_divisor=0,
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # Verify layer distribution
    print(f"\nRanks: {len(model.ranks)}")
    for rank in model.ranks:
        print(f"  Rank {rank.rank}: layers {rank.layer_start}-{rank.layer_end-1} on {rank.device}, "
              f"embed={rank.has_embedding}, lm_head={rank.has_lm_head}")

    # Verify layers are on correct devices
    for i, layer in enumerate(model.layers):
        expected_rank = model._get_rank_for_layer(i)
        expected_dev = model.ranks[expected_rank].device
        assert str(layer.device) == expected_dev, \
            f"Layer {i} on {layer.device}, expected {expected_dev}"
    print("Layer device assignment: OK")

    # Verify VRAM usage per GPU
    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        print(f"GPU {i}: {alloc:.0f} MB allocated")

    # Test 1: Simple generation
    print("\n--- Test 1: Simple generation ---")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")

    t0 = time.time()
    out = model.generate(prompt, max_new_tokens=32, temperature=0.0, top_k=1)
    gen_time = time.time() - t0
    text = model.tokenizer.decode(out)
    print(f"Output ({len(out)} tokens, {gen_time:.2f}s): {repr(text)}")

    assert len(out) > 0, "No output generated!"
    assert "4" in text, f"Expected '4' in output, got: {repr(text)}"
    print("PASS")

    # Test 2: Longer generation
    print("\n--- Test 2: Longer generation ---")
    messages2 = [{"role": "user", "content": "Count from 1 to 10."}]
    prompt2 = model.tokenizer.apply_chat_template(messages2)
    out2 = model.generate(prompt2, max_new_tokens=64, temperature=0.0, top_k=1)
    text2 = model.tokenizer.decode(out2)
    print(f"Output ({len(out2)} tokens): {repr(text2)}")

    assert len(out2) > 5, "Expected more output tokens"
    print("PASS")

    # Decode speed
    if len(out2) > 1:
        # Exclude prefill from speed calc — approximate
        speed = len(out2) / gen_time
        print(f"Approx decode: {speed:.1f} tok/s")

    print("\n" + "=" * 60)
    print("V2-Lite PP=2: ALL TESTS PASSED")
    print("=" * 60)


def test_pp1_regression():
    """Ensure PP=1 still works."""
    print("\n" + "=" * 60)
    print("V2-Lite PP=1 Regression Test")
    print("=" * 60)

    quant = QuantConfig(attention="awq", shared_expert="int8", dense_mlp="int8", lm_head="int8")

    print("\nLoading with PP=1...")
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=[27],
        num_gpus=1,
        kv_dtype=torch.bfloat16,
        krasis_threads=16,
        gpu_prefill=False,
        quant_cfg=quant,
        expert_divisor=0,
    )
    model.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Verify single rank
    assert len(model.ranks) == 1
    assert model.ranks[0].layer_start == 0
    assert model.ranks[0].layer_end == 27
    assert model.ranks[0].has_embedding
    assert model.ranks[0].has_lm_head
    print("PP=1 rank config: OK")

    # Generate
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    out = model.generate(prompt, max_new_tokens=32, temperature=0.0, top_k=1)
    text = model.tokenizer.decode(out)
    print(f"Output ({len(out)} tokens): {repr(text)}")

    assert len(out) > 0, "No output generated!"
    assert "4" in text, f"Expected '4' in output, got: {repr(text)}"

    print("\n" + "=" * 60)
    print("V2-Lite PP=1: PASS")
    print("=" * 60)


if __name__ == "__main__":
    if "--pp1-only" in sys.argv:
        test_pp1_regression()
    elif "--pp2-only" in sys.argv:
        test_pp2()
    else:
        test_pp1_regression()
        # Free GPU memory before PP=2 test
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        test_pp2()

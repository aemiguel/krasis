#!/usr/bin/env python3
"""Test Qwen3-Coder-Next generation: verify hybrid model produces coherent output."""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"

def main():
    print("=" * 60)
    print("Test: Qwen3-Coder-Next generation (hybrid linear+GQA)")
    print("=" * 60)

    print("\nLoading model PP=2 (24+24 layers) with GPU prefill...")
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        expert_divisor=0,  # chunked mode: 512 experts×48 layers too large for persistent
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # Print model info
    print(f"\nModel config:")
    print(f"  Layers: {model.cfg.num_hidden_layers}")
    print(f"  Hybrid: {model.cfg.is_hybrid}")
    print(f"  Full attention layers: {model.cfg.num_full_attention_layers}")
    print(f"  Linear attention layers: {model.cfg.num_hidden_layers - model.cfg.num_full_attention_layers}")
    print(f"  Experts: {model.cfg.n_routed_experts} routed, top-{model.cfg.num_experts_per_tok}")
    print(f"  Has shared_expert_gate: {model._has_shared_expert_gate}")

    # GPU memory
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**2)
        print(f"  GPU{i}: {alloc:.0f} MB")

    # Test 1: Simple math
    print("\n--- Test 1: Simple math ---")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    t1 = time.time()
    text = model.chat(messages, max_new_tokens=32, temperature=0.0, top_k=1)
    t1_elapsed = time.time() - t1
    print(f"Output: {repr(text)}")
    print(f"Time: {t1_elapsed:.1f}s")
    has_4 = "4" in text
    print(f"Contains '4': {has_4}")

    # Test 2: Counting
    print("\n--- Test 2: Counting ---")
    messages2 = [{"role": "user", "content": "Count from 1 to 5. Just list the numbers."}]
    t2 = time.time()
    text2 = model.chat(messages2, max_new_tokens=64, temperature=0.0, top_k=1)
    t2_elapsed = time.time() - t2
    print(f"Output: {repr(text2)}")
    print(f"Time: {t2_elapsed:.1f}s")
    has_numbers = all(str(n) in text2 for n in range(1, 6))
    print(f"Contains 1-5: {has_numbers}")

    # Summary
    print("\n" + "=" * 60)
    non_empty = len(text.strip()) > 0 and len(text2.strip()) > 0
    print(f"Non-empty output: {non_empty}")
    print(f"Test 1 pass: {has_4}")
    print(f"Test 2 pass: {has_numbers}")
    overall = non_empty and has_4
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

if __name__ == "__main__":
    main()

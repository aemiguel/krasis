#!/usr/bin/env python3
"""V2-Lite dual-format validation: correctness + GPU prefill vs CPU decode speed."""

import logging
import time
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_v2lite_dual")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

from krasis.model import KrasisModel
from krasis.config import QuantConfig


def load_model(gpu_prefill: bool, gpu_prefill_threshold: int = 300):
    """Load V2-Lite with dual-format cache."""
    qcfg = QuantConfig(
        attention="bf16",
        shared_expert="bf16",
        dense_mlp="bf16",
        lm_head="bf16",
        gpu_expert_bits=4,
        cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        gpu_prefill=gpu_prefill,
        gpu_prefill_threshold=gpu_prefill_threshold,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.bfloat16,
    )
    model.load()
    return model


def test_correctness(model):
    """Run a couple of short prompts to verify model responds correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Correctness — short prompts")
    print("=" * 60)

    tests = [
        {
            "name": "simple_math",
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "expect_contains": "4",
        },
        {
            "name": "factual",
            "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
            "expect_contains": "Paris",
        },
    ]

    for t in tests:
        print(f"\n--- {t['name']} ---")
        prompt_tokens = model.tokenizer.apply_chat_template(t["messages"])
        print(f"Prompt: {len(prompt_tokens)} tokens")

        start = time.perf_counter()
        generated = model.generate(prompt_tokens, max_new_tokens=64, temperature=0.0, top_k=1)
        elapsed = time.perf_counter() - start

        text = model.tokenizer.decode(generated)
        tok_per_sec = len(generated) / elapsed if elapsed > 0 else 0

        print(f"Output ({len(generated)} tokens, {elapsed:.2f}s, {tok_per_sec:.1f} tok/s): {text[:200]}")

        if t["expect_contains"] in text:
            print(f"  PASS — contains '{t['expect_contains']}'")
        else:
            print(f"  WARN — expected '{t['expect_contains']}' not found in output")


def test_gpu_prefill_speed(model):
    """Send a large prompt to measure GPU prefill speed vs CPU decode."""
    print("\n" + "=" * 60)
    print("TEST 2: GPU Prefill Speed — large prompt")
    print("=" * 60)

    # Build a long prompt (~500 tokens)
    long_content = (
        "Please summarize the following text:\n\n"
        + "The quick brown fox jumps over the lazy dog. " * 80
        + "\n\nProvide a one-sentence summary."
    )
    messages = [{"role": "user", "content": long_content}]
    prompt_tokens = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens (threshold={model.gpu_prefill_threshold})")

    # Prefill phase
    from krasis.kv_cache import SequenceKVState
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    logits = model.forward(prompt_tensor, positions, seq_states)
    torch.cuda.synchronize()
    prefill_elapsed = time.perf_counter() - prefill_start

    prefill_tok_per_sec = len(prompt_tokens) / prefill_elapsed
    print(f"Prefill: {len(prompt_tokens)} tokens in {prefill_elapsed:.3f}s = {prefill_tok_per_sec:.1f} tok/s")

    # Now decode a few tokens to measure decode speed
    from krasis.sampler import sample

    next_logits = logits[-1:, :]
    next_token = sample(next_logits, 0.0, 1, 1.0).item()
    generated = [next_token]

    decode_times = []
    for step in range(19):  # 20 decode tokens total
        pos = len(prompt_tokens) + step
        token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        logits = model.forward(token_tensor, pos_tensor, seq_states)
        torch.cuda.synchronize()
        decode_elapsed = time.perf_counter() - decode_start
        decode_times.append(decode_elapsed)

        next_token = sample(logits, 0.0, 1, 1.0).item()
        generated.append(next_token)

        if next_token == model.cfg.eos_token_id:
            break

    # Free KV
    for s in seq_states:
        s.free()

    # Stats
    if decode_times:
        avg_decode = sum(decode_times) / len(decode_times)
        decode_tok_per_sec = 1.0 / avg_decode if avg_decode > 0 else 0
        print(f"Decode: {len(decode_times)} tokens, avg {avg_decode*1000:.1f}ms/tok = {decode_tok_per_sec:.1f} tok/s")

    text = model.tokenizer.decode(generated)
    print(f"Generated text: {text[:200]}")

    print(f"\n--- Summary ---")
    print(f"GPU prefill: {prefill_tok_per_sec:.1f} tok/s ({len(prompt_tokens)} tokens)")
    if decode_times:
        print(f"CPU decode:  {decode_tok_per_sec:.1f} tok/s ({len(decode_times)} tokens)")
        print(f"Speedup:     {prefill_tok_per_sec / decode_tok_per_sec:.1f}x")


def test_cpu_only_prefill(model_cpu):
    """Same large prompt but with GPU prefill disabled, for comparison."""
    print("\n" + "=" * 60)
    print("TEST 3: CPU-only Prefill Speed — same large prompt (baseline)")
    print("=" * 60)

    long_content = (
        "Please summarize the following text:\n\n"
        + "The quick brown fox jumps over the lazy dog. " * 80
        + "\n\nProvide a one-sentence summary."
    )
    messages = [{"role": "user", "content": long_content}]
    prompt_tokens = model_cpu.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens (GPU prefill DISABLED)")

    from krasis.kv_cache import SequenceKVState
    device = torch.device(model_cpu.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model_cpu.kv_caches]

    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    logits = model_cpu.forward(prompt_tensor, positions, seq_states)
    torch.cuda.synchronize()
    prefill_elapsed = time.perf_counter() - prefill_start

    prefill_tok_per_sec = len(prompt_tokens) / prefill_elapsed
    print(f"CPU Prefill: {len(prompt_tokens)} tokens in {prefill_elapsed:.3f}s = {prefill_tok_per_sec:.1f} tok/s")

    for s in seq_states:
        s.free()

    return prefill_tok_per_sec


def main():
    print("Loading V2-Lite with GPU prefill (dual-format cache)...")
    model = load_model(gpu_prefill=True, gpu_prefill_threshold=10)  # low threshold to ensure GPU prefill triggers

    # Test 1: Correctness
    test_correctness(model)

    # Test 2: GPU prefill speed
    test_gpu_prefill_speed(model)

    # Free the model
    del model
    torch.cuda.empty_cache()

    # Test 3: CPU-only baseline
    print("\n\nLoading V2-Lite WITHOUT GPU prefill (CPU-only baseline)...")
    model_cpu = load_model(gpu_prefill=False)
    cpu_prefill_speed = test_cpu_only_prefill(model_cpu)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

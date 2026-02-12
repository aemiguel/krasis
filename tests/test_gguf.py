#!/usr/bin/env python3
"""Test GGUF expert loading with V2-Lite model."""

import logging
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("gguf-test")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
GGUF_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite.Q4_K_M.gguf"

from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample


def profile_decode(model, prompt_tokens, num_tokens=30):
    """Prefill then decode with per-token timing."""
    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    token_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    pf_start = time.perf_counter()
    logits = model.forward(token_ids, positions, seq_states)
    torch.cuda.synchronize()
    pf_elapsed = time.perf_counter() - pf_start

    next_logits = logits[-1:, :]
    next_token = sample(next_logits, 0.0, 1, 1.0).item()
    generated = [next_token]
    decode_times = []

    for step in range(num_tokens - 1):
        pos = len(prompt_tokens) + step
        token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(token_tensor, pos_tensor, seq_states)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        decode_times.append(elapsed)

        next_token = sample(logits, 0.0, 1, 1.0).item()
        generated.append(next_token)
        if next_token == model.cfg.eos_token_id:
            break

    for s in seq_states:
        s.free()

    text = model.tokenizer.decode(generated)
    return pf_elapsed, decode_times, text


def run_test(label, model):
    """Run correctness and speed tests."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")

    # Correctness: 2+2
    messages = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    pf, dt, text = profile_decode(model, tokens, num_tokens=10)
    first = text.strip().split()[0] if text.strip() else ""
    passed = "4" in first
    print(f"  2+2 = '{text.strip()[:50]}' → {'PASS' if passed else 'FAIL'}")

    # Correctness: Capital of France
    messages = [{"role": "user", "content": "What is the capital of France? One word only."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    pf2, dt2, text2 = profile_decode(model, tokens, num_tokens=10)
    first2 = text2.strip().split()[0] if text2.strip() else ""
    passed2 = "paris" in first2.lower()
    print(f"  Capital = '{text2.strip()[:50]}' → {'PASS' if passed2 else 'FAIL'}")

    # Decode speed (30 tokens)
    messages = [{"role": "user", "content": "Write a short poem about computers."}]
    tokens = model.tokenizer.apply_chat_template(messages)
    pf3, dt3, text3 = profile_decode(model, tokens, num_tokens=30)

    if dt3:
        times_ms = [t * 1000 for t in dt3]
        avg_ms = sum(times_ms) / len(times_ms)
        print(f"  Prefill: {len(tokens)} tokens in {pf3:.3f}s ({len(tokens)/pf3:.0f} tok/s)")
        print(f"  Decode: avg {avg_ms:.1f}ms = {1000/avg_ms:.2f} tok/s")
        print(f"  Text: {text3[:100]}...")

    return passed and passed2


def main():
    import os
    if not os.path.exists(GGUF_PATH):
        print(f"GGUF file not found: {GGUF_PATH}")
        print("Download with: huggingface-cli download mradermacher/DeepSeek-V2-Lite-GGUF DeepSeek-V2-Lite.Q4_K_M.gguf")
        return

    qcfg = QuantConfig(
        attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )

    # Test 1: Normal mode (BF16 → dual cache)
    print("\n" + "="*60)
    print("Loading model WITHOUT GGUF (normal dual-cache mode)...")
    print("="*60)
    model_normal = KrasisModel(
        model_path=MODEL_PATH, num_gpus=1,
        gpu_prefill=True, gpu_prefill_threshold=10,
        krasis_threads=16, quant_cfg=qcfg, kv_dtype=torch.bfloat16,
        expert_divisor=1,
    )
    t0 = time.perf_counter()
    model_normal.load()
    t_normal = time.perf_counter() - t0
    print(f"Normal load: {t_normal:.1f}s")

    ok1 = run_test("Normal (BF16 dual-cache)", model_normal)

    # Free model
    del model_normal
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)

    # Test 2: GGUF mode
    print("\n" + "="*60)
    print(f"Loading model WITH GGUF: {GGUF_PATH}")
    print("="*60)
    model_gguf = KrasisModel(
        model_path=MODEL_PATH, num_gpus=1,
        gpu_prefill=True, gpu_prefill_threshold=10,
        krasis_threads=16, quant_cfg=qcfg, kv_dtype=torch.bfloat16,
        expert_divisor=1,
        gguf_path=GGUF_PATH,
    )
    t0 = time.perf_counter()
    model_gguf.load()
    t_gguf = time.perf_counter() - t0
    print(f"GGUF load: {t_gguf:.1f}s")

    ok2 = run_test("GGUF (Q4_K_M CPU experts)", model_gguf)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Normal load:  {t_normal:.1f}s")
    print(f"  GGUF load:    {t_gguf:.1f}s")
    print(f"  Normal tests: {'PASS' if ok1 else 'FAIL'}")
    print(f"  GGUF tests:   {'PASS' if ok2 else 'FAIL'}")


if __name__ == "__main__":
    main()

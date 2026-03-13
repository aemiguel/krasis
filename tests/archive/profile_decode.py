#!/usr/bin/env python3
"""Profile decode: detailed per-layer breakdown of attention vs MoE time.

Runs 10 decode tokens with KRASIS_DECODE_TIMING=1 to show where time goes.
"""
import logging, sys, time, os
os.environ["KRASIS_DECODE_TIMING"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("profile")

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"


def profile_decode(label, threshold, num_tokens=10):
    print(f"\n{'='*70}")
    print(f"PROFILE: {label} (threshold={threshold})")
    print(f"{'='*70}")

    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=threshold,
        expert_divisor=-1,
    )
    model.load()

    # Configure pinning
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")

    # Warmup
    warmup = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
    model.generate(warmup, max_new_tokens=4, temperature=0.0, top_k=1)

    # Prefill a prompt, then do decode tokens with full timing
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"\nPrompt: {len(prompt)} tokens")
    print(f"Generating {num_tokens} decode tokens with per-layer timing...")
    print(f"{'='*70}")

    t_gen_start = time.perf_counter()
    out = model.generate(prompt, max_new_tokens=num_tokens, temperature=0.0, top_k=1)
    t_gen_end = time.perf_counter()
    text = model.tokenizer.decode(out)
    print(f"\nOutput: {repr(text)}")
    print(f"Wall-clock: {(t_gen_end - t_gen_start)*1000:.1f}ms for {len(out)} tokens")
    if len(out) > 1:
        # Exclude first token (prefill)
        decode_time = t_gen_end - t_gen_start  # includes prefill
        print(f"  (includes prefill — see DECODE-STEP lines for per-token breakdown)")

    del model
    torch.cuda.empty_cache()


def main():
    profile_decode("GPU Decode", threshold=1, num_tokens=10)


if __name__ == "__main__":
    main()

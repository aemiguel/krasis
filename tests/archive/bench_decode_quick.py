#!/usr/bin/env python3
"""Quick decode benchmark for Qwen3-Coder-Next: uses Rust CPU decode path."""

import logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.timing import TIMING

MODEL = "/home/main/.krasis/models/Qwen3-Coder-Next"
NUM_DECODE_TOKENS = 100
NUM_RUNS = 1

def main():
    instrument = os.environ.get("INSTRUMENT", "0") == "1"
    if instrument:
        TIMING.decode = True
        print("INSTRUMENTED run (timing enabled)")
    else:
        print("CLEAN run (timing disabled)")

    print("Loading Qwen3-Coder-Next PP=2, CPU decode via Rust decode_step...")
    t0 = time.time()
    qcfg = QuantConfig(gpu_expert_bits=4, cpu_expert_bits=4)
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,
        quant_cfg=qcfg,
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    prompts = [
        "Write a short poem about the ocean.",
        "Explain why the sky is blue in one paragraph.",
        "Count from 1 to 10 and explain each number briefly.",
    ]

    for run in range(NUM_RUNS):
        messages = [{"role": "user", "content": prompts[run % len(prompts)]}]
        prompt_tokens = model.tokenizer.apply_chat_template(messages)
        print(f"\nRun {run+1}: prompt={len(prompt_tokens)} tokens, generating {NUM_DECODE_TOKENS}...")

        # Use model.generate() which goes through Rust CPU decode path
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=NUM_DECODE_TOKENS,
            temperature=0.6,
            top_k=50,
            top_p=0.95,
        )

        # model.generate() stores timing internally
        decode_tps = model._last_decode_tok_s
        decode_time = model._last_decode_time
        n_decode = len(generated) - 1  # first token from prefill
        ms_per_tok = (decode_time / n_decode * 1000) if n_decode > 0 else 0

        text = model.tokenizer.decode(generated)
        print(f"  {decode_tps:.2f} tok/s ({ms_per_tok:.1f}ms/tok, {n_decode} decode tokens)")
        if run == 0:
            print(f"  Output: {repr(text[:120])}")

    print("\nDone.")

if __name__ == "__main__":
    main()

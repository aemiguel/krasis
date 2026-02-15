#!/usr/bin/env python3
"""Quick decode benchmark for Qwen3-Coder-Next: short prompt, decode timing."""

import logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.timing import TIMING

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
NUM_DECODE_TOKENS = 32
NUM_RUNS = 3

def main():
    # Enable decode timing for instrumented run
    instrument = os.environ.get("INSTRUMENT", "0") == "1"
    if instrument:
        TIMING.decode = True
        print("INSTRUMENTED run (timing enabled)")
    else:
        print("CLEAN run (timing disabled)")

    print("Loading Qwen3-Coder-Next PP=2, active_only prefill + CPU decode (inlined)...")
    t0 = time.time()
    qcfg = QuantConfig(gpu_expert_bits=4, cpu_expert_bits=4)
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,  # GPU for M>=300, CPU for M=1
        expert_divisor=-1,  # active_only (fast prefill, no GPU decode)
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # Short prompt
    messages = [{"role": "user", "content": "Write a short poem about the ocean."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens, generating {NUM_DECODE_TOKENS} × {NUM_RUNS} runs")

    from krasis.sampler import sample
    from krasis.kv_cache import SequenceKVState

    for run in range(NUM_RUNS):
        # Reset states
        if model.cfg.is_hybrid:
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        seq_states = [
            SequenceKVState(c, seq_id=0) if c is not None else None
            for c in model.kv_caches
        ]
        device = torch.device(model.ranks[0].device)

        # Prefill
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
        positions = torch.arange(len(prompt), dtype=torch.int32, device=device)

        with torch.inference_mode():
            logits = model.forward(prompt_tensor, positions, seq_states)
        torch.cuda.synchronize()

        # Decode
        next_logits = logits[-1:, :]
        next_token = sample(next_logits, 0.6, 50, 1.0).item()
        tokens = [next_token]

        t_decode = time.perf_counter()
        with torch.inference_mode():
            for step in range(NUM_DECODE_TOKENS - 1):
                pos = len(prompt) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                next_token = sample(logits, 0.6, 50, 1.0).item()
                tokens.append(next_token)
        decode_time = time.perf_counter() - t_decode
        decode_tps = (NUM_DECODE_TOKENS - 1) / decode_time
        ms_per_tok = decode_time / (NUM_DECODE_TOKENS - 1) * 1000

        for s in seq_states:
            if s is not None:
                s.free()

        text = model.tokenizer.decode(tokens)
        print(f"\nRun {run+1}: {decode_tps:.2f} tok/s ({ms_per_tok:.1f}ms/tok)")
        if run == 0:
            print(f"  Output: {repr(text[:120])}")

    print("\nDone.")

if __name__ == "__main__":
    main()

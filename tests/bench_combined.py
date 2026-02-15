#!/usr/bin/env python3
"""Combined prefill+decode benchmark: active_only prefill + CPU decode (inlined)."""

import logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
PROMPT_FILE = "/home/main/Documents/Claude/benchmark_prompt.txt"
NUM_DECODE_TOKENS = 32

def main():
    with open(PROMPT_FILE) as f:
        prompt_text = f.read()

    print("Loading Qwen3-Coder-Next PP=2, active_only prefill + CPU decode (inlined)...")
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,  # GPU for M>=300, CPU for M=1
        expert_divisor=4,  # layer_grouped (fast prefill)
    )
    model.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    messages = [{"role": "user", "content": prompt_text}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens, generating {NUM_DECODE_TOKENS} tokens")

    from krasis.sampler import sample
    from krasis.kv_cache import SequenceKVState

    for run in range(2):
        if model.cfg.is_hybrid:
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        seq_states = [
            SequenceKVState(c, seq_id=0) if c is not None else None
            for c in model.kv_caches
        ]
        device = torch.device(model.ranks[0].device)
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
        positions = torch.arange(len(prompt), dtype=torch.int32, device=device)

        # Prefill
        t_prefill = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward(prompt_tensor, positions, seq_states)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - t_prefill
        prefill_tps = len(prompt) / prefill_time

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

        for s in seq_states:
            if s is not None:
                s.free()

        text = model.tokenizer.decode(tokens)
        print(f"\nRun {run+1}:")
        print(f"  Prefill: {prefill_tps:.1f} tok/s, TTFT={prefill_time:.1f}s")
        print(f"  Decode: {decode_tps:.2f} tok/s ({decode_time/(NUM_DECODE_TOKENS-1)*1000:.0f}ms/tok)")
        if run == 0:
            print(f"  Output: {repr(text[:120])}")

if __name__ == "__main__":
    main()

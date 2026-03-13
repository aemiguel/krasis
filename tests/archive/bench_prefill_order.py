#!/usr/bin/env python3
"""Test prefill with reversed order: threshold=1 first, then threshold=300."""

import logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
PROMPT_FILE = "/home/main/Documents/Claude/benchmark_prompt.txt"

def run_prefill(threshold, prompt_text):
    print(f"\nConfig: active_only, threshold={threshold}")
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

    messages = [{"role": "user", "content": prompt_text}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")

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

        t_prefill = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward(prompt_tensor, positions, seq_states)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - t_prefill
        prefill_tps = len(prompt) / prefill_time

        for s in seq_states:
            if s is not None:
                s.free()
        print(f"  Run {run+1}: {prefill_tps:.1f} tok/s, TTFT={prefill_time:.1f}s")

    del model
    torch.cuda.empty_cache()

with open(PROMPT_FILE) as f:
    prompt_text = f.read()

# REVERSED ORDER: threshold=1 first, threshold=300 second
run_prefill(1, prompt_text)
run_prefill(300, prompt_text)

#!/usr/bin/env python3
"""Benchmark Qwen3-Coder-Next with ~8K token prompt: prefill + GPU decode."""

import logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"
PROMPT_FILE = "/home/main/Documents/Claude/benchmark_prompt.txt"
NUM_DECODE_TOKENS = 32
NUM_RUNS = 3


def main():
    # Load prompt
    with open(PROMPT_FILE) as f:
        prompt_text = f.read()

    # Load model — threshold=300 for prefill, M=1 decode always routes through GPU
    print("Loading Qwen3-Coder-Next PP=2, active_only + static_pin, threshold=300...")
    t0 = time.time()
    qcfg = QuantConfig(gpu_expert_bits=4, cpu_expert_bits=4)
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,  # CPU path for short, GPU for 300+
        expert_divisor=-1,
    )
    model.load()

    # Configure static pinning
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")

    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # Tokenize
    messages = [{"role": "user", "content": prompt_text}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")
    print(f"Generating {NUM_DECODE_TOKENS} tokens × {NUM_RUNS} runs")
    print()

    # Warmup (triggers pinning)
    print("Warmup...")
    warmup_prompt = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}])
    model.generate(warmup_prompt, max_new_tokens=8, temperature=0.0, top_k=1)
    print("Warmup done, experts pinned")

    for i in range(min(2, torch.cuda.device_count())):
        alloc = torch.cuda.memory_allocated(i) / (1024**2)
        print(f"  GPU{i}: {alloc:.0f} MB")
    print()

    # Benchmark
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

        t_prefill = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward(prompt_tensor, positions, seq_states)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - t_prefill
        prefill_tps = len(prompt) / prefill_time

        # Decode
        next_logits = logits[-1:, :]
        next_token = sample(next_logits, 0.0, 1, 1.0).item()
        tokens = [next_token]

        t_decode = time.perf_counter()
        with torch.inference_mode():
            for step in range(NUM_DECODE_TOKENS - 1):
                pos = len(prompt) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                next_token = sample(logits, 0.0, 1, 1.0).item()
                tokens.append(next_token)
        decode_time = time.perf_counter() - t_decode
        decode_tps = (NUM_DECODE_TOKENS - 1) / decode_time
        ms_per_tok = decode_time / (NUM_DECODE_TOKENS - 1) * 1000

        # Free KV
        for s in seq_states:
            if s is not None:
                s.free()

        text = model.tokenizer.decode(tokens)

        print(f"Run {run+1}:")
        print(f"  TTFT (prefill): {prefill_time:.1f}s ({prefill_tps:.1f} tok/s)")
        print(f"  Decode: {decode_tps:.2f} tok/s ({ms_per_tok:.0f}ms/tok)")
        if run == 0:
            print(f"  Output: {repr(text[:120])}")

    # Stats
    for dev_str, manager in model.gpu_prefill_managers.items():
        stats = manager.get_ao_stats()
        print(f"  {dev_str}: pinned={len(manager._pinned)}, "
              f"hits={stats['cache_hits']}, misses={stats['cache_misses']}")


if __name__ == "__main__":
    main()

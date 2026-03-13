#!/usr/bin/env python3
"""Generic model benchmark: reports quant config, prefill + decode speed.

Usage:
    python tests/bench_model.py --model models/DeepSeek-V2-Lite --pp 14,13
    python tests/bench_model.py --model models/Qwen3-Coder-Next --pp 24,24
    INSTRUMENT=1 python tests/bench_model.py --model ...   # with timing
"""

import argparse, logging, sys, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.timing import TIMING

NUM_DECODE_TOKENS = 32
NUM_RUNS = 3

def main():
    parser = argparse.ArgumentParser(description="Krasis model benchmark")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--pp", required=True, help="PP partition (e.g. '14,13' or '24,24')")
    parser.add_argument("--gpu-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--cpu-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--kv-dtype", default="fp8", choices=["fp8", "bf16"])
    parser.add_argument("--attn-quant", default="awq", choices=["bf16", "awq"])
    parser.add_argument("--divisor", type=int, default=-1, help="expert_divisor (-1=active_only)")
    parser.add_argument("--threshold", type=int, default=300, help="gpu_prefill_threshold")
    parser.add_argument("--tokens", type=int, default=NUM_DECODE_TOKENS)
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    parser.add_argument("--prefill-prompt", default=None, help="Path to large prompt file for prefill test")
    args = parser.parse_args()

    instrument = os.environ.get("INSTRUMENT", "0") == "1"
    if instrument:
        TIMING.decode = True
        print("INSTRUMENTED run (timing enabled, CUDA graphs disabled)")
    else:
        print("CLEAN run (timing disabled, CUDA graphs active)")

    pp = [int(x.strip()) for x in args.pp.split(",")]
    devices = [f"cuda:{i}" for i in range(len(pp))]
    kv_dtype = torch.float8_e4m3fn if args.kv_dtype == "fp8" else torch.bfloat16

    qcfg = QuantConfig(
        gpu_expert_bits=args.gpu_bits,
        cpu_expert_bits=args.cpu_bits,
        attention=args.attn_quant,
        shared_expert=args.attn_quant,
        dense_mlp=args.attn_quant,
        lm_head=args.attn_quant,
    )

    model_name = os.path.basename(args.model.rstrip("/"))
    print(f"\n{'='*60}")
    print(f"Model:      {model_name}")
    print(f"PP:         {pp} ({len(pp)} GPUs)")
    print(f"GPU expert: INT{args.gpu_bits} (Marlin)")
    print(f"CPU expert: INT{args.cpu_bits}")
    print(f"Attention:  {args.attn_quant}")
    print(f"KV cache:   {'FP8 E4M3' if args.kv_dtype == 'fp8' else 'BF16'}")
    print(f"Divisor:    {args.divisor}")
    print(f"Threshold:  {args.threshold}")
    print(f"{'='*60}\n")

    print("Loading model...")
    t0 = time.time()
    model = KrasisModel(
        args.model,
        pp_partition=pp,
        devices=devices,
        kv_dtype=kv_dtype,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=args.threshold,
        expert_divisor=args.divisor,
        quant_cfg=qcfg,
    )
    model.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Report VRAM
    for i in range(len(pp)):
        alloc = torch.cuda.memory_allocated(i) / 1e6
        print(f"  GPU{i} VRAM: {alloc:.0f} MB")

    from krasis.sampler import sample
    from krasis.kv_cache import SequenceKVState

    # Short prompt for decode test
    messages = [{"role": "user", "content": "Write a short poem about the ocean."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"\nDecode test: {len(prompt)} tok prompt, {args.tokens} decode tokens × {args.runs} runs")

    for run in range(args.runs):
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

        with torch.inference_mode():
            logits = model.forward(prompt_tensor, positions, seq_states)
        torch.cuda.synchronize()

        next_token = sample(logits[-1:, :], 0.6, 50, 1.0).item()
        tokens = [next_token]

        t_decode = time.perf_counter()
        with torch.inference_mode():
            for step in range(args.tokens - 1):
                pos = len(prompt) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                next_token = sample(logits, 0.6, 50, 1.0).item()
                tokens.append(next_token)
        decode_time = time.perf_counter() - t_decode
        decode_tps = (args.tokens - 1) / decode_time
        ms_per_tok = decode_time / (args.tokens - 1) * 1000

        for s in seq_states:
            if s is not None:
                s.free()

        text = model.tokenizer.decode(tokens)
        print(f"  Run {run+1}: {decode_tps:.2f} tok/s ({ms_per_tok:.1f}ms/tok)")
        if run == 0:
            print(f"    Output: {repr(text[:120])}")

    # Prefill test if large prompt provided
    if args.prefill_prompt:
        with open(args.prefill_prompt) as f:
            prompt_text = f.read()
        messages = [{"role": "user", "content": prompt_text}]
        large_prompt = model.tokenizer.apply_chat_template(messages)
        print(f"\nPrefill test: {len(large_prompt)} tokens × {args.runs} runs")

        for run in range(args.runs):
            if model.cfg.is_hybrid:
                for layer in model.layers:
                    if layer.layer_type == "linear_attention":
                        layer.attention.reset_state()

            seq_states = [
                SequenceKVState(c, seq_id=0) if c is not None else None
                for c in model.kv_caches
            ]
            device = torch.device(model.ranks[0].device)
            prompt_tensor = torch.tensor(large_prompt, dtype=torch.long, device=device)
            positions = torch.arange(len(large_prompt), dtype=torch.int32, device=device)

            t_prefill = time.perf_counter()
            with torch.inference_mode():
                logits = model.forward(prompt_tensor, positions, seq_states)
            torch.cuda.synchronize()
            prefill_time = time.perf_counter() - t_prefill
            prefill_tps = len(large_prompt) / prefill_time

            for s in seq_states:
                if s is not None:
                    s.free()

            print(f"  Run {run+1}: {prefill_tps:.1f} tok/s, TTFT={prefill_time:.2f}s")

    print("\nDone.")

if __name__ == "__main__":
    main()

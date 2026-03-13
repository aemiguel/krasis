#!/usr/bin/env python3
"""Measure prefill speed at different prompt lengths to identify fixed costs and scaling.

Outputs results to token-scaling-analysis.md in the repo root.
"""

import os
import sys
import time

# Add krasis python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.timing import TIMING

# Disable instrumentation for clean timing
TIMING.decode = False
TIMING.prefill = False
os.environ.pop("KRASIS_DECODE_TIMING", None)

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")
PROMPT_SIZES = [100, 500, 2000, 10000, 25000]
RUNS_PER_SIZE = 3


def build_prompt(model, target_tokens):
    """Build a prompt of approximately target_tokens length using Gutenberg text."""
    from prompt_utils import build_prompt_tokens
    return build_prompt_tokens(model, target_tokens)


def main():
    print("Loading model...")
    quant_cfg = QuantConfig(
        attention="awq", shared_expert="int8", dense_mlp="int8",
        lm_head="int8", gpu_expert_bits=4, cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=MODEL_PATH, num_gpus=1, layer_group_size=2,
        kv_dtype=torch.float8_e4m3fn, quant_cfg=quant_cfg,
        krasis_threads=48, gpu_prefill_threshold=300, kv_cache_mb=1000,
        stream_attention=True,
    )
    model.load()

    # Warmup CUDA runtime
    devices = [torch.device("cuda:0")]
    model.warmup_cuda_runtime(devices)

    # Warmup with a medium prompt
    print("Warmup...")
    warmup_tokens = build_prompt(model, 2000)
    for _ in range(3):
        with torch.inference_mode():
            model.generate(warmup_tokens, max_new_tokens=1, temperature=0.6)

    # Run scaling test
    results = []
    for size in PROMPT_SIZES:
        print(f"\n--- {size} tokens ---")
        tokens = build_prompt(model, size)
        actual_len = len(tokens)
        print(f"  Actual tokens: {actual_len}")

        runs = []
        for r in range(RUNS_PER_SIZE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.generate(tokens, max_new_tokens=1, temperature=0.6)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ttft = getattr(model, '_last_ttft', t1 - t0)
            wall = t1 - t0
            tok_s = actual_len / ttft if ttft > 0 else 0
            runs.append({"ttft": ttft, "wall": wall, "tok_s": tok_s})
            print(f"  Run {r+1}: TTFT={ttft:.3f}s, wall={wall:.3f}s, {tok_s:.0f} tok/s")

        avg_ttft = sum(r["ttft"] for r in runs) / len(runs)
        avg_wall = sum(r["wall"] for r in runs) / len(runs)
        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs)

        results.append({
            "target": size,
            "actual": actual_len,
            "avg_ttft": avg_ttft,
            "avg_wall": avg_wall,
            "avg_tok_s": avg_tok_s,
            "runs": runs,
        })
        print(f"  AVG: TTFT={avg_ttft:.3f}s, {avg_tok_s:.0f} tok/s")

    # Write results
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    out_path = os.path.join(repo_root, "token-scaling-analysis.md")

    lines = []
    lines.append("# Token Scaling Analysis — Qwen3-Coder-Next (1 GPU, INT4/INT4)")
    lines.append("")
    lines.append("Measures how prefill speed scales with prompt length.")
    lines.append("Identifies fixed per-prefill costs vs linear scaling.")
    lines.append("")
    lines.append(f"Config: 1x RTX 2000 Ada, layer_group_size=2, stream_attention, FP8 KV, {RUNS_PER_SIZE} runs/size")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"| {'Tokens':>8s} | {'TTFT (s)':>10s} | {'tok/s':>10s} | {'ms/tok':>10s} | {'Fixed cost est':>16s} |")
    lines.append(f"| {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*16} |")

    for r in results:
        ms_per_tok = (r["avg_ttft"] / r["actual"] * 1000) if r["actual"] > 0 else 0
        lines.append(
            f"| {r['actual']:>8,d} | {r['avg_ttft']:>10.3f} | {r['avg_tok_s']:>10.0f} | {ms_per_tok:>10.3f} | {'':>16s} |"
        )

    # Estimate fixed cost: extrapolate from two smallest sizes
    # TTFT = fixed + tokens * marginal_cost
    # Using 100 and 10000 token runs to estimate
    if len(results) >= 4:
        r_small = results[0]  # 100 tokens
        r_large = results[3]  # 10000 tokens
        marginal = (r_large["avg_ttft"] - r_small["avg_ttft"]) / (r_large["actual"] - r_small["actual"])
        fixed = r_small["avg_ttft"] - marginal * r_small["actual"]
        marginal_tok_s = 1.0 / marginal if marginal > 0 else 0

        lines.append("")
        lines.append("## Analysis")
        lines.append("")
        lines.append(f"Linear regression (100 → 10,000 tokens):")
        lines.append(f"- **Fixed cost per prefill**: {fixed*1000:.1f} ms")
        lines.append(f"- **Marginal cost per token**: {marginal*1000:.4f} ms ({marginal_tok_s:.0f} tok/s throughput)")
        lines.append(f"- **At 100 tokens**: fixed cost is {fixed/r_small['avg_ttft']*100:.0f}% of total time")
        lines.append(f"- **At 10K tokens**: fixed cost is {fixed/r_large['avg_ttft']*100:.0f}% of total time")

        if len(results) >= 5:
            r_25k = results[4]
            predicted_25k = fixed + marginal * r_25k["actual"]
            lines.append(f"- **At 25K tokens**: predicted {predicted_25k:.2f}s, actual {r_25k['avg_ttft']:.2f}s "
                         f"({'linear' if abs(predicted_25k - r_25k['avg_ttft']) / r_25k['avg_ttft'] < 0.15 else 'SUPERLINEAR'})")

    # Per-run details
    lines.append("")
    lines.append("## Per-Run Detail")
    lines.append("")
    for r in results:
        lines.append(f"### {r['actual']:,} tokens")
        for i, run in enumerate(r["runs"]):
            lines.append(f"- Run {i+1}: TTFT={run['ttft']:.3f}s, wall={run['wall']:.3f}s, {run['tok_s']:.0f} tok/s")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

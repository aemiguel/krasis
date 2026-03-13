#!/usr/bin/env python3
"""Benchmark hot_cached_static vs static_pin.

Compares 3 configs:
  1. static_pin (current best) — baseline
  2. hot_cached_static (no CUDA graphs)
  3. hot_cached_static (with CUDA graphs)

For each: 8K prompt prefill + 32 decode tokens × 3 runs.

Usage:
    # First, generate a heatmap (optional — HCS can work without one):
    python tests/bench_hot_cached_static.py --model-path /path/to/model --generate-heatmap

    # Then benchmark:
    python tests/bench_hot_cached_static.py --model-path /path/to/model

    # With custom heatmap:
    python tests/bench_hot_cached_static.py --model-path /path/to/model --heatmap-path heatmap.json
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

# Add parent to path for krasis imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krasis.model import KrasisModel
from krasis.config import QuantConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("bench_hcs")


def run_benchmark(model, prompt_tokens, num_decode=32, num_runs=3, label=""):
    """Run prefill + decode benchmark."""
    results = []

    for run in range(num_runs):
        # Force GC
        torch.cuda.empty_cache()

        # Prefill
        t_prefill_start = time.perf_counter()
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=num_decode,
            temperature=0.6,
            stop_token_ids=[],  # Don't stop early
        )
        t_end = time.perf_counter()

        total_time = t_end - t_prefill_start
        num_prompt = len(prompt_tokens)
        num_gen = len(generated)

        # Approximate prefill vs decode split:
        # TTFT is hard to measure externally, so we estimate
        # total = prefill_time + num_gen * decode_time_per_tok
        # We'll report total time and throughput
        if num_gen > 0:
            decode_tok_s = num_gen / total_time  # rough estimate (includes prefill)

        result = {
            "run": run + 1,
            "prompt_tokens": num_prompt,
            "generated_tokens": num_gen,
            "total_time_s": total_time,
            "approx_tok_s": num_gen / total_time if num_gen > 0 else 0,
        }
        results.append(result)

        logger.info(
            "%s run %d/%d: %d prompt, %d gen, %.1fs total, ~%.2f tok/s",
            label, run + 1, num_runs,
            num_prompt, num_gen, total_time,
            result["approx_tok_s"],
        )

    # Summary
    avg_time = sum(r["total_time_s"] for r in results) / len(results)
    avg_tok_s = sum(r["approx_tok_s"] for r in results) / len(results)
    logger.info(
        "%s AVERAGE: %.1fs total, ~%.2f tok/s",
        label, avg_time, avg_tok_s,
    )
    return results


def build_prompt(model, target_tokens=8000):
    """Build a prompt of approximately target_tokens length using Gutenberg text."""
    from prompt_utils import build_prompt_tokens
    tokens = build_prompt_tokens(model, target_tokens)
    logger.info("Prompt: %d tokens (target: %d)", len(tokens), target_tokens)
    return tokens


def generate_heatmap(model, output_path, num_warmup_prompts=3):
    """Generate expert activation heatmap from diverse prompts."""
    prompts = [
        "Write a comprehensive guide to Python programming.",
        "Explain quantum computing and its applications in cryptography.",
        "Describe the history of artificial intelligence from 1950 to present.",
    ]

    for i, text in enumerate(prompts[:num_warmup_prompts]):
        messages = [{"role": "user", "content": text}]
        tokens = model.tokenizer.apply_chat_template(messages)
        logger.info("Warmup %d/%d: %d tokens...", i + 1, num_warmup_prompts, len(tokens))
        model.generate(tokens, max_new_tokens=50, temperature=0.6)

    # Save heatmap from each manager
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.save_heatmap(output_path)
        logger.info("Heatmap saved to %s from %s", output_path, dev_str)
        break  # One heatmap is enough (same engine)


def main():
    parser = argparse.ArgumentParser(description="Benchmark hot_cached_static")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pp-partition", default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--heatmap-path", default=None,
                        help="Path to expert_heatmap.json (generated if --generate-heatmap)")
    parser.add_argument("--generate-heatmap", action="store_true",
                        help="Generate heatmap and exit")
    parser.add_argument("--prompt-tokens", type=int, default=8000)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip static_pin baseline")
    parser.add_argument("--skip-cuda-graphs", action="store_true",
                        help="Skip CUDA graph benchmark")
    args = parser.parse_args()

    pp_partition = None
    if args.pp_partition:
        pp_partition = [int(x.strip()) for x in args.pp_partition.split(",")]

    quant_cfg = QuantConfig(
        attention="awq", shared_expert="int8",
        dense_mlp="int8", lm_head="int8",
        gpu_expert_bits=4, cpu_expert_bits=4,
    )

    # ── Heatmap generation ──
    if args.generate_heatmap:
        heatmap_output = args.heatmap_path or "expert_heatmap.json"
        logger.info("=== Generating heatmap ===")

        model = KrasisModel(
            model_path=args.model_path,
            pp_partition=pp_partition,
            num_gpus=args.num_gpus,
            kv_dtype=torch.float8_e4m3fn,
            quant_cfg=quant_cfg,
            expert_divisor=-1,  # active_only for warmup
            gpu_prefill_threshold=1,
        )
        model.load()

        # Configure pinning to enable heatmap tracking
        for dev_str, manager in model.gpu_prefill_managers.items():
            manager.configure_pinning(budget_mb=0, warmup_requests=999, strategy="uniform")

        generate_heatmap(model, heatmap_output)
        del model
        torch.cuda.empty_cache()
        logger.info("Heatmap generation complete. Run without --generate-heatmap to benchmark.")
        return

    # ── Benchmark configs ──
    configs = []

    if not args.skip_baseline:
        configs.append({
            "label": "static_pin",
            "expert_divisor": -1,
            "cache_strategy": "static_pin",
            "heatmap_path": None,
            "cuda_graphs": False,
        })

    configs.append({
        "label": "hot_cached_static",
        "expert_divisor": -3,
        "cache_strategy": "hot_cached_static",
        "heatmap_path": args.heatmap_path,
        "cuda_graphs": False,
    })

    if not args.skip_cuda_graphs:
        configs.append({
            "label": "hot_cached_static+graphs",
            "expert_divisor": -3,
            "cache_strategy": "hot_cached_static",
            "heatmap_path": args.heatmap_path,
            "cuda_graphs": True,
        })

    all_results = {}

    for cfg in configs:
        logger.info("=" * 60)
        logger.info("=== Config: %s ===", cfg["label"])
        logger.info("=" * 60)

        model = KrasisModel(
            model_path=args.model_path,
            pp_partition=pp_partition,
            num_gpus=args.num_gpus,
            kv_dtype=torch.float8_e4m3fn,
            quant_cfg=quant_cfg,
            expert_divisor=cfg["expert_divisor"],
            gpu_prefill_threshold=1,
        )
        model.load()

        # Strategy-specific init
        if cfg["cache_strategy"] == "static_pin":
            for dev_str, manager in model.gpu_prefill_managers.items():
                manager.configure_pinning(budget_mb=0, warmup_requests=1, strategy="uniform")
        elif cfg["cache_strategy"] == "hot_cached_static":
            for dev_str, manager in model.gpu_prefill_managers.items():
                manager._init_hot_cached_static(heatmap_path=cfg["heatmap_path"])
                if cfg["cuda_graphs"]:
                    manager._init_cuda_graphs()

        # Build prompt
        prompt_tokens = build_prompt(model, target_tokens=args.prompt_tokens)

        # Warmup (for static_pin: triggers pinning)
        logger.info("Warmup run...")
        model.generate(prompt_tokens[:500], max_new_tokens=5, temperature=0.6)
        if cfg["cache_strategy"] == "static_pin":
            for dev_str, manager in model.gpu_prefill_managers.items():
                manager.notify_request_done()

        # Benchmark
        results = run_benchmark(
            model, prompt_tokens,
            num_decode=args.decode_tokens,
            num_runs=args.num_runs,
            label=cfg["label"],
        )
        all_results[cfg["label"]] = results

        # VRAM usage
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e6
            logger.info("GPU%d VRAM: %.1f MB", i, alloc)

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("=== RESULTS SUMMARY ===")
    logger.info("=" * 60)
    for label, results in all_results.items():
        avg_time = sum(r["total_time_s"] for r in results) / len(results)
        avg_tok_s = sum(r["approx_tok_s"] for r in results) / len(results)
        best_tok_s = max(r["approx_tok_s"] for r in results)
        logger.info(
            "%-30s  avg=%.1fs  avg_tok/s=%.2f  best_tok/s=%.2f",
            label, avg_time, avg_tok_s, best_tok_s,
        )


if __name__ == "__main__":
    main()

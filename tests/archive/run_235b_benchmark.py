#!/usr/bin/env python3
"""Benchmark Qwen3-235B-A22B with HCS hybrid strategy.

Measures TTFT, prefill tok/s, and decode tok/s separately using
model.generate() built-in timing instrumentation.

Strategy: layer_grouped prefill (auto-divisor) + HCS decode (zero DMA)
"""

import gc
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("bench_235b")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from krasis.model import KrasisModel
from krasis.config import QuantConfig

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-235B-A22B"
HEATMAP_PATH = "/home/main/Documents/Claude/krasis/tests/qwen235b_heatmap.json"
PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "benchmarks", "prompts", "prompt1_moby_dick.txt")
PP = [32, 31, 31]  # 94 layers across 3 GPUs
DEVICES = ["cuda:0", "cuda:1", "cuda:2"]
NUM_DECODE = 50  # Match existing benchmark


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def gpu_mem():
    return {i: torch.cuda.memory_allocated(i) / 1e6
            for i in range(torch.cuda.device_count())}


def build_prompt(model):
    """Load the standard benchmark prompt (~8600 tokens)."""
    with open(PROMPT_FILE) as f:
        prompt_text = f.read()

    messages = [{"role": "user", "content": prompt_text}]
    tokens = model.tokenizer.apply_chat_template(messages)
    print(f"  Prompt: {len(prompt_text)} chars → {len(tokens)} tokens")
    return tokens


def benchmark_long_prompt(model, prompt_tokens, num_decode=50, num_runs=2, label=""):
    """Benchmark with 8K+ prompt — uses model's built-in TTFT/decode timing."""
    results = []

    for run in range(num_runs):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        generated = model.generate(
            prompt_tokens,
            max_new_tokens=num_decode,
            temperature=0.6,
            top_k=50,
        )
        torch.cuda.synchronize()

        n_prompt = len(prompt_tokens)
        n_gen = len(generated)
        ttft = getattr(model, '_last_ttft', 0.0)
        decode_tok_s = getattr(model, '_last_decode_tok_s', 0.0)
        decode_time = getattr(model, '_last_decode_time', 0.0)
        prefill_tok_s = n_prompt / ttft if ttft > 0 else 0

        result = {
            "run": run + 1,
            "prompt_tokens": n_prompt,
            "generated_tokens": n_gen,
            "ttft_s": ttft,
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
            "decode_time_s": decode_time,
        }
        results.append(result)
        print(f"  [{label}] run {run+1}/{num_runs}: {n_prompt} prompt, {n_gen} gen")
        print(f"    TTFT={ttft:.1f}s, prefill={prefill_tok_s:.1f} tok/s, decode={decode_tok_s:.2f} tok/s")

        # Print generated text (first 200 chars) to verify correctness
        text = model.tokenizer.decode(generated)
        print(f"    Output: {text[:200]}...")

    return results


def benchmark_decode_only(model, num_decode=50, num_runs=3, label=""):
    """Short prompt to measure pure decode speed."""
    messages = [{"role": "user", "content": "Hello"}]
    tokens = model.tokenizer.apply_chat_template(messages)

    results = []
    for run in range(num_runs):
        torch.cuda.synchronize()

        generated = model.generate(
            tokens,
            max_new_tokens=num_decode,
            temperature=0.6,
            top_k=50,
        )
        torch.cuda.synchronize()

        n_gen = len(generated)
        decode_tok_s = getattr(model, '_last_decode_tok_s', 0.0)
        decode_time = getattr(model, '_last_decode_time', 0.0)

        results.append({
            "run": run + 1,
            "generated": n_gen,
            "decode_tok_s": decode_tok_s,
            "decode_time_s": decode_time,
        })
        print(f"  [{label}] run {run+1}/{num_runs}: {n_gen} tok, {decode_tok_s:.2f} tok/s")

    avg = sum(r["decode_tok_s"] for r in results) / len(results)
    best = max(r["decode_tok_s"] for r in results)
    print(f"  [{label}] AVG: {avg:.2f} tok/s (best: {best:.2f})")
    return results


# ── Step 1: Generate heatmap ──

def step_generate_heatmap():
    print("\n" + "=" * 70)
    print("STEP 1: Generate expert activation heatmap")
    print("=" * 70)

    clear_gpu()
    model = KrasisModel(
        MODEL,
        pp_partition=PP,
        devices=DEVICES,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,
        expert_divisor=-1,  # active_only
    )
    model.load()

    # Configure pinning to track heatmap (high warmup so it doesn't pin yet)
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.configure_pinning(budget_mb=0, warmup_requests=9999, strategy="uniform")

    # Run diverse prompts to build heatmap
    prompts = [
        "Write a comprehensive guide to Python programming including decorators, generators, and async/await patterns.",
        "Explain quantum computing and its applications in cryptography, drug discovery, and optimization problems.",
        "Describe the complete history of artificial intelligence from Alan Turing to modern large language models.",
        "Write a detailed technical analysis of GPU architecture including CUDA cores, memory hierarchy, and parallelism.",
        "Explain the mathematical foundations of transformer models including attention mechanisms and positional encodings.",
    ]

    for i, text in enumerate(prompts):
        messages = [{"role": "user", "content": text}]
        tokens = model.tokenizer.apply_chat_template(messages)
        print(f"  Warmup {i+1}/{len(prompts)}: {len(tokens)} prompt tokens...")
        t0 = time.time()
        gen = model.generate(tokens, max_new_tokens=32, temperature=0.6)
        print(f"    Generated {len(gen)} tokens in {time.time()-t0:.1f}s")

    # Save heatmap from first manager
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager.save_heatmap(HEATMAP_PATH)
        heatmap_size = len(manager._heatmap)
        print(f"  Heatmap saved: {heatmap_size} entries to {HEATMAP_PATH}")
        break

    del model
    clear_gpu()
    print("  Heatmap generation complete.\n")


# ── Step 2: HCS hybrid benchmark ──

def step_hcs_hybrid():
    print("\n" + "=" * 70)
    print("STEP 2: HCS hybrid (layer_grouped prefill + HCS decode) PP=3")
    print("=" * 70)

    clear_gpu()
    model = KrasisModel(
        MODEL,
        pp_partition=PP,
        devices=DEVICES,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=1,  # GPU decode
        expert_divisor=-3,  # hot_cached_static
    )
    model.load()

    # Initialize HCS with heatmap
    for dev_str, manager in model.gpu_prefill_managers.items():
        manager._init_hot_cached_static(heatmap_path=HEATMAP_PATH)
        n_layers = len(manager._hcs_buffers)
        n_experts = sum(manager._hcs_num_pinned.values())
        print(f"    {dev_str}: {n_experts} hot experts across {n_layers} layers")

    mem = gpu_mem()
    print(f"  VRAM: " + ", ".join(f"GPU{i}={mem[i]:.0f} MB" for i in sorted(mem)))

    # ── Pure decode speed (skip if QUICK mode) ──
    quick = os.environ.get("QUICK", "") == "1"
    if quick:
        print("\n  --- QUICK mode: skipping pure decode ---")
        decode_results = []
    else:
        print("\n  --- Pure decode speed ---")
        decode_results = benchmark_decode_only(model, num_decode=NUM_DECODE, num_runs=3, label="hcs_hybrid")

    # ── 8600 token prompt (TTFT + decode) ──
    num_runs = 1 if quick else 2
    num_decode = 10 if quick else NUM_DECODE
    print(f"\n  --- 8600 token prompt ({num_runs} runs, {num_decode} decode) ---")
    prompt = build_prompt(model)
    long_results = benchmark_long_prompt(model, prompt, num_decode=num_decode, num_runs=num_runs, label="hcs_hybrid-8.6K")

    del model
    clear_gpu()

    decode_avg = sum(r["decode_tok_s"] for r in decode_results) / len(decode_results) if decode_results else 0
    decode_best = max((r["decode_tok_s"] for r in decode_results), default=0)

    return {
        "decode": decode_results,
        "long": long_results,
        "vram": mem,
        "decode_avg": decode_avg,
        "decode_best": decode_best,
    }


# ── Main ──

def main():
    print("=" * 70)
    print("QWEN3-235B-A22B BENCHMARK: HCS Hybrid")
    print("  Strategy: layer_grouped prefill + HCS decode")
    print(f"  PP=3 ({PP[0]}+{PP[1]}+{PP[2]}), 3x RTX 2000 Ada 16GB")
    print("=" * 70)

    # Step 1: Generate heatmap if needed
    if not os.path.exists(HEATMAP_PATH):
        step_generate_heatmap()
    else:
        print(f"\nHeatmap exists at {HEATMAP_PATH}, skipping generation.")
        with open(HEATMAP_PATH) as f:
            hm = json.load(f)
        print(f"  {len(hm)} entries")

    # Step 2: HCS hybrid benchmark
    results = step_hcs_hybrid()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — Qwen3-235B-A22B HCS Hybrid PP=3")
    print("  Strategy: layer_grouped prefill + HCS decode (zero DMA)")
    print("=" * 70)
    print(f"  Decode speed: {results['decode_avg']:.2f} avg / {results['decode_best']:.2f} best tok/s")

    for r in results["long"]:
        print(f"  8.6K prompt run {r['run']}: TTFT={r['ttft_s']:.1f}s, prefill={r['prefill_tok_s']:.1f} tok/s, decode={r['decode_tok_s']:.2f} tok/s")

    vram = results["vram"]
    vram_str = "+".join(f"{vram[i]:.0f}" for i in sorted(vram))
    print(f"  VRAM: {vram_str} MB")
    print("=" * 70)

    # Save results
    results_path = "/home/main/Documents/Claude/krasis/tests/qwen235b_hcs_results.json"
    save_data = {
        "decode": results["decode"],
        "long": results["long"],
        "vram": {str(k): v for k, v in results["vram"].items()},
        "decode_avg": results["decode_avg"],
        "decode_best": results["decode_best"],
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

"""Test QuantConfig: BF16 attention + INT8 Marlin GPU prefill vs defaults.

Runs V2-Lite with three configs:
1. Default (INT8 attention, INT4 Marlin GPU prefill) — baseline
2. BF16 attention + INT8 Marlin GPU prefill — new config
3. CPU-only (no GPU prefill) — reference

Reports VRAM, prefill speed, decode speed, output quality.
"""

import gc
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_quant_config")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

# Long prompt for GPU prefill testing
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 30
SHORT_PROMPT = "The capital of France is"


def measure_gpu_memory(device="cuda:0"):
    """Return (allocated_mb, reserved_mb)."""
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.max_memory_allocated(device) / (1024**2)
    return alloc, reserved


def run_config(label, quant_cfg=None, gpu_prefill=True):
    """Run V2-Lite with given config, return results dict."""
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel

    logger.info("=" * 60)
    logger.info("CONFIG: %s", label)
    if quant_cfg:
        logger.info("  QuantConfig: %s", quant_cfg)
    logger.info("  gpu_prefill: %s", gpu_prefill)
    logger.info("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    mem_before = torch.cuda.memory_allocated("cuda:0") / (1024**2)

    # Load model
    load_start = time.perf_counter()
    model = KrasisModel(
        MODEL_PATH,
        num_gpus=1,
        gpu_prefill=gpu_prefill,
        gpu_prefill_threshold=100,
        quant_cfg=quant_cfg,
    )
    model.load()
    load_time = time.perf_counter() - load_start

    mem_after_load, peak_load = measure_gpu_memory()
    weight_mem = mem_after_load - mem_before

    logger.info("Load time: %.1fs", load_time)
    logger.info("GPU weight memory: %.0f MB", weight_mem)

    results = {
        "label": label,
        "weight_mem_mb": weight_mem,
        "load_time_s": load_time,
    }

    # --- Long prompt (GPU prefill path) ---
    messages = [{"role": "user", "content": LONG_PROMPT + "\n\nSummarize in one sentence."}]
    prompt_tokens = model.tokenizer.apply_chat_template(messages)
    num_prompt = len(prompt_tokens)
    logger.info("Long prompt: %d tokens", num_prompt)

    start = time.perf_counter()
    gen_long = model.generate(prompt_tokens, max_new_tokens=30, temperature=0.0)
    total_long = time.perf_counter() - start

    text_long = model.tokenizer.decode(gen_long)
    logger.info("Long prompt output (%d tokens, %.1fs): %s", len(gen_long), total_long, text_long[:200])

    # Estimate prefill vs decode split
    # First token = prefill, rest = decode
    if len(gen_long) > 1:
        est_decode_per_tok = (total_long) / len(gen_long)  # rough average
    else:
        est_decode_per_tok = total_long

    results["long_prompt_tokens"] = num_prompt
    results["long_gen_tokens"] = len(gen_long)
    results["long_total_s"] = total_long
    results["long_output"] = text_long[:200]
    results["long_gen_token_ids"] = gen_long[:20]

    # --- Short prompt (decode-dominated) ---
    short_tokens = model.tokenizer.encode(SHORT_PROMPT)
    logger.info("Short prompt: %d tokens", len(short_tokens))

    start = time.perf_counter()
    gen_short = model.generate(short_tokens, max_new_tokens=30, temperature=0.0)
    total_short = time.perf_counter() - start

    text_short = model.tokenizer.decode(gen_short)
    logger.info("Short prompt output (%d tokens, %.1fs): %s", len(gen_short), total_short, text_short[:200])

    if len(gen_short) > 1:
        decode_speed = (len(gen_short) - 1) / (total_short)  # rough
    else:
        decode_speed = 0

    results["short_gen_tokens"] = len(gen_short)
    results["short_total_s"] = total_short
    results["short_output"] = text_short[:200]
    results["short_gen_token_ids"] = gen_short[:20]
    results["decode_tok_s"] = decode_speed

    # Peak memory
    _, peak = measure_gpu_memory()
    results["peak_mem_mb"] = peak

    logger.info("Peak GPU memory: %.0f MB", peak)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def compare_results(all_results):
    """Print comparison table."""
    print()
    print("=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    # Header
    labels = [r["label"] for r in all_results]
    print(f"{'Metric':<35}", end="")
    for label in labels:
        print(f"  {label:<20}", end="")
    print()
    print("-" * (35 + 22 * len(labels)))

    # Rows
    metrics = [
        ("GPU weight memory (MB)", "weight_mem_mb", ".0f"),
        ("Peak GPU memory (MB)", "peak_mem_mb", ".0f"),
        ("Load time (s)", "load_time_s", ".1f"),
        ("Long prompt total (s)", "long_total_s", ".1f"),
        ("Long prompt tokens", "long_prompt_tokens", "d"),
        ("Long gen tokens", "long_gen_tokens", "d"),
        ("Short prompt total (s)", "short_total_s", ".1f"),
        ("Short gen tokens", "short_gen_tokens", "d"),
        ("~Decode speed (tok/s)", "decode_tok_s", ".1f"),
    ]

    for name, key, fmt in metrics:
        print(f"{name:<35}", end="")
        for r in all_results:
            val = r.get(key, "N/A")
            if val != "N/A":
                print(f"  {val:<20{fmt}}", end="")
            else:
                print(f"  {'N/A':<20}", end="")
        print()

    # Output comparison
    print()
    print("OUTPUT COMPARISON (Long prompt, first 20 token IDs):")
    print("-" * 80)
    for r in all_results:
        print(f"  {r['label']}: {r.get('long_gen_token_ids', [])}")

    print()
    print("OUTPUT COMPARISON (Short prompt, first 20 token IDs):")
    print("-" * 80)
    for r in all_results:
        print(f"  {r['label']}: {r.get('short_gen_token_ids', [])}")

    # Token match analysis
    if len(all_results) >= 2:
        print()
        print("TOKEN MATCH (Long prompt):")
        base = all_results[0]["long_gen_token_ids"]
        for r in all_results[1:]:
            other = r["long_gen_token_ids"]
            match = sum(1 for a, b in zip(base, other) if a == b)
            total = min(len(base), len(other))
            print(f"  {all_results[0]['label']} vs {r['label']}: {match}/{total} match")

        print()
        print("TOKEN MATCH (Short prompt):")
        base = all_results[0]["short_gen_token_ids"]
        for r in all_results[1:]:
            other = r["short_gen_token_ids"]
            match = sum(1 for a, b in zip(base, other) if a == b)
            total = min(len(base), len(other))
            print(f"  {all_results[0]['label']} vs {r['label']}: {match}/{total} match")

    print()
    print("TEXT OUTPUTS (Short prompt):")
    print("-" * 80)
    for r in all_results:
        print(f"  {r['label']}: {r['short_output']}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    from krasis.config import QuantConfig

    all_results = []

    # Config 1: Default (INT8 attn, INT4 GPU experts) — baseline
    r1 = run_config("Default(INT8+INT4)")
    all_results.append(r1)

    # Config 2: BF16 attention + INT8 Marlin GPU prefill
    qcfg = QuantConfig(attention="bf16", gpu_expert_bits=8)
    r2 = run_config("BF16attn+INT8exp", quant_cfg=qcfg)
    all_results.append(r2)

    # Config 3: CPU-only decode (no GPU prefill) for reference
    r3 = run_config("CPU-only", gpu_prefill=False)
    all_results.append(r3)

    compare_results(all_results)

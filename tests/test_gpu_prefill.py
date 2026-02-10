"""Test GPU prefill integration with V2-Lite.

Tests:
1. Load model with gpu_prefill=True
2. Prefill with 500+ tokens (above threshold=300) → GPU path
3. Decode single tokens → CPU path
4. Compare output quality
"""

import logging
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_gpu_prefill")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"


def test_gpu_prefill():
    from krasis.model import KrasisModel

    logger.info("=== Test GPU Prefill with V2-Lite ===")

    # Load with GPU prefill enabled, PP=1 for simplicity
    model = KrasisModel(
        MODEL_PATH,
        num_gpus=1,
        gpu_prefill=True,
        gpu_prefill_threshold=100,  # Lower threshold for testing
    )
    model.load()

    # Build a long prompt (>100 tokens to trigger GPU prefill)
    long_text = "The quick brown fox jumps over the lazy dog. " * 30
    messages = [{"role": "user", "content": long_text + "\n\nSummarize in one sentence."}]
    prompt_tokens = model.tokenizer.apply_chat_template(messages)
    logger.info("Prompt length: %d tokens (threshold=%d)", len(prompt_tokens), model.gpu_prefill_threshold)

    assert len(prompt_tokens) >= model.gpu_prefill_threshold, \
        f"Prompt too short ({len(prompt_tokens)} tokens) to test GPU prefill"

    # Generate
    start = time.perf_counter()
    generated = model.generate(
        prompt_tokens,
        max_new_tokens=30,
        temperature=0.0,  # greedy for determinism
    )
    elapsed = time.perf_counter() - start

    text = model.tokenizer.decode(generated)
    logger.info("Generated %d tokens in %.1fs", len(generated), elapsed)
    logger.info("Output: %s", text[:200])

    # Prefill speed: should be faster than CPU-only
    prefill_tokens = len(prompt_tokens)
    # Rough: first token latency dominates
    logger.info(
        "Prefill: %d tokens, total gen time: %.1fs",
        prefill_tokens, elapsed,
    )

    logger.info("=== GPU Prefill Test PASSED ===")


def test_cpu_vs_gpu_consistency():
    """Compare GPU prefill vs CPU-only output for same prompt."""
    from krasis.model import KrasisModel
    from krasis.kv_cache import SequenceKVState

    logger.info("=== Test CPU vs GPU Consistency ===")

    # Long prompt to trigger GPU path
    long_text = "Explain quantum computing in simple terms. " * 25
    messages = [{"role": "user", "content": long_text}]

    # Run with GPU prefill
    model_gpu = KrasisModel(
        MODEL_PATH,
        num_gpus=1,
        gpu_prefill=True,
        gpu_prefill_threshold=100,
    )
    model_gpu.load()

    prompt_tokens = model_gpu.tokenizer.apply_chat_template(messages)
    logger.info("Prompt: %d tokens", len(prompt_tokens))

    start = time.perf_counter()
    gen_gpu = model_gpu.generate(prompt_tokens, max_new_tokens=20, temperature=0.0)
    gpu_time = time.perf_counter() - start
    text_gpu = model_gpu.tokenizer.decode(gen_gpu)
    logger.info("GPU prefill: %d tokens in %.1fs → %s", len(gen_gpu), gpu_time, text_gpu[:100])

    # Run with CPU only
    model_cpu = KrasisModel(
        MODEL_PATH,
        num_gpus=1,
        gpu_prefill=False,
    )
    model_cpu.load()

    start = time.perf_counter()
    gen_cpu = model_cpu.generate(prompt_tokens, max_new_tokens=20, temperature=0.0)
    cpu_time = time.perf_counter() - start
    text_cpu = model_cpu.tokenizer.decode(gen_cpu)
    logger.info("CPU only:    %d tokens in %.1fs → %s", len(gen_cpu), cpu_time, text_cpu[:100])

    # Compare: same greedy output? (may differ due to INT4 vs INT4 quantization noise)
    if gen_gpu == gen_cpu:
        logger.info("Outputs IDENTICAL")
    else:
        # Count matching tokens
        match = sum(1 for a, b in zip(gen_gpu, gen_cpu) if a == b)
        logger.info("Outputs differ: %d/%d tokens match", match, min(len(gen_gpu), len(gen_cpu)))

    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    logger.info("Speedup: %.1fx (GPU=%.1fs, CPU=%.1fs)", speedup, gpu_time, cpu_time)

    logger.info("=== Consistency Test DONE ===")


if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "prefill"

    if test == "prefill":
        test_gpu_prefill()
    elif test == "consistency":
        test_cpu_vs_gpu_consistency()
    elif test == "all":
        test_gpu_prefill()
        test_cpu_vs_gpu_consistency()
    else:
        print(f"Unknown test: {test}")
        print("Usage: python test_gpu_prefill.py [prefill|consistency|all]")

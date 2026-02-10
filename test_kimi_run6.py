#!/usr/bin/env python3
"""Run 6: Test Kimi K2.5 with Bug #4 fix (shared expert double-application removed)."""

import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run6")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

def main():
    import torch

    logger.info("=== Run 6: Kimi K2.5 — Bug #4 fix test ===")
    logger.info("GPU count: %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info("GPU%d: %s, %.1f GB", i, props.name, props.total_memory / 1e9)

    from krasis.model import KrasisModel

    t0 = time.perf_counter()
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=2,
        gpu_prefill=False,  # CPU-only for now (faster to test)
        krasis_threads=16,
    )
    logger.info("Loading model...")
    model.load()
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", load_time)

    # Report VRAM
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info("GPU%d VRAM: %.2f GB", i, alloc)

    # Test 1: Simple prompt
    logger.info("--- Test 1: Short prompt ---")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    t0 = time.perf_counter()
    response = model.chat(messages, max_new_tokens=32, temperature=0.0)
    elapsed = time.perf_counter() - t0
    logger.info("Response (%.1fs): %s", elapsed, repr(response))

    # Test 2: Slightly longer prompt
    logger.info("--- Test 2: Reasoning prompt ---")
    messages = [{"role": "user", "content": "Explain why the sky is blue in exactly two sentences."}]
    t0 = time.perf_counter()
    response = model.chat(messages, max_new_tokens=128, temperature=0.0)
    elapsed = time.perf_counter() - t0
    logger.info("Response (%.1fs): %s", elapsed, repr(response))

    # Test 3: Code prompt
    logger.info("--- Test 3: Code prompt ---")
    messages = [{"role": "user", "content": "Write a Python function that checks if a number is prime. Just the code, no explanation."}]
    t0 = time.perf_counter()
    response = model.chat(messages, max_new_tokens=256, temperature=0.0)
    elapsed = time.perf_counter() - t0
    logger.info("Response (%.1fs): %s", elapsed, repr(response))

    logger.info("=== Run 6 complete ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU decode timing for Qwen3.5-35B-A3B INT4/INT4."""

import sys
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("decode_timing")

# Enable timing with frequent reports
os.environ["KRASIS_CPU_DECODE_TIMING"] = "1"
os.environ["KRASIS_TIMING_INTERVAL"] = "20"

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3.5-35B-A3B")

def main():
    qcfg = QuantConfig(
        gpu_expert_bits=4,
        cpu_expert_bits=4,
        attention="bf16",
        lm_head="int8",
        shared_expert="int8",
        dense_mlp="int8",
    )

    logger.info("Loading Qwen3.5-35B-A3B model...")
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        layer_group_size=2,
        quant_cfg=qcfg,
    )
    model.load()
    logger.info("Model loaded")

    tokenizer = model.tokenizer

    # Warmup: short generation to get caches warm
    logger.info("=== WARMUP ===")
    warmup_ids = tokenizer.encode("Hello")
    model.generate(warmup_ids, max_new_tokens=10, temperature=0.1)

    # Test 1: Short prompt, long generation (pure decode timing)
    logger.info("=== TEST 1: Short prompt, 200 decode tokens ===")
    prompt1 = "Explain the concept of recursion in programming. Give examples and discuss base cases, stack frames, and tail recursion optimization. Be thorough."
    p1_ids = tokenizer.encode(prompt1)
    logger.info("Prompt tokens: %d", len(p1_ids))
    t0 = time.perf_counter()
    gen1 = model.generate(p1_ids, max_new_tokens=200, temperature=0.3)
    t1 = time.perf_counter()
    text1 = tokenizer.decode(gen1)
    decode_tokens = len(gen1) - 1  # first token from prefill
    decode_time = t1 - t0
    logger.info("Generated %d tokens in %.2fs (%.2f tok/s overall)", len(gen1), decode_time, len(gen1)/decode_time)
    logger.info("Output preview: %s", text1[:200])

    logger.info("=== DONE ===")

if __name__ == "__main__":
    main()

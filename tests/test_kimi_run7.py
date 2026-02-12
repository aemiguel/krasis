#!/usr/bin/env python3
"""Run 7: Test Kimi K2.5 with BF16 attention (isolate INT8 attention quality issue)."""

import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run7")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

def main():
    import torch
    from krasis.config import QuantConfig

    logger.info("=== Run 7: Kimi K2.5 — BF16 attention (isolating INT8 quality) ===")

    from krasis.model import KrasisModel

    # BF16 attention to isolate INT8 as potential quality issue
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    logger.info("QuantConfig: attention=%s, shared=%s, dense=%s, lm_head=%s",
                qcfg.attention, qcfg.shared_expert, qcfg.dense_mlp, qcfg.lm_head)

    t0 = time.perf_counter()
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=2,
        gpu_prefill=False,
        krasis_threads=16,
        quant_cfg=qcfg,
    )
    logger.info("Loading model...")
    model.load()
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", load_time)

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info("GPU%d VRAM: %.2f GB", i, alloc)

    # Also test with BF16 KV cache (eliminate FP8 as variable)
    logger.info("--- Test 1: Short prompt (BF16 attn, FP8 KV) ---")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    t0 = time.perf_counter()
    response = model.chat(messages, max_new_tokens=64, temperature=0.0)
    elapsed = time.perf_counter() - t0
    logger.info("Response (%.1fs): %s", elapsed, repr(response))

    logger.info("--- Test 2: Reasoning prompt ---")
    messages = [{"role": "user", "content": "Explain why the sky is blue in exactly two sentences."}]
    t0 = time.perf_counter()
    response = model.chat(messages, max_new_tokens=128, temperature=0.6)
    elapsed = time.perf_counter() - t0
    logger.info("Response (%.1fs): %s", elapsed, repr(response))

    logger.info("=== Run 7 complete ===")

if __name__ == "__main__":
    main()

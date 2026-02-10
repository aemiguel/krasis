#!/usr/bin/env python3
"""Quick Kimi K2.5 generation test on Krasis standalone.

Tests correctness with configurable precision.
Uses PP=2 (GPU0+GPU1) with BF16 KV cache.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("kimi-test")

import torch

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

TEST_CASES = [
    {
        "name": "simple_math",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 128,
        "expect_contains": ["4"],
    },
    {
        "name": "counting",
        "messages": [{"role": "user", "content": "Count from 1 to 5, separated by commas."}],
        "max_tokens": 128,
        "expect_contains": ["1", "2", "3", "4", "5"],
    },
    {
        "name": "factual",
        "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
        "max_tokens": 128,
        "expect_contains": ["Paris"],
    },
]


def main():
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel

    logger.info("Loading Kimi K2.5 PP=2...")
    t0 = time.perf_counter()

    # Default QuantConfig: INT8 attention, INT8 shared expert, INT8 dense MLP, INT8 lm_head
    # This is the production config — halves GPU VRAM vs BF16
    qcfg = QuantConfig()  # all defaults: int8 everywhere
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=2,
        gpu_prefill=True,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.float8_e4m3fn,
    )
    model.load()
    load_time = time.perf_counter() - t0
    logger.info("Kimi K2.5 loaded in %.1fs", load_time)

    passed = 0
    total = 0

    for tc in TEST_CASES:
        name = tc["name"]
        logger.info("\n--- %s ---", name)

        prompt = model.tokenizer.apply_chat_template(tc["messages"])
        logger.info("Prompt: %d tokens", len(prompt))

        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=tc["max_tokens"], temperature=0.0, top_k=1)
        elapsed = time.perf_counter() - t0

        text = model.tokenizer.decode(out)
        logger.info("Output (%d tokens, %.2fs): %s", len(out), elapsed, repr(text))

        ok = True
        for expected in tc["expect_contains"]:
            if expected.lower() not in text.lower():
                logger.warning("  MISSING expected: %s", expected)
                ok = False

        total += 1
        if ok:
            logger.info("  PASS")
            passed += 1
        else:
            logger.warning("  FAIL")

    logger.info("\n=== Kimi K2.5 results: %d/%d passed ===", passed, total)


if __name__ == "__main__":
    main()

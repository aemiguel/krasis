#!/usr/bin/env python3
"""GPU prefill speed test for Krasis standalone.

Tests with varying prompt sizes to measure GPU prefill throughput.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("prefill-test")

import torch

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

# Prompt sizes to test (tokens)
PROMPT_SIZES = [500, 1000, 2000, 5000]
MAX_NEW_TOKENS = 5  # small output, we're measuring prefill


def main():
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel

    logger.info("Loading Kimi K2.5 PP=2...")
    t0 = time.perf_counter()

    qcfg = QuantConfig()
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=2,
        gpu_prefill=True,
        quant_cfg=qcfg,
        kv_dtype=torch.float8_e4m3fn,
    )
    model.load()
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", load_time)

    # Build a large prompt by repeating a paragraph
    base_text = (
        "The history of computing is rich and varied. From the earliest mechanical "
        "calculators to modern quantum computers, humanity has always sought to "
        "automate calculation and reasoning. Charles Babbage designed the Analytical "
        "Engine in the 1830s, which contained many features of modern computers. "
        "Ada Lovelace wrote what is considered the first computer program for this "
        "machine. In the 20th century, Alan Turing formalized the concept of "
        "computation with his Turing machine model. The first electronic computers "
        "like ENIAC and Colossus were built during World War II. "
    )

    # Tokenize the base text to know how many tokens per repetition
    base_tokens = model.tokenizer.encode(base_text)
    logger.info("Base text: %d tokens per repetition", len(base_tokens))

    chat_prefix = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}]
    )

    for target_size in PROMPT_SIZES:
        # Build prompt of approximately target_size tokens
        reps = max(1, (target_size - len(chat_prefix)) // len(base_tokens))
        full_text = base_text * reps + "\n\nSummarize the above in one sentence."
        prompt = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_text}]
        )
        actual_size = len(prompt)

        logger.info("\n--- Prompt: %d tokens (target %d) ---", actual_size, target_size)

        # Warmup: first run may have cold caches
        if target_size == PROMPT_SIZES[0]:
            logger.info("  Warmup run...")
            _ = model.generate(prompt, max_new_tokens=1, temperature=0.0, top_k=1)
            # KV cache is per-sequence, no global clear needed

        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, top_k=1)
        elapsed = time.perf_counter() - t0

        text = model.tokenizer.decode(out)
        prefill_tok_s = actual_size / elapsed
        decode_tok_s = len(out) / elapsed if elapsed > 0 else 0

        logger.info("  Total: %.2fs | Prefill: %.1f tok/s | Output: %d tokens",
                     elapsed, prefill_tok_s, len(out))
        logger.info("  Response: %s", repr(text[:200]))

        # KV cache is per-sequence, no global clear needed

    logger.info("\n=== Prefill speed test complete ===")


if __name__ == "__main__":
    main()

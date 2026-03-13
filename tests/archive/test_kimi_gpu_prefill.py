#!/usr/bin/env python3
"""Test GPU prefill on Kimi K2.5 with a long prompt.

This triggers the GPU Marlin MoE path (M >= threshold).
First run will be slow (precomputing Marlin weights for 384 experts × 60 layers).
Subsequent runs use disk cache.
"""

import logging, os, sys, time

# Enable synchronous CUDA error reporting for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KRASIS_DEBUG_SYNC"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("gpu-prefill-test")

import torch

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

# A long prompt (~500 tokens) to trigger GPU prefill (threshold=300)
LONG_PROMPT = """You are an expert mathematician. Please solve the following problem step by step.

Problem: A farmer has a rectangular field that is 120 meters long and 80 meters wide. He wants to divide it into smaller rectangular plots, each 10 meters by 8 meters. He also wants to build a fence around each plot. The fence costs $5 per meter.

Additionally, the farmer wants to plant different crops in each plot. He has the following crops available: wheat, corn, soybeans, rice, barley, oats, rye, and potatoes. He wants to ensure that no two adjacent plots (sharing an edge) have the same crop.

Furthermore, each crop has a different yield per plot:
- Wheat: 200 kg
- Corn: 350 kg
- Soybeans: 150 kg
- Rice: 280 kg
- Barley: 180 kg
- Oats: 160 kg
- Rye: 140 kg
- Potatoes: 400 kg

And each crop has a different market price per kg:
- Wheat: $0.30
- Corn: $0.25
- Soybeans: $0.45
- Rice: $0.35
- Barley: $0.28
- Oats: $0.22
- Rye: $0.20
- Potatoes: $0.15

Calculate: 1) The total number of plots, 2) The total fencing cost, 3) The optimal crop assignment that maximizes total revenue while respecting the adjacency constraint. Show your work."""


def main():
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel

    logger.info("Loading Kimi K2.5 PP=2 with GPU prefill...")
    t0 = time.perf_counter()

    qcfg = QuantConfig()  # INT8 defaults
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=2,
        gpu_prefill=True,
        gpu_prefill_threshold=1,  # Force GPU prefill even for short prompts
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.float8_e4m3fn,
    )
    model.load()
    load_time = time.perf_counter() - t0
    logger.info("Kimi K2.5 loaded in %.1fs", load_time)

    # Tokenize the long prompt
    messages = [{"role": "user", "content": LONG_PROMPT}]
    prompt = model.tokenizer.apply_chat_template(messages)
    logger.info("Prompt: %d tokens (threshold=1, should trigger GPU prefill)", len(prompt))

    # Prefill + short decode
    t0 = time.perf_counter()
    out = model.generate(prompt, max_new_tokens=32, temperature=0.0, top_k=1)
    elapsed = time.perf_counter() - t0

    text = model.tokenizer.decode(out)
    prefill_tokens = len(prompt)
    decode_tokens = len(out)

    logger.info("Total: %.2fs for %d prefill + %d decode tokens", elapsed, prefill_tokens, decode_tokens)
    logger.info("Output: %s", repr(text[:200]))

    # Quick correctness check
    if len(out) > 0 and not any(text.lower().count(w) for w in ["error", "nan", "inf"]):
        logger.info("PASS — coherent output produced")
    else:
        logger.warning("FAIL — output may be garbled")

    logger.info("\n=== GPU prefill test complete ===")


if __name__ == "__main__":
    main()

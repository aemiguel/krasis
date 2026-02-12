#!/usr/bin/env python3
"""Run 8: Test Kimi K2.5 with BF16 KV cache (eliminate FP8 as variable).

This test uses BF16 for EVERYTHING (attention, KV cache, shared expert, dense, lm_head)
to establish a baseline. If this still loops, the bug is in MoE (INT4) or attention algorithm.
"""

import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run8")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

def main():
    import torch
    from krasis.config import QuantConfig

    logger.info("=== Run 8: Kimi K2.5 — ALL BF16 + BF16 KV cache ===")

    from krasis.model import KrasisModel

    # ALL BF16 to establish baseline
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
        kv_dtype=torch.bfloat16,  # BF16 KV cache (not FP8)
    )
    logger.info("Loading model...")
    model.load()
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", load_time)

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logger.info("GPU%d VRAM: %.2f GB", i, alloc)

    # Test with per-token logging
    logger.info("--- Test 1: Short math prompt (BF16 everything, per-token log) ---")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    prompt_tokens = model.tokenizer.apply_chat_template(messages)
    logger.info("Prompt tokens (%d): %s", len(prompt_tokens), prompt_tokens)
    logger.info("Prompt text: %s", repr(model.tokenizer.decode(prompt_tokens)))

    # Manual generate with per-token logging
    device = torch.device(model.ranks[0].device)
    from krasis.kv_cache import SequenceKVState
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    generated = []
    try:
        # Prefill
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)
        logits = model.forward(prompt_tensor, positions, seq_states)

        # First token
        next_logits = logits[-1:, :]
        topk_vals, topk_ids = next_logits[0].topk(10)
        tok_strs = [repr(model.tokenizer.decode([tid.item()])) for tid in topk_ids]
        logger.info("Prefill top10: %s", list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

        next_token = topk_ids[0].item()  # greedy
        generated.append(next_token)
        logger.info("Token 0: id=%d text=%s", next_token, repr(model.tokenizer.decode([next_token])))

        # Decode loop with per-token logging
        for step in range(63):  # max 64 total tokens
            pos = len(prompt_tokens) + step
            token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
            pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

            logits = model.forward(token_tensor, pos_tensor, seq_states)

            topk_vals, topk_ids = logits[0].topk(5)
            tok_strs = [repr(model.tokenizer.decode([tid.item()])) for tid in topk_ids]

            next_token = topk_ids[0].item()  # greedy
            generated.append(next_token)

            text = model.tokenizer.decode([next_token])
            logger.info("Token %d (pos=%d): id=%d text=%s | top5=%s",
                       step+1, pos, next_token, repr(text),
                       list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

            if next_token == model.cfg.eos_token_id:
                logger.info("Hit EOS at step %d", step+1)
                break

        full_text = model.tokenizer.decode(generated)
        logger.info("Full output: %s", repr(full_text))

    finally:
        for s in seq_states:
            s.free()

    logger.info("=== Run 8 complete ===")

if __name__ == "__main__":
    main()

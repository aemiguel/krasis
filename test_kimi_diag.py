#!/usr/bin/env python3
"""Kimi K2.5 diagnostic: detailed per-layer logging for first few decode steps."""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("diag")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

def main():
    import torch
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel
    from krasis.kv_cache import SequenceKVState

    logger.info("Loading Kimi K2.5 PP=2, BF16 everything, BF16 KV...")
    t0 = time.perf_counter()
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=2,
        gpu_prefill=False,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.bfloat16,
    )
    model.load()
    logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

    # Short prompt
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    prompt_tokens = model.tokenizer.apply_chat_template(messages)
    logger.info("Prompt tokens (%d): %s", len(prompt_tokens), prompt_tokens)
    logger.info("Prompt text: %s", repr(model.tokenizer.decode(prompt_tokens)))

    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    generated = []
    try:
        # Prefill
        logger.info("=== PREFILL ===")
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)
        logits = model.forward(prompt_tensor, positions, seq_states)

        topk_vals, topk_ids = logits[-1].topk(10)
        tok_strs = [repr(model.tokenizer.decode([tid.item()])) for tid in topk_ids]
        logger.info("Prefill top10: %s", list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

        next_token = topk_ids[0].item()  # greedy
        generated.append(next_token)
        logger.info("Token 0: id=%d text=%s", next_token, repr(model.tokenizer.decode([next_token])))

        # Decode 20 tokens
        logger.info("=== DECODE ===")
        for step in range(19):
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

    logger.info("=== Test complete ===")

if __name__ == "__main__":
    main()

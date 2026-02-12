#!/usr/bin/env python3
"""Compare prefill vs token-at-a-time decode logits.

For a prompt of N tokens, compare:
1. Prefill: forward(all N tokens) → logits[-1]  (what we get from bulk prefill)
2. Sequential: forward(token[0]), forward(token[1]), ..., forward(token[N-1]) → logits[-1]

If they match, the KV cache + decode attention is correct.
If they diverge, the bug is in KV cache writing/reading or decode attention.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("pfvd")

import torch

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"


def main():
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel
    from krasis.kv_cache import SequenceKVState

    logger.info("Loading V2-Lite PP=1, BF16 everything...")
    t0 = time.perf_counter()
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        gpu_prefill=False,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.bfloat16,
    )
    model.load()
    logger.info("Loaded in %.1fs", time.perf_counter() - t0)

    device = torch.device(model.ranks[0].device)

    # Short prompt
    messages = [{"role": "user", "content": "Hello"}]
    prompt = model.tokenizer.apply_chat_template(messages)
    N = len(prompt)
    logger.info("Prompt: %d tokens: %s", N, prompt)

    # ═══════════════════════════════════════════════
    # Test 1: Prefill all N tokens at once
    # ═══════════════════════════════════════════════
    logger.info("\n=== TEST 1: Bulk prefill (all %d tokens) ===", N)
    seq_states_bulk = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]
    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
    positions = torch.arange(N, dtype=torch.int32, device=device)

    bulk_logits = model.forward(prompt_tensor, positions, seq_states_bulk)
    bulk_last = bulk_logits[-1].float().cpu()  # [vocab_size]

    topk_vals, topk_ids = bulk_last.topk(10)
    tok_strs = [repr(model.tokenizer.decode([tid.item()])) for tid in topk_ids]
    logger.info("Bulk prefill last-token top10: %s",
                list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

    # Free the bulk states
    for s in seq_states_bulk:
        s.free()

    # ═══════════════════════════════════════════════
    # Test 2: Sequential token-at-a-time processing
    # ═══════════════════════════════════════════════
    logger.info("\n=== TEST 2: Sequential (token by token) ===")
    seq_states_seq = [SequenceKVState(c, seq_id=1) for c in model.kv_caches]

    per_token_logits = []
    for i in range(N):
        tok_tensor = torch.tensor([prompt[i]], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([i], dtype=torch.int32, device=device)

        logits_i = model.forward(tok_tensor, pos_tensor, seq_states_seq)
        per_token_logits.append(logits_i[0].float().cpu())

        if i < 3 or i == N - 1:
            topk_v, topk_i = per_token_logits[-1].topk(5)
            tok_strs_i = [repr(model.tokenizer.decode([tid.item()])) for tid in topk_i]
            logger.info("Token %d top5: %s", i,
                        list(zip(tok_strs_i, [f"{v:.1f}" for v in topk_v.tolist()])))

    seq_last = per_token_logits[-1]  # [vocab_size]

    topk_vals2, topk_ids2 = seq_last.topk(10)
    tok_strs2 = [repr(model.tokenizer.decode([tid.item()])) for tid in topk_ids2]
    logger.info("Sequential last-token top10: %s",
                list(zip(tok_strs2, [f"{v:.1f}" for v in topk_vals2.tolist()])))

    for s in seq_states_seq:
        s.free()

    # ═══════════════════════════════════════════════
    # Compare
    # ═══════════════════════════════════════════════
    logger.info("\n=== COMPARISON ===")
    cos_sim = torch.nn.functional.cosine_similarity(
        bulk_last.flatten(), seq_last.flatten(), dim=0
    ).item()
    max_diff = (bulk_last - seq_last).abs().max().item()
    mean_diff = (bulk_last - seq_last).abs().mean().item()

    logger.info("Cosine similarity:  %.6f", cos_sim)
    logger.info("Max abs diff:       %.4f", max_diff)
    logger.info("Mean abs diff:      %.4f", mean_diff)

    # Compare top-k predictions
    bulk_top10 = set(topk_ids.tolist())
    seq_top10 = set(topk_ids2.tolist())
    overlap = len(bulk_top10 & seq_top10)
    greedy_match = topk_ids[0].item() == topk_ids2[0].item()

    logger.info("Top-10 overlap:     %d/10", overlap)
    logger.info("Greedy match:       %s (bulk=%s, seq=%s)",
                greedy_match,
                repr(model.tokenizer.decode([topk_ids[0].item()])),
                repr(model.tokenizer.decode([topk_ids2[0].item()])))

    if cos_sim > 0.999:
        logger.info("PASS: Prefill and sequential match perfectly")
    elif cos_sim > 0.99:
        logger.info("CLOSE: Minor differences (acceptable for INT4 experts)")
    elif cos_sim > 0.95:
        logger.info("WARN: Noticeable divergence")
    else:
        logger.info("FAIL: Prefill vs sequential diverge significantly!")

    # Also compare intermediate token logits from bulk vs sequential
    logger.info("\n=== Per-token comparison (bulk vs sequential) ===")
    for i in [0, N//2, N-1]:
        b = bulk_logits[i].float().cpu()
        s = per_token_logits[i]
        cos_i = torch.nn.functional.cosine_similarity(
            b.flatten(), s.flatten(), dim=0
        ).item()
        max_d = (b - s).abs().max().item()
        b_top = b.topk(3)
        s_top = s.topk(3)
        b_strs = [repr(model.tokenizer.decode([t.item()])) for t in b_top.indices]
        s_strs = [repr(model.tokenizer.decode([t.item()])) for t in s_top.indices]
        logger.info("Token %d: cos=%.6f max_diff=%.4f bulk_top3=%s seq_top3=%s",
                     i, cos_i, max_d, b_strs, s_strs)

    logger.info("\n=== Test complete ===")


if __name__ == "__main__":
    main()

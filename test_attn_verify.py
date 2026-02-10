#!/usr/bin/env python3
"""Verify MLA attention: compare FlashInfer MLA kernel vs manual torch implementation.

Loads one layer of Kimi K2.5, runs prefill + 1 decode step through both:
1. Our FlashInfer-based MLA attention (from attention.py)
2. A manual torch.matmul implementation (ground truth)

If they match, the attention is correct. If they differ, we found the bug.
"""

import logging, sys, time, math
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("attn-verify")

import torch
import flashinfer

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"


def manual_mla_attention(
    q_nope_absorbed,  # [M, H, kv_lora_rank]
    q_pe,             # [M, H, qk_rope_dim]
    ckv_all,          # [T, kv_lora_rank] — all tokens' compressed KV
    kpe_all,          # [T, qk_rope_dim] — all tokens' k_pe
    sm_scale,
    causal=True,
):
    """Manual MLA attention using torch.matmul (no FlashInfer).

    Args:
        q_nope_absorbed: [M, H, D_ckv] query with w_kc absorbed
        q_pe: [M, H, D_kpe] query position embeddings
        ckv_all: [T, D_ckv] all tokens' compressed KV (T = total seq len)
        kpe_all: [T, D_kpe] all tokens' key position embeddings
        sm_scale: softmax scale factor
        causal: apply causal mask

    Returns:
        attn_out: [M, H, D_ckv] attention output in compressed KV space
    """
    M, H, D_ckv = q_nope_absorbed.shape
    T = ckv_all.shape[0]
    D_kpe = q_pe.shape[-1]

    # Compute attention scores
    # score_nope[m, h, t] = q_nope_absorbed[m, h] @ ckv[t]
    score_nope = torch.einsum("mhd,td->mht", q_nope_absorbed.float(), ckv_all.float())

    # score_pe[m, h, t] = q_pe[m, h] @ kpe[t]
    score_pe = torch.einsum("mhd,td->mht", q_pe.float(), kpe_all.float())

    scores = (score_nope + score_pe) * sm_scale  # [M, H, T]

    # Causal mask: token at position p can only attend to positions <= p
    if causal and M > 1:
        # During prefill: M queries, T keys. Query i corresponds to position (T-M+i)
        # For our case: positions are [0..M-1] and T=M during prefill
        # mask[i, j] = True if j <= i (for prefill where T=M)
        mask = torch.triu(torch.ones(M, T, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [M, H, T]

    # Weighted sum over ckv
    # attn_out[m, h, d] = sum_t attn_weights[m, h, t] * ckv[t, d]
    attn_out = torch.einsum("mht,td->mhd", attn_weights, ckv_all.float())

    return attn_out.to(torch.bfloat16)


def main():
    from krasis.config import ModelConfig, QuantConfig
    from krasis.weight_loader import WeightLoader
    from krasis.kv_cache import PagedKVCache, SequenceKVState
    from krasis.attention import MLAAttention

    logger.info("Loading Kimi K2.5 config + 1 layer...")
    cfg = ModelConfig.from_model_path(MODEL_PATH)
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    loader = WeightLoader(cfg, qcfg)

    device = torch.device("cuda:0")

    # Load embedding + layer 1 (first MoE layer, has q_lora)
    embedding = loader.load_embedding(device)
    weights_l1 = loader.load_layer(1, device)

    # Create attention module
    attn = MLAAttention(cfg, 1, weights_l1["attention"], device)

    # Create KV cache (1 layer only)
    cache = PagedKVCache(cfg, num_layers=1, device=device, kv_dtype=torch.bfloat16, max_pages=64)
    seq_state = SequenceKVState(cache, seq_id=0)

    # Create a short prompt
    from krasis.tokenizer import Tokenizer
    tokenizer = Tokenizer(MODEL_PATH)
    messages = [{"role": "user", "content": "Hello"}]
    prompt_tokens = tokenizer.apply_chat_template(messages)
    M = len(prompt_tokens)
    logger.info("Prompt: %d tokens", M)

    # Embed
    token_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    hidden = embedding[token_ids]  # [M, hidden_size]

    # Apply input layernorm (to get the actual attention input)
    input_norm_weight = weights_l1["norms"]["input_layernorm"]
    hidden_normed = flashinfer.norm.rmsnorm(hidden, input_norm_weight, cfg.rms_norm_eps)

    # ── Our FlashInfer-based attention ──
    logger.info("=== FlashInfer MLA attention ===")
    seq_state.ensure_capacity(M)
    positions = torch.arange(M, dtype=torch.int32, device=device)

    flashinfer_out = attn.forward(
        hidden_normed, positions, cache, seq_state, layer_offset=0, num_new_tokens=M
    )
    logger.info("FlashInfer output shape: %s, rms=%.6f", flashinfer_out.shape,
                flashinfer_out.float().pow(2).mean().sqrt().item())

    # ── Manual attention for comparison ──
    logger.info("=== Manual MLA attention ===")

    # Reproduce the attention steps manually
    H = cfg.num_attention_heads
    qk_nope_dim = cfg.qk_nope_head_dim
    qk_rope_dim = cfg.qk_rope_head_dim
    kv_lora_rank = cfg.kv_lora_rank
    v_head_dim = cfg.v_head_dim
    head_dim = qk_nope_dim + qk_rope_dim

    # Q projection (q_lora path)
    from krasis.layer import _linear
    q_compressed = _linear(hidden_normed, attn.q_a_proj)
    q_compressed = flashinfer.norm.rmsnorm(q_compressed, attn.q_a_norm_weight, cfg.rms_norm_eps)
    q_full = _linear(q_compressed, attn.q_b_proj)
    q_full = q_full.reshape(M, H, head_dim)
    q_nope = q_full[:, :, :qk_nope_dim]
    q_pe = q_full[:, :, qk_nope_dim:]

    # KV projection
    kv_out = _linear(hidden_normed, attn.kv_a_proj)
    kv_compressed = kv_out[:, :kv_lora_rank]
    k_pe = kv_out[:, kv_lora_rank:]
    kv_compressed = flashinfer.norm.rmsnorm(kv_compressed, attn.kv_a_norm_weight, cfg.rms_norm_eps)

    # RoPE
    k_pe_heads = k_pe.unsqueeze(1)
    q_pe_rope, k_pe_rope = attn._apply_rope(q_pe, k_pe_heads, positions)
    k_pe_flat = k_pe_rope.squeeze(1)  # [M, qk_rope_dim]

    # Absorb w_kc into query
    q_nope_absorbed = torch.einsum("mhi,hid->mhd", q_nope.float(), attn.w_kc.float()).to(torch.bfloat16)

    # Manual attention (no paged cache, direct computation)
    manual_attn_out = manual_mla_attention(
        q_nope_absorbed, q_pe_rope,
        kv_compressed,  # [M, kv_lora_rank]
        k_pe_flat,      # [M, qk_rope_dim]
        attn.sm_scale,
        causal=True,
    )

    # Post-attention: attn_out @ w_vc^T
    manual_projected = torch.einsum(
        "mhd,hod->mho", manual_attn_out.float(), attn.w_vc.float()
    ).to(torch.bfloat16)
    manual_flat = manual_projected.reshape(M, H * v_head_dim)
    manual_output = _linear(manual_flat, attn.o_proj)

    logger.info("Manual output shape: %s, rms=%.6f", manual_output.shape,
                manual_output.float().pow(2).mean().sqrt().item())

    # ── Compare ──
    fi = flashinfer_out.float()
    mn = manual_output.float()

    cos_sim = torch.nn.functional.cosine_similarity(fi.flatten(), mn.flatten(), dim=0).item()
    max_diff = (fi - mn).abs().max().item()
    mean_diff = (fi - mn).abs().mean().item()

    logger.info("=== COMPARISON ===")
    logger.info("  Cosine similarity: %.6f", cos_sim)
    logger.info("  Max abs diff: %.6f", max_diff)
    logger.info("  Mean abs diff: %.6f", mean_diff)

    # Per-token comparison (last token is most important for generation)
    for i in [0, M-1]:
        cos_t = torch.nn.functional.cosine_similarity(
            fi[i].flatten(), mn[i].flatten(), dim=0
        ).item()
        logger.info("  Token %d cosine: %.6f", i, cos_t)

    if cos_sim > 0.999:
        logger.info("PASS: FlashInfer MLA matches manual attention")
    elif cos_sim > 0.99:
        logger.info("CLOSE: Small differences (precision?)")
    else:
        logger.info("MISMATCH: FlashInfer MLA DIFFERS from manual attention!")

        # Debug: check intermediate values
        logger.info("\n=== DEBUG: Intermediate values ===")
        logger.info("q_nope_absorbed rms: %.6f", q_nope_absorbed.float().pow(2).mean().sqrt().item())
        logger.info("q_pe_rope rms: %.6f", q_pe_rope.float().pow(2).mean().sqrt().item())
        logger.info("kv_compressed rms: %.6f", kv_compressed.float().pow(2).mean().sqrt().item())
        logger.info("k_pe_flat rms: %.6f", k_pe_flat.float().pow(2).mean().sqrt().item())
        logger.info("sm_scale: %.6f", attn.sm_scale)

        # Check manual vs flashinfer at attn_out level (before o_proj)
        fi_pre_o = flashinfer_out  # This is after o_proj already...
        # Can't easily get pre-o_proj from flashinfer path

    # ── Test decode step too ──
    logger.info("\n=== DECODE STEP (1 token) ===")
    seq_state.advance(M)  # Mark prefill tokens as consumed

    # Pick the greedy next token (using manual output for embedding)
    # Just use token id 1 as a test input
    test_token = torch.tensor([1], dtype=torch.long, device=device)
    test_hidden = embedding[test_token]
    test_hidden_normed = flashinfer.norm.rmsnorm(test_hidden, input_norm_weight, cfg.rms_norm_eps)
    test_pos = torch.tensor([M], dtype=torch.int32, device=device)  # position = M

    # FlashInfer decode
    seq_state.ensure_capacity(1)
    fi_decode = attn.forward(
        test_hidden_normed, test_pos, cache, seq_state, layer_offset=0, num_new_tokens=1
    )

    # Manual decode: attend to all M+1 tokens
    # Need to recompute Q for the decode token
    q_compressed_dec = _linear(test_hidden_normed, attn.q_a_proj)
    q_compressed_dec = flashinfer.norm.rmsnorm(q_compressed_dec, attn.q_a_norm_weight, cfg.rms_norm_eps)
    q_full_dec = _linear(q_compressed_dec, attn.q_b_proj)
    q_full_dec = q_full_dec.reshape(1, H, head_dim)
    q_nope_dec = q_full_dec[:, :, :qk_nope_dim]
    q_pe_dec = q_full_dec[:, :, qk_nope_dim:]

    # KV for decode token
    kv_out_dec = _linear(test_hidden_normed, attn.kv_a_proj)
    ckv_dec = kv_out_dec[:, :kv_lora_rank]
    kpe_dec = kv_out_dec[:, kv_lora_rank:]
    ckv_dec = flashinfer.norm.rmsnorm(ckv_dec, attn.kv_a_norm_weight, cfg.rms_norm_eps)

    # RoPE for decode
    kpe_dec_heads = kpe_dec.unsqueeze(1)
    q_pe_dec_rope, kpe_dec_rope = attn._apply_rope(q_pe_dec, kpe_dec_heads, test_pos)
    kpe_dec_flat = kpe_dec_rope.squeeze(1)

    # Absorb w_kc
    q_nope_dec_absorbed = torch.einsum("mhi,hid->mhd", q_nope_dec.float(), attn.w_kc.float()).to(torch.bfloat16)

    # All KV: prefill tokens + decode token
    all_ckv = torch.cat([kv_compressed, ckv_dec], dim=0)  # [M+1, kv_lora_rank]
    all_kpe = torch.cat([k_pe_flat, kpe_dec_flat], dim=0)  # [M+1, qk_rope_dim]

    manual_attn_dec = manual_mla_attention(
        q_nope_dec_absorbed, q_pe_dec_rope,
        all_ckv, all_kpe,
        attn.sm_scale,
        causal=False,  # single query token, no causal needed
    )
    manual_proj_dec = torch.einsum(
        "mhd,hod->mho", manual_attn_dec.float(), attn.w_vc.float()
    ).to(torch.bfloat16)
    manual_flat_dec = manual_proj_dec.reshape(1, H * v_head_dim)
    manual_dec = _linear(manual_flat_dec, attn.o_proj)

    fi_d = fi_decode.float()
    mn_d = manual_dec.float()
    cos_dec = torch.nn.functional.cosine_similarity(fi_d.flatten(), mn_d.flatten(), dim=0).item()
    max_diff_dec = (fi_d - mn_d).abs().max().item()

    logger.info("  FlashInfer decode rms: %.6f", fi_d.pow(2).mean().sqrt().item())
    logger.info("  Manual decode rms: %.6f", mn_d.pow(2).mean().sqrt().item())
    logger.info("  Cosine similarity: %.6f", cos_dec)
    logger.info("  Max abs diff: %.6f", max_diff_dec)

    if cos_dec > 0.999:
        logger.info("  PASS: Decode attention matches")
    elif cos_dec > 0.99:
        logger.info("  CLOSE: Small differences")
    else:
        logger.info("  MISMATCH: Decode attention differs!")

    # Clean up
    seq_state.free()
    loader.close()
    logger.info("\n=== Test complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test V2-Lite with MoE layers replaced by identity (pass-through).

If the dense layer 0 + attention work but MoE is wrong,
this will produce better logits than the full model.
Also: compare single-layer logits against manual torch computation.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("dense-test")

import torch
import flashinfer

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"


def main():
    from krasis.config import ModelConfig, QuantConfig
    from krasis.weight_loader import WeightLoader
    from krasis.kv_cache import PagedKVCache, SequenceKVState
    from krasis.attention import MLAAttention
    from krasis.layer import _linear

    logger.info("Loading V2-Lite config...")
    cfg = ModelConfig.from_model_path(MODEL_PATH)
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    loader = WeightLoader(cfg, qcfg)

    device = torch.device("cuda:0")

    # Load embedding + layer 0 (dense MLP) + layer 1 (first MoE)
    logger.info("Loading embedding...")
    embedding = loader.load_embedding(device)

    logger.info("Loading layer 0 (dense)...")
    weights_l0 = loader.load_layer(0, device)
    logger.info("  is_moe=%s", weights_l0["is_moe"])

    logger.info("Loading layer 1 (MoE)...")
    weights_l1 = loader.load_layer(1, device)
    logger.info("  is_moe=%s", weights_l1["is_moe"])

    # Create attention for layer 0
    attn_l0 = MLAAttention(cfg, 0, weights_l0["attention"], device)
    attn_l1 = MLAAttention(cfg, 1, weights_l1["attention"], device)

    # Layer norms
    input_norm_l0 = weights_l0["norms"]["input_layernorm"]
    post_attn_norm_l0 = weights_l0["norms"]["post_attention_layernorm"]
    input_norm_l1 = weights_l1["norms"]["input_layernorm"]
    post_attn_norm_l1 = weights_l1["norms"]["post_attention_layernorm"]

    # Dense MLP weights
    dense_mlp = weights_l0.get("dense_mlp")

    # Shared expert weights for layer 1
    shared_expert = weights_l1.get("shared_expert")
    gate_weight = weights_l1["gate"]["weight"]

    # Load final norm and LM head
    final_norm = loader.load_final_norm(device)
    lm_head = loader.load_lm_head(device)

    # KV cache
    cache = PagedKVCache(cfg, num_layers=2, device=device, kv_dtype=torch.bfloat16, max_pages=64)
    seq = SequenceKVState(cache, seq_id=0)

    # Tokenize
    from krasis.tokenizer import Tokenizer
    tokenizer = Tokenizer(MODEL_PATH)
    messages = [{"role": "user", "content": "Hello"}]
    prompt = tokenizer.apply_chat_template(messages)
    M = len(prompt)
    logger.info("Prompt: %d tokens: %s", M, prompt)

    token_ids = torch.tensor(prompt, dtype=torch.long, device=device)
    hidden = embedding[token_ids]  # [M, hidden_size]
    positions = torch.arange(M, dtype=torch.int32, device=device)
    logger.info("Embedding rms: %.4f", hidden.float().pow(2).mean().sqrt().item())

    # ── Layer 0: pre-norm → attention → post-norm → dense MLP ──
    logger.info("\n=== Layer 0 (dense) ===")
    seq.ensure_capacity(M)

    # Pre-attention norm (first layer: no residual add)
    residual = hidden.clone()
    hidden_normed = flashinfer.norm.rmsnorm(hidden, input_norm_l0, cfg.rms_norm_eps)
    logger.info("Pre-attn norm rms: %.4f", hidden_normed.float().pow(2).mean().sqrt().item())

    # Attention
    attn_out = attn_l0.forward(hidden_normed, positions, cache, seq, layer_offset=0, num_new_tokens=M)
    logger.info("Attn out rms: %.4f", attn_out.float().pow(2).mean().sqrt().item())

    # Post-attention: residual += attn_out, then norm
    flashinfer.norm.fused_add_rmsnorm(attn_out, residual, post_attn_norm_l0, cfg.rms_norm_eps)
    hidden = attn_out  # Now contains post-attn normed value
    logger.info("Post-attn norm rms: %.4f", hidden.float().pow(2).mean().sqrt().item())

    # Dense MLP: gate + up → SiLU → down
    gate = _linear(hidden, dense_mlp["gate_proj"])
    up = _linear(hidden, dense_mlp["up_proj"])
    activated = flashinfer.activation.silu_and_mul(torch.cat([gate, up], dim=-1))
    mlp_out = _linear(activated, dense_mlp["down_proj"])
    logger.info("Dense MLP out rms: %.4f", mlp_out.float().pow(2).mean().sqrt().item())

    # ── Layer 1: pre-norm → attention → post-norm → MoE ──
    # First, prepare for layer 1
    hidden = mlp_out  # output of layer 0's MLP
    # Need to do pre-norm with residual from previous layer
    flashinfer.norm.fused_add_rmsnorm(hidden, residual, input_norm_l1, cfg.rms_norm_eps)
    logger.info("\n=== Layer 1 (MoE) ===")
    logger.info("Pre-attn norm L1 rms: %.4f", hidden.float().pow(2).mean().sqrt().item())

    # Attention L1
    attn_out_l1 = attn_l1.forward(hidden, positions, cache, seq, layer_offset=1, num_new_tokens=M)
    logger.info("Attn L1 out rms: %.4f", attn_out_l1.float().pow(2).mean().sqrt().item())

    # Post-attention norm L1
    flashinfer.norm.fused_add_rmsnorm(attn_out_l1, residual, post_attn_norm_l1, cfg.rms_norm_eps)
    hidden_for_moe = attn_out_l1
    logger.info("Post-attn norm L1 rms: %.4f", hidden_for_moe.float().pow(2).mean().sqrt().item())

    # MoE routing (Python, no Krasis engine)
    router_logits = torch.matmul(hidden_for_moe.float(), gate_weight.float().t())
    scores = torch.softmax(router_logits, dim=-1)  # V2-Lite uses softmax
    topk = cfg.num_experts_per_tok
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    logger.info("Router: topk=%d, ids[last]=%s, weights[last]=%s",
                topk, topk_ids[-1].tolist(),
                [f"{w:.4f}" for w in topk_weights[-1].tolist()])

    # ── Test: shared expert only (skip routed experts) ──
    logger.info("\n=== Shared expert only (no routed) ===")
    if shared_expert:
        gate_se = _linear(hidden_for_moe, shared_expert["gate_proj"])
        up_se = _linear(hidden_for_moe, shared_expert["up_proj"])
        act_se = flashinfer.activation.silu_and_mul(torch.cat([gate_se, up_se], dim=-1))
        shared_out = _linear(act_se, shared_expert["down_proj"])
        logger.info("Shared expert out rms: %.4f", shared_out.float().pow(2).mean().sqrt().item())

        # Use shared expert as the MoE output (skip routed)
        moe_out_shared_only = shared_out
    else:
        logger.info("No shared expert for layer 1")
        moe_out_shared_only = torch.zeros_like(hidden_for_moe)

    # ── Now: 2-layer logits with shared expert only ──
    # Final norm: residual += moe_out, hidden = norm(residual)
    final_hidden = moe_out_shared_only.clone()
    flashinfer.norm.fused_add_rmsnorm(final_hidden, residual, final_norm, cfg.rms_norm_eps)

    # LM head
    logits = _linear(final_hidden, lm_head).float()
    logger.info("\n=== 2-layer logits (shared expert only) ===")
    last_logits = logits[-1]
    topk_vals, topk_idx = last_logits.topk(10)
    tok_strs = [repr(tokenizer.decode([tid.item()])) for tid in topk_idx]
    logger.info("Last token top10: %s", list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))
    logger.info("Logits std: %.2f", last_logits.std().item())

    # Advance KV cache
    seq.advance(M)

    # ── Decode step: next token ──
    logger.info("\n=== Decode step (greedy from 2-layer model) ===")
    next_id = topk_idx[0].item()
    next_text = tokenizer.decode([next_id])
    logger.info("Next token: id=%d text=%s", next_id, repr(next_text))

    # Clean up
    seq.free()
    loader.close()
    logger.info("\n=== Test complete ===")


if __name__ == "__main__":
    main()

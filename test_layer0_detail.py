#!/usr/bin/env python3
"""Detailed comparison of layer 0 sub-components: HF vs Krasis.

Layer 0 is a DENSE layer (no MoE), so this isolates:
1. Pre-attention RMSNorm
2. MLA Attention (q_proj, kv_proj, RoPE, attention, w_kc/w_vc, o_proj)
3. Post-attention residual + RMSNorm
4. Dense MLP (gate + up → SiLU → down)

The embedding is known to be EXACT between HF and Krasis.
"""

import logging, sys, time, gc
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("l0-detail")

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
TOKEN_IDS = [100000, 100000, 5726, 25, 37727, 185, 185, 77398, 25]


def compare(name, hf_tensor, kr_tensor, token_idx=-1):
    """Compare two tensors and report."""
    if hf_tensor.dim() >= 2 and kr_tensor.dim() >= 2:
        hf = hf_tensor[token_idx].float().flatten()
        kr = kr_tensor[token_idx].float().flatten()
    else:
        hf = hf_tensor.float().flatten()
        kr = kr_tensor.float().flatten()

    if hf.shape != kr.shape:
        logger.info("  [SHAPE MISMATCH] %s: hf=%s kr=%s", name, hf_tensor.shape, kr_tensor.shape)
        return 0.0

    cos = F.cosine_similarity(hf, kr, dim=0).item()
    max_diff = (hf - kr).abs().max().item()
    mean_diff = (hf - kr).abs().mean().item()
    exact = torch.equal(hf_tensor.cpu().to(torch.bfloat16), kr_tensor.cpu().to(torch.bfloat16))

    status = "EXACT" if exact else ("OK" if cos > 0.9999 else ("CLOSE" if cos > 0.999 else "DIVERGED"))
    logger.info("  [%s] %s: cos=%.6f max=%.4f mean=%.6f",
                status, name, cos, max_diff, mean_diff)
    return cos


def run_hf_layer0():
    """Run HF layer 0 manually, capturing all sub-component outputs."""
    from transformers import AutoModelForCausalLM

    logger.info("Loading HF model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()

    layer0 = model.model.layers[0]

    # Capture sub-component outputs via hooks
    captures = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                captures[name] = out[0].detach().clone()
            else:
                captures[name] = out.detach().clone()
            # Capture input (may be tuple, may be empty)
            if isinstance(inp, tuple) and len(inp) > 0:
                captures[name + "_input"] = inp[0].detach().clone()
        return hook

    hooks = []
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook("embed")))
    hooks.append(layer0.input_layernorm.register_forward_hook(make_hook("input_norm")))
    hooks.append(layer0.self_attn.register_forward_hook(make_hook("self_attn")))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(make_hook("post_attn_norm")))
    hooks.append(layer0.mlp.register_forward_hook(make_hook("mlp")))

    # Also hook the full layer
    hooks.append(layer0.register_forward_hook(make_hook("layer0_full")))

    # Forward
    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    # Squeeze batch dim
    result = {}
    for k, v in captures.items():
        if v.dim() == 3:
            result[k] = v.squeeze(0).cpu()
        else:
            result[k] = v.cpu()

    result["logits"] = outputs.logits.squeeze(0).float().cpu()

    del model
    gc.collect()
    return result


def run_krasis_layer0():
    """Run Krasis layer 0 manually, capturing all sub-component outputs."""
    import flashinfer
    from krasis.config import ModelConfig, QuantConfig
    from krasis.weight_loader import WeightLoader
    from krasis.kv_cache import PagedKVCache, SequenceKVState
    from krasis.attention import MLAAttention
    from krasis.layer import _linear

    logger.info("Loading Krasis layer 0 weights (BF16)...")
    cfg = ModelConfig.from_model_path(MODEL_PATH)
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    loader = WeightLoader(cfg, qcfg)

    device = torch.device("cuda:0")

    # Load just what we need
    embedding = loader.load_embedding(device)
    weights_l0 = loader.load_layer(0, device)

    result = {}

    # Token IDs → embedding
    token_ids = torch.tensor(TOKEN_IDS, dtype=torch.long, device=device)
    hidden = embedding[token_ids]  # [M, 2048]
    result["embed"] = hidden.cpu()

    M = len(TOKEN_IDS)
    positions = torch.arange(M, dtype=torch.int32, device=device)

    # ── Pre-attention RMSNorm ──
    # First layer: residual = hidden, hidden = rmsnorm(hidden)
    input_norm_weight = weights_l0["norms"]["input_layernorm"]
    residual = hidden.clone()
    hidden_normed = flashinfer.norm.rmsnorm(hidden, input_norm_weight, cfg.rms_norm_eps)
    result["input_norm"] = hidden_normed.cpu()

    # ── Attention ──
    # Set up KV cache for attention
    cache = PagedKVCache(cfg, num_layers=1, device=device, kv_dtype=torch.bfloat16, max_pages=64)
    seq = SequenceKVState(cache, seq_id=0)
    seq.ensure_capacity(M)

    attn = MLAAttention(cfg, 0, weights_l0["attention"], device)
    attn_out = attn.forward(hidden_normed, positions, cache, seq, layer_offset=0, num_new_tokens=M)
    result["self_attn"] = attn_out.cpu()

    # ── Post-attention: residual += attn_out, then norm ──
    flashinfer.norm.fused_add_rmsnorm(attn_out, residual, weights_l0["norms"]["post_attention_layernorm"], cfg.rms_norm_eps)
    # After: attn_out = rmsnorm(residual + old_attn_out), residual = residual + old_attn_out
    result["post_attn_norm"] = attn_out.cpu()
    result["post_attn_residual"] = residual.cpu()

    # ── Dense MLP ──
    dense_mlp = weights_l0.get("dense_mlp")
    gate = _linear(attn_out, dense_mlp["gate_proj"])
    up = _linear(attn_out, dense_mlp["up_proj"])
    activated = flashinfer.activation.silu_and_mul(torch.cat([gate, up], dim=-1))
    mlp_out = _linear(activated, dense_mlp["down_proj"])
    result["mlp"] = mlp_out.cpu()

    # ── Full layer output (residual + mlp_out = HF's layer output) ──
    result["layer0_full"] = (residual + mlp_out).cpu()

    seq.free()
    loader.close()
    return result


def main():
    logger.info("=" * 60)
    logger.info("Detailed Layer 0 comparison: HF vs Krasis")
    logger.info("Token IDs: %s (%d tokens)", TOKEN_IDS, len(TOKEN_IDS))
    logger.info("=" * 60)

    hf = run_hf_layer0()
    kr = run_krasis_layer0()

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON (last token)")
    logger.info("=" * 60)

    logger.info("\n--- Checkpoints ---")
    compare("Embedding", hf["embed"], kr["embed"])
    compare("Pre-attn norm (input_layernorm output)", hf["input_norm"], kr["input_norm"])
    compare("Attention output", hf["self_attn"], kr["self_attn"])
    compare("Post-attn norm output", hf["post_attn_norm"], kr["post_attn_norm"])
    compare("Dense MLP output", hf["mlp"], kr["mlp"])
    compare("Layer 0 full output", hf["layer0_full"], kr["layer0_full"])

    # Also compare at ALL token positions for the first divergence point
    logger.info("\n--- Per-token comparison at first divergence ---")
    for checkpoint in ["input_norm", "self_attn", "post_attn_norm", "mlp"]:
        if checkpoint in hf and checkpoint in kr:
            hf_t = hf[checkpoint]
            kr_t = kr[checkpoint]
            if hf_t.dim() == 2 and kr_t.dim() == 2 and hf_t.shape == kr_t.shape:
                cosines = []
                for tok in range(min(hf_t.shape[0], kr_t.shape[0])):
                    c = F.cosine_similarity(
                        hf_t[tok].float().flatten(),
                        kr_t[tok].float().flatten(), dim=0
                    ).item()
                    cosines.append(c)
                logger.info("  %s per-token cos: %s",
                            checkpoint,
                            [f"{c:.6f}" for c in cosines])

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

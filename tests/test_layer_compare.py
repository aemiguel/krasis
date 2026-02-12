#!/usr/bin/env python3
"""Layer-by-layer comparison: HF reference vs Krasis standalone.

Feeds the same token IDs to both, captures hidden states after each layer,
and reports where they diverge. The first layer that diverges is where the bug is.

HF (CPU, BF16, trust_remote_code) = ground truth (produces correct output).
Krasis (GPU, BF16 everything) = what we're debugging.
"""

import logging, sys, time, gc
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("layer-cmp")

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"

# Use the same tokens for both (HF's tokenization of "Hello" chat)
TOKEN_IDS = [100000, 100000, 5726, 25, 37727, 185, 185, 77398, 25]


def compare(name, hf_tensor, kr_tensor, token_idx=-1):
    """Compare two tensors at a specific token position and report."""
    if hf_tensor.dim() == 2 and kr_tensor.dim() == 2:
        hf = hf_tensor[token_idx].float()
        kr = kr_tensor[token_idx].float()
    elif hf_tensor.dim() == 1 and kr_tensor.dim() == 1:
        hf = hf_tensor.float()
        kr = kr_tensor.float()
    else:
        hf = hf_tensor.flatten().float()
        kr = kr_tensor.flatten().float()

    cos = F.cosine_similarity(hf, kr, dim=0).item()
    max_diff = (hf - kr).abs().max().item()
    mean_diff = (hf - kr).abs().mean().item()

    # Check if exactly identical
    exact = torch.equal(hf_tensor.cpu(), kr_tensor.cpu())

    status = "EXACT" if exact else ("OK" if cos > 0.999 else ("CLOSE" if cos > 0.99 else "DIVERGED"))
    logger.info("  [%s] %s: cos=%.6f max_diff=%.4f mean_diff=%.6f exact=%s",
                status, name, cos, max_diff, mean_diff, exact)
    return cos, exact


def run_hf_reference():
    """Run HF model on CPU, capture embedding + per-layer hidden states + logits."""
    from transformers import AutoModelForCausalLM

    logger.info("Loading HF model on CPU (BF16, trust_remote_code=True)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    logger.info("HF model loaded in %.1fs", time.perf_counter() - t0)

    # Register hooks on each decoder layer
    layer_outputs = {}
    def make_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            layer_outputs[idx] = h.detach().clone()
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Hook embedding
    embedding_out = {}
    def embed_hook(module, input, output):
        embedding_out['embed'] = output.detach().clone()
    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # Forward pass
    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    logger.info("Running HF forward (CPU, %d tokens)...", len(TOKEN_IDS))
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
    logits = outputs.logits.squeeze(0).float().cpu()
    logger.info("HF forward done in %.1fs", time.perf_counter() - t0)

    embed = embedding_out['embed'].squeeze(0).cpu()

    # Collect layer outputs as list
    n_layers = len(model.model.layers)
    layers = []
    for i in range(n_layers):
        layers.append(layer_outputs[i].squeeze(0).cpu())

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Log top predictions
    last_logits = logits[-1]
    topk_vals, topk_ids = last_logits.topk(5)
    logger.info("HF top5: %s",
                [(f"{v:.1f}", tid.item()) for v, tid in zip(topk_vals, topk_ids)])

    # Free model
    del model
    gc.collect()

    return embed, layers, logits


def run_krasis():
    """Run Krasis model, capture embedding + per-layer hidden states + logits."""
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel
    from krasis.kv_cache import SequenceKVState
    from krasis.layer import TransformerLayer

    logger.info("Loading Krasis model (BF16 everything)...")
    t0 = time.perf_counter()
    qcfg = QuantConfig(
        attention="bf16", shared_expert="bf16",
        dense_mlp="bf16", lm_head="bf16",
    )
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        gpu_prefill=False,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.bfloat16,
    )
    model.load()
    logger.info("Krasis model loaded in %.1fs", time.perf_counter() - t0)

    device = torch.device(model.ranks[0].device)

    # Capture per-layer (hidden, residual) by patching TransformerLayer.forward
    layer_data = []
    original_forward = TransformerLayer.forward

    def patched_forward(self, hidden, residual, *args, **kwargs):
        hidden_out, residual_out = original_forward(self, hidden, residual, *args, **kwargs)
        # Full residual stream after this layer = residual + hidden (delayed add)
        full_stream = (residual_out + hidden_out).detach().cpu().clone()
        layer_data.append({
            'layer_idx': self.layer_idx,
            'hidden': hidden_out.detach().cpu().clone(),
            'residual': residual_out.detach().cpu().clone(),
            'full': full_stream,
        })
        return hidden_out, residual_out

    TransformerLayer.forward = patched_forward

    # Capture embedding
    embed = model.embedding[
        torch.tensor(TOKEN_IDS, dtype=torch.long, device=device)
    ].detach().cpu().clone()

    # Forward pass
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]
    token_ids = torch.tensor(TOKEN_IDS, dtype=torch.long, device=device)
    positions = torch.arange(len(TOKEN_IDS), dtype=torch.int32, device=device)

    logger.info("Running Krasis forward (GPU, %d tokens)...", len(TOKEN_IDS))
    t0 = time.perf_counter()
    logits = model.forward(token_ids, positions, seq_states).float().cpu()
    logger.info("Krasis forward done in %.1fs", time.perf_counter() - t0)

    # Restore original forward
    TransformerLayer.forward = original_forward

    # Extract per-layer full residual stream (corresponds to HF layer output)
    layers = [d['full'] for d in layer_data]

    # Log top predictions
    last_logits = logits[-1]
    topk_vals, topk_ids = last_logits.topk(5)
    logger.info("Krasis top5: %s",
                [(f"{v:.1f}", tid.item()) for v, tid in zip(topk_vals, topk_ids)])

    # Clean up
    for s in seq_states:
        s.free()

    return embed, layers, logits


def main():
    logger.info("=" * 60)
    logger.info("Layer-by-layer comparison: HF vs Krasis")
    logger.info("Token IDs: %s (%d tokens)", TOKEN_IDS, len(TOKEN_IDS))
    logger.info("=" * 60)

    # Run HF reference
    hf_embed, hf_layers, hf_logits = run_hf_reference()
    logger.info("HF: embedding %s, %d layers, logits %s",
                hf_embed.shape, len(hf_layers), hf_logits.shape)

    # Run Krasis
    kr_embed, kr_layers, kr_logits = run_krasis()
    logger.info("Krasis: embedding %s, %d layers, logits %s",
                kr_embed.shape, len(kr_layers), kr_logits.shape)

    # ── Compare ──
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON (last token position)")
    logger.info("=" * 60)

    compare("Embedding", hf_embed, kr_embed)

    n_layers = min(len(hf_layers), len(kr_layers))
    diverged_at = None
    for i in range(n_layers):
        cos, exact = compare(f"Layer {i:2d} ({('dense' if i == 0 else 'MoE  ')})",
                             hf_layers[i], kr_layers[i])
        if cos < 0.99 and diverged_at is None:
            diverged_at = i
            logger.info("  *** FIRST SIGNIFICANT DIVERGENCE at layer %d ***", i)

    compare("Final logits", hf_logits, kr_logits)

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if diverged_at is not None:
        logger.info("First divergence: layer %d", diverged_at)
        if diverged_at == 0:
            logger.info("Bug is in: attention or dense MLP (layer 0)")
        else:
            logger.info("Bug is in: MoE layers (layer %d is first MoE)", diverged_at)
            logger.info("Previous layer (%d) was still OK", diverged_at - 1)
    else:
        logger.info("All layers match! (cos > 0.99 everywhere)")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

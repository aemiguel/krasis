#!/usr/bin/env python3
"""Attention layer ablation: skip each layer's attention one at a time, measure PPL delta.

Identifies which attention layers are critical vs redundant for output quality.
Layers with small PPL delta are candidates for VRAM offloading.

Usage:
    python -m perplexity.ablate_attention --model-path ~/.krasis/Qwen3-Coder-Next \
        --max-tokens 10000
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_krasis_root = _script_dir.parent
sys.path.insert(0, str(_krasis_root / "python"))

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.layer import _rmsnorm, _fused_add_rmsnorm
from perplexity.measure_ppl import load_dataset_text

logger = logging.getLogger(__name__)


def evaluate_ppl_with_skip(model, tokens, window_size, stride, skip_layers=None):
    """Run PPL evaluation, skipping attention for specified layers.

    When a layer is skipped, its attention output is set to zeros so the
    residual stream passes through unmodified (only norms are applied).
    """
    skip_layers = set(skip_layers or [])
    device = torch.device(model.ranks[0].device)
    total_nll = 0.0
    total_scored = 0

    starts = list(range(0, len(tokens) - 1, stride))

    for begin in starts:
        end = min(begin + window_size, len(tokens))
        win_len = end - begin
        if win_len < 2:
            break

        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

        if model.cfg.is_hybrid:
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        gpu_store = getattr(model, '_gpu_decode_store', None)
        evicted = 0
        if gpu_store is not None:
            evicted, _ = gpu_store.py_hcs_evict_for_prefill(win_len)

        try:
            token_tensor = torch.tensor(tokens[begin:end], dtype=torch.long, device=device)
            positions = torch.arange(win_len, dtype=torch.int32, device=device)

            with torch.inference_mode():
                logits = _forward_with_skip(
                    model, token_tensor, positions, seq_states, skip_layers
                )

            shift_logits = logits[:-1, :].float()
            shift_labels = token_tensor[1:]
            loss_per_pos = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="none"
            )

            if begin == 0:
                score_start = 0
            else:
                score_start = stride - 1

            scored_loss = loss_per_pos[score_start:]
            n_scored = scored_loss.shape[0]
            if n_scored > 0:
                total_nll += scored_loss.sum().item()
                total_scored += n_scored

            del logits, shift_logits, loss_per_pos, scored_loss

        finally:
            if gpu_store is not None and evicted > 0:
                gpu_store.py_hcs_reload_after_prefill()
            for s in seq_states:
                s.free()

    if total_scored == 0:
        return float("inf")
    return math.exp(total_nll / total_scored)


def _forward_with_skip(model, tokens, positions, seq_states, skip_layers):
    """Model forward pass that skips attention for specified layers.

    Monkey-patches forward_attn temporarily to produce zero attention output
    for skipped layers, so the residual passes through with only norms applied.
    """
    # We need to intercept at the layer level. The simplest approach:
    # temporarily replace forward_attn on skipped layers.

    original_fns = {}
    for layer_idx in skip_layers:
        if layer_idx < len(model.layers):
            layer = model.layers[layer_idx]
            original_fns[layer_idx] = layer.forward_attn
            layer.forward_attn = _make_skip_attn(layer)

    try:
        logits = model.forward(
            tokens, positions, seq_states,
            return_all_logits=True,
        )
    finally:
        # Restore original functions
        for layer_idx, fn in original_fns.items():
            model.layers[layer_idx].forward_attn = fn

    return logits


def _make_skip_attn(layer):
    """Create a replacement forward_attn that skips attention computation.

    Does the pre-attention and post-attention norms (to keep residual stream
    consistent) but replaces attention output with zeros.
    """
    def skip_forward_attn(hidden, residual, positions, kv_cache, seq_state,
                          layer_offset, num_new_tokens=0):
        # Pre-attention norm (same as normal)
        if residual is None:
            residual = hidden.clone()
            hidden = _rmsnorm(
                hidden, layer.input_norm_weight, layer.cfg.rms_norm_eps
            )
        else:
            _fused_add_rmsnorm(
                hidden, residual, layer.input_norm_weight, layer.cfg.rms_norm_eps
            )

        # Skip attention: zero output
        attn_out = torch.zeros_like(hidden)

        # Post-attention norm
        _fused_add_rmsnorm(
            attn_out, residual, layer.post_attn_norm_weight, layer.cfg.rms_norm_eps
        )
        return attn_out, residual

    return skip_forward_attn


def main():
    p = argparse.ArgumentParser(description="Attention layer ablation study")
    p.add_argument("--model-path", required=True)
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=10000,
                    help="Tokens to evaluate (default: 10000 for speed)")
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--dataset", default="wikitext-2")
    p.add_argument("--kv-cache-mb", type=int, default=1000)
    p.add_argument("--skip-layers", type=str, default=None,
                    help="Comma-separated layer indices to test (default: all)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    stride = args.window_size // 2

    # Load model
    print(f"Loading model from {args.model_path}...")
    quant_cfg = QuantConfig(
        attention="bf16", shared_expert="int8", dense_mlp="int8",
        lm_head="int8", gpu_expert_bits=4, cpu_expert_bits=4,
    )
    model = KrasisModel(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        layer_group_size=2,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=48,
        gpu_prefill=True,
        kv_cache_mb=args.kv_cache_mb,
    )
    model.load()

    # Load and tokenize dataset
    print(f"\nLoading {args.dataset}...")
    text = load_dataset_text(args.dataset)
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    if args.max_tokens:
        tokens = tokens[:args.max_tokens]
    print(f"Using {len(tokens):,} tokens\n")

    num_layers = len(model.layers)

    # Determine which layers to test
    if args.skip_layers:
        test_layers = [int(x) for x in args.skip_layers.split(",")]
    else:
        test_layers = list(range(num_layers))

    # Baseline PPL (no layers skipped)
    print("Computing baseline PPL (no skip)...")
    t0 = time.perf_counter()
    baseline_ppl = evaluate_ppl_with_skip(model, tokens, args.window_size, stride)
    t_baseline = time.perf_counter() - t0
    print(f"  Baseline PPL: {baseline_ppl:.4f} ({t_baseline:.1f}s)\n")

    # Per-layer ablation
    results = []
    print(f"Ablating {len(test_layers)} attention layers...\n")
    print(f"{'Layer':>5}  {'Type':>8}  {'PPL':>10}  {'Delta':>10}  {'Delta%':>8}  {'Impact':>8}")
    print("-" * 65)

    for layer_idx in test_layers:
        layer = model.layers[layer_idx]
        layer_type = layer.layer_type[:3].upper()  # GQA or LIN

        t0 = time.perf_counter()
        ppl = evaluate_ppl_with_skip(model, tokens, args.window_size, stride,
                                     skip_layers=[layer_idx])
        elapsed = time.perf_counter() - t0

        delta = ppl - baseline_ppl
        delta_pct = (delta / baseline_ppl) * 100

        # Categorize impact
        if delta_pct < 1.0:
            impact = "NONE"
        elif delta_pct < 5.0:
            impact = "LOW"
        elif delta_pct < 20.0:
            impact = "MED"
        else:
            impact = "HIGH"

        results.append({
            "layer": layer_idx,
            "type": layer.layer_type,
            "ppl": ppl,
            "delta": delta,
            "delta_pct": delta_pct,
            "impact": impact,
        })

        print(f"  {layer_idx:>3}  {layer_type:>8}  {ppl:>10.2f}  {delta:>+10.2f}  {delta_pct:>+7.1f}%  {impact:>6}")
        sys.stdout.flush()

    # Summary
    print(f"\n{'=' * 65}")
    print(f"SUMMARY")
    print(f"{'=' * 65}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Total layers: {num_layers}")

    none_layers = [r for r in results if r["impact"] == "NONE"]
    low_layers = [r for r in results if r["impact"] == "LOW"]
    med_layers = [r for r in results if r["impact"] == "MED"]
    high_layers = [r for r in results if r["impact"] == "HIGH"]

    print(f"\nNONE impact (<1% PPL delta): {len(none_layers)} layers")
    if none_layers:
        print(f"  Layers: {', '.join(str(r['layer']) for r in none_layers)}")

    print(f"LOW impact (1-5% PPL delta): {len(low_layers)} layers")
    if low_layers:
        print(f"  Layers: {', '.join(str(r['layer']) for r in low_layers)}")

    print(f"MED impact (5-20% PPL delta): {len(med_layers)} layers")
    if med_layers:
        print(f"  Layers: {', '.join(str(r['layer']) for r in med_layers)}")

    print(f"HIGH impact (>20% PPL delta): {len(high_layers)} layers")
    if high_layers:
        print(f"  Layers: {', '.join(str(r['layer']) for r in high_layers)}")

    # VRAM savings potential
    if none_layers or low_layers:
        offloadable = none_layers + low_layers
        # Estimate VRAM per attention layer
        attn_mb_per_layer = model._estimate_attention_vram() / num_layers / (1024 * 1024)
        saveable_mb = len(offloadable) * attn_mb_per_layer
        print(f"\nPotential VRAM savings: {len(offloadable)} layers x {attn_mb_per_layer:.0f} MB = {saveable_mb:.0f} MB")

    # Save results
    output_path = _script_dir / "results" / f"attention_ablation_{Path(args.model_path).name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "model": args.model_path,
        "dataset": args.dataset,
        "max_tokens": args.max_tokens,
        "baseline_ppl": baseline_ppl,
        "layers": results,
    }, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

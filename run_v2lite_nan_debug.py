"""Debug: find exactly which layer/component produces NaN in V2-Lite at long sequences."""

import os
import sys
import time
import torch
import math

os.environ.setdefault("KRASIS_LOG", "warn")

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.timing import TIMING

TIMING.decode = False
TIMING.prefill = False


def trace_prefill(model, prompt_text, label=""):
    """Manually step through prefill layer by layer, checking for NaN."""
    import flashinfer

    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt_text}]
    tokens = tokenizer.apply_chat_template(messages)
    M = len(tokens)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Prompt tokens: {M}")
    print(f"{'='*70}")

    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    for ss in seq_states:
        if ss is not None:
            ss.ensure_capacity(M)

    with torch.inference_mode():
        token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(M, dtype=torch.int32, device=device)
        hidden = model.embedding[token_ids]

        first_k = model.cfg.first_k_dense_replace

        # Get GPU prefill manager
        mgr = next(iter(model.gpu_prefill_managers.values()), None)

        residual = None
        first_nan_layer = None

        for abs_layer_idx in range(model.cfg.num_hidden_layers):
            layer = model.layers[abs_layer_idx]
            moe_layer_idx = abs_layer_idx - first_k if abs_layer_idx >= first_k else None
            kv_layer_offset = model._kv_layer_offsets.get(abs_layer_idx, 0)

            # Stream attention load
            if model._stream_attn_enabled:
                buf_idx = abs_layer_idx % 2
                if model._stream_attn_loaded.get(buf_idx) != abs_layer_idx:
                    model._stream_attn_load(abs_layer_idx, buf_idx)

            # Pre-attention norm
            if residual is None:
                residual = hidden
                hidden = flashinfer.norm.rmsnorm(
                    hidden, layer.input_norm_weight, model.cfg.rms_norm_eps
                )
            else:
                flashinfer.norm.fused_add_rmsnorm(
                    hidden, residual, layer.input_norm_weight, model.cfg.rms_norm_eps
                )

            # Check after pre-attn norm
            torch.cuda.synchronize()
            h_nan = torch.isnan(hidden).any().item()
            r_nan = torch.isnan(residual).any().item()
            if h_nan or r_nan:
                print(f"  Layer {abs_layer_idx}: NaN AFTER pre-attn norm (hidden={h_nan}, residual={r_nan})")
                if first_nan_layer is None:
                    first_nan_layer = abs_layer_idx
                break

            # Attention
            kv_cache = model.kv_caches[0]
            seq_state = seq_states[0]
            if layer.layer_type == "linear_attention":
                attn_out = layer.attention.forward(hidden, is_decode=False)
            else:
                if kv_layer_offset >= 0:
                    attn_out = layer.attention.forward(
                        hidden, positions, kv_cache, seq_state, kv_layer_offset,
                        num_new_tokens=M,
                    )
                else:
                    attn_out = layer.attention.forward(
                        hidden, positions, None, None, -1, num_new_tokens=M,
                    )

            torch.cuda.synchronize()
            attn_nan = torch.isnan(attn_out).any().item()
            if attn_nan:
                print(f"  Layer {abs_layer_idx}: NaN AFTER attention")
                # Check RoPE
                if hasattr(layer.attention, '_rope_cos_sin') and layer.attention._rope_cos_sin is not None:
                    cos, sin = layer.attention._rope_cos_sin
                    cos_nan = torch.isnan(cos[:M]).any().item()
                    sin_nan = torch.isnan(sin[:M]).any().item()
                    cos_inf = torch.isinf(cos[:M]).any().item()
                    sin_inf = torch.isinf(sin[:M]).any().item()
                    print(f"    RoPE cos NaN={cos_nan} inf={cos_inf}, sin NaN={sin_nan} inf={sin_inf}")
                    print(f"    RoPE cos range: [{cos[:M].min():.6f}, {cos[:M].max():.6f}]")
                    print(f"    RoPE sin range: [{sin[:M].min():.6f}, {sin[:M].max():.6f}]")
                if first_nan_layer is None:
                    first_nan_layer = abs_layer_idx
                break

            # Post-attention norm
            flashinfer.norm.fused_add_rmsnorm(
                attn_out, residual, layer.post_attn_norm_weight, model.cfg.rms_norm_eps
            )
            hidden = attn_out

            torch.cuda.synchronize()
            h_nan = torch.isnan(hidden).any().item()
            r_nan = torch.isnan(residual).any().item()
            if h_nan or r_nan:
                print(f"  Layer {abs_layer_idx}: NaN AFTER post-attn norm (hidden={h_nan}, residual={r_nan})")
                if first_nan_layer is None:
                    first_nan_layer = abs_layer_idx
                break

            # MoE / Dense MLP — use CPU path to isolate attention as NaN source
            if moe_layer_idx is not None:
                mlp_out = layer._moe_forward(hidden, moe_layer_idx)
            else:
                mlp_out = layer._dense_mlp_forward(hidden)

            torch.cuda.synchronize()
            mlp_nan = torch.isnan(mlp_out).any().item()
            if mlp_nan:
                print(f"  Layer {abs_layer_idx}: NaN AFTER MLP/MoE")
                if first_nan_layer is None:
                    first_nan_layer = abs_layer_idx
                break

            hidden = mlp_out
            h_norm = hidden.float().norm().item()
            r_norm = residual.float().norm().item()
            h_max = hidden.float().abs().max().item()

            status = "OK"
            if h_max > 1e4:
                status = "WARN (large values)"
            print(f"  Layer {abs_layer_idx:2d}: h_norm={h_norm:10.2f}  r_norm={r_norm:10.2f}  h_max={h_max:8.2f}  [{status}]")

        if first_nan_layer is None:
            # Got through all layers, compute logits
            last_h = hidden[-1:]
            last_r = residual[-1:]
            flashinfer.norm.fused_add_rmsnorm(
                last_h, last_r, model.final_norm, model.cfg.rms_norm_eps
            )
            from krasis.layer import _linear
            logits = _linear(last_h, model.lm_head_data).float()
            nan_count = torch.isnan(logits).sum().item()
            print(f"\n  Final logits: NaN={nan_count}, max={logits.max().item():.2f}, mean={logits.mean().item():.4f}")
        else:
            print(f"\n  FIRST NaN at layer {first_nan_layer}")

    for s in seq_states:
        s.free()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--layer-group-size", type=int, default=2)
    args = parser.parse_args()

    quant_cfg = QuantConfig(attention="int8", shared_expert="int8", dense_mlp="int8", lm_head="int8")

    model = KrasisModel(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        layer_group_size=args.layer_group_size,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=12,
    )
    model.load()

    # Warmup
    from krasis.kv_cache import SequenceKVState
    from krasis.sampler import sample
    dev = torch.device(model.ranks[0].device)
    ss = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]
    toks = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
    t = torch.tensor(toks, dtype=torch.long, device=dev)
    p = torch.arange(len(toks), dtype=torch.int32, device=dev)
    model.forward(t, p, ss)
    for s in ss:
        s.free()

    # Test 1: Medium (should work)
    medium = "Summarize: " + " ".join(
        f"Section {i}: Found {80+i%20}% accuracy with {i*10}ms latency."
        for i in range(1, 15)
    )
    trace_prefill(model, medium, "MEDIUM (~500 words) — should work")

    # Test 2: Long (should fail at ~4K tokens)
    long_prompt = (
        "Analyze: " + " ".join(
            f"Chapter {ch}: Section {s}. "
            f"Experiment {ch*100+s} found latency scales logarithmically. "
            f"Throughput achieves {90+(ch*s)%10}% efficiency."
            for ch in range(1, 11)
            for s in range(1, 8)
        )
    )
    trace_prefill(model, long_prompt, "LONG (~3K words) — likely NaN")

    # Test 3: Around the boundary (~2K tokens)
    boundary = (
        "Summarize: " + " ".join(
            f"Section {i}: Experiment {i} found {80+i%20}% accuracy."
            for i in range(1, 40)
        )
    )
    trace_prefill(model, boundary, "BOUNDARY (~1500 tokens) — boundary check")

    print("\nDone.")

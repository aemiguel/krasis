"""Compare GPU Marlin MoE output vs CPU Rust MoE output at each layer.

Identifies exactly where GPU prefill diverges from CPU.
"""

import os
import sys
import time
import torch

os.environ.setdefault("KRASIS_LOG", "warn")

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.timing import TIMING

TIMING.decode = False
TIMING.prefill = False


def compare_moe_outputs(model, prompt_text, max_layers=5):
    """Run prefill and compare GPU vs CPU MoE at each layer."""
    import flashinfer

    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt_text}]
    tokens = tokenizer.apply_chat_template(messages)
    M = len(tokens)

    print(f"  Prompt tokens: {M}")

    device = torch.device(model.ranks[0].device)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    # Ensure KV capacity
    for ss in seq_states:
        if ss is not None:
            ss.ensure_capacity(M)

    with torch.inference_mode():
        # Embedding
        token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(M, dtype=torch.int32, device=device)
        hidden = model.embedding[token_ids]

        first_k = model.cfg.first_k_dense_replace

        # Get the GPU prefill manager
        mgr = next(iter(model.gpu_prefill_managers.values()), None)
        if mgr is None:
            print("  No GPU prefill manager!")
            return

        residual = None
        compared = 0
        for abs_layer_idx in range(model.cfg.num_hidden_layers):
            layer = model.layers[abs_layer_idx]
            layer_dev = torch.device(layer.device)
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

            # Post-attention norm
            flashinfer.norm.fused_add_rmsnorm(
                attn_out, residual, layer.post_attn_norm_weight, model.cfg.rms_norm_eps
            )
            hidden = attn_out

            # MoE comparison
            if moe_layer_idx is not None and compared < max_layers:
                # 1. Routing (same for both)
                topk_ids, topk_weights = layer.compute_routing(hidden)

                # 2. GPU Marlin MoE
                # Need to load experts for this layer
                mgr.preload_layer_group([moe_layer_idx])
                gpu_output = mgr.forward(moe_layer_idx, hidden, topk_ids, topk_weights)
                mgr.free_layer_group()

                # 3. CPU Rust MoE
                cpu_output = layer._routed_expert_forward(hidden, topk_ids, topk_weights, moe_layer_idx)

                # 4. Compare
                torch.cuda.synchronize()
                diff = (gpu_output.float() - cpu_output.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                cosine = torch.nn.functional.cosine_similarity(
                    gpu_output.float().reshape(1, -1),
                    cpu_output.float().reshape(1, -1),
                ).item()

                gpu_norm = gpu_output.float().norm().item()
                cpu_norm = cpu_output.float().norm().item()

                status = "OK" if cosine > 0.99 else ("WARN" if cosine > 0.9 else "BAD")
                print(f"  Layer {abs_layer_idx} (moe={moe_layer_idx}): "
                      f"cosine={cosine:.6f} max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} "
                      f"gpu_norm={gpu_norm:.2f} cpu_norm={cpu_norm:.2f} [{status}]")

                # Use CPU output for continued forward (known correct)
                mlp_out = cpu_output
                compared += 1
            elif moe_layer_idx is not None:
                # Use CPU for remaining layers
                mlp_out = layer._moe_forward(hidden, moe_layer_idx)
            else:
                # Dense layer
                mlp_out = layer._dense_mlp_forward(hidden)

            # Update residual
            hidden = mlp_out
            # Note: hidden is now the MLP output, residual is the post-attn residual
            # On next layer, fused_add_rmsnorm will compute: residual += hidden, hidden = rmsnorm(residual)

        # Final norm + lm_head
        last_h = hidden[-1:]
        last_r = residual[-1:]
        flashinfer.norm.fused_add_rmsnorm(
            last_h, last_r, model.final_norm, model.cfg.rms_norm_eps
        )
        from krasis.layer import _linear
        logits = _linear(last_h, model.lm_head_data).float()

        nan_count = torch.isnan(logits).sum().item()
        print(f"  Final logits: NaN={nan_count}, max={logits.max().item():.2f}, mean={logits.mean().item():.4f}")

    for s in seq_states:
        s.free()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--layer-group-size", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=5)
    args = parser.parse_args()

    quant_cfg = QuantConfig(attention="awq", shared_expert="int8", dense_mlp="int8", lm_head="int8")

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
    print("Warming up...")
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
    print("Warmup done.\n")

    # Test at different lengths
    for words in [200, 500, 1000, 1500]:
        n = max(1, words // 40)
        from prompt_utils import load_prompt_text_truncated
        prompt = "Summarize: " + load_prompt_text_truncated(max_chars=words * 5)
        print(f"\n{'='*70}")
        print(f"  COMPARING GPU vs CPU MoE — ~{words} words")
        print(f"{'='*70}")
        compare_moe_outputs(model, prompt, max_layers=args.max_layers)

    print("\nDone.")

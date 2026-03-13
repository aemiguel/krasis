#!/usr/bin/env python3
"""Focused layer-0 debug: compare Python vs Rust at each substep.

Runs Python layer 0 manually step-by-step, capturing:
  1. Embedding
  2. After pre-attn norm
  3. After LA attention
  4. After post-attn norm (= MoE input)
  5. After MoE (= layer 0 output)

Then runs Rust for just layer 0 (debug_stop_layer=1) and compares.
"""

import sys
import os
import numpy as np

import torch
torch.set_default_dtype(torch.bfloat16)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel, _linear
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState
import flashinfer

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def bf16_to_f32(u16_list):
    """Convert list of BF16 u16 values to FP32 tensor."""
    arr = np.array(u16_list, dtype=np.uint16)
    buf = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return buf.float()


def show(name, t, n=8):
    """Print tensor info."""
    if isinstance(t, torch.Tensor):
        t = t.float()
        vals = t.flatten()[:n].tolist()
        print(f"  {name}: norm={t.norm().item():.6f} [{', '.join(f'{v:.6f}' for v in vals)}]")
    else:
        print(f"  {name}: {t}")


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    print(f"Loading model on GPU {gpu_id}...")
    quant = QuantConfig(
        gpu_expert_bits=4,
        cpu_expert_bits=4,
        attention="bf16",
        shared_expert="int8",
        dense_mlp="int8",
        lm_head="int8",
    )

    model = KrasisModel(
        model_path=MODEL_PATH,
        pp_partition=[48],
        num_gpus=1,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=16,
        quant_cfg=quant,
        layer_group_size=2,
        gpu_prefill_threshold=1,
        stream_attention=True,
    )
    model.load()
    tokenizer = Tokenizer(MODEL_PATH)

    prompt = "What is 2+2?"
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # === Prefill ===
    seq_states = [
        SequenceKVState(c, seq_id=0) if c is not None else None
        for c in model.kv_caches
    ]
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()

    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    with torch.inference_mode():
        logits = model.forward(prompt_tensor, positions, seq_states)
        first_token = logits[-1:, :].argmax(dim=-1).item()

    print(f"First token after prefill: {first_token} = '{tokenizer.decode([first_token])}'")

    # === Save post-prefill LA states ===
    saved_conv = {}
    saved_recur = {}
    for layer_idx, layer in enumerate(model.layers):
        if layer.layer_type == "linear_attention":
            saved_conv[layer_idx] = layer.attention._conv_state.clone()
            saved_recur[layer_idx] = layer.attention._recurrent_state.clone()

    # === PYTHON: Manual layer 0 step-by-step ===
    print("\n" + "=" * 60)
    print("PYTHON LAYER 0 STEP-BY-STEP")
    print("=" * 60)

    # Restore LA states
    for li in saved_conv:
        model.layers[li].attention._conv_state = saved_conv[li].clone()
        model.layers[li].attention._recurrent_state = saved_recur[li].clone()

    layer0 = model.layers[0]
    eps = model.cfg.rms_norm_eps
    next_pos = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)

    # Load streaming attention weights for layer 0
    if hasattr(model, '_stream_attn_enabled') and model._stream_attn_enabled:
        model._stream_attn_load(0, buf_idx=0)
        print("  (loaded streaming attention weights for layer 0)")

    with torch.inference_mode():
        # Step 1: Embedding
        py_emb = model.embedding[first_token].unsqueeze(0).to(device)
        show("PY embedding", py_emb[0])

        # Step 2: Pre-attn norm (first layer: residual = hidden, hidden = rmsnorm(hidden))
        py_residual = py_emb.clone()
        py_hidden = flashinfer.norm.rmsnorm(
            py_emb, layer0.input_norm_weight, eps
        )
        show("PY after pre-attn norm (hidden)", py_hidden[0])
        show("PY after pre-attn norm (residual)", py_residual[0])

        # Step 3: LA attention
        py_attn_out = layer0.attention.forward(py_hidden, is_decode=True)
        show("PY after LA attention", py_attn_out[0])

        # Step 4: Post-attn norm (fused: residual += attn_out, attn_out = rmsnorm(residual))
        flashinfer.norm.fused_add_rmsnorm(
            py_attn_out, py_residual, layer0.post_attn_norm_weight, eps
        )
        show("PY after post-attn norm (hidden=MoE input)", py_attn_out[0])
        show("PY after post-attn norm (residual)", py_residual[0])

        # Step 5: MoE
        # First show routing
        router_logits = torch.matmul(py_attn_out.float(), layer0.gate_weight.float().t())
        scores = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(scores, model.cfg.num_experts_per_tok, dim=-1)
        topk_weights_orig = scores.gather(1, topk_ids)
        if model.cfg.norm_topk_prob:
            topk_weights_norm = topk_weights_orig / topk_weights_orig.sum(dim=-1, keepdim=True)
        else:
            topk_weights_norm = topk_weights_orig
        print(f"  PY routing: ids={topk_ids[0].tolist()}")
        print(f"  PY routing: weights={[f'{w:.6f}' for w in topk_weights_norm[0].tolist()]}")
        print(f"  PY routing: weight_sum={topk_weights_norm[0].sum().item():.6f}")

        py_moe_out = layer0._moe_forward(py_attn_out, moe_layer_idx=0)
        show("PY after MoE (layer 0 output)", py_moe_out[0])

    # === RUST: Layer 0 only ===
    print("\n" + "=" * 60)
    print("RUST LAYER 0 (debug_stop_layer=1)")
    print("=" * 60)

    # Restore LA states
    for li in saved_conv:
        model.layers[li].attention._conv_state = saved_conv[li].clone()
        model.layers[li].attention._recurrent_state = saved_recur[li].clone()

    # Set up Rust decode store
    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Run Rust for just layer 0
    store.set_debug_capture_layers(True)
    store.set_debug_stop_layer(1)
    store.py_gpu_decode_step(first_token, len(prompt_tokens))
    torch.cuda.synchronize()

    # Get Rust layer 0 output
    rust_captures = store.download_layer_captures()
    rust_hidden = bf16_to_f32(store.download_hidden_bf16())
    rust_residual = bf16_to_f32(store.download_residual_bf16())
    rust_moe_out = bf16_to_f32(store.download_moe_out_bf16())

    show("RUST d_hidden (after layer 0 MoE)", rust_hidden)
    show("RUST d_residual", rust_residual)
    show("RUST d_moe_out", rust_moe_out)

    if rust_captures:
        rust_l0 = bf16_to_f32(rust_captures[0])
        show("RUST capture[0] (d_hidden after layer 0)", rust_l0)

    # === Comparison ===
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    py_out = py_moe_out[0].float().cpu()
    rust_out = rust_hidden.cpu()

    print(f"  Layer 0 MoE output cos_sim: {cos_sim(py_out, rust_out):.6f}")
    print(f"  py_norm={py_out.norm():.4f}, rust_norm={rust_out.norm():.4f}")

    # Compare residuals
    py_res = py_residual[0].float().cpu()
    rust_res = rust_residual.cpu()
    print(f"  Residual cos_sim: {cos_sim(py_res, rust_res):.6f}")
    print(f"  py_res_norm={py_res.norm():.4f}, rust_res_norm={rust_res.norm():.4f}")

    # Compare MoE out
    py_moe = py_moe_out[0].float().cpu()
    rust_moe = rust_moe_out.cpu()
    print(f"  d_moe_out cos_sim: {cos_sim(py_moe, rust_moe):.6f}")

    # Now let's also try to isolate: is the attention output correct?
    # We can upload Python's post-attn-norm hidden to Rust d_hidden,
    # then run just the MoE, and see if that output matches.
    print("\n" + "=" * 60)
    print("ISOLATING MoE: Upload Python's post-attn hidden -> Rust MoE")
    print("=" * 60)

    # Convert Python hidden to BF16 u16 for upload
    py_moe_input = py_attn_out[0].cpu()  # This is the post-attn normed hidden
    py_moe_input_bf16 = py_moe_input.contiguous().view(torch.int16).numpy().astype(np.uint16).tolist()

    store.upload_hidden_bf16(py_moe_input_bf16)
    torch.cuda.synchronize()

    # Run just MoE for layer 0
    route_ms, dma_ms, compute_ms, total_ms = store.moe_forward_gpu(0)
    torch.cuda.synchronize()

    rust_moe_from_py_input = bf16_to_f32(store.download_moe_out_bf16())
    show("RUST MoE(py_hidden) output", rust_moe_from_py_input)
    show("PY MoE output (reference)", py_moe_out[0])

    cos = cos_sim(py_moe[0:2048] if len(py_moe) >= 2048 else py_moe,
                  rust_moe_from_py_input[0:2048] if len(rust_moe_from_py_input) >= 2048 else rust_moe_from_py_input)
    print(f"  MoE isolation cos_sim: {cos:.6f}")

    if cos > 0.9:
        print("  => MoE is OK! Bug is in attention or norms.")
    else:
        print("  => MoE is WRONG even with correct input. Bug is in MoE kernels.")

        # Further isolate: show routing comparison
        # Rust routing is already logged via RUST_LOG=info

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

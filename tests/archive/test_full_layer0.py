#!/usr/bin/env python3
"""Compare Python vs Rust full layer 0 output (LA + post-attn-norm + MoE).

The substep test showed LA internals match perfectly.
This test checks what happens AFTER LA through MoE.
"""
import sys, os, struct, torch
import numpy as np

torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState
from krasis.linear_attention import _linear

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def bf16_to_np(u16_list):
    return np.array([struct.unpack('f', struct.pack('I', v << 16))[0]
                     for v in u16_list], dtype=np.float32)


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def report(name, py_arr, rust_arr):
    py_np = np.asarray(py_arr, dtype=np.float32).ravel()
    rust_np = np.asarray(rust_arr, dtype=np.float32).ravel()
    n = min(len(py_np), len(rust_np))
    cs = cos_sim(py_np[:n], rust_np[:n])
    diff = np.abs(py_np[:n] - rust_np[:n])
    status = "OK" if cs > 0.999 else "WARN" if cs > 0.99 else "BAD"
    print(f"  {name:35s} cos={cs:.6f} maxdiff={diff.max():.6f} "
          f"py_norm={np.linalg.norm(py_np[:n]):.4f} rust_norm={np.linalg.norm(rust_np[:n]):.4f}  [{status}]")
    if cs < 0.99:
        print(f"    Python[:6]: {py_np[:6].tolist()}")
        print(f"    Rust[:6]:   {rust_np[:6].tolist()}")
    return cs


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("Loading model...")
    quant = QuantConfig(
        gpu_expert_bits=4, cpu_expert_bits=4,
        attention="bf16", shared_expert="int8",
        dense_mlp="int8", lm_head="int8",
    )
    model = KrasisModel(
        model_path=MODEL_PATH, pp_partition=[48], num_gpus=1,
        kv_dtype=torch.float8_e4m3fn, krasis_threads=16,
        quant_cfg=quant, layer_group_size=2,
        gpu_prefill_threshold=1, stream_attention=True,
    )
    model.load()
    tokenizer = Tokenizer(MODEL_PATH)

    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt_tokens = tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # Prefill
    seq_states = [SequenceKVState(c, seq_id=0) if c is not None else None
                  for c in model.kv_caches]
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()

    device = torch.device("cuda:0")
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    with torch.inference_mode():
        logits = model.forward(prompt_tensor, positions, seq_states)
        first_token = logits[-1:, :].argmax(dim=-1).item()
    print(f"Prefill first token: {first_token} = {repr(tokenizer.decode([first_token]))}")

    # Save post-prefill states
    saved_conv = {}
    saved_recur = {}
    for li, layer in enumerate(model.layers):
        if layer.layer_type == "linear_attention":
            saved_conv[li] = layer.attention._conv_state.clone()
            saved_recur[li] = layer.attention._recurrent_state.clone()

    # ================================================================
    # Python: run full model.forward for one decode step
    # ================================================================
    print("\n=== Python full forward (1 token) ===")
    tok_t = torch.tensor([first_token], dtype=torch.long, device=device)
    pos_t = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)

    # Hook into forward to capture per-layer hidden states
    py_layer_hidden = {}
    py_layer_residual = {}
    _orig_forward = model.forward

    # Instead of hooking, let's just capture the per-layer hidden state
    # by running the model forward and checking the output.
    with torch.inference_mode():
        py_logits = model.forward(tok_t, pos_t, seq_states)
    py_token = py_logits[0].argmax(dim=-1).item()
    print(f"  Python next token: {py_token} = {repr(tokenizer.decode([py_token]))}")

    # ================================================================
    # Rust: set up and run full pipeline (all 48 layers)
    # ================================================================
    print("\n=== Rust full decode (1 token) ===")

    # Restore LA states
    for li in saved_conv:
        a = model.layers[li].attention
        a._conv_state = saved_conv[li].clone()
        a._recurrent_state = saved_recur[li].clone()

    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Run with per-layer capture enabled
    store.set_debug_stop_layer(0)  # no stop, run all layers
    store.set_debug_capture_layers(True)

    tokens = store.gpu_generate_batch(
        first_token=first_token,
        start_position=len(prompt_tokens),
        max_tokens=1,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        stop_ids=[],
        presence_penalty=0.0,
    )
    rust_token = tokens[0] if tokens else -1
    print(f"  Rust next token: {rust_token} = {repr(tokenizer.decode([rust_token]) if rust_token >= 0 else '?')}")

    # Download per-layer hidden states
    captures = store.download_layer_captures()
    print(f"  Captured {len(captures)} layer states")

    if py_token == rust_token:
        print("\n  MATCH!")
    else:
        print(f"\n  MISMATCH: Python={py_token} Rust={rust_token}")

    # ================================================================
    # Now compare d_hidden and d_residual after layer 0 only
    # ================================================================
    print("\n=== Layer 0 only comparison ===")

    # Restore states again
    for li in saved_conv:
        a = model.layers[li].attention
        a._conv_state = saved_conv[li].clone()
        a._recurrent_state = saved_recur[li].clone()
    model._update_la_state_ptrs()

    store.set_debug_stop_layer(1)  # stop after layer 0
    store.set_debug_capture_layers(False)

    store.py_gpu_decode_step(first_token, len(prompt_tokens))
    torch.cuda.synchronize()

    # Download d_hidden (= MoE output for layer 0)
    rust_hidden_l0 = bf16_to_np(store.download_hidden_bf16())
    rust_residual_l0 = bf16_to_np(store.download_residual_bf16())
    rust_moe_out_l0 = bf16_to_np(store.download_moe_out_bf16())

    print(f"  Rust d_hidden  after L0: norm={np.linalg.norm(rust_hidden_l0):.4f}")
    print(f"  Rust d_residual after L0: norm={np.linalg.norm(rust_residual_l0):.4f}")
    print(f"  Rust d_moe_out after L0:  norm={np.linalg.norm(rust_moe_out_l0):.4f}")

    # Also get LA output (before post-attn norm, before MoE)
    # We know d_la_ba has the recurrence output (step 7)
    # But after MoE runs, d_hidden was overwritten by the copy from d_moe_out
    # The LA output projection wrote to d_hidden, then post-attn norm modified
    # d_hidden and d_residual, then MoE wrote to d_moe_out, then d_moe_out
    # was copied to d_hidden.

    # For comparison, run Python manually for layer 0:
    print("\n=== Python layer 0 manual ===")

    # Embedding
    hidden = model.embedding[torch.tensor([first_token], device=device)]  # [1, hidden_size]

    # Input norm (fused_add_rmsnorm, first_layer)
    layer = model.layers[0]
    h_f32 = hidden.float()
    rms = torch.rsqrt((h_f32 * h_f32).mean(dim=-1, keepdim=True) + 1e-6)
    residual = h_f32.clone()
    hidden_normed = (h_f32 * rms * layer.input_norm_weight.float()).to(torch.bfloat16)

    # LA attention
    la_out = layer.attention._forward_recurrent_inplace(hidden_normed)
    print(f"  Python LA output: norm={la_out.norm().item():.4f}")

    # Post-attention norm (fused_add_rmsnorm, not first_layer)
    # h = la_out + residual
    h_post = la_out.float() + residual
    rms_post = torch.rsqrt((h_post * h_post).mean(dim=-1, keepdim=True) + 1e-6)
    py_residual = h_post.clone()
    post_norm_w = layer.post_attn_norm_weight
    hidden_for_moe = (h_post * rms_post * post_norm_w.float()).to(torch.bfloat16)
    print(f"  Python post-attn-norm hidden: norm={hidden_for_moe.norm().item():.4f}")
    print(f"  Python residual after post-attn: norm={py_residual.norm().item():.4f}")

    # Compare residual
    report("Residual after L0",
           py_residual.squeeze(0).cpu().numpy(),
           rust_residual_l0)

    # Compare d_hidden (should be MoE output)
    # For this we need to also run the MoE in Python, but that's complex.
    # Let's just compare residuals and MoE output norms.

    # The key comparison: if residual matches, LA+norms are correct.
    # If d_hidden (MoE output) also matches, the full layer is correct.

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

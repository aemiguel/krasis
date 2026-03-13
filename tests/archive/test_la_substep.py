#!/usr/bin/env python3
"""Compare Python vs Rust LA attention at every substep for layer 0.

Runs prefill, saves states, then runs ONE decode step through both paths,
comparing QKVZ projection, uninterleave, conv1d, gate/beta, recurrence, etc.

Usage: CUDA_VISIBLE_DEVICES=1 python tests/test_la_substep.py
"""
import sys, os, struct
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState
from krasis.linear_attention import _linear, _l2norm

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dot = np.dot(a.ravel(), b.ravel())
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def bf16_to_np(u16_list):
    return np.array([struct.unpack('f', struct.pack('I', v << 16))[0]
                     for v in u16_list], dtype=np.float32)


def report(name, py_arr, rust_arr):
    py_np = np.asarray(py_arr, dtype=np.float32).ravel()
    rust_np = np.asarray(rust_arr, dtype=np.float32).ravel()
    n = min(len(py_np), len(rust_np))
    py_np, rust_np = py_np[:n], rust_np[:n]
    cs = cos_sim(py_np, rust_np)
    diff = np.abs(py_np - rust_np)
    max_d = diff.max()
    mean_d = diff.mean()
    py_norm = np.linalg.norm(py_np)
    rust_norm = np.linalg.norm(rust_np)
    status = "OK" if cs > 0.999 else "WARN" if cs > 0.99 else "BAD"
    print(f"  {name:30s} cos={cs:.6f} maxdiff={max_d:.6f} meandiff={mean_d:.6f} "
          f"py_norm={py_norm:.4f} rust_norm={rust_norm:.4f}  [{status}]")
    if cs < 0.99:
        # Show first few values
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

    # Prompt
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt_tokens = tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # Prefill
    seq_states = [SequenceKVState(c, seq_id=0) if c is not None else None for c in model.kv_caches]
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()

    device = torch.device("cuda:0")
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    with torch.inference_mode():
        logits = model.forward(prompt_tensor, positions, seq_states)
        first_token = logits[-1:, :].argmax(dim=-1).item()
    print(f"Prefill first token: {first_token} = '{tokenizer.decode([first_token])}'")

    # Save post-prefill states
    saved_conv = {}
    saved_recur = {}
    for li, layer in enumerate(model.layers):
        if layer.layer_type == "linear_attention":
            saved_conv[li] = layer.attention._conv_state.clone()
            saved_recur[li] = layer.attention._recurrent_state.clone()

    # ================================================================
    # PYTHON: run layer 0 LA decode step manually, saving intermediates
    # ================================================================
    print("\n=== Python LA layer 0 decode ===")

    attn = model.layers[0].attention
    nk = attn.num_k_heads
    nv = attn.num_v_heads
    dk = attn.k_head_dim
    dv = attn.v_head_dim
    hr = attn.head_ratio
    key_dim = attn.key_dim
    conv_dim = attn.conv_dim
    kd = attn.kernel_dim
    print(f"  nk={nk} nv={nv} dk={dk} dv={dv} hr={hr} key_dim={key_dim} conv_dim={conv_dim} kd={kd}")

    # Get embedding for first_token
    token_t = torch.tensor([first_token], dtype=torch.long, device=device)
    hidden_py = model.embedding[token_t].squeeze(0)  # [hidden_size]

    # Apply input norm (layer 0, first_layer=True: just rmsnorm)
    input_norm_w = model.layers[0].input_norm_weight
    h_f32 = hidden_py.float()
    rms = torch.sqrt((h_f32 * h_f32).mean() + 1e-6)
    h_normed_bf16 = (h_f32 / rms * input_norm_w.float()).to(torch.bfloat16)
    # This is what goes into the LA attention as input
    py_hidden_normed = h_normed_bf16.unsqueeze(0)  # [1, hidden_size]

    # Step 1: Projections
    py_qkvz = _linear(py_hidden_normed, attn.in_proj_qkvz).float()
    py_ba = _linear(py_hidden_normed, attn.in_proj_ba).float()

    # Step 2: Un-interleave
    # Do this manually to match CUDA kernel exactly
    group_dim = 2 * dk + 2 * hr * dv
    qkvz_grouped = py_qkvz.view(1, nk, group_dim)  # [1, nk, group_dim]

    # Split
    py_q = qkvz_grouped[:, :, :dk]                        # [1, nk, dk]
    py_k = qkvz_grouped[:, :, dk:2*dk]                    # [1, nk, dk]
    py_v = qkvz_grouped[:, :, 2*dk:2*dk+hr*dv]            # [1, nk, hr*dv]
    py_z = qkvz_grouped[:, :, 2*dk+hr*dv:]                # [1, nk, hr*dv]

    # Flatten to conv_input layout: [q_flat(key_dim), k_flat(key_dim), v_flat(nv*dv)]
    py_conv_input = torch.cat([
        py_q.reshape(1, key_dim),
        py_k.reshape(1, key_dim),
        py_v.reshape(1, nv * dv),
    ], dim=-1).squeeze(0)  # [conv_dim]
    py_z_flat = py_z.reshape(1, nv * dv).squeeze(0)  # [nv*dv]

    # Step 3: Conv1d
    mixed_qkv = py_conv_input.unsqueeze(0).unsqueeze(-1)  # [1, conv_dim, 1]
    conv_input = torch.cat([saved_conv[0], mixed_qkv], dim=-1)  # [1, conv_dim, kd+1]
    new_conv_state = conv_input[:, :, -kd:]  # save for comparison

    conv_out = F.conv1d(
        conv_input.to(attn.conv1d_weight.dtype),
        attn.conv1d_weight, bias=None, padding=0, groups=conv_dim,
    )
    conv_out_silu = F.silu(conv_out[:, :, -1:]).float()  # [1, conv_dim, 1]
    conv_out_flat = conv_out_silu.squeeze(0).squeeze(-1)  # [conv_dim]

    # Split conv output into q, k, v
    py_conv_q = conv_out_flat[:key_dim].reshape(nk, dk)
    py_conv_k = conv_out_flat[key_dim:2*key_dim].reshape(nk, dk)
    py_conv_v = conv_out_flat[2*key_dim:].reshape(nv, dv)

    # Step 4: Gate and beta from BA
    ba_grouped = py_ba.view(1, nk, 2 * hr)
    py_b = ba_grouped[:, :, :hr].reshape(1, nv)
    py_a = ba_grouped[:, :, hr:].reshape(1, nv)
    py_beta = torch.sigmoid(py_b[0])
    py_g_raw = -attn.A_log.float().exp() * F.softplus(py_a[0].float() + attn.dt_bias)
    py_gate = py_g_raw.exp()

    # Step 5: Repeat interleave
    if hr > 1:
        py_conv_q_ri = py_conv_q.unsqueeze(1).expand(nk, hr, dk).reshape(nv, dk)
        py_conv_k_ri = py_conv_k.unsqueeze(1).expand(nk, hr, dk).reshape(nv, dk)
    else:
        py_conv_q_ri = py_conv_q
        py_conv_k_ri = py_conv_k

    # Step 6: L2 norm + scale
    py_q_normed = _l2norm(py_conv_q_ri.unsqueeze(0), dim=-1).squeeze(0) * attn.scale
    py_k_normed = _l2norm(py_conv_k_ri.unsqueeze(0), dim=-1).squeeze(0)

    # Step 7: Recurrence
    state = saved_recur[0].clone().float()  # [1, nv, dk, dv]
    state.mul_(py_gate.unsqueeze(-1).unsqueeze(-1))

    kv_mem = (state.squeeze(0) * py_k_normed.unsqueeze(-1)).sum(dim=-2)
    delta = (py_conv_v - kv_mem) * py_beta.unsqueeze(-1)
    state.add_(
        py_k_normed.unsqueeze(-1).unsqueeze(0) * delta.unsqueeze(-2).unsqueeze(0)
    )
    py_recur_out = (state.squeeze(0) * py_q_normed.unsqueeze(-1)).sum(dim=-2)

    # Step 8: Gated RMSNorm + SiLU
    py_gated_out = attn._gated_rmsnorm(
        py_recur_out.unsqueeze(0),
        py_z_flat.reshape(1, nv, dv)
    )

    # Step 9: Output projection
    py_la_output = _linear(
        py_gated_out.reshape(1, nv * dv).to(torch.bfloat16),
        attn.out_proj
    )

    print(f"  Python LA output norm: {py_la_output.norm().item():.6f}")
    print(f"  Python recurrence out norm: {py_recur_out.norm().item():.6f}")

    # ================================================================
    # RUST: run layer 0 decode step, download intermediates
    # ================================================================
    print("\n=== Rust LA layer 0 decode ===")

    # Restore LA states
    for li in saved_conv:
        a = model.layers[li].attention
        a._conv_state = saved_conv[li].clone()
        a._recurrent_state = saved_recur[li].clone()

    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Run one decode step, stop after layer 1 (layer 0 complete)
    store.set_debug_stop_layer(1)
    store.py_gpu_decode_step(first_token, len(prompt_tokens))
    torch.cuda.synchronize()

    # Download intermediates
    # After layer 0 LA:
    # d_la_qkvz: Step 1 output (after GEMV) = interleaved QKVZ projection, FP32
    #            After Step 3: conv output [q, k, v] with SiLU
    # d_la_ba: Step 1 output (after GEMV) = interleaved BA projection, FP32
    #          After Step 7: recurrence output [nv*dv]
    # d_la_conv_out: Step 2 output = uninterleaved [q_flat, k_flat, v_flat]
    #               After Step 4: [gate(nv), beta(nv)] at start
    #               After Step 8: gated rmsnorm output [nv*dv]
    # d_la_recur_out: Step 2 output = z_flat [nv*dv] (then saved to gated_out)
    #                 After Step 5: [q_ri(nv*dk), k_ri(nv*dk)]
    # d_la_gated_out: z_flat [nv*dv] (saved from recur_out before step 5)

    # These are FINAL values after all LA steps complete

    # Download the recurrence output (in d_la_ba after step 7)
    rust_recur_out = np.array(store.download_la_buffer_f32("ba", nv * dv), dtype=np.float32)

    # Download gated rmsnorm output (in d_la_conv_out after step 8)
    rust_gated_out = np.array(store.download_la_buffer_f32("conv_out", nv * dv), dtype=np.float32)

    # Download z values (saved in d_la_gated_out)
    rust_z = np.array(store.download_la_buffer_f32("gated_out", nv * dv), dtype=np.float32)

    # Download gate and beta (in d_la_conv_out, but they were overwritten by step 8)
    # Can't get them post-hoc. Need to check them differently.

    # Download conv output q,k (in d_la_qkvz after step 3, but step 5 used them for repeat_interleave)
    # After step 5, d_la_recur_out has [q_ri(nv*dk), k_ri(nv*dk)]
    rust_qk_ri = np.array(store.download_la_buffer_f32("recur_out", 2 * nv * dk), dtype=np.float32)
    rust_q_ri = rust_qk_ri[:nv*dk]
    rust_k_ri = rust_qk_ri[nv*dk:2*nv*dk]

    # After step 6, q_ri and k_ri are L2-normalized in-place
    # So rust_q_ri and rust_k_ri have the L2-normed+scaled values

    # Download d_hidden (LA output projection result, BF16)
    rust_hidden = bf16_to_np(store.download_hidden_bf16())

    print(f"  Rust recurrence out norm: {np.linalg.norm(rust_recur_out):.6f}")
    print(f"  Rust gated out norm: {np.linalg.norm(rust_gated_out):.6f}")

    # ================================================================
    # COMPARE
    # ================================================================
    print("\n=== Substep Comparison ===")

    # Z values (saved before overwrite)
    report("Z (saved)", py_z_flat.cpu().numpy(), rust_z)

    # Q after L2norm+scale (step 6)
    report("Q normed (nv*dk)", py_q_normed.cpu().numpy(), rust_q_ri)

    # K after L2norm (step 6)
    report("K normed (nv*dk)", py_k_normed.cpu().numpy(), rust_k_ri)

    # Recurrence output (step 7)
    report("Recurrence out (nv*dv)", py_recur_out.cpu().numpy(), rust_recur_out)

    # Gated RMSNorm output (step 8)
    report("Gated RMSNorm out (nv*dv)", py_gated_out.cpu().numpy(), rust_gated_out)

    # Final LA output (d_hidden after step 9, BF16)
    # Note: d_hidden after step 9 contains the LA output added to... wait
    # Actually d_hidden after the full layer contains MoE output, not LA output.
    # But debug_stop_layer=1 stops AFTER the full layer. Hmm.
    # The final d_hidden is MoE output for layer 0.
    # What I really want is d_hidden after the LA output projection (step 9).
    # That gets overwritten by MoE. Let me check what d_la_conv_out has after step 8.

    # Actually, the gated_out comparison tells us if the LA pipeline is correct
    # up to the output projection. If gated_out matches, the issue is in output projection
    # or downstream. If gated_out diverges, trace back to recurrence.

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

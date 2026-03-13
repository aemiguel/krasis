#!/usr/bin/env python3
"""Compare Python vs Rust per-layer hidden states using monkey-patched capture.

Patches model.forward() to capture hidden state after each layer, then
compares against Rust debug_capture_layers output.
"""
import sys, os, struct, torch
import numpy as np

torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState

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
    if na < 1e-12 or nb < 1e-12: return 0.0
    return dot / (na * nb)

def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print("Loading model...")
    quant = QuantConfig(
        gpu_expert_bits=4, cpu_expert_bits=4, attention="bf16",
        shared_expert="int8", dense_mlp="int8", lm_head="int8",
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

    # Save states
    saved_conv = {}
    saved_recur = {}
    for li, layer in enumerate(model.layers):
        if layer.layer_type == "linear_attention":
            saved_conv[li] = layer.attention._conv_state.clone()
            saved_recur[li] = layer.attention._recurrent_state.clone()

    # ================================================================
    # Python: capture per-layer hidden state
    # ================================================================
    print("\n=== Python M=1 forward with per-layer capture ===")
    py_captures = []

    # Monkey-patch the forward to capture hidden after each layer
    orig_is_moe = {}
    orig_moe_fwd = {}
    orig_dense_fwd = {}
    orig_forward_attn = {}

    # Simpler approach: hook into each layer's forward_attn return
    class LayerCapture:
        def __init__(self):
            self.captures = []

    cap = LayerCapture()

    # Patch: wrap the "unified path" section
    # Instead of patching, let's add a diagnostic capture. The forward
    # returns hidden after all layers. We can't easily hook into the middle.
    # Alternative: run model.forward with TIMING.diag = True and capture from logs.
    # OR: run the forward manually, layer by layer.

    # Manual layer-by-layer forward:
    tok_t = torch.tensor([first_token], dtype=torch.long, device=device)
    pos_t = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)

    with torch.inference_mode():
        hidden = model.embedding[tok_t.to(device)]  # [1, hidden_size]
        residual = None

        for ss in seq_states:
            if ss is not None:
                ss.ensure_capacity(1)

        first_k = model.cfg.first_k_dense_replace
        _stream_attn = model._stream_attn_enabled
        if _stream_attn:
            model._stream_attn_load(0, buf_idx=0)

        kv_layer_offset_map = model._kv_layer_offsets

        for abs_layer_idx in range(model.cfg.num_hidden_layers):
            layer = model.layers[abs_layer_idx]
            kv_cache = model.kv_caches[0]
            seq_state = seq_states[0]
            moe_layer_idx = abs_layer_idx - first_k if abs_layer_idx >= first_k else None
            kv_layer_offset = kv_layer_offset_map.get(abs_layer_idx, 0)

            if _stream_attn:
                buf_idx = abs_layer_idx % 2
                if model._stream_attn_loaded.get(buf_idx) != abs_layer_idx:
                    model._stream_attn_load(abs_layer_idx, buf_idx)
                next_layer = abs_layer_idx + 1
                if next_layer < model.cfg.num_hidden_layers:
                    next_buf = next_layer % 2
                    model._stream_attn_prefetch(next_layer, next_buf)

            # forward_attn
            if kv_layer_offset < 0:
                hidden, residual = layer.forward_attn(
                    hidden, residual, pos_t, None, None, -1, num_new_tokens=1)
            else:
                hidden, residual = layer.forward_attn(
                    hidden, residual, pos_t, kv_cache, seq_state,
                    kv_layer_offset, num_new_tokens=1)

            # MLP/MoE
            if layer.is_moe:
                hidden = layer._moe_forward(hidden, moe_layer_idx)
            else:
                hidden = layer._dense_mlp_forward(hidden)

            # Capture hidden after this layer (BF16)
            cap.captures.append(hidden[0].clone())

        # Advance KV seq_len
        for ss in seq_states:
            if ss is not None:
                ss.advance(1)

        # Final norm + LM head
        import flashinfer
        flashinfer.norm.fused_add_rmsnorm(
            hidden, residual, model.final_norm, model.cfg.rms_norm_eps)
        from krasis.linear_attention import _linear
        py_logits = _linear(hidden, model.lm_head_data).float()

    py_token = py_logits[0].argmax(dim=-1).item()
    print(f"  Python next token: {py_token} = {repr(tokenizer.decode([py_token]))}")
    print(f"  Captured {len(cap.captures)} layer states")

    # ================================================================
    # Rust: run full decode with capture
    # ================================================================
    print("\n=== Rust full decode with per-layer capture ===")

    model.setup_gpu_decode_store()

    # Restore LA states AFTER setup (setup calls _init_state which zeros them)
    for li in saved_conv:
        a = model.layers[li].attention
        a._conv_state = saved_conv[li].clone()
        a._recurrent_state = saved_recur[li].clone()

    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store
    store.set_debug_stop_layer(0)
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

    rust_captures = store.download_layer_captures()
    print(f"  Captured {len(rust_captures)} layer states")

    # ================================================================
    # Compare per-layer
    # ================================================================
    print("\n=== Per-Layer Comparison (hidden state after MoE) ===")
    print(f"{'Layer':>5s}  {'cos_sim':>10s}  {'py_norm':>10s}  {'rust_norm':>10s}  {'maxdiff':>10s}  {'status':>6s}")

    first_bad = None
    for i in range(min(len(cap.captures), len(rust_captures))):
        py_np = cap.captures[i].float().cpu().numpy()
        rust_np = bf16_to_np(rust_captures[i])
        cs = cos_sim(py_np, rust_np)
        diff = np.abs(py_np - rust_np)
        status = "OK" if cs > 0.999 else "WARN" if cs > 0.99 else "BAD"
        print(f"  L{i:3d}  {cs:10.6f}  {np.linalg.norm(py_np):10.4f}  {np.linalg.norm(rust_np):10.4f}  {diff.max():10.6f}  {status:>6s}")
        if cs < 0.99 and first_bad is None:
            first_bad = i
            print(f"    >>> FIRST DIVERGENCE at layer {i}")
            print(f"    Python[:8]: {py_np[:8].tolist()}")
            print(f"    Rust[:8]:   {rust_np[:8].tolist()}")

    if first_bad is None:
        print("\nAll layers match!")
    else:
        print(f"\nFirst divergence at layer {first_bad}")

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()
    print("\nDone.")


if __name__ == "__main__":
    main()

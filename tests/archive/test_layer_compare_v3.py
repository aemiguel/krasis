#!/usr/bin/env python3
"""Per-layer comparison: Python GPU forward vs Rust GPU decode.

Runs Python model.forward() for M=1 (handles streaming attention internally),
then runs Rust decode with per-layer capture enabled, and compares.

Both paths run on the same GPU with the same model weights.
"""

import sys
import os
import struct

import torch
torch.set_default_dtype(torch.bfloat16)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel, _linear
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def bf16_list_to_tensor(bf16_u16_list):
    """Convert list of BF16 u16 values to FP32 tensor."""
    import numpy as np
    arr = np.array(bf16_u16_list, dtype=np.uint16)
    buf = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return buf.float()


def compare_tensors(name, py_t, rust_t):
    """Compare two tensors and report divergence."""
    diff = (py_t - rust_t).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    py_norm = py_t.norm().item()
    rust_norm = rust_t.norm().item()

    cos_sim = torch.nn.functional.cosine_similarity(
        py_t.unsqueeze(0), rust_t.unsqueeze(0)
    ).item()

    status = "OK" if cos_sim > 0.99 else ("WARN" if cos_sim > 0.95 else "DIVERGED")
    print(f"  {name}: cos={cos_sim:.6f} max_diff={max_diff:.4f} "
          f"mean_diff={mean_diff:.6f} py_norm={py_norm:.2f} "
          f"rust_norm={rust_norm:.2f} [{status}]")

    if status == "DIVERGED":
        print(f"    PY[0:8]:   {py_t[:8].tolist()}")
        print(f"    RUST[0:8]: {rust_t[:8].tolist()}")

    return cos_sim


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

    # === PHASE 1: Python M=1 forward via model.forward() ===
    # Use model.forward() which handles streaming attention correctly
    print("\n=== Python M=1 decode via model.forward() ===")

    # Restore LA states
    for li in saved_conv:
        model.layers[li].attention._conv_state = saved_conv[li].clone()
        model.layers[li].attention._recurrent_state = saved_recur[li].clone()

    next_tensor = torch.tensor([first_token], dtype=torch.long, device=device)
    next_pos = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)

    # Inject per-layer capture into model.forward by monkeypatching
    py_hidden_per_layer = {}

    # Patch each layer's _moe_forward and _dense_mlp_forward to capture output
    original_moe_fwd = {}
    original_dense_fwd = {}

    for i, layer in enumerate(model.layers):
        if layer.is_moe:
            orig = layer._moe_forward
            original_moe_fwd[i] = orig
            def make_wrapper(idx, orig_fn):
                def wrapper(hidden, moe_layer_idx):
                    result = orig_fn(hidden, moe_layer_idx)
                    py_hidden_per_layer[idx] = result[0].float().cpu().clone()
                    return result
                return wrapper
            layer._moe_forward = make_wrapper(i, orig)
        else:
            orig = layer._dense_mlp_forward
            original_dense_fwd[i] = orig
            def make_dense_wrapper(idx, orig_fn):
                def wrapper(hidden):
                    result = orig_fn(hidden)
                    py_hidden_per_layer[idx] = result[0].float().cpu().clone()
                    return result
                return wrapper
            layer._dense_mlp_forward = make_dense_wrapper(i, orig)

    with torch.inference_mode():
        py_logits = model.forward(next_tensor, next_pos, seq_states)
        py_token = py_logits[0].argmax(dim=-1).item()
        py_top5 = py_logits[0].float().topk(5)

    # Restore original methods
    for i, orig in original_moe_fwd.items():
        model.layers[i]._moe_forward = orig
    for i, orig in original_dense_fwd.items():
        model.layers[i]._dense_mlp_forward = orig

    print(f"Python decode token: {py_token} = '{tokenizer.decode([py_token])}'")
    for i in range(5):
        tid = py_top5.indices[i].item()
        val = py_top5.values[i].item()
        print(f"  {tid} ({val:.3f}) '{tokenizer.decode([tid])}'")

    print(f"Captured {len(py_hidden_per_layer)} Python layer outputs")

    # === PHASE 2: Rust GPU decode with per-layer capture ===
    print("\n=== Rust GPU decode with per-layer capture ===")

    # Restore LA states
    for li in saved_conv:
        model.layers[li].attention._conv_state = saved_conv[li].clone()
        model.layers[li].attention._recurrent_state = saved_recur[li].clone()

    # Reset KV advance from Python path
    for ss in seq_states:
        if ss is not None:
            ss.seq_len -= 1

    # Set up Rust decode store
    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Enable per-layer capture and run full decode step
    store.set_debug_capture_layers(True)
    store.set_debug_stop_layer(0)  # run all layers
    store.py_gpu_decode_step(first_token, len(prompt_tokens))
    torch.cuda.synchronize()

    # Get Rust logits
    rust_hidden_u16 = store.download_hidden_bf16()
    rust_hidden_final = bf16_list_to_tensor(rust_hidden_u16)

    # Download per-layer captures
    rust_captures = store.download_layer_captures()
    print(f"Captured {len(rust_captures)} Rust layer outputs")

    # === PHASE 3: Run Rust full decode for token comparison ===
    # Restore states again for generate_batch
    for li in saved_conv:
        model.layers[li].attention._conv_state = saved_conv[li].clone()
        model.layers[li].attention._recurrent_state = saved_recur[li].clone()
    model._update_la_state_ptrs()

    # Re-export KV (states were consumed by py_gpu_decode_step)
    for ss in seq_states:
        if ss is not None:
            ss.seq_len -= 1  # undo advance from Python
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    rust_tokens = store.gpu_generate_batch(
        first_token=first_token,
        start_position=len(prompt_tokens),
        max_tokens=1,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        stop_ids=[],
        presence_penalty=0.0,
    )
    rust_token = rust_tokens[0] if rust_tokens else -1
    print(f"Rust decode token: {rust_token} = '{tokenizer.decode([rust_token]) if rust_token >= 0 else '?'}'")

    # === PHASE 4: Comparison ===
    print(f"\n=== Token comparison ===")
    print(f"Python: {py_token} = '{tokenizer.decode([py_token])}'")
    print(f"Rust:   {rust_token} = '{tokenizer.decode([rust_token]) if rust_token >= 0 else '?'}'")
    if py_token == rust_token:
        print("MATCH!")
    else:
        print("MISMATCH")

    print(f"\n=== Per-layer hidden state comparison ===")
    print("(After MoE/MLP output for each layer)")
    print()

    first_diverged = None
    for i in range(min(len(rust_captures), 48)):
        if i not in py_hidden_per_layer:
            print(f"  L{i:2d}: no Python capture")
            continue

        rust_h = bf16_list_to_tensor(rust_captures[i])
        py_h = py_hidden_per_layer[i]
        layer_type = model.layers[i].layer_type
        is_moe = model.layers[i].is_moe
        tag = f"{'LA' if layer_type == 'linear_attention' else 'GQA'}+{'MoE' if is_moe else 'Dense'}"

        cos = compare_tensors(f"L{i:2d} ({tag})", py_h, rust_h)

        if cos < 0.95 and first_diverged is None:
            first_diverged = i

    # Summary
    print(f"\n=== Summary ===")
    if first_diverged is not None:
        print(f"First major divergence at layer {first_diverged}")
        layer = model.layers[first_diverged]
        print(f"  Type: {layer.layer_type}, MoE: {layer.is_moe}")
        print(f"\nFocus debugging on layer {first_diverged}.")
    else:
        print("All layers within tolerance!")

    print(f"\nPython token: {py_token} = '{tokenizer.decode([py_token])}'")
    print(f"Rust token:   {rust_token} = '{tokenizer.decode([rust_token]) if rust_token >= 0 else '?'}'")

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

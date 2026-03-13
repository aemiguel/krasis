#!/usr/bin/env python3
"""Compare Python GPU forward (M=1) vs Rust GPU decode hidden states per-layer.

Uses set_debug_stop_layer() to stop the Rust decode after N layers
and download_hidden_bf16() to read back the hidden state.

Also hooks into Python forward pass to capture per-layer hidden states
for direct comparison.
"""

import sys
import os
import time
import struct

import torch
torch.set_default_dtype(torch.bfloat16)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def bf16_list_to_tensor(bf16_u16_list):
    """Convert list of u16 BF16 bit patterns to a float32 torch tensor."""
    floats = []
    for u16val in bf16_u16_list:
        bits = u16val << 16
        floats.append(struct.unpack('f', struct.pack('I', bits))[0])
    return torch.tensor(floats, dtype=torch.float32)


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("Loading model...")
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

    # Prompt
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt_tokens = tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # ── Prefill ──
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

    print(f"First token: {first_token} = '{tokenizer.decode([first_token])}'")

    # Save post-prefill states
    saved_conv = {}
    saved_recur = {}
    saved_kv_states = []  # For KV cache state restoration
    for layer_idx, layer in enumerate(model.layers):
        if layer.layer_type == "linear_attention":
            saved_conv[layer_idx] = layer.attention._conv_state.clone()
            saved_recur[layer_idx] = layer.attention._recurrent_state.clone()

    # ── Python M=1 forward ──
    print("\n=== Python M=1 forward ===")
    next_tensor = torch.tensor([first_token], dtype=torch.long, device=device)
    next_pos = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)

    with torch.inference_mode():
        python_logits = model.forward(next_tensor, next_pos, seq_states)

    python_token = python_logits[0].argmax(dim=-1).item()
    print(f"Python next token: {python_token} = '{tokenizer.decode([python_token])}'")
    python_top5 = python_logits[0].topk(5)
    for i in range(5):
        tid = python_top5.indices[i].item()
        val = python_top5.values[i].item()
        print(f"  {tid}({val:.3f}) '{tokenizer.decode([tid])}'")

    # ── Set up Rust ──
    print("\n=== Setting up Rust GPU decode ===")

    # Restore LA states
    for layer_idx in saved_conv:
        attn = model.layers[layer_idx].attention
        attn._conv_state = saved_conv[layer_idx].clone()
        attn._recurrent_state = saved_recur[layer_idx].clone()

    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    # Export KV cache from FlashInfer to Rust (critical for GQA attention!)
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Full Rust decode
    print("\n=== Rust full decode ===")
    store.set_debug_stop_layer(0)

    # Restore states again (setup may have modified them)
    for layer_idx in saved_conv:
        attn = model.layers[layer_idx].attention
        attn._conv_state = saved_conv[layer_idx].clone()
        attn._recurrent_state = saved_recur[layer_idx].clone()
    model._update_la_state_ptrs()

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
    print(f"Rust next token: {rust_token} = '{tokenizer.decode([rust_token]) if rust_token >= 0 else '?'}'")

    if python_token == rust_token:
        print("MATCH!")
    else:
        print(f"MISMATCH: Python={python_token} Rust={rust_token}")

    # ── Per-layer hidden state norms ──
    print("\n=== Rust hidden state after each layer ===")
    test_layers = list(range(1, 49))  # All 48 layers

    for n_layers in test_layers:
        # Restore states
        for layer_idx in saved_conv:
            attn = model.layers[layer_idx].attention
            attn._conv_state = saved_conv[layer_idx].clone()
            attn._recurrent_state = saved_recur[layer_idx].clone()
        model._update_la_state_ptrs()

        store.set_debug_stop_layer(n_layers)
        store.py_gpu_decode_step(first_token, len(prompt_tokens))
        torch.cuda.synchronize()

        rust_hidden_u16 = store.download_hidden_bf16()
        rust_hidden = bf16_list_to_tensor(rust_hidden_u16)
        rust_norm = rust_hidden.norm().item()
        rust_max = rust_hidden.abs().max().item()
        rust_mean = rust_hidden.mean().item()

        # Check for NaN
        has_nan = torch.isnan(rust_hidden).any().item()
        nan_str = " NAN!" if has_nan else ""

        print(f"  L{n_layers:2d}: norm={rust_norm:10.4f}  max={rust_max:8.4f}  mean={rust_mean:+.6f}{nan_str}")

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

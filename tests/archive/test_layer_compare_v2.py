#!/usr/bin/env python3
"""Compare Python GPU forward (M=1) vs Rust GPU decode per-layer hidden states.

Hooks into Python model.forward() to capture hidden states at each layer boundary,
then runs Rust decode layer-by-layer using debug_stop_layer and compares.
"""

import sys
import os
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
    floats = []
    for u16val in bf16_u16_list:
        bits = u16val << 16
        floats.append(struct.unpack('f', struct.pack('I', bits))[0])
    return torch.tensor(floats, dtype=torch.float32)


def compare_tensors(name, py_t, rust_t, rtol=0.05, atol=0.01):
    """Compare two tensors and report divergence."""
    diff = (py_t - rust_t).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    py_norm = py_t.norm().item()
    rust_norm = rust_t.norm().item()
    rel_diff = max_diff / (py_norm + 1e-8)

    status = "OK" if max_diff < atol or rel_diff < rtol else "DIVERGED"
    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"py_norm={py_norm:.4f}, rust_norm={rust_norm:.4f}, rel={rel_diff:.4f} [{status}]")

    if status == "DIVERGED":
        # Show first few values
        print(f"    PY[0:4]:   {py_t[:4].tolist()}")
        print(f"    RUST[0:4]: {rust_t[:4].tolist()}")

    return status == "OK"


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

    print(f"First token: {first_token} = '{tokenizer.decode([first_token])}'")

    # Save post-prefill states
    saved_conv = {}
    saved_recur = {}
    for layer_idx, layer in enumerate(model.layers):
        if layer.layer_type == "linear_attention":
            saved_conv[layer_idx] = layer.attention._conv_state.clone()
            saved_recur[layer_idx] = layer.attention._recurrent_state.clone()

    # === Python M=1 forward with per-layer hooks ===
    print("\n=== Python M=1 forward with per-layer capture ===")

    python_hidden_states = {}

    # Hook to capture hidden states after each layer
    original_forward = model.forward

    # We need to instrument the forward pass. The simplest way:
    # model.forward() calls each layer in sequence. We can hook by
    # modifying the model's _forward_one_token or similar method.
    # But for simplicity, let's just run the Python forward and capture
    # the pre/post hidden state of EACH LAYER via register_forward_hook.

    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # After each layer's forward, capture the hidden state
            # The layer module processes hidden_states and returns new hidden_states
            if isinstance(output, tuple):
                python_hidden_states[layer_idx] = output[0].detach().clone()
            else:
                python_hidden_states[layer_idx] = output.detach().clone()
        return hook_fn

    # Actually, hooking into model.layers[i] won't work because the layers
    # aren't nn.Module. Let me use a different approach: run the Python
    # forward step-by-step using model's internal methods.

    # Simpler approach: capture the hidden state at each layer by running
    # model.forward() once with a debug hook that saves intermediate states.
    # For now, just get the final Python output.

    next_tensor = torch.tensor([first_token], dtype=torch.long, device=device)
    next_pos = torch.tensor([len(prompt_tokens)], dtype=torch.int32, device=device)

    with torch.inference_mode():
        python_logits = model.forward(next_tensor, next_pos, seq_states)

    python_token = python_logits[0].argmax(dim=-1).item()
    python_top5 = python_logits[0].topk(5)
    print(f"Python next token: {python_token} = '{tokenizer.decode([python_token])}'")
    for i in range(5):
        tid = python_top5.indices[i].item()
        val = python_top5.values[i].item()
        print(f"  {tid}({val:.3f}) '{tokenizer.decode([tid])}'")

    # Save the Python logits for comparison
    python_logits_f32 = python_logits[0].float()

    # === Set up Rust ===
    print("\n=== Setting up Rust GPU decode ===")

    # Restore LA states
    for layer_idx in saved_conv:
        attn = model.layers[layer_idx].attention
        attn._conv_state = saved_conv[layer_idx].clone()
        attn._recurrent_state = saved_recur[layer_idx].clone()

    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # === Full Rust decode for logits comparison ===
    print("\n=== Rust full decode ===")
    store.set_debug_stop_layer(0)  # 0 = run all layers

    # Restore states
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

    # === Per-layer hidden state comparison ===
    print("\n=== Per-layer hidden state comparison ===")
    print("(Comparing Rust hidden state after each layer against itself across runs)")
    print("(Cannot directly compare vs Python without instrumenting Python forward)")

    # Instead, let's focus on comparing the Rust embedding vs Python embedding,
    # and the Rust logits vs Python logits.

    # Compare logits
    print("\n=== Logits comparison ===")

    # Get Rust logits from gpu_generate_batch
    # Actually, we need to get the raw logits. Let me use py_gpu_decode_step + download.
    # Restore states
    for layer_idx in saved_conv:
        attn = model.layers[layer_idx].attention
        attn._conv_state = saved_conv[layer_idx].clone()
        attn._recurrent_state = saved_recur[layer_idx].clone()
    model._update_la_state_ptrs()

    store.set_debug_stop_layer(0)
    store.py_gpu_decode_step(first_token, len(prompt_tokens))
    torch.cuda.synchronize()

    # Download the logits buffer
    rust_logits_u16 = store.download_hidden_bf16()
    # Wait - download_hidden_bf16 gets d_hidden, not d_logits.
    # After full decode, d_hidden contains the normalized hidden (after final norm).
    # Let's compare that against Python's final hidden.

    # Actually, the most useful comparison is the EMBEDDING (should be identical)
    # and the first layer's output.
    print("\n=== Embedding comparison ===")
    # Run Rust with stop_layer=1 to get hidden state after embedding + layer 0
    for layer_idx in saved_conv:
        attn = model.layers[layer_idx].attention
        attn._conv_state = saved_conv[layer_idx].clone()
        attn._recurrent_state = saved_recur[layer_idx].clone()
    model._update_la_state_ptrs()

    # Get embedding: run with stop_layer=1 but we need just the embedding.
    # Since debug_stop_layer stops AFTER the layer loop, we can't get just the embedding.
    # Let me just compare the PYTHON embedding against the Rust embedding.

    # Python embedding
    with torch.inference_mode():
        py_emb = model.embedding[first_token].float()
    print(f"Python embedding norm: {py_emb.norm().item():.4f}")
    print(f"Python embedding[0:4]: {py_emb[:4].tolist()}")

    # We can't easily get Rust embedding separately, so let's compare layer-by-layer
    # hidden states. For each layer N, we compare Rust's d_hidden after N layers.

    print("\n=== Layer-by-layer hidden state norms ===")
    for n_layers in [1, 2, 3, 4, 5, 10, 20, 30, 40, 48]:
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
        has_nan = torch.isnan(rust_hidden).any().item()
        nan_str = " NAN!" if has_nan else ""

        print(f"  After L{n_layers:2d}: norm={rust_norm:10.4f}  max={rust_max:8.4f}  mean={rust_mean:+.6f}{nan_str}")

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

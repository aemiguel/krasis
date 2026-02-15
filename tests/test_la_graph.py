#!/usr/bin/env python3
"""Test CUDA graph for linear attention produces correct output."""

import logging, sys
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"

print("Loading model...")
model = KrasisModel(
    MODEL,
    pp_partition=[24, 24],
    devices=["cuda:0", "cuda:1"],
    kv_dtype=torch.float8_e4m3fn,
    krasis_threads=48,
    gpu_prefill=True,
    gpu_prefill_threshold=300,
    expert_divisor=-1,
)
model.load()

prompt = model.tokenizer.apply_chat_template([{"role": "user", "content": "Write a short poem about the ocean."}])
print(f"Prompt: {len(prompt)} tokens")

def generate(use_graph, num_tokens=20):
    """Generate with or without CUDA graph."""
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()

    seq_states = [
        SequenceKVState(c, seq_id=0) if c is not None else None
        for c in model.kv_caches
    ]
    device = torch.device(model.ranks[0].device)
    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt), dtype=torch.int32, device=device)

    # Disable or enable graph in forward
    if not use_graph:
        # Disable graph by setting _la_input to None (will stay on fallback path)
        for layer in model.layers:
            if layer.layer_type == "linear_attention":
                layer.attention._la_input = None  # disable graph capture

    with torch.inference_mode():
        logits = model.forward(prompt_tensor, positions, seq_states)

    if not use_graph:
        # Keep graph disabled for decode too
        for layer in model.layers:
            if layer.layer_type == "linear_attention":
                layer.attention._la_graph = None
                layer.attention._la_input = None

    next_token = sample(logits[-1:, :], 0.0, 1, 1.0).item()  # greedy
    tokens = [next_token]

    with torch.inference_mode():
        for step in range(num_tokens - 1):
            pos = len(prompt) + step
            token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
            pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

            if not use_graph:
                for layer in model.layers:
                    if layer.layer_type == "linear_attention":
                        layer.attention._la_graph = None
                        layer.attention._la_input = None

            logits = model.forward(token_tensor, pos_tensor, seq_states)
            next_token = sample(logits, 0.0, 1, 1.0).item()
            tokens.append(next_token)

    for s in seq_states:
        if s is not None:
            s.free()
    return tokens

print("\n=== Without CUDA graph ===")
tokens_no_graph = generate(False)
text_no_graph = model.tokenizer.decode(tokens_no_graph)
print(f"Tokens: {tokens_no_graph[:10]}...")
print(f"Text: {repr(text_no_graph[:200])}")

print("\n=== With CUDA graph ===")
tokens_graph = generate(True)
text_graph = model.tokenizer.decode(tokens_graph)
print(f"Tokens: {tokens_graph[:10]}...")
print(f"Text: {repr(text_graph[:200])}")

print(f"\nMatch: {tokens_no_graph == tokens_graph}")
if tokens_no_graph != tokens_graph:
    for i, (a, b) in enumerate(zip(tokens_no_graph, tokens_graph)):
        if a != b:
            print(f"  First mismatch at token {i}: no_graph={a}, graph={b}")
            break

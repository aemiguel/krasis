#!/usr/bin/env python3
"""Quick test: compare _forward_recurrent vs _forward_recurrent_inplace."""

import logging, sys, time
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

def generate(use_inplace, num_tokens=20):
    """Generate with original or inplace recurrent forward."""
    # Reset
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()
            # Disable CUDA graph
            layer.attention._la_graph = None
            layer.attention._la_input = None
            layer.attention._la_output = None

    seq_states = [
        SequenceKVState(c, seq_id=0) if c is not None else None
        for c in model.kv_caches
    ]
    device = torch.device(model.ranks[0].device)
    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt), dtype=torch.int32, device=device)

    with torch.inference_mode():
        logits = model.forward(prompt_tensor, positions, seq_states)

    # Monkey-patch: force inplace or original
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention._use_inplace_test = use_inplace

    next_token = sample(logits[-1:, :], 0.0, 1, 1.0).item()  # greedy
    tokens = [next_token]

    with torch.inference_mode():
        for step in range(num_tokens - 1):
            pos = len(prompt) + step
            token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
            pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
            logits = model.forward(token_tensor, pos_tensor, seq_states)
            next_token = sample(logits, 0.0, 1, 1.0).item()
            tokens.append(next_token)

    for s in seq_states:
        if s is not None:
            s.free()

    return tokens

# Monkey-patch forward to use inplace or original based on flag
import krasis.linear_attention as la
original_forward = la.GatedDeltaNetAttention.forward

def patched_forward(self, hidden, is_decode):
    self._init_state()
    if is_decode and getattr(self, '_use_inplace_test', False):
        return self._forward_recurrent_inplace(hidden)
    elif is_decode:
        return self._forward_recurrent(hidden)
    else:
        return self._forward_chunked(hidden)

la.GatedDeltaNetAttention.forward = patched_forward

print("\n=== Original forward ===")
tokens_orig = generate(False)
text_orig = model.tokenizer.decode(tokens_orig)
print(f"Tokens: {tokens_orig[:10]}...")
print(f"Text: {repr(text_orig[:200])}")

print("\n=== Inplace forward ===")
tokens_inplace = generate(True)
text_inplace = model.tokenizer.decode(tokens_inplace)
print(f"Tokens: {tokens_inplace[:10]}...")
print(f"Text: {repr(text_inplace[:200])}")

print(f"\nMatch: {tokens_orig == tokens_inplace}")
if tokens_orig != tokens_inplace:
    for i, (a, b) in enumerate(zip(tokens_orig, tokens_inplace)):
        if a != b:
            print(f"  First mismatch at token {i}: orig={a}, inplace={b}")
            break

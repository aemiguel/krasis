#!/usr/bin/env python3
"""Compare Krasis GQA attention output vs HF reference for Qwen3-Coder-Next."""

import logging, sys
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from krasis.config import ModelConfig, QuantConfig
from krasis.weight_loader import WeightLoader

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"

def main():
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )

    # Layer 3 is first GQA (full attention) layer
    hf_layer3 = hf_model.model.layers[3]
    hf_attn = hf_layer3.self_attn
    print(f"HF layer 3 attention: {type(hf_attn).__name__}")
    print(f"  num_heads={hf_attn.config.num_attention_heads}")
    print(f"  num_kv_heads={hf_attn.config.num_key_value_heads}")
    print(f"  head_dim={hf_attn.head_dim}")
    print(f"  q_proj shape={hf_attn.q_proj.weight.shape}")
    print(f"  k_proj shape={hf_attn.k_proj.weight.shape}")
    print(f"  v_proj shape={hf_attn.v_proj.weight.shape}")
    print(f"  o_proj shape={hf_attn.o_proj.weight.shape}")

    # Load our weights
    cfg = ModelConfig.from_model_path(MODEL)
    loader = WeightLoader(cfg, QuantConfig(attention="bf16"))
    weights = loader.load_layer(3, torch.device("cuda:0"))
    loader.close()

    print(f"\nOur GQA config:")
    print(f"  num_heads={cfg.num_attention_heads}")
    print(f"  num_kv_heads={cfg.num_key_value_heads}")
    print(f"  head_dim={cfg.gqa_head_dim}")
    print(f"  partial_rotary_factor={cfg.partial_rotary_factor}")
    print(f"  rotary_dim={cfg.rotary_dim}")

    # Compare raw weight dimensions
    our_q = weights["attention"]["q_proj"]
    our_k = weights["attention"]["k_proj"]
    our_v = weights["attention"]["v_proj"]
    our_o = weights["attention"]["o_proj"]
    print(f"  our q_proj shape={our_q.shape}")
    print(f"  our k_proj shape={our_k.shape}")
    print(f"  our v_proj shape={our_v.shape}")
    print(f"  our o_proj shape={our_o.shape}")

    # Compare weight values
    d = (hf_attn.q_proj.weight.float() - our_q.cpu().float()).abs()
    print(f"  q_proj diff: max={d.max():.6f}")
    d = (hf_attn.k_proj.weight.float() - our_k.cpu().float()).abs()
    print(f"  k_proj diff: max={d.max():.6f}")

    # Check q_norm, k_norm
    hf_q_norm = hf_attn.q_norm.weight
    hf_k_norm = hf_attn.k_norm.weight
    our_q_norm = weights["attention"].get("q_norm")
    our_k_norm = weights["attention"].get("k_norm")
    print(f"\n  hf q_norm shape={hf_q_norm.shape}")
    print(f"  our q_norm: {'exists' if our_q_norm is not None else 'MISSING!'}")
    if our_q_norm is not None:
        d = (hf_q_norm.float() - our_q_norm.cpu().float()).abs()
        print(f"  q_norm diff: max={d.max():.6f}")
    if our_k_norm is not None:
        d = (hf_k_norm.float() - our_k_norm.cpu().float()).abs()
        print(f"  k_norm diff: max={d.max():.6f}")

    # Test forward: compare projections
    M = 5
    hidden = torch.randn(M, cfg.hidden_size, dtype=torch.bfloat16)
    hidden_3d = hidden.unsqueeze(0)  # [1, M, hidden] for HF

    # HF forward step by step
    with torch.no_grad():
        # HF q_proj + chunk
        hf_q_raw = hf_attn.q_proj(hidden_3d)  # [1, M, num_heads * head_dim * 2]
        print(f"\nHF q_proj output shape: {hf_q_raw.shape}")
        hf_q_split = hf_q_raw.view(1, M, -1, hf_attn.head_dim * 2)
        hf_q_states, hf_gate = torch.chunk(hf_q_split, 2, dim=-1)
        print(f"HF query shape after chunk: {hf_q_states.shape}")
        print(f"HF gate shape: {hf_gate.shape}")

        # Apply q_norm
        hf_q_normed = hf_attn.q_norm(hf_q_states)
        print(f"HF q after norm: mean={hf_q_normed.float().mean():.6f}, std={hf_q_normed.float().std():.6f}")

        # Our q_proj
        our_q_raw = F.linear(hidden, our_q.cpu())  # [M, num_heads * head_dim * 2]
        print(f"\nOur q_proj output shape: {our_q_raw.shape}")
        our_q_view = our_q_raw.view(M, cfg.num_attention_heads, cfg.gqa_head_dim * 2)
        our_q_split, our_gate = our_q_view.chunk(2, dim=-1)
        print(f"Our query shape after chunk: {our_q_split.shape}")

        # Compare before norm
        d = (hf_q_states.squeeze(0).float() - our_q_split.float()).abs()
        print(f"Q before norm diff: max={d.max():.6f}")

        # Compare gate
        hf_gate_flat = hf_gate.reshape(1, M, -1).squeeze(0)
        our_gate_flat = our_gate.reshape(M, -1)
        d = (hf_gate_flat.float() - our_gate_flat.float()).abs()
        print(f"Gate diff: max={d.max():.6f}")

    # Test RoPE
    print("\n--- RoPE comparison ---")
    print(f"  HF partial_rotary_factor: {hf_attn.config.partial_rotary_factor}")
    print(f"  Our rotary_dim: {cfg.rotary_dim} of {cfg.gqa_head_dim}")

    # Check if HF RoPE uses partial
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
    # HF model uses rotary_emb in the parent decoder layer
    hf_rotary = hf_model.model.rotary_emb
    print(f"  HF rotary type: {type(hf_rotary).__name__}")
    print(f"  HF rotary dim: {hf_rotary.dim}")

    print("\nDone.")

if __name__ == "__main__":
    main()

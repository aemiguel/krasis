#!/usr/bin/env python3
"""Compare Krasis linear attention output vs HF reference, layer by layer."""

import logging, sys
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from krasis.config import ModelConfig
from krasis.weight_loader import WeightLoader, QuantConfig
from krasis.linear_attention import GatedDeltaNetAttention

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"

def main():
    print("Loading HF model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )

    # Get the first linear attention layer (layer 0)
    hf_layer0 = hf_model.model.layers[0]
    hf_attn = hf_layer0.linear_attn

    print(f"HF layer 0 type: {type(hf_attn).__name__}")
    print(f"  num_k_heads={hf_attn.num_k_heads}, num_v_heads={hf_attn.num_v_heads}")
    print(f"  head_k_dim={hf_attn.head_k_dim}, head_v_dim={hf_attn.head_v_dim}")
    print(f"  conv_dim={hf_attn.conv_dim}")

    # Load our weights on GPU then move to CPU for comparison
    cfg = ModelConfig.from_model_path(MODEL)
    loader = WeightLoader(cfg, QuantConfig(attention="bf16"))  # BF16 to match HF
    weights = loader.load_layer(0, torch.device("cuda:0"))

    # Move weights to CPU for comparison
    def to_cpu(d):
        if isinstance(d, dict):
            return {k: to_cpu(v) for k, v in d.items()}
        elif isinstance(d, torch.Tensor):
            return d.cpu()
        return d
    weights_cpu = to_cpu(weights)

    our_attn = GatedDeltaNetAttention(cfg, 0, weights_cpu["linear_attention"], torch.device("cpu"))

    # Create test input
    hidden = torch.randn(1, 5, cfg.hidden_size, dtype=torch.bfloat16)  # [B, M, hidden]
    hidden_2d = hidden.squeeze(0)  # [M, hidden] for our code

    # ---- Run HF ----
    print("\n--- HF forward ---")
    # HF expects [B, M, hidden]
    with torch.no_grad():
        hf_out = hf_attn(hidden, cache_params=None, cache_position=None, attention_mask=None)
    print(f"HF output: shape={hf_out.shape}, mean={hf_out.float().mean():.6f}, std={hf_out.float().std():.6f}")

    # ---- Run ours ----
    print("\n--- Krasis forward (recurrent) ---")
    our_attn.reset_state()
    with torch.no_grad():
        our_out = our_attn.forward(hidden_2d, is_decode=False)  # currently forced to recurrent
    print(f"Our output: shape={our_out.shape}, mean={our_out.float().mean():.6f}, std={our_out.float().std():.6f}")

    # Compare
    hf_out_2d = hf_out.squeeze(0)  # [M, hidden]
    diff = (hf_out_2d.float() - our_out.float()).abs()
    cos_sim = F.cosine_similarity(hf_out_2d.float().flatten(), our_out.float().flatten(), dim=0)
    print(f"\nDiff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    # Check intermediate values: un-interleaving
    print("\n--- Checking un-interleaving ---")
    with torch.no_grad():
        # HF
        hf_qkvz = hf_attn.in_proj_qkvz(hidden)
        hf_ba = hf_attn.in_proj_ba(hidden)
        hf_q, hf_k, hf_v, hf_z, hf_b, hf_a = hf_attn.fix_query_key_value_ordering(hf_qkvz, hf_ba)

        # Ours
        our_qkvz_raw = torch.nn.functional.linear(hidden_2d, our_attn.in_proj_qkvz)
        our_ba_raw = torch.nn.functional.linear(hidden_2d, our_attn.in_proj_ba)
        our_q, our_k, our_v, our_z, our_b, our_a = our_attn._fix_query_key_value_ordering(our_qkvz_raw, our_ba_raw)

    # HF shapes are [B, M, nh, hd], ours are [M, nh, hd]
    def compare(name, hf_t, our_t):
        hf_2d = hf_t.squeeze(0) if hf_t.dim() > our_t.dim() else hf_t
        if hf_2d.shape != our_t.shape:
            print(f"  {name}: SHAPE MISMATCH hf={hf_2d.shape} vs ours={our_t.shape}")
            return
        d = (hf_2d.float() - our_t.float()).abs()
        cs = F.cosine_similarity(hf_2d.float().flatten(), our_t.float().flatten(), dim=0)
        print(f"  {name}: max_diff={d.max():.6f}, cos_sim={cs:.6f}")

    compare("q", hf_q, our_q)
    compare("k", hf_k, our_k)
    compare("v", hf_v, our_v)
    compare("z", hf_z, our_z)
    compare("b", hf_b, our_b)
    compare("a", hf_a, our_a)

    # Check conv1d weights match
    print("\n--- Conv1d weight check ---")
    hf_conv_w = hf_attn.conv1d.weight  # [conv_dim, 1, kernel_size]
    our_conv_w = our_attn.conv1d_weight  # [conv_dim, 1, kernel_size]
    d = (hf_conv_w.float() - our_conv_w.float()).abs()
    print(f"  conv1d weight diff: max={d.max():.6f}")

    # Check out_proj weights
    hf_out_w = hf_attn.out_proj.weight
    our_out_w = our_attn.out_proj
    d = (hf_out_w.float() - our_out_w.float()).abs()
    print(f"  out_proj weight diff: max={d.max():.6f}")

    # Check A_log, dt_bias, norm
    d = (hf_attn.A_log.float() - our_attn.A_log.float()).abs()
    print(f"  A_log diff: max={d.max():.6f}")
    d = (hf_attn.dt_bias.float() - our_attn.dt_bias.float()).abs()
    print(f"  dt_bias diff: max={d.max():.6f}")
    d = (hf_attn.norm.weight.float() - our_attn.norm_weight.float()).abs()
    print(f"  norm.weight diff: max={d.max():.6f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

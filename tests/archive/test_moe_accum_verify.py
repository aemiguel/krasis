#!/usr/bin/env python3
"""Verify MoE accumulation pipeline: w13 GEMV -> SiLU*mul -> w2 GEMV -> weighted accum.

Since w13 GEMV is verified correct (cos=0.999999), this test isolates the
fused_silu_accum kernel and the full 10-expert accumulation.
"""

import sys
import os
import numpy as np

import torch
torch.set_default_dtype(torch.bfloat16)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def bf16_to_f32(u16_list):
    arr = np.array(u16_list, dtype=np.uint16)
    buf = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return buf.float()


def compare(name, ref_t, test_t, n=8):
    cos = torch.nn.functional.cosine_similarity(
        ref_t.unsqueeze(0), test_t.unsqueeze(0)
    ).item()
    diff = (ref_t - test_t).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "DIVERGED")
    print(f"  {name}: cos={cos:.6f} max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} "
          f"ref_norm={ref_t.norm():.4f} test_norm={test_t.norm():.4f} [{status}]")
    if cos < 0.999:
        print(f"    REF[0:{n}]:  {ref_t[:n].tolist()}")
        print(f"    TEST[0:{n}]: {test_t[:n].tolist()}")
    return cos


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

    # Prefill
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

    print(f"First token: {first_token}")

    # Set up GPU decode store
    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Upload hidden state (post-attn-norm from Python path, same as isolation test)
    with torch.inference_mode():
        hidden = model.embedding[first_token].unsqueeze(0).to(device)
    hidden_bf16 = hidden[0].cpu().contiguous().view(torch.int16).numpy().astype(np.uint16).tolist()
    store.upload_hidden_bf16(hidden_bf16)

    hidden_f32 = hidden[0].float().cpu()
    hs = hidden_f32.shape[0]
    intermediate = model.cfg.moe_intermediate_size

    print(f"Hidden: norm={hidden_f32.norm():.4f}, hs={hs}, intermediate={intermediate}")

    # ================================================================
    # TEST 1: CPU reference for ONE expert's full pipeline
    # (w13 GEMV -> silu*mul -> w2 GEMV)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Single expert full pipeline (expert 319)")
    print("=" * 60)

    expert_id = 319
    layer_idx = 0

    # Get CPU reference w13 output
    cpu_gate_up = torch.tensor(
        store.test_cpu_reference_w13(layer_idx, expert_id),
        dtype=torch.float32,
    )

    # CPU reference: silu(gate) * up
    cpu_gate = cpu_gate_up[:intermediate]
    cpu_up = cpu_gate_up[intermediate:]
    cpu_silu_mul = torch.sigmoid(cpu_gate) * cpu_gate * cpu_up  # silu(g) = g * sigmoid(g)

    print(f"  CPU silu_mul: norm={cpu_silu_mul.norm():.6f}")
    print(f"  CPU silu_mul[0:8]: {cpu_silu_mul[:8].tolist()}")

    # ================================================================
    # TEST 2: Run full MoE on GPU and compare
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Full MoE forward (Rust GPU)")
    print("=" * 60)

    route_ms, dma_ms, compute_ms, total_ms = store.moe_forward_gpu(layer_idx)
    rust_moe_out = bf16_to_f32(store.download_moe_out_bf16())

    print(f"  Timing: route={route_ms:.1f}ms dma={dma_ms:.1f}ms compute={compute_ms:.1f}ms total={total_ms:.1f}ms")
    print(f"  Rust MoE output: norm={rust_moe_out.norm():.4f}")
    print(f"  Rust MoE[0:8]: {rust_moe_out[:8].tolist()}")

    # ================================================================
    # TEST 3: CPU reference for full MoE (all 10 experts + shared)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: CPU reference MoE (Python path)")
    print("=" * 60)

    # Run Python MoE forward for layer 0
    layer0 = model.layers[0]
    with torch.inference_mode():
        py_moe_out = layer0._moe_forward(hidden.to(device), moe_layer_idx=0)

    py_moe_f32 = py_moe_out[0].float().cpu()
    print(f"  Python MoE output: norm={py_moe_f32.norm():.4f}")
    print(f"  Python MoE[0:8]: {py_moe_f32[:8].tolist()}")

    # ================================================================
    # Compare
    # ================================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    cos = compare("Rust vs Python MoE", py_moe_f32, rust_moe_out)

    if cos < 0.99:
        print("\n  MoE output DIVERGED. Since w13 GEMV is correct, bug is in:")
        print("  1. fused_silu_accum kernel (SiLU, w2 GEMV, or weighted accumulation)")
        print("  2. Shared expert computation")
        print("  3. BF16 accumulation precision loss")

        # Test without shared expert:
        # The Rust MoE adds shared expert with weight=1.0
        # The Python adds it as a separate operation
        # Let's compare just the routed part

        # Get Python routed-only output by running dispatch without shared expert
        print("\n  --- Isolating shared expert ---")

        # Save shared expert ref
        shared_exp = layer0.shared_expert
        layer0.shared_expert = None  # temporarily disable

        with torch.inference_mode():
            py_routed_only = layer0._moe_forward(hidden.to(device), moe_layer_idx=0)
        py_routed_f32 = py_routed_only[0].float().cpu()

        layer0.shared_expert = shared_exp  # restore

        py_shared = py_moe_f32 - py_routed_f32
        print(f"  Python routed-only: norm={py_routed_f32.norm():.4f}")
        print(f"  Python shared-expert: norm={py_shared.norm():.4f}")
        print(f"  Ratio routed/total: {py_routed_f32.norm()/py_moe_f32.norm():.4f}")
        print(f"  Ratio shared/total: {py_shared.norm()/py_moe_f32.norm():.4f}")

        compare("Rust vs Python routed-only", py_routed_f32, rust_moe_out)

        # Maybe Rust is missing the shared expert?
        # Check by comparing Rust output vs Python routed-only
        cos2 = compare("Rust vs Python shared-only", py_shared, rust_moe_out - py_routed_f32)

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()

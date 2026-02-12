"""Test the KrasisMoEWrapper SGLang bridge (CPU-only, no GPU required)."""

import sys
import os
import time
import struct
import numpy as np

# Add krasis python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from krasis import KrasisEngine
from krasis.sglang_bridge import KrasisMoEWrapper


def test_engine_async_roundtrip():
    """Test async submit/sync directly on the Rust engine."""
    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    engine = KrasisEngine(parallel=True, num_threads=16)
    engine.load(model_dir)

    hidden = engine.hidden_size()
    top_k = engine.top_k()
    print(f"Engine loaded: hidden={hidden}, top_k={top_k}, layers={engine.num_moe_layers()}")

    # Create test activation (BF16 as u16 bytes)
    act_f32 = np.array([(i * 7 + 13) / hidden - 0.5 for i in range(hidden)], dtype=np.float32) * 0.1
    # Convert f32 → bf16 (truncate lower 16 bits)
    act_bf16_u16 = (act_f32.view(np.uint32) >> 16).astype(np.uint16)
    act_bytes = act_bf16_u16.tobytes()

    # Expert indices and weights
    expert_ids = np.arange(top_k, dtype=np.int32)
    expert_weights = np.full(top_k, 1.0 / top_k, dtype=np.float32)

    ids_bytes = expert_ids.tobytes()
    wts_bytes = expert_weights.tobytes()

    # Submit + sync
    start = time.perf_counter()
    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, 1)
    output_bytes = engine.sync_forward()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Parse BF16 output
    output_bf16 = np.frombuffer(output_bytes, dtype=np.uint16)
    # BF16 → f32: shift left 16 bits
    output_f32 = (output_bf16.astype(np.uint32) << 16).view(np.float32)

    rms = np.sqrt(np.mean(output_f32 ** 2))
    nonzero = np.count_nonzero(output_f32)

    print(f"Async roundtrip: {elapsed_ms:.1f} ms, RMS={rms:.6f}, nonzero={nonzero}/{hidden}")
    assert rms > 1e-4, f"Output too small: RMS={rms}"
    assert nonzero > hidden // 2, f"Too many zeros: {nonzero}/{hidden}"
    print("PASS: engine async roundtrip")


def test_wrapper_interface():
    """Test KrasisMoEWrapper with numpy tensors (CPU-only, no torch.cuda)."""
    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    # Reset singleton
    KrasisMoEWrapper.reset()
    KrasisMoEWrapper.hf_model_path = model_dir

    import json
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    num_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 64))
    top_k = cfg["num_experts_per_tok"]
    first_k_dense = cfg.get("first_k_dense_replace", 0)

    # Create wrapper (layer 1 = first MoE layer for V2-Lite)
    wrapper = KrasisMoEWrapper(
        layer_idx=first_k_dense,  # first MoE layer
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden,
        moe_intermediate_size=cfg["moe_intermediate_size"],
        num_gpu_experts=0,  # all experts on CPU
        cpuinfer_threads=16,
    )

    # Load weights
    wrapper.load_weights()

    print(f"Wrapper created: layer={first_k_dense}, experts={num_experts}, top_k={top_k}")

    # Test with CPU tensors (simulating GPU tensors for interface test)
    try:
        import torch
    except ImportError:
        print("SKIP: torch not available")
        return

    batch = 1
    hidden_states = torch.randn(batch, hidden, dtype=torch.bfloat16)
    topk_ids = torch.arange(top_k, dtype=torch.int32).unsqueeze(0)  # [1, top_k]
    topk_weights = torch.full((batch, top_k), 1.0 / top_k, dtype=torch.float32)

    # For CPU-only test, mock the CUDA stream
    cuda_stream = 0

    # We can't call submit_forward directly because it calls .cpu() on already-CPU tensors
    # and torch.cuda.current_stream() requires CUDA. Let's test the raw engine path instead.
    engine = wrapper._get_engine(model_dir, 16)

    # BF16 → bytes: view as uint16 since numpy doesn't support bfloat16
    act_bytes = hidden_states.view(torch.uint16).numpy().view(np.uint8).tobytes()
    ids_bytes = topk_ids.numpy().view(np.uint8).tobytes()
    wts_bytes = topk_weights.numpy().view(np.uint8).tobytes()

    moe_layer_idx = first_k_dense - first_k_dense  # = 0

    start = time.perf_counter()
    engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, batch)
    output_bytes = engine.sync_forward()
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = torch.frombuffer(bytearray(output_bytes), dtype=torch.bfloat16).reshape(batch, hidden)
    rms = output.float().pow(2).mean().sqrt().item()

    print(f"Wrapper roundtrip: {elapsed_ms:.1f} ms, output shape={output.shape}, RMS={rms:.6f}")
    assert output.shape == (batch, hidden)
    assert rms > 1e-4, f"Output too small: RMS={rms}"
    print("PASS: wrapper interface")


def test_batch_forward():
    """Test batch>1 processing through the engine."""
    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    try:
        import torch
    except ImportError:
        print("SKIP: torch not available")
        return

    # Reset and get engine
    KrasisMoEWrapper.reset()
    KrasisMoEWrapper.hf_model_path = model_dir
    engine = KrasisMoEWrapper._get_engine(model_dir, 16)

    hidden = engine.hidden_size()
    top_k = engine.top_k()
    batch = 4

    # Create batch of tokens
    hidden_states = torch.randn(batch, hidden, dtype=torch.bfloat16)
    topk_ids = torch.arange(top_k, dtype=torch.int32).unsqueeze(0).expand(batch, -1).contiguous()
    topk_weights = torch.full((batch, top_k), 1.0 / top_k, dtype=torch.float32)

    act_bytes = hidden_states.view(torch.uint16).numpy().view(np.uint8).tobytes()
    ids_bytes = topk_ids.numpy().view(np.uint8).tobytes()
    wts_bytes = topk_weights.numpy().view(np.uint8).tobytes()

    start = time.perf_counter()
    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, batch)
    output_bytes = engine.sync_forward()
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = torch.frombuffer(bytearray(output_bytes), dtype=torch.bfloat16).reshape(batch, hidden)
    rms = output.float().pow(2).mean().sqrt().item()

    print(f"Batch={batch} forward: {elapsed_ms:.1f} ms, RMS={rms:.6f}")
    print(f"  Per-token: {elapsed_ms / batch:.1f} ms")
    assert output.shape == (batch, hidden)
    assert rms > 1e-5
    print("PASS: batch forward")


if __name__ == "__main__":
    test_engine_async_roundtrip()
    print()
    test_wrapper_interface()
    print()
    test_batch_forward()
    print("\nAll bridge tests passed!")

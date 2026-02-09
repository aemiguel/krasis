"""Test GPU prefill with fused_marlin_moe on DeepSeek-V2-Lite."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
import numpy as np


def test_quantize_and_pack():
    """Test INT4 quantization on GPU."""
    from krasis.gpu_prefill import _quantize_and_pack_gpu

    # Create test weight [K=128, N=256] (Marlin convention: reduction dim first)
    w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda:0")

    packed, scale = _quantize_and_pack_gpu(w, group_size=128)

    print(f"Input:  {w.shape} {w.dtype}")
    print(f"Packed: {packed.shape} {packed.dtype}")  # [K//8, N] = [16, 256]
    print(f"Scale:  {scale.shape} {scale.dtype}")    # [K//128, N] = [1, 256]

    assert packed.shape == (128 // 8, 256), f"Wrong packed shape: {packed.shape}"
    assert scale.shape == (128 // 128, 256), f"Wrong scale shape: {scale.shape}"
    assert packed.dtype == torch.int32
    assert scale.dtype == torch.bfloat16

    # Verify non-zero
    assert packed.any(), "Packed weights all zero"
    assert scale.any(), "Scales all zero"
    print("PASS: quantize_and_pack_gpu\n")


def test_gpu_prefill_v2_lite():
    """Test full GPU prefill pipeline on V2-Lite."""
    from krasis.gpu_prefill import GpuPrefillManager

    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    device = torch.device("cuda:0")

    # V2-Lite: 64 experts, hidden=2048, intermediate=1408
    # n_shared_experts=2, routed_scaling_factor=1.0
    manager = GpuPrefillManager(
        model_path=model_dir,
        device=device,
        num_experts=64,
        hidden_size=2048,
        intermediate_size=1408,
        params_dtype=torch.bfloat16,
        n_shared_experts=2,
        routed_scaling_factor=1.0,
        first_k_dense=1,
    )

    # Prepare layer 0 (first MoE layer)
    start = time.perf_counter()
    manager.prepare_layer(0)
    prep_time = time.perf_counter() - start
    print(f"Layer 0 prepared in {prep_time:.1f}s ({prep_time/64:.2f}s/expert)")

    # Verify cache
    cache = manager._cache[0]
    print(f"  w13_packed: {cache['w13_packed'].shape}")  # [64, 128, 5632]
    print(f"  w13_scale:  {cache['w13_scale'].shape}")   # [64, 16, 2816]
    print(f"  w2_packed:  {cache['w2_packed'].shape}")    # [64, 88, 4096]
    print(f"  w2_scale:   {cache['w2_scale'].shape}")     # [64, 11, 2048]

    # Test forward: batch=32 tokens, top-6 experts
    batch = 32
    hidden = 2048
    top_k = 6

    x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, 64, (batch, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(torch.randn(batch, top_k, device=device), dim=-1).float()

    # Warmup
    _ = manager.forward(0, x, topk_ids, topk_weights)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    output = manager.forward(0, x, topk_ids, topk_weights)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) * 1000

    rms = output.float().pow(2).mean().sqrt().item()
    print(f"\nForward: batch={batch}, top_k={top_k}")
    print(f"  Time: {fwd_ms:.1f} ms")
    print(f"  Output: {output.shape} {output.dtype}, RMS={rms:.6f}")

    assert output.shape == (batch, hidden)
    assert output.dtype == torch.bfloat16
    assert rms > 1e-4, f"Output too small: RMS={rms}"
    print("PASS: gpu_prefill_v2_lite\n")


def test_gpu_prefill_shared_expert():
    """Test that shared expert changes output."""
    from krasis.gpu_prefill import GpuPrefillManager

    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    device = torch.device("cuda:0")

    # Create two managers: one with shared, one without
    manager_shared = GpuPrefillManager(
        model_path=model_dir, device=device,
        num_experts=64, hidden_size=2048, intermediate_size=1408,
        n_shared_experts=2, routed_scaling_factor=1.0, first_k_dense=1,
    )
    manager_no_shared = GpuPrefillManager(
        model_path=model_dir, device=device,
        num_experts=64, hidden_size=2048, intermediate_size=1408,
        n_shared_experts=0, routed_scaling_factor=1.0, first_k_dense=1,
    )

    manager_shared.prepare_layer(0)
    manager_no_shared.prepare_layer(0)
    # Share the same routed cache to isolate shared expert effect
    manager_no_shared._cache[0] = manager_shared._cache[0]

    batch = 8
    x = torch.randn(batch, 2048, dtype=torch.bfloat16, device=device)
    topk_ids = torch.arange(6, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    topk_weights = torch.full((batch, 6), 1.0/6, dtype=torch.float32, device=device)

    out_shared = manager_shared.forward(0, x, topk_ids, topk_weights)
    out_no_shared = manager_no_shared.forward(0, x, topk_ids, topk_weights)

    diff = (out_shared - out_no_shared).float().abs().max().item()
    rms_s = out_shared.float().pow(2).mean().sqrt().item()
    rms_n = out_no_shared.float().pow(2).mean().sqrt().item()

    print(f"Shared expert test:")
    print(f"  RMS with shared:    {rms_s:.6f}")
    print(f"  RMS without shared: {rms_n:.6f}")
    print(f"  Max diff:           {diff:.6f}")

    assert diff > 1e-4, f"Shared expert should change output (diff={diff})"
    print("PASS: gpu_prefill_shared_expert\n")


def test_gpu_vs_cpu_consistency():
    """Compare GPU prefill vs CPU decode outputs (should be roughly similar)."""
    from krasis.gpu_prefill import GpuPrefillManager
    from krasis import KrasisEngine

    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    device = torch.device("cuda:0")

    # GPU path
    gpu_mgr = GpuPrefillManager(
        model_path=model_dir, device=device,
        num_experts=64, hidden_size=2048, intermediate_size=1408,
        n_shared_experts=2, routed_scaling_factor=1.0, first_k_dense=1,
    )
    gpu_mgr.prepare_layer(0)

    # CPU path
    engine = KrasisEngine(parallel=True, num_threads=16)
    engine.load(model_dir)

    batch = 1
    hidden = 2048
    top_k = 6

    # Use same activation and routing for both
    x_gpu = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    topk_ids_gpu = torch.arange(top_k, dtype=torch.int32, device=device).unsqueeze(0)
    topk_weights_gpu = torch.full((batch, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

    # GPU forward
    gpu_out = gpu_mgr.forward(0, x_gpu, topk_ids_gpu, topk_weights_gpu)

    # CPU forward
    x_cpu = x_gpu.cpu()
    act_bytes = x_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
    ids_bytes = topk_ids_gpu.cpu().numpy().view(np.uint8).tobytes()
    wts_bytes = topk_weights_gpu.cpu().numpy().view(np.uint8).tobytes()

    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, batch)
    cpu_bytes = engine.sync_forward()
    cpu_out = torch.frombuffer(bytearray(cpu_bytes), dtype=torch.bfloat16).reshape(batch, hidden)

    # Compare (not bit-exact due to different quantization: Marlin INT4 vs Krasis INT4)
    gpu_f = gpu_out.cpu().float()
    cpu_f = cpu_out.float()
    cosine = torch.nn.functional.cosine_similarity(gpu_f.flatten(), cpu_f.flatten(), dim=0).item()
    max_diff = (gpu_f - cpu_f).abs().max().item()
    rms_gpu = gpu_f.pow(2).mean().sqrt().item()
    rms_cpu = cpu_f.pow(2).mean().sqrt().item()

    print(f"GPU vs CPU consistency (V2-Lite, batch=1, top-{top_k}):")
    print(f"  GPU RMS: {rms_gpu:.6f}")
    print(f"  CPU RMS: {rms_cpu:.6f}")
    print(f"  Cosine similarity: {cosine:.4f}")
    print(f"  Max abs diff: {max_diff:.6f}")

    # They won't be identical (different quantization methods), but should be correlated
    assert cosine > 0.5, f"GPU/CPU outputs not correlated: cosine={cosine}"
    assert rms_gpu > 1e-4, f"GPU output too small"
    assert rms_cpu > 1e-4, f"CPU output too small"
    print("PASS: gpu_vs_cpu_consistency\n")


def test_scaling_performance():
    """Test GPU prefill performance at different batch sizes."""
    from krasis.gpu_prefill import GpuPrefillManager

    model_dir = os.path.expanduser("~/Documents/Claude/hf-models/DeepSeek-V2-Lite")
    if not os.path.exists(model_dir):
        print("SKIP: V2-Lite not downloaded")
        return

    device = torch.device("cuda:0")
    manager = GpuPrefillManager(
        model_path=model_dir, device=device,
        num_experts=64, hidden_size=2048, intermediate_size=1408,
        n_shared_experts=2, routed_scaling_factor=1.0, first_k_dense=1,
    )
    manager.prepare_layer(0)

    print("GPU prefill scaling (V2-Lite, 1 layer):")
    for batch in [1, 8, 32, 128, 512]:
        x = torch.randn(batch, 2048, dtype=torch.bfloat16, device=device)
        ids = torch.randint(0, 64, (batch, 6), dtype=torch.int32, device=device)
        wts = torch.softmax(torch.randn(batch, 6, device=device), dim=-1).float()

        # Warmup
        _ = manager.forward(0, x, ids, wts)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 5
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = manager.forward(0, x, ids, wts)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iters * 1000

        print(f"  batch={batch:4d}: {elapsed:6.1f} ms ({batch / elapsed * 1000:.0f} tok/s)")

    print("PASS: scaling_performance\n")


def _gpu_cleanup():
    """Force GPU cleanup between tests to avoid stale CUDA state."""
    import gc
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_quantize_and_pack()
    _gpu_cleanup()
    test_gpu_prefill_v2_lite()
    _gpu_cleanup()
    test_gpu_prefill_shared_expert()
    _gpu_cleanup()
    test_gpu_vs_cpu_consistency()
    _gpu_cleanup()
    test_scaling_performance()
    print("All GPU prefill tests passed!")

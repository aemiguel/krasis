#!/usr/bin/env python3
"""Focused test: fused_marlin_moe on GPU1 with Kimi K2.5 dimensions.

Tests the Marlin kernel in isolation on cuda:1 to diagnose
the PP boundary crash (layer 31 = first MoE on GPU1).
"""

import logging, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("marlin-gpu1-test")

import torch

# Kimi K2.5 MoE dimensions
HIDDEN = 7168
INTERMEDIATE = 2048
NUM_EXPERTS = 384
TOP_K = 8
GROUP_SIZE = 128
NUM_BITS = 4

def create_marlin_buffers(num_experts, device, num_bits=4):
    """Create properly-shaped Marlin buffers with random data."""
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    K = HIDDEN
    N = INTERMEDIATE
    nb2 = num_bits // 2

    w13_packed = torch.randint(0, 2**31, (num_experts, K // 16, 2 * N * nb2),
                               dtype=torch.int32, device=device)
    w13_scale = torch.randn(num_experts, K // GROUP_SIZE, 2 * N,
                            dtype=torch.bfloat16, device=device) * 0.01
    w2_packed = torch.randint(0, 2**31, (num_experts, N // 16, K * nb2),
                              dtype=torch.int32, device=device)
    w2_scale = torch.randn(num_experts, N // GROUP_SIZE, K,
                           dtype=torch.bfloat16, device=device) * 0.01
    g_idx = torch.empty(num_experts, 0, dtype=torch.int32, device=device)
    sort_idx = torch.empty(num_experts, 0, dtype=torch.int32, device=device)
    workspace = marlin_make_workspace(device, max_blocks_per_sm=4)

    return w13_packed, w13_scale, w2_packed, w2_scale, g_idx, sort_idx, workspace


def test_single_chunk(device, M=333):
    """Test fused_marlin_moe with all 384 experts in one chunk."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

    # Fix: sgl_kernel launches on current device, not tensor device
    torch.cuda.set_device(device)
    logger.info("=== Single chunk test on %s (E=%d, M=%d) ===", device, NUM_EXPERTS, M)

    w13_packed, w13_scale, w2_packed, w2_scale, g_idx, sort_idx, workspace = \
        create_marlin_buffers(NUM_EXPERTS, device)
    buf_mb = (w13_packed.nbytes + w13_scale.nbytes + w2_packed.nbytes + w2_scale.nbytes) / 1e6
    logger.info("Buffer allocated: %.1f MB on %s", buf_mb, device)

    x = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.randn(M, TOP_K, dtype=torch.float32, device=device).softmax(dim=-1)
    gating = torch.empty(M, NUM_EXPERTS, device=device)

    torch.cuda.synchronize(device)
    logger.info("Pre-kernel sync OK")

    t0 = time.perf_counter()
    output = fused_marlin_moe(
        hidden_states=x,
        w1=w13_packed, w2=w2_packed,
        w1_scale=w13_scale, w2_scale=w2_scale,
        gating_output=gating,
        topk_weights=topk_weights, topk_ids=topk_ids,
        global_num_experts=NUM_EXPERTS,
        expert_map=None,
        g_idx1=g_idx, g_idx2=g_idx,
        sort_indices1=sort_idx, sort_indices2=sort_idx,
        workspace=workspace,
        num_bits=NUM_BITS, is_k_full=True,
    ).to(x.dtype)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    logger.info("Single chunk: output=%s, time=%.1fms, max=%.4f", output.shape, elapsed*1000, output.abs().max().item())

    # Cleanup
    del w13_packed, w13_scale, w2_packed, w2_scale, g_idx, sort_idx, workspace, output
    torch.cuda.empty_cache()
    return True


def test_multi_chunk(device, chunk_size=223, M=333):
    """Test fused_marlin_moe with chunked expert processing (mimics GPU1 crash scenario)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    # Fix: sgl_kernel launches on current device, not tensor device
    torch.cuda.set_device(device)
    num_chunks = (NUM_EXPERTS + chunk_size - 1) // chunk_size
    logger.info("=== Multi-chunk test on %s (E=%d, chunk=%d, chunks=%d, M=%d) ===",
                device, NUM_EXPERTS, chunk_size, num_chunks, M)

    # Allocate chunk-sized buffer (like GpuPrefillManager does)
    w13_packed, w13_scale, w2_packed, w2_scale, g_idx, sort_idx, workspace = \
        create_marlin_buffers(chunk_size, device)
    buf_mb = (w13_packed.nbytes + w13_scale.nbytes + w2_packed.nbytes + w2_scale.nbytes) / 1e6
    logger.info("Chunk buffer: %.1f MB on %s", buf_mb, device)

    # Create full-model expert weights on CPU (simulating RAM cache)
    K, N = HIDDEN, INTERMEDIATE
    nb2 = NUM_BITS // 2
    cpu_w13_packed = torch.randint(0, 2**31, (NUM_EXPERTS, K // 16, 2 * N * nb2), dtype=torch.int32)
    cpu_w13_scale = torch.randn(NUM_EXPERTS, K // GROUP_SIZE, 2 * N, dtype=torch.bfloat16) * 0.01
    cpu_w2_packed = torch.randint(0, 2**31, (NUM_EXPERTS, N // 16, K * nb2), dtype=torch.int32)
    cpu_w2_scale = torch.randn(NUM_EXPERTS, N // GROUP_SIZE, K, dtype=torch.bfloat16) * 0.01

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.randn(M, TOP_K, dtype=torch.float32, device=device).softmax(dim=-1)
    gating = torch.empty(M, chunk_size, device=device)

    output = torch.zeros(M, K, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, NUM_EXPERTS)
        actual = end - start

        # Copy chunk to GPU
        w13_packed[:actual].copy_(cpu_w13_packed[start:end])
        w13_scale[:actual].copy_(cpu_w13_scale[start:end])
        w2_packed[:actual].copy_(cpu_w2_packed[start:end])
        w2_scale[:actual].copy_(cpu_w2_scale[start:end])

        # Remap expert IDs
        chunk_mask = (topk_ids >= start) & (topk_ids < end)
        chunk_ids = torch.where(chunk_mask, topk_ids - start, torch.zeros_like(topk_ids))
        chunk_wts = torch.where(chunk_mask, topk_weights, torch.zeros_like(topk_weights))

        workspace.zero_()

        torch.cuda.synchronize(device)
        logger.info("  chunk %d/%d: actual=%d, pre-kernel sync OK", chunk_idx, num_chunks, actual)

        chunk_output = fused_marlin_moe(
            hidden_states=x,
            w1=w13_packed[:actual], w2=w2_packed[:actual],
            w1_scale=w13_scale[:actual], w2_scale=w2_scale[:actual],
            gating_output=gating[:, :actual],
            topk_weights=chunk_wts, topk_ids=chunk_ids,
            global_num_experts=actual,
            expert_map=None,
            g_idx1=g_idx[:actual], g_idx2=g_idx[:actual],
            sort_indices1=sort_idx[:actual], sort_indices2=sort_idx[:actual],
            workspace=workspace,
            num_bits=NUM_BITS, is_k_full=True,
        ).to(x.dtype)

        torch.cuda.synchronize(device)
        logger.info("  chunk %d/%d: post-kernel sync OK, max=%.4f",
                    chunk_idx, num_chunks, chunk_output.abs().max().item())

        output += chunk_output

    elapsed = time.perf_counter() - t0
    logger.info("Multi-chunk: output=%s, total=%.1fms, max=%.4f",
                output.shape, elapsed*1000, output.abs().max().item())

    del w13_packed, w13_scale, w2_packed, w2_scale, g_idx, sort_idx, workspace
    del cpu_w13_packed, cpu_w13_scale, cpu_w2_packed, cpu_w2_scale
    torch.cuda.empty_cache()
    return True


def main():
    logger.info("CUDA devices: %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        logger.info("  GPU%d: %s, %.1f/%.1f GB free", i, p.name, free/1e9, total/1e9)

    # Test 1: Single chunk on GPU0 (known working baseline)
    logger.info("\n--- GPU0 Tests (baseline) ---")
    test_single_chunk(torch.device("cuda:0"), M=100)
    test_multi_chunk(torch.device("cuda:0"), chunk_size=186, M=100)

    # Test 2: Single chunk on GPU1 (crash target)
    logger.info("\n--- GPU1 Tests (crash investigation) ---")
    test_single_chunk(torch.device("cuda:1"), M=100)
    test_multi_chunk(torch.device("cuda:1"), chunk_size=223, M=100)

    # Test 3: Multi-chunk on GPU1 with exact crash M=333
    test_multi_chunk(torch.device("cuda:1"), chunk_size=223, M=333)

    logger.info("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()

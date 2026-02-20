"""GPU prefill for MoE layers using INT4 Marlin kernel.

During prefill (many tokens), GPU is much faster than CPU for expert computation.
This module handles:
- Reading expert weights from HF safetensors
- Quantizing BF16 → INT4 symmetric (group_size=128) on GPU
- Repacking to Marlin format (gptq_marlin_moe_repack)
- Caching quantized weights in RAM (and optionally disk)
- Calling fused_marlin_moe kernel with chunked expert buffer

Usage:
    manager = GpuPrefillManager(model_path, device, num_experts, hidden_size, intermediate_size)
    manager.prepare_layer(moe_layer_idx)  # one-time quantize + cache
    output = manager.forward(moe_layer_idx, hidden_states, topk_ids, topk_weights)
"""

import dataclasses
import gc
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

from krasis.timing import TIMING

# Marlin quantization constants
GROUP_SIZE = 128


def _gpt_oss_activation(gate_up: torch.Tensor, output: torch.Tensor, limit: float, alpha: float):
    """GPT OSS custom activation replacing silu_and_mul.

    Formula: gate * sigmoid(gate * alpha) * (up + 1)
    with gate clamped to (-inf, limit] and up clamped to [-limit, limit].

    Args:
        gate_up: [T, 2N] where first N columns = gate, last N columns = up
        output: [T, N] pre-allocated output buffer
        limit: clamping limit (7.0 for GPT OSS)
        alpha: activation alpha (1.702 for GPT OSS)
    """
    N = output.shape[-1]
    gate = gate_up[..., :N].clamp(max=limit)
    up = gate_up[..., N:].clamp(min=-limit, max=limit)
    output[:] = gate * torch.sigmoid(gate * alpha) * (up + 1.0)


def _fused_marlin_moe_gpt_oss(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
    swiglu_limit: float,
    activation_alpha: float,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 4,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """GPT OSS variant of fused_marlin_moe with custom activation and expert biases.

    Mirrors sglang's fused_marlin_moe but replaces silu_and_mul with:
    gate * sigmoid(gate * 1.702) * (up + 1), with clamping.
    Also supports per-expert biases via b_bias_or_none.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sgl_kernel import moe_sum_reduce

    assert hidden_states.is_contiguous()

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16  # intermediate_size (input dim for second GEMM)
    topk = topk_ids.shape[1]
    # Detect Marlin w2 padding: actual output dim from w2 packed shape
    nb2 = num_bits // 2
    K_out = w2.shape[2] // nb2  # may be > K if padded (e.g., 2944 vs 2880)
    w2_padded = K_out != K

    # Block size selection (same as original)
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K_out) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import get_scalar_type
    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk * max(2 * N, K_out),), device=hidden_states.device, dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk * 2 * N].view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk * K_out].view(-1, K_out)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )

    # First GEMM: hidden → gate_up (with optional bias)
    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        hidden_states,
        intermediate_cache1,
        w1,
        gate_up_bias,
        w1_scale,
        None,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type1.id,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # GPT OSS custom activation (replaces silu_and_mul)
    _gpt_oss_activation(
        intermediate_cache1.view(-1, 2 * N),
        intermediate_cache2,
        swiglu_limit,
        activation_alpha,
    )

    if expert_map is not None:
        intermediate_cache3.zero_()

    # Pad down_bias if w2 is padded (bias shape must match output dim)
    if w2_padded and down_bias is not None:
        pad_size = K_out - K
        down_bias = torch.nn.functional.pad(down_bias, (0, pad_size))

    # Second GEMM: activated → output (with optional bias)
    # K_out may be > K if w2 was padded for Marlin kernel compatibility
    intermediate_cache3 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        down_bias,
        w2_scale,
        None,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type2.id,
        size_m=M * topk,
        size_n=K_out,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # Slice off padding columns if w2 was padded
    if w2_padded:
        intermediate_cache3 = intermediate_cache3[:, :K].contiguous()
    intermediate_cache3 = intermediate_cache3.view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    moe_sum_reduce(intermediate_cache3, output, routed_scaling_factor)
    return output


def _quantize_and_pack_gpu(w: torch.Tensor, group_size: int = GROUP_SIZE, num_bits: int = 4):
    """Quantize [K, N] weight to INT4 or INT8 packed on GPU.

    Input weight must be transposed to [K, N] (K = input/reduction dim, N = output dim).
    This matches the GPTQ convention where packing is along the K dimension.

    Args:
        w: [K, N] weight tensor
        group_size: quantization group size
        num_bits: 4 for INT4 or 8 for INT8

    Returns:
        packed: [K//vals_per_int32, N] int32
        scale: [K//group_size, N] same dtype as w
    """
    K, N = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    assert num_bits in (4, 8), f"num_bits must be 4 or 8, got {num_bits}"

    w_float = w.float()
    w_grouped = w_float.reshape(K // group_size, group_size, N)

    if num_bits == 4:
        # Symmetric INT4: scale = max(|max|/7, |min|/8) per group
        max_val = w_grouped.amax(dim=1, keepdim=True)
        min_val = w_grouped.amin(dim=1, keepdim=True)
        scale = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0)
        scale = scale.clamp(min=1e-10)
        w_q = (w_grouped / scale).round().clamp(-8, 7).to(torch.int32) + 8
        vals_per_int32 = 8
    else:  # num_bits == 8
        # Symmetric INT8: scale = max(|max|/127, |min|/128) per group
        max_val = w_grouped.amax(dim=1, keepdim=True)
        min_val = w_grouped.amin(dim=1, keepdim=True)
        scale = torch.max(max_val.abs() / 127.0, min_val.abs() / 128.0)
        scale = scale.clamp(min=1e-10)
        w_q = (w_grouped / scale).round().clamp(-128, 127).to(torch.int32) + 128
        vals_per_int32 = 4

    w_q = w_q.reshape(K, N)
    scale = scale.squeeze(1).to(w.dtype)  # [K//group_size, N], match model dtype

    # Pack values into int32
    assert K % vals_per_int32 == 0, f"K={K} not divisible by vals_per_int32={vals_per_int32}"
    w_q = w_q.reshape(K // vals_per_int32, vals_per_int32, N)
    packed = w_q[:, 0, :].clone()
    for i in range(1, vals_per_int32):
        packed |= w_q[:, i, :] << (num_bits * i)

    return packed, scale


@dataclasses.dataclass
class HcsDeviceState:
    """Per-device HCS state for unified multi-GPU decode."""
    device: torch.device
    stream: object  # torch.cuda.Stream
    buffers: dict   # layer_idx -> {w13, w13_scale, w2, w2_scale}
    lookup: dict    # layer_idx -> tensor[num_experts] -> local_slot or -1
    num_pinned: dict  # layer_idx -> int
    biases: dict    # layer_idx -> {gate_up_bias, down_bias}

    # Primary device (rank 0) has some unique resources
    is_primary: bool = False
    g_idx: Optional[dict] = None
    sort_idx: Optional[dict] = None
    gating_1: Optional[dict] = None

    # Per-device CUDA graph state for multi-GPU decode
    cuda_graphs: Optional[dict] = None      # layer_idx -> CUDAGraph
    graph_io: Optional[dict] = None         # layer_idx -> {x, local_ids, weights, gating, output}
    workspace: Optional[torch.Tensor] = None
    lookup_on_primary: Optional[dict] = None  # layer_idx -> lookup tensor on primary device



class GpuPrefillManager:
    """Manages GPU expert buffer and INT4 Marlin weights for prefill.

    The manager owns a fixed-size GPU buffer that holds one chunk of experts.
    For models with many experts (e.g. Kimi K2.5 with 384), experts are
    processed in chunks that fit in VRAM.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
        n_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense: int = 0,
        chunk_size: Optional[int] = None,
        num_bits: int = 4,
        krasis_engine=None,
        num_moe_layers: int = 0,
        expert_divisor: int = 1,
        skip_shared_experts: bool = False,
        swiglu_limit: float = 0.0,
        rank: int = 0,
        num_ranks: int = 1,
    ):
        self.model_path = model_path
        self.device = device
        self.num_experts = num_experts
        self.rank = rank
        self.num_ranks = num_ranks
        
        # Internalize expert slicing for EP prefill
        self.expert_start = rank * (num_experts // num_ranks)
        if rank == num_ranks - 1:
            self.expert_end = num_experts
        else:
            self.expert_end = (rank + 1) * (num_experts // num_ranks)
        self.num_local_experts = self.expert_end - self.expert_start
        
        logger.info(
            "GpuPrefillManager(rank %d/%d): expert_slice=[%d, %d), local_count=%d",
            rank, num_ranks, self.expert_start, self.expert_end, self.num_local_experts
        )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.params_dtype = params_dtype
        # If skip_shared_experts, Python layer handles shared expert with gate
        self.n_shared_experts = 0 if skip_shared_experts else n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense
        self.num_bits = num_bits
        self._engine = krasis_engine
        self._num_moe_layers = num_moe_layers
        self._expert_divisor = expert_divisor
        self.swiglu_limit = swiglu_limit
        self.activation_alpha = 1.702 if swiglu_limit > 0 else 0.0
        # Per-layer expert biases for GPT OSS: {moe_layer_idx: {gate_up_bias, down_bias}}
        self._expert_biases: dict[int, dict[str, torch.Tensor]] = {}

        # Use engine's group_size if available, else default 128
        if krasis_engine is not None:
            self._group_size = krasis_engine.group_size()
        else:
            self._group_size = GROUP_SIZE
        logger.info("GPU prefill group_size=%d", self._group_size)

        # Padded hidden size for Marlin w2 kernel compatibility (e.g., 2880→2944)
        if krasis_engine is not None:
            self._w2_padded_n = krasis_engine.marlin_w2_padded_n()
        else:
            self._w2_padded_n = hidden_size
        if self._w2_padded_n != hidden_size:
            logger.info("Marlin w2 padded N: %d → %d (kernel compat)", hidden_size, self._w2_padded_n)

        # Persistent/layer-grouped expert buffers
        self._prefill_mode = "chunked"  # "persistent", "layer_grouped", "active_only", or "chunked"
        self._persistent: Optional[dict[int, dict[str, torch.Tensor]]] = None
        self._persistent_shared: Optional[dict[int, dict[str, torch.Tensor]]] = None

        # DMA pipelining: prefetch next group while current group computes
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_persistent: dict[int, dict[str, torch.Tensor]] = {}
        self._prefetch_persistent_shared: dict[int, dict[str, torch.Tensor]] = {}

        # Active-only mode state: per-layer GPU buffer + loaded expert tracking
        self._ao_gpu_w13: Optional[torch.Tensor] = None
        self._ao_gpu_w13_scale: Optional[torch.Tensor] = None
        self._ao_gpu_w2: Optional[torch.Tensor] = None
        self._ao_gpu_w2_scale: Optional[torch.Tensor] = None
        self._ao_current_layer: int = -1
        self._ao_loaded_experts: set = set()
        self._ao_dma_time: float = 0.0
        self._ao_dma_bytes: int = 0
        self._ao_cache_hits: int = 0
        self._ao_cache_misses: int = 0

        # Expert activation heatmap: {(layer, expert): count} — built during warmup
        self._heatmap: dict[tuple[int, int], int] = {}
        self._heatmap_requests: int = 0

        # Static pin state: pinned (layer, expert) → {w13, w13_scale, w2, w2_scale} on GPU
        self._pinned: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._pin_budget_mb: float = 0  # Set via configure_pinning()
        self._warmup_requests: int = 1  # Requests before pinning activates

        # LRU cache state: (layer, expert) → slot_index in _lru_gpu_* buffers
        self._lru_map: dict[tuple[int, int], int] = {}  # key → slot
        self._lru_access_time: dict[int, int] = {}  # slot → monotonic counter
        self._lru_slot_key: dict[int, tuple[int, int]] = {}  # slot → key
        self._lru_counter: int = 0  # monotonic access counter
        self._lru_gpu_w13: Optional[torch.Tensor] = None
        self._lru_gpu_w13_scale: Optional[torch.Tensor] = None
        self._lru_gpu_w2: Optional[torch.Tensor] = None
        self._lru_gpu_w2_scale: Optional[torch.Tensor] = None
        self._lru_cache_size: int = 0

        # Compact per-layer decode buffers (views into AO buffer, zero extra VRAM)
        # Each MoE layer gets its own small buffer for cross-token expert caching
        self._compact_initialized: bool = False
        self._compact_slots_per_layer: int = 21
        self._compact_next_partition: int = 0
        self._compact_bufs: dict[int, dict] = {}  # moe_layer_idx → {w13, w13_scale, w2, w2_scale}
        self._compact_maps: dict[int, dict] = {}  # moe_layer_idx → {global_eid: local_slot}
        self._compact_reverse: dict[int, dict] = {}  # moe_layer_idx → {local_slot: global_eid}
        self._compact_lru: dict[int, dict] = {}  # moe_layer_idx → {local_slot: counter}
        self._compact_next_slot: dict[int, int] = {}  # moe_layer_idx → next empty slot
        self._compact_counter: int = 0
        self._compact_g_idx: Optional[torch.Tensor] = None
        self._compact_sort_idx: Optional[torch.Tensor] = None
        self._compact_gating_1: Optional[torch.Tensor] = None

        # Pre-allocated CUDA-pinned DMA staging buffers
        # Pinned memory enables truly async .to(device, non_blocking=True)
        self._dma_bufs_initialized: bool = False
        self._dma_pinned: bool = False
        self._dma_buf_w13p: Optional[torch.Tensor] = None
        self._dma_buf_w13s: Optional[torch.Tensor] = None
        self._dma_buf_w2p: Optional[torch.Tensor] = None
        self._dma_buf_w2s: Optional[torch.Tensor] = None
        self._dma_size_w13p: int = 0
        self._dma_size_w13s: int = 0
        self._dma_size_w2p: int = 0
        self._dma_size_w2s: int = 0
        self._dma_shape_w13p: tuple = ()
        self._dma_shape_w13s: tuple = ()
        self._dma_shape_w2p: tuple = ()
        self._dma_shape_w2s: tuple = ()
        # Shared expert staging buffers (smaller, 1 expert per layer)
        self._dma_shared_w13p: Optional[torch.Tensor] = None
        self._dma_shared_w13s: Optional[torch.Tensor] = None
        self._dma_shared_w2p: Optional[torch.Tensor] = None
        self._dma_shared_w2s: Optional[torch.Tensor] = None
        self._dma_shared_initialized: bool = False
        self._dma_shared_size_w13p: int = 0
        self._dma_shared_size_w13s: int = 0
        self._dma_shared_size_w2p: int = 0
        self._dma_shared_size_w2s: int = 0
        self._dma_shared_shape_w13p: tuple = ()
        self._dma_shared_shape_w13s: tuple = ()
        self._dma_shared_shape_w2p: tuple = ()
        self._dma_shared_shape_w2s: tuple = ()

        # Hot-cached-static state: static GPU buffers + CPU cold in parallel
        self._hcs_buffers: dict[int, dict[str, torch.Tensor]] = {}  # layer → {w13, w13_scale, w2, w2_scale}
        self._hcs_lookup: dict[int, torch.Tensor] = {}  # layer → [num_experts] int32 → local slot or -1
        self._hcs_num_pinned: dict[int, int] = {}  # layer → count of hot experts
        self._hcs_g_idx: dict[int, torch.Tensor] = {}  # layer → [n_hot, 0] kernel helper
        self._hcs_sort_idx: dict[int, torch.Tensor] = {}  # layer → [n_hot, 0] kernel helper
        self._hcs_gating_1: dict[int, torch.Tensor] = {}  # layer → [1, n_hot] for M=1
        self._hcs_initialized: bool = False
        self._hcs_biases: dict[int, dict[str, torch.Tensor]] = {}  # layer → remapped biases for hot experts
        # CUDA graph state (optional acceleration for hot_cached_static)
        self._hcs_cuda_graphs: dict[int, object] = {}  # layer → CUDAGraph
        self._hcs_graph_io: dict[int, dict[str, torch.Tensor]] = {}  # layer → fixed I/O tensors
        self._hcs_cuda_graphs_enabled: bool = False
        # Unified multi-GPU HCS state
        self._hcs_devices: list[HcsDeviceState] = []
        self._hcs_device_lookup: dict[int, torch.Tensor] = {}  # layer → [num_experts] int16 → device_idx (-1=cold)

        # Prefill timing accumulators (reset per request via reset_prefill_stats)
        self._pf_cpu_submit_ms: float = 0.0
        self._pf_gpu_ms: float = 0.0
        self._pf_cpu_sync_ms: float = 0.0
        self._pf_cpu_to_gpu_ms: float = 0.0
        self._pf_combine_ms: float = 0.0
        self._pf_cpu_only_ms: float = 0.0
        self._pf_layers_total: int = 0
        self._pf_layers_cold: int = 0
        self._pf_layers_all_cold: int = 0
        self._pf_cold_experts: int = 0
        self._pf_total_experts: int = 0

        # Chunk size: how many experts fit in one GPU buffer load
        if chunk_size is None:
            chunk_size = self._auto_chunk_size()
        self.chunk_size = min(chunk_size, num_experts)
        self.num_chunks = (num_experts + self.chunk_size - 1) // self.chunk_size

        # RAM cache: layer_idx -> {w13_packed, w13_scale, w2_packed, w2_scale}
        # Only used when krasis_engine is None (legacy safetensors path)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

        # Safetensors index (lazy loaded, legacy path only)
        self._weight_map: Optional[dict[str, str]] = None
        self._shards: dict[str, object] = {}
        self._layers_prefix: Optional[str] = None

        # GPU buffer tensors (allocated lazily)
        self._gpu_w13_packed: Optional[torch.Tensor] = None
        self._gpu_w13_scale: Optional[torch.Tensor] = None
        self._gpu_w2_packed: Optional[torch.Tensor] = None
        self._gpu_w2_scale: Optional[torch.Tensor] = None
        self._gpu_g_idx: Optional[torch.Tensor] = None
        self._gpu_sort_idx: Optional[torch.Tensor] = None
        self._workspace: Optional[torch.Tensor] = None

        # Shared expert GPU cache (separate from chunked buffer, legacy path only)
        self._shared_cache: dict[int, dict[str, torch.Tensor]] = {}

        # Select prefill mode based on expert_divisor
        self._select_prefill_mode()

        mode = "engine" if self._engine is not None else "safetensors"
        logger.info(
            "GpuPrefillManager(%s): experts=%d, hidden=%d, intermediate=%d, "
            "chunk_size=%d, num_chunks=%d, shared=%d, scale=%.3f, num_bits=%d, "
            "prefill_mode=%s, expert_divisor=%d",
            mode, num_experts, hidden_size, intermediate_size,
            self.chunk_size, self.num_chunks,
            n_shared_experts, routed_scaling_factor, num_bits,
            self._prefill_mode, self._expert_divisor,
        )

        # Pre-allocate GPU buffer eagerly so KV cache auto-sizer sees the reservation
        # (not needed for persistent/layer_grouped — buffers allocated per-group)
        if self._prefill_mode == "chunked":
            self._allocate_gpu_buffer()
        elif self._prefill_mode == "active_only":
            self._allocate_ao_buffer()
        elif self._prefill_mode == "lru":
            self._allocate_ao_buffer()  # Staging buffer
            self._allocate_lru_buffer()  # LRU cache buffer

    def _call_fused_moe(
        self,
        moe_layer_idx: int,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int = -1,
        expert_map=None,
        g_idx1=None,
        g_idx2=None,
        sort_indices1=None,
        sort_indices2=None,
        workspace=None,
        num_bits=None,
        is_k_full=True,
        inplace=False,
        routed_scaling_factor=None,
        bias_gate_up: Optional[torch.Tensor] = None,
        bias_down: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dispatch to standard or GPT OSS fused_marlin_moe based on swiglu_limit.

        For GPT OSS models (swiglu_limit > 0), uses custom activation + biases.
        bias_gate_up/bias_down override auto-lookup (needed for HCS with remapped slots).
        """
        if num_bits is None:
            num_bits = self.num_bits

        if self.swiglu_limit > 0:
            # Use explicit biases if provided, else look up from layer index
            if bias_gate_up is None or bias_down is None:
                biases = self._expert_biases.get(moe_layer_idx, {})
                if bias_gate_up is None:
                    bias_gate_up = biases.get("gate_up_bias")
                if bias_down is None:
                    bias_down = biases.get("down_bias")
            return _fused_marlin_moe_gpt_oss(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                gating_output=gating_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                gate_up_bias=bias_gate_up,
                down_bias=bias_down,
                swiglu_limit=self.swiglu_limit,
                activation_alpha=self.activation_alpha,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                g_idx1=g_idx1,
                g_idx2=g_idx2,
                sort_indices1=sort_indices1,
                sort_indices2=sort_indices2,
                workspace=workspace,
                num_bits=num_bits,
                is_k_full=is_k_full,
                inplace=inplace,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                fused_marlin_moe,
            )
            return fused_marlin_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                gating_output=gating_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                g_idx1=g_idx1,
                g_idx2=g_idx2,
                sort_indices1=sort_indices1,
                sort_indices2=sort_indices2,
                workspace=workspace,
                num_bits=num_bits,
                is_k_full=is_k_full,
                inplace=inplace,
                routed_scaling_factor=routed_scaling_factor,
            )

    def _per_expert_vram_bytes(self) -> int:
        """Calculate per-expert VRAM usage in bytes for Marlin format."""
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # may be > K if padded for Marlin compat
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2  # 2 for INT4, 4 for INT8
        w13_bytes = (K // 16) * (2 * N * nb2) * 4 + (K // gs) * (2 * N) * 2
        w2_bytes = (N // 16) * (K_w2 * nb2) * 4 + (N // gs) * K_w2 * 2
        return w13_bytes + w2_bytes

    def _select_prefill_mode(self):
        """Select prefill mode based on expert_divisor setting.

        - divisor=-3: "hot_cached_static" (hot on GPU, cold on CPU, parallel, zero DMA)
        - divisor=-2: "lru" (cross-layer LRU expert caching)
        - divisor=-1: "active_only" (per-layer, load only activated experts)
        - divisor=0 or no engine: "chunked" (current DMA-per-layer behavior)
        - divisor=1: "persistent" (all experts pre-loaded in VRAM, zero DMA)
        - divisor>=2: "layer_grouped" (load experts per-group-of-layers, O(groups) DMA)
        """
        if self._expert_divisor == -3 and self._engine is not None:
            self._prefill_mode = "hot_cached_static"
            logger.info("Hot-cached-static mode: hot experts on GPU (static), cold on CPU (parallel)")
            return

        if self._expert_divisor == -1 and self._engine is not None:
            self._prefill_mode = "active_only"
            logger.info("Active-only mode: load only activated experts per-layer (zero preload)")
            return

        if self._expert_divisor == -2 and self._engine is not None:
            self._prefill_mode = "lru"
            logger.info("LRU cache mode: cross-layer expert caching with LRU eviction")
            return

        if self._expert_divisor == 0 or self._engine is None:
            self._prefill_mode = "chunked"
            return

        per_expert = self._per_expert_vram_bytes()
        num_moe = self._num_moe_layers

        if self._expert_divisor == 1:
            # Full persistent: all experts for all MoE layers
            total_vram = per_expert * self.num_experts * num_moe
            total_mb = total_vram / 1e6

            try:
                free_vram = torch.cuda.mem_get_info(self.device)[0]
                # Need room for kernel intermediates + KV cache
                budget = free_vram - 500 * 1024 * 1024  # 500 MB headroom
                if total_vram <= budget:
                    self._prefill_mode = "persistent"
                    logger.info(
                        "Persistent mode selected: %d layers × %d experts = %.1f MB "
                        "(%.1f MB free, %.1f MB budget)",
                        num_moe, self.num_experts, total_mb,
                        free_vram / 1e6, budget / 1e6,
                    )
                else:
                    # OOM fallback: try layer_grouped with divisor=2
                    logger.warning(
                        "Persistent mode needs %.1f MB but only %.1f MB budget — "
                        "falling back to layer_grouped (divisor=2)",
                        total_mb, budget / 1e6,
                    )
                    self._expert_divisor = 2
                    self._select_prefill_mode()  # recurse once
                    return
            except Exception:
                self._prefill_mode = "persistent"
                logger.warning("Could not query VRAM — assuming persistent fits")
        else:
            # Layer-grouped: experts loaded per-group of layers
            from math import ceil
            moe_per_group = ceil(num_moe / self._expert_divisor)
            num_groups = ceil(num_moe / moe_per_group)
            group_vram = per_expert * self.num_experts * moe_per_group
            group_mb = group_vram / 1e6
            self._prefill_mode = "layer_grouped"
            logger.info(
                "Layer-grouped mode: divisor=%d, %d groups of %d MoE layers, "
                "~%.1f MB per group, %d total MoE layers",
                self._expert_divisor, num_groups, moe_per_group,
                group_mb, num_moe,
            )

    def _auto_chunk_size(self) -> int:
        """Estimate chunk size that fits in available VRAM."""
        per_expert = self._per_expert_vram_bytes()

        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            # Use at most 40% of free VRAM for expert buffer,
            # minus 100 MB for kernel intermediate allocations
            # (fused_marlin_moe allocates M * topk * max(2N, K) * 2 bytes)
            budget = int(free_vram * 0.4) - 100 * 1024 * 1024
            chunk = max(1, budget // per_expert)
            chunk = min(chunk, self.num_experts)
            logger.info(
                "Auto chunk_size: %d experts (%.1f MB each, %.1f MB budget of %.1f MB free)",
                chunk, per_expert / 1e6, budget / 1e6, free_vram / 1e6,
            )
            return chunk
        except Exception:
            # Fallback: 32 experts per chunk
            return min(32, self.num_experts)

    def _load_expert_biases(self):
        """Load GPT OSS expert biases from safetensors as GPU tensors.

        Only called when swiglu_limit > 0. Biases are small (~7.5 MB per layer
        for GPT OSS 120B) so we load them all at startup.

        Stores per layer: gate_up_bias [E, 2N] and down_bias [E, K] on GPU.
        Gate and up biases are de-interleaved from the original format and
        concatenated as [gate | up] to match the Marlin w1 output layout.
        """
        if self.swiglu_limit <= 0:
            return

        import safetensors.torch
        index_path = Path(self.model_path) / "model.safetensors.index.json"
        if not index_path.exists():
            logger.warning("No safetensors index for bias loading")
            return

        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Auto-detect prefix
        prefix = None
        for key in weight_map:
            if "mlp.experts" in key and "bias" in key:
                # e.g. "model.layers.0.mlp.experts.gate_up_proj_bias"
                idx = key.find(".layers.")
                if idx >= 0:
                    prefix = key[:idx]
                    break
        if prefix is None:
            logger.warning("Could not detect prefix for expert biases")
            return

        # Use safe_open to selectively load only bias tensors (avoids loading
        # entire 4 GB shards — biases spread across all 15 shards)
        from safetensors import safe_open
        shard_handles = {}  # shard_name -> safe_open handle

        loaded = 0
        for moe_idx in range(self._num_moe_layers):
            layer_idx = moe_idx + self.first_k_dense
            gu_name = f"{prefix}.layers.{layer_idx}.mlp.experts.gate_up_proj_bias"
            dn_name = f"{prefix}.layers.{layer_idx}.mlp.experts.down_proj_bias"

            gu_shard = weight_map.get(gu_name)
            dn_shard = weight_map.get(dn_name)
            if not gu_shard or not dn_shard:
                continue

            # Open shard handles lazily
            for sn in (gu_shard, dn_shard):
                if sn not in shard_handles:
                    shard_path = Path(self.model_path) / sn
                    if shard_path.exists():
                        shard_handles[sn] = safe_open(str(shard_path), framework="pt")
            if gu_shard not in shard_handles or dn_shard not in shard_handles:
                continue

            # gate_up_proj_bias: [n_experts, 2*intermediate] interleaved
            gu_bias_raw = shard_handles[gu_shard].get_tensor(gu_name).float()  # [E, 2N]
            # De-interleave: even columns = gate, odd columns = up
            gate_bias = gu_bias_raw[:, ::2]   # [E, N]
            up_bias = gu_bias_raw[:, 1::2]    # [E, N]
            # Concatenate as [gate | up] to match Marlin w1 layout
            gate_up_bias = torch.cat([gate_bias, up_bias], dim=1)  # [E, 2N]

            # down_proj_bias: [n_experts, hidden]
            down_bias = shard_handles[dn_shard].get_tensor(dn_name).float()  # [E, K]

            self._expert_biases[moe_idx] = {
                "gate_up_bias": gate_up_bias.to(self.params_dtype).to(self.device),
                "down_bias": down_bias.to(self.params_dtype).to(self.device),
            }
            loaded += 1

        # Close handles
        shard_handles.clear()

        if loaded > 0:
            size_mb = sum(
                b["gate_up_bias"].nbytes + b["down_bias"].nbytes
                for b in self._expert_biases.values()
            ) / 1e6
            logger.info("Loaded expert biases: %d layers, %.1f MB on %s", loaded, size_mb, self.device)

    def prepare_all_layers(self):
        """Pre-prepare all MoE layers at startup.

        Persistent mode: pre-load all experts into GPU VRAM.
        Selective/Chunked engine: no-op (weights read from engine on demand).
        Legacy mode: loads from disk cache.
        """
        # Load GPT OSS expert biases if needed
        self._load_expert_biases()

        if self._engine is not None:
            if self._prefill_mode == "persistent":
                self._preload_persistent()
                return
            elif self._engine.is_marlin_format:
                logger.info("Engine path: Marlin-native DMA copy (zero conversion, zero RAM cache)")
            else:
                logger.info("Engine path: GPU prefill uses on-the-fly Marlin repack (zero RAM cache)")
            return

        first_k = self.first_k_dense
        total_moe = self.num_experts  # This is num_experts, not num_moe_layers
        # We need the caller to tell us how many MoE layers there are.
        # Discover from disk cache:
        cache_dir = Path(self.model_path) / ".marlin_cache" / f"b{self.num_bits}"
        if not cache_dir.exists():
            logger.warning("No Marlin disk cache found at %s — layers will be prepared lazily", cache_dir)
            return

        prepared = 0
        failed = 0
        for cache_file in sorted(cache_dir.glob("layer*_routed.pt")):
            # Parse layer index from filename: layer{N}_routed.pt
            name = cache_file.stem  # e.g. "layer1_routed"
            try:
                abs_layer_idx = int(name.split("_")[0].replace("layer", ""))
                moe_layer_idx = abs_layer_idx - first_k
            except (ValueError, IndexError):
                continue

            if moe_layer_idx < 0:
                continue
            if moe_layer_idx in self._cache:
                prepared += 1
                continue

            disk_data = self._load_from_disk(moe_layer_idx)
            if disk_data is not None:
                self._cache[moe_layer_idx] = disk_data
                prepared += 1
            else:
                failed += 1

        # Also prepare shared experts
        shared_prepared = 0
        for cache_file in sorted(cache_dir.glob("layer*_shared.pt")):
            name = cache_file.stem
            try:
                abs_layer_idx = int(name.split("_")[0].replace("layer", ""))
                moe_layer_idx = abs_layer_idx - first_k
            except (ValueError, IndexError):
                continue

            if moe_layer_idx < 0:
                continue
            if moe_layer_idx in self._shared_cache:
                shared_prepared += 1
                continue

            disk_data = self._load_from_disk(moe_layer_idx, kind="shared")
            if disk_data is not None:
                self._shared_cache[moe_layer_idx] = disk_data
                shared_prepared += 1

        logger.info(
            "prepare_all_layers: %d routed + %d shared layers loaded from disk cache "
            "(%d failed) on %s",
            prepared, shared_prepared, failed, self.device,
        )

    def _ensure_index(self):
        """Load safetensors index and detect expert prefix (lazy)."""
        if self._weight_map is not None:
            return

        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self._weight_map = index["weight_map"]

        # Detect prefix
        for key in self._weight_map:
            if ".layers." in key and ".mlp.experts." in key:
                pos = key.index(".layers.")
                self._layers_prefix = key[:pos]
                break

        if self._layers_prefix is None:
            raise ValueError("Could not detect expert weight prefix")

        logger.info("Expert prefix: %s", self._layers_prefix)

    def _open_shard(self, shard_name: str):
        """Open a safetensors shard (cached)."""
        if shard_name not in self._shards:
            from safetensors.torch import load_file
            path = os.path.join(self.model_path, shard_name)
            self._shards[shard_name] = load_file(path, device="cpu")
        return self._shards[shard_name]

    def _read_tensor(self, tensor_name: str) -> torch.Tensor:
        """Read a tensor from safetensors by name."""
        shard_name = self._weight_map[tensor_name]
        shard = self._open_shard(shard_name)
        return shard[tensor_name]

    def _read_expert_weight_bf16(self, prefix: str, proj: str) -> torch.Tensor:
        """Read an expert weight, dequantizing if pre-quantized. Returns BF16 [N, K]."""
        weight_key = f"{prefix}.{proj}.weight"
        if weight_key in self._weight_map:
            return self._read_tensor(weight_key).to(torch.bfloat16)

        # Pre-quantized (compressed-tensors): dequantize
        packed = self._read_tensor(f"{prefix}.{proj}.weight_packed")
        scale = self._read_tensor(f"{prefix}.{proj}.weight_scale")
        shape = self._read_tensor(f"{prefix}.{proj}.weight_shape")

        N, K = int(shape[0]), int(shape[1])
        group_size = K // scale.shape[1]

        # Unpack INT4: each int32 holds 8 values
        unpacked = torch.zeros(N, K, dtype=torch.float32)
        for i in range(8):
            unpacked[:, i::8] = ((packed >> (4 * i)) & 0xF).float() - 8.0

        # Dequantize
        unpacked = unpacked.reshape(N, K // group_size, group_size)
        scale_f = scale.float().unsqueeze(2)
        w = (unpacked * scale_f).reshape(N, K)
        return w.to(torch.bfloat16)

    def _allocate_gpu_buffer(self):
        """Allocate GPU buffer for one chunk of experts."""
        if self._gpu_w13_packed is not None:
            return

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        E = self.chunk_size
        gs = self._group_size
        nb2 = self.num_bits // 2  # 2 for INT4, 4 for INT8

        # w13 (gate+up combined): [E, K//16, 2*N*nb2]
        self._gpu_w13_packed = torch.zeros(
            E, K // 16, 2 * N * nb2,
            dtype=torch.int32, device=self.device,
        )
        # w13 scale: [E, K//gs, 2*N]
        self._gpu_w13_scale = torch.zeros(
            E, K // gs, 2 * N,
            dtype=self.params_dtype, device=self.device,
        )
        # w2 (down): [E, N//16, K_w2*nb2] (K_w2 may be padded for Marlin compat)
        self._gpu_w2_packed = torch.zeros(
            E, N // 16, K_w2 * nb2,
            dtype=torch.int32, device=self.device,
        )
        # w2 scale: [E, N//gs, K_w2]
        self._gpu_w2_scale = torch.zeros(
            E, N // gs, K_w2,
            dtype=self.params_dtype, device=self.device,
        )
        # Empty g_idx and sort_indices (no activation ordering)
        self._gpu_g_idx = torch.empty(
            E, 0, dtype=torch.int32, device=self.device,
        )
        self._gpu_sort_idx = torch.empty(
            E, 0, dtype=torch.int32, device=self.device,
        )
        # Workspace
        self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        buf_mb = (
            self._gpu_w13_packed.nbytes + self._gpu_w13_scale.nbytes
            + self._gpu_w2_packed.nbytes + self._gpu_w2_scale.nbytes
        ) / 1e6
        logger.info("GPU buffer allocated: %.1f MB for %d experts", buf_mb, E)

    def _preload_persistent(self):
        """Pre-load ALL expert weights for ALL MoE layers into GPU VRAM.

        Called once at startup when expert_divisor=1.
        After this, forward() does zero DMA — just indexes into persistent buffers.
        """
        assert self._engine is not None, "Persistent mode requires Krasis engine"

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        torch.cuda.set_device(self.device)
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2
        E = self.num_experts

        self._persistent = {}
        self._persistent_shared = {}

        total_mb = 0
        load_start = time.perf_counter()

        try:
            for moe_idx in range(self._num_moe_layers):
                # Get all expert weights from Rust engine
                w13_packed_bytes = self._engine.get_expert_w13_packed(moe_idx, 0, E)
                w13_scales_bytes = self._engine.get_expert_w13_scales(moe_idx, 0, E)
                w2_packed_bytes = self._engine.get_expert_w2_packed(moe_idx, 0, E)
                w2_scales_bytes = self._engine.get_expert_w2_scales(moe_idx, 0, E)

                # Reshape and upload to GPU
                w13_packed = torch.frombuffer(
                    bytearray(w13_packed_bytes), dtype=torch.int32
                ).reshape(E, K // 16, 2 * N * nb2).to(self.device, non_blocking=True)
                w13_scale = torch.frombuffer(
                    bytearray(w13_scales_bytes), dtype=torch.bfloat16
                ).reshape(E, K // gs, 2 * N).to(self.device, non_blocking=True)
                w2_packed = torch.frombuffer(
                    bytearray(w2_packed_bytes), dtype=torch.int32
                ).reshape(E, N // 16, K_w2 * nb2).to(self.device, non_blocking=True)
                w2_scale = torch.frombuffer(
                    bytearray(w2_scales_bytes), dtype=torch.bfloat16
                ).reshape(E, N // gs, K_w2).to(self.device, non_blocking=True)

                layer_mb = (
                    w13_packed.nbytes + w13_scale.nbytes
                    + w2_packed.nbytes + w2_scale.nbytes
                ) / 1e6
                total_mb += layer_mb

                self._persistent[moe_idx] = {
                    "w13_packed": w13_packed,
                    "w13_scale": w13_scale,
                    "w2_packed": w2_packed,
                    "w2_scale": w2_scale,
                }

                # Free CPU-side byte copies
                del w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes

                if (moe_idx + 1) % 5 == 0 or moe_idx == self._num_moe_layers - 1:
                    logger.info(
                        "  Persistent preload: %d/%d layers (%.1f MB so far)",
                        moe_idx + 1, self._num_moe_layers, total_mb,
                    )

            # Pre-load shared experts
            if self.n_shared_experts > 0:
                shared_N = self.n_shared_experts * N
                for moe_idx in range(self._num_moe_layers):
                    w13p_bytes, w13s_bytes, w2p_bytes, w2s_bytes = (
                        self._engine.get_shared_expert_weights(moe_idx)
                    )

                    w13_packed = torch.frombuffer(
                        bytearray(w13p_bytes), dtype=torch.int32
                    ).reshape(1, K // 16, 2 * shared_N * nb2).to(self.device, non_blocking=True)
                    w13_scale = torch.frombuffer(
                        bytearray(w13s_bytes), dtype=torch.bfloat16
                    ).reshape(1, K // gs, 2 * shared_N).to(self.device, non_blocking=True)
                    w2_packed = torch.frombuffer(
                        bytearray(w2p_bytes), dtype=torch.int32
                    ).reshape(1, shared_N // 16, K_w2 * nb2).to(self.device, non_blocking=True)
                    w2_scale = torch.frombuffer(
                        bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(1, shared_N // gs, K_w2).to(self.device, non_blocking=True)

                    shared_mb = (
                        w13_packed.nbytes + w13_scale.nbytes
                        + w2_packed.nbytes + w2_scale.nbytes
                    ) / 1e6
                    total_mb += shared_mb

                    self._persistent_shared[moe_idx] = {
                        "w13_packed": w13_packed,
                        "w13_scale": w13_scale,
                        "w2_packed": w2_packed,
                        "w2_scale": w2_scale,
                    }

            # Allocate workspace (single, reused across layers)
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

            # Allocate g_idx and sort_idx for full expert count
            self._gpu_g_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)

            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - load_start
            alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
            logger.info(
                "Persistent mode ready: %d MoE layers × %d experts = %.1f MB "
                "in %.1fs (GPU total: %.1f MB)",
                self._num_moe_layers, E, total_mb, elapsed, alloc_mb,
            )

            # Free the chunk-sized GPU buffers (not needed in persistent mode)
            self._gpu_w13_packed = None
            self._gpu_w13_scale = None
            self._gpu_w2_packed = None
            self._gpu_w2_scale = None
            torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
            logger.warning(
                "OOM during persistent preload at layer %d — "
                "falling back to layer_grouped (divisor=2)",
                len(self._persistent),
            )
            # Clean up partial persistent data
            self._persistent = None
            self._persistent_shared = None
            torch.cuda.empty_cache()

            self._expert_divisor = 2
            self._prefill_mode = "layer_grouped"

    def _init_dma_buffers(self):
        """Pre-allocate CUDA-pinned CPU staging buffers for truly async DMA.

        Called once before first preload_layer_group(). Allocates pinned CPU
        tensors sized for the LOCAL slice of experts in one layer. Pinned
        memory enables truly async .to(device, non_blocking=True) — the CPU
        is NOT blocked during the PCIe transfer, unlike pageable memory where
        CUDA must first stage through an internal pinned buffer synchronously.

        Combined with Rust's allow_threads GIL release during memcpy, this
        allows DMA to fully overlap with GPU compute.
        """
        if self._dma_bufs_initialized:
            return

        E_local = self.num_local_experts
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # may be > K if padded for Marlin compat
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2

        # Compute sizes in bytes (matching tensor shapes in preload_layer_group)
        w13p_size = E_local * (K // 16) * (2 * N * nb2) * 4  # int32
        w13s_size = E_local * (K // gs) * (2 * N) * 2  # bf16
        w2p_size = E_local * (N // 16) * (K_w2 * nb2) * 4  # int32 (uses padded K)
        w2s_size = E_local * (N // gs) * K_w2 * 2  # bf16 (uses padded K)

        # Store sizes for pointer-based Rust API
        self._dma_size_w13p = w13p_size
        self._dma_size_w13s = w13s_size
        self._dma_size_w2p = w2p_size
        self._dma_size_w2s = w2s_size

        # Store tensor shapes for view() calls
        self._dma_shape_w13p = (E_local, K // 16, 2 * N * nb2)
        self._dma_shape_w13s = (E_local, K // gs, 2 * N)
        self._dma_shape_w2p = (E_local, N // 16, K_w2 * nb2)
        self._dma_shape_w2s = (E_local, N // gs, K_w2)

        total = w13p_size + w13s_size + w2p_size + w2s_size
        logger.info(
            "Pre-allocating pinned DMA buffers: %.1f MB "
            "(w13p=%.1f, w13s=%.1f, w2p=%.1f, w2s=%.1f)",
            total / 1e6, w13p_size / 1e6, w13s_size / 1e6,
            w2p_size / 1e6, w2s_size / 1e6,
        )

        t0 = time.perf_counter()
        # Pinned memory: pre-committed by CUDA, no page faults, truly async DMA
        self._dma_buf_w13p = torch.empty(w13p_size, dtype=torch.uint8, pin_memory=True)
        self._dma_buf_w13s = torch.empty(w13s_size, dtype=torch.uint8, pin_memory=True)
        self._dma_buf_w2p = torch.empty(w2p_size, dtype=torch.uint8, pin_memory=True)
        self._dma_buf_w2s = torch.empty(w2s_size, dtype=torch.uint8, pin_memory=True)

        elapsed = time.perf_counter() - t0
        self._dma_bufs_initialized = True
        self._dma_pinned = True
        logger.info(
            "Pinned DMA buffers allocated: %.1f MB in %.1fs (one-time cost)",
            total / 1e6, elapsed,
        )

        # Also allocate shared expert buffers if needed
        if self.n_shared_experts > 0:
            shared_N = self.n_shared_experts * N
            sw13p = 1 * (K // 16) * (2 * shared_N * nb2) * 4
            sw13s = 1 * (K // gs) * (2 * shared_N) * 2
            sw2p = 1 * (shared_N // 16) * (K_w2 * nb2) * 4
            sw2s = 1 * (shared_N // gs) * K_w2 * 2
            self._dma_shared_w13p = torch.empty(sw13p, dtype=torch.uint8, pin_memory=True)
            self._dma_shared_w13s = torch.empty(sw13s, dtype=torch.uint8, pin_memory=True)
            self._dma_shared_w2p = torch.empty(sw2p, dtype=torch.uint8, pin_memory=True)
            self._dma_shared_w2s = torch.empty(sw2s, dtype=torch.uint8, pin_memory=True)
            self._dma_shared_size_w13p = sw13p
            self._dma_shared_size_w13s = sw13s
            self._dma_shared_size_w2p = sw2p
            self._dma_shared_size_w2s = sw2s
            self._dma_shared_shape_w13p = (1, K // 16, 2 * shared_N * nb2)
            self._dma_shared_shape_w13s = (1, K // gs, 2 * shared_N)
            self._dma_shared_shape_w2p = (1, shared_N // 16, K_w2 * nb2)
            self._dma_shared_shape_w2s = (1, shared_N // gs, K_w2)
            self._dma_shared_initialized = True

    def preload_layer_group(self, moe_layer_indices):
        """Pre-load expert weights for a group of MoE layers into GPU VRAM.

        Called per-group during layer-grouped prefill. After processing all
        chunks through this group, call free_layer_group() to release VRAM.

        Uses zero-copy path: Rust writes directly into pre-allocated bytearrays
        (pages already mapped), then torch.frombuffer creates a zero-copy view,
        then .to(device) does PCIe DMA. No intermediate allocations.
        """
        # HCS M>1 prefill: use layer-grouped DMA to GPU instead of CPU cold path.
        # CPU AVX2 engine takes 5-8s/layer for M=10K vs 40ms GPU kernel.
        # DMA one layer's experts (~830 MB) into free VRAM, compute on GPU, free.
        # (preload_layer_group is only called during M>1 prefill, not M=1 decode)

        assert self._engine is not None, "Layer-grouped mode requires Krasis engine"
        detailed = TIMING.prefill

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        torch.cuda.set_device(self.device)

        # Pre-allocate pinned DMA staging buffers (one-time)
        self._init_dma_buffers()

        self._persistent = {}
        self._persistent_shared = {}

        total_mb = 0
        load_start = time.perf_counter()

        # Accumulators for detailed timing
        t_rust = 0.0
        t_frombuf = 0.0
        t_to_gpu = 0.0
        t_sync = 0.0
        t_shared = 0.0

        for moe_idx in moe_layer_indices:
            if detailed:
                t0 = time.perf_counter()

            # Pinned path: Rust writes into pinned tensor via raw pointer,
            # releases GIL during memcpy so other threads can run in parallel.
            self._engine.write_experts_range_into_pinned(
                moe_idx,
                self.expert_start,
                self.expert_end,
                self._dma_buf_w13p.data_ptr(), self._dma_size_w13p,
                self._dma_buf_w13s.data_ptr(), self._dma_size_w13s,
                self._dma_buf_w2p.data_ptr(), self._dma_size_w2p,
                self._dma_buf_w2s.data_ptr(), self._dma_size_w2s,
            )

            if detailed:
                t1 = time.perf_counter()
                t_rust += t1 - t0

            # View pinned tensors as typed+shaped (zero-copy, still pinned)
            w13_packed = self._dma_buf_w13p.view(torch.int32).reshape(self._dma_shape_w13p)
            w13_scale = self._dma_buf_w13s.view(torch.bfloat16).reshape(self._dma_shape_w13s)
            w2_packed = self._dma_buf_w2p.view(torch.int32).reshape(self._dma_shape_w2p)
            w2_scale = self._dma_buf_w2s.view(torch.bfloat16).reshape(self._dma_shape_w2s)

            if detailed:
                t2 = time.perf_counter()
                t_frombuf += t2 - t1

            # PCIe DMA: truly async from pinned memory — CPU returns immediately,
            # DMA proceeds in background on the GPU's copy engine.
            w13_packed = w13_packed.to(self.device, non_blocking=True)
            w13_scale = w13_scale.to(self.device, non_blocking=True)
            w2_packed = w2_packed.to(self.device, non_blocking=True)
            w2_scale = w2_scale.to(self.device, non_blocking=True)

            if detailed:
                t3 = time.perf_counter()
                t_to_gpu += t3 - t2

            layer_mb = (
                w13_packed.nbytes + w13_scale.nbytes
                + w2_packed.nbytes + w2_scale.nbytes
            ) / 1e6
            total_mb += layer_mb

            self._persistent[moe_idx] = {
                "w13_packed": w13_packed,
                "w13_scale": w13_scale,
                "w2_packed": w2_packed,
                "w2_scale": w2_scale,
            }

        # Pre-load shared experts
        if self.n_shared_experts > 0:
            if detailed:
                t_sh0 = time.perf_counter()

            for moe_idx in moe_layer_indices:
                if self._dma_shared_initialized:
                    # Pinned shared expert path
                    self._engine.write_shared_expert_into_pinned(
                        moe_idx,
                        self._dma_shared_w13p.data_ptr(), self._dma_shared_size_w13p,
                        self._dma_shared_w13s.data_ptr(), self._dma_shared_size_w13s,
                        self._dma_shared_w2p.data_ptr(), self._dma_shared_size_w2p,
                        self._dma_shared_w2s.data_ptr(), self._dma_shared_size_w2s,
                    )
                    w13_packed = self._dma_shared_w13p.view(torch.int32).reshape(
                        self._dma_shared_shape_w13p).to(self.device, non_blocking=True)
                    w13_scale = self._dma_shared_w13s.view(torch.bfloat16).reshape(
                        self._dma_shared_shape_w13s).to(self.device, non_blocking=True)
                    w2_packed = self._dma_shared_w2p.view(torch.int32).reshape(
                        self._dma_shared_shape_w2p).to(self.device, non_blocking=True)
                    w2_scale = self._dma_shared_w2s.view(torch.bfloat16).reshape(
                        self._dma_shared_shape_w2s).to(self.device, non_blocking=True)
                else:
                    # Fallback: allocating path for shared experts
                    w13p_bytes, w13s_bytes, w2p_bytes, w2s_bytes = (
                        self._engine.get_shared_expert_weights(moe_idx)
                    )
                    w13_packed = torch.frombuffer(
                        bytearray(w13p_bytes), dtype=torch.int32
                    ).reshape(self._dma_shared_shape_w13p).to(self.device, non_blocking=True)
                    w13_scale = torch.frombuffer(
                        bytearray(w13s_bytes), dtype=torch.bfloat16
                    ).reshape(self._dma_shared_shape_w13s).to(self.device, non_blocking=True)
                    w2_packed = torch.frombuffer(
                        bytearray(w2p_bytes), dtype=torch.int32
                    ).reshape(self._dma_shared_shape_w2p).to(self.device, non_blocking=True)
                    w2_scale = torch.frombuffer(
                        bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(self._dma_shared_shape_w2s).to(self.device, non_blocking=True)

                shared_mb = (
                    w13_packed.nbytes + w13_scale.nbytes
                    + w2_packed.nbytes + w2_scale.nbytes
                ) / 1e6
                total_mb += shared_mb

                self._persistent_shared[moe_idx] = {
                    "w13_packed": w13_packed,
                    "w13_scale": w13_scale,
                    "w2_packed": w2_packed,
                    "w2_scale": w2_scale,
                }

            if detailed:
                t_shared += time.perf_counter() - t_sh0

        # Allocate workspace and index tensors (once, reused across groups)
        if self._workspace is None:
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)
        if self._gpu_g_idx is None:
            self._gpu_g_idx = torch.empty(self.num_experts, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(self.num_experts, 0, dtype=torch.int32, device=self.device)

        if detailed:
            t_sync0 = time.perf_counter()
        torch.cuda.synchronize(self.device)
        if detailed:
            t_sync = time.perf_counter() - t_sync0

        elapsed = time.perf_counter() - load_start
        alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6

        if detailed:
            logger.info(
                "DMA-DETAIL L%s: memcpy=%.0fms frombuf=%.0fms "
                "to_gpu=%.0fms sync=%.0fms shared=%.0fms | total=%.0fms (%.1f MB)",
                moe_layer_indices,
                t_rust * 1000, t_frombuf * 1000,
                t_to_gpu * 1000, t_sync * 1000, t_shared * 1000,
                elapsed * 1000, total_mb,
            )
        else:
            logger.info(
                "Layer group loaded: %d MoE layers = %.1f MB in %.2fs (GPU total: %.1f MB)",
                len(moe_layer_indices), total_mb, elapsed, alloc_mb,
            )

    def free_layer_group(self, clear_cache=True):
        """Free layer group's expert weights from GPU VRAM.

        Args:
            clear_cache: If True (default), call torch.cuda.empty_cache().
                Set False during DMA pipelining to avoid the ~30ms overhead;
                the allocator reuses freed blocks for the next group naturally.
        """
        detailed = TIMING.prefill
        if detailed:
            t0 = time.perf_counter()

        self._persistent = {}
        self._persistent_shared = {}
        if clear_cache:
            torch.cuda.empty_cache()

        if detailed:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("DMA-FREE: %.0fms (dict clear%s)", elapsed,
                        " + empty_cache" if clear_cache else "")

    # ── DMA pipelining: overlap next group's DMA with current compute ──

    def preload_layer_group_async(self, moe_layer_indices):
        """Start async DMA loading for next layer group (pipelined prefetch).

        Same as preload_layer_group() but:
        - DMA transfers are queued on a dedicated prefetch CUDA stream
        - Results go to _prefetch_persistent / _prefetch_persistent_shared
        - Does NOT synchronize — call swap_prefetch() when data is needed

        The prefetch stream overlaps with compute on the default stream because
        PyTorch creates streams with cudaStreamNonBlocking.
        """
        assert self._engine is not None, "Async prefetch requires Krasis engine"

        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

        torch.cuda.set_device(self.device)
        stream = self._prefetch_stream

        self._init_dma_buffers()

        self._prefetch_persistent = {}
        self._prefetch_persistent_shared = {}

        for moe_idx in moe_layer_indices:
            # CPU: Rust writes into pinned staging buffers (GIL released during memcpy)
            self._engine.write_experts_range_into_pinned(
                moe_idx,
                self.expert_start,
                self.expert_end,
                self._dma_buf_w13p.data_ptr(), self._dma_size_w13p,
                self._dma_buf_w13s.data_ptr(), self._dma_size_w13s,
                self._dma_buf_w2p.data_ptr(), self._dma_size_w2p,
                self._dma_buf_w2s.data_ptr(), self._dma_size_w2s,
            )

            # View pinned tensors as typed+shaped (zero-copy, still pinned)
            w13_packed = self._dma_buf_w13p.view(torch.int32).reshape(self._dma_shape_w13p)
            w13_scale = self._dma_buf_w13s.view(torch.bfloat16).reshape(self._dma_shape_w13s)
            w2_packed = self._dma_buf_w2p.view(torch.int32).reshape(self._dma_shape_w2p)
            w2_scale = self._dma_buf_w2s.view(torch.bfloat16).reshape(self._dma_shape_w2s)

            # GPU: Queue truly async PCIe DMA on prefetch stream
            # (pinned memory → no blocking CPU-side staging copy)
            with torch.cuda.stream(stream):
                w13_packed = w13_packed.to(self.device, non_blocking=True)
                w13_scale = w13_scale.to(self.device, non_blocking=True)
                w2_packed = w2_packed.to(self.device, non_blocking=True)
                w2_scale = w2_scale.to(self.device, non_blocking=True)

            self._prefetch_persistent[moe_idx] = {
                "w13_packed": w13_packed,
                "w13_scale": w13_scale,
                "w2_packed": w2_packed,
                "w2_scale": w2_scale,
            }

        # Shared experts
        if self.n_shared_experts > 0:
            for moe_idx in moe_layer_indices:
                if self._dma_shared_initialized:
                    self._engine.write_shared_expert_into_pinned(
                        moe_idx,
                        self._dma_shared_w13p.data_ptr(), self._dma_shared_size_w13p,
                        self._dma_shared_w13s.data_ptr(), self._dma_shared_size_w13s,
                        self._dma_shared_w2p.data_ptr(), self._dma_shared_size_w2p,
                        self._dma_shared_w2s.data_ptr(), self._dma_shared_size_w2s,
                    )
                    w13_packed = self._dma_shared_w13p.view(torch.int32).reshape(
                        self._dma_shared_shape_w13p)
                    w13_scale = self._dma_shared_w13s.view(torch.bfloat16).reshape(
                        self._dma_shared_shape_w13s)
                    w2_packed = self._dma_shared_w2p.view(torch.int32).reshape(
                        self._dma_shared_shape_w2p)
                    w2_scale = self._dma_shared_w2s.view(torch.bfloat16).reshape(
                        self._dma_shared_shape_w2s)
                    with torch.cuda.stream(stream):
                        w13_packed = w13_packed.to(self.device, non_blocking=True)
                        w13_scale = w13_scale.to(self.device, non_blocking=True)
                        w2_packed = w2_packed.to(self.device, non_blocking=True)
                        w2_scale = w2_scale.to(self.device, non_blocking=True)
                else:
                    # Fallback: allocating path for shared experts
                    w13p_bytes, w13s_bytes, w2p_bytes, w2s_bytes = (
                        self._engine.get_shared_expert_weights(moe_idx)
                    )
                    w13_packed = torch.frombuffer(
                        bytearray(w13p_bytes), dtype=torch.int32
                    ).reshape(self._dma_shared_shape_w13p)
                    w13_scale = torch.frombuffer(
                        bytearray(w13s_bytes), dtype=torch.bfloat16
                    ).reshape(self._dma_shared_shape_w13s)
                    w2_packed = torch.frombuffer(
                        bytearray(w2p_bytes), dtype=torch.int32
                    ).reshape(self._dma_shared_shape_w2p)
                    w2_scale = torch.frombuffer(
                        bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(self._dma_shared_shape_w2s)
                    with torch.cuda.stream(stream):
                        w13_packed = w13_packed.to(self.device, non_blocking=True)
                        w13_scale = w13_scale.to(self.device, non_blocking=True)
                        w2_packed = w2_packed.to(self.device, non_blocking=True)
                        w2_scale = w2_scale.to(self.device, non_blocking=True)

                self._prefetch_persistent_shared[moe_idx] = {
                    "w13_packed": w13_packed,
                    "w13_scale": w13_scale,
                    "w2_packed": w2_packed,
                    "w2_scale": w2_scale,
                }

        # Workspace and index tensors (idempotent, already allocated after first sync load)
        if self._workspace is None:
            from sglang.srt.layers.quantization.marlin_utils import (
                marlin_make_workspace,
            )
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)
        if self._gpu_g_idx is None:
            self._gpu_g_idx = torch.empty(self.num_experts, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(self.num_experts, 0, dtype=torch.int32, device=self.device)

        # NO sync — DMA continues in background on prefetch stream

    def swap_prefetch(self):
        """Wait for async prefetch to complete and make prefetched data active.

        Synchronizes the prefetch stream, then swaps _prefetch_persistent into
        _persistent so that forward() can use the prefetched expert weights.
        """
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()
        self._persistent = self._prefetch_persistent
        self._persistent_shared = self._prefetch_persistent_shared
        self._prefetch_persistent = {}
        self._prefetch_persistent_shared = {}

    def _allocate_ao_buffer(self):
        """Allocate GPU buffer for active-only mode (full [E, ...] per layer)."""
        if self._ao_gpu_w13 is not None:
            return

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        E = self.num_experts
        gs = self._group_size
        nb2 = self.num_bits // 2

        self._ao_gpu_w13 = torch.zeros(
            E, K // 16, 2 * N * nb2, dtype=torch.int32, device=self.device,
        )
        self._ao_gpu_w13_scale = torch.zeros(
            E, K // gs, 2 * N, dtype=self.params_dtype, device=self.device,
        )
        self._ao_gpu_w2 = torch.zeros(
            E, N // 16, K_w2 * nb2, dtype=torch.int32, device=self.device,
        )
        self._ao_gpu_w2_scale = torch.zeros(
            E, N // gs, K_w2, dtype=self.params_dtype, device=self.device,
        )

        if self._gpu_g_idx is None:
            self._gpu_g_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
        if self._workspace is None:
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        # Pre-allocate decode-sized gating output (M=1) to avoid per-call CUDA alloc
        self._ao_gating_output_1 = torch.empty(1, E, device=self.device)

        buf_mb = (
            self._ao_gpu_w13.nbytes + self._ao_gpu_w13_scale.nbytes
            + self._ao_gpu_w2.nbytes + self._ao_gpu_w2_scale.nbytes
        ) / 1e6
        logger.info("Active-only buffer allocated: %.1f MB for %d expert slots", buf_mb, E)

    def _allocate_lru_buffer(self):
        """Allocate GPU buffer for LRU cache mode.

        Uses available VRAM to create as many expert slots as possible.
        Each slot stores one expert's weights for one layer.
        """
        if self._lru_gpu_w13 is not None:
            return

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2
        per_expert = self._per_expert_vram_bytes()

        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            # Reserve VRAM for KV cache + attention intermediates + kernel workspace.
            # KV cache needs ~100 KB/token/layer; for 10K tokens × 94 layers ≈ 1 GB.
            # Add 1 GB for intermediates/workspace → 2 GB total headroom.
            kv_headroom = 2 * 1024 * 1024 * 1024  # 2 GB for KV cache + intermediates
            budget = free_vram - kv_headroom
            if budget < per_expert * self.num_experts:
                # Not enough VRAM for even num_experts LRU slots after KV reservation;
                # reduce headroom to minimum (still risky for large prompts)
                budget = free_vram - 500 * 1024 * 1024
            cache_size = max(self.num_experts, budget // per_expert)
        except Exception:
            cache_size = self.num_experts * 2  # Fallback

        self._lru_cache_size = cache_size
        self._lru_gpu_w13 = torch.zeros(
            cache_size, K // 16, 2 * N * nb2, dtype=torch.int32, device=self.device,
        )
        self._lru_gpu_w13_scale = torch.zeros(
            cache_size, K // gs, 2 * N, dtype=self.params_dtype, device=self.device,
        )
        self._lru_gpu_w2 = torch.zeros(
            cache_size, N // 16, K_w2 * nb2, dtype=torch.int32, device=self.device,
        )
        self._lru_gpu_w2_scale = torch.zeros(
            cache_size, N // gs, K_w2, dtype=self.params_dtype, device=self.device,
        )

        # Shared index/workspace
        E = self.num_experts
        if self._gpu_g_idx is None:
            self._gpu_g_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
        if self._workspace is None:
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        buf_mb = (
            self._lru_gpu_w13.nbytes + self._lru_gpu_w13_scale.nbytes
            + self._lru_gpu_w2.nbytes + self._lru_gpu_w2_scale.nbytes
        ) / 1e6
        logger.info(
            "LRU cache buffer allocated: %.1f MB for %d expert slots (%.1f MB/slot)",
            buf_mb, cache_size, per_expert / 1e6,
        )

    def _forward_lru(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with LRU expert cache: cross-layer, cross-request caching.

        Maintains a fixed pool of GPU expert slots. Each slot caches one
        (layer, expert) pair's weights. LRU eviction when full.

        The active_only buffer [E, ...] is used as the "staging" buffer
        that fused_marlin_moe reads from. We copy from LRU cache → staging.
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        self._allocate_ao_buffer()  # Staging buffer
        self._allocate_lru_buffer()  # LRU cache

        M = x.shape[0]
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2
        E = self.num_experts

        # Zero the staging buffer (needed since we index by expert_id)
        # For M=1 decode, skip full buffer zeroing (kernel only reads active slots)
        if moe_layer_idx != self._ao_current_layer:
            if M > 1:
                self._ao_gpu_w13.zero_()
                self._ao_gpu_w13_scale.zero_()
                self._ao_gpu_w2.zero_()
                self._ao_gpu_w2_scale.zero_()
            self._ao_loaded_experts.clear()
            self._ao_current_layer = moe_layer_idx

        active_experts = torch.unique(topk_ids).tolist()

        # Update heatmap
        if self._heatmap is not None:
            for eid in active_experts:
                key = (moe_layer_idx, eid)
                self._heatmap[key] = self._heatmap.get(key, 0) + 1

        missing_from_staging = [eid for eid in active_experts if eid not in self._ao_loaded_experts]

        for eid in missing_from_staging:
            key = (moe_layer_idx, eid)
            self._lru_counter += 1

            if key in self._lru_map:
                # LRU hit: copy from LRU slot → staging buffer
                slot = self._lru_map[key]
                self._lru_access_time[slot] = self._lru_counter
                self._ao_gpu_w13[eid:eid+1].copy_(self._lru_gpu_w13[slot:slot+1])
                self._ao_gpu_w13_scale[eid:eid+1].copy_(self._lru_gpu_w13_scale[slot:slot+1])
                self._ao_gpu_w2[eid:eid+1].copy_(self._lru_gpu_w2[slot:slot+1])
                self._ao_gpu_w2_scale[eid:eid+1].copy_(self._lru_gpu_w2_scale[slot:slot+1])
                self._ao_cache_hits += 1
            elif key in self._pinned:
                # Pinned: copy from pin → staging (also treated as hit)
                pin = self._pinned[key]
                self._ao_gpu_w13[eid:eid+1].copy_(pin["w13"])
                self._ao_gpu_w13_scale[eid:eid+1].copy_(pin["w13_scale"])
                self._ao_gpu_w2[eid:eid+1].copy_(pin["w2"])
                self._ao_gpu_w2_scale[eid:eid+1].copy_(pin["w2_scale"])
                self._ao_cache_hits += 1
            else:
                # LRU miss: collect for batched DMA
                if not hasattr(self, '_lru_dma_pending'):
                    self._lru_dma_pending = []
                self._lru_dma_pending.append((eid, key))

            self._ao_loaded_experts.add(eid)

        # Batched DMA for all LRU misses
        if hasattr(self, '_lru_dma_pending') and self._lru_dma_pending:
            dma_start = time.perf_counter()
            dma_eids = [eid for eid, _ in self._lru_dma_pending]
            dma_keys = [key for _, key in self._lru_dma_pending]

            w13_bytes = self._engine.get_experts_batch(moe_layer_idx, dma_eids, "w13_packed")
            w13s_bytes = self._engine.get_experts_batch(moe_layer_idx, dma_eids, "w13_scales")
            w2_bytes = self._engine.get_experts_batch(moe_layer_idx, dma_eids, "w2_packed")
            w2s_bytes = self._engine.get_experts_batch(moe_layer_idx, dma_eids, "w2_scales")

            n_dma = len(dma_eids)
            w13_all = torch.frombuffer(bytearray(w13_bytes), dtype=torch.int32
                ).reshape(n_dma, K // 16, 2 * N * nb2)
            w13s_all = torch.frombuffer(bytearray(w13s_bytes), dtype=torch.bfloat16
                ).reshape(n_dma, K // gs, 2 * N)
            w2_all = torch.frombuffer(bytearray(w2_bytes), dtype=torch.int32
                ).reshape(n_dma, N // 16, K_w2 * nb2)
            w2s_all = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16
                ).reshape(n_dma, N // gs, K_w2)

            for i, (eid, key) in enumerate(self._lru_dma_pending):
                # Find an LRU slot to evict (or use empty)
                if len(self._lru_map) < self._lru_cache_size:
                    slot = len(self._lru_map)
                else:
                    evict_slot = min(
                        (s for s in self._lru_access_time
                         if self._lru_slot_key.get(s) not in self._pinned),
                        key=lambda s: self._lru_access_time[s],
                    )
                    slot = evict_slot
                    evict_key = self._lru_slot_key[slot]
                    del self._lru_map[evict_key]
                    del self._lru_slot_key[slot]

                # Store in LRU cache
                self._lru_gpu_w13[slot:slot+1].copy_(w13_all[i:i+1])
                self._lru_gpu_w13_scale[slot:slot+1].copy_(w13s_all[i:i+1])
                self._lru_gpu_w2[slot:slot+1].copy_(w2_all[i:i+1])
                self._lru_gpu_w2_scale[slot:slot+1].copy_(w2s_all[i:i+1])
                self._lru_map[key] = slot
                self._lru_slot_key[slot] = key
                self._lru_access_time[slot] = self._lru_counter

                # Copy to staging buffer
                self._ao_gpu_w13[eid:eid+1].copy_(self._lru_gpu_w13[slot:slot+1])
                self._ao_gpu_w13_scale[eid:eid+1].copy_(self._lru_gpu_w13_scale[slot:slot+1])
                self._ao_gpu_w2[eid:eid+1].copy_(self._lru_gpu_w2[slot:slot+1])
                self._ao_gpu_w2_scale[eid:eid+1].copy_(self._lru_gpu_w2_scale[slot:slot+1])

            self._ao_dma_time += time.perf_counter() - dma_start
            self._ao_dma_bytes += n_dma * self._per_expert_vram_bytes()
            self._ao_cache_misses += n_dma
            self._lru_dma_pending.clear()

        # Run kernel
        if M == 1 and self._ao_gating_output_1 is not None:
            gating_output = self._ao_gating_output_1
        else:
            gating_output = torch.empty(M, E, device=self.device)
        if M > 1:
            self._workspace.zero_()

        output = self._call_fused_moe(
            moe_layer_idx=moe_layer_idx,
            hidden_states=x,
            w1=self._ao_gpu_w13,
            w2=self._ao_gpu_w2,
            w1_scale=self._ao_gpu_w13_scale,
            w2_scale=self._ao_gpu_w2_scale,
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=E,
            g_idx1=self._gpu_g_idx,
            g_idx2=self._gpu_g_idx,
            sort_indices1=self._gpu_sort_idx,
            sort_indices2=self._gpu_sort_idx,
            workspace=self._workspace,
        ).to(x.dtype)

        return output

    def _forward_active_only(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with active-only DMA: load only activated experts per-layer.

        On layer change: zero the buffer and reset loaded expert tracking.
        For each call: determine active experts, DMA only missing ones, run kernel.
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        self._allocate_ao_buffer()

        M = x.shape[0]
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2
        E = self.num_experts

        # Layer changed — zero buffer and reset tracking
        # For M=1 decode, skip full buffer zeroing (kernel only reads active slots)
        if moe_layer_idx != self._ao_current_layer:
            if M > 1:
                self._ao_gpu_w13.zero_()
                self._ao_gpu_w13_scale.zero_()
                self._ao_gpu_w2.zero_()
                self._ao_gpu_w2_scale.zero_()
            self._ao_loaded_experts.clear()
            self._ao_current_layer = moe_layer_idx

        # Determine which experts are active in this call
        # For M=1 decode, avoid GPU torch.unique + sync — use CPU set instead
        if M == 1:
            active_experts = list(set(topk_ids[0].tolist()))
        else:
            active_experts = torch.unique(topk_ids).tolist()

        # Update heatmap for pinning strategies
        if self._heatmap is not None:
            for eid in active_experts:
                key = (moe_layer_idx, eid)
                self._heatmap[key] = self._heatmap.get(key, 0) + 1

        missing = [eid for eid in active_experts if eid not in self._ao_loaded_experts]

        # Track per-expert cache hits (experts already loaded)
        hits = len(active_experts) - len(missing)
        self._ao_cache_hits += hits
        self._ao_cache_misses += len(missing)

        if missing:
            dma_start = time.perf_counter()

            # Separate pinned (GPU→GPU) vs DMA (CPU→GPU) experts
            pinned_eids = []
            dma_eids = []
            for eid in missing:
                pin_key = (moe_layer_idx, eid)
                if pin_key in self._pinned:
                    pinned_eids.append(eid)
                else:
                    dma_eids.append(eid)

            # GPU→GPU copy from pinned buffers (batched via index_copy_)
            if pinned_eids:
                pin_idx = torch.tensor(pinned_eids, dtype=torch.long, device=self.device)
                pin_w13 = torch.cat([self._pinned[(moe_layer_idx, e)]["w13"] for e in pinned_eids])
                pin_w13s = torch.cat([self._pinned[(moe_layer_idx, e)]["w13_scale"] for e in pinned_eids])
                pin_w2 = torch.cat([self._pinned[(moe_layer_idx, e)]["w2"] for e in pinned_eids])
                pin_w2s = torch.cat([self._pinned[(moe_layer_idx, e)]["w2_scale"] for e in pinned_eids])
                self._ao_gpu_w13.index_copy_(0, pin_idx, pin_w13)
                self._ao_gpu_w13_scale.index_copy_(0, pin_idx, pin_w13s)
                self._ao_gpu_w2.index_copy_(0, pin_idx, pin_w2)
                self._ao_gpu_w2_scale.index_copy_(0, pin_idx, pin_w2s)
                self._ao_loaded_experts.update(pinned_eids)
                self._ao_cache_hits += len(pinned_eids)
                self._ao_cache_misses -= len(pinned_eids)

            # Batched CPU→GPU DMA: single Rust FFI call + contiguous upload + scatter
            if dma_eids:
                # Single FFI call returns all 4 weight types (saves 3 round-trips)
                w13_bytes, w13s_bytes, w2_bytes, w2s_bytes = \
                    self._engine.get_experts_all_batch(moe_layer_idx, dma_eids)

                n_dma = len(dma_eids)
                # Use bytes directly (no bytearray copy — saves ~15 MB/layer)
                w13_all = torch.frombuffer(w13_bytes, dtype=torch.int32
                    ).reshape(n_dma, K // 16, 2 * N * nb2)
                w13s_all = torch.frombuffer(w13s_bytes, dtype=torch.bfloat16
                    ).reshape(n_dma, K // gs, 2 * N)
                w2_all = torch.frombuffer(w2_bytes, dtype=torch.int32
                    ).reshape(n_dma, N // 16, K_w2 * nb2)
                w2s_all = torch.frombuffer(w2s_bytes, dtype=torch.bfloat16
                    ).reshape(n_dma, N // gs, K_w2)

                # Contiguous CPU→GPU upload + GPU-side scatter (replaces per-expert copy_)
                dma_idx = torch.tensor(dma_eids, dtype=torch.long, device=self.device)
                self._ao_gpu_w13.index_copy_(0, dma_idx, w13_all.to(self.device, non_blocking=True))
                self._ao_gpu_w13_scale.index_copy_(0, dma_idx, w13s_all.to(self.device, non_blocking=True))
                self._ao_gpu_w2.index_copy_(0, dma_idx, w2_all.to(self.device, non_blocking=True))
                self._ao_gpu_w2_scale.index_copy_(0, dma_idx, w2s_all.to(self.device, non_blocking=True))
                self._ao_loaded_experts.update(dma_eids)

            dma_elapsed = time.perf_counter() - dma_start
            dma_bytes = len(dma_eids) * self._per_expert_vram_bytes()
            self._ao_dma_time += dma_elapsed
            self._ao_dma_bytes += dma_bytes

        # Run fused_marlin_moe with full [E, ...] buffer
        # Reuse pre-allocated gating_output for M=1 decode to avoid CUDA alloc
        if M == 1 and self._ao_gating_output_1 is not None:
            gating_output = self._ao_gating_output_1
        else:
            gating_output = torch.empty(M, E, device=self.device)
        if M > 1:
            self._workspace.zero_()

        output = self._call_fused_moe(
            moe_layer_idx=moe_layer_idx,
            hidden_states=x,
            w1=self._ao_gpu_w13,
            w2=self._ao_gpu_w2,
            w1_scale=self._ao_gpu_w13_scale,
            w2_scale=self._ao_gpu_w2_scale,
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=E,
            g_idx1=self._gpu_g_idx,
            g_idx2=self._gpu_g_idx,
            sort_indices1=self._gpu_sort_idx,
            sort_indices2=self._gpu_sort_idx,
            workspace=self._workspace,
        ).to(x.dtype)

        return output

    # ── Compact per-layer decode buffers (cross-token expert caching) ──

    def _init_compact_decode(self):
        """Initialize compact decode mode by partitioning the AO buffer.

        Each MoE layer gets a small compact buffer (21 slots) carved from the
        AO buffer [num_experts, ...]. Experts persist across tokens within each
        layer's buffer, enabling cross-token caching.

        Zero extra VRAM: compact buffers are views into the existing AO buffer.
        """
        if self._compact_initialized:
            return

        self._allocate_ao_buffer()

        slots = self._compact_slots_per_layer
        self._compact_next_partition = 0
        self._compact_bufs.clear()
        self._compact_maps.clear()
        self._compact_reverse.clear()
        self._compact_lru.clear()
        self._compact_next_slot.clear()
        self._compact_counter = 0

        # Pre-allocate kernel helpers for compact buffer size
        self._compact_g_idx = torch.empty(
            slots, 0, dtype=torch.int32, device=self.device,
        )
        self._compact_sort_idx = torch.empty(
            slots, 0, dtype=torch.int32, device=self.device,
        )
        self._compact_gating_1 = torch.empty(1, slots, device=self.device)

        self._compact_initialized = True
        max_layers = self.num_experts // slots
        logger.info(
            "Compact decode initialized: %d slots/layer, max %d layers "
            "(reusing AO buffer, zero extra VRAM)",
            slots, max_layers,
        )

    def _get_or_create_compact_buf(self, moe_layer_idx: int):
        """Get compact buffer for a MoE layer, allocating from AO pool if needed.

        On first creation, pre-populates with pinned experts (GPU→GPU copy)
        so the first decode token already has warm cache entries.
        """
        if moe_layer_idx in self._compact_bufs:
            return self._compact_bufs[moe_layer_idx]

        slots = self._compact_slots_per_layer
        start = self._compact_next_partition * slots
        end = start + slots

        if end > self.num_experts:
            return None  # No room — caller should fall back to AO path

        buf = {
            "w13": self._ao_gpu_w13[start:end],
            "w13_scale": self._ao_gpu_w13_scale[start:end],
            "w2": self._ao_gpu_w2[start:end],
            "w2_scale": self._ao_gpu_w2_scale[start:end],
        }
        emap = {}
        rmap = {}
        lru = {}
        next_slot = 0

        # Pre-populate from pinned experts (GPU→GPU, instant)
        if self._pinned:
            for (layer, eid), pin_data in self._pinned.items():
                if layer != moe_layer_idx:
                    continue
                if next_slot >= slots:
                    break  # Buffer full
                buf["w13"][next_slot:next_slot+1].copy_(pin_data["w13"])
                buf["w13_scale"][next_slot:next_slot+1].copy_(pin_data["w13_scale"])
                buf["w2"][next_slot:next_slot+1].copy_(pin_data["w2"])
                buf["w2_scale"][next_slot:next_slot+1].copy_(pin_data["w2_scale"])
                emap[eid] = next_slot
                rmap[next_slot] = eid
                lru[next_slot] = 0  # Low LRU — evicted first if not accessed
                next_slot += 1

        self._compact_bufs[moe_layer_idx] = buf
        self._compact_maps[moe_layer_idx] = emap
        self._compact_reverse[moe_layer_idx] = rmap
        self._compact_lru[moe_layer_idx] = lru
        self._compact_next_slot[moe_layer_idx] = next_slot
        self._compact_next_partition += 1

        if next_slot > 0:
            logger.debug(
                "Compact L%d: pre-populated %d pinned experts into %d slots",
                moe_layer_idx, next_slot, slots,
            )
        return buf

    def _invalidate_compact(self):
        """Invalidate compact decode state (called when AO buffer is used for prefill)."""
        if not self._compact_initialized:
            return
        self._compact_bufs.clear()
        self._compact_maps.clear()
        self._compact_reverse.clear()
        self._compact_lru.clear()
        self._compact_next_slot.clear()
        self._compact_next_partition = 0
        self._compact_initialized = False

    def _forward_compact(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with per-layer compact expert buffers for M=1 decode.

        Each MoE layer maintains its own small buffer (~21 slots) carved from
        the AO buffer. Experts persist across tokens enabling cross-token
        caching. Only newly activated experts trigger DMA.

        Key advantage over _forward_active_only: the shared AO buffer clears
        loaded_experts on every layer change, so ALL experts need fresh DMA
        every token. Compact buffers are per-layer, so cached experts survive
        layer cycling and only changed experts need DMA.
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        self._init_compact_decode()

        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2
        slots = self._compact_slots_per_layer

        # Get or create compact buffer for this layer
        buf = self._get_or_create_compact_buf(moe_layer_idx)
        if buf is None:
            # Fallback: use regular active-only path (ran out of AO buffer partitions)
            return self._forward_active_only(moe_layer_idx, x, topk_ids, topk_weights)

        emap = self._compact_maps[moe_layer_idx]
        rmap = self._compact_reverse[moe_layer_idx]
        lru = self._compact_lru[moe_layer_idx]

        # Get active experts (CPU-side, avoids GPU sync)
        # Save full topk list for remapping later (avoids second .tolist())
        topk_list = topk_ids[0].tolist()
        active_experts = list(set(topk_list))

        # Update heatmap
        if self._heatmap is not None:
            for eid in active_experts:
                key = (moe_layer_idx, eid)
                self._heatmap[key] = self._heatmap.get(key, 0) + 1

        # Find missing experts (not in compact buffer for this layer)
        missing = [eid for eid in active_experts if eid not in emap]
        hits = len(active_experts) - len(missing)
        self._ao_cache_hits += hits
        self._ao_cache_misses += len(missing)

        # Update LRU counter
        self._compact_counter += 1
        counter = self._compact_counter

        if missing:
            dma_start = time.perf_counter()

            # Assign compact slots for missing experts (evict LRU if full)
            for eid in missing:
                next_empty = self._compact_next_slot[moe_layer_idx]
                if next_empty < slots:
                    slot = next_empty
                    self._compact_next_slot[moe_layer_idx] = next_empty + 1
                else:
                    # Evict LRU: find slot with oldest access, excluding active experts
                    active_slots = {emap[e] for e in active_experts if e in emap}
                    evict_slot = min(
                        (s for s in range(slots) if s not in active_slots),
                        key=lambda s: lru.get(s, 0),
                    )
                    slot = evict_slot
                    old_eid = rmap.pop(slot, None)
                    if old_eid is not None:
                        del emap[old_eid]

                emap[eid] = slot
                rmap[slot] = eid
                lru[slot] = counter

            # Update LRU for hit experts
            for eid in active_experts:
                if eid in emap and eid not in missing:
                    lru[emap[eid]] = counter

            # Separate pinned (GPU→GPU) vs DMA (CPU→GPU)
            pinned_eids = []
            dma_eids = []
            for eid in missing:
                if (moe_layer_idx, eid) in self._pinned:
                    pinned_eids.append(eid)
                else:
                    dma_eids.append(eid)

            # Copy pinned experts to compact slots (GPU→GPU)
            if pinned_eids:
                for eid in pinned_eids:
                    slot = emap[eid]
                    pin = self._pinned[(moe_layer_idx, eid)]
                    buf["w13"][slot:slot+1].copy_(pin["w13"])
                    buf["w13_scale"][slot:slot+1].copy_(pin["w13_scale"])
                    buf["w2"][slot:slot+1].copy_(pin["w2"])
                    buf["w2_scale"][slot:slot+1].copy_(pin["w2_scale"])
                self._ao_cache_hits += len(pinned_eids)
                self._ao_cache_misses -= len(pinned_eids)

            # Batched DMA for non-pinned missing experts
            if dma_eids:
                w13_bytes, w13s_bytes, w2_bytes, w2s_bytes = \
                    self._engine.get_experts_all_batch(moe_layer_idx, dma_eids)

                n_dma = len(dma_eids)
                w13_all = torch.frombuffer(w13_bytes, dtype=torch.int32
                    ).reshape(n_dma, K // 16, 2 * N * nb2)
                w13s_all = torch.frombuffer(w13s_bytes, dtype=torch.bfloat16
                    ).reshape(n_dma, K // gs, 2 * N)
                w2_all = torch.frombuffer(w2_bytes, dtype=torch.int32
                    ).reshape(n_dma, N // 16, K_w2 * nb2)
                w2s_all = torch.frombuffer(w2s_bytes, dtype=torch.bfloat16
                    ).reshape(n_dma, N // gs, K_w2)

                # Build slot indices and scatter into compact buffer
                dma_slots = torch.tensor(
                    [emap[eid] for eid in dma_eids],
                    dtype=torch.long, device=self.device,
                )
                buf["w13"].index_copy_(0, dma_slots,
                    w13_all.to(self.device, non_blocking=True))
                buf["w13_scale"].index_copy_(0, dma_slots,
                    w13s_all.to(self.device, non_blocking=True))
                buf["w2"].index_copy_(0, dma_slots,
                    w2_all.to(self.device, non_blocking=True))
                buf["w2_scale"].index_copy_(0, dma_slots,
                    w2s_all.to(self.device, non_blocking=True))

            dma_elapsed = time.perf_counter() - dma_start
            self._ao_dma_time += dma_elapsed
            self._ao_dma_bytes += len(dma_eids) * self._per_expert_vram_bytes()
        else:
            # All hits — just update LRU
            for eid in active_experts:
                lru[emap[eid]] = counter

        # Remap topk_ids: global expert IDs → local compact slot IDs
        # Build on CPU from saved topk_list (avoids GPU→CPU transfer)
        local_ids = torch.tensor(
            [[emap[eid] for eid in topk_list]],
            dtype=topk_ids.dtype, device=topk_ids.device,
        )

        # Run fused_marlin_moe with compact buffer
        # For GPT OSS compact: build remapped biases from emap
        compact_gu_bias = None
        compact_dn_bias = None
        if self.swiglu_limit > 0 and moe_layer_idx in self._expert_biases:
            full_b = self._expert_biases[moe_layer_idx]
            # Remap: for each local slot, get the global expert's bias
            reverse = self._compact_reverse[moe_layer_idx]  # local_slot → global_eid
            slot_eids = [reverse.get(s, 0) for s in range(slots)]
            idx = torch.tensor(slot_eids, dtype=torch.long)
            compact_gu_bias = full_b["gate_up_bias"][idx]
            compact_dn_bias = full_b["down_bias"][idx]

        output = self._call_fused_moe(
            moe_layer_idx=moe_layer_idx,
            hidden_states=x,
            w1=buf["w13"],
            w2=buf["w2"],
            w1_scale=buf["w13_scale"],
            w2_scale=buf["w2_scale"],
            gating_output=self._compact_gating_1,
            topk_weights=topk_weights,
            topk_ids=local_ids,
            global_num_experts=slots,
            g_idx1=self._compact_g_idx,
            g_idx2=self._compact_g_idx,
            sort_indices1=self._compact_sort_idx,
            sort_indices2=self._compact_sort_idx,
            workspace=self._workspace,
            bias_gate_up=compact_gu_bias,
            bias_down=compact_dn_bias,
        ).to(x.dtype)

        return output

    def reset_ao_stats(self):
        """Reset active-only statistics for a new request."""
        self._ao_dma_time = 0.0
        self._ao_dma_bytes = 0
        self._ao_cache_hits = 0
        self._ao_cache_misses = 0

    def get_ao_stats(self) -> dict:
        """Get active-only DMA statistics."""
        return {
            "dma_time": self._ao_dma_time,
            "dma_bytes": self._ao_dma_bytes,
            "cache_hits": self._ao_cache_hits,
            "cache_misses": self._ao_cache_misses,
            "pinned_experts": len(self._pinned),
        }

    def configure_pinning(self, budget_mb: float = 0, warmup_requests: int = 1,
                          strategy: str = "uniform"):
        """Configure static expert pinning.

        Args:
            budget_mb: VRAM budget for pinned experts (0 = auto-detect from free VRAM)
            warmup_requests: Number of requests before pinning activates
            strategy: "uniform" (equal per layer) or "weighted" (proportional to skew)
        """
        if budget_mb <= 0:
            try:
                free_vram = torch.cuda.mem_get_info(self.device)[0]
                budget_mb = (free_vram - 200 * 1024 * 1024) / 1e6  # Leave 200 MB headroom
            except Exception:
                budget_mb = 1000  # Fallback
        self._pin_budget_mb = budget_mb
        self._warmup_requests = warmup_requests
        self._pin_strategy = strategy
        per_expert_mb = self._per_expert_vram_bytes() / 1e6
        max_slots = int(budget_mb / per_expert_mb)
        logger.info(
            "Expert pinning configured: budget=%.1f MB, warmup=%d requests, "
            "strategy=%s, max_slots=%d (%.1f MB/expert)",
            budget_mb, warmup_requests, strategy, max_slots, per_expert_mb,
        )

    def notify_request_done(self):
        """Called after each request completes. Triggers pinning after warmup."""
        self._heatmap_requests += 1
        if self._heatmap_requests == self._warmup_requests and self._pin_budget_mb > 0:
            self._pin_hot_experts()

    def _pin_hot_experts(self):
        """Pin the hottest experts based on accumulated heatmap.

        Greedily assigns pinning slots to the (layer, expert) pairs with
        highest activation counts, respecting the VRAM budget.
        """
        if not self._heatmap:
            logger.warning("No heatmap data — skipping expert pinning")
            return

        per_expert_bytes = self._per_expert_vram_bytes()
        budget_bytes = int(self._pin_budget_mb * 1e6)
        max_slots = budget_bytes // per_expert_bytes

        strategy = getattr(self, '_pin_strategy', 'uniform')

        if strategy == "weighted":
            # Weighted pinning: allocate more slots to layers with higher skew
            self._pin_weighted(max_slots)
        else:
            # Uniform: just take globally hottest experts
            self._pin_uniform(max_slots)

    def _pin_uniform(self, max_slots: int):
        """Pin globally hottest (layer, expert) pairs."""
        # Sort by activation count descending
        sorted_experts = sorted(self._heatmap.items(), key=lambda x: x[1], reverse=True)

        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2

        # Free active_only buffer temporarily to make room for pinning
        had_ao_buffer = self._ao_gpu_w13 is not None
        if had_ao_buffer:
            self._ao_gpu_w13 = None
            self._ao_gpu_w13_scale = None
            self._ao_gpu_w2 = None
            self._ao_gpu_w2_scale = None
            self._ao_loaded_experts.clear()
            self._ao_current_layer = -1
            torch.cuda.empty_cache()

        # Recalculate budget from actual free VRAM
        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            per_expert = self._per_expert_vram_bytes()
            # Leave room for active_only buffer to be re-allocated
            ao_buffer_bytes = per_expert * self.num_experts if had_ao_buffer else 0
            # Reserve: AO buffer + 3.5GB for KV cache growth, attention intermediates,
            # kernel workspace, and PyTorch allocator fragmentation overhead
            headroom = int(3.5 * 1024 * 1024 * 1024)
            available = free_vram - ao_buffer_bytes - headroom
            max_slots = min(max_slots, max(0, available // per_expert))
            logger.info(
                "Pin budget recalculated: %.1f MB free, %d slots after reserving AO buffer + 3.5GB headroom",
                free_vram / 1e6, max_slots,
            )
        except Exception:
            pass

        pin_start = time.perf_counter()
        pinned = 0
        total_mb = 0

        for (layer_idx, expert_id), count in sorted_experts:
            if pinned >= max_slots:
                break

            try:
                w13_bytes = self._engine.get_expert_w13_packed(layer_idx, expert_id, expert_id + 1)
                w13s_bytes = self._engine.get_expert_w13_scales(layer_idx, expert_id, expert_id + 1)
                w2_bytes = self._engine.get_expert_w2_packed(layer_idx, expert_id, expert_id + 1)
                w2s_bytes = self._engine.get_expert_w2_scales(layer_idx, expert_id, expert_id + 1)

                w13 = torch.frombuffer(bytearray(w13_bytes), dtype=torch.int32
                    ).reshape(1, K // 16, 2 * N * nb2).to(self.device)
                w13s = torch.frombuffer(bytearray(w13s_bytes), dtype=torch.bfloat16
                    ).reshape(1, K // gs, 2 * N).to(self.device)
                w2 = torch.frombuffer(bytearray(w2_bytes), dtype=torch.int32
                    ).reshape(1, N // 16, K_w2 * nb2).to(self.device)
                w2s = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(1, N // gs, K_w2).to(self.device)

                self._pinned[(layer_idx, expert_id)] = {
                    "w13": w13, "w13_scale": w13s,
                    "w2": w2, "w2_scale": w2s,
                }
                pinned += 1
                total_mb += self._per_expert_vram_bytes() / 1e6
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                logger.warning("OOM during pinning at slot %d — stopping", pinned)
                torch.cuda.empty_cache()
                break

        # Re-allocate the active_only buffer
        if had_ao_buffer:
            self._allocate_ao_buffer()

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - pin_start

        # Log per-layer distribution
        layers_with_pins = {}
        for (layer_idx, _), _ in sorted_experts[:pinned]:
            layers_with_pins[layer_idx] = layers_with_pins.get(layer_idx, 0) + 1

        logger.info(
            "Expert pinning complete (uniform): %d experts pinned (%.1f MB) in %.2fs, "
            "spread across %d layers, top count=%d",
            pinned, total_mb, elapsed,
            len(layers_with_pins),
            sorted_experts[0][1] if sorted_experts else 0,
        )

    def _pin_weighted(self, max_slots: int):
        """Pin experts with per-layer budget proportional to activation skew.

        Layers with more skewed distributions (fewer experts activated) get more pins,
        since pinning has higher relative impact there.
        """
        # Compute per-layer stats
        layer_stats = {}  # layer -> {total_activations, unique_experts, max_count}
        for (layer_idx, expert_id), count in self._heatmap.items():
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {"total": 0, "experts": set(), "max": 0}
            layer_stats[layer_idx]["total"] += count
            layer_stats[layer_idx]["experts"].add(expert_id)
            layer_stats[layer_idx]["max"] = max(layer_stats[layer_idx]["max"], count)

        # Skew metric: max_count / (total / n_unique) = concentration ratio
        layer_skew = {}
        for layer_idx, stats in layer_stats.items():
            n_unique = len(stats["experts"])
            avg = stats["total"] / n_unique if n_unique > 0 else 1
            layer_skew[layer_idx] = stats["max"] / avg if avg > 0 else 1

        total_skew = sum(layer_skew.values())
        if total_skew == 0:
            self._pin_uniform(max_slots)
            return

        # Allocate per-layer budgets proportional to skew
        layer_budgets = {}
        for layer_idx, skew in layer_skew.items():
            layer_budgets[layer_idx] = max(1, int(max_slots * skew / total_skew))

        # Ensure total doesn't exceed budget
        total_alloc = sum(layer_budgets.values())
        if total_alloc > max_slots:
            scale = max_slots / total_alloc
            layer_budgets = {l: max(1, int(b * scale)) for l, b in layer_budgets.items()}

        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2

        # Free active_only buffer to make room
        had_ao_buffer = self._ao_gpu_w13 is not None
        if had_ao_buffer:
            self._ao_gpu_w13 = None
            self._ao_gpu_w13_scale = None
            self._ao_gpu_w2 = None
            self._ao_gpu_w2_scale = None
            self._ao_loaded_experts.clear()
            self._ao_current_layer = -1
            torch.cuda.empty_cache()

        # Recalculate slots from actual free VRAM
        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            per_expert = self._per_expert_vram_bytes()
            ao_buffer_bytes = per_expert * self.num_experts if had_ao_buffer else 0
            headroom = int(3.5 * 1024 * 1024 * 1024)
            available = free_vram - ao_buffer_bytes - headroom
            max_slots = min(max_slots, max(0, available // per_expert))
        except Exception:
            pass

        pin_start = time.perf_counter()
        pinned = 0
        total_mb = 0

        for layer_idx in sorted(layer_budgets.keys()):
            budget = layer_budgets[layer_idx]
            layer_experts = [
                (eid, count)
                for (l, eid), count in self._heatmap.items()
                if l == layer_idx
            ]
            layer_experts.sort(key=lambda x: x[1], reverse=True)

            for expert_id, count in layer_experts[:budget]:
                if pinned >= max_slots:
                    break
                try:
                    w13_bytes = self._engine.get_expert_w13_packed(layer_idx, expert_id, expert_id + 1)
                    w13s_bytes = self._engine.get_expert_w13_scales(layer_idx, expert_id, expert_id + 1)
                    w2_bytes = self._engine.get_expert_w2_packed(layer_idx, expert_id, expert_id + 1)
                    w2s_bytes = self._engine.get_expert_w2_scales(layer_idx, expert_id, expert_id + 1)

                    w13 = torch.frombuffer(bytearray(w13_bytes), dtype=torch.int32
                        ).reshape(1, K // 16, 2 * N * nb2).to(self.device)
                    w13s = torch.frombuffer(bytearray(w13s_bytes), dtype=torch.bfloat16
                        ).reshape(1, K // gs, 2 * N).to(self.device)
                    w2 = torch.frombuffer(bytearray(w2_bytes), dtype=torch.int32
                        ).reshape(1, N // 16, K_w2 * nb2).to(self.device)
                    w2s = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16
                        ).reshape(1, N // gs, K_w2).to(self.device)

                    self._pinned[(layer_idx, expert_id)] = {
                        "w13": w13, "w13_scale": w13s,
                        "w2": w2, "w2_scale": w2s,
                    }
                    pinned += 1
                    total_mb += self._per_expert_vram_bytes() / 1e6
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    logger.warning("OOM during weighted pinning at slot %d — stopping", pinned)
                    torch.cuda.empty_cache()
                    break

        # Re-allocate active_only buffer
        if had_ao_buffer:
            self._allocate_ao_buffer()

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - pin_start

        layers_with_pins = {}
        for (layer_idx, _) in self._pinned:
            layers_with_pins[layer_idx] = layers_with_pins.get(layer_idx, 0) + 1

        logger.info(
            "Expert pinning complete (weighted): %d experts pinned (%.1f MB) in %.2fs, "
            "spread across %d layers, min/max pins per layer: %d/%d",
            pinned, total_mb, elapsed,
            len(layers_with_pins),
            min(layers_with_pins.values()) if layers_with_pins else 0,
            max(layers_with_pins.values()) if layers_with_pins else 0,
        )

    # ── Hot-Cached-Static Strategy ──

    def save_heatmap(self, path: str):
        """Save accumulated expert activation heatmap to JSON file.

        Format: {"layer,expert": count, ...}
        Can be loaded later via --heatmap-path for hot_cached_static init.
        """
        if not self._heatmap:
            logger.warning("No heatmap data to save")
            return

        data = {}
        for (layer, expert), count in sorted(self._heatmap.items()):
            data[f"{layer},{expert}"] = count

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Heatmap saved: %d entries to %s", len(data), path)

    def clear_hcs(self):
        """Tear down all HCS state so the manager can be re-initialized."""
        # Release CUDA graphs
        self._hcs_cuda_graphs.clear()
        self._hcs_graph_io.clear()
        self._hcs_cuda_graphs_enabled = False

        # Release all device states (including per-device graphs)
        for hcs_dev in self._hcs_devices:
            hcs_dev.buffers.clear()
            hcs_dev.lookup.clear()
            hcs_dev.num_pinned.clear()
            hcs_dev.biases.clear()
            if hcs_dev.cuda_graphs:
                hcs_dev.cuda_graphs.clear()
            if hcs_dev.graph_io:
                hcs_dev.graph_io.clear()
            hcs_dev.workspace = None
            if hcs_dev.g_idx:
                hcs_dev.g_idx.clear()
            if hcs_dev.sort_idx:
                hcs_dev.sort_idx.clear()
            if hcs_dev.gating_1:
                hcs_dev.gating_1.clear()
            if hcs_dev.lookup_on_primary:
                hcs_dev.lookup_on_primary.clear()
            torch.cuda.synchronize(hcs_dev.device)
        self._hcs_devices.clear()

        # Clear shared state
        self._hcs_device_lookup.clear()
        self._pinned.clear()
        self._hcs_num_pinned.clear()
        self._hcs_buffers.clear()
        self._hcs_lookup.clear()
        self._hcs_g_idx.clear()
        self._hcs_sort_idx.clear()
        self._hcs_gating_1.clear()
        self._workspace = None
        self._hcs_initialized = False

        # Full CUDA reset
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("HCS state cleared.")

    def _init_hot_cached_static(self, heatmap_path: Optional[str] = None,
                                expert_budget_mb: Optional[int] = None,
                                allocation_mode: str = "uniform",
                                devices: Optional[list] = None,
                                device_budgets: Optional[dict] = None):
        """Initialize hot_cached_static strategy with a unified multi-GPU loop.

        Loads expert heatmap, selects hot experts that fit in combined VRAM
        budget across all GPUs, copies their weights to static per-layer GPU
        buffers, and builds lookup tables for multi-GPU/CPU split dispatch.

        Args:
            heatmap_path: Path to expert_heatmap.json. If None, uses
                          accumulated warmup heatmap from self._heatmap.
            expert_budget_mb: Max MB on primary GPU for expert weights.
                             If None, uses free VRAM minus 1 GB headroom.
            allocation_mode: "greedy" (globally hottest experts) or
                            "uniform" (equal experts per layer).
            devices: List of all torch.device objects to use.
            device_budgets: Dict of {device: budget_mb} for each device.
        """
        assert self._engine is not None, "hot_cached_static requires Krasis engine"

        if devices is None:
            devices = [self.device]
        if device_budgets is None:
            device_budgets = {}

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        # Load heatmap
        heatmap: dict[tuple[int, int], int] = {}
        if heatmap_path and os.path.exists(heatmap_path):
            with open(heatmap_path) as f:
                raw = json.load(f)
            for key_str, count in raw.items():
                parts = key_str.split(",")
                heatmap[(int(parts[0]), int(parts[1]))] = count
            logger.info("Loaded heatmap from %s: %d entries", heatmap_path, len(heatmap))
        elif self._heatmap:
            heatmap = dict(self._heatmap)
            logger.info("Using accumulated warmup heatmap: %d entries", len(heatmap))
        else:
            logger.warning("No heatmap for hot_cached_static — all experts on CPU")

        # Sort by activation count descending
        sorted_experts = sorted(heatmap.items(), key=lambda x: x[1], reverse=True)

        # ── Unified VRAM Budget Calculation ──
        per_expert = self._per_expert_vram_bytes()
        all_max_experts = []
        num_moe_layers = len(sorted_experts and {k[0] for k, _ in sorted_experts} or [])

        for i, device in enumerate(devices):
            # The device_budgets passed in from server.py should represent
            # the memory usable *for experts* on that device.
            expert_budget_bytes = device_budgets.get(device, 0) * 1024 * 1024
            all_max_experts.append(max(0, expert_budget_bytes // per_expert))
        
        total_max_experts = sum(all_max_experts)

        logger.info(
            "HCS unified VRAM budget: %d devices, %.1f MB/expert, "
            "per-device max experts: %s, total max: %d",
            len(devices), per_expert / 1e6,
            {str(d): n for d, n in zip(devices, all_max_experts)},
            total_max_experts,
        )

        # Select hot experts based on allocation mode
        if allocation_mode == "uniform":
            per_layer_heatmap: dict[int, list[tuple[int, int]]] = {}
            for (layer, expert), count in heatmap.items():
                per_layer_heatmap.setdefault(layer, []).append((expert, count))
            for layer in per_layer_heatmap:
                per_layer_heatmap[layer].sort(key=lambda x: x[1], reverse=True)
            num_layers_with_experts = len(per_layer_heatmap)
            per_layer_budget = total_max_experts // max(1, num_layers_with_experts)
            hot_experts = []
            for layer in sorted(per_layer_heatmap.keys()):
                for expert, count in per_layer_heatmap[layer][:per_layer_budget]:
                    hot_experts.append((layer, expert))
            logger.info(
                "HCS uniform allocation: %d experts/layer across %d layers = %d total",
                per_layer_budget, num_layers_with_experts, len(hot_experts),
            )
        else: # "greedy"
            hot_experts = [he[0] for he in sorted_experts[:total_max_experts]]

        if not hot_experts:
            logger.info("hot_cached_static: no experts pinned (all CPU)")
            self._hcs_initialized = True
            return

        # ── Distribute experts across devices per layer ──
        # Asymmetric HCS: Primary GPU (dev 0) gets globally hottest experts (L1).
        # Secondary GPUs get the next hottest experts (L2).
        
        # Current hot_experts are already sorted by heat.
        # Assign them to devices based on device capacities.
        layer_device_assignment: dict[int, list[tuple[int, list[int]]]] = {}
        
        device_remaining_cap = list(all_max_experts)
        for layer, expert in hot_experts:
            # Find first device with remaining capacity
            assigned = False
            for dev_idx in range(len(devices)):
                if device_remaining_cap[dev_idx] > 0:
                    device_remaining_cap[dev_idx] -= 1
                    
                    if layer not in layer_device_assignment:
                        layer_device_assignment[layer] = []
                    
                    # Add to existing assignment for this device if any
                    found_dev = False
                    for d_idx, eids in layer_device_assignment[layer]:
                        if d_idx == dev_idx:
                            eids.append(expert)
                            found_dev = True
                            break
                    if not found_dev:
                        layer_device_assignment[layer].append((dev_idx, [expert]))
                    
                    assigned = True
                    break
            if not assigned:
                break # All GPU budgets full

        # Verify per-device expert counts match budget
        dev_expert_counts = [0] * len(devices)
        for layer_idx, assignments in layer_device_assignment.items():
            for dev_idx, eids in assignments:
                dev_expert_counts[dev_idx] += len(eids)
        for i, count in enumerate(dev_expert_counts):
            logger.info("HCS assignment check: device %d (%s) → %d experts (budget: %d)",
                        i, devices[i], count, all_max_experts[i])

        # ── Unified Loop for Buffer Allocation and Weight Loading ──
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2

        total_mb = 0.0
        dev_loaded_mb = [0.0] * len(devices)
        dev_loaded_experts = [0] * len(devices)
        load_start = time.perf_counter()

        # Log pre-loading memory state for each device
        for i, device in enumerate(devices):
            alloc_mb = torch.cuda.memory_allocated(device) // (1024 * 1024)
            free_mb = torch.cuda.mem_get_info(device)[0] // (1024 * 1024)
            logger.info("HCS pre-load: %s allocated=%d MB, free=%d MB", device, alloc_mb, free_mb)

        # Create a unified list of device states
        self._hcs_devices: list[HcsDeviceState] = []
        for i, device in enumerate(devices):
            self._hcs_devices.append(HcsDeviceState(
                device=device,
                stream=torch.cuda.Stream(device=device),
                buffers={}, lookup={}, num_pinned={}, biases={},
                is_primary=(i==0),
                g_idx={},
                sort_idx={},
                gating_1={},
            ))

        for layer_idx in sorted(layer_device_assignment.keys()):
            assignments = layer_device_assignment[layer_idx]
            device_lookup = torch.full((self.num_experts,), -1, dtype=torch.int16, device=self.device)

            for dev_idx, eids in assignments:
                hcs_dev = self._hcs_devices[dev_idx]
                n = len(eids)

                w13_bytes, w13s_bytes, w2_bytes, w2s_bytes = \
                    self._engine.get_experts_all_batch(layer_idx, eids)

                w13_cpu = torch.frombuffer(bytearray(w13_bytes), dtype=torch.int32).reshape(n, K // 16, 2 * N * nb2)
                w13s_cpu = torch.frombuffer(bytearray(w13s_bytes), dtype=torch.bfloat16).reshape(n, K // gs, 2 * N)
                w2_cpu = torch.frombuffer(bytearray(w2_bytes), dtype=torch.int32).reshape(n, N // 16, K_w2 * nb2)
                w2s_cpu = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16).reshape(n, N // gs, K_w2)

                with torch.cuda.stream(hcs_dev.stream):
                    torch.cuda.set_device(hcs_dev.device)
                    if hcs_dev.is_primary:
                        # Primary device: upload Marlin format for sgl_kernel
                        w13 = w13_cpu.to(hcs_dev.device, non_blocking=True)
                        w13s = w13s_cpu.to(hcs_dev.device, non_blocking=True)
                        w2 = w2_cpu.to(hcs_dev.device, non_blocking=True)
                        w2s = w2s_cpu.to(hcs_dev.device, non_blocking=True)
                        hcs_dev.g_idx[layer_idx] = torch.empty(n, 0, dtype=torch.int32, device=hcs_dev.device)
                        hcs_dev.sort_idx[layer_idx] = torch.empty(n, 0, dtype=torch.int32, device=hcs_dev.device)
                        hcs_dev.gating_1[layer_idx] = torch.empty(1, n, device=hcs_dev.device)
                    else:
                        # Non-primary devices: keep Marlin format (same as primary).
                        # Each device runs its own fused_marlin_moe for per-GPU CUDA graphs.
                        w13 = w13_cpu.to(hcs_dev.device, non_blocking=True)
                        w13s = w13s_cpu.to(hcs_dev.device, non_blocking=True)
                        w2 = w2_cpu.to(hcs_dev.device, non_blocking=True)
                        w2s = w2s_cpu.to(hcs_dev.device, non_blocking=True)
                        hcs_dev.g_idx[layer_idx] = torch.empty(n, 0, dtype=torch.int32, device=hcs_dev.device)
                        hcs_dev.sort_idx[layer_idx] = torch.empty(n, 0, dtype=torch.int32, device=hcs_dev.device)
                        hcs_dev.gating_1[layer_idx] = torch.empty(1, n, device=hcs_dev.device)

                    layer_mb = (w13.nbytes + w13s.nbytes + w2.nbytes + w2s.nbytes) / 1e6
                    total_mb += layer_mb
                    dev_loaded_mb[dev_idx] += layer_mb
                    dev_loaded_experts[dev_idx] += n

                    hcs_dev.buffers[layer_idx] = {"w13": w13, "w13_scale": w13s, "w2": w2, "w2_scale": w2s}
                    local_lookup = torch.full((self.num_experts,), -1, dtype=torch.int32, device=hcs_dev.device)
                    for local_slot, eid in enumerate(eids):
                        local_lookup[eid] = local_slot
                        device_lookup[eid] = dev_idx
                    hcs_dev.lookup[layer_idx] = local_lookup
                    hcs_dev.num_pinned[layer_idx] = n

                    # For secondary devices, copy lookup to primary for fast dispatch
                    if not hcs_dev.is_primary:
                        if hcs_dev.lookup_on_primary is None:
                            hcs_dev.lookup_on_primary = {}
                        hcs_dev.lookup_on_primary[layer_idx] = local_lookup.to(self._hcs_devices[0].device)

                    if self.swiglu_limit > 0 and layer_idx in self._expert_biases:
                        full_biases = self._expert_biases[layer_idx]
                        eids_tensor = torch.tensor(eids, dtype=torch.long)
                        hcs_dev.biases[layer_idx] = {
                            "gate_up_bias": full_biases["gate_up_bias"][eids_tensor].to(hcs_dev.device),
                            "down_bias": full_biases["down_bias"][eids_tensor].to(hcs_dev.device),
                        }

                del w13_cpu, w13s_cpu, w2_cpu, w2s_cpu
            self._hcs_device_lookup[layer_idx] = device_lookup
            # Memory tracking: log every 6 layers to diagnose OOM
            if layer_idx % 6 == 0 or layer_idx == max(layer_device_assignment.keys()):
                mem_parts = []
                for di, device in enumerate(devices):
                    alloc_mb = torch.cuda.memory_allocated(device) // (1024 * 1024)
                    free_mb = torch.cuda.mem_get_info(device)[0] // (1024 * 1024)
                    mem_parts.append(f"dev{di}: alloc={alloc_mb}MB free={free_mb}MB loaded={dev_loaded_mb[di]:.0f}MB({dev_loaded_experts[di]}exp)")
                logger.info("HCS load L%d: total=%.0f MB | %s",
                            layer_idx, total_mb, " | ".join(mem_parts))

        # Synchronize all devices and log results
        for hcs_dev in self._hcs_devices:
            hcs_dev.stream.synchronize()
        elapsed = time.perf_counter() - load_start

        # Populate legacy dicts from primary device for backward compat
        # (validate_gpu_allocation, CUDA graphs, OOM retry all use these)
        primary_dev_state = self._hcs_devices[0]
        for layer_idx, bufs in primary_dev_state.buffers.items():
            lookup = primary_dev_state.lookup[layer_idx]
            n_pinned = primary_dev_state.num_pinned.get(layer_idx, 0)
            self._hcs_num_pinned[layer_idx] = n_pinned
            self._hcs_buffers[layer_idx] = bufs
            self._hcs_lookup[layer_idx] = lookup
            if primary_dev_state.g_idx:
                self._hcs_g_idx[layer_idx] = primary_dev_state.g_idx.get(layer_idx, torch.empty(n_pinned, 0, dtype=torch.int32, device=self.device))
            if primary_dev_state.sort_idx:
                self._hcs_sort_idx[layer_idx] = primary_dev_state.sort_idx.get(layer_idx, torch.empty(n_pinned, 0, dtype=torch.int32, device=self.device))
            if primary_dev_state.gating_1:
                self._hcs_gating_1[layer_idx] = primary_dev_state.gating_1.get(layer_idx, torch.empty(1, n_pinned, device=self.device))
            for eid in range(self.num_experts):
                slot = int(lookup[eid])
                if slot >= 0:
                    self._pinned[(layer_idx, eid)] = {
                        "w13": bufs["w13"][slot:slot+1],
                        "w13_scale": bufs["w13_scale"][slot:slot+1],
                        "w2": bufs["w2"][slot:slot+1],
                        "w2_scale": bufs["w2_scale"][slot:slot+1],
                    }

        device_counts = {str(d.device): sum(d.num_pinned.values()) for d in self._hcs_devices}
        total_pinned = sum(device_counts.values())

        logger.info(
            "hot_cached_static initialized: %d experts across %d devices, "
            "%.1f MB in %.2fs. Counts: %s",
            total_pinned, len(devices), total_mb, elapsed, device_counts
        )

        if self._workspace is None:
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        # Initialize per-device workspaces and graph state
        for hcs_dev in self._hcs_devices:
            hcs_dev.cuda_graphs = {}
            hcs_dev.graph_io = {}
            if hcs_dev.workspace is None:
                hcs_dev.workspace = marlin_make_workspace(hcs_dev.device, max_blocks_per_sm=4)

        self._hcs_initialized = True
        # CUDA graphs: single-GPU uses legacy path, multi-GPU uses per-device graphs
        if len(self._hcs_devices) == 1 and self._hcs_devices[0].device == self.device:
            self._init_cuda_graphs()
        elif len(self._hcs_devices) > 1:
            self._init_cuda_graphs_multi_gpu()
        else:
            logger.info("CUDA graphs skipped: manager on %s, HCS device on %s",
                        self.device, self._hcs_devices[0].device if self._hcs_devices else "none")

    def validate_gpu_allocation(self, model, test_tokens) -> tuple:
        """Run 10K-token inference to validate GPU allocation.

        Returns (ok, info) with inference_cost_bytes measured on success.
        """
        n_experts = sum(self._hcs_num_pinned.values()) if self._hcs_num_pinned else 0
        free_after_load = torch.cuda.mem_get_info(self.device)[0]

        model._oom_retry_enabled = False
        model._validation_sync = True  # Per-layer CUDA sync to pinpoint errors
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            with torch.inference_mode():
                model.generate(test_tokens, max_new_tokens=2, temperature=0.6)

            peak = torch.cuda.max_memory_allocated(self.device)
            baseline = torch.cuda.memory_allocated(self.device)
            inference_cost = peak - baseline

            logger.info(
                "GPU validation PASS: %d experts, %d MB free after load, "
                "inference cost %d MB (peak %d MB, baseline %d MB)",
                n_experts, free_after_load // (1024*1024),
                inference_cost // (1024*1024), peak // (1024*1024),
                baseline // (1024*1024),
            )
            return True, {
                "n_experts": n_experts,
                "free_after_load_mb": free_after_load // (1024*1024),
                "inference_cost_bytes": inference_cost,
            }
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError, torch.AcceleratorError) as e:
            err_str = str(e).lower()
            if ("out of memory" in err_str
                    or isinstance(e, (torch.cuda.OutOfMemoryError,
                                      torch.OutOfMemoryError))):
                # Log per-GPU VRAM to identify which device OOMed
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    alloc = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    logger.warning(
                        "GPU validation OOM diag: cuda:%d — free=%d MB, alloc=%d MB, "
                        "reserved=%d MB, total=%d MB",
                        i, free // (1024*1024), alloc // (1024*1024),
                        reserved // (1024*1024), total // (1024*1024),
                    )
                logger.warning(
                    "GPU validation FAIL: %d experts, %d MB free after load — OOM: %s",
                    n_experts, free_after_load // (1024*1024), e,
                )
                logger.warning("OOM traceback:\n%s", traceback.format_exc())
                gc.collect()
                torch.cuda.empty_cache()
                return False, {"n_experts": n_experts}
            if "illegal memory access" in err_str or "cuda error" in err_str:
                logger.error(
                    "GPU validation FAIL: CUDA error during inference. "
                    "Run with CUDA_LAUNCH_BLOCKING=1 and KRASIS_DEBUG_SYNC=1 "
                    "for detailed error location. Error: %s", e,
                )
                raise
            raise
        finally:
            model._oom_retry_enabled = True
            model._validation_sync = False

    def reset_prefill_stats(self):
        """Reset prefill timing accumulators (call before each prefill request)."""
        self._pf_cpu_submit_ms = 0.0
        self._pf_gpu_ms = 0.0
        self._pf_cpu_sync_ms = 0.0
        self._pf_cpu_to_gpu_ms = 0.0
        self._pf_combine_ms = 0.0
        self._pf_cpu_only_ms = 0.0
        self._pf_layers_total = 0
        self._pf_layers_cold = 0
        self._pf_layers_all_cold = 0
        self._pf_cold_experts = 0
        self._pf_total_experts = 0

    def log_prefill_summary(self, chunk_idx: int = -1):
        """Log accumulated prefill timing summary."""
        total = (self._pf_cpu_submit_ms + self._pf_gpu_ms + self._pf_cpu_sync_ms
                 + self._pf_cpu_to_gpu_ms + self._pf_combine_ms + self._pf_cpu_only_ms)
        if self._pf_layers_total == 0:
            return
        cold_pct = (100.0 * self._pf_cold_experts / max(self._pf_total_experts, 1))
        label = f"chunk={chunk_idx}" if chunk_idx >= 0 else "total"
        logger.info(
            "HCS-PREFILL-SUMMARY %s: %d layers (%d cold, %d all-cold) | "
            "cpu_submit=%.0fms gpu=%.0fms cpu_sync=%.0fms cpu_to_gpu=%.0fms "
            "combine=%.0fms cpu_only=%.0fms | total=%.0fms | "
            "cold=%d/%d(%.0f%%)",
            label, self._pf_layers_total, self._pf_layers_cold,
            self._pf_layers_all_cold,
            self._pf_cpu_submit_ms, self._pf_gpu_ms,
            self._pf_cpu_sync_ms, self._pf_cpu_to_gpu_ms,
            self._pf_combine_ms, self._pf_cpu_only_ms,
            total, self._pf_cold_experts, self._pf_total_experts, cold_pct,
        )

    def _log_timing_summary(self, prefix, moe_layer_idx, M, t_start, t_gpu_done,
                            t_cpu_submitted, t_cpu_sync_start, t_cpu_sync_done,
                            t_cpu_to_gpu, has_cold, lookup_result):
        """Helper to log detailed timing for HCS forward passes."""
        t_end = time.perf_counter()
        gpu_ms = (t_gpu_done - (t_cpu_submitted if has_cold else t_start)) * 1000

        if M == 1:
            n_cold = int((lookup_result < 0).sum().item())
            n_total = lookup_result.numel()
        else:
            n_cold = int((lookup_result < 0).sum().item())
            n_total = lookup_result.numel()

        submit_ms = (t_cpu_submitted - t_start) * 1000 if has_cold else 0
        sync_ms = (t_cpu_sync_done - t_cpu_sync_start) * 1000 if has_cold else 0
        togpu_ms = (t_cpu_to_gpu - t_cpu_sync_done) * 1000 if has_cold else 0
        combine_ms = (t_end - t_cpu_to_gpu) * 1000 if has_cold else 0

        if has_cold:
            logger.info(
                "%s L%d M=%d: cpu_submit=%.1fms gpu=%.1fms cpu_sync=%.1fms cpu_to_gpu=%.1fms "
                "combine=%.1fms total=%.1fms cold=%d/%d(%.0f%%)",
                prefix, moe_layer_idx, M,
                submit_ms, gpu_ms, sync_ms, togpu_ms, combine_ms,
                (t_end - t_start) * 1000,
                n_cold, n_total, 100.0 * n_cold / max(n_total, 1),
            )
        else:
            logger.info(
                "%s L%d M=%d: gpu_only=%.1fms total=%.1fms cold=0/%d",
                prefix, moe_layer_idx, M, gpu_ms, (t_end - t_start) * 1000, n_total,
            )

        # Accumulate for summary
        self._pf_cpu_submit_ms += submit_ms
        self._pf_gpu_ms += gpu_ms
        self._pf_cpu_sync_ms += sync_ms
        self._pf_cpu_to_gpu_ms += togpu_ms
        self._pf_combine_ms += combine_ms
        self._pf_layers_total += 1
        if has_cold:
            self._pf_layers_cold += 1
        self._pf_cold_experts += n_cold
        self._pf_total_experts += n_total


    def _forward_hot_cached_static(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with hot_cached_static: hot experts across multiple GPUs, cold on CPU.

        For M=1, dispatches to the unified decode function.
        For M>1, uses the primary GPU and falls back to CPU for cold experts.
        """
        M = x.shape[0]
        timing = (TIMING.decode and M == 1) or (TIMING.prefill and M > 1)

        if timing:
            t_start = time.perf_counter()

        device_lookup = self._hcs_device_lookup.get(moe_layer_idx)
        if device_lookup is None:
            return self._forward_hcs_cpu_only(moe_layer_idx, x, topk_ids, topk_weights)

        # Check if any GPU has experts for this layer
        any_gpu_experts = False
        for hcs_dev in self._hcs_devices:
            if hcs_dev.num_pinned.get(moe_layer_idx, 0) > 0:
                any_gpu_experts = True
                break
        if not any_gpu_experts:
            return self._forward_hcs_cpu_only(moe_layer_idx, x, topk_ids, topk_weights)

        # ── M=1 decode ──
        if M == 1:
            if self._hcs_cuda_graphs_enabled:
                # Single-GPU with CUDA graphs: fast graphed path (legacy dicts)
                if len(self._hcs_devices) == 1 and moe_layer_idx in self._hcs_cuda_graphs:
                    return self._forward_hcs_graphed_decode(
                        moe_layer_idx, x, topk_ids, topk_weights, timing,
                        t_start if timing else 0,
                    )
                # Multi-GPU with per-device CUDA graphs
                elif len(self._hcs_devices) > 1:
                    return self._forward_hcs_multigpu_graphed_decode(
                        moe_layer_idx, x, topk_ids, topk_weights, timing,
                        t_start if timing else 0,
                    )
            # Fallback: unified decode with Sticky L2
            return self._forward_hcs_unified_decode(
                moe_layer_idx, x, topk_ids, topk_weights, timing,
                t_start if timing else 0,
            )

        # ── M>1 prefill: uses primary GPU + CPU only ──
        primary_dev_state = self._hcs_devices[0]
        lookup = primary_dev_state.lookup.get(moe_layer_idx)
        n_hot = primary_dev_state.num_pinned.get(moe_layer_idx, 0)
        bufs = primary_dev_state.buffers.get(moe_layer_idx)

        if n_hot == 0 or bufs is None:
            return self._forward_hcs_cpu_only(moe_layer_idx, x, topk_ids, topk_weights)

        lookup_result = lookup[topk_ids.long()]
        has_cold = (lookup_result < 0).any().item()
        if (lookup_result < 0).all().item():
            return self._forward_hcs_cpu_only(moe_layer_idx, x, topk_ids, topk_weights)

        # Submit cold experts to CPU
        if has_cold:
            if timing: t_serialize_start = time.perf_counter()
            cpu_ids = topk_ids.clone()
            cpu_ids[lookup_result >= 0] = -1
            act_cpu = x.detach().cpu().contiguous()
            ids_cpu = cpu_ids.to(torch.int32).cpu().contiguous()
            wts_cpu = topk_weights.detach().cpu().contiguous()
            act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
            ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
            wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()
            if timing: t_tobytes = time.perf_counter()
            self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, M)

        if timing:
            t_cpu_submitted = time.perf_counter()

        # Primary GPU computation for hot experts
        cold_mask = lookup_result < 0
        local_ids = lookup_result.clone().to(topk_ids.dtype)
        local_ids[cold_mask] = 0
        gpu_topk_weights = topk_weights.clone()
        gpu_topk_weights[cold_mask] = 0.0
        gating = torch.empty(M, n_hot, device=self.device)

        self._workspace.zero_()
        hcs_bias = primary_dev_state.biases.get(moe_layer_idx, {})
        gpu_output = self._call_fused_moe(
            moe_layer_idx=moe_layer_idx, hidden_states=x,
            w1=bufs["w13"], w2=bufs["w2"],
            w1_scale=bufs["w13_scale"], w2_scale=bufs["w2_scale"],
            gating_output=gating, topk_weights=gpu_topk_weights, topk_ids=local_ids,
            global_num_experts=n_hot,
            g_idx1=primary_dev_state.g_idx[moe_layer_idx], g_idx2=primary_dev_state.g_idx[moe_layer_idx],
            sort_indices1=primary_dev_state.sort_idx[moe_layer_idx], sort_indices2=primary_dev_state.sort_idx[moe_layer_idx],
            workspace=self._workspace,
            bias_gate_up=hcs_bias.get("gate_up_bias"), bias_down=hcs_bias.get("down_bias"),
        ).to(x.dtype)

        if timing:
            torch.cuda.synchronize(self.device)
            t_gpu_done = time.perf_counter()

        # Combine with CPU result
        if has_cold:
            if timing: t_cpu_sync_start = time.perf_counter()
            cpu_output_bytes = self._engine.sync_forward()
            if timing: t_cpu_sync_done = time.perf_counter()
            cpu_raw = torch.frombuffer(bytearray(cpu_output_bytes), dtype=torch.bfloat16).reshape(M, self.hidden_size).to(self.device)
            if timing: t_cpu_to_gpu = time.perf_counter()
            if self.routed_scaling_factor != 1.0:
                cpu_raw /= self.routed_scaling_factor
            gpu_output += cpu_raw

        if timing:
            self._log_timing_summary(
                "HCS-PREFILL", moe_layer_idx, M, t_start, t_gpu_done,
                t_cpu_submitted if has_cold else 0,
                t_cpu_sync_start if has_cold else 0,
                t_cpu_sync_done if has_cold else 0,
                t_cpu_to_gpu if has_cold else 0,
                has_cold, lookup_result
            )

        if self._heatmap is not None:
            for eid in torch.unique(topk_ids).tolist():
                self._heatmap[(moe_layer_idx, eid)] = self._heatmap.get((moe_layer_idx, eid), 0) + 1

        return gpu_output

    def _forward_hcs_unified_decode(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        timing: bool,
        t_start: float,
    ) -> torch.Tensor:
        """Unified multi-GPU M=1 decode dispatch with Sticky L2 Cache.

        Classifies experts by device. Experts on secondary GPUs are streamed to
        the primary GPU's Active Buffer (AO buffer) once and kept there ("Sticky").
        All hot experts (L1 and L2) are then processed in a single Marlin kernel
        call on GPU 0.
        """
        K = self.hidden_size
        device_lookup = self._hcs_device_lookup[moe_layer_idx]
        topk_list = topk_ids[0].tolist()
        weights_list = topk_weights[0].tolist()

        # Classify each expert by its target device index.
        # L2 experts on secondary GPUs now use Marlin format (same as primary),
        # so they can stream directly into the AO buffer.
        dev_indices = [int(device_lookup[eid]) for eid in topk_list]
        has_cold = any(d < 0 for d in dev_indices)

        # ── Step 1: Submit cold experts to CPU engine (async) ──
        if has_cold:
            cpu_ids_list = [eid if dev_indices[j] < 0 else -1 for j, eid in enumerate(topk_list)]
            cpu_ids = torch.tensor([cpu_ids_list], dtype=torch.int32)
            act_cpu = x.detach().cpu().contiguous()
            ids_cpu = cpu_ids.cpu().contiguous()
            wts_cpu = topk_weights.detach().cpu().contiguous()
            act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
            ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
            wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()
            self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, 1)

        if timing:
            t_cpu_submitted = time.perf_counter()

        # ── Step 2: Manage Sticky L2 Cache on Primary GPU ──
        self._allocate_ao_buffer()
        torch.cuda.set_device(self.device) # Primary device
        
        # Track L2 sticky state
        if not hasattr(self, '_l2_sticky_map'):
            self._l2_sticky_map = set() # (layer, expert)
        
        # Ensure L1 experts for this layer are in AO buffer (one-time copy per layer turn)
        if moe_layer_idx != self._ao_current_layer:
            self._ao_gpu_w13.zero_()
            self._ao_gpu_w13_scale.zero_()
            self._ao_gpu_w2.zero_()
            self._ao_gpu_w2_scale.zero_()
            self._ao_loaded_experts.clear()
            self._ao_current_layer = moe_layer_idx
            
            # Pre-populate with L1 experts from primary device's HCS buffers
            primary_hcs = self._hcs_devices[0]
            if moe_layer_idx in primary_hcs.buffers:
                bufs = primary_hcs.buffers[moe_layer_idx]
                lookup = primary_hcs.lookup[moe_layer_idx]
                # Identify which global expert IDs are in local L1 slots
                l1_eids = []
                l1_slots = []
                for eid in range(self.num_experts):
                    slot = int(lookup[eid])
                    if slot >= 0:
                        l1_eids.append(eid)
                        l1_slots.append(slot)
                
                if l1_eids:
                    dst_idx = torch.tensor(l1_eids, dtype=torch.long, device=self.device)
                    src_idx = torch.tensor(l1_slots, dtype=torch.long, device=self.device)
                    self._ao_gpu_w13.index_copy_(0, dst_idx, bufs["w13"].index_select(0, src_idx))
                    self._ao_gpu_w13_scale.index_copy_(0, dst_idx, bufs["w13_scale"].index_select(0, src_idx))
                    self._ao_gpu_w2.index_copy_(0, dst_idx, bufs["w2"].index_select(0, src_idx))
                    self._ao_gpu_w2_scale.index_copy_(0, dst_idx, bufs["w2_scale"].index_select(0, src_idx))
                    self._ao_loaded_experts.update(l1_eids)

        # Stream missing L2 experts from secondary GPUs
        for j, dev_idx in enumerate(dev_indices):
            if dev_idx > 0:
                eid = topk_list[j]
                key = (moe_layer_idx, eid)
                if key not in self._l2_sticky_map or eid not in self._ao_loaded_experts:
                    # Move from Secondary VRAM -> Primary VRAM
                    hcs_dev = self._hcs_devices[dev_idx]
                    slot = int(hcs_dev.lookup[moe_layer_idx][eid])
                    
                    # Inter-GPU copy (via P2P if working, else CPU bounce handled by torch)
                    self._ao_gpu_w13[eid:eid+1].copy_(hcs_dev.buffers[moe_layer_idx]["w13"][slot:slot+1])
                    self._ao_gpu_w13_scale[eid:eid+1].copy_(hcs_dev.buffers[moe_layer_idx]["w13_scale"][slot:slot+1])
                    self._ao_gpu_w2[eid:eid+1].copy_(hcs_dev.buffers[moe_layer_idx]["w2"][slot:slot+1])
                    self._ao_gpu_w2_scale[eid:eid+1].copy_(hcs_dev.buffers[moe_layer_idx]["w2_scale"][slot:slot+1])
                    
                    self._l2_sticky_map.add(key)
                    self._ao_loaded_experts.add(eid)

        # ── Step 3: Run Single Marlin Kernel on Primary GPU ──
        # Build topk_ids and weights for Marlin call (only includes hot experts)
        gpu_ids_list = []
        gpu_weights_list = []
        for j, dev_idx in enumerate(dev_indices):
            if dev_idx >= 0: # L1 or L2
                gpu_ids_list.append(topk_list[j])
                gpu_weights_list.append(weights_list[j])
            else: # Cold
                gpu_ids_list.append(0)
                gpu_weights_list.append(0.0)
        
        gpu_ids = torch.tensor([gpu_ids_list], dtype=topk_ids.dtype, device=self.device)
        gpu_weights = torch.tensor([gpu_weights_list], dtype=topk_weights.dtype, device=self.device)
        
        self._workspace.zero_()
        # For unified hot path, we use global EIDs as indices into the AO buffer
        # But Marlin expects indices relative to the provided weight buffers.
        # Since our weight buffers are [num_experts, ...], global EIDs are correct.
        
        # Bias management for sticky cache:
        # GPT OSS biases are small, we can load them all or just use remapped ones.
        # For simplicity, we'll let self._call_fused_moe handle standard bias lookup.
        
        gpu_output = self._call_fused_moe(
            moe_layer_idx=moe_layer_idx,
            hidden_states=x,
            w1=self._ao_gpu_w13,
            w2=self._ao_gpu_w2,
            w1_scale=self._ao_gpu_w13_scale,
            w2_scale=self._ao_gpu_w2_scale,
            gating_output=self._ao_gating_output_1,
            topk_weights=gpu_weights,
            topk_ids=gpu_ids,
            global_num_experts=self.num_experts,
            g_idx1=self._gpu_g_idx,
            g_idx2=self._gpu_g_idx,
            sort_indices1=self._gpu_sort_idx,
            sort_indices2=self._gpu_sort_idx,
            workspace=self._workspace,
        ).to(x.dtype)

        if timing:
            torch.cuda.synchronize(self.device)
            t_gpu_done = time.perf_counter()

        # ── Step 4: Sync CPU result and combine ──
        if has_cold:
            if timing: t_cpu_sync_start = time.perf_counter()
            cpu_output_bytes = self._engine.sync_forward()
            if timing: t_cpu_sync_done = time.perf_counter()
            cpu_raw = torch.frombuffer(bytearray(cpu_output_bytes), dtype=torch.bfloat16).reshape(1, K).to(self.device)
            if timing: t_cpu_to_gpu = time.perf_counter()
            if self.routed_scaling_factor != 1.0:
                cpu_raw = cpu_raw / self.routed_scaling_factor
            gpu_output += cpu_raw

        if timing:
            t_end = time.perf_counter()
            n_cold = sum(1 for d in dev_indices if d < 0)
            n_total = len(dev_indices)
            gpu_ms = (t_gpu_done - (t_cpu_submitted if has_cold else t_start)) * 1000
            total_ms = (t_end - t_start) * 1000
            logger.info(
                "HCS-STICKY-L2 L%d: gpu=%.1fms total=%.1fms cold=%d/%d devs=%d",
                moe_layer_idx, gpu_ms, total_ms, n_cold, n_total, len(self._hcs_devices)
            )
            self._pf_gpu_ms += gpu_ms
            if has_cold:
                self._pf_cpu_submit_ms += (t_cpu_submitted - t_start) * 1000
                self._pf_cpu_sync_ms += (t_cpu_sync_done - t_cpu_sync_start) * 1000
                self._pf_cpu_to_gpu_ms += (t_cpu_to_gpu - t_cpu_sync_done) * 1000
                self._pf_combine_ms += (t_end - t_cpu_to_gpu) * 1000
            self._pf_layers_total += 1
            if has_cold: self._pf_layers_cold += 1
            self._pf_cold_experts += n_cold
            self._pf_total_experts += n_total

        if self._heatmap is not None:
            for eid in topk_list:
                self._heatmap[(moe_layer_idx, eid)] = self._heatmap.get((moe_layer_idx, eid), 0) + 1

        return gpu_output

    def _forward_hcs_graphed_decode(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        timing: bool,
        t_start: float,
    ) -> torch.Tensor:
        """Fast M=1 decode using CUDA graphs for single-GPU HCS.

        Uses the captured CUDA graph (compact N-hot expert buffer) for the GPU
        hot experts, and falls back to CPU engine for cold experts.
        """
        K = self.hidden_size
        lookup = self._hcs_lookup[moe_layer_idx]
        topk_list = topk_ids[0].tolist()

        # Remap global expert IDs → local HCS slot IDs
        lookup_result = lookup[topk_ids.long()]  # [1, top_k]
        cold_mask = lookup_result < 0
        has_cold = cold_mask.any().item()

        # Submit cold experts to CPU engine (async)
        if has_cold:
            cpu_ids = topk_ids.clone()
            cpu_ids[~cold_mask] = -1  # Mask out hot experts
            act_cpu = x.detach().cpu().contiguous()
            ids_cpu = cpu_ids.to(torch.int32).cpu().contiguous()
            wts_cpu = topk_weights.detach().cpu().contiguous()
            act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
            ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
            wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()
            self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, 1)

        if timing:
            t_cpu_submitted = time.perf_counter()

        # Prepare local IDs and weights for CUDA graph
        local_ids = lookup_result.clone().to(topk_ids.dtype)
        local_ids[cold_mask] = 0
        gpu_topk_weights = topk_weights.clone()
        gpu_topk_weights[cold_mask] = 0.0

        # Replay CUDA graph (fast path)
        gpu_output = self._forward_hcs_graphed(moe_layer_idx, local_ids, gpu_topk_weights, x)

        if timing:
            torch.cuda.synchronize(self.device)
            t_gpu_done = time.perf_counter()

        # Sync CPU result and combine
        if has_cold:
            if timing: t_cpu_sync_start = time.perf_counter()
            cpu_output_bytes = self._engine.sync_forward()
            if timing: t_cpu_sync_done = time.perf_counter()
            cpu_raw = torch.frombuffer(bytearray(cpu_output_bytes), dtype=torch.bfloat16).reshape(1, K).to(self.device)
            if timing: t_cpu_to_gpu = time.perf_counter()
            if self.routed_scaling_factor != 1.0:
                cpu_raw = cpu_raw / self.routed_scaling_factor
            gpu_output = gpu_output + cpu_raw

        if timing:
            t_end = time.perf_counter()
            n_cold = cold_mask.sum().item()
            n_total = len(topk_list)
            gpu_ms = (t_gpu_done - (t_cpu_submitted if has_cold else t_start)) * 1000
            total_ms = (t_end - t_start) * 1000
            logger.info(
                "HCS-GRAPHED L%d: gpu=%.1fms total=%.1fms cold=%d/%d",
                moe_layer_idx, gpu_ms, total_ms, n_cold, n_total,
            )

        if self._heatmap is not None:
            for eid in topk_list:
                self._heatmap[(moe_layer_idx, eid)] = self._heatmap.get((moe_layer_idx, eid), 0) + 1

        return gpu_output

    def _forward_hcs_multigpu_graphed_decode(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        timing: bool,
        t_start: float,
    ) -> torch.Tensor:
        """Fast M=1 decode using per-GPU CUDA graphs for multi-GPU HCS.

        Each GPU replays its own CUDA graph for its local experts, then
        partial outputs are summed on the primary GPU.
        """
        K = self.hidden_size
        primary_device = self._hcs_devices[0].device
        device_lookup = self._hcs_device_lookup[moe_layer_idx]
        topk_list = topk_ids[0].tolist()

        # Classify experts: which device owns each, which are cold
        dev_indices = [int(device_lookup[eid]) for eid in topk_list]
        has_cold = any(d < 0 for d in dev_indices)

        if timing:
            t_classify = time.perf_counter()

        # ── Step 1: Submit cold experts to CPU engine (async) ──
        if has_cold:
            cpu_ids = topk_ids.clone()
            for j, d in enumerate(dev_indices):
                if d >= 0:
                    cpu_ids[0, j] = -1
            act_cpu = x.detach().cpu().contiguous()
            ids_cpu = cpu_ids.to(torch.int32).cpu().contiguous()
            wts_cpu = topk_weights.detach().cpu().contiguous()
            act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
            ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
            wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()
            self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, 1)

        if timing:
            t_cpu_submitted = time.perf_counter()

        # ── Step 2: Replay per-device CUDA graphs ──
        gpu_output = torch.zeros(1, K, dtype=self.params_dtype, device=primary_device)

        if timing:
            # Per-device timing: wall-clock start/end for parallelism verification
            _dev_timings = []  # (device_str, n_experts, setup_ms, replay_wall_start, replay_wall_end, accum_ms)

        for hcs_dev in self._hcs_devices:
            if moe_layer_idx not in hcs_dev.cuda_graphs:
                continue

            if timing:
                _t_dev_start = time.perf_counter()

            io = hcs_dev.graph_io[moe_layer_idx]
            graph = hcs_dev.cuda_graphs[moe_layer_idx]

            # Use lookup_on_primary for secondary devices to avoid cross-device access
            if hcs_dev.is_primary:
                lookup = hcs_dev.lookup[moe_layer_idx]
            else:
                lookup = hcs_dev.lookup_on_primary[moe_layer_idx]

            # Remap global expert IDs → local slot IDs for this device
            lookup_result = lookup[topk_ids.long()]  # [1, top_k] — on primary device
            on_this_dev = lookup_result >= 0
            n_on_dev = on_this_dev.sum().item()

            # Build local_ids and weights for this device
            local_ids = lookup_result.clone().to(topk_ids.dtype)
            local_ids[~on_this_dev] = 0
            dev_weights = topk_weights.clone()
            dev_weights[~on_this_dev] = 0.0

            # Copy inputs into this device's graph I/O buffers
            io["x"].copy_(x if hcs_dev.is_primary else x.to(hcs_dev.device))
            io["local_ids"].copy_(local_ids if hcs_dev.is_primary else local_ids.to(hcs_dev.device))
            io["weights"].copy_(dev_weights if hcs_dev.is_primary else dev_weights.to(hcs_dev.device))

            if timing:
                _t_setup_done = time.perf_counter()

            # Replay CUDA graph
            graph.replay()

            if timing:
                _t_replay_done = time.perf_counter()

            # Accumulate output on primary device
            if hcs_dev.is_primary:
                gpu_output += io["output"]
            else:
                gpu_output += io["output"].to(primary_device)

            if timing:
                _t_accum_done = time.perf_counter()
                _dev_timings.append((
                    str(hcs_dev.device), n_on_dev,
                    (_t_setup_done - _t_dev_start) * 1000,  # setup_ms
                    _t_setup_done,  # replay wall start (absolute)
                    _t_replay_done,  # replay wall end (absolute)
                    (_t_accum_done - _t_replay_done) * 1000,  # accum_ms
                ))

        if timing:
            # Sync all devices to measure true GPU completion
            torch.cuda.synchronize(primary_device)
            for hcs_dev in self._hcs_devices:
                if not hcs_dev.is_primary:
                    torch.cuda.synchronize(hcs_dev.device)
            t_gpu_done = time.perf_counter()

        # ── Step 3: Sync CPU result and combine ──
        if has_cold:
            if timing: t_cpu_sync_start = time.perf_counter()
            cpu_output_bytes = self._engine.sync_forward()
            if timing: t_cpu_sync_done = time.perf_counter()
            cpu_raw = torch.frombuffer(bytearray(cpu_output_bytes), dtype=torch.bfloat16).reshape(1, K).to(primary_device)
            if timing: t_cpu_to_gpu = time.perf_counter()
            if self.routed_scaling_factor != 1.0:
                cpu_raw = cpu_raw / self.routed_scaling_factor
            gpu_output = gpu_output + cpu_raw

        if timing:
            t_end = time.perf_counter()
            n_cold = sum(1 for d in dev_indices if d < 0)
            n_total = len(topk_list)
            classify_ms = (t_classify - t_start) * 1000
            cpu_submit_ms = (t_cpu_submitted - t_classify) * 1000
            gpu_ms = (t_gpu_done - t_cpu_submitted) * 1000
            total_ms = (t_end - t_start) * 1000

            # Per-device details
            dev_details = []
            for dev_str, n_exp, setup_ms, ws, we, accum_ms in _dev_timings:
                replay_wall_ms = (we - ws) * 1000
                dev_details.append(f"{dev_str}({n_exp}exp setup={setup_ms:.2f} replay={replay_wall_ms:.2f} accum={accum_ms:.2f})")

            # Parallelism check: did GPU replays overlap?
            parallel_note = ""
            if len(_dev_timings) >= 2:
                # Check if second device's replay started before first device's replay ended
                _, _, _, ws0, we0, _ = _dev_timings[0]
                _, _, _, ws1, we1, _ = _dev_timings[1]
                overlap_start = max(ws0, ws1)
                overlap_end = min(we0, we1)
                if overlap_end > overlap_start:
                    overlap_ms = (overlap_end - overlap_start) * 1000
                    parallel_note = f" PARALLEL overlap={overlap_ms:.2f}ms"
                else:
                    gap_ms = (max(ws0, ws1) - min(we0, we1)) * 1000
                    parallel_note = f" SEQUENTIAL gap={gap_ms:.2f}ms"

            cold_detail = ""
            if has_cold:
                cpu_wait_ms = (t_cpu_sync_done - t_cpu_sync_start) * 1000
                cpu_xfer_ms = (t_cpu_to_gpu - t_cpu_sync_done) * 1000
                cold_detail = f" cpu_wait={cpu_wait_ms:.2f} cpu_xfer={cpu_xfer_ms:.2f}"

            logger.info(
                "HCS-MGPU L%d: classify=%.2f cpu_sub=%.2f gpu=%.2f%s total=%.2fms cold=%d/%d%s | %s",
                moe_layer_idx, classify_ms, cpu_submit_ms, gpu_ms,
                cold_detail, total_ms, n_cold, n_total,
                parallel_note, " | ".join(dev_details),
            )

        if self._heatmap is not None:
            for eid in topk_list:
                self._heatmap[(moe_layer_idx, eid)] = self._heatmap.get((moe_layer_idx, eid), 0) + 1

        return gpu_output

    def _forward_hcs_cpu_only(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """CPU-only path for hot_cached_static layers with no hot experts.

        All experts are dispatched to the Rust AVX2 engine.
        """
        M = x.shape[0]
        timing = (TIMING.decode and M == 1) or (TIMING.prefill and M > 1)

        if timing:
            t_start = time.perf_counter()

        act_cpu = x.detach().cpu().contiguous()
        ids_cpu = topk_ids.detach().cpu().to(torch.int32).contiguous()
        wts_cpu = topk_weights.detach().cpu().contiguous()

        act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
        ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
        wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()

        if timing:
            t_serialized = time.perf_counter()

        self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, M)
        output_bytes = self._engine.sync_forward()

        if timing:
            t_cpu_done = time.perf_counter()

        output = torch.frombuffer(
            bytearray(output_bytes), dtype=torch.bfloat16
        ).reshape(M, self.hidden_size).to(self.device)

        # CPU engine applies routed_scaling_factor — divide it out
        if self.routed_scaling_factor != 1.0:
            output = output / self.routed_scaling_factor

        if timing:
            t_end = time.perf_counter()
            cpu_only_ms = (t_end - t_start) * 1000
            prefix = "HCS-PREFILL-CPU" if M > 1 else "HCS-CPU"
            logger.info(
                "%s L%d M=%d: serialize=%.1fms cpu_compute=%.1fms to_gpu=%.1fms total=%.1fms",
                prefix, moe_layer_idx, M,
                (t_serialized - t_start) * 1000,
                (t_cpu_done - t_serialized) * 1000,
                (t_end - t_cpu_done) * 1000,
                cpu_only_ms,
            )
            # Accumulate for summary
            self._pf_cpu_only_ms += cpu_only_ms
            self._pf_layers_total += 1
            self._pf_layers_all_cold += 1
            # All experts cold in this path
            n_total = topk_ids.numel()
            self._pf_cold_experts += n_total
            self._pf_total_experts += n_total

        return output

    # ── CUDA Graph Acceleration for hot_cached_static ──

    def _init_cuda_graphs(self):
        """Capture CUDA graphs for M=1 decode in hot_cached_static mode.

        Creates one CUDA graph per MoE layer with hot experts.
        Pre-allocates fixed-address I/O tensors for graph replay.
        """
        if not self._hcs_initialized or not self._hcs_buffers:
            logger.warning("Cannot init CUDA graphs: hot_cached_static not initialized")
            return

        K = self.hidden_size
        top_k = self._engine.top_k()
        torch.cuda.set_device(self.device)

        graphs_captured = 0
        for layer_idx, bufs in self._hcs_buffers.items():
            n_hot = self._hcs_num_pinned[layer_idx]

            # Pre-allocate fixed-address I/O tensors
            io = {
                "x": torch.zeros(1, K, dtype=self.params_dtype, device=self.device),
                "local_ids": torch.zeros(1, top_k, dtype=torch.int32, device=self.device),
                "weights": torch.zeros(1, top_k, dtype=torch.float32, device=self.device),
                "gating": torch.empty(1, n_hot, device=self.device),
                "output": torch.zeros(1, K, dtype=self.params_dtype, device=self.device),
            }
            g_idx = self._hcs_g_idx[layer_idx]
            sort_idx = self._hcs_sort_idx[layer_idx]

            # Warmup runs (3x) on current stream to trigger Triton compilation
            # Use _call_fused_marlin_moe dispatch to handle w2 padding + gpt_oss activation
            for _ in range(3):
                _ = self._call_fused_moe(
                    hidden_states=io["x"],
                    w1=bufs["w13"], w2=bufs["w2"],
                    w1_scale=bufs["w13_scale"], w2_scale=bufs["w2_scale"],
                    gating_output=io["gating"],
                    topk_weights=io["weights"], topk_ids=io["local_ids"],
                    moe_layer_idx=layer_idx,
                    global_num_experts=n_hot, expert_map=None,
                    g_idx1=g_idx, g_idx2=g_idx,
                    sort_indices1=sort_idx, sort_indices2=sort_idx,
                    workspace=self._workspace,
                    num_bits=self.num_bits, is_k_full=True,
                )
            torch.cuda.synchronize(self.device)

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                io["output"] = self._call_fused_moe(
                    hidden_states=io["x"],
                    w1=bufs["w13"], w2=bufs["w2"],
                    w1_scale=bufs["w13_scale"], w2_scale=bufs["w2_scale"],
                    gating_output=io["gating"],
                    topk_weights=io["weights"], topk_ids=io["local_ids"],
                    moe_layer_idx=layer_idx,
                    global_num_experts=n_hot, expert_map=None,
                    g_idx1=g_idx, g_idx2=g_idx,
                    sort_indices1=sort_idx, sort_indices2=sort_idx,
                    workspace=self._workspace,
                    num_bits=self.num_bits, is_k_full=True,
                ).to(self.params_dtype)

            self._hcs_cuda_graphs[layer_idx] = graph
            self._hcs_graph_io[layer_idx] = io
            graphs_captured += 1

        self._hcs_cuda_graphs_enabled = True
        logger.info("CUDA graphs captured: %d MoE layers", graphs_captured)

    def _forward_hcs_graphed(
        self,
        moe_layer_idx: int,
        local_ids: torch.Tensor,
        gpu_topk_weights: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Replay captured CUDA graph for M=1 GPU hot expert computation.

        Copies input data into graph's fixed-address tensors, replays the
        captured kernel sequence, and returns the output.
        """
        io = self._hcs_graph_io[moe_layer_idx]
        graph = self._hcs_cuda_graphs[moe_layer_idx]

        # Copy inputs into fixed-address buffers
        io["x"].copy_(x)
        io["local_ids"].copy_(local_ids)
        io["weights"].copy_(gpu_topk_weights)

        # Replay captured kernel
        graph.replay()

        # Return clone (graph tensor is in CUDA graph private pool)
        return io["output"].clone()

    def _init_cuda_graphs_multi_gpu(self):
        """Capture per-device CUDA graphs for M=1 decode in multi-GPU HCS mode.

        Creates one CUDA graph per (device, layer) pair. Each GPU runs
        fused_marlin_moe against its own compact N-hot buffer independently.
        """
        if not self._hcs_initialized:
            logger.warning("Cannot init multi-GPU CUDA graphs: HCS not initialized")
            return

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        K = self.hidden_size
        top_k = self._engine.top_k()
        graphs_captured = 0
        original_device = torch.cuda.current_device()

        try:
            for hcs_dev in self._hcs_devices:
                dev = hcs_dev.device
                torch.cuda.set_device(dev)
                # Use explicit stream on this device for graph capture
                capture_stream = torch.cuda.Stream(device=dev)

                for layer_idx, bufs in hcs_dev.buffers.items():
                    n_hot = hcs_dev.num_pinned[layer_idx]
                    if n_hot == 0:
                        continue

                    g_idx = hcs_dev.g_idx[layer_idx]
                    sort_idx = hcs_dev.sort_idx[layer_idx]

                    # Pre-allocate fixed-address I/O tensors on this device
                    with torch.cuda.device(dev):
                        io = {
                            "x": torch.zeros(1, K, dtype=self.params_dtype, device=dev),
                            "local_ids": torch.zeros(1, top_k, dtype=torch.int32, device=dev),
                            "weights": torch.zeros(1, top_k, dtype=torch.float32, device=dev),
                            "gating": torch.empty(1, n_hot, device=dev),
                            "output": torch.zeros(1, K, dtype=self.params_dtype, device=dev),
                        }

                    # Warmup runs (3x) to trigger Triton compilation
                    with torch.cuda.device(dev), torch.cuda.stream(capture_stream):
                        for _ in range(3):
                            _ = fused_marlin_moe(
                                hidden_states=io["x"],
                                w1=bufs["w13"], w2=bufs["w2"],
                                w1_scale=bufs["w13_scale"], w2_scale=bufs["w2_scale"],
                                gating_output=io["gating"],
                                topk_weights=io["weights"], topk_ids=io["local_ids"],
                                global_num_experts=n_hot, expert_map=None,
                                g_idx1=g_idx, g_idx2=g_idx,
                                sort_indices1=sort_idx, sort_indices2=sort_idx,
                                workspace=hcs_dev.workspace,
                                num_bits=self.num_bits, is_k_full=True,
                            )
                    torch.cuda.synchronize(dev)

                    # Capture graph on this device's stream
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.device(dev), torch.cuda.stream(capture_stream):
                        with torch.cuda.graph(graph, stream=capture_stream):
                            io["output"] = fused_marlin_moe(
                                hidden_states=io["x"],
                                w1=bufs["w13"], w2=bufs["w2"],
                                w1_scale=bufs["w13_scale"], w2_scale=bufs["w2_scale"],
                                gating_output=io["gating"],
                                topk_weights=io["weights"], topk_ids=io["local_ids"],
                                global_num_experts=n_hot, expert_map=None,
                                g_idx1=g_idx, g_idx2=g_idx,
                                sort_indices1=sort_idx, sort_indices2=sort_idx,
                                workspace=hcs_dev.workspace,
                                num_bits=self.num_bits, is_k_full=True,
                            ).to(self.params_dtype)

                    hcs_dev.cuda_graphs[layer_idx] = graph
                    hcs_dev.graph_io[layer_idx] = io
                    graphs_captured += 1

                logger.info("Multi-GPU CUDA graphs: %s captured %d layers",
                            dev, len(hcs_dev.cuda_graphs))

            self._hcs_cuda_graphs_enabled = True
            logger.info("Multi-GPU CUDA graphs captured: %d total across %d devices",
                        graphs_captured, len(self._hcs_devices))

        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or isinstance(e, (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)):
                logger.warning("Multi-GPU CUDA graph capture OOM on %s, falling back to unified decode: %s",
                               torch.cuda.current_device(), e)
                # Clear any partially captured graphs
                for hcs_dev in self._hcs_devices:
                    if hcs_dev.cuda_graphs:
                        hcs_dev.cuda_graphs.clear()
                    if hcs_dev.graph_io:
                        hcs_dev.graph_io.clear()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            torch.cuda.set_device(original_device)

    def _forward_persistent(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using pre-loaded persistent expert buffers. Zero DMA."""
        M = x.shape[0]
        buffers = self._persistent[moe_layer_idx]
        E_local = self.num_local_experts

        # EP remapping: when this manager owns only a slice of experts,
        # remap global IDs to local indices and zero non-local weights.
        # Same pattern as _forward_engine chunking (lines 3908-3910).
        if E_local < self.num_experts:
            local_mask = (topk_ids >= self.expert_start) & (topk_ids < self.expert_end)
            topk_ids = torch.where(local_mask, topk_ids - self.expert_start, torch.zeros_like(topk_ids))
            topk_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
            num_experts_for_kernel = E_local
        else:
            num_experts_for_kernel = self.num_experts

        gating_output = torch.empty(M, num_experts_for_kernel, device=self.device)

        self._workspace.zero_()

        # Slice g_idx/sort_idx to match local expert count
        g_idx = self._gpu_g_idx[:num_experts_for_kernel] if self._gpu_g_idx is not None else None
        sort_idx = self._gpu_sort_idx[:num_experts_for_kernel] if self._gpu_sort_idx is not None else None

        output = self._call_fused_moe(
            moe_layer_idx=moe_layer_idx,
            hidden_states=x,
            w1=buffers["w13_packed"],
            w2=buffers["w2_packed"],
            w1_scale=buffers["w13_scale"],
            w2_scale=buffers["w2_scale"],
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=num_experts_for_kernel,
            g_idx1=g_idx,
            g_idx2=g_idx,
            sort_indices1=sort_idx,
            sort_indices2=sort_idx,
            workspace=self._workspace,
        ).to(x.dtype)

        return output

    def _disk_cache_path(self, moe_layer_idx: int, kind: str = "routed") -> Path:
        """Path for disk-cached Marlin weights."""
        layer_idx = moe_layer_idx + self.first_k_dense
        cache_dir = Path(self.model_path) / ".marlin_cache" / f"b{self.num_bits}"
        return cache_dir / f"layer{layer_idx}_{kind}.pt"

    def _save_to_disk(self, moe_layer_idx: int, cache_dict: dict, kind: str = "routed"):
        """Save Marlin-repacked weights to disk cache."""
        path = self._disk_cache_path(moe_layer_idx, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_dict, str(path))
        size_mb = sum(t.nbytes for t in cache_dict.values() if isinstance(t, torch.Tensor)) / 1e6
        logger.info("Disk cache saved: %s (%.1f MB)", path.name, size_mb)

    def _load_from_disk(self, moe_layer_idx: int, kind: str = "routed") -> Optional[dict]:
        """Load Marlin-repacked weights from disk cache."""
        path = self._disk_cache_path(moe_layer_idx, kind)
        if not path.exists():
            return None
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=True)
            logger.info("Disk cache loaded: %s", path.name)
            return data
        except Exception as e:
            logger.warning("Disk cache corrupted, re-computing: %s (%s)", path.name, e)
            path.unlink(missing_ok=True)
            return None

    def prepare_layer(self, moe_layer_idx: int):
        """Quantize and cache one MoE layer's expert weights for GPU prefill.

        Reads from safetensors, quantizes BF16→INT4 on GPU, repacks to Marlin,
        then caches to CPU RAM and disk.
        """
        if moe_layer_idx in self._cache:
            return  # Already in RAM

        # Try disk cache first
        disk_data = self._load_from_disk(moe_layer_idx)
        if disk_data is not None:
            self._cache[moe_layer_idx] = disk_data
            return

        # Set device for CUDA kernel calls (gptq_marlin_repack)
        torch.cuda.set_device(self.device)

        self._ensure_index()
        layer_idx = moe_layer_idx + self.first_k_dense
        start = time.perf_counter()

        from sglang.srt.layers.quantization.gptq import (
            gptq_marlin_repack,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_permute_scales,
        )

        K = self.hidden_size
        N = self.intermediate_size

        # Collect all experts for this layer
        all_w13_packed = []
        all_w13_scale = []
        all_w2_packed = []
        all_w2_scale = []

        for eidx in range(self.num_experts):
            prefix = f"{self._layers_prefix}.layers.{layer_idx}.mlp.experts.{eidx}"

            # Read gate, up, down weights as BF16 (HF convention: [out, in])
            gate = self._read_expert_weight_bf16(prefix, "gate_proj")  # [N, K]
            up = self._read_expert_weight_bf16(prefix, "up_proj")      # [N, K]
            down = self._read_expert_weight_bf16(prefix, "down_proj")  # [K, N]

            # Concatenate gate+up → w13, then transpose to [K, 2*N] for Marlin
            w13 = torch.cat([gate, up], dim=0)  # [2*N, K]
            del gate, up
            w13 = w13.t().contiguous()  # [K, 2*N] — reduction dim first

            # Transpose down to [N, K] for Marlin (size_k=N, size_n=K)
            down = down.t().contiguous()  # [N, K]

            # Quantize on GPU
            w13_gpu = w13.to(self.device)
            w13_packed, w13_scale = _quantize_and_pack_gpu(w13_gpu, self._group_size, self.num_bits)
            del w13, w13_gpu

            down_gpu = down.to(self.device)
            w2_packed, w2_scale = _quantize_and_pack_gpu(down_gpu, self._group_size, self.num_bits)
            del down, down_gpu

            # Marlin repack (per-expert, CUDA kernel)
            perm = torch.empty(0, dtype=torch.int32, device=self.device)
            w13_repacked = gptq_marlin_repack(w13_packed, perm, K, 2 * N, self.num_bits)
            w2_repacked = gptq_marlin_repack(w2_packed, perm, N, K, self.num_bits)

            # Permute scales for Marlin
            w13_scale_perm = marlin_permute_scales(
                w13_scale, K, 2 * N, self._group_size,
            )
            w2_scale_perm = marlin_permute_scales(
                w2_scale, N, K, self._group_size,
            )

            # Move to CPU for RAM cache
            all_w13_packed.append(w13_repacked.cpu())
            all_w13_scale.append(w13_scale_perm.cpu())
            all_w2_packed.append(w2_repacked.cpu())
            all_w2_scale.append(w2_scale_perm.cpu())

            del w13_packed, w13_scale, w2_packed, w2_scale
            del w13_repacked, w2_repacked, w13_scale_perm, w2_scale_perm

        # Stack into [num_experts, ...] tensors
        cache_dict = {
            "w13_packed": torch.stack(all_w13_packed),  # [E, K//16, 2*N*2]
            "w13_scale": torch.stack(all_w13_scale),    # [E, K//128, 2*N]
            "w2_packed": torch.stack(all_w2_packed),    # [E, N//16, K*2]
            "w2_scale": torch.stack(all_w2_scale),      # [E, N//128, K]
        }
        self._cache[moe_layer_idx] = cache_dict

        # Save to disk for future runs
        self._save_to_disk(moe_layer_idx, cache_dict)

        elapsed = time.perf_counter() - start
        logger.info(
            "Layer %d prepared: %d experts in %.1fs (%.1fs/expert)",
            layer_idx, self.num_experts, elapsed, elapsed / self.num_experts,
        )

    def prepare_shared_expert(self, moe_layer_idx: int):
        """Quantize and cache the shared expert for GPU prefill."""
        if moe_layer_idx in self._shared_cache:
            return
        if self.n_shared_experts == 0:
            return

        # Try disk cache first
        disk_data = self._load_from_disk(moe_layer_idx, kind="shared")
        if disk_data is not None:
            self._shared_cache[moe_layer_idx] = disk_data
            return

        # Set device for CUDA kernel calls (gptq_marlin_repack)
        torch.cuda.set_device(self.device)

        self._ensure_index()
        layer_idx = moe_layer_idx + self.first_k_dense

        from sglang.srt.layers.quantization.gptq import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales

        prefix = f"{self._layers_prefix}.layers.{layer_idx}.mlp.shared_experts"
        gate = self._read_expert_weight_bf16(prefix, "gate_proj")  # [shared_N, K]
        up = self._read_expert_weight_bf16(prefix, "up_proj")      # [shared_N, K]
        down = self._read_expert_weight_bf16(prefix, "down_proj")  # [K, shared_N]

        shared_N = gate.shape[0]  # n_shared * intermediate
        K = self.hidden_size

        # Transpose to Marlin convention: reduction dim first
        w13 = torch.cat([gate, up], dim=0).t().contiguous().to(self.device)  # [K, 2*shared_N]
        del gate, up
        w13_packed, w13_scale = _quantize_and_pack_gpu(w13, GROUP_SIZE, self.num_bits)
        del w13

        down_gpu = down.t().contiguous().to(self.device)  # [shared_N, K]
        del down
        w2_packed, w2_scale = _quantize_and_pack_gpu(down_gpu, self._group_size, self.num_bits)
        del down_gpu

        perm = torch.empty(0, dtype=torch.int32, device=self.device)
        w13_repacked = gptq_marlin_repack(w13_packed, perm, K, 2 * shared_N, self.num_bits)
        w2_repacked = gptq_marlin_repack(w2_packed, perm, shared_N, K, self.num_bits)
        w13_scale_perm = marlin_permute_scales(w13_scale, K, 2 * shared_N, GROUP_SIZE)
        w2_scale_perm = marlin_permute_scales(w2_scale, shared_N, K, GROUP_SIZE)

        # Shared expert stored as single expert (unsqueeeze to [1, ...])
        cache_dict = {
            "w13_packed": w13_repacked.unsqueeze(0).cpu(),
            "w13_scale": w13_scale_perm.unsqueeze(0).cpu(),
            "w2_packed": w2_repacked.unsqueeze(0).cpu(),
            "w2_scale": w2_scale_perm.unsqueeze(0).cpu(),
            "shared_N": shared_N,
        }
        self._shared_cache[moe_layer_idx] = cache_dict

        # Save to disk
        self._save_to_disk(moe_layer_idx, cache_dict, kind="shared")

        del w13_packed, w13_scale, w2_packed, w2_scale
        del w13_repacked, w2_repacked, w13_scale_perm, w2_scale_perm
        logger.info("Shared expert prepared for layer %d (intermediate=%d)", layer_idx, shared_N)

    def forward(
        self,
        moe_layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_only: bool = False,
    ) -> torch.Tensor:
        """Run GPU MoE forward for one layer using cached INT4 Marlin weights.

        Args:
            moe_layer_idx: 0-based MoE layer index
            hidden_states: [M, K] input on GPU
            topk_ids: [M, top_k] expert indices on GPU
            topk_weights: [M, top_k] routing weights on GPU
            routed_only: If True, return raw routed output without
                routed_scaling_factor or shared expert (for EP partial sums).

        Returns:
            [M, K] output tensor on GPU
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        # sgl_kernel's moe_sum_reduce launches on the current CUDA device,
        # not the tensor's device. Must set device explicitly for multi-GPU.
        torch.cuda.set_device(self.device)

        M = hidden_states.shape[0]
        x = hidden_states

        # Debug flag: set KRASIS_DEBUG_SYNC=1 for synchronous CUDA error checking
        debug_sync = os.environ.get("KRASIS_DEBUG_SYNC", "") == "1"

        if debug_sync:
            for name, t in [("hidden", x), ("topk_ids", topk_ids), ("topk_weights", topk_weights)]:
                if t.device != self.device:
                    logger.error("DEVICE MISMATCH: %s on %s, expected %s", name, t.device, self.device)
            logger.info("forward() moe_layer=%d M=%d device=%s mode=%s",
                        moe_layer_idx, M, self.device, self._prefill_mode)

        if debug_sync:
            torch.cuda.synchronize(self.device)

        # Layer-grouped DMA data takes priority (M>1 prefill uses GPU for all experts)
        if self._persistent and moe_layer_idx in self._persistent:
            output = self._forward_persistent(moe_layer_idx, x, topk_ids, topk_weights)
        elif self._prefill_mode == "hot_cached_static" and self._hcs_initialized:
            output = self._forward_hot_cached_static(moe_layer_idx, x, topk_ids, topk_weights)
        elif self._prefill_mode in ("active_only", "lru") and self._engine is not None:
            if M == 1:
                # Compact per-layer buffers for decode (cross-token expert caching)
                output = self._forward_compact(moe_layer_idx, x, topk_ids, topk_weights)
            elif self._prefill_mode == "lru":
                # Invalidate compact state (AO buffer will be used for prefill)
                self._invalidate_compact()
                output = self._forward_lru(moe_layer_idx, x, topk_ids, topk_weights)
            else:
                # Invalidate compact state (AO buffer will be used for prefill)
                self._invalidate_compact()
                output = self._forward_active_only(moe_layer_idx, x, topk_ids, topk_weights)
        elif self._engine is not None:
            self._allocate_gpu_buffer()
            output = self._forward_engine(moe_layer_idx, x, topk_ids, topk_weights, debug_sync)
        else:
            # Legacy safetensors path with RAM cache
            self._allocate_gpu_buffer()
            if moe_layer_idx not in self._cache:
                self.prepare_layer(moe_layer_idx)
            cache = self._cache[moe_layer_idx]
            output = self._forward_cached(cache, x, topk_ids, topk_weights, debug_sync)

        # EP partial sum: return raw routed output (no scaling, no shared expert)
        if routed_only:
            return output

        # Apply routed scaling factor and add shared expert
        if self.n_shared_experts > 0:
            if debug_sync:
                torch.cuda.synchronize(self.device)
                logger.info("  pre-shared-expert sync OK")
            shared_output = self._shared_expert_forward(moe_layer_idx, x)
            if debug_sync:
                torch.cuda.synchronize(self.device)
                logger.info("  post-shared-expert sync OK")
            output = self.routed_scaling_factor * output + shared_output
        elif self.routed_scaling_factor != 1.0:
            output *= self.routed_scaling_factor

        return output

    def _repack_chunk_from_engine(
        self,
        moe_layer_idx: int,
        chunk_start: int,
        chunk_end: int,
    ):
        """Get expert weight data from Rust engine and load into GPU buffer.

        Marlin-native path: DMA copy directly (zero conversion).
        Legacy path: repack from old transposed format to Marlin on GPU.

        Called per-chunk during forward() with ZERO caching.
        Temporary CPU data is freed immediately after GPU upload.
        """
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        N = self.intermediate_size
        gs = self._engine.group_size()
        actual = chunk_end - chunk_start

        # Get raw bytes from Rust for just this chunk (zero-copy Rust→Python)
        w13_packed_bytes = self._engine.get_expert_w13_packed(moe_layer_idx, chunk_start, chunk_end)
        w13_scales_bytes = self._engine.get_expert_w13_scales(moe_layer_idx, chunk_start, chunk_end)
        w2_packed_bytes = self._engine.get_expert_w2_packed(moe_layer_idx, chunk_start, chunk_end)
        w2_scales_bytes = self._engine.get_expert_w2_scales(moe_layer_idx, chunk_start, chunk_end)

        nb2 = self.num_bits // 2  # 2 for INT4, 4 for INT8

        if self._engine.is_marlin_format:
            # Marlin-native: data is already GPU-ready, just DMA copy
            w13_packed = torch.frombuffer(
                bytearray(w13_packed_bytes), dtype=torch.int32
            ).reshape(actual, K // 16, 2 * N * nb2)
            w13_scales = torch.frombuffer(
                bytearray(w13_scales_bytes), dtype=torch.bfloat16
            ).reshape(actual, K // gs, 2 * N)
            w2_packed = torch.frombuffer(
                bytearray(w2_packed_bytes), dtype=torch.int32
            ).reshape(actual, N // 16, K_w2 * nb2)
            w2_scales = torch.frombuffer(
                bytearray(w2_scales_bytes), dtype=torch.bfloat16
            ).reshape(actual, N // gs, K_w2)

            self._gpu_w13_packed[:actual].copy_(w13_packed)
            self._gpu_w13_scale[:actual].copy_(w13_scales)
            self._gpu_w2_packed[:actual].copy_(w2_packed)
            self._gpu_w2_scale[:actual].copy_(w2_scales)
            return

        # Legacy path: old transposed format → repack to Marlin on GPU
        from sglang.srt.layers.quantization.gptq import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales

        w13_packed_cpu = torch.frombuffer(
            bytearray(w13_packed_bytes), dtype=torch.int32
        ).reshape(actual, K // 8, 2 * N)
        w13_scales_cpu = torch.frombuffer(
            bytearray(w13_scales_bytes), dtype=torch.bfloat16
        ).reshape(actual, K // gs, 2 * N)
        w2_packed_cpu = torch.frombuffer(
            bytearray(w2_packed_bytes), dtype=torch.int32
        ).reshape(actual, N // 8, K)
        w2_scales_cpu = torch.frombuffer(
            bytearray(w2_scales_bytes), dtype=torch.bfloat16
        ).reshape(actual, N // gs, K)

        perm = torch.empty(0, dtype=torch.int32, device=self.device)

        # Repack each expert directly into GPU buffer
        for i in range(actual):
            w13_gpu = w13_packed_cpu[i].to(self.device)
            w13_repacked = gptq_marlin_repack(w13_gpu, perm, K, 2 * N, self.num_bits)
            w13_scale_gpu = w13_scales_cpu[i].to(self.device)
            w13_scale_perm = marlin_permute_scales(w13_scale_gpu, K, 2 * N, gs)

            self._gpu_w13_packed[i].copy_(w13_repacked)
            self._gpu_w13_scale[i].copy_(w13_scale_perm)

            w2_gpu = w2_packed_cpu[i].to(self.device)
            w2_repacked = gptq_marlin_repack(w2_gpu, perm, N, K, self.num_bits)
            w2_scale_gpu = w2_scales_cpu[i].to(self.device)
            w2_scale_perm = marlin_permute_scales(w2_scale_gpu, N, K, gs)

            self._gpu_w2_packed[i].copy_(w2_repacked)
            self._gpu_w2_scale[i].copy_(w2_scale_perm)
        # CPU tensors freed when function returns

    def _forward_engine(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        debug_sync: bool,
    ) -> torch.Tensor:
        """Forward using engine weights. Marlin-native: DMA copy. Legacy: on-the-fly repack."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        M = x.shape[0]
        output = torch.zeros(M, self.hidden_size, dtype=x.dtype, device=self.device)
        gating_output = torch.empty(M, self.chunk_size, device=self.device)

        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self.num_experts)
            actual = end - start

            # Repack this chunk from engine → GPU buffer (on-the-fly, no caching)
            self._repack_chunk_from_engine(moe_layer_idx, start, end)

            # Remap expert IDs for this chunk
            chunk_mask = (topk_ids >= start) & (topk_ids < end)
            chunk_ids = torch.where(chunk_mask, topk_ids - start, torch.zeros_like(topk_ids))
            chunk_weights = torch.where(chunk_mask, topk_weights, torch.zeros_like(topk_weights))

            self._workspace.zero_()

            if debug_sync:
                torch.cuda.synchronize(self.device)
                logger.info("  chunk %d/%d: actual=%d, pre-kernel sync OK",
                            chunk_idx, self.num_chunks, actual)

            chunk_output = fused_marlin_moe(
                hidden_states=x,
                w1=self._gpu_w13_packed[:actual],
                w2=self._gpu_w2_packed[:actual],
                w1_scale=self._gpu_w13_scale[:actual],
                w2_scale=self._gpu_w2_scale[:actual],
                gating_output=gating_output[:, :actual],
                topk_weights=chunk_weights,
                topk_ids=chunk_ids,
                global_num_experts=actual,
                expert_map=None,
                g_idx1=self._gpu_g_idx[:actual],
                g_idx2=self._gpu_g_idx[:actual],
                sort_indices1=self._gpu_sort_idx[:actual],
                sort_indices2=self._gpu_sort_idx[:actual],
                workspace=self._workspace,
                num_bits=self.num_bits,
                is_k_full=True,
            ).to(x.dtype)

            if debug_sync:
                torch.cuda.synchronize(self.device)
                logger.info("  chunk %d/%d: post-kernel sync OK", chunk_idx, self.num_chunks)

            output += chunk_output

        return output

    def _forward_cached(
        self,
        cache: dict,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        debug_sync: bool,
    ) -> torch.Tensor:
        """Forward using pre-cached Marlin weights in RAM (legacy path)."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        M = x.shape[0]

        if self.num_chunks == 1:
            self._gpu_w13_packed.copy_(cache["w13_packed"])
            self._gpu_w13_scale.copy_(cache["w13_scale"])
            self._gpu_w2_packed.copy_(cache["w2_packed"])
            self._gpu_w2_scale.copy_(cache["w2_scale"])

            gating_output = torch.empty(M, self.num_experts, device=self.device)

            output = fused_marlin_moe(
                hidden_states=x,
                w1=self._gpu_w13_packed,
                w2=self._gpu_w2_packed,
                w1_scale=self._gpu_w13_scale,
                w2_scale=self._gpu_w2_scale,
                gating_output=gating_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=self.num_experts,
                expert_map=None,
                g_idx1=self._gpu_g_idx,
                g_idx2=self._gpu_g_idx,
                sort_indices1=self._gpu_sort_idx,
                sort_indices2=self._gpu_sort_idx,
                workspace=self._workspace,
                num_bits=self.num_bits,
                is_k_full=True,
            ).to(x.dtype)
        else:
            output = torch.zeros(M, self.hidden_size, dtype=x.dtype, device=self.device)
            gating_output = torch.empty(M, self.chunk_size, device=self.device)

            for chunk_idx in range(self.num_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, self.num_experts)
                actual = end - start

                self._gpu_w13_packed[:actual].copy_(cache["w13_packed"][start:end])
                self._gpu_w13_scale[:actual].copy_(cache["w13_scale"][start:end])
                self._gpu_w2_packed[:actual].copy_(cache["w2_packed"][start:end])
                self._gpu_w2_scale[:actual].copy_(cache["w2_scale"][start:end])

                chunk_mask = (topk_ids >= start) & (topk_ids < end)
                chunk_ids = torch.where(chunk_mask, topk_ids - start, torch.zeros_like(topk_ids))
                chunk_weights = torch.where(chunk_mask, topk_weights, torch.zeros_like(topk_weights))

                self._workspace.zero_()

                chunk_output = fused_marlin_moe(
                    hidden_states=x,
                    w1=self._gpu_w13_packed[:actual],
                    w2=self._gpu_w2_packed[:actual],
                    w1_scale=self._gpu_w13_scale[:actual],
                    w2_scale=self._gpu_w2_scale[:actual],
                    gating_output=gating_output[:, :actual],
                    topk_weights=chunk_weights,
                    topk_ids=chunk_ids,
                    global_num_experts=actual,
                    expert_map=None,
                    g_idx1=self._gpu_g_idx[:actual],
                    g_idx2=self._gpu_g_idx[:actual],
                    sort_indices1=self._gpu_sort_idx[:actual],
                    sort_indices2=self._gpu_sort_idx[:actual],
                    workspace=self._workspace,
                    num_bits=self.num_bits,
                    is_k_full=True,
                ).to(x.dtype)

                output += chunk_output

        return output

    def _shared_expert_forward(
        self,
        moe_layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute shared expert forward on GPU using Marlin kernel."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        M = hidden_states.shape[0]

        if self._persistent_shared is not None and moe_layer_idx in self._persistent_shared:
            # Persistent mode: use pre-loaded shared expert buffers (zero DMA)
            cache = self._persistent_shared[moe_layer_idx]
            w13_packed = cache["w13_packed"]
            w13_scale = cache["w13_scale"]
            w2_packed = cache["w2_packed"]
            w2_scale = cache["w2_scale"]
        elif self._engine is not None:
            # On-the-fly repack from engine (single expert, tiny)
            w13_packed, w13_scale, w2_packed, w2_scale = self._repack_shared_expert(moe_layer_idx)
        else:
            if moe_layer_idx not in self._shared_cache:
                self.prepare_shared_expert(moe_layer_idx)
            cache = self._shared_cache[moe_layer_idx]
            w13_packed = cache["w13_packed"].to(self.device)
            w13_scale = cache["w13_scale"].to(self.device)
            w2_packed = cache["w2_packed"].to(self.device)
            w2_scale = cache["w2_scale"].to(self.device)

        g_idx = torch.empty(1, 0, dtype=torch.int32, device=self.device)
        sort_idx = torch.empty(1, 0, dtype=torch.int32, device=self.device)
        workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        # All tokens routed to expert 0 (the only shared expert) with weight 1.0
        topk_ids = torch.zeros(M, 1, dtype=torch.int32, device=self.device)
        topk_weights = torch.ones(M, 1, dtype=torch.float32, device=self.device)
        gating_output = torch.empty(M, 1, device=self.device)

        output = fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w13_packed,
            w2=w2_packed,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=1,
            expert_map=torch.empty(1, device=self.device),
            g_idx1=g_idx,
            g_idx2=g_idx,
            sort_indices1=sort_idx,
            sort_indices2=sort_idx,
            workspace=workspace,
            num_bits=self.num_bits,
            is_k_full=True,
        ).to(hidden_states.dtype)

        return output

    def _repack_shared_expert(self, moe_layer_idx: int):
        """Get shared expert weights from engine and prepare for GPU.

        Marlin-native: reshape and DMA copy directly (zero conversion).
        Legacy: repack from old transposed format to Marlin on GPU.
        """
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # padded K for w2 output dim (Marlin compat)
        shared_N = self.n_shared_experts * self.intermediate_size
        gs = self._engine.group_size()
        nb2 = self.num_bits // 2  # 2 for INT4

        w13p_bytes, w13s_bytes, w2p_bytes, w2s_bytes = self._engine.get_shared_expert_weights(moe_layer_idx)

        if self._engine.is_marlin_format:
            # Marlin-native: data is already GPU-ready
            w13_packed = torch.frombuffer(
                bytearray(w13p_bytes), dtype=torch.int32
            ).reshape(1, K // 16, 2 * shared_N * nb2).to(self.device)
            w13_scale = torch.frombuffer(
                bytearray(w13s_bytes), dtype=torch.bfloat16
            ).reshape(1, K // gs, 2 * shared_N).to(self.device)
            w2_packed = torch.frombuffer(
                bytearray(w2p_bytes), dtype=torch.int32
            ).reshape(1, shared_N // 16, K_w2 * nb2).to(self.device)
            w2_scale = torch.frombuffer(
                bytearray(w2s_bytes), dtype=torch.bfloat16
            ).reshape(1, shared_N // gs, K_w2).to(self.device)
            return w13_packed, w13_scale, w2_packed, w2_scale

        # Legacy path: repack to Marlin on GPU
        from sglang.srt.layers.quantization.gptq import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales

        w13p = torch.frombuffer(bytearray(w13p_bytes), dtype=torch.int32).reshape(K // 8, 2 * shared_N)
        w13s = torch.frombuffer(bytearray(w13s_bytes), dtype=torch.bfloat16).reshape(K // gs, 2 * shared_N)
        w2p = torch.frombuffer(bytearray(w2p_bytes), dtype=torch.int32).reshape(shared_N // 8, K)
        w2s = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16).reshape(shared_N // gs, K)

        perm = torch.empty(0, dtype=torch.int32, device=self.device)

        w13_packed = gptq_marlin_repack(w13p.to(self.device), perm, K, 2 * shared_N, self.num_bits).unsqueeze(0)
        w13_scale = marlin_permute_scales(w13s.to(self.device), K, 2 * shared_N, gs).unsqueeze(0)
        w2_packed = gptq_marlin_repack(w2p.to(self.device), perm, shared_N, K, self.num_bits).unsqueeze(0)
        w2_scale = marlin_permute_scales(w2s.to(self.device), shared_N, K, gs).unsqueeze(0)

        return w13_packed, w13_scale, w2_packed, w2_scale

    def is_layer_cached(self, moe_layer_idx: int) -> bool:
        """Check if a layer is already prepared in RAM cache."""
        return moe_layer_idx in self._cache

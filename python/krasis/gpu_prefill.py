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

import json
import logging
import os
import time
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
    ):
        self.model_path = model_path
        self.device = device
        self.num_experts = num_experts
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

        # Pre-allocated DMA staging buffers (zero-copy path)
        # Allocated once, pages pre-faulted — eliminates page fault overhead
        self._dma_bufs_initialized: bool = False
        self._dma_buf_w13p: Optional[bytearray] = None
        self._dma_buf_w13s: Optional[bytearray] = None
        self._dma_buf_w2p: Optional[bytearray] = None
        self._dma_buf_w2s: Optional[bytearray] = None
        # Shared expert staging buffers (smaller, 1 expert per layer)
        self._dma_shared_w13p: Optional[bytearray] = None
        self._dma_shared_w13s: Optional[bytearray] = None
        self._dma_shared_w2p: Optional[bytearray] = None
        self._dma_shared_w2s: Optional[bytearray] = None
        self._dma_shared_initialized: bool = False

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
        # w2 (down): [E, N//16, K*nb2]
        self._gpu_w2_packed = torch.zeros(
            E, N // 16, K * nb2,
            dtype=torch.int32, device=self.device,
        )
        # w2 scale: [E, N//gs, K]
        self._gpu_w2_scale = torch.zeros(
            E, N // gs, K,
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
                ).reshape(E, N // 16, K * nb2).to(self.device, non_blocking=True)
                w2_scale = torch.frombuffer(
                    bytearray(w2_scales_bytes), dtype=torch.bfloat16
                ).reshape(E, N // gs, K).to(self.device, non_blocking=True)

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
                    ).reshape(1, shared_N // 16, K * nb2).to(self.device, non_blocking=True)
                    w2_scale = torch.frombuffer(
                        bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(1, shared_N // gs, K).to(self.device, non_blocking=True)

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

        except torch.cuda.OutOfMemoryError:
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
        """Pre-allocate CPU staging buffers for zero-copy DMA.

        Called once before first preload_layer_group(). Allocates 4 bytearrays
        sized for ALL experts in one layer, then touches all pages to map them.
        Subsequent writes from Rust are raw memcpy into pre-faulted memory,
        eliminating ~1430ms/layer of page fault overhead.
        """
        if self._dma_bufs_initialized:
            return

        E = self.num_experts
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # may be > K if padded for Marlin compat
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2

        # Compute sizes in bytes (matching tensor shapes in preload_layer_group)
        w13p_size = E * (K // 16) * (2 * N * nb2) * 4  # int32
        w13s_size = E * (K // gs) * (2 * N) * 2  # bf16
        w2p_size = E * (N // 16) * (K_w2 * nb2) * 4  # int32 (uses padded K)
        w2s_size = E * (N // gs) * K_w2 * 2  # bf16 (uses padded K)

        total = w13p_size + w13s_size + w2p_size + w2s_size
        logger.info(
            "Pre-allocating DMA buffers: %.1f MB "
            "(w13p=%.1f, w13s=%.1f, w2p=%.1f, w2s=%.1f)",
            total / 1e6, w13p_size / 1e6, w13s_size / 1e6,
            w2p_size / 1e6, w2s_size / 1e6,
        )

        t0 = time.perf_counter()
        self._dma_buf_w13p = bytearray(w13p_size)
        self._dma_buf_w13s = bytearray(w13s_size)
        self._dma_buf_w2p = bytearray(w2p_size)
        self._dma_buf_w2s = bytearray(w2s_size)

        # Touch all pages to force mapping (eliminate page faults on first use)
        for buf in [self._dma_buf_w13p, self._dma_buf_w13s,
                    self._dma_buf_w2p, self._dma_buf_w2s]:
            for i in range(0, len(buf), 4096):
                buf[i] = 0

        elapsed = time.perf_counter() - t0
        self._dma_bufs_initialized = True
        logger.info(
            "DMA buffers pre-faulted: %.1f MB in %.1fs (one-time cost)",
            total / 1e6, elapsed,
        )

        # Also allocate shared expert buffers if needed
        if self.n_shared_experts > 0:
            shared_N = self.n_shared_experts * N
            sw13p = 1 * (K // 16) * (2 * shared_N * nb2) * 4
            sw13s = 1 * (K // gs) * (2 * shared_N) * 2
            sw2p = 1 * (shared_N // 16) * (K * nb2) * 4
            sw2s = 1 * (shared_N // gs) * K * 2
            self._dma_shared_w13p = bytearray(sw13p)
            self._dma_shared_w13s = bytearray(sw13s)
            self._dma_shared_w2p = bytearray(sw2p)
            self._dma_shared_w2s = bytearray(sw2s)
            for buf in [self._dma_shared_w13p, self._dma_shared_w13s,
                        self._dma_shared_w2p, self._dma_shared_w2s]:
                for i in range(0, len(buf), 4096):
                    buf[i] = 0
            self._dma_shared_initialized = True

    def preload_layer_group(self, moe_layer_indices):
        """Pre-load expert weights for a group of MoE layers into GPU VRAM.

        Called per-group during layer-grouped prefill. After processing all
        chunks through this group, call free_layer_group() to release VRAM.

        Uses zero-copy path: Rust writes directly into pre-allocated bytearrays
        (pages already mapped), then torch.frombuffer creates a zero-copy view,
        then .to(device) does PCIe DMA. No intermediate allocations.
        """
        assert self._engine is not None, "Layer-grouped mode requires Krasis engine"
        detailed = TIMING.prefill

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        torch.cuda.set_device(self.device)
        K = self.hidden_size
        K_w2 = self._w2_padded_n  # may be > K if padded for Marlin compat
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2
        E = self.num_experts

        # Pre-allocate DMA staging buffers (one-time, pages pre-faulted)
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

            # Zero-copy path: Rust writes directly into pre-allocated buffers
            # (raw memcpy into pre-faulted pages, no allocation overhead)
            self._engine.write_experts_all_into(
                moe_idx,
                self._dma_buf_w13p,
                self._dma_buf_w13s,
                self._dma_buf_w2p,
                self._dma_buf_w2s,
            )

            if detailed:
                t1 = time.perf_counter()
                t_rust += t1 - t0

            # Zero-copy views into pre-allocated buffers (no data movement)
            w13_packed = torch.frombuffer(self._dma_buf_w13p, dtype=torch.int32
                ).reshape(E, K // 16, 2 * N * nb2)
            w13_scale = torch.frombuffer(self._dma_buf_w13s, dtype=torch.bfloat16
                ).reshape(E, K // gs, 2 * N)
            w2_packed = torch.frombuffer(self._dma_buf_w2p, dtype=torch.int32
                ).reshape(E, N // 16, K_w2 * nb2)
            w2_scale = torch.frombuffer(self._dma_buf_w2s, dtype=torch.bfloat16
                ).reshape(E, N // gs, K_w2)

            if detailed:
                t2 = time.perf_counter()
                t_frombuf += t2 - t1

            # PCIe DMA to GPU — synchronous because we reuse the staging buffers
            # for the next layer (must complete before overwriting)
            w13_packed = w13_packed.to(self.device)
            w13_scale = w13_scale.to(self.device)
            w2_packed = w2_packed.to(self.device)
            w2_scale = w2_scale.to(self.device)

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

            shared_N = self.n_shared_experts * N
            for moe_idx in moe_layer_indices:
                if self._dma_shared_initialized:
                    # Zero-copy shared expert path
                    self._engine.write_shared_expert_into(
                        moe_idx,
                        self._dma_shared_w13p,
                        self._dma_shared_w13s,
                        self._dma_shared_w2p,
                        self._dma_shared_w2s,
                    )
                    w13_packed = torch.frombuffer(
                        self._dma_shared_w13p, dtype=torch.int32
                    ).reshape(1, K // 16, 2 * shared_N * nb2).to(self.device)
                    w13_scale = torch.frombuffer(
                        self._dma_shared_w13s, dtype=torch.bfloat16
                    ).reshape(1, K // gs, 2 * shared_N).to(self.device)
                    w2_packed = torch.frombuffer(
                        self._dma_shared_w2p, dtype=torch.int32
                    ).reshape(1, shared_N // 16, K * nb2).to(self.device)
                    w2_scale = torch.frombuffer(
                        self._dma_shared_w2s, dtype=torch.bfloat16
                    ).reshape(1, shared_N // gs, K).to(self.device)
                else:
                    # Fallback: allocating path for shared experts
                    w13p_bytes, w13s_bytes, w2p_bytes, w2s_bytes = (
                        self._engine.get_shared_expert_weights(moe_idx)
                    )
                    w13_packed = torch.frombuffer(
                        bytearray(w13p_bytes), dtype=torch.int32
                    ).reshape(1, K // 16, 2 * shared_N * nb2).to(self.device)
                    w13_scale = torch.frombuffer(
                        bytearray(w13s_bytes), dtype=torch.bfloat16
                    ).reshape(1, K // gs, 2 * shared_N).to(self.device)
                    w2_packed = torch.frombuffer(
                        bytearray(w2p_bytes), dtype=torch.int32
                    ).reshape(1, shared_N // 16, K * nb2).to(self.device)
                    w2_scale = torch.frombuffer(
                        bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(1, shared_N // gs, K).to(self.device)

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
            self._gpu_g_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)

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

    def free_layer_group(self):
        """Free layer group's expert weights from GPU VRAM."""
        detailed = TIMING.prefill
        if detailed:
            t0 = time.perf_counter()

        self._persistent = {}
        self._persistent_shared = {}
        torch.cuda.empty_cache()

        if detailed:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("DMA-FREE: %.0fms (dict clear + empty_cache)", elapsed)

    def _allocate_ao_buffer(self):
        """Allocate GPU buffer for active-only mode (full [E, ...] per layer)."""
        if self._ao_gpu_w13 is not None:
            return

        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
        )

        K = self.hidden_size
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
            E, N // 16, K * nb2, dtype=torch.int32, device=self.device,
        )
        self._ao_gpu_w2_scale = torch.zeros(
            E, N // gs, K, dtype=self.params_dtype, device=self.device,
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
            cache_size, N // 16, K * nb2, dtype=torch.int32, device=self.device,
        )
        self._lru_gpu_w2_scale = torch.zeros(
            cache_size, N // gs, K, dtype=self.params_dtype, device=self.device,
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
                ).reshape(n_dma, N // 16, K * nb2)
            w2s_all = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16
                ).reshape(n_dma, N // gs, K)

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
                    ).reshape(n_dma, N // 16, K * nb2)
                w2s_all = torch.frombuffer(w2s_bytes, dtype=torch.bfloat16
                    ).reshape(n_dma, N // gs, K)

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
                    ).reshape(n_dma, N // 16, K * nb2)
                w2s_all = torch.frombuffer(w2s_bytes, dtype=torch.bfloat16
                    ).reshape(n_dma, N // gs, K)

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
                    ).reshape(1, N // 16, K * nb2).to(self.device)
                w2s = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16
                    ).reshape(1, N // gs, K).to(self.device)

                self._pinned[(layer_idx, expert_id)] = {
                    "w13": w13, "w13_scale": w13s,
                    "w2": w2, "w2_scale": w2s,
                }
                pinned += 1
                total_mb += self._per_expert_vram_bytes() / 1e6
            except torch.cuda.OutOfMemoryError:
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
                        ).reshape(1, N // 16, K * nb2).to(self.device)
                    w2s = torch.frombuffer(bytearray(w2s_bytes), dtype=torch.bfloat16
                        ).reshape(1, N // gs, K).to(self.device)

                    self._pinned[(layer_idx, expert_id)] = {
                        "w13": w13, "w13_scale": w13s,
                        "w2": w2, "w2_scale": w2s,
                    }
                    pinned += 1
                    total_mb += self._per_expert_vram_bytes() / 1e6
                except torch.cuda.OutOfMemoryError:
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

    def _init_hot_cached_static(self, heatmap_path: Optional[str] = None):
        """Initialize hot_cached_static strategy.

        Loads expert heatmap, selects hot experts that fit in VRAM budget,
        copies their Marlin weights to static per-layer GPU buffers,
        and builds lookup tables for GPU/CPU split dispatch.

        Args:
            heatmap_path: Path to expert_heatmap.json. If None, uses
                          accumulated warmup heatmap from self._heatmap.
        """
        assert self._engine is not None, "hot_cached_static requires Krasis engine"

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

        # Compute VRAM budget
        per_expert = self._per_expert_vram_bytes()
        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            # Reserve 3.5 GB for KV cache, attention intermediates, kernel workspace
            headroom = int(3.5 * 1024 * 1024 * 1024)
            budget = free_vram - headroom
            max_experts = max(0, budget // per_expert)
            logger.info(
                "HCS VRAM budget: %.1f MB free, %.1f MB headroom, %d max experts (%.1f MB/expert)",
                free_vram / 1e6, headroom / 1e6, max_experts, per_expert / 1e6,
            )
        except Exception:
            max_experts = 1000

        # Select hot experts (globally hottest, up to budget)
        hot_experts = []
        for (layer, expert), count in sorted_experts:
            if len(hot_experts) >= max_experts:
                break
            hot_experts.append((layer, expert))

        if not hot_experts:
            logger.info("hot_cached_static: no experts pinned (all CPU)")
            self._hcs_initialized = True
            return

        # Group by layer
        layer_experts: dict[int, list[int]] = {}
        for layer, expert in hot_experts:
            layer_experts.setdefault(layer, []).append(expert)

        # Allocate per-layer static buffers and copy weights from engine
        torch.cuda.set_device(self.device)
        K = self.hidden_size
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2

        total_mb = 0.0
        load_start = time.perf_counter()

        for layer_idx in sorted(layer_experts.keys()):
            eids = sorted(layer_experts[layer_idx])
            n = len(eids)

            # Batched weight fetch from Rust engine
            w13_bytes, w13s_bytes, w2_bytes, w2s_bytes = \
                self._engine.get_experts_all_batch(layer_idx, eids)

            w13_cpu = torch.frombuffer(
                bytearray(w13_bytes), dtype=torch.int32
            ).reshape(n, K // 16, 2 * N * nb2)
            w13s_cpu = torch.frombuffer(
                bytearray(w13s_bytes), dtype=torch.bfloat16
            ).reshape(n, K // gs, 2 * N)
            w2_cpu = torch.frombuffer(
                bytearray(w2_bytes), dtype=torch.int32
            ).reshape(n, N // 16, K * nb2)
            w2s_cpu = torch.frombuffer(
                bytearray(w2s_bytes), dtype=torch.bfloat16
            ).reshape(n, N // gs, K)

            # Upload to GPU
            w13 = w13_cpu.to(self.device, non_blocking=True)
            w13s = w13s_cpu.to(self.device, non_blocking=True)
            w2 = w2_cpu.to(self.device, non_blocking=True)
            w2s = w2s_cpu.to(self.device, non_blocking=True)

            self._hcs_buffers[layer_idx] = {
                "w13": w13, "w13_scale": w13s,
                "w2": w2, "w2_scale": w2s,
            }

            # Build lookup table: global expert ID → local slot (-1 if cold)
            lookup = torch.full(
                (self.num_experts,), -1, dtype=torch.int32, device=self.device,
            )
            for local_slot, eid in enumerate(eids):
                lookup[eid] = local_slot
            self._hcs_lookup[layer_idx] = lookup
            self._hcs_num_pinned[layer_idx] = n

            # Kernel helpers per layer
            self._hcs_g_idx[layer_idx] = torch.empty(
                n, 0, dtype=torch.int32, device=self.device,
            )
            self._hcs_sort_idx[layer_idx] = torch.empty(
                n, 0, dtype=torch.int32, device=self.device,
            )
            self._hcs_gating_1[layer_idx] = torch.empty(
                1, n, device=self.device,
            )

            # Remap expert biases for hot experts (GPT OSS)
            if self.swiglu_limit > 0 and layer_idx in self._expert_biases:
                full_biases = self._expert_biases[layer_idx]
                gu_full = full_biases["gate_up_bias"]  # [E, 2N]
                dn_full = full_biases["down_bias"]     # [E, K]
                eids_tensor = torch.tensor(eids, dtype=torch.long)
                self._hcs_biases[layer_idx] = {
                    "gate_up_bias": gu_full[eids_tensor],  # [n_hot, 2N]
                    "down_bias": dn_full[eids_tensor],     # [n_hot, K]
                }

            layer_mb = (w13.nbytes + w13s.nbytes + w2.nbytes + w2s.nbytes) / 1e6
            total_mb += layer_mb

        # Ensure workspace is allocated
        if self._workspace is None:
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)

        # Populate self._pinned so active_only prefill path skips DMA for hot experts.
        # Uses views into HCS buffers — zero extra VRAM.
        for layer_idx in sorted(layer_experts.keys()):
            eids = sorted(layer_experts[layer_idx])
            bufs = self._hcs_buffers[layer_idx]
            for local_slot, eid in enumerate(eids):
                self._pinned[(layer_idx, eid)] = {
                    "w13": bufs["w13"][local_slot:local_slot+1],
                    "w13_scale": bufs["w13_scale"][local_slot:local_slot+1],
                    "w2": bufs["w2"][local_slot:local_slot+1],
                    "w2_scale": bufs["w2_scale"][local_slot:local_slot+1],
                }

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - load_start

        # Per-layer summary
        layers_with_pins = {l: len(e) for l, e in layer_experts.items()}
        logger.info(
            "hot_cached_static initialized on %s: %d experts across %d layers "
            "(%.1f MB) in %.2fs, min/max per layer: %d/%d",
            self.device, len(hot_experts), len(layer_experts), total_mb, elapsed,
            min(layers_with_pins.values()), max(layers_with_pins.values()),
        )

        self._hcs_initialized = True

    def _forward_hot_cached_static(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with hot_cached_static: hot experts on GPU, cold on CPU, parallel.

        For each MoE layer:
          1. Lookup table splits experts into GPU (hot) vs CPU (cold)
          2. Submit cold experts to CPU engine (async, non-blocking)
             with hot expert IDs set to -1 (engine skips them)
          3. Run fused_marlin_moe on static GPU buffer for hot experts
             with cold expert weights zeroed
          4. Sync CPU result + combine: output = gpu_output + cpu_output

        For M=1 decode: CPU finishes within GPU's ~1.5ms Marlin kernel window.
        """

        M = x.shape[0]
        timing = TIMING.decode and M == 1

        if timing:
            t_start = time.perf_counter()

        # Get layer's hot expert info
        bufs = self._hcs_buffers.get(moe_layer_idx)
        lookup = self._hcs_lookup.get(moe_layer_idx)
        n_hot = self._hcs_num_pinned.get(moe_layer_idx, 0)

        if n_hot == 0 or bufs is None:
            # No hot experts for this layer — all on CPU
            return self._forward_hcs_cpu_only(moe_layer_idx, x, topk_ids, topk_weights)

        # Determine GPU vs CPU split
        if M == 1:
            topk_list = topk_ids[0].tolist()
            local_slots = [int(lookup[eid]) for eid in topk_list]
            has_cold = any(s < 0 for s in local_slots)
            all_cold = all(s < 0 for s in local_slots)
        else:
            lookup_result = lookup[topk_ids.long()]  # [M, topk]
            has_cold = (lookup_result < 0).any().item()
            all_cold = (lookup_result < 0).all().item()

        if all_cold:
            return self._forward_hcs_cpu_only(moe_layer_idx, x, topk_ids, topk_weights)

        # ── Step 1: Submit cold experts to CPU engine (async) ──
        if has_cold:
            if M == 1:
                cpu_ids_list = [
                    eid if local_slots[j] < 0 else -1
                    for j, eid in enumerate(topk_list)
                ]
                cpu_ids = torch.tensor([cpu_ids_list], dtype=torch.int32)
            else:
                cpu_ids = topk_ids.clone()
                hot_mask = lookup[topk_ids.long()] >= 0
                cpu_ids[hot_mask] = -1
                cpu_ids = cpu_ids.to(torch.int32)

            act_cpu = x.detach().cpu().contiguous()
            ids_cpu = cpu_ids.cpu().contiguous()
            wts_cpu = topk_weights.detach().cpu().contiguous()

            act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
            ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
            wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()

            self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, M)

        if timing:
            t_cpu_submitted = time.perf_counter()

        # ── Step 2: GPU hot expert computation ──
        if M == 1:
            # Build local IDs and masked weights
            gpu_local_ids = []
            gpu_weights = topk_weights[0].tolist()
            for j, slot in enumerate(local_slots):
                if slot >= 0:
                    gpu_local_ids.append(slot)
                else:
                    gpu_local_ids.append(0)  # dummy slot, weight zeroed
                    gpu_weights[j] = 0.0

            local_ids = torch.tensor(
                [gpu_local_ids], dtype=topk_ids.dtype, device=self.device,
            )
            gpu_topk_weights = torch.tensor(
                [gpu_weights], dtype=topk_weights.dtype, device=self.device,
            )
            gating = self._hcs_gating_1[moe_layer_idx]
        else:
            # Batch remap for M > 1
            lookup_result = lookup[topk_ids.long()]  # [M, topk]
            cold_mask = lookup_result < 0
            local_ids = lookup_result.clone().to(topk_ids.dtype)
            local_ids[cold_mask] = 0  # dummy slot
            gpu_topk_weights = topk_weights.clone()
            gpu_topk_weights[cold_mask] = 0.0
            gating = torch.empty(M, n_hot, device=self.device)

        # Use CUDA graph if available (M=1 only), otherwise standard kernel
        if (M == 1 and self._hcs_cuda_graphs_enabled
                and moe_layer_idx in self._hcs_cuda_graphs):
            gpu_output = self._forward_hcs_graphed(
                moe_layer_idx, local_ids, gpu_topk_weights, x,
            )
        else:
            g_idx = self._hcs_g_idx[moe_layer_idx]
            sort_idx = self._hcs_sort_idx[moe_layer_idx]

            if M > 1:
                self._workspace.zero_()

            # Get remapped biases for hot experts (GPT OSS)
            hcs_bias = self._hcs_biases.get(moe_layer_idx, {})
            gpu_output = self._call_fused_moe(
                moe_layer_idx=moe_layer_idx,
                hidden_states=x,
                w1=bufs["w13"],
                w2=bufs["w2"],
                w1_scale=bufs["w13_scale"],
                w2_scale=bufs["w2_scale"],
                gating_output=gating,
                topk_weights=gpu_topk_weights,
                topk_ids=local_ids,
                global_num_experts=n_hot,
                g_idx1=g_idx,
                g_idx2=g_idx,
                sort_indices1=sort_idx,
                sort_indices2=sort_idx,
                workspace=self._workspace,
                bias_gate_up=hcs_bias.get("gate_up_bias"),
                bias_down=hcs_bias.get("down_bias"),
            ).to(x.dtype)

        if timing:
            torch.cuda.synchronize(self.device)
            t_gpu_done = time.perf_counter()

        # ── Step 3: Sync CPU result and combine ──
        if has_cold:
            cpu_output_bytes = self._engine.sync_forward()
            cpu_raw = torch.frombuffer(
                bytearray(cpu_output_bytes), dtype=torch.bfloat16
            ).reshape(M, self.hidden_size).to(self.device)

            # CPU engine applies routed_scaling_factor — divide it out
            # so forward() can apply it uniformly to the combined output
            if self.routed_scaling_factor != 1.0:
                cpu_raw = cpu_raw / self.routed_scaling_factor

            gpu_output = gpu_output + cpu_raw

        if timing:
            t_end = time.perf_counter()
            gpu_ms = (t_gpu_done - (t_cpu_submitted if has_cold else t_start)) * 1000
            logger.info(
                "HCS-TIMING L%d: cpu_submit=%.1fms gpu=%.1fms cpu_sync=%.1fms total=%.1fms cold=%s",
                moe_layer_idx,
                (t_cpu_submitted - t_start) * 1000 if has_cold else 0,
                gpu_ms,
                (t_end - t_gpu_done) * 1000 if has_cold else 0,
                (t_end - t_start) * 1000,
                has_cold,
            )

        # Update heatmap for potential re-pinning
        if self._heatmap is not None:
            if M == 1:
                for eid in topk_list:
                    key = (moe_layer_idx, eid)
                    self._heatmap[key] = self._heatmap.get(key, 0) + 1
            else:
                for eid in torch.unique(topk_ids).tolist():
                    key = (moe_layer_idx, eid)
                    self._heatmap[key] = self._heatmap.get(key, 0) + 1

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
        act_cpu = x.detach().cpu().contiguous()
        ids_cpu = topk_ids.detach().cpu().to(torch.int32).contiguous()
        wts_cpu = topk_weights.detach().cpu().contiguous()

        act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
        ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
        wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()

        self._engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, M)
        output_bytes = self._engine.sync_forward()

        output = torch.frombuffer(
            bytearray(output_bytes), dtype=torch.bfloat16
        ).reshape(M, self.hidden_size).to(self.device)

        # CPU engine applies routed_scaling_factor — divide it out
        if self.routed_scaling_factor != 1.0:
            output = output / self.routed_scaling_factor

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

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

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
                    workspace=self._workspace,
                    num_bits=self.num_bits, is_k_full=True,
                )
            torch.cuda.synchronize(self.device)

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                io["output"] = fused_marlin_moe(
                    hidden_states=io["x"],
                    w1=bufs["w13"], w2=bufs["w2"],
                    w1_scale=bufs["w13_scale"], w2_scale=bufs["w2_scale"],
                    gating_output=io["gating"],
                    topk_weights=io["weights"], topk_ids=io["local_ids"],
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

        # Return clone (graph tensor is reused next call)
        return io["output"].clone()

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
        gating_output = torch.empty(M, self.num_experts, device=self.device)

        self._workspace.zero_()

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
            global_num_experts=self.num_experts,
            g_idx1=self._gpu_g_idx,
            g_idx2=self._gpu_g_idx,
            sort_indices1=self._gpu_sort_idx,
            sort_indices2=self._gpu_sort_idx,
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
    ) -> torch.Tensor:
        """Run GPU MoE forward for one layer using cached INT4 Marlin weights.

        Args:
            moe_layer_idx: 0-based MoE layer index
            hidden_states: [M, K] input on GPU
            topk_ids: [M, top_k] expert indices on GPU
            topk_weights: [M, top_k] routing weights on GPU

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

        if self._prefill_mode == "hot_cached_static" and self._hcs_initialized:
            if M == 1:
                # M=1 decode: hot experts on GPU static buffers, cold on CPU (zero DMA)
                output = self._forward_hot_cached_static(moe_layer_idx, x, topk_ids, topk_weights)
            elif self._persistent and moe_layer_idx in self._persistent:
                # M>1 prefill: use layer_grouped preloaded experts (all experts in VRAM)
                output = self._forward_persistent(moe_layer_idx, x, topk_ids, topk_weights)
            else:
                # M>1 prefill fallback: DMA-based active_only
                self._invalidate_compact()
                output = self._forward_active_only(moe_layer_idx, x, topk_ids, topk_weights)
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
        elif self._prefill_mode in ("persistent", "layer_grouped") and \
                self._persistent and moe_layer_idx in self._persistent:
            output = self._forward_persistent(moe_layer_idx, x, topk_ids, topk_weights)
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
            ).reshape(actual, N // 16, K * nb2)
            w2_scales = torch.frombuffer(
                bytearray(w2_scales_bytes), dtype=torch.bfloat16
            ).reshape(actual, N // gs, K)

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
            ).reshape(1, shared_N // 16, K * nb2).to(self.device)
            w2_scale = torch.frombuffer(
                bytearray(w2s_bytes), dtype=torch.bfloat16
            ).reshape(1, shared_N // gs, K).to(self.device)
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

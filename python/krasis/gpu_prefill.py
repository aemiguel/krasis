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

import torch

logger = logging.getLogger(__name__)

# Marlin quantization constants
GROUP_SIZE = 128


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

        # Use engine's group_size if available, else default 128
        if krasis_engine is not None:
            self._group_size = krasis_engine.group_size()
        else:
            self._group_size = GROUP_SIZE
        logger.info("GPU prefill group_size=%d", self._group_size)

        # Persistent/layer-grouped expert buffers
        self._prefill_mode = "chunked"  # "persistent", "layer_grouped", or "chunked"
        self._persistent: Optional[dict[int, dict[str, torch.Tensor]]] = None
        self._persistent_shared: Optional[dict[int, dict[str, torch.Tensor]]] = None

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

    def _per_expert_vram_bytes(self) -> int:
        """Calculate per-expert VRAM usage in bytes for Marlin format."""
        K = self.hidden_size
        N = self.intermediate_size
        gs = self._group_size
        nb2 = self.num_bits // 2  # 2 for INT4, 4 for INT8
        w13_bytes = (K // 16) * (2 * N * nb2) * 4 + (K // gs) * (2 * N) * 2
        w2_bytes = (N // 16) * (K * nb2) * 4 + (N // gs) * K * 2
        return w13_bytes + w2_bytes

    def _select_prefill_mode(self):
        """Select prefill mode based on expert_divisor setting.

        - divisor=0 or no engine: "chunked" (current DMA-per-layer behavior)
        - divisor=1: "persistent" (all experts pre-loaded in VRAM, zero DMA)
        - divisor>=2: "layer_grouped" (load experts per-group-of-layers, O(groups) DMA)
        """
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

    def prepare_all_layers(self):
        """Pre-prepare all MoE layers at startup.

        Persistent mode: pre-load all experts into GPU VRAM.
        Selective/Chunked engine: no-op (weights read from engine on demand).
        Legacy mode: loads from disk cache.
        """
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

    def preload_layer_group(self, moe_layer_indices):
        """Pre-load expert weights for a group of MoE layers into GPU VRAM.

        Called per-group during layer-grouped prefill. After processing all
        chunks through this group, call free_layer_group() to release VRAM.
        """
        assert self._engine is not None, "Layer-grouped mode requires Krasis engine"

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

        for moe_idx in moe_layer_indices:
            w13_packed_bytes = self._engine.get_expert_w13_packed(moe_idx, 0, E)
            w13_scales_bytes = self._engine.get_expert_w13_scales(moe_idx, 0, E)
            w2_packed_bytes = self._engine.get_expert_w2_packed(moe_idx, 0, E)
            w2_scales_bytes = self._engine.get_expert_w2_scales(moe_idx, 0, E)

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

            del w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes

        # Pre-load shared experts
        if self.n_shared_experts > 0:
            shared_N = self.n_shared_experts * N
            for moe_idx in moe_layer_indices:
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

        # Allocate workspace and index tensors (once, reused across groups)
        if self._workspace is None:
            self._workspace = marlin_make_workspace(self.device, max_blocks_per_sm=4)
        if self._gpu_g_idx is None:
            self._gpu_g_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)
            self._gpu_sort_idx = torch.empty(E, 0, dtype=torch.int32, device=self.device)

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - load_start
        alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
        logger.info(
            "Layer group loaded: %d MoE layers = %.1f MB in %.2fs (GPU total: %.1f MB)",
            len(moe_layer_indices), total_mb, elapsed, alloc_mb,
        )

    def free_layer_group(self):
        """Free layer group's expert weights from GPU VRAM."""
        self._persistent = {}
        self._persistent_shared = {}
        torch.cuda.empty_cache()

    def _forward_persistent(
        self,
        moe_layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using pre-loaded persistent expert buffers. Zero DMA."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        M = x.shape[0]
        buffers = self._persistent[moe_layer_idx]
        gating_output = torch.empty(M, self.num_experts, device=self.device)

        self._workspace.zero_()

        output = fused_marlin_moe(
            hidden_states=x,
            w1=buffers["w13_packed"],
            w2=buffers["w2_packed"],
            w1_scale=buffers["w13_scale"],
            w2_scale=buffers["w2_scale"],
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

        torch.cuda.synchronize(self.device)

        if self._prefill_mode in ("persistent", "layer_grouped") and \
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

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
NUM_BITS = 4


def _quantize_and_pack_gpu(w: torch.Tensor, group_size: int = GROUP_SIZE):
    """Quantize [K, N] weight to INT4 packed on GPU.

    Input weight must be transposed to [K, N] (K = input/reduction dim, N = output dim).
    This matches the GPTQ convention where packing is along the K dimension.

    Returns:
        packed: [K//8, N] int32 — 8 INT4 values per int32, packed along K
        scale: [K//group_size, N] same dtype as w
    """
    K, N = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    assert K % 8 == 0, f"K={K} not divisible by 8"

    w_float = w.float()
    w_grouped = w_float.reshape(K // group_size, group_size, N)

    # Symmetric quantization: scale = max(|max|/7, |min|/8) per group
    max_val = w_grouped.amax(dim=1, keepdim=True)
    min_val = w_grouped.amin(dim=1, keepdim=True)
    scale = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0)
    scale = scale.clamp(min=1e-10)  # avoid division by zero

    # Quantize to [-8, 7], then bias to [0, 15] for unsigned INT4 packing
    w_q = (w_grouped / scale).round().clamp(-8, 7).to(torch.int32) + 8
    w_q = w_q.reshape(K, N)
    scale = scale.squeeze(1).to(w.dtype)  # [K//group_size, N], match model dtype

    # Pack 8 INT4 values into one int32 (bits 0-3, 4-7, ..., 28-31)
    w_q = w_q.reshape(K // 8, 8, N)
    packed = w_q[:, 0, :].clone()
    for i in range(1, 8):
        packed |= w_q[:, i, :] << (4 * i)

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
    ):
        self.model_path = model_path
        self.device = device
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.params_dtype = params_dtype
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense

        # Chunk size: how many experts fit in one GPU buffer load
        if chunk_size is None:
            chunk_size = self._auto_chunk_size()
        self.chunk_size = min(chunk_size, num_experts)
        self.num_chunks = (num_experts + self.chunk_size - 1) // self.chunk_size

        # RAM cache: layer_idx -> {w13_packed, w13_scale, w2_packed, w2_scale}
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

        # Safetensors index (lazy loaded)
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

        # Shared expert GPU cache (separate from chunked buffer)
        self._shared_cache: dict[int, dict[str, torch.Tensor]] = {}

        logger.info(
            "GpuPrefillManager: experts=%d, hidden=%d, intermediate=%d, "
            "chunk_size=%d, num_chunks=%d, shared=%d, scale=%.3f",
            num_experts, hidden_size, intermediate_size,
            self.chunk_size, self.num_chunks,
            n_shared_experts, routed_scaling_factor,
        )

    def _auto_chunk_size(self) -> int:
        """Estimate chunk size that fits in available VRAM."""
        # Per-expert VRAM for INT4 Marlin:
        # w13: [K//16, 2*N*2] int32 + [K//128, 2*N] scale
        # w2: [N//16, K*2] int32 + [N//128, K] scale
        K = self.hidden_size
        N = self.intermediate_size
        w13_bytes = (K // 16) * (2 * N * 2) * 4 + (K // GROUP_SIZE) * (2 * N) * 2
        w2_bytes = (N // 16) * (K * 2) * 4 + (N // GROUP_SIZE) * K * 2
        per_expert = w13_bytes + w2_bytes

        try:
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            # Use at most 50% of free VRAM for expert buffer
            budget = int(free_vram * 0.5)
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

        # w13 (gate+up combined): [E, K//16, 2*N*(NUM_BITS//2)]
        self._gpu_w13_packed = torch.zeros(
            E, K // 16, 2 * N * (NUM_BITS // 2),
            dtype=torch.int32, device=self.device,
        )
        # w13 scale: [E, K//GROUP_SIZE, 2*N]
        self._gpu_w13_scale = torch.zeros(
            E, K // GROUP_SIZE, 2 * N,
            dtype=self.params_dtype, device=self.device,
        )
        # w2 (down): [E, N//16, K*(NUM_BITS//2)]
        self._gpu_w2_packed = torch.zeros(
            E, N // 16, K * (NUM_BITS // 2),
            dtype=torch.int32, device=self.device,
        )
        # w2 scale: [E, N//GROUP_SIZE, K]
        self._gpu_w2_scale = torch.zeros(
            E, N // GROUP_SIZE, K,
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

    def prepare_layer(self, moe_layer_idx: int):
        """Quantize and cache one MoE layer's expert weights for GPU prefill.

        Reads from safetensors, quantizes BF16→INT4 on GPU, repacks to Marlin,
        then moves result to CPU RAM cache.
        """
        if moe_layer_idx in self._cache:
            return  # Already cached

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
            w13_packed, w13_scale = _quantize_and_pack_gpu(w13_gpu, GROUP_SIZE)
            del w13, w13_gpu

            down_gpu = down.to(self.device)
            w2_packed, w2_scale = _quantize_and_pack_gpu(down_gpu, GROUP_SIZE)
            del down, down_gpu

            # Marlin repack (per-expert, CUDA kernel)
            perm = torch.empty(0, dtype=torch.int32, device=self.device)
            w13_repacked = gptq_marlin_repack(w13_packed, perm, K, 2 * N, NUM_BITS)
            w2_repacked = gptq_marlin_repack(w2_packed, perm, N, K, NUM_BITS)

            # Permute scales for Marlin
            w13_scale_perm = marlin_permute_scales(
                w13_scale, K, 2 * N, GROUP_SIZE,
            )
            w2_scale_perm = marlin_permute_scales(
                w2_scale, N, K, GROUP_SIZE,
            )

            # Move to CPU for RAM cache
            all_w13_packed.append(w13_repacked.cpu())
            all_w13_scale.append(w13_scale_perm.cpu())
            all_w2_packed.append(w2_repacked.cpu())
            all_w2_scale.append(w2_scale_perm.cpu())

            del w13_packed, w13_scale, w2_packed, w2_scale
            del w13_repacked, w2_repacked, w13_scale_perm, w2_scale_perm

        # Stack into [num_experts, ...] tensors
        self._cache[moe_layer_idx] = {
            "w13_packed": torch.stack(all_w13_packed),  # [E, K//16, 2*N*2]
            "w13_scale": torch.stack(all_w13_scale),    # [E, K//128, 2*N]
            "w2_packed": torch.stack(all_w2_packed),    # [E, N//16, K*2]
            "w2_scale": torch.stack(all_w2_scale),      # [E, N//128, K]
        }

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
        w13_packed, w13_scale = _quantize_and_pack_gpu(w13, GROUP_SIZE)
        del w13

        down_gpu = down.t().contiguous().to(self.device)  # [shared_N, K]
        del down
        w2_packed, w2_scale = _quantize_and_pack_gpu(down_gpu, GROUP_SIZE)
        del down_gpu

        perm = torch.empty(0, dtype=torch.int32, device=self.device)
        w13_repacked = gptq_marlin_repack(w13_packed, perm, K, 2 * shared_N, NUM_BITS)
        w2_repacked = gptq_marlin_repack(w2_packed, perm, shared_N, K, NUM_BITS)
        w13_scale_perm = marlin_permute_scales(w13_scale, K, 2 * shared_N, GROUP_SIZE)
        w2_scale_perm = marlin_permute_scales(w2_scale, shared_N, K, GROUP_SIZE)

        # Shared expert stored as single expert (unsqueeeze to [1, ...])
        self._shared_cache[moe_layer_idx] = {
            "w13_packed": w13_repacked.unsqueeze(0).cpu(),
            "w13_scale": w13_scale_perm.unsqueeze(0).cpu(),
            "w2_packed": w2_repacked.unsqueeze(0).cpu(),
            "w2_scale": w2_scale_perm.unsqueeze(0).cpu(),
            "shared_N": shared_N,
        }

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

        self._allocate_gpu_buffer()

        # Ensure this layer is prepared
        if moe_layer_idx not in self._cache:
            self.prepare_layer(moe_layer_idx)

        cache = self._cache[moe_layer_idx]
        M = hidden_states.shape[0]
        x = hidden_states

        if self.num_chunks == 1:
            # Fast path: all experts fit in one buffer
            self._gpu_w13_packed.copy_(cache["w13_packed"])
            self._gpu_w13_scale.copy_(cache["w13_scale"])
            self._gpu_w2_packed.copy_(cache["w2_packed"])
            self._gpu_w2_scale.copy_(cache["w2_scale"])

            # Fake gating_output (only used for shape assertion inside kernel)
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
                num_bits=NUM_BITS,
                is_k_full=True,
            ).to(x.dtype)
        else:
            # Multi-chunk: process experts in chunks, accumulate output
            output = torch.zeros(M, self.hidden_size, dtype=x.dtype, device=self.device)
            gating_output = torch.empty(M, self.chunk_size, device=self.device)

            for chunk_idx in range(self.num_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, self.num_experts)
                actual = end - start

                # Load chunk into GPU buffer
                self._gpu_w13_packed[:actual].copy_(cache["w13_packed"][start:end])
                self._gpu_w13_scale[:actual].copy_(cache["w13_scale"][start:end])
                self._gpu_w2_packed[:actual].copy_(cache["w2_packed"][start:end])
                self._gpu_w2_scale[:actual].copy_(cache["w2_scale"][start:end])

                # Remap expert IDs for this chunk
                chunk_mask = (topk_ids >= start) & (topk_ids < end)
                chunk_ids = torch.where(chunk_mask, topk_ids - start, torch.zeros_like(topk_ids))
                chunk_weights = torch.where(chunk_mask, topk_weights, torch.zeros_like(topk_weights))

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
                    num_bits=NUM_BITS,
                    is_k_full=True,
                ).to(x.dtype)

                output += chunk_output

        # Apply routed scaling factor and add shared expert
        if self.n_shared_experts > 0:
            shared_output = self._shared_expert_forward(moe_layer_idx, x)
            output = self.routed_scaling_factor * output + shared_output
        elif self.routed_scaling_factor != 1.0:
            output *= self.routed_scaling_factor

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

        if moe_layer_idx not in self._shared_cache:
            self.prepare_shared_expert(moe_layer_idx)

        cache = self._shared_cache[moe_layer_idx]
        M = hidden_states.shape[0]

        # Load shared expert to GPU (single expert)
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
            num_bits=NUM_BITS,
            is_k_full=True,
        ).to(hidden_states.dtype)

        return output

    def is_layer_cached(self, moe_layer_idx: int) -> bool:
        """Check if a layer is already prepared in RAM cache."""
        return moe_layer_idx in self._cache

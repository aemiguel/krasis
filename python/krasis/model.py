"""Full model: embedding → N transformer layers → LM head.

Streaming attention architecture: ALL attention on GPU0, experts
split across all GPUs via EP. GPU1+ reserved for EP prefill and
HCS decode experts (zero attention allocation).

Loading sequence:
  Phase 0: System RAM budget check (refuse to run if insufficient)
  Phase 1: GPU weights — all on GPU0 (streaming BF16→INT8, one tensor at a time)
  Phase 2: CPU expert weights (Krasis Rust engine, INT4)
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
import torch

from krasis.timing import TIMING

from krasis.config import ModelConfig, PPRankConfig, QuantConfig, build_pp_ranks, compute_pp_partition, cache_dir_for_model
from krasis.weight_loader import WeightLoader, int8_linear
from krasis.layer import TransformerLayer
from krasis.kv_cache import PagedKVCache, SequenceKVState
from krasis.sampler import sample

# Debug: env var to enable per-layer decode logging
_DEBUG_DECODE = os.environ.get("KRASIS_DEBUG_DECODE", "") == "1"
from krasis.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _weight_dtype_code(t: torch.Tensor) -> int:
    """Map torch dtype to Rust GpuWeight dtype code: 0=BF16, 1=FP32, 2=FP16."""
    if t.dtype == torch.float32:
        return 1
    elif t.dtype == torch.float16:
        return 2
    return 0  # BF16 (default)


# GPU-to-GPU P2P transfer may silently fail on some systems (returns zeros).
# Detect this once at import time and use CPU bounce if needed.
_p2p_works: Optional[bool] = None

def _check_p2p() -> bool:
    """Test if direct GPU-to-GPU transfer works."""
    global _p2p_works
    if _p2p_works is not None:
        return _p2p_works
    if torch.cuda.device_count() < 2:
        _p2p_works = True
        return True
    test = torch.tensor([42, 137], dtype=torch.float32, device='cuda:0')
    transferred = test.to('cuda:1')
    torch.cuda.synchronize()
    _p2p_works = bool((transferred.cpu() == test.cpu()).all())
    if not _p2p_works:
        logger.warning("GPU P2P transfer broken — using CPU bounce for cross-device transfers")
    return _p2p_works


def _to_device(tensor: torch.Tensor, target: torch.device) -> torch.Tensor:
    """Transfer tensor to target device, using CPU bounce if P2P is broken."""
    if tensor.device == target:
        return tensor
    if _check_p2p():
        return tensor.to(target)
    return tensor.cpu().to(target)


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to INT8 or BF16 linear based on weight type."""
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)


def _read_meminfo():
    """Read /proc/meminfo and return dict of key -> value in KB."""
    result = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    result[key] = int(parts[1])
    except (OSError, ValueError):
        pass
    return result


def _vram_checkpoint(label: str, devices=None):
    """Print VRAM usage at a checkpoint for diagnostics."""
    import torch
    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    for dev in devices:
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        free, total = torch.cuda.mem_get_info(dev)
        used = total - free
        print(
            f"  \033[33m[VRAM {label}]\033[0m {dev}: "
            f"alloc={alloc // (1024*1024)} MB, reserved={reserved // (1024*1024)} MB, "
            f"used={used // (1024*1024)} MB, free={free // (1024*1024)} MB, total={total // (1024*1024)} MB",
            flush=True,
        )
        logger.info(
            "VRAM_CHECKPOINT [%s] %s: alloc=%d MB, reserved=%d MB, driver_used=%d MB, free=%d MB, total=%d MB",
            label, dev, alloc >> 20, reserved >> 20, used >> 20, free >> 20, total >> 20,
        )


def _read_vmrss_kb() -> int:
    """Read VmRSS from /proc/self/status in KB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _estimate_expert_ram_gb(cfg, cpu_expert_bits: int = 4, group_size: int = 128) -> float:
    """Estimate system RAM needed for CPU expert weights in GB.

    Calculates per-expert byte sizes for the unified weight format:
      w13 (gate+up concat, transposed): packed [K//8, 2*N] + scales [K//gs, 2*N]
      w2 (down, transposed):            packed [N//8, K] + scales [N//gs, K]

    Plus shared experts if any.
    """
    h = cfg.hidden_size           # K for w13, N_down for w2
    m = cfg.moe_intermediate_size  # N for w13, K_down for w2
    gs = group_size
    n_experts = cfg.n_routed_experts
    num_moe_layers = cfg.num_moe_layers

    if cpu_expert_bits == 4:
        # w13 packed: (h//8) * 2*m * 4 bytes, w13 scales: (h//gs) * 2*m * 2 bytes
        w13_bytes = (h // 8) * (2 * m) * 4 + (h // gs) * (2 * m) * 2
        # w2 packed: (m//8) * h * 4 bytes, w2 scales: (m//gs) * h * 2 bytes
        w2_bytes = (m // 8) * h * 4 + (m // gs) * h * 2
    else:
        # INT8: gate/up each m*h + scales, down h*m + scales
        w13_bytes = 2 * m * h + (h // gs) * (2 * m) * 2
        w2_bytes = h * m + (m // gs) * h * 2

    per_expert = w13_bytes + w2_bytes
    routed_total = num_moe_layers * n_experts * per_expert

    # Shared experts
    shared_total = 0
    if cfg.n_shared_experts > 0:
        shared_m = cfg.n_shared_experts * m
        if cpu_expert_bits == 4:
            s_w13 = (h // 8) * (2 * shared_m) * 4 + (h // gs) * (2 * shared_m) * 2
            s_w2 = (shared_m // 8) * h * 4 + (shared_m // gs) * h * 2
        else:
            s_w13 = 2 * shared_m * h + (h // gs) * (2 * shared_m) * 2
            s_w2 = h * shared_m + (shared_m // gs) * h * 2
        shared_total = num_moe_layers * (s_w13 + s_w2)

    return (routed_total + shared_total) / 1e9


def _check_system_ram(cfg, cpu_expert_bits: int = 4, force_load: bool = False):
    """Check system RAM budget before loading expert weights.

    Reads /proc/meminfo, estimates RAM needed, and refuses to run if
    the model would exceed 95% of MemTotal. Warns at 90%.

    Args:
        cfg: ModelConfig with model dimensions
        cpu_expert_bits: 4 or 8
        force_load: If True, log error but don't raise

    Raises:
        RuntimeError: If insufficient RAM and force_load=False
    """
    meminfo = _read_meminfo()
    if not meminfo:
        logger.warning("Could not read /proc/meminfo — skipping RAM budget check")
        return

    mem_total_gb = meminfo.get("MemTotal", 0) / 1024 / 1024
    mem_avail_gb = meminfo.get("MemAvailable", 0) / 1024 / 1024

    expert_ram_gb = _estimate_expert_ram_gb(cfg, cpu_expert_bits)
    # Compute non-expert RAM overhead from model dimensions:
    # - GPU weights (attention, embedding, lm_head, norms, gates, shared experts) are loaded
    #   to GPU but PyTorch/CUDA uses staging buffers in system RAM during loading.
    # - Estimate ~2x the BF16 weight size for staging overhead (load + transfer).
    # - Add 10% of total expert RAM for PyTorch/CUDA runtime, page tables, etc.
    h = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    # Attention projection params per layer
    if cfg.kv_lora_rank is not None:
        # MLA: Q (or q_a + q_b), KV_A, KV_B, O
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim or 128
        qk_rope = cfg.qk_rope_head_dim or 64
        v_hd = cfg.v_head_dim or 128
        q_params = h * n_heads * (qk_nope + qk_rope)
        if cfg.q_lora_rank:
            q_params = h * cfg.q_lora_rank + cfg.q_lora_rank * n_heads * (qk_nope + qk_rope)
        kv_a_params = h * (kv_lora + qk_rope)
        kv_b_params = kv_lora * n_heads * (qk_nope + v_hd)
        o_params = n_heads * v_hd * h
        attn_params = q_params + kv_a_params + kv_b_params + o_params
    else:
        # GQA: Q + K + V + O
        head_dim = cfg.gqa_head_dim or (h // n_heads)
        attn_params = h * n_heads * head_dim + 2 * h * n_kv * head_dim + n_heads * head_dim * h
    attn_bytes = attn_params * 2 * n_layers  # BF16
    # Embedding + lm_head
    embed_bytes = cfg.vocab_size * h * 2
    lm_head_bytes = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h * 2
    # Gates, norms, shared experts
    n_moe = n_layers - cfg.first_k_dense_replace
    gate_bytes = h * cfg.n_routed_experts * 2 * n_moe if cfg.n_routed_experts > 0 else 0
    norm_bytes = 2 * h * 2 * n_layers
    shared_inter = cfg.shared_expert_intermediate_size or (cfg.n_shared_experts * cfg.moe_intermediate_size)
    shared_bytes = 3 * h * shared_inter * 2 * n_moe if shared_inter > 0 else 0
    gpu_weight_bytes = attn_bytes + embed_bytes + lm_head_bytes + gate_bytes + norm_bytes + shared_bytes
    # Staging overhead (2x GPU weights for load/transfer) + runtime overhead (10% of expert RAM)
    overhead_gb = (gpu_weight_bytes * 2 / 1e9) + (expert_ram_gb * 0.10)
    overhead_gb = max(overhead_gb, 2.0)  # floor at 2 GB for minimal PyTorch/CUDA runtime
    total_needed_gb = expert_ram_gb + overhead_gb

    logger.info(
        "RAM budget: experts=%.1f GB, overhead=%.1f GB, total=%.1f GB needed | "
        "system: %.1f GB total, %.1f GB available",
        expert_ram_gb, overhead_gb, total_needed_gb, mem_total_gb, mem_avail_gb,
    )

    hard_limit = 0.95 * mem_total_gb
    warn_limit = 0.90 * mem_total_gb

    if total_needed_gb > hard_limit:
        msg = (
            f"INSUFFICIENT RAM: model needs ~{total_needed_gb:.1f} GB but system has "
            f"{mem_total_gb:.1f} GB total (95% limit = {hard_limit:.1f} GB). "
            f"Breakdown: {expert_ram_gb:.1f} GB experts + {overhead_gb:.1f} GB overhead. "
            f"Available: {mem_avail_gb:.1f} GB. "
            f"This WILL cause OOM and crash system processes."
        )
        if force_load:
            logger.error("FORCE LOAD: %s", msg)
        else:
            raise RuntimeError(msg)
    elif total_needed_gb > warn_limit:
        logger.warning(
            "LOW RAM HEADROOM: model needs ~%.1f GB, system has %.1f GB total "
            "(%.1f GB available). Monitor memory usage closely.",
            total_needed_gb, mem_total_gb, mem_avail_gb,
        )
    else:
        logger.info(
            "RAM budget OK: %.1f GB needed, %.1f GB headroom",
            total_needed_gb, mem_total_gb - total_needed_gb,
        )


def _check_actual_rss(estimated_gb: float):
    """After loading, compare actual RSS to estimate and warn if >10% deviation."""
    actual_kb = _read_vmrss_kb()
    if actual_kb == 0:
        return
    actual_gb = actual_kb / 1024 / 1024
    if estimated_gb > 0:
        deviation = abs(actual_gb - estimated_gb) / estimated_gb
        if deviation > 0.10:
            logger.warning(
                "RAM estimate deviation: estimated %.1f GB, actual VmRSS %.1f GB "
                "(%.0f%% off) — estimates may need recalibration",
                estimated_gb, actual_gb, deviation * 100,
            )
        else:
            logger.info(
                "RAM estimate accurate: estimated %.1f GB, actual VmRSS %.1f GB (%.0f%% deviation)",
                estimated_gb, actual_gb, deviation * 100,
            )


def _compute_layer_groups(
    rank,
    cfg,
    divisor: int,
) -> List[Tuple[List[int], List[int]]]:
    """Compute layer groups for layer-grouped prefill.

    Splits a rank's layers into groups where each group has at most
    ceil(num_moe_layers / divisor) MoE layers. Dense layers (below
    first_k_dense_replace) are included in the first group.

    Args:
        rank: PPRankConfig with layer_start, layer_end
        cfg: ModelConfig with first_k_dense_replace
        divisor: Number of groups to split MoE layers into

    Returns:
        List of (abs_layer_indices, moe_layer_indices) tuples.
        abs_layer_indices: absolute layer indices for self.layers indexing
        moe_layer_indices: 0-based MoE indices for GpuPrefillManager
    """
    first_k = cfg.first_k_dense_replace
    all_layers = list(range(rank.layer_start, rank.layer_end))

    # Build abs_layer → moe_index mapping for hybrid models
    # For standard models: MoE layers are first_k..N, so moe_idx = l - first_k
    # For hybrid (Nemotron): MoE layers are scattered, need sequential mapping
    if cfg.layer_types is not None:
        _abs_to_moe = {}
        _moe_seq = 0
        for i, lt in enumerate(cfg.layer_types):
            if lt == "moe":
                _abs_to_moe[i] = _moe_seq
                _moe_seq += 1
        dense_layers = [l for l in all_layers if l not in _abs_to_moe]
        moe_layers = [l for l in all_layers if l in _abs_to_moe]
    else:
        _abs_to_moe = None
        dense_layers = [l for l in all_layers if l < first_k]
        moe_layers = [l for l in all_layers if l >= first_k]

    if not moe_layers:
        # All dense layers — single group, no experts
        return [(all_layers, [])]

    def _to_moe_idx(abs_layer):
        if _abs_to_moe is not None:
            return _abs_to_moe[abs_layer]
        return abs_layer - first_k

    # Active-only or single group: all layers in one group
    if divisor <= 1:
        moe_indices = [_to_moe_idx(l) for l in moe_layers]
        return [(all_layers, moe_indices)]

    # Split MoE layers into groups
    num_moe = len(moe_layers)
    moe_per_group = ceil(num_moe / divisor)

    groups = []
    for g in range(divisor):
        start = g * moe_per_group
        end = min(start + moe_per_group, num_moe)
        if start >= num_moe:
            break
        group_moe = moe_layers[start:end]
        group_moe_indices = [_to_moe_idx(l) for l in group_moe]

        if _abs_to_moe is not None:
            # Hybrid model: include interleaved non-MoE layers in correct position.
            # Each group spans from first_moe to last_moe (inclusive), plus any
            # non-MoE layers before the first group or between groups.
            range_start = 0 if g == 0 else group_moe[0]
            range_end = group_moe[-1] + 1
            group_all = [l for l in all_layers if range_start <= l < range_end]
        elif g == 0:
            # Standard model: dense layers go in the first group
            group_all = dense_layers + group_moe
        else:
            group_all = list(group_moe)

        groups.append((group_all, group_moe_indices))

    return groups


class CPUHubManager:
    """Manages multi-GPU token coordination using CPU as a star-hub aggregator.

    Owns pinned CPU buffers for N GPUs. 
    Workflow: 
      1. gather(gpu_tensors) -> copies from VRAM to pinned RAM
      2. reduce()            -> calls Rust engine to sum BF16 buffers
      3. broadcast(gpu_tensors) -> copies from pinned RAM to all VRAM
    """

    def __init__(self, num_gpus: int, hidden_size: int, engine):
        self.num_gpus = num_gpus
        self.hidden_size = hidden_size
        self.engine = engine
        
        # Max chunk size for prefill buffers (e.g. 5000 tokens)
        self.max_tokens = 5000
        self.buffer_elements = self.max_tokens * hidden_size
        
        # Pinned input buffers (one per GPU)
        self.input_bufs = [
            torch.empty(self.buffer_elements, dtype=torch.bfloat16, pin_memory=True)
            for _ in range(num_gpus)
        ]
        # Pinned output buffer (result)
        self.output_buf = torch.empty(
            self.buffer_elements, dtype=torch.bfloat16, pin_memory=True
        )
        
        self.input_ptrs = [buf.data_ptr() for buf in self.input_bufs]
        self.output_ptr = self.output_buf.data_ptr()

    def reduce_sum(self, num_tokens: int):
        """Call Rust engine to sum partial results."""
        num_elements = num_tokens * self.hidden_size
        if num_elements > self.buffer_elements:
            raise ValueError(f"Too many tokens for CPU hub: {num_tokens} > {self.max_tokens}")
            
        self.engine.reduce_sum_bf16(
            self.input_ptrs,
            self.output_ptr,
            num_elements
        )

    def gather(self, gpu_tensors: List[torch.Tensor]):
        """Copy from GPUs to pinned CPU memory (asynchronous)."""
        assert len(gpu_tensors) == self.num_gpus
        for i, t in enumerate(gpu_tensors):
            # Non-blocking copy to pinned memory
            M = t.shape[0]
            count = M * self.hidden_size
            self.input_bufs[i][:count].copy_(t.view(-1), non_blocking=True)

    def broadcast(self, gpu_tensors: List[torch.Tensor]):
        """Copy from pinned CPU memory to all GPUs (asynchronous)."""
        # num_tokens is derived from the target tensor shapes
        for i, t in enumerate(gpu_tensors):
            M = t.shape[0]
            count = M * self.hidden_size
            t.view(-1).copy_(self.output_buf[:count], non_blocking=True)

    def all_gather(self, gpu_slices: List[Optional[torch.Tensor]],
                   targets: List[torch.Tensor]):
        """Gather partial hidden states from all GPUs and broadcast full set to all.

        Each GPU owns a slice of tokens. This concatenates all slices via pinned
        CPU memory and copies the full result to every target tensor.

        Args:
            gpu_slices: Per-GPU partial hidden states (may contain None for inactive GPUs).
            targets: Pre-allocated [total_tokens, hidden_size] tensors on each GPU.
        """
        # Copy each GPU's slice into contiguous pinned memory
        offset = 0
        for t in gpu_slices:
            if t is None:
                continue
            count = t.shape[0] * self.hidden_size
            self.output_buf[offset:offset + count].copy_(t.view(-1), non_blocking=True)
            offset += count
        torch.cuda.synchronize()
        # Broadcast concatenated result to all target GPUs
        for target in targets:
            count = target.shape[0] * self.hidden_size
            target.view(-1).copy_(self.output_buf[:count], non_blocking=True)


class KrasisModel:
    """Full model with streaming attention (GPU0) + EP MoE (all GPUs) + CPU experts."""

    # Mapping from _extract_layer_weights dict keys to attention module attribute names.
    # Only entries where the names differ need to be listed.
    _WEIGHT_KEY_TO_ATTN_ATTR = {
        "kv_a_proj_with_mqa": "kv_a_proj",
        "kv_a_layernorm": "kv_a_norm_weight",
        "q_a_layernorm": "q_a_norm_weight",
    }

    def __init__(
        self,
        model_path: str,
        pp_partition: Optional[List[int]] = None,
        num_gpus: Optional[int] = None,
        devices: Optional[List[str]] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        krasis_threads: int = 40,
        gpu_prefill: bool = False,  # Rust prefill replaces Python GpuPrefillManager
        gpu_prefill_threshold: int = 300,
        attention_backend: str = "rust",  # "rust" (Rust prefill) or "trtllm" (unsupported)
        quant_cfg: QuantConfig = None,
        force_load: bool = False,
        layer_group_size: int = 1,
        gguf_path: Optional[str] = None,
        gguf_native: bool = False,
        kv_cache_mb: int = 1000,  # MB for KV cache
        stream_attention: bool = False,
    ):
        self.cfg = ModelConfig.from_model_path(model_path)
        self.quant_cfg = quant_cfg or QuantConfig()
        self.gguf_path = gguf_path
        self.gguf_native = gguf_native

        # Determine PP partition
        if pp_partition is None:
            if num_gpus is None:
                num_gpus = torch.cuda.device_count()
            pp_partition = compute_pp_partition(self.cfg.num_hidden_layers, num_gpus)

        self.pp_partition = pp_partition
        self.ranks = build_pp_ranks(self.cfg, pp_partition, devices)
        # all_devices = all GPUs for EP replication, not just PP-ranked GPUs
        effective_num_gpus = num_gpus or len(self.ranks)
        self._num_gpus = effective_num_gpus
        self.all_devices = [torch.device(f"cuda:{i}") for i in range(effective_num_gpus)]
        self.kv_dtype = kv_dtype
        self.kv_cache_mb = kv_cache_mb
        self.krasis_threads = krasis_threads
        self.gpu_prefill_enabled = gpu_prefill
        self.gpu_prefill_threshold = gpu_prefill_threshold
        self.attention_backend = attention_backend
        self.force_load = force_load
        # When attention is quantized, it's permanently VRAM-resident (much smaller),
        # so streaming from CPU is unnecessary and would overwrite MarlinWeight attrs.
        if stream_attention and quant_cfg.attention != "bf16":
            logger.info("Disabling stream_attention: quantized attention (%s) is permanently VRAM-resident",
                        quant_cfg.attention)
            stream_attention = False
        self.stream_attention = stream_attention

        # Double-buffered streaming attention needs layer_group_size >= 2
        if stream_attention and layer_group_size >= 1:
            if layer_group_size < 2:
                logger.warning("stream_attention: auto-adjusting layer_group_size %d → 2 (minimum for double-buffering)", layer_group_size)
                layer_group_size = 2
            elif layer_group_size % 2 != 0:
                new_size = layer_group_size + 1
                logger.warning("stream_attention: auto-adjusting layer_group_size %d → %d (must be even for double-buffering)", layer_group_size, new_size)
                layer_group_size = new_size
        self.layer_group_size = layer_group_size

        hybrid_info = ""
        if self.cfg.is_hybrid:
            hybrid_info = f", hybrid={self.cfg.num_full_attention_layers} full + {self.cfg.num_hidden_layers - self.cfg.num_full_attention_layers} linear"
        logger.info(
            "KrasisModel: %d layers, PP=%s, %d GPUs, attn=%s%s",
            self.cfg.num_hidden_layers, pp_partition, len(self.ranks),
            attention_backend, hybrid_info,
        )

        # Will be populated by load()
        self.layers: List[TransformerLayer] = []
        self.embedding: Optional[torch.Tensor] = None
        self.final_norm: Optional[torch.Tensor] = None
        self.lm_head_data = None  # (int8, scale) tuple or plain BF16 tensor
        self.kv_caches: List[PagedKVCache] = []
        self.krasis_engine = None
        self.gpu_prefill_managers: dict = {}  # device -> GpuPrefillManager
        self.tokenizer: Optional[Tokenizer] = None
        self._loaded = False

        # Per-device state (streaming attention: GPU0 only).
        self._device_state: dict = {}
        self._active_device: Optional[str] = None  # current device for use_device()

        # Streaming attention: all layers on GPU0.
        # _layer_split = [(0, L)] — single entry covering all layers.
        # _get_gpu_for_layer(i) always returns 0.
        self._layer_split: List[Tuple[int, int]] = []

        # HCS device: GPU where ALL MoE decode happens (most free VRAM, typically GPU1).
        # Set by server.py after HCS allocation. None = use layer's own device.
        self._hcs_device: Optional[torch.device] = None
        # Multi-GPU HCS: experts pinned on ALL GPUs, GPU0 manager dispatches internally.
        # When True, _hcs_device is None and decode calls GPU0 manager directly.
        self._multi_gpu_hcs: bool = False

        # Decode mode: "gpu" (default) or "cpu".
        # GPU decode runs model.forward() per token on GPU.
        # GPU decode uses GpuDecodeStore in Rust (GIL-free).
        self.decode_mode: str = "gpu"

        # Streaming attention decode: weights on CPU pinned memory, DMA'd per-layer.
        # When enabled, attention weights are NOT permanently on GPU.
        self._stream_attn_enabled = False  # set True after offload
        self._stream_attn_cpu: dict = {}   # layer_idx -> {attr_name: pinned_tensor}
        self._stream_attn_gpu_bufs: list = [{}, {}]  # [buf0, buf1] ping-pong GPU buffers
        self._stream_attn_loaded: dict = {}  # {buf_idx: layer_idx} — tracks what's in each buffer
        self._stream_attn_dma_stream: Optional[torch.cuda.Stream] = None

        # For hybrid models: maps absolute layer idx → KV cache layer offset
        # within its device's cache. Linear attention layers get -1 (no KV).
        self._kv_layer_offsets: dict = {}  # layer_idx -> offset or -1

        # Maps absolute layer index → 0-based MoE sequential index.
        # For standard models: moe_idx = abs_layer - first_k_dense_replace
        # For hybrid (Nemotron): MoE layers are scattered, this provides the mapping.
        self._abs_to_moe_idx: dict = {}
        if self.cfg.layer_types is not None:
            moe_seq = 0
            for i, lt in enumerate(self.cfg.layer_types):
                if lt == "moe":
                    self._abs_to_moe_idx[i] = moe_seq
                    moe_seq += 1
        else:
            for i in range(self.cfg.first_k_dense_replace, self.cfg.num_hidden_layers):
                self._abs_to_moe_idx[i] = i - self.cfg.first_k_dense_replace

        # Detect shared_expert_gate from weight map (Qwen3-Next)
        self._has_shared_expert_gate = self._detect_shared_expert_gate()

    def _detect_shared_expert_gate(self) -> bool:
        """Check if model has shared_expert_gate weights (Qwen3-Next sigmoid gate)."""
        import json
        index_path = os.path.join(self.cfg.model_path, "model.safetensors.index.json")
        try:
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            # Check layer 0 for shared_expert_gate
            gate_key = f"{self.cfg.layers_prefix}.layers.0.mlp.shared_expert_gate.weight"
            return gate_key in weight_map
        except (OSError, KeyError):
            return False

    def load(self, gpu_only: bool = True):
        """Load all weights: GPU (streaming INT8) + CPU (Krasis INT4 experts).

        Args:
            gpu_only: If True, skip CPU expert cache and CPU decode weights.
                Saves ~40 GB RAM and ~60s load time. Only GPU prefill and
                GPU decode (with HCS) will work. CPU decode will not be available.
        """
        start = time.perf_counter()

        _vram_checkpoint("before-load")

        # Phase 0a: GPU VRAM sanity check — abort early if GPU is occupied
        # (e.g. zombie process holding memory after a crash)
        for dev in self.all_devices:
            free_mb, total_mb = torch.cuda.mem_get_info(dev)
            free_mb //= (1024 * 1024)
            total_mb //= (1024 * 1024)
            used_pct = 100 * (1 - free_mb / total_mb)
            if used_pct > 50:
                msg = (f"GPU {dev} has only {free_mb} MB free ({used_pct:.0f}% used). "
                       f"A zombie process may be holding GPU memory. "
                       f"Try: sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) "
                       f"or reboot to clear stuck GPU memory.")
                logger.error(msg)
                raise RuntimeError(msg)

        # Phase 0b: System RAM budget check (skip in gpu_only — no CPU experts loaded)
        if self.cfg.n_routed_experts > 0 and not gpu_only:
            _check_system_ram(
                self.cfg,
                cpu_expert_bits=self.quant_cfg.cpu_expert_bits,
                force_load=self.force_load,
            )
            self._estimated_expert_ram_gb = _estimate_expert_ram_gb(
                self.cfg, self.quant_cfg.cpu_expert_bits,
            )
        else:
            self._estimated_expert_ram_gb = 0.0

        # Start RAM watchdog BEFORE loading — protects during the entire load
        self._start_ram_watchdog()

        # Phase 1: GPU weights
        print(f"\n\033[1m\033[36m▸ Loading GPU weights\033[0m", flush=True)
        logger.info("Phase 1: Loading GPU weights (streaming INT8)...")
        loader = WeightLoader(self.cfg, self.quant_cfg)
        self._load_gpu_weights(loader)
        loader.close()

        gpu_elapsed = time.perf_counter() - start
        for i, rank in enumerate(self.ranks):
            dev = torch.device(rank.device)
            alloc_mb = torch.cuda.memory_allocated(dev) / (1024**2)
            logger.info("GPU%d: %.0f MB allocated", i, alloc_mb)
        logger.info("GPU weights loaded in %.1fs", gpu_elapsed)
        _vram_checkpoint("after-phase1-gpu-weights")

        # Phase 1b: Streaming attention offload (if enabled) or resident check
        if self.stream_attention:
            print(f"\n\033[1m\033[36m▸ Offloading attention for streaming decode\033[0m", flush=True)
            self._init_stream_attention()
        else:
            dev = self.all_devices[0]
            free_mb = torch.cuda.mem_get_info(dev)[0] >> 20
            if self.quant_cfg.attention == "awq":
                # AWQ: attention weights are on CPU, will be quantized to INT4/INT8
                # and uploaded to GPU in setup_gpu_decode_store(). BF16 never touches GPU.
                logger.info("Attention on CPU (AWQ: will quantize and upload in decode setup), GPU free: %d MB",
                            free_mb)
                print(f"  \033[0;32mAttention on CPU (AWQ pending), {free_mb} MB free\033[0m", flush=True)
            else:
                # BF16: attention weights permanently resident on GPU
                attn_mb = self._estimate_attention_vram() >> 20
                logger.info("Attention resident on GPU: %d MB, GPU free: %d MB",
                            attn_mb, free_mb)
                print(f"  \033[0;32mAttention resident on GPU ({attn_mb} MB), {free_mb} MB free\033[0m", flush=True)

        # Phase 2: Expert weights (Rust engine) — GPU Marlin cache only (CPU cache no longer used)
        cpu_start = time.perf_counter()
        gpu_bits = self.quant_cfg.gpu_expert_bits
        cache_dir = cache_dir_for_model(self.cfg.model_path)
        has_gpu_cache = False
        if gpu_bits == 16:
            print(f"\n\033[1m\033[36m▸ Loading BF16 expert weights from safetensors (validation mode)\033[0m", flush=True)
            logger.info("Phase 2: Loading GPU expert weights (BF16 validation)...")
        else:
            has_gpu_cache = os.path.isfile(os.path.join(cache_dir, f"experts_marlin_int{gpu_bits}_g128.bin"))
            if has_gpu_cache:
                print(f"\n\033[1m\033[36m▸ Loading GPU expert weights from cache\033[0m", flush=True)
            else:
                print(f"\n\033[1m\033[36m▸ Building GPU INT{gpu_bits} Marlin expert cache (one-time)\033[0m", flush=True)
                print(f"  \033[2mCache will be saved to {cache_dir} for instant loading next time.\033[0m", flush=True)
            logger.info("Phase 2: Loading GPU expert weights (INT%d)...", gpu_bits)
        self._load_cpu_experts(gpu_only=gpu_only)
        cpu_elapsed = time.perf_counter() - cpu_start
        logger.info("Expert weights loaded in %.1fs", cpu_elapsed)
        if gpu_bits == 16:
            print(f"  \033[0;32mBF16 expert weights loaded in {cpu_elapsed:.0f}s.\033[0m", flush=True)
        elif has_gpu_cache:
            print(f"  \033[0;32mExpert weights loaded in {cpu_elapsed:.0f}s.\033[0m", flush=True)
        else:
            print(f"  \033[0;32mExpert cache built in {cpu_elapsed:.0f}s — next launch will be much faster.\033[0m", flush=True)

        _vram_checkpoint("after-phase2-expert-weights")

        # Post-load RSS check: verify RAM estimate accuracy
        if self._estimated_expert_ram_gb > 0:
            _check_actual_rss(self._estimated_expert_ram_gb)

        # Phase 3: GPU prefill managers (one per device)
        if self.gpu_prefill_enabled and self.cfg.n_routed_experts > 0:
            print(f"\n\033[1m\033[36m▸ Initializing GPU prefill managers\033[0m", flush=True)
            self._init_gpu_prefill()

        _vram_checkpoint("after-phase3-prefill-managers")

        # Allocate KV caches
        self._init_kv_caches()
        _vram_checkpoint("after-kv-cache-init")

        # Phase 4: CPU Hub — disabled.  Multi-GPU uses layer-split decode,
        # not Expert Parallelism.  Prefill runs on primary GPU only.
        self.cpu_hub = None

        # Load tokenizer — override eos_token_id from tokenizer (authoritative)
        # config.json may have wrong eos_token_id (e.g. Qwen3.5 has endoftext=248044
        # in config but im_end=248046 is the real stop token from tokenizer_config.json)
        self.tokenizer = Tokenizer(self.cfg.model_path)
        if self.tokenizer.eos_token_id and self.tokenizer.eos_token_id != self.cfg.eos_token_id:
            logger.info("Overriding eos_token_id: config=%d -> tokenizer=%d",
                        self.cfg.eos_token_id, self.tokenizer.eos_token_id)
            self.cfg.eos_token_id = self.tokenizer.eos_token_id

        # CPU decode has been removed — GPU decode via Rust GpuDecodeStore only.
        # CpuDecoder is preserved in frozen-oracle for verification.

        self._loaded = True
        total = time.perf_counter() - start
        _vram_checkpoint("after-full-load")
        logger.info("Model fully loaded in %.1fs", total)

    def warmup_cuda_runtime(self, devices: List[torch.device]):
        """Trigger ALL lazy CUDA runtime allocations on ALL devices before HCS expert loading."""
        import gc
        from krasis.triton_moe import triton_moe_decode, inverse_marlin_repack, inverse_scale_permute

        logger.info("Warming up CUDA runtime on all devices: %s", [str(d) for d in devices])
        free_before = {str(d): torch.cuda.mem_get_info(d)[0] for d in devices}

        for device in devices:
            torch.cuda.set_device(device)
            is_primary = device.index == 0

            # ── 1. cuBLAS workspace (for int8 matmuls) ──
            try:
                # Find largest non-expert weight to stress cuBLAS
                max_k, max_n = 0, 0
                for layer in self.layers:
                    if layer.device != device: continue
                    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                         w = getattr(layer.attention, key, None)
                         if w is None:
                             w = layer.shared_expert.get(key) if layer.shared_expert else None
                         if w is not None:
                            wt = w[0] if isinstance(w, (tuple, list)) else w
                            k, n = wt.shape if len(wt.shape) == 2 else (wt.shape[1], wt.shape[0])
                            if k * n > max_k * max_n: max_k, max_n = k, n

                if max_k > 0:
                    for m_size in (32, 512):
                        x = torch.randint(-127, 127, (m_size, max_k), dtype=torch.int8, device=device)
                        w = torch.randint(-127, 127, (max_n, max_k), dtype=torch.int8, device=device)
                        _ = torch._int_mm(x, w.t())
            except Exception as e:
                raise RuntimeError(
                    f"CUDA warmup (cuBLAS) on {device} failed: {e}\n"
                    "cuBLAS is required for INT8 matmuls (attention/gates). "
                    "Cannot start without working cuBLAS."
                ) from e

            # ── 2. Triton/Marlin JIT Compilation ──
            try:
                K, N, bits = self.cfg.hidden_size, self.cfg.moe_intermediate_size, 4
                # Match the group_size auto-adjustment in the Rust cache builder
                gs = 128
                if K % gs != 0 or N % gs != 0:
                    gs = 64
                nb2 = bits // 2
                n_dummy = 2
                # Gated (silu): w13 width = 2*N (gate+up), ungated (relu2): w13 width = N (up only)
                gated = self.cfg.mlp_hidden_act != "relu2"
                w13_n = 2 * N if gated else N

                if is_primary:
                    # Fused MoE warmup no longer needed — Rust prefill dlopens
                    # the vendored MarlinDefault kernel directly from libkrasis_marlin.so.
                    # Just warm up the vendored Marlin GEMM via a small dummy call.
                    from krasis.marlin_utils import gptq_marlin_gemm, marlin_make_workspace, get_scalar_type
                    ws = marlin_make_workspace(device, max_blocks_per_sm=4)
                    st = get_scalar_type(bits)
                    dummy_a = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
                    dummy_b = torch.zeros(K // 16, N * (bits // 2), dtype=torch.int32, device=device)
                    dummy_s = torch.zeros(K // gs, N, dtype=torch.bfloat16, device=device)
                    gptq_marlin_gemm(
                        a=dummy_a, c=None, b_q_weight=dummy_b, b_scales=dummy_s,
                        global_scale=None, b_zeros=None, g_idx=None, perm=None,
                        workspace=ws, b_q_type=st, size_m=1, size_n=N, size_k=K,
                        is_k_full=True,
                    )
                    torch.cuda.synchronize(device)
                else:
                    # Warm up the Triton kernel with dummy standard-GPTQ data
                    w13_marlin = torch.zeros(n_dummy, K // 16, w13_n * nb2, dtype=torch.int32)
                    w13s_marlin = torch.zeros(n_dummy, K // gs, w13_n, dtype=torch.bfloat16)
                    w13_std = inverse_marlin_repack(w13_marlin, K, w13_n, bits).to(device)
                    w13s_std = inverse_scale_permute(w13s_marlin, K, w13_n, gs).to(device)
                    # w2 (down projection) has different dimensions: input=N, output=K
                    w2_marlin = torch.zeros(n_dummy, N // 16, K * nb2, dtype=torch.int32)
                    w2s_marlin = torch.zeros(n_dummy, N // gs, K, dtype=torch.bfloat16)
                    w2_std = inverse_marlin_repack(w2_marlin, N, K, bits).to(device)
                    w2s_std = inverse_scale_permute(w2s_marlin, N, K, gs).to(device)
                    lookup = torch.tensor([0, 1], dtype=torch.int32, device=device)
                    x = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
                    triton_moe_decode(x, [0,1], [0.5,0.5], w13_std, w13s_std, w2_std, w2s_std, lookup, None, None, self.cfg.swiglu_limit, 1.702, gs)

            except Exception as e:
                raise RuntimeError(
                    f"CUDA warmup (Triton/Marlin) on {device} failed: {e}\n"
                    "Marlin kernels are required for expert computation. "
                    "Cannot start without working Triton/Marlin JIT."
                ) from e
            
            torch.cuda.synchronize(device)

        # ── 3. Linear attention torch.compile warmup ──
        # Must set device back to primary BEFORE torch.compile — after the per-device
        # loop above, the current CUDA device may be a secondary GPU with a different
        # SM architecture. Inductor generates PTX for the current device's arch, so
        # compiling while cuda:1 (e.g. Ada sm_89) is current would produce kernels
        # that fail on cuda:0 (e.g. Blackwell sm_120).
        if self.cfg.linear_num_value_heads > 0:
            try:
                from krasis.linear_attention import warmup_compiled_chunk_step
                primary = devices[0]
                torch.cuda.set_device(primary)
                la_warmup_chunk = 64
                startup_diag = os.environ.get("KRASIS_STARTUP_DIAG", "") == "1"
                la_t0 = time.perf_counter() if startup_diag else 0.0
                if startup_diag:
                    logger.info(
                        "Linear attention compile warmup starting on %s: nv=%d dk=%d dv=%d chunk=%d",
                        primary,
                        self.cfg.linear_num_value_heads,
                        self.cfg.linear_key_head_dim,
                        self.cfg.linear_value_head_dim,
                        la_warmup_chunk,
                    )
                warmup_compiled_chunk_step(
                    primary,
                    nv=self.cfg.linear_num_value_heads,
                    dk=self.cfg.linear_key_head_dim,
                    dv=self.cfg.linear_value_head_dim,
                    chunk_size=la_warmup_chunk,
                )
                if startup_diag:
                    logger.info(
                        "Linear attention compile warmup finished on %s in %.3fs",
                        primary,
                        time.perf_counter() - la_t0,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Linear attention torch.compile warmup failed: {e}\n"
                    "Linear attention layers require torch.compile. "
                    "Cannot start without working compiled LA kernels."
                ) from e

        # ── Clean up and measure ──
        gc.collect()
        torch.cuda.empty_cache()
        
        for device in devices:
            free_after = torch.cuda.mem_get_info(device)[0]
            consumed = free_before[str(device)] - free_after
            logger.info(
                "CUDA runtime warmup on %s complete: %.0f MB consumed "
                "(%.0f MB free before → %.0f MB free after)",
                device, consumed / 1e6, free_before[str(device)] / 1e6, free_after / 1e6,
            )

    @staticmethod
    def _move_weight(w, device):
        """Copy a weight (tensor or INT8 (tensor, scale) tuple) to device."""
        if w is None:
            return None
        if isinstance(w, tuple):
            return tuple(t.to(device) for t in w)
        if isinstance(w, torch.Tensor):
            return w.to(device)
        return w

    @staticmethod
    def _extract_layer_weights(layer, src_device) -> dict:
        """Extract the weights dict from an existing TransformerLayer.

        Reconstructs the dict format that TransformerLayer.__init__ expects,
        so we can copy it to another device and create a new layer from it.
        """
        mv = KrasisModel._move_weight
        result = {
            "norms": {
                "input_layernorm": layer.input_norm_weight,
                "post_attention_layernorm": layer.post_attn_norm_weight,
            },
            "is_moe": layer.is_moe,
            "layer_type": layer.layer_type,
        }

        # ── Attention weights ──
        attn = layer.attention
        if layer.layer_type == "mamba2":
            result["mamba2"] = layer.mamba2_weights
        elif layer.layer_type == "moe" and attn is None:
            pass  # MoE-only layer, no attention weights
        elif layer.layer_type == "linear_attention":
            result["linear_attention"] = {
                "in_proj_qkvz": attn.in_proj_qkvz,
                "in_proj_ba": attn.in_proj_ba,
                "out_proj": attn.out_proj,
                "conv1d_weight": attn.conv1d_weight,
                "A_log": attn.A_log,
                "dt_bias": attn.dt_bias,
                "norm_weight": attn.norm_weight,
            }
        elif hasattr(attn, "kv_a_proj"):
            # MLA
            attn_d = {
                "kv_a_proj_with_mqa": attn.kv_a_proj,
                "o_proj": attn.o_proj,
                "kv_a_layernorm": attn.kv_a_norm_weight,
                "w_kc": attn.w_kc,
                "w_vc": attn.w_vc,
            }
            if attn.has_q_lora:
                attn_d["q_a_proj"] = attn.q_a_proj
                attn_d["q_b_proj"] = attn.q_b_proj
                attn_d["q_a_layernorm"] = attn.q_a_norm_weight
            else:
                attn_d["q_proj"] = attn.q_proj
            result["attention"] = attn_d
        elif hasattr(layer, 'gqa_weights') and layer.gqa_weights:
            # GQA — weights stored directly on layer (no attention wrapper)
            result["attention"] = dict(layer.gqa_weights)
        else:
            # GQA with legacy attention object (shouldn't happen but safe fallback)
            attn_d = {
                "q_proj": attn.q_proj,
                "k_proj": attn.k_proj,
                "v_proj": attn.v_proj,
                "o_proj": attn.o_proj,
            }
            for name in ("q_proj_bias", "k_proj_bias", "v_proj_bias",
                         "o_proj_bias", "sinks", "q_norm", "k_norm"):
                val = getattr(attn, name, None)
                if val is not None:
                    attn_d[name] = val
            result["attention"] = attn_d

        # ── MoE weights ──
        if layer.is_moe:
            result["gate"] = {"weight": layer.gate_weight}
            if layer.gate_bias is not None:
                result["gate"]["bias"] = layer.gate_bias
            if layer.e_score_correction_bias is not None:
                result["gate"]["e_score_correction_bias"] = layer.e_score_correction_bias
            if layer.shared_expert is not None:
                se = {}
                if "gate_up_proj" in layer.shared_expert:
                    # Standard MoE: reconstruct original gate_proj/up_proj from fused gate_up_proj
                    gate_up = layer.shared_expert["gate_up_proj"]
                    if isinstance(gate_up, tuple):
                        # INT8: split (weight, scale) along dim 0
                        mid_w = gate_up[0].shape[0] // 2
                        mid_s = gate_up[1].shape[0] // 2
                        se["gate_proj"] = (gate_up[0][:mid_w], gate_up[1][:mid_s])
                        se["up_proj"] = (gate_up[0][mid_w:], gate_up[1][mid_s:])
                    else:
                        mid = gate_up.shape[0] // 2
                        se["gate_proj"] = gate_up[:mid]
                        se["up_proj"] = gate_up[mid:]
                else:
                    # Nemotron MoE: up_proj + down_proj only (no gate_proj)
                    se["up_proj"] = layer.shared_expert["up_proj"]
                se["down_proj"] = layer.shared_expert["down_proj"]
                if layer.shared_expert_gate is not None:
                    se["shared_expert_gate"] = layer.shared_expert_gate
                result["shared_expert"] = se
        elif layer.dense_mlp is not None:
            result["dense_mlp"] = dict(layer.dense_mlp)

        # ── Latent MoE projections (Nemotron) ──
        if hasattr(layer, 'latent_proj') and layer.latent_proj is not None:
            result["latent_proj"] = dict(layer.latent_proj)

        return result

    @staticmethod
    def _copy_weights_dict(weights: dict, device) -> dict:
        """Deep-copy a layer weights dict, moving all tensors to device."""
        mv = KrasisModel._move_weight
        result = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                result[k] = KrasisModel._copy_weights_dict(v, device)
            elif isinstance(v, (torch.Tensor, tuple)):
                result[k] = mv(v, device)
            else:
                result[k] = v
        return result

    def _load_gpu_weights(self, loader: WeightLoader):
        """Stream-load GPU weights: all attention on GPU0.

        Streaming attention architecture: ALL attention weights, norms, gate
        weights, shared expert weights, embedding, final_norm, and lm_head
        live on GPU0. GPU1+ are reserved entirely for EP expert parallelism
        during prefill and HCS expert cache during decode.

        When stream_attention is enabled, attention weights are loaded to GPU
        one layer at a time, immediately offloaded to CPU, then freed from GPU
        before loading the next layer. This bounds peak VRAM to ~1 layer of
        attention + permanent weights (norms, gates, shared experts, embedding,
        lm_head) instead of the full model.
        """
        self.layers = []
        primary_dev = self.all_devices[0]
        torch.cuda.set_device(primary_dev)
        L = self.cfg.num_hidden_layers

        # All attention on GPU0 — single split covering all layers
        self._layer_split = [(0, L)]
        if self.stream_attention:
            logger.info("Streaming attention: all %d layers on GPU0, %d GPUs for EP",
                         L, len(self.all_devices))
        else:
            logger.info("Resident attention: all %d layers permanently on GPU0, %d GPUs for EP",
                         L, len(self.all_devices))

        # ── Load layers ──
        logger.info("Loading full base model to %s...", primary_dev)
        self.embedding = loader.load_embedding(primary_dev)

        if self.stream_attention:
            # Incremental load: attention goes directly to CPU (never touches GPU),
            # while norms, gates, and shared experts go to GPU (small, permanent).
            # This avoids the GPU roundtrip and ensures no OOM from accumulating
            # attention weights on GPU during loading.
            self._attn_cpu_weights = {}
            import torch as _torch
            _cpu = _torch.device('cpu')
            for layer_idx in range(L):
                weights = loader.load_layer(layer_idx, primary_dev, attn_device=_cpu)
                layer = TransformerLayer(
                    self.cfg, layer_idx, weights, primary_dev,
                    krasis_engine=None,
                    gpu_prefill_manager=None,
                    gpu_prefill_threshold=self.gpu_prefill_threshold,
                    attention_backend=self.attention_backend,
                )
                # Extract attention weights to CPU, null GPU refs
                w = self._extract_layer_weights(layer, layer.device)
                if layer.layer_type == "linear_attention":
                    attn_key = "linear_attention"
                elif layer.layer_type == "mamba2":
                    attn_key = "mamba2"
                else:
                    attn_key = "attention"
                attn_dict = w.get(attn_key, {})
                cpu_attn = self._copy_weights_dict(attn_dict, 'cpu') if attn_dict else {}
                # Store for _init_stream_attention to pin later
                self._attn_cpu_weights[layer_idx] = {
                    "norms": w["norms"],
                    attn_key: cpu_attn,
                    "is_moe": w["is_moe"],
                    "layer_type": w.get("layer_type", "full_attention"),
                }
                if "gate" in w:
                    self._attn_cpu_weights[layer_idx]["gate"] = w["gate"]
                if "shared_expert" in w:
                    self._attn_cpu_weights[layer_idx]["shared_expert"] = w["shared_expert"]
                if "dense_mlp" in w:
                    self._attn_cpu_weights[layer_idx]["dense_mlp"] = w["dense_mlp"]
                if "latent_proj" in w:
                    self._attn_cpu_weights[layer_idx]["latent_proj"] = w["latent_proj"]
                # Null attention on GPU (keep norms/gate/shared — already on GPU)
                attn = layer.attention
                if attn is not None:
                    for attr_name in list(vars(attn).keys()):
                        val = getattr(attn, attr_name, None)
                        if isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                            setattr(attn, attr_name, None)
                        elif isinstance(val, tuple) and len(val) == 2:
                            if all(isinstance(t, torch.Tensor) for t in val):
                                setattr(attn, attr_name, None)
                self.layers.append(layer)
                if (layer_idx + 1) % 10 == 0 or layer_idx == L - 1:
                    torch.cuda.empty_cache()
                    alloc_mb = torch.cuda.memory_allocated(primary_dev) / (1024**2)
                    logger.info("Layer %d/%d loaded+offloaded (GPU alloc: %.0f MB)",
                                layer_idx + 1, L, alloc_mb)
            self._attn_offloaded = True
            logger.info("Incremental attention offload: %d layers, peak VRAM bounded to ~1 layer", L)
        else:
            # When AWQ is configured, load attention weights to CPU (not GPU).
            # AWQ quantizes BF16→INT4 on CPU then uploads only the packed INT4 to GPU
            # in setup_gpu_decode_store(). Loading BF16 to GPU first wastes VRAM and
            # risks OOM on large models (e.g. Q235B: ~12.8 GB BF16 attention).
            awq_active = self.quant_cfg.attention == "awq"
            _attn_dev = torch.device('cpu') if awq_active else None
            for layer_idx in range(L):
                weights = loader.load_layer(layer_idx, primary_dev, attn_device=_attn_dev)
                layer = TransformerLayer(
                    self.cfg, layer_idx, weights, primary_dev,
                    krasis_engine=None,
                    gpu_prefill_manager=None,
                    gpu_prefill_threshold=self.gpu_prefill_threshold,
                    attention_backend=self.attention_backend,
                )
                self.layers.append(layer)

        self.final_norm = loader.load_final_norm(primary_dev)
        self.lm_head_data = loader.load_lm_head(primary_dev)

        # ── VRAM checkpoint ──
        def _vram_snap(label):
            for di, d in enumerate(self.all_devices):
                free, total = torch.cuda.mem_get_info(d)
                alloc = torch.cuda.memory_allocated(d)
                resv = torch.cuda.memory_reserved(d)
                logger.info(
                    "VRAM[%s] GPU%d: free=%d MB, alloc=%d MB, reserved=%d MB, "
                    "non-pytorch=%d MB",
                    label, di, free >> 20, alloc >> 20, resv >> 20,
                    (total - free - resv) >> 20,
                )
        _vram_snap("after-layers-loaded")

        # ── Build per-device state (GPU0 only) ──
        self._active_device = str(primary_dev)
        dev_str = str(primary_dev)
        self._device_state[dev_str] = {
            "layers": self.layers,
            "embedding": self.embedding,
            "final_norm": self.final_norm,
            "lm_head_data": self.lm_head_data,
        }

    # ── Attention streaming: offload/reload for large models ──

    def _estimate_attention_vram(self) -> int:
        """Estimate total GPU VRAM used by attention weights across all layers.

        Returns bytes consumed by all attention-related parameters:
        norms, attention projections, gate weights, shared experts.
        Does NOT include KV cache, embedding, or lm_head.
        """
        total = 0
        for layer in self.layers:
            # Norms
            if layer.input_norm_weight is not None:
                total += layer.input_norm_weight.nelement() * layer.input_norm_weight.element_size()
            if layer.post_attn_norm_weight is not None:
                total += layer.post_attn_norm_weight.nelement() * layer.post_attn_norm_weight.element_size()

            # Attention weights (skip Mamba2/MoE-only layers which have no attention)
            attn = layer.attention
            if attn is not None:
                for attr_name in dir(attn):
                    val = getattr(attn, attr_name, None)
                    if isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                        total += val.nelement() * val.element_size()
                    elif isinstance(val, tuple) and len(val) == 2:
                        # INT8 (weight, scale) tuple
                        for t in val:
                            if isinstance(t, torch.Tensor) and t.device.type == 'cuda':
                                total += t.nelement() * t.element_size()

            # Gate weights + shared expert (MoE layers)
            if layer.is_moe:
                if layer.gate_weight is not None:
                    total += layer.gate_weight.nelement() * layer.gate_weight.element_size()
                if layer.gate_bias is not None:
                    total += layer.gate_bias.nelement() * layer.gate_bias.element_size()
                if layer.e_score_correction_bias is not None:
                    total += layer.e_score_correction_bias.nelement() * layer.e_score_correction_bias.element_size()
                if layer.shared_expert is not None:
                    for v in layer.shared_expert.values():
                        if isinstance(v, torch.Tensor):
                            total += v.nelement() * v.element_size()
                        elif isinstance(v, tuple):
                            for t in v:
                                if isinstance(t, torch.Tensor):
                                    total += t.nelement() * t.element_size()
                if layer.shared_expert_gate is not None:
                    total += layer.shared_expert_gate.nelement() * layer.shared_expert_gate.element_size()

            # Dense MLP (non-MoE layers)
            if not layer.is_moe and hasattr(layer, 'mlp') and layer.mlp is not None:
                for v in layer.mlp.values():
                    if isinstance(v, torch.Tensor):
                        total += v.nelement() * v.element_size()
                    elif isinstance(v, tuple):
                        for t in v:
                            if isinstance(t, torch.Tensor):
                                total += t.nelement() * t.element_size()

        return total

    def _should_stream_attention(self, headroom_mb: int = 2000) -> bool:
        """Check if attention weights should be streamed (offloaded) during prefill.

        Returns True if total attention VRAM exceeds what GPU0 can hold after
        accounting for headroom needed for KV cache, activations, and DMA buffers.
        """
        if not self._loaded or not self.layers:
            return False
        dev = self.all_devices[0]
        free, total = torch.cuda.mem_get_info(dev)
        attn_vram = self._estimate_attention_vram()
        # Headroom for KV cache, activations, DMA buffers, CUDA overhead
        available = free + attn_vram  # If we offloaded, we'd reclaim attn_vram
        needed = attn_vram + headroom_mb * 1024 * 1024
        should_stream = needed > total
        if should_stream:
            logger.info(
                "Attention streaming ENABLED: %d MB attention, %d MB total VRAM, "
                "%d MB needed (with %d MB headroom)",
                attn_vram >> 20, total >> 20, needed >> 20, headroom_mb,
            )
        return should_stream

    def _offload_attention_to_ram(self):
        """Move all attention weights from GPU0 to pinned CPU RAM for streaming.

        After this, layers still exist as TransformerLayer objects but their
        attention weights are on CPU. Gate/shared expert weights also offloaded.
        """
        self._attn_offloaded = True
        self._attn_cpu_weights = {}  # layer_idx -> {attr: cpu_tensor}
        t0 = time.perf_counter()
        total_bytes = 0

        for layer_idx, layer in enumerate(self.layers):
            w = self._extract_layer_weights(layer, layer.device)
            cpu_w = self._copy_weights_dict(w, 'cpu')
            self._attn_cpu_weights[layer_idx] = cpu_w
            total_bytes += sum(
                t.nelement() * t.element_size()
                for t in self._flatten_weights(cpu_w)
            )
            # Null GPU refs to free VRAM (keep layer object for structure)
            self._null_layer_gpu_weights(layer)

        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Attention offloaded: %d layers, %d MB to CPU RAM in %.1fs",
            len(self.layers), total_bytes >> 20, elapsed,
        )

    def _load_attention_group(self, layer_indices: List[int]):
        """Load attention weights for a group of layers from CPU RAM to GPU0.

        Call before processing a layer group during prefill when streaming.
        """
        if not getattr(self, '_attn_offloaded', False):
            return
        dev = self.all_devices[0]
        for layer_idx in layer_indices:
            cpu_w = self._attn_cpu_weights.get(layer_idx)
            if cpu_w is None:
                continue
            gpu_w = self._copy_weights_dict(cpu_w, dev)
            old_layer = self.layers[layer_idx]
            new_layer = TransformerLayer(
                self.cfg, layer_idx, gpu_w, dev,
                krasis_engine=old_layer.krasis_engine,
                gpu_prefill_manager=old_layer.gpu_prefill_manager,
                gpu_prefill_threshold=self.gpu_prefill_threshold,
                attention_backend=self.attention_backend,
            )
            # Preserve CPU decode buffers
            for buf_name in ('_cpu_act_buf', '_cpu_ids_buf', '_cpu_wts_buf',
                             '_cpu_out_buf', '_gpu_out_buf'):
                buf = getattr(old_layer, buf_name, None)
                if buf is not None:
                    setattr(new_layer, buf_name, buf)
            self.layers[layer_idx] = new_layer

    def _free_attention_group(self, layer_indices: List[int]):
        """Free GPU attention weights for a group of layers after processing.

        Nulls GPU refs so VRAM can be reclaimed for the next group's DMA.
        """
        if not getattr(self, '_attn_offloaded', False):
            return
        for layer_idx in layer_indices:
            layer = self.layers[layer_idx]
            self._null_layer_gpu_weights(layer)
        torch.cuda.empty_cache()

    def _reload_all_attention(self):
        """Reload all attention weights to GPU0 permanently (for decode after prefill).

        After prefill with streaming, decode needs all attention weights resident
        on GPU0 for the full layer loop.
        """
        if not getattr(self, '_attn_offloaded', False):
            return
        dev = self.all_devices[0]
        t0 = time.perf_counter()
        for layer_idx in range(len(self.layers)):
            cpu_w = self._attn_cpu_weights.get(layer_idx)
            if cpu_w is None:
                continue
            gpu_w = self._copy_weights_dict(cpu_w, dev)
            old_layer = self.layers[layer_idx]
            new_layer = TransformerLayer(
                self.cfg, layer_idx, gpu_w, dev,
                krasis_engine=old_layer.krasis_engine,
                gpu_prefill_manager=old_layer.gpu_prefill_manager,
                gpu_prefill_threshold=self.gpu_prefill_threshold,
                attention_backend=self.attention_backend,
            )
            for buf_name in ('_cpu_act_buf', '_cpu_ids_buf', '_cpu_wts_buf',
                             '_cpu_out_buf', '_gpu_out_buf'):
                buf = getattr(old_layer, buf_name, None)
                if buf is not None:
                    setattr(new_layer, buf_name, buf)
            self.layers[layer_idx] = new_layer
        self._attn_offloaded = False
        elapsed = time.perf_counter() - t0
        logger.info("Attention reloaded to GPU0: %d layers in %.1fs", len(self.layers), elapsed)

    # ── Streaming attention decode: per-layer DMA from CPU pinned memory ──

    def _init_stream_attention(self):
        """Offload attention weights to CPU pinned memory for streaming decode.

        Uses the existing _offload_attention_to_ram() for prefill compatibility,
        then creates pinned CPU copies and pre-allocated GPU buffers for fast
        per-layer decode DMA.

        When _load_gpu_weights already did incremental offloading (stream_attention
        was True during load), _attn_cpu_weights is pre-populated and norms/gate/
        shared are already on GPU. We skip Step 1 and Step 2.

        Prefill: uses _load_attention_group / _free_attention_group (existing)
        Decode: uses _stream_attn_load / _stream_attn_prefetch (new, fast)
        """
        dev = self.all_devices[0]
        t0 = time.perf_counter()

        already_offloaded = getattr(self, '_attn_offloaded', False)

        if not already_offloaded:
            # Step 1: Use existing offload to move weights to CPU (populates _attn_cpu_weights)
            self._offload_attention_to_ram()

            # Step 2: Reload non-attention components to GPU permanently
            # (norms, gate weights, shared expert, dense MLP — small, needed every decode)
            for layer_idx in range(len(self.layers)):
                cpu_w = self._attn_cpu_weights.get(layer_idx)
                if cpu_w is None:
                    continue
                layer = self.layers[layer_idx]

                # Norms (tiny, ~28 KB per layer)
                norms = cpu_w.get("norms", {})
                if "input_layernorm" in norms:
                    layer.input_norm_weight = norms["input_layernorm"].to(dev)
                if "post_attention_layernorm" in norms:
                    layer.post_attn_norm_weight = norms["post_attention_layernorm"].to(dev)

                # Gate weights (MoE routing)
                if layer.is_moe and "gate" in cpu_w:
                    gate_d = cpu_w["gate"]
                    layer.gate_weight = gate_d["weight"].to(dev)
                    layer._gate_weight_f32 = layer.gate_weight.float()
                    if "bias" in gate_d:
                        layer.gate_bias = gate_d["bias"].to(dev)
                        layer._gate_bias_f32 = layer.gate_bias.float()
                    if "e_score_correction_bias" in gate_d:
                        layer.e_score_correction_bias = gate_d["e_score_correction_bias"].to(dev)
                        layer._e_score_correction_bias_f32 = layer.e_score_correction_bias.float()

                # Shared expert (MoE)
                if layer.is_moe and "shared_expert" in cpu_w:
                    se = self._copy_weights_dict(cpu_w["shared_expert"], dev)
                    layer.shared_expert = se
                    # Re-fuse gate+up proj
                    gp = se.get("gate_proj")
                    up = se.get("up_proj")
                    if gp is not None and up is not None:
                        if isinstance(gp, tuple):
                            se["gate_up_proj"] = (
                                torch.cat([gp[0], up[0]], dim=0),
                                torch.cat([gp[1], up[1]], dim=0),
                            )
                        else:
                            se["gate_up_proj"] = torch.cat([gp, up], dim=0)
                        del se["gate_proj"], se["up_proj"]
                    if "shared_expert_gate" in se:
                        layer.shared_expert_gate = se["shared_expert_gate"]

                # Dense MLP (non-MoE layers)
                if not layer.is_moe and "dense_mlp" in cpu_w:
                    layer.dense_mlp = self._copy_weights_dict(cpu_w["dense_mlp"], dev)
        else:
            logger.info("Attention already offloaded during load — skipping Steps 1-2")

        # Step 3: Create pinned CPU copies of ATTENTION-ONLY weights for fast DMA
        total_bytes = 0
        self._stream_attn_keys = {}

        for layer_idx in range(len(self.layers)):
            cpu_w = self._attn_cpu_weights.get(layer_idx)
            if cpu_w is None:
                self._stream_attn_cpu[layer_idx] = {}
                self._stream_attn_keys[layer_idx] = []
                continue

            # Extract attention-only weights and pin them
            attn_dict = cpu_w.get("attention") or cpu_w.get("linear_attention", {})
            pinned = {}
            keys = []
            for attr_name, val in attn_dict.items():
                if isinstance(val, tuple) and len(val) == 2:
                    pw = val[0].pin_memory() if not val[0].is_pinned() else val[0]
                    ps = val[1].pin_memory() if not val[1].is_pinned() else val[1]
                    pinned[attr_name] = (pw, ps)
                    total_bytes += pw.nelement() * pw.element_size()
                    total_bytes += ps.nelement() * ps.element_size()
                    keys.append(attr_name)
                elif isinstance(val, torch.Tensor):
                    p = val.pin_memory() if not val.is_pinned() else val
                    pinned[attr_name] = p
                    total_bytes += p.nelement() * p.element_size()
                    keys.append(attr_name)

            self._stream_attn_cpu[layer_idx] = pinned
            self._stream_attn_keys[layer_idx] = keys

        # Step 3: Pre-allocate GPU buffers per layer type (DOUBLE-BUFFERED)
        # Hybrid models (e.g. QCN) have different attention architectures
        # per layer (linear attention vs GQA), so we need separate buffer
        # sets keyed by the set of attribute names.
        # We allocate TWO buffer sets (ping-pong) so DMA into buf (N+1)%2
        # can overlap with compute on buf N%2.
        self._stream_attn_gpu_bufs: list = [{}, {}]  # [buf0, buf1], each: frozenset -> {attr: gpu_buf}
        self._stream_attn_layer_key: dict = {}   # layer_idx -> frozenset(attr_names)
        gpu_buf_bytes = 0

        for layer_idx in range(len(self.layers)):
            cpu_tensors = self._stream_attn_cpu[layer_idx]
            if not cpu_tensors:
                self._stream_attn_layer_key[layer_idx] = frozenset()
                continue
            key = frozenset(cpu_tensors.keys())
            self._stream_attn_layer_key[layer_idx] = key
            for buf_idx in range(2):
                if key not in self._stream_attn_gpu_bufs[buf_idx]:
                    bufs = {}
                    for attr_name, val in cpu_tensors.items():
                        if isinstance(val, tuple):
                            bufs[attr_name] = (
                                torch.empty_like(val[0], device=dev),
                                torch.empty_like(val[1], device=dev),
                            )
                            gpu_buf_bytes += val[0].nelement() * val[0].element_size()
                            gpu_buf_bytes += val[1].nelement() * val[1].element_size()
                        else:
                            bufs[attr_name] = torch.empty_like(val, device=dev)
                            gpu_buf_bytes += val.nelement() * val.element_size()
                    self._stream_attn_gpu_bufs[buf_idx][key] = bufs

        num_types = len(self._stream_attn_gpu_bufs[0])
        logger.info("Streaming attention: %d buffer set(s) for %d layer type(s) (double-buffered)",
                     num_types, num_types)

        # Step 4: DMA stream for async prefetch
        self._stream_attn_dma_stream = torch.cuda.Stream(dev)
        self._stream_attn_enabled = True
        self._stream_attn_loaded: dict = {}  # {buf_idx: layer_idx} — tracks what's in each buffer

        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        gpu_free = torch.cuda.mem_get_info(dev)[0] >> 20
        logger.info(
            "Streaming attention decode: %d layers, %d MB pinned, "
            "%d MB GPU buffers (%d type(s), 2x ping-pong), GPU free: %d MB, %.1fs",
            len(self.layers), total_bytes >> 20, gpu_buf_bytes >> 20,
            num_types, gpu_free, elapsed,
        )

    def _get_stream_bufs(self, layer_idx: int, buf_idx: int) -> dict:
        """Get the GPU staging buffers for this layer's attention type.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        key = self._stream_attn_layer_key.get(layer_idx, frozenset())
        return self._stream_attn_gpu_bufs[buf_idx].get(key, {})

    def _stream_attn_load(self, layer_idx: int, buf_idx: int):
        """Copy one layer's attention weights from CPU pinned to GPU buffers.

        Sets the layer's attention attributes to point to the pre-allocated
        GPU buffers. Norms, gate, shared expert are permanently on GPU.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        if self._stream_attn_loaded.get(buf_idx) == layer_idx:
            return
        cpu_tensors = self._stream_attn_cpu[layer_idx]
        gpu_bufs = self._get_stream_bufs(layer_idx, buf_idx)
        attn = self.layers[layer_idx].attention
        for dict_key, gpu_buf in gpu_bufs.items():
            src = cpu_tensors.get(dict_key)
            if src is None:
                continue
            if isinstance(gpu_buf, tuple):
                gpu_buf[0].copy_(src[0], non_blocking=True)
                gpu_buf[1].copy_(src[1], non_blocking=True)
            else:
                gpu_buf.copy_(src, non_blocking=True)
            # Map weight dict key to the actual attention attribute name
            attr_name = self._WEIGHT_KEY_TO_ATTN_ATTR.get(dict_key, dict_key)
            setattr(attn, attr_name, gpu_buf)
        self._stream_attn_loaded[buf_idx] = layer_idx

    def _stream_attn_prefetch(self, layer_idx: int, buf_idx: int):
        """Start async DMA for a layer's attention weights on the DMA stream.

        Call this while CPU is busy with MoE experts. Before using the
        prefetched weights, sync the DMA stream.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        if self._stream_attn_loaded.get(buf_idx) == layer_idx:
            return
        dma_stream = self._stream_attn_dma_stream
        cpu_tensors = self._stream_attn_cpu[layer_idx]
        gpu_bufs = self._get_stream_bufs(layer_idx, buf_idx)
        # The DMA stream must wait for the default stream before overwriting
        # a ping-pong buffer. The buffer we're writing to was read 2 layers
        # ago on the default stream — without this wait, the DMA copy can
        # race with those still-executing kernels.
        dma_stream.wait_stream(torch.cuda.current_stream(self.all_devices[0]))
        with torch.cuda.stream(dma_stream):
            for dict_key, gpu_buf in gpu_bufs.items():
                src = cpu_tensors.get(dict_key)
                if src is None:
                    continue
                if isinstance(gpu_buf, tuple):
                    gpu_buf[0].copy_(src[0], non_blocking=True)
                    gpu_buf[1].copy_(src[1], non_blocking=True)
                else:
                    gpu_buf.copy_(src, non_blocking=True)

    def _stream_attn_sync_prefetch(self, layer_idx: int, buf_idx: int):
        """Sync DMA stream and point layer's attention to GPU buffers.

        buf_idx: 0 or 1 (ping-pong buffer index).
        """
        torch.cuda.current_stream(self.all_devices[0]).wait_stream(self._stream_attn_dma_stream)
        cpu_tensors = self._stream_attn_cpu[layer_idx]
        gpu_bufs = self._get_stream_bufs(layer_idx, buf_idx)
        attn = self.layers[layer_idx].attention
        for dict_key, gpu_buf in gpu_bufs.items():
            if dict_key in cpu_tensors:
                attr_name = self._WEIGHT_KEY_TO_ATTN_ATTR.get(dict_key, dict_key)
                setattr(attn, attr_name, gpu_buf)
        self._stream_attn_loaded[buf_idx] = layer_idx

    @staticmethod
    def _flatten_weights(w_dict: dict) -> List[torch.Tensor]:
        """Flatten a nested weight dict into a list of tensors."""
        tensors = []
        for v in w_dict.values():
            if isinstance(v, dict):
                tensors.extend(KrasisModel._flatten_weights(v))
            elif isinstance(v, torch.Tensor):
                tensors.append(v)
            elif isinstance(v, tuple):
                for t in v:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
        return tensors

    @staticmethod
    def _null_layer_gpu_weights(layer):
        """Null out GPU weight references on a layer to free VRAM."""
        layer.input_norm_weight = None
        layer.post_attn_norm_weight = None

        # Attention
        attn = layer.attention
        for attr_name in list(vars(attn).keys()):
            val = getattr(attn, attr_name, None)
            if isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                setattr(attn, attr_name, None)
            elif isinstance(val, tuple) and len(val) == 2:
                if all(isinstance(t, torch.Tensor) for t in val):
                    setattr(attn, attr_name, None)

        # Gate + shared expert
        if layer.is_moe:
            layer.gate_weight = None
            layer.gate_bias = None
            layer.e_score_correction_bias = None
            layer.shared_expert = None
            layer.shared_expert_gate = None

    def _load_cpu_experts(self, gpu_only: bool = False):
        """Load expert weights into Krasis Rust engine.

        Args:
            gpu_only: If True, skip CPU expert cache loading (saves ~40 GB RAM
                and ~45s load time). GPU Marlin experts are still loaded for
                prefill and HCS decode. CPU decode will not work in this mode.
        """
        from krasis import KrasisEngine

        cpu_bits = self.quant_cfg.cpu_expert_bits
        gpu_bits = self.quant_cfg.gpu_expert_bits

        # If model has shared_expert_gate, Python/GPU handles shared expert with gate
        # → tell Rust engine to skip shared experts to avoid double-counting
        skip_shared = self._has_shared_expert_gate
        if skip_shared:
            logger.info("shared_expert_gate detected — Rust engine will skip shared experts (handled on GPU)")
        engine = KrasisEngine(parallel=True, num_threads=self.krasis_threads, skip_shared_experts=skip_shared)

        if self.gguf_path:
            logger.info("Loading CPU experts from GGUF: %s (native=%s)", self.gguf_path, self.gguf_native)
            engine.load(
                self.cfg.model_path,
                cpu_num_bits=cpu_bits,
                gpu_num_bits=gpu_bits,
                gguf_path=self.gguf_path,
                gguf_native=self.gguf_native,
            )
        else:
            engine.load(self.cfg.model_path, cpu_num_bits=cpu_bits, gpu_num_bits=gpu_bits, gpu_only=gpu_only)

        self.krasis_engine = engine

        # Wire engine to MoE layers on ALL devices + allocate pinned buffers
        # (CPU MoE buffers only needed when NOT gpu_only — they're for CPU decode path)
        first_k = self.cfg.first_k_dense_replace
        for layer in self.layers:
            if layer.is_moe:
                layer.krasis_engine = engine
                if not gpu_only and layer._cpu_act_buf is None:
                    layer._cpu_act_buf = torch.empty(1, self.cfg.hidden_size, dtype=torch.bfloat16, pin_memory=True)
                    layer._cpu_ids_buf = torch.empty(1, self.cfg.num_experts_per_tok, dtype=torch.int32, pin_memory=True)
                    layer._cpu_wts_buf = torch.empty(1, self.cfg.num_experts_per_tok, dtype=torch.float32, pin_memory=True)
                    layer._cpu_out_buf = torch.empty(1, self.cfg.hidden_size, dtype=torch.bfloat16, pin_memory=True)
                    layer._gpu_out_buf = torch.empty(1, self.cfg.hidden_size, dtype=torch.bfloat16, device=layer.device)

        logger.info(
            "Krasis engine: %d MoE layers, %d experts, hidden=%d",
            engine.num_moe_layers(), engine.num_experts(), engine.hidden_size(),
        )

        # ── Send routing weights to Rust for fused routing+MoE (M=1 decode) ──
        num_moe_layers = self.cfg.num_moe_layers
        engine.set_routing_config(
            scoring_func=self.cfg.scoring_func,
            norm_topk_prob=self.cfg.norm_topk_prob,
            topk=self.cfg.num_experts_per_tok,
            n_experts=self.cfg.n_routed_experts,
            hidden_size=self.cfg.hidden_size,
            num_moe_layers=num_moe_layers,
        )
        moe_idx = 0
        for layer in self.layers:
            if layer.is_moe:
                gw = layer.gate_weight.cpu().contiguous()
                gw_bytes = gw.view(torch.uint16).numpy().view(np.uint8).tobytes()
                bias_bytes = None
                if layer.e_score_correction_bias is not None:
                    bias = layer.e_score_correction_bias.cpu().float().contiguous()
                    bias_bytes = bias.numpy().view(np.uint8).tobytes()
                elif self.cfg.swiglu_limit > 0 and layer.gate_bias is not None:
                    # GPT OSS: send gate bias as correction_bias for Rust routing fallback
                    bias = layer.gate_bias.cpu().float().contiguous()
                    bias_bytes = bias.numpy().view(np.uint8).tobytes()
                engine.set_routing_weights(moe_idx, gw_bytes, bias_bytes)
                moe_idx += 1
        logger.info("Routing weights sent to Rust engine (%d MoE layers)", moe_idx)

    def _init_gpu_prefill(self):
        """No-op: Python GPU prefill has been replaced by Rust prefill engine."""
        pass
        num_ranks = 1

        for i, dev in enumerate(prefill_devices):
            dev_str = str(dev)
            if dev_str not in self.gpu_prefill_managers:
                manager = GpuPrefillManager(
                    model_path=self.cfg.model_path,
                    device=dev,
                    num_experts=self.cfg.n_routed_experts,
                    hidden_size=self.cfg.hidden_size,
                    intermediate_size=self.cfg.moe_intermediate_size,
                    params_dtype=torch.bfloat16,
                    n_shared_experts=self.cfg.n_shared_experts,
                    routed_scaling_factor=self.cfg.routed_scaling_factor,
                    first_k_dense=self.cfg.first_k_dense_replace,
                    num_bits=self.quant_cfg.gpu_expert_bits,
                    krasis_engine=engine_ref,
                    num_moe_layers=num_moe_layers,
                    layer_group_size=self.layer_group_size,
                    skip_shared_experts=self._has_shared_expert_gate,
                    swiglu_limit=self.cfg.swiglu_limit,
                    rank=i,
                    num_ranks=num_ranks,
                    num_experts_per_tok=self.cfg.num_experts_per_tok,
                    shared_expert_intermediate_size=self.cfg.effective_shared_expert_intermediate,
                )
                # Set gated mode for ungated (relu2) expert architectures
                if self.cfg.mlp_hidden_act == "relu2":
                    manager.set_gated_experts(False)
                self.gpu_prefill_managers[dev_str] = manager
                logger.info("GPU prefill manager created for %s (rank %d/%d)", dev_str, i, num_ranks)

        # Pre-prepare all MoE layers from disk cache at startup
        for dev_str, manager in self.gpu_prefill_managers.items():
            manager.prepare_all_layers()

        # Build per-layer pinned buffers for zero-copy prefill DMA.
        # Eliminates per-request CPU memcpy during layer streaming.
        for dev_str, manager in self.gpu_prefill_managers.items():
            logger.info("Building prefill pinned buffers for %s...", dev_str)
            manager.build_prefill_pinned_bufs()

        # Wire manager to MoE layers (each layer gets its own device's manager)
        for layer in self.layers:
            if layer.is_moe:
                dev_str = str(layer.device)
                layer.gpu_prefill_manager = self.gpu_prefill_managers.get(dev_str)

        logger.info(
            "GPU prefill: %d managers, threshold=%d tokens",
            len(self.gpu_prefill_managers), self.gpu_prefill_threshold,
        )

    def _start_ram_watchdog(self, floor_pct: float = 0.5):
        """Start daemon thread that monitors system RAM and exits if too low.

        Checks /proc/meminfo every second. If (MemAvailable + SwapFree) drops
        below floor_pct% of MemTotal, logs an error and calls os._exit() to
        prevent a full system OOM that kills desktop processes.

        Swap is counted as available headroom because NVMe swap is fast enough
        to absorb transient spikes (e.g. during WriteCombined expert migration).

        Args:
            floor_pct: Minimum % free RAM+swap before forced exit (default 0.5%).
                       Override via KRASIS_RAM_FLOOR_PERCENT env var.
        """
        floor_pct = float(os.environ.get("KRASIS_RAM_FLOOR_PERCENT", str(floor_pct)))

        def _watchdog():
            while True:
                time.sleep(1.0)
                meminfo = _read_meminfo()
                if not meminfo:
                    continue
                total_kb = meminfo.get("MemTotal", 0)
                avail_kb = meminfo.get("MemAvailable", 0) + meminfo.get("SwapFree", 0)
                if total_kb == 0:
                    continue
                pct_free = 100.0 * avail_kb / total_kb
                if pct_free < floor_pct:
                    logger.error(
                        "RAM WATCHDOG: %.1f%% free (%.1f GB available+swap / %.1f GB total) "
                        "— below %.1f%% floor. Exiting to prevent system OOM!",
                        pct_free, avail_kb / 1024 / 1024,
                        total_kb / 1024 / 1024, floor_pct,
                    )
                    os._exit(137)

        t = threading.Thread(target=_watchdog, daemon=True, name="ram-watchdog")
        t.start()
        logger.info("RAM watchdog started: will exit if < %.1f%% free (RAM+swap)", floor_pct)

    # ── Multi-GPU calibration: replicate weights + measure inference cost ──

    @staticmethod
    def _move_weight(w, device):
        """Move a weight (tensor or INT8 (tensor, scale) tuple) to device."""
        if w is None:
            return None
        if isinstance(w, tuple):
            return tuple(t.to(device) for t in w)
        if isinstance(w, torch.Tensor):
            return w.to(device)
        return w

    def _replicate_to_device(self, device: torch.device) -> dict:
        """Copy all model weights to `device`, return saved references.

        Handles top-level weights (embedding, final_norm, lm_head),
        per-layer norms, attention weights (MLA/GQA/linear), MoE gate
        + shared expert weights, and dense MLP weights.

        Device-bound caches (RoPE, CUDA graphs)
        are invalidated — they re-create lazily on the new device.
        """
        mv = self._move_weight
        saved = {"layers": []}

        # ── Top-level weights ──
        saved["embedding"] = self.embedding
        saved["final_norm"] = self.final_norm
        saved["lm_head_data"] = self.lm_head_data
        self.embedding = self.embedding.to(device)
        self.final_norm = self.final_norm.to(device)
        self.lm_head_data = mv(self.lm_head_data, device)

        # ── Per-layer weights ──
        for layer in self.layers:
            ls = {"device": layer.device}

            # Norms
            ls["input_norm_weight"] = layer.input_norm_weight
            ls["post_attn_norm_weight"] = layer.post_attn_norm_weight
            layer.input_norm_weight = layer.input_norm_weight.to(device)
            layer.post_attn_norm_weight = layer.post_attn_norm_weight.to(device)

            # GPU output buffer (MoE layers)
            if layer._gpu_out_buf is not None:
                ls["_gpu_out_buf"] = layer._gpu_out_buf
                layer._gpu_out_buf = torch.empty_like(layer._gpu_out_buf, device=device)

            # ── Attention weights ──
            attn = layer.attention
            attn_saved = {}

            if attn is None:
                # GQA attention handled by Rust prefill — no Python attention object
                attn_saved = {"type": "rust_prefill"}
            elif layer.layer_type == "linear_attention":
                # GatedDeltaNetAttention
                attn_saved["device"] = attn.device
                for name in ("in_proj_qkvz", "in_proj_ba", "out_proj",
                             "conv1d_weight", "A_log", "dt_bias", "norm_weight"):
                    attn_saved[name] = getattr(attn, name)
                    setattr(attn, name, mv(getattr(attn, name), device))
                # Invalidate linear attention state and CUDA graph
                attn_saved["_conv_state"] = attn._conv_state
                attn_saved["_recurrent_state"] = attn._recurrent_state
                attn_saved["_la_graph"] = attn._la_graph
                attn_saved["_la_input"] = attn._la_input
                attn_saved["_la_output"] = attn._la_output
                attn_saved["_la_stream"] = attn._la_stream
                attn._conv_state = None
                attn._recurrent_state = None
                attn._la_graph = None
                attn._la_input = None
                attn._la_output = None
                attn._la_stream = torch.cuda.Stream(device=device)
                attn.device = device
            else:
                raise NotImplementedError(
                    f"MLA attention replication not supported. "
                    f"Layer {layer.layer_idx} has unsupported attention type."
                )

            ls["attention"] = attn_saved

            # ── MoE weights ──
            if layer.is_moe:
                moe_saved = {}
                for name in ("gate_weight", "gate_bias", "e_score_correction_bias"):
                    val = getattr(layer, name, None)
                    if val is not None:
                        moe_saved[name] = val
                        setattr(layer, name, val.to(device))

                # Shared expert dict
                if layer.shared_expert is not None:
                    moe_saved["shared_expert"] = dict(layer.shared_expert)
                    layer.shared_expert = {
                        k: mv(v, device) for k, v in layer.shared_expert.items()
                    }
                    # shared_expert_gate lives in shared_expert dict AND layer attr
                    if layer.shared_expert_gate is not None:
                        moe_saved["shared_expert_gate"] = layer.shared_expert_gate
                        layer.shared_expert_gate = layer.shared_expert.get("shared_expert_gate")

                # Recompute f32 caches
                moe_saved["_gate_weight_f32"] = layer._gate_weight_f32
                moe_saved["_gate_bias_f32"] = layer._gate_bias_f32
                moe_saved["_e_score_correction_bias_f32"] = layer._e_score_correction_bias_f32
                layer._gate_weight_f32 = layer.gate_weight.float() if layer.gate_weight is not None else None
                layer._gate_bias_f32 = layer.gate_bias.float() if layer.gate_bias is not None else None
                layer._e_score_correction_bias_f32 = (
                    layer.e_score_correction_bias.float()
                    if layer.e_score_correction_bias is not None else None
                )

                # Nullify CUDA graph (device-bound)
                moe_saved["_se_graph"] = layer._se_graph
                moe_saved["_se_input"] = layer._se_input
                moe_saved["_se_output"] = layer._se_output
                moe_saved["_shared_stream"] = layer._shared_stream
                layer._se_graph = None
                layer._se_input = None
                layer._se_output = None
                layer._shared_stream = (
                    torch.cuda.Stream(device=device)
                    if layer.shared_expert_gate is not None else None
                )

                ls["moe"] = moe_saved

            elif layer.dense_mlp is not None:
                ls["dense_mlp"] = dict(layer.dense_mlp)
                layer.dense_mlp = {
                    k: mv(v, device) for k, v in layer.dense_mlp.items()
                }

            layer.device = device
            saved["layers"].append(ls)

        return saved

    def _restore_from_device(self, saved: dict):
        """Restore all tensor references from saved state and free copies."""
        import gc

        # ── Top-level ──
        self.embedding = saved["embedding"]
        self.final_norm = saved["final_norm"]
        self.lm_head_data = saved["lm_head_data"]

        # ── Per-layer ──
        for layer, ls in zip(self.layers, saved["layers"]):
            layer.device = ls["device"]
            layer.input_norm_weight = ls["input_norm_weight"]
            layer.post_attn_norm_weight = ls["post_attn_norm_weight"]

            if "_gpu_out_buf" in ls:
                layer._gpu_out_buf = ls["_gpu_out_buf"]

            # Attention
            attn = layer.attention
            attn_saved = ls["attention"]

            if attn is None:
                # GQA handled by Rust prefill — nothing to restore
                pass
            elif layer.layer_type == "linear_attention":
                attn.device = attn_saved["device"]
                for name in ("in_proj_qkvz", "in_proj_ba", "out_proj",
                             "conv1d_weight", "A_log", "dt_bias", "norm_weight"):
                    setattr(attn, name, attn_saved[name])
                attn._conv_state = attn_saved["_conv_state"]
                attn._recurrent_state = attn_saved["_recurrent_state"]
                attn._la_graph = attn_saved["_la_graph"]
                attn._la_input = attn_saved["_la_input"]
                attn._la_output = attn_saved["_la_output"]
                attn._la_stream = attn_saved["_la_stream"]
            else:
                pass  # MLA not supported

            # MoE
            if "moe" in ls:
                ms = ls["moe"]
                for name in ("gate_weight", "gate_bias", "e_score_correction_bias"):
                    if name in ms:
                        setattr(layer, name, ms[name])
                if "shared_expert" in ms:
                    layer.shared_expert = ms["shared_expert"]
                if "shared_expert_gate" in ms:
                    layer.shared_expert_gate = ms["shared_expert_gate"]
                layer._gate_weight_f32 = ms["_gate_weight_f32"]
                layer._gate_bias_f32 = ms["_gate_bias_f32"]
                layer._e_score_correction_bias_f32 = ms["_e_score_correction_bias_f32"]
                layer._se_graph = ms["_se_graph"]
                layer._se_input = ms["_se_input"]
                layer._se_output = ms["_se_output"]
                layer._shared_stream = ms["_shared_stream"]

            elif "dense_mlp" in ls:
                layer.dense_mlp = ls["dense_mlp"]

        # Free any remaining copies
        del saved
        gc.collect()
        torch.cuda.empty_cache()

    def calibrate_on_device(self, device: torch.device, test_tokens) -> dict:
        """Temporarily replicate model to `device`, run inference, measure VRAM cost.

        Disables GPU prefill so we measure only non-expert VRAM
        (attention, norms, activations, KV cache). Expert VRAM is
        managed separately by HCS budgets.

        Returns dict with 'inference_cost_bytes'.
        """
        import gc
        from krasis.config import PPRankConfig

        logger.info("Calibrating inference cost on %s...", device)

        # ── Save current state ──
        saved_gpu_prefill_enabled = self.gpu_prefill_enabled
        saved_ranks = self.ranks
        saved_kv_caches = self.kv_caches
        saved_kv_layer_offsets = self._kv_layer_offsets
        saved_layer_split = self._layer_split
        saved_all_devices = self.all_devices
        saved_hcs_device = self._hcs_device
        saved_multi_gpu_hcs = self._multi_gpu_hcs

        # Save and null out per-layer gpu_prefill_manager
        saved_layer_managers = []
        for layer in self.layers:
            saved_layer_managers.append(layer.gpu_prefill_manager)
            layer.gpu_prefill_manager = None

        # Disable GPU prefill — forces CPU MoE path
        self.gpu_prefill_enabled = False

        # ── Replicate weights to target device ──
        saved_weights = self._replicate_to_device(device)

        # ── Temporary single-device config ──
        N = self.cfg.num_hidden_layers
        self._layer_split = [(0, N)]
        self.all_devices = [device]
        self._hcs_device = None
        self._multi_gpu_hcs = False
        orig_rank = saved_ranks[0]
        self.ranks = [PPRankConfig(
            rank=0,
            device=str(device),
            layer_start=0,
            layer_end=N,
            num_layers=N,
            has_embedding=True,
            has_lm_head=True,
        )]

        # Re-init KV caches on target device
        self._init_kv_caches()

        # ── Measure inference cost ──
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        inference_cost = 0

        try:
            self._oom_retry_enabled = False
            with torch.inference_mode():
                self.generate(test_tokens, max_new_tokens=1, temperature=0.6)

            peak = torch.cuda.max_memory_allocated(device)
            inference_cost = peak - baseline
            logger.info(
                "Calibration on %s: inference_cost=%d MB "
                "(peak=%d MB, baseline=%d MB)",
                device,
                inference_cost // (1024 * 1024),
                peak // (1024 * 1024),
                baseline // (1024 * 1024),
            )
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            logger.error("Calibration OOM on %s: %s", device, e)
            gc.collect()
            torch.cuda.empty_cache()
            raise
        finally:
            self._oom_retry_enabled = True

            # ── Restore everything ──
            self.gpu_prefill_enabled = saved_gpu_prefill_enabled
            self.ranks = saved_ranks
            self.kv_caches = saved_kv_caches
            self._kv_layer_offsets = saved_kv_layer_offsets
            self._layer_split = saved_layer_split
            self.all_devices = saved_all_devices
            self._hcs_device = saved_hcs_device
            self._multi_gpu_hcs = saved_multi_gpu_hcs

            for layer, mgr in zip(self.layers, saved_layer_managers):
                layer.gpu_prefill_manager = mgr

            self._restore_from_device(saved_weights)

        return {"inference_cost_bytes": inference_cost}

    def _init_kv_caches(self):
        """Allocate paged KV caches per GPU split group.

        Streaming attention: all attention on GPU0, so single KV cache on GPU0.
        For hybrid models, linear attention layers get -1 in _kv_layer_offsets.

        self.kv_caches[gpu_idx] = KV cache for the gpu_idx'th split group.
        """
        self.kv_caches = []
        self._kv_layer_offsets = {}
        use_combined = (self.attention_backend == "trtllm")
        kv_mb = self.kv_cache_mb

        for gpu_idx, (start, end) in enumerate(self._layer_split):
            dev = self.all_devices[gpu_idx]

            # Count full attention layers in this GPU's split
            kv_offset = 0
            for layer_idx in range(start, end):
                if self.cfg.is_full_attention_layer(layer_idx):
                    self._kv_layer_offsets[layer_idx] = kv_offset
                    kv_offset += 1
                else:
                    self._kv_layer_offsets[layer_idx] = -1

            num_kv_layers = kv_offset

            if num_kv_layers > 0:
                cache = PagedKVCache(
                    self.cfg,
                    num_layers=num_kv_layers,
                    device=dev,
                    kv_dtype=self.kv_dtype,
                    combined=use_combined,
                    max_mb=kv_mb,
                    kv_format=self.quant_cfg.kv_cache_format,
                )
            else:
                cache = None
            self.kv_caches.append(cache)

        if self.cfg.is_hybrid:
            total_kv = sum(1 for v in self._kv_layer_offsets.values() if v >= 0)
            total_linear = sum(1 for v in self._kv_layer_offsets.values() if v < 0)
            logger.info(
                "Hybrid model: %d full attention layers (KV cache), %d linear attention layers",
                total_kv, total_linear,
            )

    def get_max_context_tokens(self) -> int:
        """Maximum prompt + generation tokens supported by the KV cache."""
        if not self.kv_caches:
            return 0
        # Bottleneck is the smallest KV cache across GPU splits
        return min(
            c.max_context_tokens for c in self.kv_caches if c is not None
        )

    def _get_rank_for_layer(self, global_layer_idx: int) -> int:
        """Get the PP rank index that owns a given layer."""
        offset = 0
        for i, rank in enumerate(self.ranks):
            if global_layer_idx < offset + rank.num_layers:
                return i
            offset += rank.num_layers
        raise ValueError(f"Layer {global_layer_idx} out of range")

    def _get_gpu_for_layer(self, layer_idx: int) -> int:
        """Get GPU index owning a given layer (streaming attention: always 0)."""
        for gpu_idx, (start, end) in enumerate(self._layer_split):
            if start <= layer_idx < end:
                return gpu_idx
        raise ValueError(f"Layer {layer_idx} not in any GPU split: {self._layer_split}")

    def _cross_device_moe(
        self,
        layer: TransformerLayer,
        hidden_on_layer_dev: torch.Tensor,
        moe_layer_idx: int,
    ) -> torch.Tensor:
        """Run MoE for a layer whose attention is on a different GPU than HCS.

        Routing + shared expert run on layer_dev (where gate/shared weights are).
        Routed experts run on HCS device (where Marlin experts + compact buffers are).
        Result transferred back to layer_dev.

        Args:
            layer: The transformer layer (on layer_dev)
            hidden_on_layer_dev: [M, hidden_size] post-attn-norm hidden state
            moe_layer_idx: 0-based MoE layer index

        Returns:
            MoE output [M, hidden_size] on layer_dev
        """
        layer_dev = layer.device
        hcs_dev = self._hcs_device
        timing = TIMING.decode

        if timing:
            torch.cuda.synchronize(layer_dev)
            t0 = time.perf_counter()

        # 1. Routing on layer_dev (tiny gate matmul)
        topk_ids, topk_weights = layer.compute_routing(hidden_on_layer_dev)

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_routing = time.perf_counter()

        # 2. Shared expert on layer_dev (CUDA graph or direct)
        shared_output = None
        has_shared = layer._shared_stream is not None and layer.shared_expert is not None
        if has_shared:
            if layer._se_graph is None:
                layer._capture_shared_expert_graph()
            if layer._se_graph is not None:
                layer._se_input.copy_(hidden_on_layer_dev)
                layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                layer._se_graph.replay()
            else:
                layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                with torch.cuda.stream(layer._shared_stream):
                    shared_output = layer._shared_expert_forward(hidden_on_layer_dev)
        elif layer.shared_expert_gate is not None and layer.shared_expert is not None:
            shared_output = layer._shared_expert_forward(hidden_on_layer_dev)

        if timing:
            # Don't sync shared stream yet — it should overlap with routed experts
            t_shared_launched = time.perf_counter()

        # 3. Transfer to HCS device
        if _DEBUG_DECODE:
            import sys
            torch.cuda.synchronize(layer_dev)
            _xdev_t0 = time.perf_counter()
        hidden_hcs = _to_device(hidden_on_layer_dev, hcs_dev)
        topk_ids_hcs = _to_device(topk_ids, hcs_dev)
        topk_weights_hcs = _to_device(topk_weights, hcs_dev)

        if _DEBUG_DECODE:
            torch.cuda.synchronize(hcs_dev)
            _xdev_xfer_ms = (time.perf_counter() - _xdev_t0) * 1000

        if timing:
            torch.cuda.synchronize(hcs_dev)
            t_xfer_to_hcs = time.perf_counter()

        # 4. Routed experts on HCS device via its manager
        hcs_mgr = self.gpu_prefill_managers.get(str(hcs_dev))
        if _DEBUG_DECODE:
            _xdev_t1 = time.perf_counter()
        if hcs_mgr is not None:
            routed_output = hcs_mgr.forward(
                moe_layer_idx, hidden_hcs, topk_ids_hcs, topk_weights_hcs,
                routed_only=True,
            )
            if _DEBUG_DECODE:
                torch.cuda.synchronize(hcs_dev)
                _xdev_fwd_ms = (time.perf_counter() - _xdev_t1) * 1000
        else:
            raise RuntimeError(
                f"HCS manager is None during GPU decode at MoE layer {moe_layer_idx}. "
                "GPU decode requires HCS to be initialized. This indicates a bug in "
                "server initialization -- HCS should always be set up before decode starts."
            )

        if timing:
            torch.cuda.synchronize(hcs_dev)
            t_hcs_done = time.perf_counter()

        # 5. Transfer routed output back to layer_dev
        if _DEBUG_DECODE:
            _xdev_t2 = time.perf_counter()
        routed_on_layer = _to_device(routed_output, layer_dev)

        if _DEBUG_DECODE:
            torch.cuda.synchronize(layer_dev)
            _xdev_back_ms = (time.perf_counter() - _xdev_t2) * 1000

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_xfer_back = time.perf_counter()

        # 6. Apply scaling + combine with shared expert
        if _DEBUG_DECODE:
            _xdev_t3 = time.perf_counter()
        rsf = self.cfg.routed_scaling_factor
        if rsf != 1.0:
            routed_on_layer = routed_on_layer * rsf
        if has_shared:
            torch.cuda.current_stream(layer_dev).wait_stream(layer._shared_stream)
            if layer._se_graph is not None:
                shared_output = layer._se_output
            routed_on_layer = routed_on_layer + shared_output
        elif shared_output is not None:
            routed_on_layer = routed_on_layer + shared_output

        if _DEBUG_DECODE:
            torch.cuda.synchronize(layer_dev)
            _xdev_combine_ms = (time.perf_counter() - _xdev_t3) * 1000
            print(
                f"[DBG]   xdev L{moe_layer_idx}: xfer→hcs={_xdev_xfer_ms:.1f}ms fwd={_xdev_fwd_ms:.1f}ms xfer←={_xdev_back_ms:.1f}ms combine={_xdev_combine_ms:.1f}ms",
                file=sys.stderr, flush=True,
            )

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_combine = time.perf_counter()
            logger.info(
                "XDEV-MOE L%d: route=%.2f shared_launch=%.2f xfer_to=%.2f hcs_fwd=%.2f xfer_back=%.2f combine=%.2f total=%.2fms",
                moe_layer_idx,
                (t_routing - t0) * 1000,
                (t_shared_launched - t_routing) * 1000,
                (t_xfer_to_hcs - t_shared_launched) * 1000,
                (t_hcs_done - t_xfer_to_hcs) * 1000,
                (t_xfer_back - t_hcs_done) * 1000,
                (t_combine - t_xfer_back) * 1000,
                (t_combine - t0) * 1000,
            )

        return routed_on_layer

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """Full forward pass with streaming attention architecture.

        ALL attention runs on GPU0. During decode (M=1):
        - Attention runs on GPU0 (where all weights live)
        - MoE runs on HCS device via _cross_device_moe (8 KB hidden transfer)
        - No cross-GPU hidden state bouncing for attention

        Args:
            token_ids: [M] int64 token IDs
            positions: [M] int32 position indices
            seq_states: One SequenceKVState per GPU split group
            return_all_logits: If True, return logits for ALL positions [M, V]
                instead of just the last token [1, V]. Used for perplexity measurement.

        Returns:
            logits: [M, vocab_size] or [1, vocab_size] float32
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        timing = TIMING.decode and M == 1

        if timing:
            t_fwd_start = time.perf_counter()

        # ── Layer-grouped prefill routing ──
        if (
            M > 1
            and self.layer_group_size >= 1
            and M >= self.gpu_prefill_threshold
            and self.gpu_prefill_enabled
            and self.cfg.n_routed_experts > 0
        ):
            return self._forward_prefill_with_oom_retry(
                token_ids, positions, seq_states, return_all_logits=return_all_logits)

        # ── Embedding (GPU0) ──
        first_dev = self.all_devices[0]
        hidden = self.embedding[token_ids.to(first_dev)]  # [M, hidden_size]

        # Diagnostics
        diag = False
        if TIMING.diag:
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            self._diag_count += 1
            diag = self._diag_count <= 12
            if diag:
                h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                logger.info("DIAG[%d] embed: mean=%.4f std=%.4f max=%.4f nan=%d",
                            self._diag_count, h.float().mean(), h.float().std(),
                            h.float().abs().max(), h.isnan().sum().item())

        # ── Ensure KV capacity (once per GPU split group) ──
        for gpu_idx in range(len(self._layer_split)):
            ss = seq_states[gpu_idx]
            if ss is not None:
                ss.ensure_capacity(M)

        # ── Transformer layers ──
        residual = None
        first_k = self.cfg.first_k_dense_replace
        hcs_dev = self._hcs_device  # None if single GPU or HCS not set
        prev_layer_dev = first_dev
        positions_cache = {}  # device -> positions tensor (avoid repeated transfers)
        positions_cache[str(first_dev)] = positions.to(first_dev) if positions.device != first_dev else positions

        # Per-layer timing accumulators (only when timing enabled)
        if timing:
            _t_attn_total = 0.0  # total attention time (all layer types)
            _t_moe_total = 0.0   # total MoE time (cross-device or unified)
            _t_xfer_total = 0.0  # total cross-device hidden transfers
            _t_dma_total = 0.0   # total attention DMA time (streaming only)
            _t_lin_attn_total = 0.0  # linear attention time
            _t_gqa_attn_total = 0.0  # GQA attention time
            _t_lin_attn_layers = 0  # count of linear attention layers
            _t_gqa_layers = 0      # count of GQA attention layers
            _t_moe_total_gpu = 0.0  # GPU hot expert time within MoE
            _t_moe_layers = 0      # count of MoE layers
            _t_unified_layers = 0  # count of unified (same-device) layers

        # Streaming attention: double-buffered (ping-pong) DMA
        # Enable for ANY forward pass where attention weights are offloaded to CPU
        _stream_attn = self._stream_attn_enabled
        _prefetch_started = False  # whether an async prefetch is in flight
        _prefetch_buf_idx = -1     # which buffer the prefetch targets

        # Pre-load layer 0 synchronously into buf 0 before the loop
        if _stream_attn:
            self._stream_attn_load(0, buf_idx=0)
            if _DEBUG_DECODE:
                import sys
                print(f"[DBG] pre-loaded layer 0 into buf 0", file=sys.stderr, flush=True)

        for abs_layer_idx in range(self.cfg.num_hidden_layers):
            layer = self.layers[abs_layer_idx]
            layer_dev = torch.device(layer.device)
            gpu_idx = self._get_gpu_for_layer(abs_layer_idx)
            kv_cache = self.kv_caches[gpu_idx]
            seq_state = seq_states[gpu_idx]

            # MoE layer index for Krasis engine (0-based sequential)
            moe_layer_idx = self._abs_to_moe_idx.get(abs_layer_idx)
            kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, 0)

            # Transfer hidden to layer device if needed
            if prev_layer_dev != layer_dev:
                if timing:
                    torch.cuda.synchronize(prev_layer_dev)
                    _t_xfer0 = time.perf_counter()
                hidden = _to_device(hidden, layer_dev)
                if residual is not None:
                    residual = _to_device(residual, layer_dev)
                prev_layer_dev = layer_dev
                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_xfer_total += time.perf_counter() - _t_xfer0

            # Cache positions per device
            dev_str = str(layer_dev)
            if dev_str not in positions_cache:
                positions_cache[dev_str] = _to_device(positions, layer_dev)
            layer_positions = positions_cache[dev_str]

            # Double-buffered streaming attention
            _dbg_stream_ms = 0.0  # default for non-streaming layers
            if _stream_attn:
                buf_idx = abs_layer_idx % 2
                if _DEBUG_DECODE:
                    import sys
                    torch.cuda.synchronize(layer_dev)
                    _dbg_t0 = time.perf_counter()

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_dma0 = time.perf_counter()

                # Sync prefetch if one was started for this layer
                if _prefetch_started and _prefetch_buf_idx == buf_idx:
                    self._stream_attn_sync_prefetch(abs_layer_idx, buf_idx)
                    _prefetch_started = False
                elif self._stream_attn_loaded.get(buf_idx) != abs_layer_idx:
                    # Fallback: synchronous load (first layer already loaded above)
                    self._stream_attn_load(abs_layer_idx, buf_idx)

                if _DEBUG_DECODE:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_stream_ms = (time.perf_counter() - _dbg_t0) * 1000

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_dma_total += time.perf_counter() - _t_dma0

                # Start prefetch for NEXT layer into opposite buffer BEFORE compute
                next_layer = abs_layer_idx + 1
                if next_layer < self.cfg.num_hidden_layers:
                    next_buf = next_layer % 2
                    self._stream_attn_prefetch(next_layer, next_buf)
                    _prefetch_started = True
                    _prefetch_buf_idx = next_buf

            # Decide execution path:
            # (a) Multi-GPU HCS: attention + routing + shared on layer_dev,
            #     routed experts dispatched to all GPUs via primary manager
            # (b) Single-GPU HCS: attention on layer_dev, MoE on HCS device
            # (c) Unified: everything on layer_dev
            use_multi_gpu_hcs = (
                M == 1
                and layer.is_moe
                and self._multi_gpu_hcs
            )
            use_cross_device_moe = (
                M == 1
                and layer.is_moe
                and not use_multi_gpu_hcs
                and hcs_dev is not None
                and torch.device(hcs_dev) != layer_dev
            )

            if use_multi_gpu_hcs:
                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la = time.perf_counter()

                # Attention on layer_dev (GPU0)
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        None, None, -1, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la_done = time.perf_counter()
                    _dt = _t_la_done - _t_la
                    _t_attn_total += _dt
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_total += _dt
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_attn_total += _dt
                        _t_gqa_layers += 1

                # Routing on layer_dev (tiny gate matmul)
                topk_ids, topk_weights = layer.compute_routing(hidden)

                # Shared expert on layer_dev (CUDA graph or direct)
                shared_output = None
                has_shared = layer._shared_stream is not None and layer.shared_expert is not None
                if has_shared:
                    if layer._se_graph is None:
                        layer._capture_shared_expert_graph()
                    if layer._se_graph is not None:
                        layer._se_input.copy_(hidden)
                        layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                        layer._se_graph.replay()
                    else:
                        layer._shared_stream.wait_stream(torch.cuda.current_stream(layer_dev))
                        with torch.cuda.stream(layer._shared_stream):
                            shared_output = layer._shared_expert_forward(hidden)
                elif layer.shared_expert_gate is not None and layer.shared_expert is not None:
                    shared_output = layer._shared_expert_forward(hidden)

                # Routed experts via primary manager — dispatches to all GPUs internally
                primary_mgr = self.gpu_prefill_managers.get(str(layer_dev))
                if primary_mgr is not None:
                    routed_output = primary_mgr.forward(
                        moe_layer_idx, hidden, topk_ids, topk_weights,
                        routed_only=True,
                    )
                else:
                    raise RuntimeError(
                        f"GPU prefill manager is None at MoE layer {moe_layer_idx} on {layer_dev}. "
                        "GPU prefill requires the prefill manager to be initialized. "
                        "This indicates a bug in server initialization."
                    )

                # Combine: apply routed scaling + shared expert
                rsf = self.cfg.routed_scaling_factor
                if rsf != 1.0:
                    routed_output = routed_output * rsf
                if has_shared:
                    torch.cuda.current_stream(layer_dev).wait_stream(layer._shared_stream)
                    if layer._se_graph is not None:
                        shared_output = layer._se_output
                    hidden = routed_output + shared_output
                elif shared_output is not None:
                    hidden = routed_output + shared_output
                else:
                    hidden = routed_output

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_moe_done = time.perf_counter()
                    _t_moe_total += _t_moe_done - _t_la_done
                    _t_moe_layers += 1

            elif use_cross_device_moe:
                if _DEBUG_DECODE:
                    import sys
                    torch.cuda.synchronize(layer_dev)
                    _dbg_attn_t0 = time.perf_counter()

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la = time.perf_counter()

                # Split path: attention on layer_dev, MoE on HCS device
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        None, None, -1, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la_done = time.perf_counter()
                    _dt = _t_la_done - _t_la
                    _t_attn_total += _dt
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_total += _dt
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_attn_total += _dt
                        _t_gqa_layers += 1

                if _DEBUG_DECODE:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_attn_ms = (time.perf_counter() - _dbg_attn_t0) * 1000
                    _dbg_moe_t0 = time.perf_counter()

                # MoE on HCS device, result back on layer_dev
                hidden = self._cross_device_moe(layer, hidden, moe_layer_idx)

                if _DEBUG_DECODE:
                    _dbg_moe_ms = (time.perf_counter() - _dbg_moe_t0) * 1000
                    print(f"[DBG] L{abs_layer_idx} stream={_dbg_stream_ms:.1f}ms attn={_dbg_attn_ms:.1f}ms moe={_dbg_moe_ms:.1f}ms", file=sys.stderr, flush=True)

                if timing:
                    # _cross_device_moe already syncs internally when timing
                    _t_moe_done = time.perf_counter()
                    _t_moe_total += _t_moe_done - _t_la_done
                    _t_moe_layers += 1
            else:
                if _DEBUG_DECODE:
                    import sys
                    torch.cuda.synchronize(layer_dev)
                    _dbg_uni_t0 = time.perf_counter()

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la = time.perf_counter()

                # Unified path: attention first, then MLP/MoE
                # Split into forward_attn + MoE to get separate timing
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        None, None, -1, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward_attn(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_la_done = time.perf_counter()
                    _dt = _t_la_done - _t_la
                    _t_attn_total += _dt
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_total += _dt
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_attn_total += _dt
                        _t_gqa_layers += 1

                # MLP / MoE
                if layer.is_moe:
                    hidden = layer._moe_forward(hidden, moe_layer_idx)
                else:
                    hidden = layer._dense_mlp_forward(hidden)

                if _DEBUG_DECODE:
                    torch.cuda.synchronize(layer_dev)
                    _dbg_uni_ms = (time.perf_counter() - _dbg_uni_t0) * 1000
                    print(f"[DBG] L{abs_layer_idx} stream={_dbg_stream_ms:.1f}ms unified={_dbg_uni_ms:.1f}ms (moe={layer.is_moe})", file=sys.stderr, flush=True)

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_moe_done = time.perf_counter()
                    _t_unified_layers += 1
                    if layer.is_moe:
                        _t_moe_total += _t_moe_done - _t_la_done
                        _t_moe_layers += 1

            if TIMING.diag and diag and abs_layer_idx in (0, 1, self.cfg.num_hidden_layers // 2, self.cfg.num_hidden_layers - 1):
                h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                r = residual[-1] if residual.shape[0] > 1 else residual[0]
                logger.info("DIAG[%d] L%d: hid std=%.4f max=%.4f | res std=%.4f max=%.4f",
                            self._diag_count, abs_layer_idx,
                            h.float().std(), h.float().abs().max(),
                            r.float().std(), r.float().abs().max())

        # Advance KV cache seq_len (once per GPU split group)
        for gpu_idx in range(len(self._layer_split)):
            ss = seq_states[gpu_idx]
            if ss is not None:
                ss.advance(M)

        if timing:
            torch.cuda.synchronize()
            t_after_layers = time.perf_counter()

        # ── Final norm + LM head (always on GPU0) ──
        hidden = _to_device(hidden, first_dev)
        if residual is not None:
            residual = _to_device(residual, first_dev)

        from krasis.layer import _fused_add_rmsnorm
        _fused_add_rmsnorm(
            hidden, residual, self.final_norm, self.cfg.rms_norm_eps
        )

        if M > 1 and not return_all_logits:
            hidden = hidden[-1:, :]
        logits = _linear(hidden, self.lm_head_data)
        logits = logits.float()

        if TIMING.diag and diag:
            last_logits = logits[-1]
            topk_vals, topk_ids = last_logits.topk(5)
            tok_strs = []
            if self.tokenizer:
                for tid in topk_ids.tolist():
                    tok_strs.append(repr(self.tokenizer.decode([tid])))
            logger.info("DIAG[%d] logits: std=%.2f top5=%s",
                        self._diag_count, last_logits.std(),
                        list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

        if timing:
            torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()
            total_ms = (t_fwd_end - t_fwd_start) * 1000
            layers_ms = (t_after_layers - t_fwd_start) * 1000
            post_ms = (t_fwd_end - t_after_layers) * 1000
            attn_ms = _t_attn_total * 1000
            moe_ms = _t_moe_total * 1000
            xfer_ms = _t_xfer_total * 1000
            dma_ms = _t_dma_total * 1000
            logger.info(
                "DECODE-TOKEN: total=%.1fms (layers=%.1fms post=%.1fms)",
                total_ms, layers_ms, post_ms,
            )
            la_ms = _t_lin_attn_total * 1000
            gqa_ms = _t_gqa_attn_total * 1000
            dma_str = f" attn_dma=%.1fms" % dma_ms if dma_ms > 0 else ""
            la_per = la_ms / _t_lin_attn_layers if _t_lin_attn_layers else 0
            gqa_per = gqa_ms / _t_gqa_layers if _t_gqa_layers else 0
            moe_per = moe_ms / _t_moe_layers if _t_moe_layers else 0
            logger.info(
                "  BREAKDOWN: attn=%.1fms [LA=%.1fms (%d×%.1f) GQA=%.1fms (%d×%.1f)] "
                "moe=%.1fms (%d×%.1f) xfer=%.1fms unified=%d%s",
                attn_ms, la_ms, _t_lin_attn_layers, la_per,
                gqa_ms, _t_gqa_layers, gqa_per,
                moe_ms, _t_moe_layers, moe_per,
                xfer_ms, _t_unified_layers, dma_str,
            )
            if layers_ms > 0:
                dma_pct = f" dma=%.0f%%" % (dma_ms / total_ms * 100) if dma_ms > 0 else ""
                logger.info(
                    "  PERCENT: LA=%.0f%% GQA=%.0f%% moe=%.0f%% xfer=%.0f%% post=%.0f%%%s unaccounted=%.0f%%",
                    la_ms / total_ms * 100,
                    gqa_ms / total_ms * 100,
                    moe_ms / total_ms * 100,
                    xfer_ms / total_ms * 100,
                    post_ms / total_ms * 100,
                    dma_pct,
                    (total_ms - attn_ms - moe_ms - xfer_ms - dma_ms - post_ms) / total_ms * 100,
                )

        return logits

    def _forward_prefill_with_oom_retry(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """Wrapper around forward_prefill_layer_grouped with OOM recovery.

        On CUDA OOM:
        1. Halve the max chunk size (forces smaller chunks on retry)
        2. Free CUDA graphs if present (reclaims ~1 GB)
        3. Reset KV state and retry
        4. If minimum chunk still OOMs, raise the error
        """
        max_chunk_override = None
        max_retries = 0 if not getattr(self, '_oom_retry_enabled', True) else 3
        freed_graphs = False

        # Defragment the PyTorch allocator before prefill.  Decode leaves many
        # small cached-but-free blocks that prevent contiguous DMA allocations.
        # This reclaims those blocks so layer group DMA can allocate freely.
        torch.cuda.empty_cache()

        for attempt in range(max_retries + 1):
            try:
                result = self.forward_prefill_layer_grouped(
                    token_ids, positions, seq_states,
                    max_chunk_override=max_chunk_override,
                    return_all_logits=return_all_logits,
                )

                # Re-capture CUDA graphs if we freed them during OOM recovery.
                # Prefill is done so the activation VRAM is free again.
                if freed_graphs:
                    for manager in self.gpu_prefill_managers.values():
                        if manager._hcs_initialized and (manager._hcs_buffers or manager._hcs_devices):
                            try:
                                torch.cuda.empty_cache()
                                if len(manager._hcs_devices) > 1:
                                    manager._init_cuda_graphs_multi_gpu()
                                else:
                                    manager._init_cuda_graphs()
                                logger.info("CUDA graphs re-captured after OOM recovery")
                            except torch.OutOfMemoryError as graph_e:
                                raise RuntimeError(
                                    "Could not re-capture CUDA graphs after OOM recovery. "
                                    "Decode requires CUDA graphs for acceptable performance. "
                                    "Reduce KV cache size or expert cache budget to free VRAM."
                                ) from graph_e

                return result
            except torch.OutOfMemoryError:
                if attempt >= max_retries:
                    raise

                # Determine new chunk size
                old_chunk = max_chunk_override or token_ids.shape[0]
                new_chunk = max(128, old_chunk // 2)

                if new_chunk == max_chunk_override:
                    # Already at minimum, can't shrink further
                    raise

                logger.warning(
                    "Prefill OOM (attempt %d/%d): reducing chunk_size %d → %d",
                    attempt + 1, max_retries, old_chunk, new_chunk,
                )

                # Free CUDA intermediates
                torch.cuda.empty_cache()

                # Free CUDA graphs if present (reclaims ~1 GB)
                if not freed_graphs:
                    for manager in self.gpu_prefill_managers.values():
                        # Legacy single-GPU graphs
                        if manager._hcs_cuda_graphs:
                            logger.warning(
                                "Freeing %d CUDA graphs to reclaim VRAM for prefill",
                                len(manager._hcs_cuda_graphs),
                            )
                            manager._hcs_cuda_graphs.clear()
                            manager._hcs_graph_io.clear()
                            manager._hcs_cuda_graphs_enabled = False
                            freed_graphs = True
                        # Multi-GPU per-device graphs
                        for hcs_dev in getattr(manager, '_hcs_devices', []):
                            if hcs_dev.cuda_graphs:
                                logger.warning(
                                    "Freeing %d multi-GPU CUDA graphs on %s for prefill",
                                    len(hcs_dev.cuda_graphs), hcs_dev.device,
                                )
                                hcs_dev.cuda_graphs.clear()
                                if hcs_dev.graph_io:
                                    hcs_dev.graph_io.clear()
                                manager._hcs_cuda_graphs_enabled = False
                                freed_graphs = True
                        if freed_graphs:
                            torch.cuda.empty_cache()

                # Reset KV state for retry (prefill will re-fill from scratch)
                for seq_state in seq_states:
                    if seq_state is not None:
                        seq_state.seq_len = 0

                max_chunk_override = new_chunk

    def forward_prefill_layer_grouped(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
        max_chunk_override: int = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """GPU prefill with layer-grouped expert loading.

        Loop structure (group-outer, chunk-inner):
            for each group → DMA experts once → for each chunk → for each layer → compute

        Streaming attention: ALL attention runs on GPU0. Expert Parallelism
        splits MoE experts across all GPUs for parallel compute. No cross-GPU
        hidden state transfers during attention (only for EP MoE dispatch).
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        num_gpus = len(self.all_devices)
        first_k = self.cfg.first_k_dense_replace
        # EP disabled: prefill runs entirely on primary GPU (GPU0).
        # Aux GPUs are decode-only (HCS layer split).  The decomposed EP path
        # (separate attn/routing/expert dispatch + cross-GPU transfers) adds
        # overhead with no benefit on PCIe — single layer.forward() is faster.
        use_ep = False

        # ── Embedding (GPU0) ──
        first_dev = self.all_devices[0]
        all_hidden = self.embedding[token_ids.to(first_dev)]

        # ── Pre-allocate KV pages for all tokens ──
        for seq_state in seq_states:
            if seq_state is not None:
                seq_state.ensure_capacity(M)

        # ── Token chunking ──
        chunk_size = min(M, max_chunk_override or 5000)
        num_chunks = ceil(M / chunk_size)

        chunk_hidden = []
        chunk_positions = []
        chunk_residual: List[Optional[torch.Tensor]] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, M)
            chunk_hidden.append(all_hidden[start:end])
            chunk_positions.append(positions[start:end])
            chunk_residual.append(None)
        del all_hidden

        # ── Layer group computation ──
        # layer_group_size = how many MoE layers to load at once.
        # Convert to divisor for _build_layer_groups: divisor = ceil(N / group_size).
        first_mgr = next(iter(self.gpu_prefill_managers.values()), None)
        first_k = self.cfg.first_k_dense_replace
        rank = self.ranks[0]
        num_moe = sum(1 for l in range(rank.layer_start, rank.layer_end) if l >= first_k)
        group_size = max(1, self.layer_group_size)
        effective_divisor = max(1, -(-num_moe // group_size))  # ceil division

        # Use rank 0 as the reference for groups (PP=1: all layers in one rank).
        # Streaming attention: all layers on GPU0, groups for expert DMA chunking.
        rank = self.ranks[0]
        dev = torch.device(rank.device)  # initial device (GPU0)

        # Collect all managers ordered by device index
        all_managers = []
        for d in self.all_devices:
            mgr = self.gpu_prefill_managers.get(str(d))
            if mgr is not None:
                all_managers.append(mgr)

        manager = all_managers[0] if all_managers else None

        # Single-GPU: manager must load ALL experts, not just its EP slice.
        if not use_ep and manager and manager.num_local_experts < self.cfg.n_routed_experts:
            _saved_ep = (manager.expert_start, manager.expert_end, manager.num_local_experts)
            manager.expert_start = 0
            manager.expert_end = self.cfg.n_routed_experts
            manager.num_local_experts = self.cfg.n_routed_experts
            manager._dma_bufs_initialized = False
        else:
            _saved_ep = None

        groups = _compute_layer_groups(rank, self.cfg, effective_divisor)

        is_active_only = manager and manager._prefill_mode in ("active_only", "lru")

        # Thread pool for DMA operations (parallel EP loads + pipelined prefetch)
        _dma_managers = all_managers if use_ep else ([manager] if manager else [])
        dma_pool = ThreadPoolExecutor(max_workers=max(len(_dma_managers), 1)) if _dma_managers else None

        # Per-GPU CUDA streams for EP compute.
        # GPU0 runs attention on default stream, EP streams handle MoE dispatch.
        ep_streams = {}
        if use_ep:
            for mgr in all_managers:
                ep_streams[mgr.device] = torch.cuda.Stream(device=mgr.device)

        # ── Async EP transfer: pinned CPU bounce buffers + copy stream ──
        # When P2P is broken, cross-GPU transfers go through CPU.  Without
        # pinned buffers + a dedicated copy stream, h.cpu() blocks the host
        # thread until GPU0's default stream drains (serializing GPU0 forward
        # and GPU1 forward).  The copy stream runs D2H in parallel with GPU0's
        # forward, and event-based sync eliminates host-blocking synchronize().
        _ep_copy_stream = None
        _ep_pinned_h = None
        _ep_pinned_ids = None
        _ep_pinned_wts = None
        _ep_pinned_outs = {}       # device -> pinned output buffer
        _ep_input_read_event = None  # GPU1 done reading pinned input
        _ep_has_secondary = use_ep and any(m.device != dev for m in all_managers)
        if _ep_has_secondary and not _check_p2p():
            _ep_copy_stream = torch.cuda.Stream(device=dev)

        # ── Lightweight per-group timing (DMA vs compute breakdown) ──
        _layer_timing = use_ep and os.environ.get("KRASIS_LAYER_TIMING") == "1"
        if _layer_timing:
            _lt_dma_total = 0.0
            _lt_compute_total = 0.0
            _lt_free_total = 0.0
            _lt_count = 0

        # ── EP timing instrumentation (gated on TIMING.prefill) ──
        ep_timing = TIMING.prefill and use_ep
        if ep_timing:
            _ep_times = {
                "dma_preload": 0.0,
                "attention": 0.0,
                "attention_gqa": 0.0,
                "attention_linear": 0.0,
                "routing": 0.0,
                "shared_expert": 0.0,
                "output_transfer": 0.0,
                "free": 0.0,
                "layer_total": 0.0,
            }
            # Per-GPU timing entries
            for _gi, _mgr in enumerate(all_managers):
                _ep_times[f"gpu{_gi}_forward"] = 0.0
                if ep_streams.get(_mgr.device) is not None:
                    _ep_times[f"gpu{_gi}_input_copy"] = 0.0
            _ep_layer_count = 0
            _ep_dense_count = 0
            _ep_dense_time = 0.0

        # ── DMA pipelining: overlap next group's DMA with current compute ──
        # Disabled during active_only mode and validation sync.
        # EP timing now uses CUDA events (no sync overhead), so pipelining stays enabled.
        _validation = getattr(self, '_validation_sync', False)
        _no_pipeline = os.environ.get("KRASIS_NO_PIPELINE", "") == "1"
        _pipeline_enabled = bool(_dma_managers) and not is_active_only and not _validation and not _no_pipeline
        _has_prefetch = False
        _prefetch_futures = []
        if _pipeline_enabled:
            logger.info("DMA pipelining ENABLED (%d managers, %d groups)",
                        len(_dma_managers), len(groups))

        # Use heavy attention streaming (new TransformerLayer per group) only
        # if attention is offloaded AND decode streaming is NOT active.
        # When _stream_attn_enabled, the lightweight per-layer DMA handles attention.
        _attn_streaming = getattr(self, '_attn_offloaded', False) and not self._stream_attn_enabled

        # Double-buffered streaming attention prefill state
        if self._stream_attn_enabled:
            self._prefill_prefetch_started = False
            self._prefill_prefetch_buf = -1
            # Pre-load first layer into buf 0
            first_group_layers = groups[0][0] if groups else []
            if first_group_layers:
                self._stream_attn_load(first_group_layers[0], buf_idx=first_group_layers[0] % 2)

        for group_idx, (group_layers, group_moe_indices) in enumerate(groups):
            need_load = group_moe_indices and not is_active_only

            # 0. Load attention weights for this group (streaming mode)
            if _attn_streaming:
                self._load_attention_group(group_layers)

            # 1. Ensure current group's experts are loaded
            if need_load:
                if _layer_timing:
                    _lt_dma0 = time.perf_counter()

                if _has_prefetch:
                    # Wait for async prefetch started in previous iteration
                    for f in _prefetch_futures:
                        f.result()  # propagate exceptions
                    _prefetch_futures = []
                    for mgr in _dma_managers:
                        mgr.swap_prefetch()
                    _has_prefetch = False
                else:
                    # Synchronous load (first group, or pipelining disabled)
                    if ep_timing:
                        for d in self.all_devices:
                            torch.cuda.synchronize(d)
                        _t_dma0 = time.perf_counter()

                    futures = [dma_pool.submit(mgr.preload_layer_group, group_moe_indices)
                               for mgr in _dma_managers]
                    for f in futures:
                        f.result()  # propagate exceptions

                    if ep_timing and group_moe_indices:
                        for d in self.all_devices:
                            torch.cuda.synchronize(d)
                        _ep_times["dma_preload"] += time.perf_counter() - _t_dma0

                if _layer_timing:
                    _lt_dma_total += time.perf_counter() - _lt_dma0

            # 2. Start async prefetch for next group (overlaps with compute below)
            if _pipeline_enabled and need_load:
                next_idx = group_idx + 1
                next_moe = groups[next_idx][1] if next_idx < len(groups) else None
                if next_moe:
                    _prefetch_futures = [dma_pool.submit(mgr.preload_layer_group_async, next_moe)
                                         for mgr in _dma_managers]
                    _has_prefetch = True

            # Reset seq_state for this group (all attention on GPU0)
            if seq_states[0] is not None:
                seq_states[0].seq_len = 0
            if _layer_timing and need_load:
                _lt_compute0 = time.perf_counter()

            # 3. Process all chunks through this group's layers
            for c in range(num_chunks):
                # Use _to_device (CPU bounce) for cross-GPU transfers.
                # PyTorch's .to() with broken P2P silently produces garbage
                # data on the target device (proven by positions corruption).
                h = _to_device(chunk_hidden[c], dev)
                r = chunk_residual[c]
                if r is not None:
                    r = _to_device(r, dev)
                pos = _to_device(chunk_positions[c], dev)
                chunk_M = h.shape[0]

                for li, abs_layer_idx in enumerate(group_layers):
                    layer = self.layers[abs_layer_idx]
                    moe_layer_idx = self._abs_to_moe_idx.get(abs_layer_idx)
                    kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, 0)

                    # Streaming attention: double-buffered load
                    if self._stream_attn_enabled:
                        buf_idx = abs_layer_idx % 2
                        if c == 0:
                            # First chunk: sync prefetch or synchronous load
                            if hasattr(self, '_prefill_prefetch_started') and self._prefill_prefetch_started and self._prefill_prefetch_buf == buf_idx:
                                self._stream_attn_sync_prefetch(abs_layer_idx, buf_idx)
                                self._prefill_prefetch_started = False
                            elif self._stream_attn_loaded.get(buf_idx) != abs_layer_idx:
                                self._stream_attn_load(abs_layer_idx, buf_idx)

                            # Prefetch next layer within group (safe: uses opposite buffer)
                            next_li = li + 1
                            if next_li < len(group_layers):
                                next_layer = group_layers[next_li]
                                next_buf = next_layer % 2
                                self._stream_attn_prefetch(next_layer, next_buf)
                                self._prefill_prefetch_started = True
                                self._prefill_prefetch_buf = next_buf

                        # Cross-group prefetch: only on the LAST chunk.
                        # Earlier chunks still need the current group's buffers,
                        # and the cross-group prefetch would overwrite them.
                        is_last_layer_in_group = (li == len(group_layers) - 1)
                        is_last_chunk = (c == num_chunks - 1)
                        if is_last_layer_in_group and is_last_chunk:
                            next_group_idx = group_idx + 1
                            if next_group_idx < len(groups):
                                next_layer = groups[next_group_idx][0][0]
                                next_buf = next_layer % 2
                                self._stream_attn_prefetch(next_layer, next_buf)
                                self._prefill_prefetch_started = True
                                self._prefill_prefetch_buf = next_buf

                    # Streaming attention: all layers on GPU0, no cross-GPU transfers
                    kv_cache = self.kv_caches[0]
                    seq_state = seq_states[0]

                    # Dense or linear attention layers, or single-GPU: use standard path
                    if not use_ep or moe_layer_idx is None:
                        if ep_timing:
                            torch.cuda.synchronize(dev)
                            _t_dense0 = time.perf_counter()
                        if kv_layer_offset < 0:
                            h, r = layer.forward(
                                h, r, pos,
                                None, None, -1,
                                moe_layer_idx, num_new_tokens=chunk_M,
                            )
                        else:
                            h, r = layer.forward(
                                h, r, pos,
                                kv_cache, seq_state, kv_layer_offset,
                                moe_layer_idx, num_new_tokens=chunk_M,
                            )
                        if ep_timing:
                            torch.cuda.synchronize(dev)
                            _ep_dense_time += time.perf_counter() - _t_dense0
                            _ep_dense_count += 1
                        # Per-layer sync for validation: catch CUDA errors precisely
                        if getattr(self, '_validation_sync', False):
                            try:
                                torch.cuda.synchronize(dev)
                            except Exception as e:
                                logger.error(
                                    "CUDA error at layer %d (moe_idx=%s, chunk=%d/%d, M=%d): %s",
                                    abs_layer_idx, moe_layer_idx, c+1, num_chunks, chunk_M, e,
                                )
                                raise
                        continue

                    # ── Multi-GPU EP path for MoE layers ──
                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_layer0 = time.perf_counter()

                    # ── Attention ──
                    if kv_layer_offset < 0:
                        h, r = layer.forward_attn(
                            h, r, pos, None, None, -1,
                            num_new_tokens=chunk_M,
                        )
                    else:
                        h, r = layer.forward_attn(
                            h, r, pos, kv_cache, seq_state, kv_layer_offset,
                            num_new_tokens=chunk_M,
                        )

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_attn = time.perf_counter()
                        _attn_dt = _t_attn - _t_layer0
                        _ep_times["attention"] += _attn_dt
                        dev_key = f"attn_{dev}"
                        if dev_key not in _ep_times:
                            _ep_times[dev_key] = 0.0
                            _ep_times[f"attn_{dev}_count"] = 0
                        _ep_times[dev_key] += _attn_dt
                        _ep_times[f"attn_{dev}_count"] += 1
                        if layer.layer_type == "linear_attention":
                            _ep_times["attention_linear"] += _attn_dt
                        else:
                            _ep_times["attention_gqa"] += _attn_dt

                    # (b) Routing on GPU 0 (tiny: gate matmul)
                    topk_ids, topk_weights = layer.compute_routing(h)

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_route = time.perf_counter()
                        _ep_times["routing"] += _t_route - _t_attn

                    # (c) Shared expert on layer_dev (async overlap with routed experts)
                    shared_output = None
                    has_shared_overlap = (layer._shared_stream is not None
                                          and layer.shared_expert is not None)
                    if has_shared_overlap:
                        layer._shared_stream.wait_stream(torch.cuda.current_stream(dev))
                        with torch.cuda.stream(layer._shared_stream):
                            shared_output = layer._shared_expert_forward(h)
                    elif layer.shared_expert is not None:
                        shared_output = layer._shared_expert_forward(h)

                    if ep_timing:
                        _t_shared = time.perf_counter()

                    # (d) Routed experts — ALWAYS parallel, with CUDA event timing
                    h_ready = torch.cuda.current_stream(dev).record_event()

                    # CUDA events for per-GPU kernel timing (no sync required)
                    if ep_timing:
                        _gpu_events = {}  # gpu_idx -> (start_event, end_event, device)

                    # ── Async pinned bounce: start D2H on copy stream ──
                    # The copy stream runs D2H in parallel with GPU0's forward
                    # on the default stream.  Without this, h.cpu() blocks the
                    # host until GPU0's default stream drains (serializing the
                    # two GPU forwards).
                    _ep_copy_done = None
                    if _ep_copy_stream is not None:
                        # Lazy-allocate pinned buffers on first use
                        _M = h.shape[0]
                        if _ep_pinned_h is None or _ep_pinned_h.shape[0] < _M:
                            _ep_pinned_h = torch.empty(
                                _M, h.shape[1], dtype=h.dtype,
                                device='cpu', pin_memory=True)
                            _ep_pinned_ids = torch.empty(
                                _M, topk_ids.shape[1], dtype=topk_ids.dtype,
                                device='cpu', pin_memory=True)
                            _ep_pinned_wts = torch.empty(
                                _M, topk_weights.shape[1], dtype=topk_weights.dtype,
                                device='cpu', pin_memory=True)
                        # Wait for any previous GPU1 H2D reads to finish
                        # before overwriting the pinned buffer.
                        if _ep_input_read_event is not None:
                            _ep_copy_stream.wait_event(_ep_input_read_event)
                        _ep_copy_stream.wait_event(h_ready)
                        with torch.cuda.stream(_ep_copy_stream):
                            _ep_pinned_h[:_M].copy_(h, non_blocking=True)
                            _ep_pinned_ids[:_M].copy_(topk_ids, non_blocking=True)
                            _ep_pinned_wts[:_M].copy_(topk_weights, non_blocking=True)
                        _ep_copy_done = _ep_copy_stream.record_event()

                    partial_outputs = []
                    for _gi, mgr in enumerate(all_managers):
                        if mgr.device == dev:
                            if ep_timing:
                                _ev_start = torch.cuda.Event(enable_timing=True)
                                _ev_end = torch.cuda.Event(enable_timing=True)
                                _ev_start.record()
                            out = mgr.forward(moe_layer_idx, h, topk_ids, topk_weights,
                                              routed_only=True)
                            if ep_timing:
                                _ev_end.record()
                                _gpu_events[_gi] = (_ev_start, _ev_end, dev)
                            partial_outputs.append((mgr.device, out))
                        elif _ep_copy_stream is not None:
                            # ── Async path: copy from pinned buffer ──
                            stream = ep_streams[mgr.device]
                            stream.wait_event(_ep_copy_done)
                            with torch.cuda.stream(stream):
                                if ep_timing:
                                    _ev_start = torch.cuda.Event(enable_timing=True)
                                    _ev_end = torch.cuda.Event(enable_timing=True)
                                    _ev_start.record(stream)
                                _M = h.shape[0]
                                h_dev = _ep_pinned_h[:_M].to(mgr.device, non_blocking=True)
                                ids_dev = _ep_pinned_ids[:_M].to(mgr.device, non_blocking=True)
                                wts_dev = _ep_pinned_wts[:_M].to(mgr.device, non_blocking=True)
                                # Mark pinned input buffer as consumed
                                _ep_input_read_event = stream.record_event()
                                out = mgr.forward(moe_layer_idx, h_dev, ids_dev, wts_dev,
                                                  routed_only=True)
                                if ep_timing:
                                    _ev_end.record(stream)
                                    _gpu_events[_gi] = (_ev_start, _ev_end, mgr.device)
                                partial_outputs.append((mgr.device, out))
                        else:
                            # ── Fallback: synchronous _to_device (P2P works) ──
                            stream = ep_streams[mgr.device]
                            stream.wait_event(h_ready)
                            with torch.cuda.stream(stream):
                                if ep_timing:
                                    _ev_start = torch.cuda.Event(enable_timing=True)
                                    _ev_end = torch.cuda.Event(enable_timing=True)
                                    _ev_start.record(stream)
                                h_dev = _to_device(h, mgr.device)
                                ids_dev = _to_device(topk_ids, mgr.device)
                                wts_dev = _to_device(topk_weights, mgr.device)
                                out = mgr.forward(moe_layer_idx, h_dev, ids_dev, wts_dev,
                                                  routed_only=True)
                                if ep_timing:
                                    _ev_end.record(stream)
                                    _gpu_events[_gi] = (_ev_start, _ev_end, mgr.device)
                                partial_outputs.append((mgr.device, out))

                    if ep_timing:
                        _t_dispatch_done = time.perf_counter()

                    if has_shared_overlap:
                        torch.cuda.current_stream(dev).wait_stream(layer._shared_stream)

                    routed_output = None
                    for po_device, po_out in partial_outputs:
                        if po_device == dev:
                            routed_output = po_out
                        elif _ep_copy_stream is not None:
                            # ── Async output gather via pinned buffer ──
                            _M = po_out.shape[0]
                            if po_device not in _ep_pinned_outs or _ep_pinned_outs[po_device].shape[0] < _M:
                                _ep_pinned_outs[po_device] = torch.empty(
                                    _M, po_out.shape[1], dtype=po_out.dtype,
                                    device='cpu', pin_memory=True)
                            po_pinned = _ep_pinned_outs[po_device]
                            # D2H on secondary GPU's ep_stream (sequenced after forward)
                            with torch.cuda.stream(ep_streams[po_device]):
                                po_pinned[:_M].copy_(po_out, non_blocking=True)
                            out_ready = ep_streams[po_device].record_event()
                            # H2D on GPU0's default stream (wait for D2H via event)
                            torch.cuda.current_stream(dev).wait_event(out_ready)
                            out_on_primary = po_pinned[:_M].to(dev, non_blocking=True)
                            if routed_output is None:
                                routed_output = out_on_primary
                            else:
                                routed_output = routed_output + out_on_primary
                        else:
                            ep_streams[po_device].synchronize()
                            out_on_primary = _to_device(po_out, dev)
                            if routed_output is None:
                                routed_output = out_on_primary
                            else:
                                routed_output = routed_output + out_on_primary

                    if ep_timing:
                        # Sync all devices to ensure kernels have completed
                        for _sync_dev in self.all_devices:
                            try:
                                torch.cuda.synchronize(_sync_dev)
                            except Exception as _sync_err:
                                logger.warning("EP timing: sync %s failed: %s", _sync_dev, _sync_err)
                        _t_sync_done = time.perf_counter()
                        # Collect CUDA event timings (GPU kernel times)
                        for _gi, (_ev_s, _ev_e, _ev_dev) in _gpu_events.items():
                            try:
                                _ev_e.synchronize()
                                _gpu_ms = _ev_s.elapsed_time(_ev_e)
                                _ep_times[f"gpu{_gi}_forward"] += _gpu_ms / 1000.0
                            except Exception as _ev_err:
                                logger.warning("EP timing: gpu%d event error: %s", _gi, _ev_err)
                        _ep_times["shared_expert"] += (_t_shared - _t_route) if has_shared_overlap else 0
                        _ep_times["output_transfer"] += _t_sync_done - _t_dispatch_done
                        _ep_layer_count += 1

                    # (f) Apply routed_scaling_factor + shared expert on GPU 0
                    rsf = self.cfg.routed_scaling_factor
                    if rsf != 1.0:
                        routed_output *= rsf
                    if shared_output is not None:
                        h = routed_output + shared_output
                    else:
                        h = routed_output

                    if ep_timing:
                        try:
                            for _sync_dev in self.all_devices:
                                torch.cuda.synchronize(_sync_dev)
                        except Exception:
                            pass
                        _ep_times["layer_total"] += time.perf_counter() - _t_layer0

                # Save for next group
                chunk_hidden[c] = h
                chunk_residual[c] = r

                # Advance KV seq_state (all attention on GPU0 → single cache)
                if seq_states[0] is not None:
                    seq_states[0].advance(chunk_M)
            # 4. Free experts for this group
            if _layer_timing and need_load:
                torch.cuda.synchronize(dev)
                _lt_compute_total += time.perf_counter() - _lt_compute0

            if _layer_timing and need_load:
                _lt_free0 = time.perf_counter()

            if need_load:
                _clear_cache = True

                # Sync default stream before freeing: empty_cache() can return
                # blocks to CUDA that pending kernels still reference.
                torch.cuda.synchronize(dev)

                if ep_timing:
                    for d in self.all_devices:
                        torch.cuda.synchronize(d)
                    _t_free0 = time.perf_counter()

                for mgr in _dma_managers:
                    mgr.free_layer_group(clear_cache=_clear_cache)

                if ep_timing and group_moe_indices:
                    for d in self.all_devices:
                        torch.cuda.synchronize(d)
                    _ep_times["free"] += time.perf_counter() - _t_free0

            if _layer_timing and need_load:
                _lt_free_total += time.perf_counter() - _lt_free0
                _lt_count += 1

            # 5. Free attention weights for this group (streaming mode)
            if _attn_streaming:
                self._free_attention_group(group_layers)

        # Reload all attention weights for decode (streaming mode)
        # Skip reload if streaming decode is enabled — weights stay on CPU
        if _attn_streaming and not self._stream_attn_enabled:
            self._reload_all_attention()

        # Restore EP slicing state (single-GPU only)
        if _saved_ep is not None:
            manager.expert_start, manager.expert_end, manager.num_local_experts = _saved_ep
            manager._dma_bufs_initialized = False

        if dma_pool is not None:
            dma_pool.shutdown(wait=False)

        # ── EP timing summary ──
        if ep_timing:
            _total_measured = sum(v for k, v in _ep_times.items()
                                 if k not in ("layer_total", "attention_gqa", "attention_linear"))
            logger.info("=" * 70)
            logger.info("EP TIMING BREAKDOWN (%d MoE layers, %d dense layers, %d chunks)",
                        _ep_layer_count, _ep_dense_count, num_chunks)
            logger.info("-" * 70)
            _skip_phases = {"layer_total", "attention_gqa", "attention_linear"}
            # Also skip per-device attn_* keys (shown separately below)
            _skip_phases.update(k for k in _ep_times if k.startswith("attn_"))
            for phase, t in _ep_times.items():
                if phase in _skip_phases:
                    continue
                per_layer = (t / _ep_layer_count * 1000) if _ep_layer_count > 0 else 0
                pct = (t / _ep_times["layer_total"] * 100) if _ep_times["layer_total"] > 0 else 0
                logger.info("  %-20s %8.1f ms total  %6.2f ms/layer  %5.1f%%",
                            phase, t * 1000, per_layer, pct)
            logger.info("-" * 70)
            # Per-layer-type attention averages (multiply by num_chunks for layer-chunks)
            num_gqa_chunks = (self.cfg.num_full_attention_layers if self.cfg.is_hybrid else _ep_layer_count) * num_chunks
            num_linear_chunks = _ep_layer_count - num_gqa_chunks
            if num_gqa_chunks > 0:
                logger.info("  %-20s %6.2f ms/chunk  (%d layer-chunks)",
                            "  avg GQA attn", _ep_times["attention_gqa"] / num_gqa_chunks * 1000, num_gqa_chunks)
            if num_linear_chunks > 0:
                logger.info("  %-20s %6.2f ms/chunk  (%d layer-chunks)",
                            "  avg linear attn", _ep_times["attention_linear"] / num_linear_chunks * 1000, num_linear_chunks)
            # Per-device attention averages
            for d in self.all_devices:
                dk = f"attn_{d}"
                ck = f"attn_{d}_count"
                if dk in _ep_times and _ep_times[ck] > 0:
                    logger.info("  %-20s %6.2f ms/chunk  (%d chunks, %.0f ms total)",
                                f"  attn on {d}",
                                _ep_times[dk] / _ep_times[ck] * 1000,
                                int(_ep_times[ck]),
                                _ep_times[dk] * 1000)
            logger.info("-" * 70)
            logger.info("  %-20s %8.1f ms total  %6.2f ms/layer",
                        "LAYER TOTAL", _ep_times["layer_total"] * 1000,
                        _ep_times["layer_total"] / _ep_layer_count * 1000 if _ep_layer_count else 0)
            logger.info("  %-20s %8.1f ms total  %6.2f ms/layer",
                        "DENSE LAYERS", _ep_dense_time * 1000,
                        _ep_dense_time / _ep_dense_count * 1000 if _ep_dense_count else 0)
            logger.info("  %-20s %8.1f ms total",
                        "SUM (measured)", _total_measured * 1000)
            logger.info("=" * 70)

        # ── Layer timing summary (DMA vs compute vs free) ──
        if _layer_timing and _lt_count > 0:
            _lt_total = _lt_dma_total + _lt_compute_total + _lt_free_total
            logger.info("=" * 70)
            logger.info("LAYER TIMING (%d groups, %d chunks)", _lt_count, num_chunks)
            logger.info("-" * 70)
            logger.info("  DMA load:  %8.1f ms total  %6.2f ms/group  %5.1f%%",
                        _lt_dma_total * 1000, _lt_dma_total / _lt_count * 1000,
                        _lt_dma_total / _lt_total * 100 if _lt_total > 0 else 0)
            logger.info("  Compute:   %8.1f ms total  %6.2f ms/group  %5.1f%%",
                        _lt_compute_total * 1000, _lt_compute_total / _lt_count * 1000,
                        _lt_compute_total / _lt_total * 100 if _lt_total > 0 else 0)
            logger.info("  Free:      %8.1f ms total  %6.2f ms/group  %5.1f%%",
                        _lt_free_total * 1000, _lt_free_total / _lt_count * 1000,
                        _lt_free_total / _lt_total * 100 if _lt_total > 0 else 0)
            logger.info("  TOTAL:     %8.1f ms", _lt_total * 1000)
            per_chunk = _lt_compute_total / (_lt_count * num_chunks) * 1000
            logger.info("  Compute per chunk: %.2f ms", per_chunk)
            logger.info("=" * 70)

        # ── Final result: logits ──
        # final_norm and lm_head are always on GPU0 (first device)
        from krasis.layer import _fused_add_rmsnorm
        first_dev = self.all_devices[0]

        if return_all_logits:
            # Concatenate ALL chunk hidden/residual states for perplexity measurement
            all_h = torch.cat(chunk_hidden, dim=0)      # [M, hidden_size]
            all_r = torch.cat(chunk_residual, dim=0)     # [M, hidden_size]
            all_h = _to_device(all_h, first_dev)
            all_r = _to_device(all_r, first_dev)
            _fused_add_rmsnorm(
                all_h, all_r, self.final_norm, self.cfg.rms_norm_eps
            )
            final_logits = _linear(all_h, self.lm_head_data).float()
        else:
            # Standard path: only last token's logits (for generation)
            last_h = chunk_hidden[-1][-1:]
            last_r = chunk_residual[-1][-1:]
            last_h = _to_device(last_h, first_dev)
            last_r = _to_device(last_r, first_dev)
            _fused_add_rmsnorm(
                last_h, last_r, self.final_norm, self.cfg.rms_norm_eps
            )
            final_logits = _linear(last_h, self.lm_head_data).float()

        return final_logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_token_ids: Optional[List[int]] = None,
        presence_penalty: float = 0.0,
    ) -> List[int]:
        """Generate tokens autoregressively.

        Args:
            prompt_tokens: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            stop_token_ids: Stop generation on these tokens
            presence_penalty: Penalty for already-generated tokens (0 = disabled)

        Returns:
            List of generated token IDs (excluding prompt)
        """
        if stop_token_ids is None:
            stop_token_ids = [self.cfg.eos_token_id] + list(self.cfg.extra_stop_token_ids)

        # Guard: ensure prompt_tokens is a list of ints
        if isinstance(prompt_tokens, str):
            prompt_tokens = self.tokenizer.encode(prompt_tokens)
        elif isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens["input_ids"]
        elif hasattr(prompt_tokens, "input_ids"):
            prompt_tokens = prompt_tokens.input_ids
        if isinstance(prompt_tokens, list) and prompt_tokens and not isinstance(prompt_tokens[0], int):
            prompt_tokens = [int(t) for t in prompt_tokens]

        # Create per-GPU-split sequence states (one per KV cache / GPU)
        seq_states_per_rank = [
            SequenceKVState(c, seq_id=0) if c is not None else None
            for c in self.kv_caches
        ]

        # Reset linear attention states for new sequence
        if self.cfg.is_hybrid:
            for layer in self.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        # Reset Mamba2 SSM states for new sequence (zero conv_state and ssm_state)
        # Prefill will write fresh final states, but zeroing ensures no stale state leaks
        decode_states = getattr(self, '_mamba2_decode_states', None)
        if decode_states:
            for buffers in decode_states.values():
                buffers['conv_state'].zero_()
                buffers['ssm_state'].zero_()

        device = torch.device(self.ranks[0].device)
        generated = []

        _t_gen_start = time.perf_counter()

        try:
            # Multi-GPU: re-upload prefill-only attention if previously freed
            self._upload_prefill_only_attention()

            # ── Prefill ── (Single-slot AWQ: Marlin already in GPU slots)
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

            logits = self.forward(prompt_tensor, positions, seq_states_per_rank)
            # Take last token's logits
            next_logits = logits[-1:, :]

            # Track generated tokens for presence penalty
            generated_set = set()
            next_token = sample(
                next_logits, temperature, top_k, top_p,
                presence_penalty=presence_penalty,
                generated_tokens=generated_set,
            ).item()
            generated.append(next_token)
            generated_set.add(next_token)

            torch.cuda.synchronize()
            self._last_ttft = time.perf_counter() - _t_gen_start

            if next_token in stop_token_ids:
                return generated

            # ── Decode (GPU) — entire loop in Rust, zero Python per token ──
            gpu_store = getattr(self, '_gpu_decode_store', None)
            if gpu_store is None:
                raise RuntimeError("GPU decode store not configured. Call setup_gpu_decode_store() first.")
            self._export_kv_to_rust(seq_states_per_rank, len(prompt_tokens))
            self._transfer_mamba2_states()
            self._update_la_state_ptrs()
            self._update_la_state_ptrs_aux()
            # Multi-GPU: free prefill-only attention before decode
            self._free_prefill_only_attention()
            # Single-slot AWQ: swap simple INT4 into GPU slots for decode
            gpu_store.swap_to_simple_int4()
            decode_tokens = gpu_store.gpu_generate_batch(
                first_token=next_token,
                start_position=len(prompt_tokens),
                max_tokens=max_new_tokens - 1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_ids=list(stop_token_ids),
                presence_penalty=presence_penalty,
            )
            generated.extend(decode_tokens)

        finally:
            n_decode = len(generated) - 1  # First token is from prefill
            if n_decode > 0:
                self._last_decode_time = gpu_store.last_decode_elapsed_s if gpu_store else 0.0
                self._last_decode_tok_s = n_decode / self._last_decode_time if self._last_decode_time > 0 else 0.0
            else:
                self._last_decode_time = 0.0
                self._last_decode_tok_s = 0.0
            # Single-slot AWQ: restore Marlin into GPU slots for next prefill
            if gpu_store is not None:
                gpu_store.swap_to_marlin()
            # Free KV cache pages
            for s in seq_states_per_rank:
                if s is not None:
                    s.free()

        return generated

    def chat(
        self,
        messages: List[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Chat completion: format messages, generate, decode."""
        prompt_tokens = self.tokenizer.apply_chat_template(messages)
        logger.info("Prompt: %d tokens", len(prompt_tokens))

        generated = self.generate(
            prompt_tokens, max_new_tokens, temperature, top_k, top_p,
        )
        return self.tokenizer.decode(generated)

    # ──────────────────────────────────────────────────────
    # Rust server interface
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _dense_mlp_tensor(w) -> torch.Tensor:
        """Extract a BF16 tensor from a dense MLP weight value.

        Dense MLP weights are either plain BF16 tensors or (weight_int8, scale)
        tuples from per-channel INT8 quantization.  The Rust decode store needs
        plain BF16, so dequantize on the fly when necessary.
        """
        if isinstance(w, tuple):
            w_int8, scale = w
            return (w_int8.float() * scale.float().unsqueeze(1)).to(torch.bfloat16)
        return w

    def setup_gpu_decode_store(self) -> "GpuDecodeStore":
        """Create and configure a GpuDecodeStore for Rust-native GPU decode.

        Registers all attention weights, MoE layers, RoPE tables, and KV cache.
        Called once at startup. Returns the configured store.
        """
        from krasis import GpuDecodeStore

        device = torch.device(self.ranks[0].device)
        gpu_idx = device.index or 0

        store = GpuDecodeStore(gpu_idx)
        # Compute max QKV buffer size across all layer types.
        # Use dimension attributes (not weight tensors) since streaming may have
        # offloaded weights to CPU, leaving tensor attributes as None.
        max_qkv = self.cfg.hidden_size * 3  # default for standard GQA
        for layer in self.layers:
            if layer.attention is None:
                # Mamba2 or MoE-only layers have no attention QKV buffers
                if layer.layer_type == "mamba2" and layer.mamba2_weights is not None:
                    # Mamba2 in_proj output is large: d_inner*2 + conv_dim + num_heads
                    m2 = self.cfg
                    in_proj_out = m2.mamba_d_inner * 2 + m2.mamba_conv_dim + m2.mamba_num_heads
                    max_qkv = max(max_qkv, in_proj_out)
            elif layer.layer_type == "linear_attention":
                attn = layer.attention
                # in_proj_qkvz output dim = nk*(dk + dk + hr*dv + hr*dv)
                qkvz_out = attn.num_k_heads * (2 * attn.k_head_dim + 2 * attn.head_ratio * attn.v_head_dim)
                max_qkv = max(max_qkv, qkvz_out)
            elif hasattr(layer.attention, 'kv_a_proj'):
                # MLA: q_absorbed [num_heads * ckv_dim] is the largest buffer
                ma = layer.attention
                q_out = ma.num_heads * (ma.qk_nope_dim + ma.qk_rope_dim)
                q_absorbed = ma.num_heads * ma.ckv_dim  # padded to 512 for MLA
                kv_out = ma.ckv_dim + ma.qk_rope_dim
                max_qkv = max(max_qkv, q_out, q_absorbed, kv_out)
            elif hasattr(layer.attention, 'num_heads'):
                ga = layer.attention
                q_sz = ga.num_heads * ga.head_dim * (2 if ga.gated_attention else 1)
                kv_sz = ga.num_kv_heads * ga.head_dim
                max_qkv = max(max_qkv, q_sz + kv_sz * 2)

        # Compute max intermediate from actual layer types (not unused config fields)
        has_dense_mlp = any(l.dense_mlp is not None for l in self.layers)
        max_inter = self.cfg.moe_intermediate_size
        if has_dense_mlp:
            max_inter = max(max_inter, self.cfg.intermediate_size)

        store.configure(
            hidden_size=self.cfg.hidden_size,
            num_layers=len(self.layers),
            vocab_size=self.cfg.vocab_size,
            eps=self.cfg.rms_norm_eps,
            max_experts_per_tok=self.cfg.num_experts_per_tok,
            max_intermediate_size=max_inter,
            max_qkv_size=max_qkv,
            group_size=128,
            expert_bits=self.quant_cfg.gpu_expert_bits,
            moe_intermediate_size=self.cfg.moe_intermediate_size,
        )

        # Register embedding
        store.set_embedding(self.embedding.data_ptr())

        # Register final norm
        store.set_final_norm(self.final_norm.data_ptr(), self.cfg.hidden_size)

        # Register LM head (always as BF16 for cuBLAS GEMV compatibility)
        # Dequantize on CPU to avoid OOM on GPU0 (float32 intermediate can be ~3 GB for large vocabs)
        lm_head_w = self.lm_head_data
        if isinstance(lm_head_w, tuple):
            # INT8 (weight, scale) — dequantize to BF16 on CPU, then copy to GPU
            w_int8, scale = lm_head_w
            lm_head_bf16 = (w_int8.cpu().float() * scale.cpu().unsqueeze(1)).to(torch.bfloat16).contiguous().to(device)
            self._rust_lm_head = lm_head_bf16  # prevent GC
            lm_head_ptr = lm_head_bf16.data_ptr()
            lm_head_rows = lm_head_bf16.shape[0]
            lm_head_cols = lm_head_bf16.shape[1]
        else:
            lm_head_ptr = lm_head_w.data_ptr()
            lm_head_rows = lm_head_w.shape[0]
            lm_head_cols = lm_head_w.shape[1]
        store.set_lm_head(
            store.register_weight(lm_head_ptr, lm_head_rows, lm_head_cols, 0)  # 0 = BF16
        )

        # Norm bias flag (Qwen3 uses (1+w)*x)
        store.set_norm_bias_one(getattr(self.cfg, 'norm_bias_one', False))

        # Track weight IDs for LA layers (needed for re-registration after state reset)
        self._la_wids = {}
        rope_set = False

        # When streaming attention is enabled, the layer's weight attributes point to
        # ping-pong GPU buffers that may contain a DIFFERENT layer's data.  We must
        # create permanent GPU copies from the CPU-pinned originals so Rust decode
        # always sees correct weights for every layer.
        self._rust_decode_weights = []  # prevent GC of permanent GPU copies

        attn_quant = self.quant_cfg.attention  # "bf16" or "awq"
        marlin_gs = 128  # Marlin group size for both INT8 and INT4

        # AWQ template: per-layer channel scales from calibration
        _awq_template = None
        if attn_quant == "awq":
            from krasis.awq_calibrate import load_template
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "templates", "attention")
            _awq_template = load_template(template_dir, self.cfg.model_path)
            if _awq_template is not None:
                tmpl_version = _awq_template.get("version", 1)
                if tmpl_version >= 2:
                    summary = _awq_template["summary"]
                    logger.info("AWQ v2 template loaded: %d AWQ-scaled, %d plain INT4, %d BF16 tensors "
                                "(avg error reduction: %.1f%%)",
                                summary["awq_scaled_tensors"],
                                summary["plain_int4_tensors"],
                                summary["bf16_tensors"],
                                summary["avg_error_reduction"] * 100)
                else:
                    # v1 template (old per-tensor decisions, no per-channel scaling)
                    n_int4 = _awq_template["summary"]["int4"]
                    n_int8 = _awq_template["summary"]["int8"]
                    n_bf16 = _awq_template["summary"]["bf16"]
                    logger.info("AWQ v1 template loaded: %d INT4, %d INT8, %d BF16 tensors",
                                n_int4, n_int8, n_bf16)
            else:
                raise RuntimeError(
                    f"AWQ attention requested but no calibration template found for model "
                    f"{self.cfg.model_path}. No template available locally or from GitHub. "
                    f"Run './dev awq-calibrate <config>' to generate one, or use "
                    f"--attention-quant bf16 for unquantized attention."
                )

        # Marlin quantization helpers (lazy import, only when quantizing)
        _marlin_workspace = None
        _marlin_scalar_type_int4 = None
        _marlin_scalar_type_int8 = None
        if attn_quant == "awq":
            from krasis.marlin_utils import marlin_make_workspace, get_scalar_type
            _marlin_workspace = marlin_make_workspace(device)
            _marlin_scalar_type_int4 = get_scalar_type(4, False)
            _marlin_scalar_type_int8 = get_scalar_type(8, False)

        def _register_attn_weight(w: torch.Tensor, layer_idx: int = -1,
                                  layer_type: str = "", tensor_name: str = "",
                                  awq_scales: Optional[torch.Tensor] = None) -> int:
            """Register an attention weight as BF16, Marlin INT8, or Marlin INT4.

            For Marlin quantization: Rust does quantize + repack on CPU,
            Python uploads packed result to GPU as PyTorch tensors (shared
            between prefill GEMM and Rust decode GEMV). Same repack format
            for both paths. No BF16 ever touches VRAM.

            For BF16: weight must already be on GPU.

            Args:
                awq_scales: Optional per-channel scales [K] from AWQ calibration.
                    When provided, weight columns are scaled by s[j] before
                    quantization: W[:,j] *= s[j]. The caller is responsible
                    for folding 1/s[j] into the preceding RMSNorm weight.

            Returns weight ID in the Rust store. For quantized weights, also
            stores (packed, scales, workspace, scalar_type, N, K) on
            self._marlin_attn_weights[wid] for prefill GEMM."""
            # Stash BF16 on CPU for aux store quantization (only when streaming is off)
            if self._aux_bf16_stash is not None and layer_idx >= 0:
                w_stash = w.cpu().contiguous() if w.is_cuda else w.contiguous()
                self._aux_bf16_stash[(layer_idx, tensor_name)] = w_stash

            # Determine effective quantization for this tensor
            effective_quant = attn_quant
            if attn_quant == "awq":
                if _awq_template is not None:
                    from krasis.awq_calibrate import get_tensor_decision
                    effective_quant = get_tensor_decision(
                        _awq_template, layer_idx, layer_type, tensor_name)
                else:
                    # This should be unreachable — we raise at template load time above
                    raise RuntimeError(
                        "AWQ attention active but no template loaded — this is a bug"
                    )

            if effective_quant in ("int8", "int4") and w.dtype == torch.bfloat16:
                use_int4 = (effective_quant == "int4")
                num_bits = 4 if use_int4 else 8
                n, k = w.shape[0], w.shape[1]
                if n % 64 == 0 and k % 16 == 0 and k % marlin_gs == 0:
                    # Step 1: Get contiguous CPU BF16 data
                    w_cpu = w.cpu().contiguous() if w.is_cuda else w.contiguous()

                    # Step 1.5: AWQ per-channel scaling (if provided)
                    # Scale weight columns: W[:,j] *= s[j] to protect channels
                    # with high activation magnitude during quantization.
                    if awq_scales is not None:
                        assert awq_scales.shape[0] == k, \
                            f"AWQ scales size {awq_scales.shape[0]} != weight K dim {k}"
                        # Apply in float32 for precision, then convert back to BF16
                        w_cpu = (w_cpu.float() * awq_scales.float().unsqueeze(0)).to(torch.bfloat16)
                        w_cpu = w_cpu.contiguous()

                    # Step 2: Rust quantizes + repacks on CPU (same format as expert weights)
                    if use_int4:
                        packed_bytes, scales_bytes, rn, rk = store.repack_marlin_int4_cpu(
                            w_cpu.data_ptr(), n, k, marlin_gs)
                    else:
                        packed_bytes, scales_bytes, rn, rk = store.repack_marlin_int8_cpu(
                            w_cpu.data_ptr(), n, k, marlin_gs)
                    del w_cpu

                    # Step 3: Create GPU tensors from Rust-repacked data
                    # CRITICAL: scales are raw BF16 bits stored as u16 — must use
                    # .view(torch.bfloat16) to reinterpret bits, NOT .to(torch.bfloat16)
                    # which would numerically convert (e.g. 15890 -> BF16(15872.0) instead
                    # of BF16(0.1429)).
                    import numpy as np
                    packed_np = np.frombuffer(packed_bytes, dtype=np.uint32)
                    scales_np = np.frombuffer(scales_bytes, dtype=np.uint16)
                    repacked = torch.from_numpy(packed_np.copy()).to(
                        dtype=torch.int32, device=device)
                    scale_raw = torch.from_numpy(scales_np.copy()).to(torch.int16)

                    # Single-slot AWQ: allocate scales at FP32 size (2x BF16) so the
                    # same GPU buffer holds either BF16 Marlin scales or FP32 simple scales.
                    # BF16 scales occupy the first half; gptq_marlin_gemm gets a BF16 view.
                    n_scale_elements = (k // marlin_gs) * n
                    if use_int4:
                        # Allocate slot: 2x BF16 elements as int16 = FP32 byte capacity
                        scales_slot = torch.empty(n_scale_elements * 2, dtype=torch.int16, device=device)
                        # Fill first half with BF16 data
                        bf16_on_gpu = scale_raw.view(torch.bfloat16).to(device)
                        scales_slot[:n_scale_elements].copy_(bf16_on_gpu.view(torch.int16).reshape(-1))
                        # BF16 view for gptq_marlin_gemm (same underlying memory, first half)
                        scale_perm = scales_slot[:n_scale_elements].view(torch.bfloat16).reshape(k // marlin_gs, n)
                        del bf16_on_gpu
                    else:
                        # INT8: no single-slot swap, standard BF16 scales
                        scale_perm = scale_raw.view(torch.bfloat16).to(device)

                    # Reshape for gptq_marlin_gemm: packed [K/16, pack_factor*N], scales [K/gs, N]
                    if use_int4:
                        repacked = repacked.reshape(k // 16, 2 * n)
                    else:
                        repacked = repacked.reshape(k // 16, 4 * n)
                    if not use_int4:
                        scale_perm = scale_perm.reshape(k // marlin_gs, n)

                    # Step 4: Register GPU pointer with Rust store
                    # For INT4 single-slot: scales_ptr points to the full FP32-sized slot
                    scalar_type = _marlin_scalar_type_int4 if use_int4 else _marlin_scalar_type_int8
                    if use_int4:
                        scales_data_ptr = scales_slot.data_ptr() if use_int4 else scale_perm.data_ptr()
                        wid = store.register_marlin_int4_weight(
                            repacked.data_ptr(), scales_data_ptr,
                            n, k, marlin_gs)
                    else:
                        wid = store.register_marlin_int8_weight(
                            repacked.data_ptr(), scale_perm.data_ptr(),
                            n, k, marlin_gs)

                    # Step 5: Keep PyTorch tensors alive and accessible for prefill.
                    # Single-slot: tensors stay on GPU permanently (they ARE the slots).
                    # For INT4, also keep scales_slot alive to prevent GC.
                    self._marlin_attn_weights[wid] = (
                        repacked, scale_perm, _marlin_workspace,
                        scalar_type, n, k,
                    )
                    if use_int4:
                        # Keep the full slot tensor alive (scale_perm is a view into it)
                        if not hasattr(self, '_single_slot_scales'):
                            self._single_slot_scales = []
                        self._single_slot_scales.append(scales_slot)
                    return wid
                else:
                    logger.warning("Attention weight [%d×%d] not Marlin-compatible "
                                   "(need N%%64==0, K%%16==0, K%%gs==0), using BF16", n, k)
            # BF16 path: weight must be on GPU
            if not w.is_cuda:
                w = w.to(device, non_blocking=True)
                self._rust_decode_weights.append(w)  # prevent GC
            return store.register_weight(
                w.data_ptr(), w.shape[0], w.shape[1], _weight_dtype_code(w))

        self._marlin_attn_weights = {}  # wid -> (packed, scales, workspace, scalar_type, N, K)

        # Stash BF16 CPU copies of attention weights for aux store AWQ quantization.
        # Needed when streaming attention is not enabled and attrs get replaced by MarlinWeight.
        if not getattr(self, '_stream_attn_enabled', False):
            self._aux_bf16_stash = {}  # {(layer_idx, attr_name): cpu_tensor}
        else:
            self._aux_bf16_stash = None  # streaming has its own CPU copies

        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attention
            inp_norm = layer.input_norm_weight
            post_norm = layer.post_attn_norm_weight

            py_diag = os.environ.get("KRASIS_PY_DIAG") == "1"

            # DIAG: check raw norm values BEFORE AWQ fold
            if py_diag and layer_idx == 0:
                import sys
                _rv = inp_norm.data.cpu().float()[:10].tolist()
                _rl2 = inp_norm.data.cpu().float().norm().item()
                print(f"[PY-DIAG] layer0 inp_norm BEFORE_AWQ first10={_rv}", file=sys.stderr)
                print(f"[PY-DIAG] layer0 inp_norm BEFORE_AWQ L2={_rl2:.6f}", file=sys.stderr)

            # ── Nemotron Mamba2 layer registration ──
            if layer.layer_type == "mamba2":
                self._register_mamba2_layer(store, layer_idx, layer, device)
                store.register_mlp(layer_idx, "none")
                continue

            # ── Nemotron MoE-only layer (no attention) ──
            if layer.layer_type == "moe":
                self._register_nemotron_moe_only_layer(store, layer_idx, layer, device)
                continue

            # AWQ v2: load per-layer per-channel scales for input projections
            _layer_awq_scales = None
            if attn_quant == "awq" and _awq_template is not None:
                from krasis.awq_calibrate import (
                    get_layer_scales, is_awq_scaled_tensor)
                _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)

            if layer.layer_type == "linear_attention":
                # Source projection weights — when quantizing, keep on CPU to avoid
                # putting full BF16 in VRAM. Only upload to GPU for BF16 mode.
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_w = self._stream_attn_cpu[layer_idx]
                    if attn_quant == "bf16":
                        qkvz_w = cpu_w.get("in_proj_qkvz", attn.in_proj_qkvz).to(device, non_blocking=True)
                        ba_w = cpu_w.get("in_proj_ba", attn.in_proj_ba).to(device, non_blocking=True)
                        out_w = cpu_w.get("out_proj", attn.out_proj).to(device, non_blocking=True)
                        self._rust_decode_weights.extend([qkvz_w, ba_w, out_w])
                    else:
                        # Quantizing: pass CPU tensors directly, never touch VRAM with BF16
                        qkvz_w = cpu_w.get("in_proj_qkvz", attn.in_proj_qkvz)
                        ba_w = cpu_w.get("in_proj_ba", attn.in_proj_ba)
                        out_w = cpu_w.get("out_proj", attn.out_proj)
                else:
                    qkvz_w = attn.in_proj_qkvz
                    ba_w = attn.in_proj_ba
                    out_w = attn.out_proj

                # AWQ v2: pass per-channel scales for input projections only
                _qkvz_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_qkvz")
                ) else None
                _ba_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_ba")
                ) else None

                qkvz_wid = _register_attn_weight(qkvz_w, layer_idx, "linear_attention",
                                                  "in_proj_qkvz", awq_scales=_qkvz_scales)
                ba_wid = _register_attn_weight(ba_w, layer_idx, "linear_attention",
                                               "in_proj_ba", awq_scales=_ba_scales)
                out_wid = _register_attn_weight(out_w, layer_idx, "linear_attention", "out_proj")
                self._la_wids[layer_idx] = (qkvz_wid, ba_wid, out_wid)

                # Replace attention weight attributes with MarlinWeight for prefill
                if qkvz_wid in self._marlin_attn_weights:
                    from krasis.attention import MarlinWeight
                    mw = self._marlin_attn_weights[qkvz_wid]
                    attn.in_proj_qkvz = MarlinWeight(*mw)
                if ba_wid in self._marlin_attn_weights:
                    from krasis.attention import MarlinWeight
                    mw = self._marlin_attn_weights[ba_wid]
                    attn.in_proj_ba = MarlinWeight(*mw)
                if out_wid in self._marlin_attn_weights:
                    from krasis.attention import MarlinWeight
                    mw = self._marlin_attn_weights[out_wid]
                    attn.out_proj = MarlinWeight(*mw)

                # AWQ v2: fold 1/s into input_norm_weight.
                # This makes RMSNorm output X_scaled = X / s, which compensates
                # for the W * s scaling applied to weight columns above.
                # Mathematical identity: (X/s) @ (W*s)^T = X @ W^T
                if _layer_awq_scales is not None and (_qkvz_scales is not None
                                                       or _ba_scales is not None):
                    s = _layer_awq_scales.to(inp_norm.device)
                    # In-place modification: norm_weight[j] /= s[j]
                    inp_norm.data.copy_(
                        (inp_norm.float() / s.float()).to(inp_norm.dtype))
                    logger.debug("AWQ: folded scales into input_norm for layer %d "
                                 "(mean_scale=%.4f)", layer_idx, s.mean().item())

                # DIAG: trace input_norm pointer and values at registration time
                if py_diag and layer_idx == 0:
                    import sys
                    _nv = inp_norm.data.cpu().float()[:10].tolist()
                    _nl2 = inp_norm.data.cpu().float().norm().item()
                    print(f"[PY-DIAG] layer0 inp_norm ptr={inp_norm.data_ptr():#x} "
                          f"numel={inp_norm.numel()} dtype={inp_norm.dtype} "
                          f"device={inp_norm.device}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 inp_norm first10={_nv}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 inp_norm L2={_nl2:.6f}", file=sys.stderr)
                    # Also check post_norm
                    _pv = post_norm.data.cpu().float()[:10].tolist()
                    _pl2 = post_norm.data.cpu().float().norm().item()
                    print(f"[PY-DIAG] layer0 post_norm ptr={post_norm.data_ptr():#x} "
                          f"numel={post_norm.numel()} dtype={post_norm.dtype}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 post_norm first10={_pv}", file=sys.stderr)
                    print(f"[PY-DIAG] layer0 post_norm L2={_pl2:.6f}", file=sys.stderr)

                # Conv weight: [conv_dim, 1, kernel_dim] -> [conv_dim, kernel_dim]
                # Rust LA kernels work in FP32, so create FP32 copies of BF16 tensors.
                # When streaming, attn.* may be None or stale — use CPU-pinned source.
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_la = self._stream_attn_cpu[layer_idx]
                    conv_src = cpu_la.get("conv1d_weight", attn.conv1d_weight)
                    a_log_src = cpu_la.get("A_log", attn.A_log)
                    dt_bias_src = cpu_la.get("dt_bias", attn.dt_bias)
                    norm_w_src = cpu_la.get("norm_weight", attn.norm_weight)
                else:
                    conv_src = attn.conv1d_weight
                    a_log_src = attn.A_log
                    dt_bias_src = attn.dt_bias
                    norm_w_src = attn.norm_weight
                conv_w = conv_src.squeeze(1).contiguous().float().to(device)
                attn._rust_conv_weight = conv_w

                attn._init_state(batch_size=1)
                # conv_state is BF16 from _init_state, Rust kernel needs FP32
                attn._rust_conv_state = attn._conv_state.squeeze(0).float().contiguous()
                # norm_weight is BF16, Rust gated_rmsnorm_silu kernel needs FP32
                attn._rust_norm_weight = norm_w_src.float().contiguous().to(device)
                # recurrent_state may need FP32 conversion for Rust kernel
                attn._rust_recur_state = attn._recurrent_state.squeeze(0).float().contiguous()
                # A_log and dt_bias are BF16 from weight loader, Rust kernel needs FP32
                attn._rust_a_log = a_log_src.float().contiguous().to(device)
                attn._rust_dt_bias = dt_bias_src.float().contiguous().to(device)

                store.register_la_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                    in_proj_qkvz_wid=qkvz_wid, in_proj_ba_wid=ba_wid, out_proj_wid=out_wid,
                    conv_weight_ptr=conv_w.data_ptr(),
                    a_log_ptr=attn._rust_a_log.data_ptr(),
                    dt_bias_ptr=attn._rust_dt_bias.data_ptr(),
                    norm_weight_ptr=attn._rust_norm_weight.data_ptr(),
                    conv_state_ptr=attn._rust_conv_state.data_ptr(),
                    recur_state_ptr=attn._rust_recur_state.data_ptr(),
                    nk=attn.num_k_heads, nv=attn.num_v_heads,
                    dk=attn.k_head_dim, dv=attn.v_head_dim,
                    hr=attn.head_ratio,
                    kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                    scale=attn.scale,
                )
            elif hasattr(attn, 'kv_a_proj'):
                # MLA attention — register projections and absorbed weights
                # AWQ v2: per-channel scales for MLA input projections
                _qa_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "q_a_proj")
                ) else None
                _qproj_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                ) else None
                _kva_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "kv_a_proj")
                ) else None

                # Q projection weights
                if attn.has_q_lora:
                    qa_w = attn.q_a_proj
                    qb_w = attn.q_b_proj
                    qa_wid = _register_attn_weight(qa_w, layer_idx, "mla", "q_a_proj",
                                                    awq_scales=_qa_scales)
                    qb_wid = _register_attn_weight(qb_w, layer_idx, "mla", "q_b_proj")
                    # q_a layernorm: BF16 → FP32 for Rust per_head_rmsnorm
                    attn._rust_q_a_norm = attn.q_a_norm_weight.float().contiguous().to(device)
                    q_a_norm_ptr = attn._rust_q_a_norm.data_ptr()
                    self._rust_decode_weights.append(attn._rust_q_a_norm)
                    q_proj_wid = None
                else:
                    q_w = attn.q_proj
                    q_proj_wid = _register_attn_weight(q_w, layer_idx, "mla", "q_proj",
                                                        awq_scales=_qproj_scales)
                    qa_wid = None
                    qb_wid = None
                    q_a_norm_ptr = 0

                # KV projection
                kva_w = attn.kv_a_proj
                kva_wid = _register_attn_weight(kva_w, layer_idx, "mla", "kv_a_proj",
                                                 awq_scales=_kva_scales)

                # O projection
                o_w = attn.o_proj
                o_wid = _register_attn_weight(o_w, layer_idx, "mla", "o_proj")

                # Replace MLA attention weight attributes with MarlinWeight for prefill
                from krasis.attention import MarlinWeight
                if attn.has_q_lora:
                    if qa_wid in self._marlin_attn_weights:
                        attn.q_a_proj = MarlinWeight(*self._marlin_attn_weights[qa_wid])
                    if qb_wid in self._marlin_attn_weights:
                        attn.q_b_proj = MarlinWeight(*self._marlin_attn_weights[qb_wid])
                else:
                    if q_proj_wid is not None and q_proj_wid in self._marlin_attn_weights:
                        attn.q_proj = MarlinWeight(*self._marlin_attn_weights[q_proj_wid])
                if kva_wid in self._marlin_attn_weights:
                    attn.kv_a_proj = MarlinWeight(*self._marlin_attn_weights[kva_wid])
                if o_wid in self._marlin_attn_weights:
                    attn.o_proj = MarlinWeight(*self._marlin_attn_weights[o_wid])

                # AWQ v2: fold 1/s into input_norm_weight for MLA layers
                if _layer_awq_scales is not None and (
                    _qa_scales is not None or _qproj_scales is not None
                    or _kva_scales is not None
                ):
                    s = _layer_awq_scales.to(inp_norm.device)
                    inp_norm.data.copy_(
                        (inp_norm.float() / s.float()).to(inp_norm.dtype))
                    logger.debug("AWQ: folded scales into input_norm for MLA layer %d "
                                 "(mean_scale=%.4f)", layer_idx, s.mean().item())

                # kv_a layernorm: BF16 → FP32
                attn._rust_kv_a_norm = attn.kv_a_norm_weight.float().contiguous().to(device)
                self._rust_decode_weights.append(attn._rust_kv_a_norm)

                # w_kc and w_vc: keep as BF16 on GPU (read by CUDA kernels directly)
                # Shape: [num_heads, dim_a, dim_b] — ensure contiguous
                attn._rust_w_kc = attn.w_kc.contiguous().to(device)
                attn._rust_w_vc = attn.w_vc.contiguous().to(device)
                self._rust_decode_weights.extend([attn._rust_w_kc, attn._rust_w_vc])

                # MLA FP8 KV caches: share paged cache directly
                # For single-sequence decode, paged layout [pages, page_size, dim]
                # is identical to flat layout [position, dim] in memory.
                cache = self.kv_caches[0]
                if not hasattr(self, '_mla_cache_offset'):
                    self._mla_cache_offset = 0
                mla_offset = self._mla_cache_offset
                self._mla_cache_offset += 1
                ckv_layer = cache.ckv_cache[mla_offset]  # [pages, page_size, kv_lora_rank]
                kpe_layer = cache.kpe_cache[mla_offset]  # [pages, page_size, qk_rope_dim]
                max_seq = cache.max_pages * cache.page_size

                store.register_mla_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                    kv_a_proj_wid=kva_wid, o_proj_wid=o_wid,
                    kv_a_norm_ptr=attn._rust_kv_a_norm.data_ptr(),
                    w_kc_ptr=attn._rust_w_kc.data_ptr(),
                    w_vc_ptr=attn._rust_w_vc.data_ptr(),
                    num_heads=attn.num_heads,
                    kv_lora_rank=attn.kv_lora_rank,  # real kv_lora_rank (256 for Mistral)
                    qk_nope_dim=attn.qk_nope_dim,
                    qk_rope_dim=attn.qk_rope_dim,
                    v_head_dim=attn.v_head_dim,
                    sm_scale=attn.sm_scale,
                    rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                    ckv_cache_ptr=ckv_layer.data_ptr(),
                    kpe_cache_ptr=kpe_layer.data_ptr(),
                    q_a_proj_wid=qa_wid,
                    q_b_proj_wid=qb_wid,
                    q_a_norm_ptr=q_a_norm_ptr,
                    q_proj_wid=q_proj_wid,
                    q_lora_rank=attn.q_lora_rank if attn.has_q_lora else 0,
                    ckv_cache_dim=attn.ckv_dim,  # padded to ≥512 for MLA decode
                )

                # Set up RoPE tables from first MLA layer
                if not rope_set:
                    cos, sin = attn._get_rope_cos_sin(max_seq)
                    cos_f32 = cos.float().contiguous()
                    sin_f32 = sin.float().contiguous()
                    self._rust_rope_cos = cos_f32
                    self._rust_rope_sin = sin_f32
                    store.set_rope_tables(
                        cos_f32.data_ptr(), sin_f32.data_ptr(),
                        cos_f32.shape[1], max_seq,
                    )
                    rope_set = True

            else:
                # GQA attention — get weights from layer.gqa_weights dict
                # (attn object was removed when FlashInfer was dropped)
                gqa_w = layer.gqa_weights
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_w = self._stream_attn_cpu[layer_idx]
                    if attn_quant == "bf16":
                        q_w = cpu_w.get("q_proj", gqa_w["q_proj"]).to(device, non_blocking=True)
                        k_w = cpu_w.get("k_proj", gqa_w["k_proj"]).to(device, non_blocking=True)
                        v_w = cpu_w.get("v_proj", gqa_w["v_proj"]).to(device, non_blocking=True)
                        o_w = cpu_w.get("o_proj", gqa_w["o_proj"]).to(device, non_blocking=True)
                        self._rust_decode_weights.extend([q_w, k_w, v_w, o_w])
                    else:
                        # Quantizing: pass CPU tensors directly, never touch VRAM with BF16
                        q_w = cpu_w.get("q_proj", gqa_w["q_proj"])
                        k_w = cpu_w.get("k_proj", gqa_w["k_proj"])
                        v_w = cpu_w.get("v_proj", gqa_w["v_proj"])
                        o_w = cpu_w.get("o_proj", gqa_w["o_proj"])
                else:
                    q_w = gqa_w["q_proj"]
                    k_w = gqa_w["k_proj"]
                    v_w = gqa_w["v_proj"]
                    o_w = gqa_w["o_proj"]

                # AWQ v2: pass per-channel scales for input projections (q/k/v)
                _q_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                ) else None
                _k_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "k_proj")
                ) else None
                _v_scales = _layer_awq_scales if (
                    _layer_awq_scales is not None and _awq_template is not None
                    and is_awq_scaled_tensor(_awq_template, layer_idx, "v_proj")
                ) else None

                q_wid = _register_attn_weight(q_w, layer_idx, "gqa", "q_proj",
                                              awq_scales=_q_scales)
                k_wid = _register_attn_weight(k_w, layer_idx, "gqa", "k_proj",
                                              awq_scales=_k_scales)
                v_wid = _register_attn_weight(v_w, layer_idx, "gqa", "v_proj",
                                              awq_scales=_v_scales)
                o_wid = _register_attn_weight(o_w, layer_idx, "gqa", "o_proj")

                # Replace attention weight references with MarlinWeight for prefill
                from krasis.attention import MarlinWeight
                gqa_w = layer.gqa_weights if hasattr(layer, 'gqa_weights') else None
                if q_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[q_wid])
                    if gqa_w is not None:
                        gqa_w["q_proj"] = mw
                    elif attn is not None:
                        attn.q_proj = mw
                if k_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[k_wid])
                    if gqa_w is not None:
                        gqa_w["k_proj"] = mw
                    elif attn is not None:
                        attn.k_proj = mw
                if v_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[v_wid])
                    if gqa_w is not None:
                        gqa_w["v_proj"] = mw
                    elif attn is not None:
                        attn.v_proj = mw
                if o_wid in self._marlin_attn_weights:
                    mw = MarlinWeight(*self._marlin_attn_weights[o_wid])
                    if gqa_w is not None:
                        gqa_w["o_proj"] = mw
                    elif attn is not None:
                        attn.o_proj = mw

                # AWQ v2: fold 1/s into input_norm_weight for GQA layers
                if _layer_awq_scales is not None and (
                    _q_scales is not None or _k_scales is not None
                    or _v_scales is not None
                ):
                    s = _layer_awq_scales.to(inp_norm.device)
                    inp_norm.data.copy_(
                        (inp_norm.float() / s.float()).to(inp_norm.dtype))
                    logger.debug("AWQ: folded scales into input_norm for GQA layer %d "
                                 "(mean_scale=%.4f)", layer_idx, s.mean().item())

                # QK norm weights are BF16, Rust per_head_rmsnorm kernel needs FP32
                # When streaming, use CPU-pinned source for these small weights
                if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                    cpu_gqa = self._stream_attn_cpu[layer_idx]
                    q_norm_src = cpu_gqa.get("q_norm", gqa_w.get("q_norm"))
                    k_norm_src = cpu_gqa.get("k_norm", gqa_w.get("k_norm"))
                else:
                    q_norm_src = gqa_w.get("q_norm")
                    k_norm_src = gqa_w.get("k_norm")
                if q_norm_src is not None:
                    _rust_q_norm = q_norm_src.float().contiguous().to(device)
                    self._rust_decode_weights.append(_rust_q_norm)
                    q_norm_ptr = _rust_q_norm.data_ptr()
                else:
                    q_norm_ptr = 0
                if k_norm_src is not None:
                    _rust_k_norm = k_norm_src.float().contiguous().to(device)
                    self._rust_decode_weights.append(_rust_k_norm)
                    k_norm_ptr = _rust_k_norm.data_ptr()
                else:
                    k_norm_ptr = 0

                post_norm_ptr = post_norm.data_ptr() if post_norm is not None else 0
                post_norm_size = post_norm.numel() if post_norm is not None else 0
                _gqa_head_dim = self.cfg.gqa_head_dim or self.cfg.head_dim
                _gqa_sm_scale = 1.0 / (_gqa_head_dim ** 0.5)
                _gqa_gated = hasattr(self.cfg, 'gated_attention') and self.cfg.gated_attention
                store.register_gqa_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm_ptr, post_attn_norm_size=post_norm_size,
                    q_proj_wid=q_wid, k_proj_wid=k_wid,
                    v_proj_wid=v_wid, o_proj_wid=o_wid,
                    fused_qkv_wid=None,
                    num_heads=self.cfg.num_attention_heads,
                    num_kv_heads=self.cfg.num_key_value_heads,
                    head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                    q_norm_ptr=q_norm_ptr, k_norm_ptr=k_norm_ptr,
                    gated=_gqa_gated,
                )

                # Set up RoPE tables from first GQA layer
                if not rope_set:
                    max_seq = max(
                        c.max_context_tokens for c in self.kv_caches if c is not None
                    )
                    rope_half = self.cfg.rotary_dim // 2
                    inv_freq = 1.0 / (self.cfg.rope_theta ** (
                        torch.arange(0, rope_half * 2, 2, dtype=torch.float32) / (rope_half * 2)))
                    t = torch.arange(max_seq, dtype=torch.float32)
                    freqs = torch.outer(t, inv_freq)
                    cos_f32 = freqs.cos().contiguous().to(device)
                    sin_f32 = freqs.sin().contiguous().to(device)
                    self._rust_rope_cos = cos_f32
                    self._rust_rope_sin = sin_f32
                    store.set_rope_tables(
                        cos_f32.data_ptr(), sin_f32.data_ptr(),
                        cos_f32.shape[1], max_seq,
                    )
                    rope_set = True

            # Register MLP type
            if layer.is_moe:
                store.register_mlp(layer_idx, "moe")
            elif layer.dense_mlp is not None:
                gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"])
                up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"])
                dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"])
                self._rust_decode_weights.extend([gp, up, dp])
                gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                store.register_mlp(layer_idx, "dense",
                                   gate_proj_wid=gp_wid, up_proj_wid=up_wid, down_proj_wid=dp_wid)
            else:
                store.register_mlp(layer_idx, "none")

        # Register MoE expert data from engine (all MoE layers at once)
        if self.krasis_engine is not None:
            store.setup_from_engine(self.krasis_engine)

            # WriteCombined DMA staging: migrate expert weights from heap to WC memory.
            # Must happen after setup_from_engine (pointers exist) but before
            # allocate_prefill_engine (which reads the updated pointers).
            if getattr(self, 'wc_alloc', False):
                wc_msg = store.setup_wc_expert_memory(self.krasis_engine)
                logger.info("WC expert memory: %s", wc_msg)

        # Register Nemotron MoE config (relu2, ungated, latent projections) — must come
        # after setup_from_engine which populates moe_layers[abs_layer_idx].
        if self.cfg.model_type == "nemotron_h":
            for layer_idx, layer in enumerate(self.layers):
                if layer.layer_type == "moe":
                    latent = layer.latent_proj
                    if latent is not None:
                        fc1_down = latent["fc1_latent_proj"]
                        fc2_up = latent["fc2_latent_proj"]
                        self._rust_decode_weights.extend([fc1_down, fc2_up])
                        ld_wid = store.register_weight(
                            fc1_down.data_ptr(), fc1_down.shape[0], fc1_down.shape[1], 0)
                        lu_wid = store.register_weight(
                            fc2_up.data_ptr(), fc2_up.shape[0], fc2_up.shape[1], 0)
                    else:
                        ld_wid = None
                        lu_wid = None
                    store.set_moe_nemotron_config(
                        layer_idx=layer_idx,
                        activation_type=1,
                        gated_experts=False,
                        latent_down_wid=ld_wid,
                        latent_up_wid=lu_wid,
                        moe_input_size=self.cfg.moe_latent_size,
                    )
                    logger.info("Set Nemotron MoE config for layer %d (latent_down=%s, latent_up=%s)",
                                layer_idx, ld_wid, lu_wid)

        # Register shared_expert_gate weights (sigmoid gate for shared expert output)
        # These are BF16 tensors loaded by Python, not in the Rust engine cache
        self._rust_shared_gate_refs = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_moe and layer.shared_expert_gate is not None:
                sg = layer.shared_expert_gate
                sg_wid = store.register_weight(sg.data_ptr(), sg.shape[0], sg.shape[1], 0)
                store.set_moe_shared_gate_wid(layer_idx, sg_wid)
                self._rust_shared_gate_refs.append(sg)  # prevent GC
                logger.info("Registered shared_expert_gate for layer %d: wid=%d shape=%s",
                            layer_idx, sg_wid, sg.shape)

        # Register shared KV cache pointers — Rust decode reads/writes the
        # same GPU buffers that Rust prefill writes. No separate
        # allocation, no D2D export copy between prefill and decode.
        cache = self.kv_caches[0]
        if cache is not None and cache.kv_format == 2 and cache.k_radius_cache is not None:
            # Polar4 KV cache: 4 pointer sets (radius + angles for K and V)
            polar4_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    kr = cache.k_radius_cache[gqa_cache_idx]
                    vr = cache.v_radius_cache[gqa_cache_idx]
                    ka = cache.k_angles_cache[gqa_cache_idx]
                    va = cache.v_angles_cache[gqa_cache_idx]
                    gqa_cache_idx += 1
                    polar4_ptrs.append((layer_idx,
                                        kr.data_ptr(), vr.data_ptr(),
                                        ka.data_ptr(), va.data_ptr()))
            max_seq = cache.max_pages * cache.page_size
            num_blocks = (cache.num_kv_heads * cache.gqa_head_dim) // 16
            store.set_kv_cache_ptrs_polar4(polar4_ptrs, max_seq, num_blocks)
            logger.info("Shared Polar4 KV cache: %d GQA layers, max_seq=%d (%d pages × %d), %d blocks",
                        len(polar4_ptrs), max_seq, cache.max_pages, cache.page_size, num_blocks)
        elif cache is not None and cache.k_cache is not None:
            kv_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                # Only full GQA attention layers have KV cache slots.
                # Skip: linear_attention (no KV), MLA (has kv_a_proj), mamba2 (SSM state), moe-only (no attention).
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    k_layer = cache.k_cache[gqa_cache_idx]  # [max_pages, page_size, nkv, hd]
                    v_layer = cache.v_cache[gqa_cache_idx]
                    gqa_cache_idx += 1
                    kv_ptrs.append((layer_idx, k_layer.data_ptr(), v_layer.data_ptr()))
            max_seq = cache.max_pages * cache.page_size
            store.set_kv_cache_ptrs(kv_ptrs, max_seq)
            logger.info("Shared FP8 KV cache: %d GQA layers, max_seq=%d (%d pages × %d)",
                        len(kv_ptrs), max_seq, cache.max_pages, cache.page_size)
        elif cache is not None and cache.ckv_cache is not None:
            # MLA-only model (no GQA layers): still need to set max_seq for Rust decode
            max_seq = cache.max_pages * cache.page_size
            store.set_kv_cache_ptrs([], max_seq)
            logger.info("MLA-only KV cache: max_seq=%d (%d pages × %d)",
                        max_seq, cache.max_pages, cache.page_size)

        self._gpu_decode_store = store

        # Enable per-component timing if KRASIS_DECODE_TIMING=1
        if os.environ.get("KRASIS_DECODE_TIMING", "") == "1":
            store.set_timing(True)
            logger.info("GPU decode timing enabled (KRASIS_DECODE_TIMING=1)")

        # Log actual attention VRAM after AWQ quantization (if applicable)
        if self.quant_cfg.attention == "awq":
            attn_mb = self._estimate_attention_vram() >> 20
            dev = torch.device(self.ranks[0].device)
            free_mb = torch.cuda.mem_get_info(dev)[0] >> 20
            logger.info("Attention after AWQ quantization: %d MB, GPU free: %d MB", attn_mb, free_mb)
            print(f"  \033[0;32mAttention after AWQ: {attn_mb} MB, {free_mb} MB free\033[0m", flush=True)

        logger.info("GPU decode store configured: %d layers, store_addr=%d",
                     len(self.layers), store.gpu_store_addr())

        # Single-slot AWQ: Marlin data is in the GPU slots from registration.
        # PyTorch tensors stay on GPU permanently (they ARE the slots).
        # Rust handles swapping slot contents between Marlin and simple INT4.

        return store

    def _register_mamba2_layer(self, store, layer_idx, layer, device):
        """Register a Mamba2 SSM layer with the Rust decode store."""
        inp_norm = layer.input_norm_weight
        m2 = layer.mamba2_weights
        cfg = self.cfg

        # Mamba2 projection weights: in_proj [in_proj_out, hidden_size], out_proj [hidden_size, d_inner]
        # Register as Marlin GEMV weights for decode
        in_proj = m2["in_proj"]
        out_proj = m2["out_proj"]
        self._rust_decode_weights.extend([in_proj, out_proj])
        in_proj_wid = store.register_weight(
            in_proj.data_ptr(), in_proj.shape[0], in_proj.shape[1], 0)
        out_proj_wid = store.register_weight(
            out_proj.data_ptr(), out_proj.shape[0], out_proj.shape[1], 0)

        # Conv1d weight: [conv_dim, conv_kernel] FP32 on GPU
        conv_w = m2["conv1d_weight"].float().contiguous().to(device)
        self._rust_decode_weights.append(conv_w)

        # Conv1d bias: [conv_dim] FP32 on GPU (optional)
        conv_bias = m2.get("conv1d_bias")
        if conv_bias is not None:
            conv_bias = conv_bias.float().contiguous().to(device)
            self._rust_decode_weights.append(conv_bias)
            store.set_mamba2_conv_bias(layer_idx, conv_bias.data_ptr())

        # A_log, D, dt_bias, norm: all FP32 on GPU
        a_log = m2["A_log"].float().contiguous().to(device)
        d_param = m2["D"].float().contiguous().to(device)
        dt_bias = m2["dt_bias"].float().contiguous().to(device)
        norm_w = m2["norm_weight"].float().contiguous().to(device)
        self._rust_decode_weights.extend([a_log, d_param, dt_bias, norm_w])

        # Allocate SSM state buffers on GPU (persistent across tokens)
        # Conv state: [conv_dim, conv_kernel] FP32
        # SSM state: [num_heads, head_dim, state_size] FP32
        conv_dim = cfg.mamba_conv_dim
        conv_state = torch.zeros(conv_dim, cfg.mamba_conv_kernel,
                                 dtype=torch.float32, device=device)
        ssm_state = torch.zeros(cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size,
                                dtype=torch.float32, device=device)
        self._rust_decode_weights.extend([conv_state, ssm_state])
        # Save references for prefill -> decode state transfer
        if not hasattr(self, '_mamba2_decode_states'):
            self._mamba2_decode_states = {}
        self._mamba2_decode_states[layer_idx] = {
            'conv_state': conv_state,
            'ssm_state': ssm_state,
        }

        store.register_mamba2_layer(
            layer_idx=layer_idx,
            input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
            post_attn_norm_ptr=0, post_attn_norm_size=0,  # Nemotron: no post_attn_norm
            in_proj_wid=in_proj_wid, out_proj_wid=out_proj_wid,
            conv_weight_ptr=conv_w.data_ptr(),
            a_ptr=a_log.data_ptr(), d_ptr=d_param.data_ptr(),
            dt_bias_ptr=dt_bias.data_ptr(), norm_weight_ptr=norm_w.data_ptr(),
            conv_state_ptr=conv_state.data_ptr(), ssm_state_ptr=ssm_state.data_ptr(),
            num_heads=cfg.mamba_num_heads, head_dim=cfg.mamba_head_dim,
            state_size=cfg.ssm_state_size,
            expand=cfg.mamba_expand, conv_kernel=cfg.mamba_conv_kernel,
        )
        # Set n_groups for B/C sharing (same for all Mamba2 layers)
        store.set_mamba2_n_groups(cfg.mamba_n_groups)

        logger.info("Registered Mamba2 layer %d (conv_dim=%d, heads=%d, state=%d)",
                     layer_idx, conv_dim, cfg.mamba_num_heads, cfg.ssm_state_size)

    def _register_nemotron_moe_only_layer(self, store, layer_idx, layer, device):
        """Register a Nemotron MoE-only layer (no attention) with the Rust decode store.

        These layers have: input_norm → LatentMoE → residual (no attention, no post_attn_norm).
        Since the Rust decode loop expects input_norm + attn + post_attn_norm + MLP, we register
        the layer with a 'None' attention type and the MoE as the MLP.
        """
        inp_norm = layer.input_norm_weight

        # Register as GQA with 0 heads (skips attention computation).
        # post_attn_norm ptr=0/size=0 tells Rust to skip post-norm (Nemotron has no post_attn_norm).
        store.register_gqa_layer(
            layer_idx=layer_idx,
            input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
            post_attn_norm_ptr=0, post_attn_norm_size=0,
            q_proj_wid=0, k_proj_wid=0, v_proj_wid=0, o_proj_wid=0,
            fused_qkv_wid=None,
            num_heads=0, num_kv_heads=0, head_dim=0, sm_scale=0.0,
            q_norm_ptr=0, k_norm_ptr=0, gated=False,
        )

        # Register MoE placeholder — actual expert data is wired by setup_from_engine,
        # and Nemotron config (relu2, latent projections) is set after that.
        store.register_mlp(layer_idx, "moe")

        logger.info("Registered Nemotron MoE-only layer %d (MoE config deferred to post-engine setup)",
                     layer_idx)

    def setup_gpu_decode_store_aux(self, gpu_idx: int, split_layer: int, layer_end: int = 0) -> "GpuDecodeStore":
        """Create an auxiliary GpuDecodeStore for multi-GPU decode.

        Sets up a store on gpu_idx for layers [split_layer..layer_end).
        If layer_end=0, defaults to num_layers (last GPU in pipeline).
        Registers attention weights (copied to aux GPU), KV cache (allocated on
        aux GPU), MoE layers, final norm, LM head, and RoPE on the aux GPU.
        Only the last segment (layer_end == num_layers) gets real final norm + LM head.
        """
        from krasis import GpuDecodeStore
        import torch

        num_layers = len(self.layers)
        if layer_end <= 0:
            layer_end = num_layers
        is_last_segment = (layer_end == num_layers)

        aux_device = torch.device(f"cuda:{gpu_idx}")
        primary_device = torch.device(self.ranks[0].device)

        store = GpuDecodeStore(gpu_idx)

        # Same configure as primary
        max_qkv = self.cfg.hidden_size * 3
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                attn = layer.attention
                qkvz_out = attn.num_k_heads * (2 * attn.k_head_dim + 2 * attn.head_ratio * attn.v_head_dim)
                max_qkv = max(max_qkv, qkvz_out)
            elif hasattr(layer.attention, 'kv_a_proj'):
                # MLA: same calculation as primary store
                ma = layer.attention
                q_out = ma.num_heads * (ma.qk_nope_dim + ma.qk_rope_dim)
                q_absorbed = ma.num_heads * ma.ckv_dim
                kv_out = ma.ckv_dim + ma.qk_rope_dim
                max_qkv = max(max_qkv, q_out, q_absorbed, kv_out)
            elif hasattr(layer.attention, 'num_heads'):
                ga = layer.attention
                q_sz = ga.num_heads * ga.head_dim * (2 if ga.gated_attention else 1)
                kv_sz = ga.num_kv_heads * ga.head_dim
                max_qkv = max(max_qkv, q_sz + kv_sz * 2)

        has_dense_mlp = any(l.dense_mlp is not None for l in self.layers)
        max_inter = self.cfg.moe_intermediate_size
        if has_dense_mlp:
            max_inter = max(max_inter, self.cfg.intermediate_size)

        store.configure(
            hidden_size=self.cfg.hidden_size,
            num_layers=len(self.layers),
            vocab_size=self.cfg.vocab_size,
            eps=self.cfg.rms_norm_eps,
            max_experts_per_tok=self.cfg.num_experts_per_tok,
            max_intermediate_size=max_inter,
            max_qkv_size=max_qkv,
            group_size=128,
            expert_bits=self.quant_cfg.gpu_expert_bits,
            moe_intermediate_size=self.cfg.moe_intermediate_size,
        )

        # Embedding — not needed on aux (segment_skip_embedding=true), but register
        # a dummy so configure doesn't complain. Use primary embedding ptr (not accessed).
        store.set_embedding(self.embedding.data_ptr())

        if is_last_segment:
            # Final norm — copy to aux GPU (only needed on last segment)
            fn_aux = self.final_norm.to(aux_device).contiguous()
            if not hasattr(self, '_aux_final_norms'):
                self._aux_final_norms = []
            self._aux_final_norms.append(fn_aux)
            store.set_final_norm(fn_aux.data_ptr(), self.cfg.hidden_size)

            # LM head — copy to aux GPU (needed for final segment only)
            # Dequantize on CPU to avoid OOM on GPU0 (LM head is ~1.2 GB in BF16)
            lm_head_w = self.lm_head_data
            if isinstance(lm_head_w, tuple):
                w_int8, scale = lm_head_w
                lm_head_bf16 = (w_int8.cpu().float() * scale.cpu().unsqueeze(1)).to(torch.bfloat16).contiguous().to(aux_device)
            else:
                lm_head_bf16 = lm_head_w.to(aux_device).contiguous()
            if not hasattr(self, '_aux_lm_heads'):
                self._aux_lm_heads = []
            self._aux_lm_heads.append(lm_head_bf16)
            lm_head_wid = store.register_weight(lm_head_bf16.data_ptr(), lm_head_bf16.shape[0], lm_head_bf16.shape[1], 0)
            store.set_lm_head(lm_head_wid)
        else:
            # Intermediate segment — register primary GPU's pointers as placeholders (never accessed)
            store.set_final_norm(self.final_norm.data_ptr(), self.cfg.hidden_size)
            lm_head_w = self.lm_head_data
            if isinstance(lm_head_w, tuple):
                lm_head_wid = store.register_weight(lm_head_w[0].data_ptr(), lm_head_w[0].shape[0], lm_head_w[0].shape[1], 0)
            else:
                lm_head_wid = store.register_weight(lm_head_w.data_ptr(), lm_head_w.shape[0], lm_head_w.shape[1], 0)
            store.set_lm_head(lm_head_wid)

        store.set_norm_bias_one(getattr(self.cfg, 'norm_bias_one', False))

        # ── AWQ support for aux stores ──
        # Aux GPUs never do prefill, so they only need simple INT4 (no Marlin, no swap).
        attn_quant = self.quant_cfg.attention  # "bf16" or "awq"
        marlin_gs = 128
        _awq_template = None
        if attn_quant == "awq":
            from krasis.awq_calibrate import load_template
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "templates", "attention")
            _awq_template = load_template(template_dir, self.cfg.model_path)
            if _awq_template is None:
                raise RuntimeError(
                    "AWQ attention: template loaded on primary store but missing for aux")
            logger.info("Aux store cuda:%d: AWQ simple INT4 attention (no Marlin swap needed)",
                        gpu_idx)

        def _get_bf16_source(attn, attr_name, layer_idx):
            """Get BF16 source tensor, even if primary store replaced attr with MarlinWeight."""
            # Check streaming attention CPU copies first
            if self._stream_attn_enabled and layer_idx in self._stream_attn_cpu:
                cpu_w = self._stream_attn_cpu[layer_idx]
                if attr_name in cpu_w:
                    return cpu_w[attr_name]
            # Check BF16 stash (populated during primary store setup)
            stash = getattr(self, '_aux_bf16_stash', None)
            if stash is not None and (layer_idx, attr_name) in stash:
                return stash[(layer_idx, attr_name)]
            # Direct attribute or dict lookup (if still a tensor)
            if isinstance(attn, dict):
                w = attn.get(attr_name)
            else:
                w = getattr(attn, attr_name, None)
            if w is not None and isinstance(w, torch.Tensor):
                return w
            raise RuntimeError(
                f"Cannot get BF16 source for {attr_name} on layer {layer_idx} "
                f"(type={type(w).__name__}). No streaming attention or BF16 stash available.")

        def _register_aux_attn(w_bf16, layer_idx, layer_type, tensor_name,
                               awq_scales=None):
            """Register attention weight on aux store.
            AWQ → quantize to simple INT4 on aux GPU (Rust alloc + upload).
            BF16 → copy tensor to aux GPU.
            """
            if attn_quant == "awq" and _awq_template is not None:
                from krasis.awq_calibrate import get_tensor_decision
                effective_quant = get_tensor_decision(
                    _awq_template, layer_idx, layer_type, tensor_name)
            else:
                effective_quant = "bf16"

            if effective_quant in ("int4", "int8") and w_bf16.dtype == torch.bfloat16:
                use_int4 = (effective_quant == "int4")
                if use_int4:
                    n, k = w_bf16.shape
                    if n % 64 == 0 and k % 16 == 0 and k % marlin_gs == 0:
                        w_cpu = w_bf16.cpu().contiguous() if w_bf16.is_cuda else w_bf16.contiguous()
                        if awq_scales is not None:
                            w_cpu = (w_cpu.float() * awq_scales.float().unsqueeze(0)).to(
                                torch.bfloat16).contiguous()
                        store.repack_marlin_int4_cpu(w_cpu.data_ptr(), n, k, marlin_gs)
                        del w_cpu
                        return store.register_simple_int4_only(n, k, marlin_gs)
                    else:
                        logger.warning("Aux attn [%d×%d] not Marlin-compatible, using BF16",
                                       n, k)
                # INT8 or non-compatible: fall through to BF16

            # BF16 path
            w_gpu = w_bf16.to(aux_device, non_blocking=True)
            self._aux_decode_weights.append(w_gpu)
            return store.register_weight(
                w_gpu.data_ptr(), w_gpu.shape[0], w_gpu.shape[1],
                _weight_dtype_code(w_gpu))

        # Keep references to aux GPU weight copies
        if not hasattr(self, '_aux_decode_weights_all'):
            self._aux_decode_weights_all = []
        self._aux_decode_weights = []
        aux_rope_set = False
        gqa_count_before_split = 0

        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attention
            inp_norm = layer.input_norm_weight
            post_norm = layer.post_attn_norm_weight

            # Count GQA layers before split for gqa_cache_offset
            if layer_idx < split_layer and layer.layer_type != "linear_attention" \
                    and not hasattr(attn, 'kv_a_proj'):
                gqa_count_before_split += 1

            # Determine if this layer is in this aux GPU's segment
            in_segment = split_layer <= layer_idx < layer_end

            if layer.layer_type == "linear_attention":
                if in_segment:
                    # Get BF16 source weights (may have been replaced by MarlinWeight on primary)
                    qkvz_src = _get_bf16_source(attn, "in_proj_qkvz", layer_idx)
                    ba_src = _get_bf16_source(attn, "in_proj_ba", layer_idx)
                    out_src = _get_bf16_source(attn, "out_proj", layer_idx)

                    # AWQ scales for input projections
                    _layer_awq_scales = None
                    _qkvz_scales = None
                    _ba_scales = None
                    if attn_quant == "awq" and _awq_template is not None:
                        from krasis.awq_calibrate import (
                            get_layer_scales, is_awq_scaled_tensor)
                        _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)
                        _qkvz_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_qkvz")
                        ) else None
                        _ba_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "in_proj_ba")
                        ) else None

                    qkvz_wid = _register_aux_attn(qkvz_src, layer_idx,
                                                   "linear_attention", "in_proj_qkvz",
                                                   awq_scales=_qkvz_scales)
                    ba_wid = _register_aux_attn(ba_src, layer_idx,
                                                 "linear_attention", "in_proj_ba",
                                                 awq_scales=_ba_scales)
                    out_wid = _register_aux_attn(out_src, layer_idx,
                                                  "linear_attention", "out_proj")

                    # AWQ scales already folded into inp_norm in-place by primary store setup.
                    # Just copy the already-folded norm to aux GPU (no second fold).
                    inp_norm_aux = inp_norm.to(aux_device, non_blocking=True)
                    post_norm_aux = post_norm.to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([inp_norm_aux, post_norm_aux])

                    # Copy LA-specific weights
                    conv_w = attn.conv1d_weight.squeeze(1).contiguous().float().to(aux_device)
                    attn._aux_conv_state = attn._conv_state.squeeze(0).float().contiguous().to(aux_device) if attn._conv_state is not None else torch.zeros(attn.conv_dim, attn.kernel_dim, device=aux_device, dtype=torch.float32)
                    attn._aux_norm_weight = attn.norm_weight.float().contiguous().to(aux_device)
                    attn._aux_recur_state = attn._recurrent_state.squeeze(0).float().contiguous().to(aux_device) if attn._recurrent_state is not None else torch.zeros(attn.num_k_heads, attn.k_head_dim, attn.head_ratio * attn.v_head_dim, device=aux_device, dtype=torch.float32)
                    attn._aux_a_log = attn.A_log.float().contiguous().to(aux_device)
                    attn._aux_dt_bias = attn.dt_bias.float().contiguous().to(aux_device)
                    self._aux_decode_weights.extend([conv_w, attn._aux_conv_state, attn._aux_norm_weight, attn._aux_recur_state, attn._aux_a_log, attn._aux_dt_bias])

                    store.register_la_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm_aux.data_ptr(), input_norm_size=inp_norm_aux.numel(),
                        post_attn_norm_ptr=post_norm_aux.data_ptr(), post_attn_norm_size=post_norm_aux.numel(),
                        in_proj_qkvz_wid=qkvz_wid, in_proj_ba_wid=ba_wid, out_proj_wid=out_wid,
                        conv_weight_ptr=conv_w.data_ptr(),
                        a_log_ptr=attn._aux_a_log.data_ptr(),
                        dt_bias_ptr=attn._aux_dt_bias.data_ptr(),
                        norm_weight_ptr=attn._aux_norm_weight.data_ptr(),
                        conv_state_ptr=attn._aux_conv_state.data_ptr(),
                        recur_state_ptr=attn._aux_recur_state.data_ptr(),
                        nk=attn.num_k_heads, nv=attn.num_v_heads,
                        dk=attn.k_head_dim, dv=attn.v_head_dim,
                        hr=attn.head_ratio,
                        kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                        scale=attn.scale,
                    )
                else:
                    # Placeholder for non-segment LA layers (never executed on this GPU).
                    # Use dummy weights — ptr=0 is safe since decode segment skips these.
                    dummy_wid = store.register_weight(0, 1, 1, 0)
                    store.register_la_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                        in_proj_qkvz_wid=dummy_wid, in_proj_ba_wid=dummy_wid,
                        out_proj_wid=dummy_wid,
                        conv_weight_ptr=0, a_log_ptr=0, dt_bias_ptr=0,
                        norm_weight_ptr=0, conv_state_ptr=0, recur_state_ptr=0,
                        nk=attn.num_k_heads, nv=attn.num_v_heads,
                        dk=attn.k_head_dim, dv=attn.v_head_dim,
                        hr=attn.head_ratio,
                        kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                        scale=attn.scale,
                    )
            elif hasattr(attn, 'kv_a_proj'):
                # MLA attention
                if in_segment:
                    # Get BF16 sources for MLA projections
                    _layer_awq_scales = None
                    _qa_scales = None
                    _qproj_scales = None
                    _kva_scales = None
                    if attn_quant == "awq" and _awq_template is not None:
                        from krasis.awq_calibrate import (
                            get_layer_scales, is_awq_scaled_tensor)
                        _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)
                        _qa_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "q_a_proj")
                        ) else None
                        _qproj_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                        ) else None
                        _kva_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "kv_a_proj")
                        ) else None

                    # Q projection
                    if attn.has_q_lora:
                        qa_src = _get_bf16_source(attn, "q_a_proj", layer_idx)
                        qb_src = _get_bf16_source(attn, "q_b_proj", layer_idx)
                        qa_wid = _register_aux_attn(qa_src, layer_idx, "mla", "q_a_proj",
                                                     awq_scales=_qa_scales)
                        qb_wid = _register_aux_attn(qb_src, layer_idx, "mla", "q_b_proj")
                        attn._aux_q_a_norm = attn.q_a_norm_weight.float().contiguous().to(aux_device)
                        self._aux_decode_weights.append(attn._aux_q_a_norm)
                        q_a_norm_ptr = attn._aux_q_a_norm.data_ptr()
                        q_proj_wid = None
                    else:
                        q_src = _get_bf16_source(attn, "q_proj", layer_idx)
                        q_proj_wid = _register_aux_attn(q_src, layer_idx, "mla", "q_proj",
                                                         awq_scales=_qproj_scales)
                        qa_wid = None
                        qb_wid = None
                        q_a_norm_ptr = 0

                    # KV projection
                    kva_src = _get_bf16_source(attn, "kv_a_proj", layer_idx)
                    kva_wid = _register_aux_attn(kva_src, layer_idx, "mla", "kv_a_proj",
                                                  awq_scales=_kva_scales)

                    # O projection
                    o_src = _get_bf16_source(attn, "o_proj", layer_idx)
                    o_wid = _register_aux_attn(o_src, layer_idx, "mla", "o_proj")

                    # AWQ scales already folded into inp_norm in-place by primary store setup.
                    # Just copy the already-folded norm to aux GPU (no second fold).
                    inp_norm_aux = inp_norm.to(aux_device, non_blocking=True)
                    post_norm_aux = post_norm.to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([inp_norm_aux, post_norm_aux])

                    # kv_a layernorm, w_kc, w_vc on aux GPU
                    attn._aux_kv_a_norm = attn.kv_a_norm_weight.float().contiguous().to(aux_device)
                    attn._aux_w_kc = attn.w_kc.contiguous().to(aux_device)
                    attn._aux_w_vc = attn.w_vc.contiguous().to(aux_device)
                    self._aux_decode_weights.extend([
                        attn._aux_kv_a_norm, attn._aux_w_kc, attn._aux_w_vc])

                    # MLA KV cache on aux GPU
                    cache = self.kv_caches[0]
                    if not hasattr(self, '_aux_mla_cache_offset'):
                        self._aux_mla_cache_offset = 0
                    mla_offset = self._aux_mla_cache_offset
                    self._aux_mla_cache_offset += 1
                    ckv_src = cache.ckv_cache[mla_offset]
                    kpe_src = cache.kpe_cache[mla_offset]
                    ckv_aux = torch.empty_like(ckv_src, device=aux_device)
                    kpe_aux = torch.empty_like(kpe_src, device=aux_device)
                    if not hasattr(self, '_aux_mla_caches'):
                        self._aux_mla_caches = []
                    self._aux_mla_caches.extend([ckv_aux, kpe_aux])
                    max_seq = cache.max_pages * cache.page_size

                    store.register_mla_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm_aux.data_ptr(),
                        input_norm_size=inp_norm_aux.numel(),
                        post_attn_norm_ptr=post_norm_aux.data_ptr(),
                        post_attn_norm_size=post_norm_aux.numel(),
                        kv_a_proj_wid=kva_wid, o_proj_wid=o_wid,
                        kv_a_norm_ptr=attn._aux_kv_a_norm.data_ptr(),
                        w_kc_ptr=attn._aux_w_kc.data_ptr(),
                        w_vc_ptr=attn._aux_w_vc.data_ptr(),
                        num_heads=attn.num_heads,
                        kv_lora_rank=attn.kv_lora_rank,
                        qk_nope_dim=attn.qk_nope_dim,
                        qk_rope_dim=attn.qk_rope_dim,
                        v_head_dim=attn.v_head_dim,
                        sm_scale=attn.sm_scale,
                        rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                        ckv_cache_ptr=ckv_aux.data_ptr(),
                        kpe_cache_ptr=kpe_aux.data_ptr(),
                        q_a_proj_wid=qa_wid,
                        q_b_proj_wid=qb_wid,
                        q_a_norm_ptr=q_a_norm_ptr,
                        q_proj_wid=q_proj_wid,
                        q_lora_rank=attn.q_lora_rank if attn.has_q_lora else 0,
                        ckv_cache_dim=attn.ckv_dim,
                    )

                    # RoPE from first MLA layer in segment
                    if not aux_rope_set:
                        cos, sin = attn._get_rope_cos_sin(max_seq)
                        cos_f32 = cos.float().contiguous().to(aux_device)
                        sin_f32 = sin.float().contiguous().to(aux_device)
                        self._aux_rope_cos = cos_f32
                        self._aux_rope_sin = sin_f32
                        store.set_rope_tables(
                            cos_f32.data_ptr(), sin_f32.data_ptr(),
                            cos_f32.shape[1], max_seq,
                        )
                        aux_rope_set = True
                else:
                    # Placeholder MLA (never executed on this GPU)
                    dummy_wid = store.register_weight(0, 1, 1, 0)
                    cache = self.kv_caches[0]
                    if not hasattr(self, '_aux_mla_cache_offset'):
                        self._aux_mla_cache_offset = 0
                    mla_offset = self._aux_mla_cache_offset
                    self._aux_mla_cache_offset += 1
                    max_seq = cache.max_pages * cache.page_size
                    store.register_mla_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(),
                        input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(),
                        post_attn_norm_size=post_norm.numel(),
                        kv_a_proj_wid=dummy_wid, o_proj_wid=dummy_wid,
                        kv_a_norm_ptr=0, w_kc_ptr=0, w_vc_ptr=0,
                        num_heads=attn.num_heads,
                        kv_lora_rank=attn.kv_lora_rank,
                        qk_nope_dim=attn.qk_nope_dim,
                        qk_rope_dim=attn.qk_rope_dim,
                        v_head_dim=attn.v_head_dim,
                        sm_scale=attn.sm_scale,
                        rope_interleave=getattr(self.cfg, 'rope_interleave', True),
                        ckv_cache_ptr=cache.ckv_cache[mla_offset].data_ptr(),
                        kpe_cache_ptr=cache.kpe_cache[mla_offset].data_ptr(),
                        q_a_proj_wid=None, q_b_proj_wid=None, q_a_norm_ptr=0,
                        q_proj_wid=None, q_lora_rank=0,
                        ckv_cache_dim=attn.ckv_dim,
                    )
            else:
                # GQA attention — attn object is None for GQA models (weights in layer.gqa_weights)
                _gqa_head_dim = self.cfg.gqa_head_dim or self.cfg.head_dim
                _gqa_sm_scale = 1.0 / (_gqa_head_dim ** 0.5)
                _gqa_gated = hasattr(self.cfg, 'gated_attention') and self.cfg.gated_attention
                gqa_w = layer.gqa_weights if hasattr(layer, 'gqa_weights') else None

                if in_segment:
                    # Get BF16 source weights from gqa_weights dict or stash
                    q_src = _get_bf16_source(gqa_w, "q_proj", layer_idx)
                    k_src = _get_bf16_source(gqa_w, "k_proj", layer_idx)
                    v_src = _get_bf16_source(gqa_w, "v_proj", layer_idx)
                    o_src = _get_bf16_source(gqa_w, "o_proj", layer_idx)

                    # AWQ scales
                    _layer_awq_scales = None
                    _q_scales = None
                    _k_scales = None
                    _v_scales = None
                    if attn_quant == "awq" and _awq_template is not None:
                        from krasis.awq_calibrate import (
                            get_layer_scales, is_awq_scaled_tensor)
                        _layer_awq_scales = get_layer_scales(_awq_template, layer_idx)
                        _q_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "q_proj")
                        ) else None
                        _k_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "k_proj")
                        ) else None
                        _v_scales = _layer_awq_scales if (
                            _layer_awq_scales is not None
                            and is_awq_scaled_tensor(_awq_template, layer_idx, "v_proj")
                        ) else None

                    q_wid = _register_aux_attn(q_src, layer_idx, "gqa", "q_proj",
                                               awq_scales=_q_scales)
                    k_wid = _register_aux_attn(k_src, layer_idx, "gqa", "k_proj",
                                               awq_scales=_k_scales)
                    v_wid = _register_aux_attn(v_src, layer_idx, "gqa", "v_proj",
                                               awq_scales=_v_scales)
                    o_wid = _register_aux_attn(o_src, layer_idx, "gqa", "o_proj")

                    # AWQ scales already folded into inp_norm in-place by primary store setup.
                    # Just copy the already-folded norm to aux GPU (no second fold).
                    inp_norm_aux = inp_norm.to(aux_device, non_blocking=True)
                    post_norm_aux = post_norm.to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([inp_norm_aux, post_norm_aux])

                    # QK norms from gqa_weights dict
                    q_norm_src = gqa_w.get("q_norm") if gqa_w else None
                    k_norm_src = gqa_w.get("k_norm") if gqa_w else None
                    if q_norm_src is not None:
                        q_norm_aux = q_norm_src.float().contiguous().to(aux_device)
                        self._aux_decode_weights.append(q_norm_aux)
                        q_norm_ptr = q_norm_aux.data_ptr()
                    else:
                        q_norm_ptr = 0
                    if k_norm_src is not None:
                        k_norm_aux = k_norm_src.float().contiguous().to(aux_device)
                        self._aux_decode_weights.append(k_norm_aux)
                        k_norm_ptr = k_norm_aux.data_ptr()
                    else:
                        k_norm_ptr = 0

                    store.register_gqa_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm_aux.data_ptr(), input_norm_size=inp_norm_aux.numel(),
                        post_attn_norm_ptr=post_norm_aux.data_ptr(), post_attn_norm_size=post_norm_aux.numel(),
                        q_proj_wid=q_wid, k_proj_wid=k_wid,
                        v_proj_wid=v_wid, o_proj_wid=o_wid,
                        fused_qkv_wid=None,
                        num_heads=self.cfg.num_attention_heads,
                        num_kv_heads=self.cfg.num_key_value_heads,
                        head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                        q_norm_ptr=q_norm_ptr, k_norm_ptr=k_norm_ptr,
                        gated=_gqa_gated,
                    )

                    # RoPE tables on aux GPU (from first GQA layer in segment)
                    if not aux_rope_set:
                        max_seq = max(
                            c.max_context_tokens for c in self.kv_caches if c is not None
                        )
                        rope_half = self.cfg.rotary_dim // 2
                        inv_freq = 1.0 / (self.cfg.rope_theta ** (
                            torch.arange(0, rope_half * 2, 2, dtype=torch.float32) / (rope_half * 2)))
                        t = torch.arange(max_seq, dtype=torch.float32)
                        freqs = torch.outer(t, inv_freq)
                        cos_f32 = torch.cos(freqs).contiguous().to(aux_device)
                        sin_f32 = torch.sin(freqs).contiguous().to(aux_device)
                        self._aux_rope_cos = cos_f32
                        self._aux_rope_sin = sin_f32
                        store.set_rope_tables(
                            cos_f32.data_ptr(), sin_f32.data_ptr(),
                            cos_f32.shape[1], max_seq,
                        )
                        aux_rope_set = True
                else:
                    # Placeholder GQA (never executed on this GPU)
                    dummy_wid = store.register_weight(0, 1, 1, 0)
                    store.register_gqa_layer(
                        layer_idx=layer_idx,
                        input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                        post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                        q_proj_wid=dummy_wid, k_proj_wid=dummy_wid,
                        v_proj_wid=dummy_wid, o_proj_wid=dummy_wid,
                        fused_qkv_wid=None,
                        num_heads=self.cfg.num_attention_heads,
                        num_kv_heads=self.cfg.num_key_value_heads,
                        head_dim=_gqa_head_dim, sm_scale=_gqa_sm_scale,
                        q_norm_ptr=0, k_norm_ptr=0,
                        gated=_gqa_gated,
                    )

            # Register MLP type
            if layer.is_moe:
                store.register_mlp(layer_idx, "moe")
            elif layer.dense_mlp is not None:
                if in_segment:
                    gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"]).to(aux_device, non_blocking=True)
                    up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"]).to(aux_device, non_blocking=True)
                    dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"]).to(aux_device, non_blocking=True)
                    self._aux_decode_weights.extend([gp, up, dp])
                    gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                    up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                    dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                else:
                    gp = self._dense_mlp_tensor(layer.dense_mlp["gate_proj"])
                    up = self._dense_mlp_tensor(layer.dense_mlp["up_proj"])
                    dp = self._dense_mlp_tensor(layer.dense_mlp["down_proj"])
                    self._rust_decode_weights.extend([gp, up, dp])
                    gp_wid = store.register_weight(gp.data_ptr(), gp.shape[0], gp.shape[1], 0)
                    up_wid = store.register_weight(up.data_ptr(), up.shape[0], up.shape[1], 0)
                    dp_wid = store.register_weight(dp.data_ptr(), dp.shape[0], dp.shape[1], 0)
                store.register_mlp(layer_idx, "dense",
                                   gate_proj_wid=gp_wid, up_proj_wid=up_wid, down_proj_wid=dp_wid)
            else:
                store.register_mlp(layer_idx, "none")

        # Register MoE expert data from engine (same engine, same CPU RAM data)
        if self.krasis_engine is not None:
            store.setup_from_engine(self.krasis_engine)

        # Shared expert gates for aux layers
        if not hasattr(self, '_aux_shared_gate_refs_all'):
            self._aux_shared_gate_refs_all = []
        self._aux_shared_gate_refs = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_moe and layer.shared_expert_gate is not None:
                if split_layer <= layer_idx < layer_end:
                    sg = layer.shared_expert_gate.to(aux_device, non_blocking=True)
                    self._aux_shared_gate_refs.append(sg)
                    sg_wid = store.register_weight(sg.data_ptr(), sg.shape[0], sg.shape[1], 0)
                else:
                    sg = layer.shared_expert_gate
                    sg_wid = store.register_weight(sg.data_ptr(), sg.shape[0], sg.shape[1], 0)
                store.set_moe_shared_gate_wid(layer_idx, sg_wid)

        # Allocate KV cache on aux GPU for GQA layers in this segment [split_layer..layer_end)
        cache = self.kv_caches[0]
        if cache is not None and cache.kv_format == 2 and cache.k_radius_cache is not None:
            # Polar4 KV cache: 4 tensors per layer (radius BF16 + angles uint8 for K and V)
            polar4_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        kr_src = cache.k_radius_cache[gqa_cache_idx]
                        vr_src = cache.v_radius_cache[gqa_cache_idx]
                        ka_src = cache.k_angles_cache[gqa_cache_idx]
                        va_src = cache.v_angles_cache[gqa_cache_idx]
                        kr_aux = torch.empty_like(kr_src, device=aux_device)
                        vr_aux = torch.empty_like(vr_src, device=aux_device)
                        ka_aux = torch.empty_like(ka_src, device=aux_device)
                        va_aux = torch.empty_like(va_src, device=aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([kr_aux, vr_aux, ka_aux, va_aux])
                        polar4_ptrs.append((layer_idx,
                                            kr_aux.data_ptr(), vr_aux.data_ptr(),
                                            ka_aux.data_ptr(), va_aux.data_ptr()))
                    else:
                        kr = cache.k_radius_cache[gqa_cache_idx]
                        vr = cache.v_radius_cache[gqa_cache_idx]
                        ka = cache.k_angles_cache[gqa_cache_idx]
                        va = cache.v_angles_cache[gqa_cache_idx]
                        polar4_ptrs.append((layer_idx,
                                            kr.data_ptr(), vr.data_ptr(),
                                            ka.data_ptr(), va.data_ptr()))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            num_blocks = (cache.num_kv_heads * cache.gqa_head_dim) // 16
            store.set_kv_cache_ptrs_polar4(polar4_ptrs, max_seq, num_blocks)
            logger.info("Aux Polar4 KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d, blocks=%d",
                        gpu_idx, len(polar4_ptrs), split_layer, layer_end, max_seq, num_blocks)
        elif cache is not None and cache.k_cache is not None:
            # FP8/BF16 KV cache: 2 tensors per layer (K and V)
            kv_ptrs = []
            gqa_cache_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                if (layer.layer_type not in ("linear_attention", "mamba2", "moe")
                    and not hasattr(layer.attention, 'kv_a_proj')):
                    if split_layer <= layer_idx < layer_end:
                        k_src = cache.k_cache[gqa_cache_idx]
                        v_src = cache.v_cache[gqa_cache_idx]
                        k_aux = torch.empty_like(k_src, device=aux_device)
                        v_aux = torch.empty_like(v_src, device=aux_device)
                        if not hasattr(self, '_aux_kv_caches'):
                            self._aux_kv_caches = []
                        self._aux_kv_caches.extend([k_aux, v_aux])
                        kv_ptrs.append((layer_idx, k_aux.data_ptr(), v_aux.data_ptr()))
                    else:
                        k_layer = cache.k_cache[gqa_cache_idx]
                        v_layer = cache.v_cache[gqa_cache_idx]
                        kv_ptrs.append((layer_idx, k_layer.data_ptr(), v_layer.data_ptr()))
                    gqa_cache_idx += 1
            max_seq = cache.max_pages * cache.page_size
            store.set_kv_cache_ptrs(kv_ptrs, max_seq)
            logger.info("Aux KV cache on cuda:%d: %d GQA layers for [%d..%d), max_seq=%d",
                        gpu_idx, len(kv_ptrs), split_layer, layer_end, max_seq)

        torch.cuda.synchronize(aux_device)

        # Store references for GC protection and multi-GPU bookkeeping
        if not hasattr(self, '_aux_gpu_decode_stores'):
            self._aux_gpu_decode_stores = []
        self._aux_gpu_decode_stores.append(store)
        self._aux_decode_weights_all.extend(self._aux_decode_weights)
        self._aux_shared_gate_refs_all.extend(self._aux_shared_gate_refs)

        # Legacy single-aux-store attributes (still set for backward compat)
        self._aux_gpu_decode_store = store
        self._aux_gpu_idx = gpu_idx
        self._multi_gpu_split_layer = split_layer
        self._multi_gpu_gqa_offset = gqa_count_before_split

        if os.environ.get("KRASIS_DECODE_TIMING", "") == "1":
            store.set_timing(True)

        logger.info("Aux GPU decode store configured: gpu=%d, layers=[%d..%d), gqa_offset=%d, store_addr=%d",
                     gpu_idx, split_layer, layer_end, gqa_count_before_split, store.gpu_store_addr())
        return store

    def _export_kv_to_rust(self, seq_states, prompt_len: int):
        """Set KV cache position for Rust decode after prefill.

        The KV cache is shared — Rust prefill already wrote FP8 data into the
        same GPU buffers that Rust decode reads. No data copy needed, just
        tell Rust how many tokens are valid.
        """
        store = self._gpu_decode_store

        # Guard: don't overflow shared KV buffer
        rust_max_seq = getattr(store, 'kv_max_seq', 0)
        if rust_max_seq > 0 and prompt_len > rust_max_seq:
            logger.warning(
                "Prompt length %d exceeds KV cache max_seq (%d), "
                "skipping decode for this request", prompt_len, rust_max_seq)
            return

        store.set_kv_position(prompt_len)

    def _transfer_mamba2_states(self):
        """Copy Mamba2 conv_state and ssm_state from Python prefill into Rust decode buffers.

        After prefill, each Mamba2 layer has saved its final conv_state and ssm_state
        on the layer object. These must be copied into the pre-allocated GPU buffers
        that the Rust decode engine uses. The buffers have fixed addresses (registered
        during _register_mamba2_layer), so we copy in-place to preserve pointers.
        """
        decode_states = getattr(self, '_mamba2_decode_states', None)
        if not decode_states:
            return

        for layer_idx, buffers in decode_states.items():
            layer = self.layers[layer_idx]

            # Conv state: prefill produces [1, conv_dim, conv_kernel], Rust expects [conv_dim, conv_kernel] FP32
            if hasattr(layer, '_mamba2_conv_state') and layer._mamba2_conv_state is not None:
                conv_src = layer._mamba2_conv_state.squeeze(0).float().contiguous()
                buffers['conv_state'].copy_(conv_src)
                layer._mamba2_conv_state = None  # Free prefill tensor

            # SSM state: prefill produces [1, num_heads, head_dim, state_size], Rust expects [num_heads, head_dim, state_size] FP32
            if hasattr(layer, '_mamba2_ssm_state') and layer._mamba2_ssm_state is not None:
                ssm_src = layer._mamba2_ssm_state.squeeze(0).float().contiguous()
                buffers['ssm_state'].copy_(ssm_src)
                layer._mamba2_ssm_state = None  # Free prefill tensor

    def _update_la_state_ptrs(self):
        """Re-register LA state pointers after prefill (states may have been reallocated).
        Prefill may reassign _conv_state and _recurrent_state tensors, so we
        need to create fresh FP32 copies and re-register the new pointers."""
        store = self._gpu_decode_store
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type == "linear_attention":
                attn = layer.attention
                if attn._conv_state is None or attn._recurrent_state is None:
                    continue
                wids = self._la_wids.get(layer_idx)
                if wids is None:
                    continue
                inp_norm = layer.input_norm_weight
                post_norm = layer.post_attn_norm_weight
                conv_w = attn._rust_conv_weight

                # Prefill may have updated conv_state and recurrent_state (BF16) -- convert to FP32 for Rust.
                # IMPORTANT: copy INTO existing buffers to preserve fixed GPU addresses for CUDA graph replay.
                # Only allocate on first call; subsequent calls copy data in-place.
                new_conv = attn._conv_state.squeeze(0).float().contiguous()
                new_recur = attn._recurrent_state.squeeze(0).float().contiguous()
                if hasattr(attn, '_rust_conv_state') and attn._rust_conv_state is not None \
                        and attn._rust_conv_state.shape == new_conv.shape:
                    attn._rust_conv_state.copy_(new_conv)
                    attn._rust_recur_state.copy_(new_recur)
                else:
                    attn._rust_conv_state = new_conv
                    attn._rust_recur_state = new_recur

                store.register_la_layer(
                    layer_idx=layer_idx,
                    input_norm_ptr=inp_norm.data_ptr(), input_norm_size=inp_norm.numel(),
                    post_attn_norm_ptr=post_norm.data_ptr(), post_attn_norm_size=post_norm.numel(),
                    in_proj_qkvz_wid=wids[0], in_proj_ba_wid=wids[1], out_proj_wid=wids[2],
                    conv_weight_ptr=conv_w.data_ptr(),
                    a_log_ptr=attn._rust_a_log.data_ptr(),
                    dt_bias_ptr=attn._rust_dt_bias.data_ptr(),
                    norm_weight_ptr=attn._rust_norm_weight.data_ptr(),
                    conv_state_ptr=attn._rust_conv_state.data_ptr(),
                    recur_state_ptr=attn._rust_recur_state.data_ptr(),
                    nk=attn.num_k_heads, nv=attn.num_v_heads,
                    dk=attn.k_head_dim, dv=attn.v_head_dim,
                    hr=attn.head_ratio,
                    kernel_dim=attn.kernel_dim, conv_dim=attn.conv_dim,
                    scale=attn.scale,
                )

    def _update_la_state_ptrs_aux(self):
        """Re-register LA state pointers on all aux stores after prefill.
        Copies post-prefill conv/recur states from GPU0 to each aux GPU."""
        aux_stores = getattr(self, '_aux_gpu_decode_stores', None)
        if not aux_stores:
            # Fallback to legacy single-store attribute
            single = getattr(self, '_aux_gpu_decode_store', None)
            if single is None:
                return
            aux_stores = [single]

        import torch

        # For each aux store, update LA state pointers for layers in its segment.
        # We need to know each store's layer range. We can infer this from the store's
        # graph config (segment_layer_start/end were set during setup). For simplicity,
        # iterate all LA layers >= first split and update any aux store that has them.
        split_layer = getattr(self, '_multi_gpu_split_layer', 0)

        for aux_store in aux_stores:
            aux_gpu_idx = aux_store.gpu_index() if hasattr(aux_store, 'gpu_index') else 0
            aux_dev = torch.device(f"cuda:{aux_gpu_idx}")

            for layer_idx, layer in enumerate(self.layers):
                if layer_idx < split_layer:
                    continue
                if layer.layer_type != "linear_attention":
                    continue
                attn = layer.attention
                if attn._conv_state is None or attn._recurrent_state is None:
                    continue
                # Copy updated states to aux GPU -- preserve fixed addresses for CUDA graph replay.
                # Use per-store attribute names to avoid collisions between aux stores.
                attr_conv = f'_aux_conv_state_{aux_gpu_idx}'
                attr_recur = f'_aux_recur_state_{aux_gpu_idx}'
                new_conv_aux = attn._conv_state.squeeze(0).float().contiguous().to(aux_dev)
                new_recur_aux = attn._recurrent_state.squeeze(0).float().contiguous().to(aux_dev)
                existing_conv = getattr(attn, attr_conv, None)
                if existing_conv is not None and existing_conv.shape == new_conv_aux.shape:
                    existing_conv.copy_(new_conv_aux)
                    getattr(attn, attr_recur).copy_(new_recur_aux)
                else:
                    setattr(attn, attr_conv, new_conv_aux)
                    setattr(attn, attr_recur, new_recur_aux)
                # Also update the legacy attributes for backward compat
                if not hasattr(attn, '_aux_conv_state') or attn._aux_conv_state is None:
                    attn._aux_conv_state = getattr(attn, attr_conv)
                    attn._aux_recur_state = getattr(attn, attr_recur)
                aux_store.update_la_state_ptrs(
                    layer_idx,
                    getattr(attn, attr_conv).data_ptr(),
                    getattr(attn, attr_recur).data_ptr(),
                )

    def restrict_to_decode_segment(self, decode_start: int, decode_end: int):
        """Multi-GPU: restrict GPU 0's single-slot AWQ swaps to its decode segment only.

        Called after set_decode_segment(). Removes swap entries for layers outside
        [decode_start, decode_end) so swap_to_simple_int4/swap_to_marlin only touch
        decode-segment weights. Stashes prefill-only layer info for free/upload cycle.
        """
        gpu_store = getattr(self, '_gpu_decode_store', None)
        if gpu_store is None:
            return

        # Tell Rust to remove swap entries for non-decode layers
        removed = gpu_store.restrict_swaps_to_decode_segment()

        # Track which MarlinWeight attributes are "prefill-only" (outside decode segment)
        # These can be freed after prefill and re-uploaded before next prefill.
        from krasis.attention import MarlinWeight
        self._prefill_only_attn = []  # [(layer_idx, attn, attr_name, MarlinWeight)]
        self._prefill_only_freed = False

        for layer_idx, layer in enumerate(self.layers):
            if decode_start <= layer_idx < decode_end:
                continue  # In decode segment — managed by single-slot swap
            attn = layer.attention
            if layer.layer_type == "linear_attention":
                for attr_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                    mw = getattr(attn, attr_name, None)
                    if isinstance(mw, MarlinWeight):
                        self._prefill_only_attn.append((layer_idx, attn, attr_name, mw))
            elif hasattr(attn, 'kv_a_proj'):
                for attr_name in ("q_a_proj", "q_b_proj", "q_proj", "kv_a_proj", "o_proj"):
                    mw = getattr(attn, attr_name, None)
                    if isinstance(mw, MarlinWeight):
                        self._prefill_only_attn.append((layer_idx, attn, attr_name, mw))
            else:
                for attr_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    mw = getattr(attn, attr_name, None)
                    if isinstance(mw, MarlinWeight):
                        self._prefill_only_attn.append((layer_idx, attn, attr_name, mw))

        n_tensors = len(self._prefill_only_attn)
        if n_tensors > 0:
            # Estimate VRAM that will be freed after prefill
            total_bytes = 0
            for _, _, _, mw in self._prefill_only_attn:
                total_bytes += mw.packed.nelement() * mw.packed.element_size()
                total_bytes += mw.scales.nelement() * mw.scales.element_size()
            logger.info("Prefill-only attention: %d tensors for layers outside [%d, %d), "
                        "~%.0f MB freeable after prefill",
                        n_tensors, decode_start, decode_end, total_bytes / 1024 / 1024)

    def _free_prefill_only_attention(self):
        """Free prefill-only attention tensors from GPU after prefill.

        Moves packed/scales to CPU and clears GPU references so PyTorch can
        reclaim VRAM. The CPU copies are kept for re-upload before next prefill.
        """
        entries = getattr(self, '_prefill_only_attn', None)
        if not entries or getattr(self, '_prefill_only_freed', False):
            return

        import gc
        from krasis.attention import MarlinWeight
        freed_bytes = 0
        new_entries = []

        for layer_idx, attn, attr_name, mw in entries:
            packed = mw.packed
            scales = mw.scales
            freed_bytes += packed.nelement() * packed.element_size()
            freed_bytes += scales.nelement() * scales.element_size()
            # Move to CPU (creates CPU copy, releases GPU tensor on next GC)
            packed_cpu = packed.cpu()
            scales_cpu = scales.cpu()
            cpu_mw = MarlinWeight(packed_cpu, scales_cpu, mw.workspace, mw.scalar_type, mw.n, mw.k)
            # Clear GPU reference on attention module
            setattr(attn, attr_name, None)
            # Also remove from _marlin_attn_weights if tracked there
            marlin_weights = getattr(self, '_marlin_attn_weights', {})
            to_remove = [wid for wid, info in marlin_weights.items()
                         if info[0] is packed]
            for wid in to_remove:
                del marlin_weights[wid]
            # Release scales_slot from _single_slot_scales if it backs this scales view
            slot_scales = getattr(self, '_single_slot_scales', [])
            scales_base = scales.storage().data_ptr() if scales.is_cuda else 0
            self._single_slot_scales = [
                s for s in slot_scales
                if s.data_ptr() != scales_base
            ]
            # Keep CPU copy for re-upload
            new_entries.append((layer_idx, attn, attr_name, cpu_mw))

        self._prefill_only_attn = new_entries
        gc.collect()
        torch.cuda.empty_cache()
        self._prefill_only_freed = True
        logger.info("Freed prefill-only attention: ~%.0f MB", freed_bytes / 1024 / 1024)

    def _upload_prefill_only_attention(self):
        """Re-upload prefill-only attention tensors to GPU before prefill.

        Restores MarlinWeight references on attention modules. Called before prefill
        in multi-GPU mode when the tensors were previously freed.
        """
        entries = getattr(self, '_prefill_only_attn', None)
        if not entries or not getattr(self, '_prefill_only_freed', False):
            return  # Nothing freed, nothing to upload

        device = torch.device(self.ranks[0].device)
        uploaded_bytes = 0
        new_entries = []

        from krasis.attention import MarlinWeight
        for layer_idx, attn, attr_name, mw in entries:
            # Re-upload to GPU
            if mw.packed.is_cuda:
                # Already on GPU (shouldn't happen if freed, but be safe)
                new_entries.append((layer_idx, attn, attr_name, mw))
                continue
            packed_gpu = mw.packed.to(device, non_blocking=True)
            scales_gpu = mw.scales.to(device, non_blocking=True)
            uploaded_bytes += packed_gpu.nelement() * packed_gpu.element_size()
            uploaded_bytes += scales_gpu.nelement() * scales_gpu.element_size()

            new_mw = MarlinWeight(packed_gpu, scales_gpu, mw.workspace, mw.scalar_type, mw.n, mw.k)
            setattr(attn, attr_name, new_mw)
            new_entries.append((layer_idx, attn, attr_name, new_mw))

        self._prefill_only_attn = new_entries
        self._prefill_only_freed = False
        torch.cuda.synchronize(device)
        logger.info("Uploaded prefill-only attention: ~%.0f MB", uploaded_bytes / 1024 / 1024)

    def server_cleanup(self):
        """Free server request state (KV cache pages, etc.).

        Also handles single-slot AWQ: restore Marlin data into GPU slots
        for instant prefill on the next request.
        """
        # Single-slot AWQ: restore Marlin into slots
        gpu_store = getattr(self, '_gpu_decode_store', None)
        if gpu_store is not None:
            gpu_store.swap_to_marlin()

        states = getattr(self, '_server_seq_states', None)
        if states:
            for s in states:
                if s is not None:
                    s.free()
            self._server_seq_states = None
        self._rust_kv_refs = None

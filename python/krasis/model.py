"""Full model: embedding → N transformer layers → LM head.

Single process, N GPUs (PP=N). Each GPU owns a slice of layers
plus its local KV cache. Hidden states transfer at PP boundaries.

Loading sequence:
  Phase 0: System RAM budget check (refuse to run if insufficient)
  Phase 1: GPU weights (streaming BF16→INT8, one tensor at a time)
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

from krasis.config import ModelConfig, PPRankConfig, QuantConfig, build_pp_ranks, compute_pp_partition
from krasis.weight_loader import WeightLoader, int8_linear
from krasis.layer import TransformerLayer
from krasis.kv_cache import PagedKVCache, SequenceKVState
from krasis.sampler import sample
from krasis.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

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
    num_moe_layers = cfg.num_hidden_layers - cfg.first_k_dense_replace

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
    overhead_gb = 20.0  # system overhead estimate
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

    # Separate dense and MoE layers within this rank
    dense_layers = [l for l in all_layers if l < first_k]
    moe_layers = [l for l in all_layers if l >= first_k]

    if not moe_layers:
        # All dense layers — single group, no experts
        return [(all_layers, [])]

    # Active-only or single group: all layers in one group
    if divisor <= 1:
        moe_indices = [l - first_k for l in moe_layers]
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
        group_moe_indices = [l - first_k for l in group_moe]

        if g == 0:
            # Dense layers go in the first group
            group_all = dense_layers + group_moe
        else:
            group_all = group_moe

        groups.append((group_all, group_moe_indices))

    return groups


def _hcs_prefill_divisor(manager, rank, cfg) -> int:
    """Auto-compute layer group divisor for HCS prefill.

    HCS stores hot experts statically on GPU(s) — prefill runs hot experts on
    GPU in parallel with cold experts on CPU, zero DMA. The divisor controls
    how layers are grouped for the forward loop (preload_layer_group() is a
    no-op under HCS since experts are already loaded).
    """
    first_k = cfg.first_k_dense_replace
    moe_layers = [l for l in range(rank.layer_start, rank.layer_end) if l >= first_k]
    if not moe_layers:
        return 1

    num_moe = len(moe_layers)

    # Always use 1-layer groups for HCS prefill.
    # Smaller expert groups = more VRAM for activation intermediates = larger token
    # chunks = more tokens per expert = dramatically better GPU utilization.
    # The extra DMA cost (2x groups) is small (~5% of total) vs kernel speedup.
    logger.info(
        "HCS prefill: %d MoE layers, 1-layer groups (maximize chunk size)",
        num_moe,
    )
    return num_moe


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
    """Full model with pipeline-parallel GPU inference + CPU MoE experts."""

    def __init__(
        self,
        model_path: str,
        pp_partition: Optional[List[int]] = None,
        num_gpus: Optional[int] = None,
        devices: Optional[List[str]] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        krasis_threads: int = 48,
        gpu_prefill: bool = True,
        gpu_prefill_threshold: int = 300,
        attention_backend: str = "flashinfer",  # "flashinfer" or "trtllm"
        quant_cfg: QuantConfig = None,
        force_load: bool = False,
        expert_divisor: int = 1,
        gguf_path: Optional[str] = None,
        gguf_native: bool = False,
        kv_cache_mb: int = 2000,
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
        self.expert_divisor = expert_divisor

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

        # Per-device weight copies for multi-GPU calibration/prefill.
        # Populated by _load_gpu_weights when all_devices has >1 GPU.
        # Keys are device strings ("cuda:0", "cuda:1", ...).
        # Values: {"layers": [...], "embedding": T, "final_norm": T,
        #          "lm_head_data": T, "ranks": [...], "kv_caches": [...]}
        self._device_state: dict = {}
        self._active_device: Optional[str] = None  # current device for use_device()

        # For hybrid models: maps absolute layer idx → KV cache layer offset
        # within its rank's cache. Linear attention layers get -1 (no KV).
        self._kv_layer_offsets: dict = {}  # layer_idx -> offset or -1

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

    def load(self):
        """Load all weights: GPU (streaming INT8) + CPU (Krasis INT4 experts)."""
        start = time.perf_counter()

        # Phase 0: System RAM budget check
        if self.cfg.n_routed_experts > 0:
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

        # Phase 2: CPU expert weights
        cpu_start = time.perf_counter()
        logger.info("Phase 2: Loading CPU expert weights (Krasis INT4)...")
        self._load_cpu_experts()
        cpu_elapsed = time.perf_counter() - cpu_start
        logger.info("CPU experts loaded in %.1fs", cpu_elapsed)

        # Post-load RSS check: verify RAM estimate accuracy
        if self._estimated_expert_ram_gb > 0:
            _check_actual_rss(self._estimated_expert_ram_gb)

        # Phase 3: GPU prefill managers (one per device)
        if self.gpu_prefill_enabled and self.cfg.n_routed_experts > 0:
            self._init_gpu_prefill()

        # Allocate KV caches
        self._init_kv_caches()

        # Phase 4: CPU Hub (for multi-GPU prefill coordination)
        if len(self.all_devices) > 1 and self.krasis_engine is not None:
            self.cpu_hub = CPUHubManager(
                num_gpus=len(self.all_devices),
                hidden_size=self.cfg.hidden_size,
                engine=self.krasis_engine,
            )
            logger.info("CPU Hub initialized for %d GPUs", len(self.all_devices))
        else:
            self.cpu_hub = None

        # Load tokenizer
        self.tokenizer = Tokenizer(self.cfg.model_path)

        self._loaded = True
        total = time.perf_counter() - start
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
                         w = getattr(layer.attention, key, None) or \
                             (layer.shared_expert.get(key) if layer.shared_expert else None)
                         if w is not None:
                            wt = w[0] if isinstance(w, (tuple, list)) else w
                            k, n = wt.shape if len(wt.shape) == 2 else (wt.shape[1], wt.shape[0])
                            if k * n > max_k * max_n: max_k, max_n = k, n

                if max_k > 0:
                    for m_size in (1, 16, 512):
                        x = torch.randint(-127, 127, (m_size, max_k), dtype=torch.int8, device=device)
                        w = torch.randint(-127, 127, (max_n, max_k), dtype=torch.int8, device=device)
                        _ = torch._int_mm(x, w.t())
            except Exception as e:
                logger.warning("CUDA warmup (cuBLAS) on %s failed: %s", device, e)

            # ── 2. Triton/Marlin JIT Compilation ──
            try:
                K, N, gs, bits = self.cfg.hidden_size, self.cfg.moe_intermediate_size, 128, 4
                nb2 = bits // 2
                n_dummy = 2

                if is_primary:
                    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
                    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
                    ws = marlin_make_workspace(device, max_blocks_per_sm=4)
                    w13 = torch.zeros(n_dummy, K // 16, 2 * N * nb2, dtype=torch.int32, device=device)
                    w13s = torch.zeros(n_dummy, K // gs, 2 * N, dtype=torch.bfloat16, device=device)
                    x = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
                    g = torch.empty(1, n_dummy, device=device)
                    tw = torch.zeros(1, 1, dtype=torch.float32, device=device)
                    ti = torch.zeros(1, 1, dtype=torch.int32, device=device)
                    fused_marlin_moe(x, w13, w13, w13s, w13s, g, tw, ti, n_dummy, None,
                                     torch.empty(n_dummy,0,dtype=torch.int32,device=device),
                                     torch.empty(n_dummy,0,dtype=torch.int32,device=device),
                                     torch.empty(n_dummy,0,dtype=torch.int32,device=device),
                                     torch.empty(n_dummy,0,dtype=torch.int32,device=device),
                                     ws, bits, True)
                else:
                    # Warm up the Triton kernel with dummy standard-GPTQ data
                    w13_marlin = torch.zeros(n_dummy, K // 16, 2 * N * nb2, dtype=torch.int32)
                    w13s_marlin = torch.zeros(n_dummy, K // gs, 2 * N, dtype=torch.bfloat16)
                    w13_std = inverse_marlin_repack(w13_marlin, K, 2*N, bits).to(device)
                    w13s_std = inverse_scale_permute(w13s_marlin, K, 2*N, gs).to(device)
                    lookup = torch.tensor([0, 1], dtype=torch.int32, device=device)
                    x = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
                    triton_moe_decode(x, [0,1], [0.5,0.5], w13_std, w13s_std, w13_std, w13s_std, lookup, None, None, self.cfg.swiglu_limit, 1.702, gs)

            except Exception as e:
                logger.warning("CUDA warmup (Triton/Marlin) on %s failed: %s", device, e)
            
            torch.cuda.synchronize(device)

        # ── 3. Linear attention torch.compile warmup ──
        if self.cfg.linear_num_value_heads > 0:
            try:
                from krasis.linear_attention import warmup_compiled_chunk_step
                primary = devices[0]
                warmup_compiled_chunk_step(
                    primary,
                    nv=self.cfg.linear_num_value_heads,
                    dk=self.cfg.linear_key_head_dim,
                    dv=self.cfg.linear_value_head_dim,
                )
            except Exception as e:
                logger.warning("Linear attention torch.compile warmup failed: %s", e)

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
        if layer.layer_type == "linear_attention":
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
        else:
            # GQA
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
                # Reconstruct original gate_proj/up_proj from fused gate_up_proj
                se = {}
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
                se["down_proj"] = layer.shared_expert["down_proj"]
                if layer.shared_expert_gate is not None:
                    se["shared_expert_gate"] = layer.shared_expert_gate
                result["shared_expert"] = se
        elif layer.dense_mlp is not None:
            result["dense_mlp"] = dict(layer.dense_mlp)

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
        """Stream-load all GPU weights to primary device, then copy to all devices."""
        self.layers = []
        primary_dev = self.all_devices[0]
        torch.cuda.set_device(primary_dev)

        # ── Load FULL model to primary device (from safetensors) ──
        logger.info("Loading full base model to primary device %s...", primary_dev)
        self.embedding = loader.load_embedding(primary_dev)

        for layer_idx in range(self.cfg.num_hidden_layers):
            weights = loader.load_layer(layer_idx, primary_dev)
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

        # Store primary device state
        self._active_device = str(primary_dev)
        self._device_state[str(primary_dev)] = {
            "layers": self.layers,
            "embedding": self.embedding,
            "final_norm": self.final_norm,
            "lm_head_data": self.lm_head_data,
        }

        # ── Copy to each additional device ──
        for extra_dev in self.all_devices:
            if extra_dev == primary_dev:
                continue
            t0 = time.perf_counter()
            logger.info("Copying full base model weights to %s...", extra_dev)

            dev_embedding = self.embedding.to(extra_dev)
            dev_final_norm = self.final_norm.to(extra_dev)
            dev_lm_head = self._move_weight(self.lm_head_data, extra_dev)

            dev_layers = []
            for layer in self.layers:
                w = self._extract_layer_weights(layer, primary_dev)
                w_copy = self._copy_weights_dict(w, extra_dev)
                new_layer = TransformerLayer(
                    self.cfg, layer.layer_idx, w_copy, extra_dev,
                    krasis_engine=None,
                    gpu_prefill_manager=None,
                    gpu_prefill_threshold=self.gpu_prefill_threshold,
                    attention_backend=self.attention_backend,
                )
                dev_layers.append(new_layer)

            self._device_state[str(extra_dev)] = {
                "layers": dev_layers,
                "embedding": dev_embedding,
                "final_norm": dev_final_norm,
                "lm_head_data": dev_lm_head,
            }

            alloc_mb = torch.cuda.memory_allocated(extra_dev) / (1024**2)
            elapsed = time.perf_counter() - t0
            logger.info(
                "Full base model copied to %s in %.1fs (%.0f MB allocated)",
                extra_dev, elapsed, alloc_mb,
            )

        # ── Split attention weights on extra devices for tensor parallelism ──
        # Currently disabled: split-attention showed no speedup (473 vs 476 tok/s)
        # due to Python dispatch overhead. Set _split_attention_active = True to re-enable.
        self._split_attention_active = False
        if self._split_attention_active and len(self.all_devices) > 1:
            for extra_dev in self.all_devices[1:]:
                self._split_attention_weights(extra_dev, half=1)

    @staticmethod
    def _split_weight_rows(w, start_row: int, end_row: int):
        """Slice rows from a weight (tensor or INT8 (tensor, scale) tuple)."""
        if isinstance(w, tuple):
            # INT8: (weight_int8, scale) — both indexed by output rows
            return (w[0][start_row:end_row].contiguous(),
                    w[1][start_row:end_row].contiguous())
        return w[start_row:end_row].contiguous()

    def _split_attention_weights(self, device: torch.device, half: int = 1):
        """Split attention weights on `device` for tensor-parallel split-head attention.

        After this, the device's attention objects hold only their half of heads.
        GPU 0 keeps the first half (rows [0:mid]), extra GPUs get the second half
        (rows [mid:end]).

        For GQA: q_proj is row-split; k_proj/v_proj replicated; o_proj deleted.
        For linear attention: in_proj_qkvz/in_proj_ba group-split; conv1d/A_log/dt_bias
        channel-split; out_proj deleted.
        """
        dev_str = str(device)
        dev_layers = self._device_state[dev_str]["layers"]
        freed_bytes = 0

        for layer in dev_layers:
            attn = layer.attention

            if layer.layer_type == "linear_attention":
                # ── Linear attention: split by key-head groups ──
                nk = attn.num_k_heads        # 16
                nv = attn.num_v_heads         # 32
                dk = attn.k_head_dim          # 128
                dv = attn.v_head_dim          # 128
                ratio = attn.head_ratio       # 2

                # group_dim per key-head: q(dk) + k(dk) + v(ratio*dv) + z(ratio*dv)
                group_dim = 2 * dk + 2 * dv * ratio  # 768
                half_nk = nk // 2  # 8

                # in_proj_qkvz: output is [nk * group_dim, hidden]
                # GPU 1 gets groups [half_nk:nk]
                qkvz_start = half_nk * group_dim
                qkvz_end = nk * group_dim
                attn.in_proj_qkvz = self._split_weight_rows(
                    attn.in_proj_qkvz, qkvz_start, qkvz_end)

                # in_proj_ba: output is [nk * 2*ratio, hidden]
                ba_group_dim = 2 * ratio  # 4
                ba_start = half_nk * ba_group_dim
                ba_end = nk * ba_group_dim
                attn.in_proj_ba = self._split_weight_rows(
                    attn.in_proj_ba, ba_start, ba_end)

                # conv1d_weight: [conv_dim, 1, kernel_dim]
                # conv_dim = key_dim*2 + value_dim = nk*dk*2 + nv*dv
                # After split: half key_dim*2 + half value_dim
                half_key_dim = half_nk * dk     # 1024
                half_value_dim = (nv // 2) * dv  # 2048
                # Channel layout: [q_flat(key_dim), k_flat(key_dim), v_flat(value_dim)]
                # We want the second half of each section
                key_dim = nk * dk               # 2048
                value_dim = nv * dv             # 4096
                old_conv = attn.conv1d_weight
                q_channels = old_conv[half_key_dim:key_dim]          # q second half
                k_channels = old_conv[key_dim + half_key_dim:key_dim * 2]  # k second half
                v_channels = old_conv[key_dim * 2 + half_value_dim:]  # v second half
                attn.conv1d_weight = torch.cat([q_channels, k_channels, v_channels], dim=0).contiguous()
                del old_conv

                # A_log, dt_bias: [num_value_heads] → second half
                half_nv = nv // 2
                attn.A_log = attn.A_log[half_nv:].contiguous()
                attn.dt_bias = attn.dt_bias[half_nv:].contiguous()

                # norm_weight: [value_head_dim] — shared across heads, replicate (keep as-is)

                # Update head counts
                attn.num_k_heads = half_nk
                attn.num_v_heads = half_nv
                attn.key_dim = half_nk * dk
                attn.value_dim = half_nv * dv
                attn.conv_dim = attn.key_dim * 2 + attn.value_dim

                # Delete out_proj (only needed on GPU 0)
                if hasattr(attn, 'out_proj'):
                    old_out = attn.out_proj
                    if isinstance(old_out, tuple):
                        freed_bytes += sum(t.nbytes for t in old_out)
                    elif isinstance(old_out, torch.Tensor):
                        freed_bytes += old_out.nbytes
                    attn.out_proj = None

                # Reset state (will be re-initialized with new dimensions)
                attn._conv_state = None
                attn._recurrent_state = None
                attn._la_graph = None
                attn._la_input = None
                attn._la_output = None
                attn._split_half = 1

            elif hasattr(attn, 'q_proj'):
                # ── GQA attention: row-split q_proj ──
                num_heads = attn.num_heads
                head_dim = attn.head_dim
                half_heads = num_heads // 2

                if attn.gated_attention:
                    # q_proj output: [num_heads * head_dim * 2, hidden]
                    # Layout: interleaved [head0_q, head0_gate, head1_q, head1_gate, ...]
                    # Per-head: head_dim * 2 (q + gate)
                    per_head_dim = head_dim * 2
                    q_start = half_heads * per_head_dim
                    q_end = num_heads * per_head_dim
                else:
                    # q_proj output: [num_heads * head_dim, hidden]
                    q_start = half_heads * head_dim
                    q_end = num_heads * head_dim

                attn.q_proj = self._split_weight_rows(attn.q_proj, q_start, q_end)

                # k_proj, v_proj: keep full (replicated, small: 4 KV heads)
                # q_norm, k_norm: keep full (shared per-head-dim, tiny)

                # Update num_heads for split attention
                attn.num_heads = half_heads
                attn.num_groups = half_heads // attn.num_kv_heads

                # q_proj_bias: split if present
                if attn.q_proj_bias is not None:
                    if attn.gated_attention:
                        attn.q_proj_bias = attn.q_proj_bias[q_start:q_end].contiguous()
                    else:
                        attn.q_proj_bias = attn.q_proj_bias[q_start:q_end].contiguous()

                # sinks: split if present (per-head)
                if attn.sinks is not None:
                    attn.sinks = attn.sinks[half_heads:].contiguous()

                # Delete o_proj (only needed on GPU 0)
                old_o = attn.o_proj
                if isinstance(old_o, tuple):
                    freed_bytes += sum(t.nbytes for t in old_o)
                elif isinstance(old_o, torch.Tensor):
                    freed_bytes += old_o.nbytes
                attn.o_proj = None

                # o_proj_bias: also only on GPU 0
                if attn.o_proj_bias is not None:
                    freed_bytes += attn.o_proj_bias.nbytes
                    attn.o_proj_bias = None
                attn._split_half = 1

        # Mark GPU 0's attention objects with split_half=0 and create prefill
        # weight views.  GPU 0 keeps FULL weights (for decode CUDA graphs) but
        # forward_split uses *views* of the first-half rows so it only computes
        # half the Q/V heads – same FLOPS as GPU 1.
        primary_layers = self._device_state[str(self.all_devices[0])]["layers"]
        for layer in primary_layers:
            attn = layer.attention
            attn._split_half = 0

            if layer.layer_type == "linear_attention":
                nk = attn.num_k_heads
                nv = attn.num_v_heads
                dk = attn.k_head_dim
                dv = attn.v_head_dim
                ratio = attn.head_ratio
                half_nk = nk // 2
                half_nv = nv // 2

                # in_proj_qkvz: first half of key-head groups
                group_dim = 2 * dk + 2 * dv * ratio
                attn._prefill_qkvz = self._split_weight_rows(
                    attn.in_proj_qkvz, 0, half_nk * group_dim)

                # in_proj_ba: first half
                ba_group_dim = 2 * ratio
                attn._prefill_ba = self._split_weight_rows(
                    attn.in_proj_ba, 0, half_nk * ba_group_dim)

                # conv1d_weight: first half of each section
                key_dim = nk * dk
                value_dim = nv * dv
                half_key_dim = half_nk * dk
                half_value_dim = half_nv * dv
                old_conv = attn.conv1d_weight
                q_ch = old_conv[:half_key_dim]
                k_ch = old_conv[key_dim:key_dim + half_key_dim]
                v_ch = old_conv[key_dim * 2:key_dim * 2 + half_value_dim]
                attn._prefill_conv1d = torch.cat(
                    [q_ch, k_ch, v_ch], dim=0).contiguous()

                # A_log, dt_bias: first half of value heads
                attn._prefill_A_log = attn.A_log[:half_nv].contiguous()
                attn._prefill_dt_bias = attn.dt_bias[:half_nv].contiguous()

                # Dimensions for the prefill path
                attn._prefill_nk = half_nk
                attn._prefill_nv = half_nv
                attn._prefill_key_dim = half_nk * dk
                attn._prefill_value_dim = half_nv * dv
                attn._prefill_conv_dim = half_nk * dk * 2 + half_nv * dv

            elif hasattr(attn, 'q_proj'):
                # GQA: first-half q_proj view
                num_heads = attn.num_heads
                head_dim = attn.head_dim
                half_heads = num_heads // 2
                if attn.gated_attention:
                    q_end = half_heads * head_dim * 2
                else:
                    q_end = half_heads * head_dim
                attn._prefill_q_proj = self._split_weight_rows(
                    attn.q_proj, 0, q_end)
                attn._prefill_num_heads = half_heads
                # sinks: first half
                if attn.sinks is not None:
                    attn._prefill_sinks = attn.sinks[:half_heads].contiguous()
                # q_proj_bias: first half
                if attn.q_proj_bias is not None:
                    attn._prefill_q_proj_bias = attn.q_proj_bias[:q_end].contiguous()

        torch.cuda.empty_cache()
        freed_mb = freed_bytes / (1024**2)
        alloc_after = torch.cuda.memory_allocated(device) / (1024**2)
        logger.info(
            "Split attention weights on %s: freed ~%.0f MB (o_proj/out_proj), "
            "%.0f MB allocated after split",
            device, freed_mb, alloc_after,
        )

    def _load_cpu_experts(self):
        """Load expert weights into Krasis Rust engine."""
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
            engine.load(self.cfg.model_path, cpu_num_bits=cpu_bits, gpu_num_bits=gpu_bits)

        self.krasis_engine = engine

        # Wire engine to MoE layers on ALL devices + allocate pinned buffers
        first_k = self.cfg.first_k_dense_replace
        for dev_str, state in self._device_state.items():
            for layer in state["layers"]:
                if layer.is_moe:
                    layer.krasis_engine = engine
                    if layer._cpu_act_buf is None:
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
        num_moe_layers = self.cfg.num_hidden_layers - self.cfg.first_k_dense_replace
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
        """Create GpuPrefillManager per device and wire to MoE layers on ALL devices."""
        from krasis.gpu_prefill import GpuPrefillManager

        engine_ref = getattr(self, 'krasis_engine', None)
        num_moe_layers = self.cfg.num_hidden_layers - self.cfg.first_k_dense_replace
        num_ranks = len(self.all_devices)

        for i, dev in enumerate(self.all_devices):
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
                    expert_divisor=self.expert_divisor,
                    skip_shared_experts=self._has_shared_expert_gate,
                    swiglu_limit=self.cfg.swiglu_limit,
                    rank=i,
                    num_ranks=num_ranks,
                )
                self.gpu_prefill_managers[dev_str] = manager
                logger.info("GPU prefill manager created for %s (rank %d/%d)", dev_str, i, num_ranks)

        # Pre-prepare all MoE layers from disk cache at startup
        for dev_str, manager in self.gpu_prefill_managers.items():
            manager.prepare_all_layers()

        # Wire manager to MoE layers on ALL devices
        for state in self._device_state.values():
            for layer in state["layers"]:
                if layer.is_moe:
                    dev_str = str(layer.device)
                    layer.gpu_prefill_manager = self.gpu_prefill_managers.get(dev_str)

        logger.info(
            "GPU prefill: %d managers, threshold=%d tokens",
            len(self.gpu_prefill_managers), self.gpu_prefill_threshold,
        )

    def _start_ram_watchdog(self, floor_pct: float = 5.0):
        """Start daemon thread that monitors system RAM and exits if too low.

        Checks /proc/meminfo every second. If MemAvailable drops below
        floor_pct% of MemTotal, logs an error and calls os._exit() to
        prevent a full system OOM that kills desktop processes.

        Args:
            floor_pct: Minimum % free RAM before forced exit (default 5%)
        """
        def _watchdog():
            while True:
                time.sleep(1.0)
                meminfo = _read_meminfo()
                if not meminfo:
                    continue
                total_kb = meminfo.get("MemTotal", 0)
                avail_kb = meminfo.get("MemAvailable", 0)
                if total_kb == 0:
                    continue
                pct_free = 100.0 * avail_kb / total_kb
                if pct_free < floor_pct:
                    logger.error(
                        "RAM WATCHDOG: %.1f%% free (%.1f GB available / %.1f GB total) "
                        "— below %.1f%% floor. Exiting to prevent system OOM!",
                        pct_free, avail_kb / 1024 / 1024,
                        total_kb / 1024 / 1024, floor_pct,
                    )
                    os._exit(137)

        t = threading.Thread(target=_watchdog, daemon=True, name="ram-watchdog")
        t.start()
        logger.info("RAM watchdog started: will exit if < %.1f%% free", floor_pct)

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

        Device-bound caches (RoPE, FlashInfer wrappers, CUDA graphs)
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
            attn_saved = {"device": attn.device}

            if layer.layer_type == "linear_attention":
                # GatedDeltaNetAttention
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

            elif hasattr(attn, "kv_a_proj"):
                # MLAAttention
                mla_weights = ["kv_a_proj", "o_proj", "kv_a_norm_weight", "w_kc", "w_vc"]
                if attn.has_q_lora:
                    mla_weights += ["q_a_proj", "q_b_proj", "q_a_norm_weight"]
                else:
                    mla_weights += ["q_proj"]
                for name in mla_weights:
                    attn_saved[name] = getattr(attn, name)
                    setattr(attn, name, mv(getattr(attn, name), device))
                # Invalidate device-bound caches
                attn_saved["_rope_cos_sin"] = attn._rope_cos_sin
                attn_saved["_attn_wrapper"] = attn._attn_wrapper
                attn._rope_cos_sin = None
                import flashinfer
                workspace_buf = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=device
                )
                attn._attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                    workspace_buf,
                )

            else:
                # GQAAttention
                gqa_weights = ["q_proj", "k_proj", "v_proj", "o_proj"]
                gqa_optional = ["q_proj_bias", "k_proj_bias", "v_proj_bias",
                                "o_proj_bias", "sinks", "q_norm", "k_norm"]
                for name in gqa_weights:
                    attn_saved[name] = getattr(attn, name)
                    setattr(attn, name, mv(getattr(attn, name), device))
                for name in gqa_optional:
                    val = getattr(attn, name, None)
                    if val is not None:
                        attn_saved[name] = val
                        setattr(attn, name, mv(val, device))
                # Invalidate device-bound caches
                attn_saved["_rope_cos_sin"] = attn._rope_cos_sin
                attn_saved["_prefill_wrapper"] = attn._prefill_wrapper
                attn._rope_cos_sin = None
                import flashinfer
                workspace_buf = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=device
                )
                attn._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace_buf,
                )

            attn.device = device
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
            attn.device = attn_saved["device"]

            if layer.layer_type == "linear_attention":
                for name in ("in_proj_qkvz", "in_proj_ba", "out_proj",
                             "conv1d_weight", "A_log", "dt_bias", "norm_weight"):
                    setattr(attn, name, attn_saved[name])
                attn._conv_state = attn_saved["_conv_state"]
                attn._recurrent_state = attn_saved["_recurrent_state"]
                attn._la_graph = attn_saved["_la_graph"]
                attn._la_input = attn_saved["_la_input"]
                attn._la_output = attn_saved["_la_output"]
                attn._la_stream = attn_saved["_la_stream"]

            elif hasattr(attn, "kv_a_proj"):
                # MLA
                mla_weights = ["kv_a_proj", "o_proj", "kv_a_norm_weight", "w_kc", "w_vc"]
                if attn.has_q_lora:
                    mla_weights += ["q_a_proj", "q_b_proj", "q_a_norm_weight"]
                else:
                    mla_weights += ["q_proj"]
                for name in mla_weights:
                    setattr(attn, name, attn_saved[name])
                attn._rope_cos_sin = attn_saved["_rope_cos_sin"]
                attn._attn_wrapper = attn_saved["_attn_wrapper"]

            else:
                # GQA
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(attn, name, attn_saved[name])
                for name in ("q_proj_bias", "k_proj_bias", "v_proj_bias",
                             "o_proj_bias", "sinks", "q_norm", "k_norm"):
                    if name in attn_saved:
                        setattr(attn, name, attn_saved[name])
                attn._rope_cos_sin = attn_saved["_rope_cos_sin"]
                attn._prefill_wrapper = attn_saved["_prefill_wrapper"]

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

        # Save and null out per-layer gpu_prefill_manager
        saved_layer_managers = []
        for layer in self.layers:
            saved_layer_managers.append(layer.gpu_prefill_manager)
            layer.gpu_prefill_manager = None

        # Disable GPU prefill — forces CPU MoE path
        self.gpu_prefill_enabled = False

        # ── Replicate weights to target device ──
        saved_weights = self._replicate_to_device(device)

        # ── Temporary single-rank config ──
        orig_rank = saved_ranks[0]
        self.ranks = [PPRankConfig(
            rank=0,
            device=str(device),
            layer_start=orig_rank.layer_start,
            layer_end=orig_rank.layer_end,
            num_layers=orig_rank.num_layers,
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

            for layer, mgr in zip(self.layers, saved_layer_managers):
                layer.gpu_prefill_manager = mgr

            self._restore_from_device(saved_weights)

        return {"inference_cost_bytes": inference_cost}

    def _init_kv_caches(self):
        """Allocate paged KV caches per GPU.

        For hybrid models, only full attention layers need KV cache.
        Builds _kv_layer_offsets mapping absolute layer idx → offset within
        the rank's KV cache (-1 for linear attention layers).
        """
        self.kv_caches = []
        self._kv_layer_offsets = {}
        use_combined = (self.attention_backend == "trtllm")

        for rank in self.ranks:
            dev = torch.device(rank.device)

            # Count full attention layers in this rank
            kv_offset = 0
            for layer_idx in rank.layer_range:
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
                    max_mb=self.kv_cache_mb,
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

        # ── Allocate secondary KV caches on extra GPUs for split-attention EP ──
        self._ep_kv_caches = {}
        if getattr(self, '_split_attention_active', False) and self.cfg.is_gqa:
            # Count full attention layers (same as primary)
            num_kv_layers = sum(1 for v in self._kv_layer_offsets.values() if v >= 0)
            if num_kv_layers > 0:
                # Smaller cache for EP GPUs: they only need prefill capacity
                # Use 120 MB (enough for ~10K tokens with FP8 GQA)
                ep_kv_mb = min(120, self.kv_cache_mb)
                for extra_dev in self.all_devices[1:]:
                    ep_cache = PagedKVCache(
                        self.cfg,
                        num_layers=num_kv_layers,
                        device=extra_dev,
                        kv_dtype=self.kv_dtype,
                        combined=use_combined,
                        max_mb=ep_kv_mb,
                    )
                    self._ep_kv_caches[str(extra_dev)] = ep_cache
                    logger.info(
                        "EP KV cache on %s: %d layers, %d MB",
                        extra_dev, num_kv_layers, ep_kv_mb,
                    )

    def _get_rank_for_layer(self, global_layer_idx: int) -> int:
        """Get the PP rank index that owns a given layer."""
        offset = 0
        for i, rank in enumerate(self.ranks):
            if global_layer_idx < offset + rank.num_layers:
                return i
            offset += rank.num_layers
        raise ValueError(f"Layer {global_layer_idx} out of range")

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            token_ids: [M] int64 token IDs
            positions: [M] int32 position indices
            seq_states: One SequenceKVState per sequence in the batch

        Returns:
            logits: [M, vocab_size] float32
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        timing = TIMING.decode and M == 1

        if timing:
            t_fwd_start = time.perf_counter()

        # ── Layer-grouped / active-only / LRU prefill routing ──
        # Only use grouped path for actual prefill (M > 1). M=1 decode uses
        # normal forward() — per-layer GPU dispatch in _moe_forward handles it.
        if (
            M > 1
            and (self.expert_divisor >= 2 or self.expert_divisor < 0)
            and M >= self.gpu_prefill_threshold
            and self.gpu_prefill_enabled
            and self.cfg.n_routed_experts > 0
        ):
            return self._forward_prefill_with_oom_retry(token_ids, positions, seq_states)

        # ── Embedding (GPU0) ──
        first_dev = torch.device(self.ranks[0].device)
        hidden = self.embedding[token_ids.to(first_dev)]  # [M, hidden_size]

        # Diagnostics: track first N forward calls (gated behind KRASIS_DIAG=1)
        diag = False
        if TIMING.diag:
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            self._diag_count += 1
            diag = self._diag_count <= 12  # prefill + first 11 decode steps
            if diag:
                h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                logger.info("DIAG[%d] embed: mean=%.4f std=%.4f max=%.4f nan=%d",
                            self._diag_count, h.float().mean(), h.float().std(),
                            h.float().abs().max(), h.isnan().sum().item())

        # ── Transformer layers ──
        residual = None
        layer_global_idx = 0
        first_k = self.cfg.first_k_dense_replace

        for rank_idx, rank in enumerate(self.ranks):
            dev = torch.device(rank.device)
            kv_cache = self.kv_caches[rank_idx]
            seq_state = seq_states[rank_idx]

            # Ensure KV cache capacity for new tokens (once per rank, not per layer)
            if seq_state is not None:
                seq_state.ensure_capacity(M)

            # Transfer hidden state to this rank's device (CPU bounce if P2P broken)
            hidden = _to_device(hidden, dev)
            if residual is not None:
                residual = _to_device(residual, dev)

            # Transfer positions once per rank
            rank_positions = _to_device(positions, dev)

            for local_offset in range(rank.num_layers):
                abs_layer_idx = rank.layer_start + local_offset
                layer = self.layers[layer_global_idx]

                # MoE layer index for Krasis engine
                moe_layer_idx = None
                if layer.is_moe:
                    moe_layer_idx = abs_layer_idx - first_k

                # KV layer offset: -1 for linear attention layers
                kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, local_offset)

                if kv_layer_offset < 0:
                    # Linear attention layer: no KV cache
                    hidden, residual = layer.forward(
                        hidden, residual, rank_positions,
                        None, None, -1,
                        moe_layer_idx, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward(
                        hidden, residual, rank_positions,
                        kv_cache, seq_state, kv_layer_offset,
                        moe_layer_idx, num_new_tokens=M,
                    )

                if TIMING.diag and diag and abs_layer_idx in (0, 1, self.cfg.num_hidden_layers // 2, self.cfg.num_hidden_layers - 1):
                    h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                    r = residual[-1] if residual.shape[0] > 1 else residual[0]
                    logger.info("DIAG[%d] L%d: hid std=%.4f max=%.4f | res std=%.4f max=%.4f",
                                self._diag_count, abs_layer_idx,
                                h.float().std(), h.float().abs().max(),
                                r.float().std(), r.float().abs().max())

                layer_global_idx += 1

            # Advance KV cache seq_len (once per rank, after all layers processed)
            if seq_state is not None:
                seq_state.advance(M)

        if timing:
            torch.cuda.synchronize()
            t_after_layers = time.perf_counter()

        # ── Final norm ──
        last_dev = torch.device(self.ranks[-1].device)
        hidden = _to_device(hidden, last_dev)
        if residual is not None:
            residual = _to_device(residual, last_dev)

        # Final fused_add_rmsnorm (in-place: residual += hidden, hidden = rmsnorm(residual))
        import flashinfer
        flashinfer.norm.fused_add_rmsnorm(
            hidden, residual, self.final_norm, self.cfg.rms_norm_eps
        )

        # ── LM head ──
        # Only compute logits for last token during prefill (saves ~4 GB VRAM at 10K tokens)
        if M > 1:
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
            logger.info(
                "DECODE-TOKEN: total=%.1fms (layers=%.1fms post=%.1fms, %d layers)",
                (t_fwd_end - t_fwd_start) * 1000,
                (t_after_layers - t_fwd_start) * 1000,
                (t_fwd_end - t_after_layers) * 1000,
                self.cfg.num_hidden_layers,
            )

        return logits

    def _forward_prefill_with_oom_retry(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_states: List[SequenceKVState],
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

        for attempt in range(max_retries + 1):
            try:
                result = self.forward_prefill_layer_grouped(
                    token_ids, positions, seq_states,
                    max_chunk_override=max_chunk_override,
                )

                # Re-capture CUDA graphs if we freed them during OOM recovery.
                # Prefill is done so the activation VRAM is free again.
                if freed_graphs:
                    for manager in self.gpu_prefill_managers.values():
                        if manager._hcs_initialized and manager._hcs_buffers:
                            try:
                                torch.cuda.empty_cache()
                                manager._init_cuda_graphs()
                                logger.info("CUDA graphs re-captured after OOM recovery")
                            except torch.OutOfMemoryError:
                                logger.warning(
                                    "Could not re-capture CUDA graphs after OOM recovery "
                                    "(decode will use non-graphed path)"
                                )

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
                        if manager._hcs_cuda_graphs:
                            logger.warning(
                                "Freeing %d CUDA graphs to reclaim VRAM for prefill",
                                len(manager._hcs_cuda_graphs),
                            )
                            manager._hcs_cuda_graphs.clear()
                            manager._hcs_graph_io.clear()
                            manager._hcs_cuda_graphs_enabled = False
                            freed_graphs = True
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
    ) -> torch.Tensor:
        """GPU prefill with layer-grouped expert loading.

        Loop structure (group-outer, chunk-inner):
            for each group → DMA experts once → for each chunk → for each layer → compute

        For multi-GPU (num_gpus > 1), uses Expert Parallelism: each GPU runs
        its local expert slice on dedicated CUDA streams, GPU-direct summation.
        For single GPU, uses standard layer.forward() path.
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        num_gpus = len(self.all_devices)
        first_k = self.cfg.first_k_dense_replace
        use_ep = num_gpus > 1 and self.cpu_hub is not None

        # ── Embedding (GPU0) ──
        first_dev = self.all_devices[0]
        all_hidden = self.embedding[token_ids.to(first_dev)]

        # ── Pre-allocate KV pages for all tokens ──
        for seq_state in seq_states:
            if seq_state is not None:
                seq_state.ensure_capacity(M)

        # Pre-allocate EP KV cache pages on extra GPUs for split-attention
        _ep_seq_states = {}
        if getattr(self, '_split_attention_active', False) and use_ep:
            for dev_str, ep_cache in self._ep_kv_caches.items():
                ep_ss = SequenceKVState(ep_cache, seq_id=0)
                ep_ss.ensure_capacity(M)
                _ep_seq_states[dev_str] = ep_ss

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
        first_mgr = next(iter(self.gpu_prefill_managers.values()), None)
        if first_mgr and first_mgr._prefill_mode == "hot_cached_static":
            effective_divisor = _hcs_prefill_divisor(first_mgr, self.ranks[0], self.cfg)
        else:
            effective_divisor = max(1, self.expert_divisor)

        rank = self.ranks[0]
        dev = torch.device(rank.device)
        kv_cache = self.kv_caches[0]
        seq_state = seq_states[0]

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

        # Per-GPU CUDA streams for EP compute (GPU 0 uses default stream)
        ep_streams = {}
        if use_ep:
            for mgr in all_managers[1:]:
                ep_streams[mgr.device] = torch.cuda.Stream(device=mgr.device)

        # ── Lightweight split-attention timing ──
        # KRASIS_SPLIT_TIMING=1: total-only (preserves parallelism)
        # KRASIS_SPLIT_TIMING=2: sub-phase detail (adds sync, serializes GPUs)
        _split_timing_val = os.environ.get("KRASIS_SPLIT_TIMING", "0") if use_ep else "0"
        _split_timing = _split_timing_val in ("1", "2")
        _split_timing_detail = _split_timing_val == "2"
        if _split_timing:
            _st = {"attn": 0.0, "copy_h": 0.0, "gpu0_fwd": 0.0,
                   "gpu1_wait": 0.0, "copy_back": 0.0,
                   "gather_oproj": 0.0, "expert_total": 0.0, "layer_total": 0.0}
            _st_count = 0

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
        # Disabled during ep_timing (detailed sync-heavy timing) and active_only mode.
        _pipeline_enabled = bool(_dma_managers) and not is_active_only and not ep_timing
        _has_prefetch = False
        _prefetch_futures = []
        if _pipeline_enabled:
            logger.info("DMA pipelining ENABLED (%d managers, %d groups)",
                        len(_dma_managers), len(groups))

        for group_idx, (group_layers, group_moe_indices) in enumerate(groups):
            need_load = group_moe_indices and not is_active_only

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

            # Reset seq_state for this group
            if seq_state is not None:
                seq_state.seq_len = 0
            for ep_ss in _ep_seq_states.values():
                ep_ss.seq_len = 0

            if _layer_timing and need_load:
                _lt_compute0 = time.perf_counter()

            # 3. Process all chunks through this group's layers
            for c in range(num_chunks):
                h = chunk_hidden[c].to(dev)
                r = chunk_residual[c]
                if r is not None:
                    r = r.to(dev)
                pos = chunk_positions[c].to(dev)
                chunk_M = h.shape[0]

                for abs_layer_idx in group_layers:
                    layer = self._device_state[str(dev)]["layers"][abs_layer_idx]
                    moe_layer_idx = abs_layer_idx - first_k if abs_layer_idx >= first_k else None
                    kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, 0)

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
                        continue

                    # ── Multi-GPU EP path for MoE layers ──
                    # Split-attention: both GPUs run half-head attention in parallel,
                    # then both GPUs run their expert slices in parallel.
                    use_split_attn = getattr(self, '_split_attention_active', False) and len(self.all_devices) >= 2
                    gpu1_dev = self.all_devices[1] if len(self.all_devices) >= 2 else None
                    gpu1_layer = self._device_state[str(gpu1_dev)]["layers"][abs_layer_idx] if gpu1_dev is not None else None
                    attn_stream = ep_streams.get(gpu1_dev) if gpu1_dev is not None else None

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_layer0 = time.perf_counter()

                    if _split_timing:
                        torch.cuda.synchronize(dev)
                        _st_t0 = time.perf_counter()

                    if use_split_attn:
                        # ── Split-attention path ──

                        # (a) Pre-attention norm on GPU 0
                        h, r = layer.pre_attn_norm(h, r)

                        # (b) Copy h_normed to GPU 1 (overlapped on CUDA stream)
                        if attn_stream is not None:
                            attn_stream.wait_stream(torch.cuda.current_stream(dev))
                            with torch.cuda.stream(attn_stream):
                                h_gpu1 = _to_device(h, gpu1_dev)
                                # Also copy positions if needed for GQA
                                if kv_layer_offset >= 0:
                                    pos_gpu1 = _to_device(pos, gpu1_dev)

                        if _split_timing_detail:
                            _st_t_copy = time.perf_counter()

                        # (c) GPU 0: half-head attention on default stream
                        if layer.layer_type == "linear_attention":
                            partial_0 = layer.attention.forward_split(h)
                        else:
                            ep_kv_cache = kv_cache  # GPU 0's cache
                            partial_0 = layer.attention.forward_split(
                                h, pos, ep_kv_cache, seq_state, kv_layer_offset,
                                num_new_tokens=chunk_M,
                            )

                        if _split_timing_detail:
                            torch.cuda.synchronize(dev)
                            _st_t_g0 = time.perf_counter()

                        # (d) GPU 1: half-head attention on CUDA stream
                        if attn_stream is not None:
                            with torch.cuda.stream(attn_stream):
                                if gpu1_layer.layer_type == "linear_attention":
                                    partial_1 = gpu1_layer.attention.forward_split(h_gpu1)
                                else:
                                    ep_kv_gpu1 = self._ep_kv_caches.get(str(gpu1_dev))
                                    ep_ss_gpu1 = _ep_seq_states.get(str(gpu1_dev))
                                    partial_1 = gpu1_layer.attention.forward_split(
                                        h_gpu1, pos_gpu1, ep_kv_gpu1, ep_ss_gpu1,
                                        kv_layer_offset, num_new_tokens=chunk_M,
                                    )

                        # (e) Wait for GPU 1, copy partial result to GPU 0
                        if attn_stream is not None:
                            attn_stream.synchronize()

                        if _split_timing_detail:
                            _st_t_g1 = time.perf_counter()

                        partial_1_on_0 = _to_device(partial_1, dev)

                        # (f) Concat halves + apply o_proj on GPU 0
                        attn_concat = torch.cat([partial_0, partial_1_on_0], dim=-1)
                        attn_out = layer.apply_attn_o_proj(attn_concat)

                        # (g) Post-attention norm
                        h, r = layer.post_attn_norm(attn_out, r)

                        if _split_timing:
                            torch.cuda.synchronize(dev)
                            _st_ta = time.perf_counter()
                            _st["attn"] += _st_ta - _st_t0
                        if _split_timing_detail:
                            _st["copy_h"] += _st_t_copy - _st_t0
                            _st["gpu0_fwd"] += _st_t_g0 - _st_t_copy
                            _st["gpu1_wait"] += _st_t_g1 - _st_t_g0
                            _st["copy_back"] += _st_ta - _st_t_g1
                            _st["gather_oproj"] += _st_ta - _st_t_g1  # approx

                    else:
                        # ── Original non-split attention path ──
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
                        if _split_timing:
                            torch.cuda.synchronize(dev)
                            _st_ta2 = time.perf_counter()
                            _st["attn"] += _st_ta2 - _st_t0

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _t_attn = time.perf_counter()
                        _attn_dt = _t_attn - _t_layer0
                        _ep_times["attention"] += _attn_dt
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

                    # (c) Shared expert on GPU 0 — run SERIALLY for timing
                    shared_output = None
                    if ep_timing:
                        # Force serial shared expert for accurate measurement
                        if layer.shared_expert is not None:
                            shared_output = layer._shared_expert_forward(h)
                        torch.cuda.synchronize(dev)
                        _t_shared = time.perf_counter()
                        _ep_times["shared_expert"] += _t_shared - _t_route
                    else:
                        has_shared_overlap = (layer._shared_stream is not None
                                              and layer.shared_expert is not None)
                        if has_shared_overlap:
                            layer._shared_stream.wait_stream(torch.cuda.current_stream(dev))
                            with torch.cuda.stream(layer._shared_stream):
                                shared_output = layer._shared_expert_forward(h)
                        elif layer.shared_expert is not None:
                            shared_output = layer._shared_expert_forward(h)

                    # (d) Routed experts — SERIAL for timing, parallel otherwise
                    if ep_timing:
                        # Serial per-GPU forward for accurate timing
                        partial_outputs = []
                        for _gi, mgr in enumerate(all_managers):
                            stream = ep_streams.get(mgr.device)
                            if stream is None:
                                # Primary GPU forward
                                torch.cuda.synchronize(dev)
                                _t_fwd0 = time.perf_counter()
                                out = mgr.forward(moe_layer_idx, h, topk_ids, topk_weights,
                                                  routed_only=True)
                                torch.cuda.synchronize(dev)
                                _ep_times[f"gpu{_gi}_forward"] += time.perf_counter() - _t_fwd0
                                partial_outputs.append((mgr.device, out))
                            else:
                                # Secondary GPU: measure input copy separately
                                _t_copy0 = time.perf_counter()
                                h_dev = _to_device(h, mgr.device)
                                ids_dev = _to_device(topk_ids, mgr.device)
                                wts_dev = _to_device(topk_weights, mgr.device)
                                torch.cuda.synchronize(mgr.device)
                                _ep_times[f"gpu{_gi}_input_copy"] += time.perf_counter() - _t_copy0

                                # Secondary GPU forward
                                _t_fwd0 = time.perf_counter()
                                out = mgr.forward(moe_layer_idx, h_dev, ids_dev, wts_dev,
                                                  routed_only=True)
                                torch.cuda.synchronize(mgr.device)
                                _ep_times[f"gpu{_gi}_forward"] += time.perf_counter() - _t_fwd0
                                partial_outputs.append((mgr.device, out))

                        # Output transfer + summation
                        torch.cuda.synchronize(dev)
                        _t_xfer0 = time.perf_counter()
                        routed_output = None
                        for po_device, po_out in partial_outputs:
                            if po_device == dev:
                                routed_output = po_out
                            else:
                                out_on_primary = _to_device(po_out, dev)
                                if routed_output is None:
                                    routed_output = out_on_primary
                                else:
                                    routed_output = routed_output + out_on_primary
                        torch.cuda.synchronize(dev)
                        _ep_times["output_transfer"] += time.perf_counter() - _t_xfer0

                    else:
                        # Normal parallel path (no timing)
                        h_ready = torch.cuda.current_stream(dev).record_event()

                        partial_outputs = []
                        for mgr in all_managers:
                            stream = ep_streams.get(mgr.device)
                            if stream is not None:
                                stream.wait_event(h_ready)
                                with torch.cuda.stream(stream):
                                    h_dev = _to_device(h, mgr.device)
                                    ids_dev = _to_device(topk_ids, mgr.device)
                                    wts_dev = _to_device(topk_weights, mgr.device)
                                    out = mgr.forward(moe_layer_idx, h_dev, ids_dev, wts_dev,
                                                      routed_only=True)
                                    partial_outputs.append((mgr.device, out))
                            else:
                                out = mgr.forward(moe_layer_idx, h, topk_ids, topk_weights,
                                                  routed_only=True)
                                partial_outputs.append((mgr.device, out))

                        if has_shared_overlap:
                            torch.cuda.current_stream(dev).wait_stream(layer._shared_stream)

                        routed_output = None
                        for po_device, po_out in partial_outputs:
                            if po_device == dev:
                                routed_output = po_out
                            else:
                                ep_streams[po_device].synchronize()
                                out_on_primary = _to_device(po_out, dev)
                                if routed_output is None:
                                    routed_output = out_on_primary
                                else:
                                    routed_output = routed_output + out_on_primary

                    # (f) Apply routed_scaling_factor + shared expert on GPU 0
                    rsf = self.cfg.routed_scaling_factor
                    if rsf != 1.0:
                        routed_output *= rsf
                    if shared_output is not None:
                        h = routed_output + shared_output
                    else:
                        h = routed_output

                    if _split_timing:
                        torch.cuda.synchronize(dev)
                        if gpu1_dev is not None:
                            torch.cuda.synchronize(gpu1_dev)
                        _st_te = time.perf_counter()
                        _st["expert_total"] += _st_te - (_st_ta if use_split_attn else _st_t0)
                        _st["layer_total"] += _st_te - _st_t0
                        _st_count += 1

                    if ep_timing:
                        torch.cuda.synchronize(dev)
                        _ep_times["layer_total"] += time.perf_counter() - _t_layer0
                        _ep_layer_count += 1

                # Save for next group
                chunk_hidden[c] = h
                chunk_residual[c] = r

                # Advance seq_state for this chunk
                if seq_state is not None:
                    seq_state.advance(chunk_M)
                for ep_ss in _ep_seq_states.values():
                    ep_ss.advance(chunk_M)

            # 4. Free experts for this group
            if _layer_timing and need_load:
                torch.cuda.synchronize(dev)
                _lt_compute_total += time.perf_counter() - _lt_compute0

            if _layer_timing and need_load:
                _lt_free0 = time.perf_counter()

            if need_load:
                # Skip empty_cache during pipelining: allocator reuses freed blocks
                # naturally, saving ~30ms/group. Only clear cache on the last group.
                _clear_cache = not _has_prefetch
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

        # Restore EP slicing state (single-GPU only)
        if _saved_ep is not None:
            manager.expert_start, manager.expert_end, manager.num_local_experts = _saved_ep
            manager._dma_bufs_initialized = False

        if dma_pool is not None:
            dma_pool.shutdown(wait=False)

        # ── Reset GPU 1 linear attention states after prefill ──
        # GPU 0 runs full computation (all heads) during split-attention prefill,
        # so its recurrent_state and conv_state are already correct for decode.
        # GPU 1's states are partial (second half) and not needed — just reset.
        if getattr(self, '_split_attention_active', False) and use_ep:
            for extra_dev in self.all_devices[1:]:
                dev_str = str(extra_dev)
                gpu1_layers = self._device_state[dev_str]["layers"]
                for layer in gpu1_layers:
                    if layer.layer_type == "linear_attention":
                        layer.attention.reset_state()

            # Free EP KV cache pages (not needed for decode)
            for ep_ss in _ep_seq_states.values():
                ep_ss.free()

        # ── EP timing summary ──
        if ep_timing:
            _total_measured = sum(v for k, v in _ep_times.items()
                                 if k not in ("layer_total", "attention_gqa", "attention_linear"))
            logger.info("=" * 70)
            logger.info("EP TIMING BREAKDOWN (%d MoE layers, %d dense layers, %d chunks)",
                        _ep_layer_count, _ep_dense_count, num_chunks)
            logger.info("-" * 70)
            for phase, t in _ep_times.items():
                if phase == "layer_total":
                    continue
                per_layer = (t / _ep_layer_count * 1000) if _ep_layer_count > 0 else 0
                pct = (t / _ep_times["layer_total"] * 100) if _ep_times["layer_total"] > 0 else 0
                logger.info("  %-20s %8.1f ms total  %6.2f ms/layer  %5.1f%%",
                            phase, t * 1000, per_layer, pct)
            logger.info("-" * 70)
            # Per-layer-type attention averages
            num_gqa = self.cfg.num_full_attention_layers if self.cfg.is_hybrid else _ep_layer_count
            num_linear = _ep_layer_count - num_gqa
            if num_gqa > 0:
                logger.info("  %-20s %6.2f ms/layer  (%d layers)",
                            "  avg GQA attn", _ep_times["attention_gqa"] / num_gqa * 1000, num_gqa)
            if num_linear > 0:
                logger.info("  %-20s %6.2f ms/layer  (%d layers)",
                            "  avg linear attn", _ep_times["attention_linear"] / num_linear * 1000, num_linear)
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

        # ── Split-attention timing summary ──
        if _split_timing and _st_count > 0:
            logger.info("=" * 70)
            logger.info("SPLIT-ATTENTION TIMING (%d layers)", _st_count)
            logger.info("-" * 70)
            for phase in ("attn", "copy_h", "gpu0_fwd", "gpu1_wait", "copy_back",
                         "gather_oproj", "expert_total", "layer_total"):
                t = _st[phase]
                per_layer = t / _st_count * 1000
                pct = (t / _st["layer_total"] * 100) if _st["layer_total"] > 0 else 0
                logger.info("  %-20s %8.1f ms total  %6.2f ms/layer  %5.1f%%",
                            phase, t * 1000, per_layer, pct)
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

        # ── Final result: last token's logits ──
        import flashinfer
        last_h = chunk_hidden[-1][-1:]
        last_r = chunk_residual[-1][-1:]
        state = self._device_state[str(dev)]
        flashinfer.norm.fused_add_rmsnorm(
            last_h, last_r, state["final_norm"], self.cfg.rms_norm_eps
        )
        final_logits = _linear(last_h, state["lm_head_data"]).float()

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
    ) -> List[int]:
        """Generate tokens autoregressively.

        Args:
            prompt_tokens: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            stop_token_ids: Stop generation on these tokens

        Returns:
            List of generated token IDs (excluding prompt)
        """
        if stop_token_ids is None:
            stop_token_ids = [self.cfg.eos_token_id]

        # Create per-rank sequence states (one per KV cache / GPU)
        seq_states_per_rank = [
            SequenceKVState(c, seq_id=0) if c is not None else None
            for c in self.kv_caches
        ]

        # Reset linear attention states for new sequence
        if self.cfg.is_hybrid:
            for layer in self.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        device = torch.device(self.ranks[0].device)
        generated = []

        _t_gen_start = time.perf_counter()

        try:
            # ── Prefill ──
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

            logits = self.forward(prompt_tensor, positions, seq_states_per_rank)
            # Take last token's logits
            next_logits = logits[-1:, :]

            next_token = sample(next_logits, temperature, top_k, top_p).item()
            generated.append(next_token)

            torch.cuda.synchronize()
            self._last_ttft = time.perf_counter() - _t_gen_start

            if next_token in stop_token_ids:
                return generated

            # ── Decode ──
            # Pre-allocate reusable tensors to avoid per-step CUDA allocations
            _decode_token_buf = torch.empty(1, dtype=torch.long, device=device)
            _decode_pos_buf = torch.empty(1, dtype=torch.int32, device=device)
            _t_decode_start = time.perf_counter()

            for step in range(max_new_tokens - 1):
                if TIMING.decode:
                    torch.cuda.synchronize()
                    _t_step_start = time.perf_counter()

                pos = len(prompt_tokens) + step
                _decode_token_buf[0] = next_token
                _decode_pos_buf[0] = pos

                if TIMING.decode:
                    _t_prep = time.perf_counter()

                logits = self.forward(_decode_token_buf, _decode_pos_buf, seq_states_per_rank)

                if TIMING.decode:
                    torch.cuda.synchronize()
                    _t_forward = time.perf_counter()

                next_token = sample(logits, temperature, top_k, top_p).item()

                if TIMING.decode:
                    _t_sample = time.perf_counter()
                    logger.info(
                        "DECODE-STEP %d: prep=%.1fms forward=%.1fms sample=%.1fms total=%.1fms",
                        step,
                        (_t_prep - _t_step_start) * 1000,
                        (_t_forward - _t_prep) * 1000,
                        (_t_sample - _t_forward) * 1000,
                        (_t_sample - _t_step_start) * 1000,
                    )

                generated.append(next_token)

                if next_token in stop_token_ids:
                    break

        finally:
            torch.cuda.synchronize()
            n_decode = len(generated) - 1  # First token is from prefill
            if n_decode > 0:
                self._last_decode_time = time.perf_counter() - _t_decode_start
                self._last_decode_tok_s = n_decode / self._last_decode_time
            else:
                self._last_decode_time = 0.0
                self._last_decode_tok_s = 0.0
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

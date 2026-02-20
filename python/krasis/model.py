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
    """Full model with streaming attention (GPU0) + EP MoE (all GPUs) + CPU experts."""

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

        # For hybrid models: maps absolute layer idx → KV cache layer offset
        # within its device's cache. Linear attention layers get -1 (no KV).
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

        # Phase 2: CPU expert weights
        cpu_start = time.perf_counter()
        print(f"\n\033[1m\033[36m▸ Loading CPU expert weights\033[0m", flush=True)
        logger.info("Phase 2: Loading CPU expert weights (Krasis INT4)...")
        self._load_cpu_experts()
        cpu_elapsed = time.perf_counter() - cpu_start
        logger.info("CPU experts loaded in %.1fs", cpu_elapsed)

        # Post-load RSS check: verify RAM estimate accuracy
        if self._estimated_expert_ram_gb > 0:
            _check_actual_rss(self._estimated_expert_ram_gb)

        # Phase 3: GPU prefill managers (one per device)
        if self.gpu_prefill_enabled and self.cfg.n_routed_experts > 0:
            print(f"\n\033[1m\033[36m▸ Initializing GPU prefill managers\033[0m", flush=True)
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
                    for m_size in (32, 512):
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
                    # w1 (gate+up): [E, K//16, 2*N*nb2], scale: [E, K//gs, 2*N]
                    w1 = torch.zeros(n_dummy, K // 16, 2 * N * nb2, dtype=torch.int32, device=device)
                    w1s = torch.zeros(n_dummy, K // gs, 2 * N, dtype=torch.bfloat16, device=device)
                    # w2 (down): [E, N//16, K*nb2], scale: [E, N//gs, K]
                    w2 = torch.zeros(n_dummy, N // 16, K * nb2, dtype=torch.int32, device=device)
                    w2s = torch.zeros(n_dummy, N // gs, K, dtype=torch.bfloat16, device=device)
                    x = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
                    g = torch.empty(1, n_dummy, device=device)
                    tw = torch.zeros(1, 1, dtype=torch.float32, device=device)
                    ti = torch.zeros(1, 1, dtype=torch.int32, device=device)
                    fused_marlin_moe(
                        hidden_states=x, w1=w1, w2=w2,
                        w1_scale=w1s, w2_scale=w2s,
                        gating_output=g, topk_weights=tw, topk_ids=ti,
                        global_num_experts=n_dummy, expert_map=None,
                        g_idx1=torch.empty(n_dummy, 0, dtype=torch.int32, device=device),
                        g_idx2=torch.empty(n_dummy, 0, dtype=torch.int32, device=device),
                        sort_indices1=torch.empty(n_dummy, 0, dtype=torch.int32, device=device),
                        sort_indices2=torch.empty(n_dummy, 0, dtype=torch.int32, device=device),
                        w1_zeros=None, w2_zeros=None,
                        workspace=ws, num_bits=bits, is_k_full=True,
                    )
                    torch.cuda.synchronize(device)
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
        """Stream-load GPU weights: all attention on GPU0.

        Streaming attention architecture: ALL attention weights, norms, gate
        weights, shared expert weights, embedding, final_norm, and lm_head
        live on GPU0. GPU1+ are reserved entirely for EP expert parallelism
        during prefill and HCS expert cache during decode.

        This eliminates cross-GPU hidden state transfers during attention
        (which can't be parallelized due to layer data dependencies) and
        frees GPU1+ VRAM entirely for HCS decode experts.
        """
        self.layers = []
        primary_dev = self.all_devices[0]
        torch.cuda.set_device(primary_dev)
        L = self.cfg.num_hidden_layers

        # All attention on GPU0 — single split covering all layers
        self._layer_split = [(0, L)]
        logger.info("Streaming attention: all %d layers on GPU0, %d GPUs for EP",
                     L, len(self.all_devices))

        # ── Load ALL layers to primary device ──
        logger.info("Loading full base model to %s...", primary_dev)
        self.embedding = loader.load_embedding(primary_dev)

        for layer_idx in range(L):
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
        _vram_snap("after-all-layers-loaded")

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
            total += layer.input_norm_weight.nelement() * layer.input_norm_weight.element_size()
            total += layer.post_attn_norm_weight.nelement() * layer.post_attn_norm_weight.element_size()

            # Attention weights
            attn = layer.attention
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
        for layer in self.layers:
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

        # Wire manager to MoE layers (each layer gets its own device's manager)
        for layer in self.layers:
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
        saved_layer_split = self._layer_split
        saved_all_devices = self.all_devices
        saved_hcs_device = self._hcs_device

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
        hidden_hcs = _to_device(hidden_on_layer_dev, hcs_dev)
        topk_ids_hcs = _to_device(topk_ids, hcs_dev)
        topk_weights_hcs = _to_device(topk_weights, hcs_dev)

        if timing:
            torch.cuda.synchronize(hcs_dev)
            t_xfer_to_hcs = time.perf_counter()

        # 4. Routed experts on HCS device via its manager
        hcs_mgr = self.gpu_prefill_managers.get(str(hcs_dev))
        if hcs_mgr is not None:
            routed_output = hcs_mgr.forward(
                moe_layer_idx, hidden_hcs, topk_ids_hcs, topk_weights_hcs,
                routed_only=True,
            )
        else:
            # Fallback: CPU path (should not happen with HCS but be safe)
            routed_output = layer._routed_expert_forward(
                hidden_on_layer_dev, topk_ids, topk_weights, moe_layer_idx,
            )
            # Already on layer_dev, skip transfer back
            rsf = self.cfg.routed_scaling_factor
            if rsf != 1.0:
                routed_output = routed_output * rsf
            if has_shared:
                torch.cuda.current_stream(layer_dev).wait_stream(layer._shared_stream)
                shared_output = layer._se_output if layer._se_graph is not None else shared_output
                routed_output = routed_output + shared_output
            elif shared_output is not None:
                routed_output = routed_output + shared_output
            return routed_output

        if timing:
            torch.cuda.synchronize(hcs_dev)
            t_hcs_done = time.perf_counter()

        # 5. Transfer routed output back to layer_dev
        routed_on_layer = _to_device(routed_output, layer_dev)

        if timing:
            torch.cuda.synchronize(layer_dev)
            t_xfer_back = time.perf_counter()

        # 6. Apply scaling + combine with shared expert
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

        Returns:
            logits: [M, vocab_size] float32
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]
        timing = TIMING.decode and M == 1

        if timing:
            t_fwd_start = time.perf_counter()

        # ── Layer-grouped / active-only / LRU prefill routing ──
        if (
            M > 1
            and (self.expert_divisor >= 2 or self.expert_divisor < 0)
            and M >= self.gpu_prefill_threshold
            and self.gpu_prefill_enabled
            and self.cfg.n_routed_experts > 0
        ):
            return self._forward_prefill_with_oom_retry(token_ids, positions, seq_states)

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
            _t_lin_attn_layers = 0  # count of linear attention layers
            _t_gqa_layers = 0      # count of GQA attention layers
            _t_moe_layers = 0      # count of MoE layers
            _t_unified_layers = 0  # count of unified (same-device) layers

        for abs_layer_idx in range(self.cfg.num_hidden_layers):
            layer = self.layers[abs_layer_idx]
            layer_dev = torch.device(layer.device)
            gpu_idx = self._get_gpu_for_layer(abs_layer_idx)
            kv_cache = self.kv_caches[gpu_idx]
            seq_state = seq_states[gpu_idx]

            # MoE layer index for Krasis engine
            moe_layer_idx = abs_layer_idx - first_k if abs_layer_idx >= first_k else None
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

            # Decide execution path: split (attention on layer_dev, MoE on HCS) or unified
            use_cross_device_moe = (
                M == 1
                and layer.is_moe
                and hcs_dev is not None
                and torch.device(hcs_dev) != layer_dev
            )

            if use_cross_device_moe:
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
                    _t_attn_total += _t_la_done - _t_la
                    if layer.layer_type == "linear_attention":
                        _t_lin_attn_layers += 1
                    else:
                        _t_gqa_layers += 1

                # MoE on HCS device, result back on layer_dev
                hidden = self._cross_device_moe(layer, hidden, moe_layer_idx)

                if timing:
                    # _cross_device_moe already syncs internally when timing
                    _t_moe_done = time.perf_counter()
                    _t_moe_total += _t_moe_done - _t_la_done
                    _t_moe_layers += 1
            else:
                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_uni0 = time.perf_counter()

                # Unified path: everything on layer_dev
                if kv_layer_offset < 0:
                    hidden, residual = layer.forward(
                        hidden, residual, layer_positions,
                        None, None, -1,
                        moe_layer_idx, num_new_tokens=M,
                    )
                else:
                    hidden, residual = layer.forward(
                        hidden, residual, layer_positions,
                        kv_cache, seq_state, kv_layer_offset,
                        moe_layer_idx, num_new_tokens=M,
                    )

                if timing:
                    torch.cuda.synchronize(layer_dev)
                    _t_uni1 = time.perf_counter()
                    _t_unified_layers += 1
                    # Unified layers include both attn + MoE in one call
                    # Attribute to MoE if it's a MoE layer, else attn
                    if layer.is_moe:
                        _t_moe_total += _t_uni1 - _t_uni0
                        _t_moe_layers += 1
                    else:
                        _t_attn_total += _t_uni1 - _t_uni0

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

        import flashinfer
        flashinfer.norm.fused_add_rmsnorm(
            hidden, residual, self.final_norm, self.cfg.rms_norm_eps
        )

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
            total_ms = (t_fwd_end - t_fwd_start) * 1000
            layers_ms = (t_after_layers - t_fwd_start) * 1000
            post_ms = (t_fwd_end - t_after_layers) * 1000
            attn_ms = _t_attn_total * 1000
            moe_ms = _t_moe_total * 1000
            xfer_ms = _t_xfer_total * 1000
            logger.info(
                "DECODE-TOKEN: total=%.1fms (layers=%.1fms post=%.1fms)",
                total_ms, layers_ms, post_ms,
            )
            logger.info(
                "  BREAKDOWN: attn=%.1fms (%d lin + %d gqa) moe=%.1fms (%d layers) xfer=%.1fms unified=%d",
                attn_ms, _t_lin_attn_layers, _t_gqa_layers,
                moe_ms, _t_moe_layers, xfer_ms, _t_unified_layers,
            )
            if layers_ms > 0:
                logger.info(
                    "  PERCENT: attn=%.0f%% moe=%.0f%% xfer=%.0f%% post=%.0f%% unaccounted=%.0f%%",
                    attn_ms / total_ms * 100,
                    moe_ms / total_ms * 100,
                    xfer_ms / total_ms * 100,
                    post_ms / total_ms * 100,
                    (total_ms - attn_ms - moe_ms - xfer_ms - post_ms) / total_ms * 100,
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

        # Defragment the PyTorch allocator before prefill.  Decode leaves many
        # small cached-but-free blocks that prevent contiguous DMA allocations.
        # This reclaims those blocks so layer group DMA can allocate freely.
        torch.cuda.empty_cache()

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

        Streaming attention: ALL attention runs on GPU0. Expert Parallelism
        splits MoE experts across all GPUs for parallel compute. No cross-GPU
        hidden state transfers during attention (only for EP MoE dispatch).
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
        _pipeline_enabled = bool(_dma_managers) and not is_active_only and not _validation
        _has_prefetch = False
        _prefetch_futures = []
        if _pipeline_enabled:
            logger.info("DMA pipelining ENABLED (%d managers, %d groups)",
                        len(_dma_managers), len(groups))

        _attn_streaming = getattr(self, '_attn_offloaded', False)

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

                for abs_layer_idx in group_layers:
                    layer = self.layers[abs_layer_idx]
                    moe_layer_idx = abs_layer_idx - first_k if abs_layer_idx >= first_k else None
                    kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, 0)

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
                        else:
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

            # 5. Free attention weights for this group (streaming mode)
            if _attn_streaming:
                self._free_attention_group(group_layers)

        # Reload all attention weights for decode (streaming mode)
        if _attn_streaming:
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

        # ── Final result: last token's logits ──
        # final_norm and lm_head are always on GPU0 (first device)
        import flashinfer
        last_h = chunk_hidden[-1][-1:]
        last_r = chunk_residual[-1][-1:]
        first_dev = self.all_devices[0]
        last_h = _to_device(last_h, first_dev)
        last_r = _to_device(last_r, first_dev)
        flashinfer.norm.fused_add_rmsnorm(
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

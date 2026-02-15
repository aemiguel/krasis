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
    """Auto-compute layer group divisor for HCS prefill based on VRAM budget.

    HCS stores hot experts statically for decode, but prefill needs ALL experts
    loaded per-group via preload_layer_group(). This computes how many groups
    are needed so each group fits in available VRAM.
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
        self.kv_dtype = kv_dtype
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

        # Load tokenizer
        self.tokenizer = Tokenizer(self.cfg.model_path)

        self._loaded = True
        total = time.perf_counter() - start
        logger.info("Model fully loaded in %.1fs", total)

    def _load_gpu_weights(self, loader: WeightLoader):
        """Stream-load all GPU weights layer by layer."""
        self.layers = []

        for rank in self.ranks:
            dev = torch.device(rank.device)

            # Embedding on first rank
            if rank.has_embedding:
                self.embedding = loader.load_embedding(dev)

            # Layers
            for layer_idx in rank.layer_range:
                weights = loader.load_layer(layer_idx, dev)
                layer = TransformerLayer(
                    self.cfg, layer_idx, weights, dev,
                    krasis_engine=None,  # set after CPU load
                    gpu_prefill_manager=None,  # set after _init_gpu_prefill
                    gpu_prefill_threshold=self.gpu_prefill_threshold,
                    attention_backend=self.attention_backend,
                )
                self.layers.append(layer)

            # Final norm + LM head on last rank
            if rank.has_lm_head:
                self.final_norm = loader.load_final_norm(dev)
                self.lm_head_data = loader.load_lm_head(dev)

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

        # Wire engine to all MoE layers + allocate pinned buffers for fused dispatch
        first_k = self.cfg.first_k_dense_replace
        for layer in self.layers:
            if layer.is_moe:
                layer.krasis_engine = engine
                # Allocate pinned CPU buffers for zero-alloc M=1 dispatch
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
        """Create GpuPrefillManager per device and wire to MoE layers."""
        from krasis.gpu_prefill import GpuPrefillManager

        for rank in self.ranks:
            dev = torch.device(rank.device)
            dev_str = str(dev)
            if dev_str not in self.gpu_prefill_managers:
                # Pass Krasis engine for unified weight access (eliminates
                # the ~438 GB Python Marlin RAM cache)
                engine_ref = getattr(self, 'krasis_engine', None)
                num_moe_layers = self.cfg.num_hidden_layers - self.cfg.first_k_dense_replace
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
                )
                self.gpu_prefill_managers[dev_str] = manager
                logger.info("GPU prefill manager created for %s", dev_str)

        # Pre-prepare all MoE layers from disk cache at startup
        # (avoids lazy quantization during forward() which causes VRAM exhaustion)
        for dev_str, manager in self.gpu_prefill_managers.items():
            manager.prepare_all_layers()

        # Wire manager to MoE layers
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

    def _init_kv_caches(self):
        """Allocate paged KV caches per GPU.

        For hybrid models, only full attention layers need KV cache.
        Builds _kv_layer_offsets mapping absolute layer idx → offset within
        the rank's KV cache (-1 for linear attention layers).
        """
        self.kv_caches = []
        self._kv_layer_offsets = {}
        use_combined = (self.attention_backend == "trtllm")

        # Compute VRAM reservation for layer-grouped expert buffers
        # so the KV cache auto-sizer doesn't over-allocate.
        # Use per-rank MoE layer count (not total) since each GPU only loads its rank's layers.
        vram_reserve = 0
        if self.expert_divisor >= 2 and self.gpu_prefill_enabled and self.cfg.n_routed_experts > 0:
            # Find max per-rank MoE layer count
            max_rank_moe = max(
                sum(1 for li in rank.layer_range if self.cfg.is_moe_layer(li))
                for rank in self.ranks
            )
            for dev_str, manager in self.gpu_prefill_managers.items():
                per_expert = manager._per_expert_vram_bytes()
                moe_per_group = ceil(max_rank_moe / self.expert_divisor)
                vram_reserve = per_expert * self.cfg.n_routed_experts * moe_per_group
                logger.info(
                    "VRAM reservation for layer-grouped prefill: %.1f MB "
                    "(moe_per_group=%d, %d experts, %.1f MB/expert)",
                    vram_reserve / 1e6, moe_per_group,
                    self.cfg.n_routed_experts, per_expert / 1e6,
                )
                break  # same reservation per GPU

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
                    vram_reserve_bytes=vram_reserve,
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
        max_retries = 3
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

        Instead of loading experts per-layer-per-chunk (O(chunks × layers) DMA),
        loads experts per-group (O(groups) DMA ≈ O(1 model load)).

        Loop structure:
            for each group → DMA experts once → for each chunk → for each layer → compute

        Args:
            token_ids: [M] int64 token IDs
            positions: [M] int32 position indices
            seq_states: One SequenceKVState per PP rank

        Returns:
            logits: [vocab_size] or [1, vocab_size] float32 (last token only)
        """
        assert self._loaded, "Model not loaded. Call load() first."
        M = token_ids.shape[0]

        # ── Embedding (GPU0) ──
        first_dev = torch.device(self.ranks[0].device)
        all_hidden = self.embedding[token_ids.to(first_dev)]  # [M, hidden_size]

        # ── Pre-allocate KV pages for all ranks ──
        for seq_state in seq_states:
            if seq_state is not None:
                seq_state.ensure_capacity(M)

        # ── Token chunking ──
        # Compute max chunk size based on fused_marlin_moe intermediate allocation.
        # The kernel allocates per token: topk × (N + max(2N, K)) × dtype_size bytes.
        # Larger chunks = more tokens per expert = better GPU SM utilization.
        topk = self.cfg.num_experts_per_tok
        N_moe = self.cfg.moe_intermediate_size
        K_moe = self.cfg.hidden_size
        dtype_size = 2  # BF16
        intermediate_per_token = topk * (N_moe + max(2 * N_moe, K_moe)) * dtype_size
        # Hidden state + residual per token (carried through layers)
        hidden_per_token = K_moe * dtype_size * 2  # hidden + residual

        # Estimate available VRAM for activation intermediates.
        # Include both CUDA driver-level free memory AND PyTorch's reserved-but-unallocated
        # pool (memory PyTorch pre-reserved from CUDA but hasn't assigned to tensors).
        # This gives a more accurate picture of what PyTorch can actually allocate.
        try:
            free_per_dev = {}
            for r in self.ranks:
                dev = torch.device(r.device)
                cuda_free = torch.cuda.mem_get_info(dev)[0]
                pytorch_pool = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
                free_per_dev[str(dev)] = cuda_free + max(0, pytorch_pool)
            min_free = min(free_per_dev.values())
            first_dev_str = str(torch.device(self.ranks[0].device))
            first_manager = self.gpu_prefill_managers.get(first_dev_str)
            if first_manager and intermediate_per_token > 0:
                per_expert = first_manager._per_expert_vram_bytes()
                # 1-layer expert group VRAM (including shared experts)
                expert_group_bytes = self.cfg.n_routed_experts * per_expert
                if self.cfg.n_shared_experts > 0:
                    expert_group_bytes += per_expert  # ~1 shared expert worth
                # Safety margin: kernel workspace, PyTorch allocator fragmentation,
                # attention intermediates, hidden states between chunks.
                # Use 2x safety factor on the per-token estimate to account for
                # peak memory during kernel execution (multiple concurrent buffers).
                safety_factor = 2.0
                total_per_token = (intermediate_per_token + hidden_per_token) * safety_factor
                activation_budget = min_free - expert_group_bytes
                if activation_budget > 0:
                    max_chunk = int(activation_budget / total_per_token)
                    chunk_size = max(256, min(M, max_chunk))
                else:
                    chunk_size = 512
                logger.info(
                    "Chunk calc: free=%s, min_free=%.0f MB, expert_group=%.0f MB, "
                    "per_tok=%.0f KB (%.0f KB intermediate + %.0f KB hidden, %.1fx safety), "
                    "budget=%.0f MB, max_chunk=%d → chunk_size=%d",
                    {k: f"{v/1e6:.0f}MB" for k, v in free_per_dev.items()},
                    min_free / 1e6, expert_group_bytes / 1e6,
                    total_per_token / 1024, intermediate_per_token / 1024,
                    hidden_per_token / 1024, safety_factor,
                    activation_budget / 1e6 if activation_budget > 0 else 0,
                    max_chunk if activation_budget > 0 else 0, chunk_size,
                )
            else:
                chunk_size = 2048
                logger.info("Chunk calc: no manager (key=%s, managers=%s), defaulting to %d",
                            first_dev_str, list(self.gpu_prefill_managers.keys()), chunk_size)
        except Exception as e:
            chunk_size = 2048
            logger.warning("Chunk calc exception: %s, defaulting to %d", e, chunk_size)

        # Apply OOM retry override (from _forward_prefill_with_oom_retry)
        if max_chunk_override is not None and max_chunk_override < chunk_size:
            logger.info("Chunk size capped by OOM retry: %d → %d", chunk_size, max_chunk_override)
            chunk_size = max_chunk_override

        num_chunks = ceil(M / chunk_size)
        logger.info(
            "Prefill chunking: M=%d, chunk_size=%d (%d chunks), "
            "intermediate=%.0f KB/tok, %.0f MB peak per chunk",
            M, chunk_size, num_chunks,
            intermediate_per_token / 1024,
            chunk_size * intermediate_per_token / 1e6,
        )

        chunk_hidden = []
        chunk_positions = []
        chunk_residual: List[Optional[torch.Tensor]] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, M)
            chunk_hidden.append(all_hidden[start:end])
            chunk_positions.append(positions[start:end])
            chunk_residual.append(None)

        del all_hidden  # Free embedding memory

        # ── Process layer groups per rank ──
        first_k = self.cfg.first_k_dense_replace

        for rank_idx, rank in enumerate(self.ranks):
            dev = torch.device(rank.device)
            kv_cache = self.kv_caches[rank_idx]
            seq_state = seq_states[rank_idx]

            # GPU prefill manager for this device
            manager = self.gpu_prefill_managers.get(str(dev))

            # Base index into self.layers for this rank
            rank_layer_base = sum(r.num_layers for r in self.ranks[:rank_idx])

            # Compute layer groups
            # HCS mode uses layer_grouped prefill — auto-compute divisor from VRAM
            if manager and manager._prefill_mode == "hot_cached_static":
                effective_divisor = _hcs_prefill_divisor(manager, rank, self.cfg)
            else:
                effective_divisor = self.expert_divisor
            groups = _compute_layer_groups(rank, self.cfg, effective_divisor)

            logger.info(
                "Layer-grouped prefill: rank %d, %d groups, %d layers, %d chunks of %d tokens",
                rank_idx, len(groups), rank.num_layers, num_chunks, chunk_size,
            )

            # Active-only/LRU mode: skip group preloading (DMA/dispatch in forward())
            # HCS uses layer_grouped preloading for prefill (M>1), HCS decode (M=1) separately
            is_active_only = manager and manager._prefill_mode in ("active_only", "lru")
            if is_active_only:
                manager.reset_ao_stats()

            # Track per-rank prefill timing
            rank_dma_time = 0.0
            rank_compute_time = 0.0

            for group_idx, (group_layers, group_moe_indices) in enumerate(groups):
                # Load experts for this group (skip for active_only — DMA in forward())
                if TIMING.prefill:
                    torch.cuda.synchronize(dev)
                    t_dma_start = time.perf_counter()

                if manager and group_moe_indices and not is_active_only:
                    manager.preload_layer_group(group_moe_indices)

                if TIMING.prefill:
                    torch.cuda.synchronize(dev)
                    dma_elapsed = time.perf_counter() - t_dma_start
                    rank_dma_time += dma_elapsed
                    t_compute_start = time.perf_counter()

                # Reset seq_state for this group (each group's layers start with 0 KV)
                seq_state.seq_len = 0

                # Process all chunks through this group
                for c in range(num_chunks):
                    h = _to_device(chunk_hidden[c], dev)
                    r = _to_device(chunk_residual[c], dev) if chunk_residual[c] is not None else None
                    pos = _to_device(chunk_positions[c], dev)
                    chunk_M = h.shape[0]

                    for abs_layer_idx in group_layers:
                        local_offset = abs_layer_idx - rank.layer_start
                        global_idx = rank_layer_base + local_offset
                        layer = self.layers[global_idx]

                        moe_layer_idx = None
                        if layer.is_moe:
                            moe_layer_idx = abs_layer_idx - first_k

                        kv_layer_offset = self._kv_layer_offsets.get(abs_layer_idx, local_offset)

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

                    # Save hidden/residual for this chunk
                    chunk_hidden[c] = h
                    chunk_residual[c] = r

                    # Advance seq_state: these layers have now seen chunk_M more tokens
                    if seq_state is not None:
                        seq_state.advance(chunk_M)

                if TIMING.prefill:
                    torch.cuda.synchronize(dev)
                    compute_elapsed = time.perf_counter() - t_compute_start
                    rank_compute_time += compute_elapsed

                # Free experts for this group (skip for active_only)
                if manager and group_moe_indices and not is_active_only:
                    manager.free_layer_group()

            if TIMING.prefill:
                total = rank_dma_time + rank_compute_time
                logger.info(
                    "PREFILL-RANK %d: DMA=%.2fs (%.0f%%), compute=%.2fs (%.0f%%), total=%.2fs",
                    rank_idx,
                    rank_dma_time, (rank_dma_time / total * 100) if total > 0 else 0,
                    rank_compute_time, (rank_compute_time / total * 100) if total > 0 else 0,
                    total,
                )

            # Log active-only stats for this rank
            if is_active_only:
                stats = manager.get_ao_stats()
                logger.info(
                    "Active-only rank %d: DMA=%.2fs (%.1f MB), "
                    "misses=%d, hits=%d, pinned=%d",
                    rank_idx, stats["dma_time"],
                    stats["dma_bytes"] / 1e6,
                    stats["cache_misses"], stats["cache_hits"],
                    stats["pinned_experts"],
                )
                # Notify for warmup/pinning
                manager.notify_request_done()

            # After all groups, seq_state.seq_len should equal M
            if seq_state is not None:
                assert seq_state.seq_len == M, (
                    f"seq_len mismatch after rank {rank_idx}: {seq_state.seq_len} != {M}"
                )

        # ── Final norm (last chunk only, contains last token) ──
        last_dev = torch.device(self.ranks[-1].device)
        last_hidden = _to_device(chunk_hidden[-1], last_dev)
        last_residual = _to_device(chunk_residual[-1], last_dev)

        import flashinfer
        flashinfer.norm.fused_add_rmsnorm(
            last_hidden, last_residual, self.final_norm, self.cfg.rms_norm_eps
        )

        # ── LM head (last token only) ──
        hidden = last_hidden[-1:, :]
        logits = _linear(hidden, self.lm_head_data)
        logits = logits.float()

        return logits

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

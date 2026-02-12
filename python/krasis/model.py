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
from typing import List, Optional

import torch

# Gate forward() diagnostics behind env var to avoid GPU sync overhead (.item() calls)
_DIAG_ENABLED = os.environ.get("KRASIS_DIAG", "") == "1"

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
    ):
        self.cfg = ModelConfig.from_model_path(model_path)
        self.quant_cfg = quant_cfg or QuantConfig()

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

        logger.info(
            "KrasisModel: %d layers, PP=%s, %d GPUs, attn=%s",
            self.cfg.num_hidden_layers, pp_partition, len(self.ranks),
            attention_backend,
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
        engine = KrasisEngine(parallel=True, num_threads=self.krasis_threads)
        engine.load(self.cfg.model_path, cpu_num_bits=cpu_bits, gpu_num_bits=gpu_bits)
        self.krasis_engine = engine

        # Wire engine to all MoE layers
        first_k = self.cfg.first_k_dense_replace
        for layer in self.layers:
            if layer.is_moe:
                layer.krasis_engine = engine

        logger.info(
            "Krasis engine: %d MoE layers, %d experts, hidden=%d",
            engine.num_moe_layers(), engine.num_experts(), engine.hidden_size(),
        )

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
        """Allocate paged KV caches per GPU."""
        self.kv_caches = []
        use_combined = (self.attention_backend == "trtllm")
        for rank in self.ranks:
            dev = torch.device(rank.device)
            cache = PagedKVCache(
                self.cfg,
                num_layers=rank.num_layers,
                device=dev,
                kv_dtype=self.kv_dtype,
                combined=use_combined,
            )
            self.kv_caches.append(cache)

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

        # ── Embedding (GPU0) ──
        first_dev = torch.device(self.ranks[0].device)
        hidden = self.embedding[token_ids.to(first_dev)]  # [M, hidden_size]

        # Diagnostics: track first N forward calls (gated behind KRASIS_DIAG=1)
        diag = False
        if _DIAG_ENABLED:
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
            seq_state.ensure_capacity(M)

            # Transfer hidden state to this rank's device (CPU bounce if P2P broken)
            hidden = _to_device(hidden, dev)
            if residual is not None:
                residual = _to_device(residual, dev)

            # Transfer positions once per rank
            rank_positions = _to_device(positions, dev)

            for layer_offset in range(rank.num_layers):
                abs_layer_idx = rank.layer_start + layer_offset
                layer = self.layers[layer_global_idx]

                # MoE layer index for Krasis engine
                moe_layer_idx = None
                if layer.is_moe:
                    moe_layer_idx = abs_layer_idx - first_k

                hidden, residual = layer.forward(
                    hidden,
                    residual,
                    rank_positions,
                    kv_cache,
                    seq_state,
                    layer_offset,
                    moe_layer_idx,
                    num_new_tokens=M,
                )

                if _DIAG_ENABLED and diag and abs_layer_idx in (0, 1, self.cfg.num_hidden_layers // 2, self.cfg.num_hidden_layers - 1):
                    h = hidden[-1] if hidden.shape[0] > 1 else hidden[0]
                    r = residual[-1] if residual.shape[0] > 1 else residual[0]
                    logger.info("DIAG[%d] L%d: hid std=%.4f max=%.4f | res std=%.4f max=%.4f",
                                self._diag_count, abs_layer_idx,
                                h.float().std(), h.float().abs().max(),
                                r.float().std(), r.float().abs().max())

                layer_global_idx += 1

            # Advance KV cache seq_len (once per rank, after all layers processed)
            seq_state.advance(M)

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

        if _DIAG_ENABLED and diag:
            last_logits = logits[-1]
            topk_vals, topk_ids = last_logits.topk(5)
            tok_strs = []
            if self.tokenizer:
                for tid in topk_ids.tolist():
                    tok_strs.append(repr(self.tokenizer.decode([tid])))
            logger.info("DIAG[%d] logits: std=%.2f top5=%s",
                        self._diag_count, last_logits.std(),
                        list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

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
        seq_states_per_rank = [SequenceKVState(c, seq_id=0) for c in self.kv_caches]

        device = torch.device(self.ranks[0].device)
        generated = []

        try:
            # ── Prefill ──
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

            logits = self.forward(prompt_tensor, positions, seq_states_per_rank)
            # Take last token's logits
            next_logits = logits[-1:, :]

            next_token = sample(next_logits, temperature, top_k, top_p).item()
            generated.append(next_token)

            if next_token in stop_token_ids:
                return generated

            # ── Decode ──
            for step in range(max_new_tokens - 1):
                pos = len(prompt_tokens) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

                logits = self.forward(token_tensor, pos_tensor, seq_states_per_rank)
                next_token = sample(logits, temperature, top_k, top_p).item()
                generated.append(next_token)

                if next_token in stop_token_ids:
                    break

        finally:
            # Free KV cache pages
            for s in seq_states_per_rank:
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

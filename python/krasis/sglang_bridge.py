"""Drop-in replacement for kt_kernel.KTMoEWrapper.

This module provides KrasisMoEWrapper which matches the interface that
SGLang's KTEPWrapperMethod expects from KTMoEWrapper. It handles:
- GPU ↔ CPU tensor transfers
- Async submit/sync pattern via Krasis Rust engine
- Expert ID masking (GPU experts = -1)
- GPU prefill: fused_marlin_moe kernel for batch_size > threshold

Usage in SGLang:
    # Before server start:
    from krasis.sglang_bridge import KrasisMoEWrapper
    KrasisMoEWrapper.hf_model_path = "/path/to/hf-model"

    # Then in kt_ep_wrapper.py, replace:
    #   from kt_kernel import KTMoEWrapper
    # with:
    #   from krasis.sglang_bridge import KrasisMoEWrapper as KTMoEWrapper
"""

import json
import logging
import os
import sys
import threading
from typing import Optional

import numpy as np
import torch

from krasis import KrasisEngine

logger = logging.getLogger(__name__)


class KrasisMoEWrapper:
    """SGLang-compatible MoE wrapper backed by Krasis Rust CPU engine.

    Implements the same interface as kt_kernel.KTMoEWrapper:
    - submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
    - sync_forward(hidden_states, cuda_stream)
    - load_weights(physical_to_logical_map_cpu)

    GPU Prefill:
    When batch_size >= gpu_prefill_threshold, uses fused_marlin_moe on GPU
    instead of CPU. This is 10-20x faster for long prompts (>300 tokens).
    """

    # ── Class-level configuration ─────────────────────────────────
    # Set before creating any wrapper instances.
    hf_model_path: Optional[str] = None
    num_threads: int = 16
    gpu_prefill_threshold: int = 300  # tokens; 0 = disabled
    cpu_expert_bits: int = 4  # 4 for INT4, 8 for INT8 CPU expert quantization

    # Shared engine singleton (loaded once, used by all layer wrappers)
    _shared_engine: Optional[KrasisEngine] = None
    _engine_lock = threading.Lock()
    _first_k_dense: Optional[int] = None

    # PP (pipeline parallel) state — set during load_weights / _get_engine
    _pp_first_layer_idx: Optional[int] = None  # first absolute layer this rank sees
    pp_start_moe_layer: Optional[int] = None   # first MoE layer index for this rank
    pp_num_moe_layers: Optional[int] = None    # number of MoE layers on this rank

    # Shared GPU prefill manager singleton
    _shared_gpu_prefill = None
    _gpu_prefill_lock = threading.Lock()

    # VRAM budget verification (set once per rank)
    _vram_verified = False

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_gpu_experts: int,
        cpuinfer_threads: int = 16,
        threadpool_count: int = 1,
        weight_path: str = "",
        chunked_prefill_size: int = 8192,
        method: str = "KRASIS",
        max_deferred_experts_per_token: Optional[int] = None,
        cpu_save: bool = False,
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_gpu_experts = num_gpu_experts

        # Resolve HF model path: class var > env var > weight_path fallback
        self._model_path = (
            KrasisMoEWrapper.hf_model_path
            or os.environ.get("KRASIS_MODEL_PATH")
            or weight_path
        )
        self._cpuinfer_threads = cpuinfer_threads

        # State for async forward
        self._device: Optional[torch.device] = None
        self._batch_size: int = 0
        self._engine: Optional[KrasisEngine] = None
        self._gpu_output: Optional[torch.Tensor] = None  # GPU prefill result

        logger.info(
            "KrasisMoEWrapper: layer=%d, experts=%d, gpu_experts=%d, hidden=%d",
            layer_idx, num_experts, num_gpu_experts, hidden_size,
        )

    # ── Shared engine management ──────────────────────────────────

    @classmethod
    def _get_engine(cls, model_path: str, num_threads: int) -> KrasisEngine:
        """Get or create the shared KrasisEngine singleton.

        For PP (pipeline parallel) setups:
        - Reads SGLANG_PP_LAYER_PARTITION env var (e.g. "20,21,20")
        - Uses _pp_first_layer_idx to infer which rank this is
        - Computes which MoE layers this rank owns
        - Passes start_layer/max_layers to engine.load() for partial loading
        """
        with cls._engine_lock:
            if cls._shared_engine is not None:
                return cls._shared_engine

            # Parse first_k_dense_replace from config
            cls._first_k_dense = cls._read_first_k_dense(model_path)
            first_k = cls._first_k_dense or 0

            # Determine PP layer range for this rank
            start_layer = None
            max_layers = None
            partition_str = os.environ.get("SGLANG_PP_LAYER_PARTITION", "")
            if partition_str and cls._pp_first_layer_idx is not None:
                partitions = [int(x.strip()) for x in partition_str.split(",")]
                first_layer = cls._pp_first_layer_idx

                # Infer rank from first_layer_idx (range check — first MoE layer
                # may not be at partition boundary if dense layers exist)
                cumsum = 0
                rank = -1
                for i, count in enumerate(partitions):
                    if cumsum <= first_layer < cumsum + count:
                        rank = i
                        break
                    cumsum += count

                if rank >= 0:
                    rank_start = sum(partitions[:rank])
                    rank_end = rank_start + partitions[rank]

                    # Compute which MoE layers fall in [rank_start, rank_end)
                    # MoE layers are [first_k .. num_hidden_layers)
                    moe_start_abs = max(rank_start, first_k)
                    moe_end_abs = rank_end  # all layers >= first_k in range are MoE

                    if moe_end_abs > first_k and moe_start_abs < rank_end:
                        start_layer = moe_start_abs - first_k  # 0-based MoE index
                        max_layers = moe_end_abs - moe_start_abs
                        cls.pp_start_moe_layer = start_layer
                        cls.pp_num_moe_layers = max_layers
                        logger.info(
                            "PP rank %d: layers [%d..%d), MoE layers [%d..%d) "
                            "(start_moe=%d, count=%d)",
                            rank, rank_start, rank_end,
                            moe_start_abs, moe_end_abs,
                            start_layer, max_layers,
                        )
                    else:
                        # This rank has no MoE layers (all dense)
                        cls.pp_start_moe_layer = 0
                        cls.pp_num_moe_layers = 0
                        logger.info(
                            "PP rank %d: layers [%d..%d) — no MoE layers",
                            rank, rank_start, rank_end,
                        )
                else:
                    logger.warning(
                        "PP: could not infer rank from first_layer_idx=%d, "
                        "partition=%s — loading all MoE layers",
                        first_layer, partition_str,
                    )

            # Log system RAM before loading
            try:
                import psutil
                mem = psutil.virtual_memory()
                logger.info(
                    "System RAM before Krasis load: used=%.1f/%.1f GiB (%.0f%% used), "
                    "available=%.1f GiB",
                    mem.used / (1024**3), mem.total / (1024**3),
                    mem.percent, mem.available / (1024**3))
            except ImportError:
                pass

            logger.info("Loading Krasis engine from %s (%d threads)", model_path, num_threads)
            if start_layer is not None:
                logger.info(
                    "  Partial load: start_layer=%d, max_layers=%s",
                    start_layer, max_layers)
            sys.stdout.flush()
            sys.stderr.flush()

            # -- CRASH DIAGNOSTIC: fine-grained logging around engine creation --
            logger.info("[DIAG] Creating KrasisEngine(parallel=True, num_threads=%d, skip_shared=True)...", num_threads)
            sys.stdout.flush()
            # skip_shared_experts=True: SGLang handles shared experts on GPU
            engine = KrasisEngine(parallel=True, num_threads=num_threads, skip_shared_experts=True)
            logger.info("[DIAG] KrasisEngine created OK, calling engine.load()...")
            sys.stdout.flush()

            try:
                import psutil
                mem = psutil.virtual_memory()
                logger.info(
                    "[DIAG] RAM before engine.load(): used=%.1f/%.1f GiB, available=%.1f GiB, "
                    "page_cache=%.1f GiB",
                    mem.used / (1024**3), mem.total / (1024**3),
                    mem.available / (1024**3),
                    (mem.cached + mem.buffers) / (1024**3) if hasattr(mem, 'cached') else -1,
                )
                sys.stdout.flush()
            except Exception:
                pass

            # This is the main load call — Rust loads safetensors, quantizes to INT4/INT8
            engine.load(
                model_path,
                start_layer=start_layer,
                max_layers=max_layers,
                cpu_num_bits=cls.cpu_expert_bits,
                gpu_num_bits=4,  # Marlin INT4 for GPU prefill
            )

            logger.info("[DIAG] engine.load() returned OK!")
            sys.stdout.flush()
            cls._shared_engine = engine

            logger.info(
                "Krasis engine loaded: %d MoE layers, %d experts, hidden=%d, first_k_dense=%d",
                engine.num_moe_layers(), engine.num_experts(),
                engine.hidden_size(), cls._first_k_dense,
            )

            # Log system RAM after loading
            try:
                mem = psutil.virtual_memory()
                logger.info(
                    "System RAM after Krasis load: used=%.1f/%.1f GiB (%.0f%% used), "
                    "available=%.1f GiB",
                    mem.used / (1024**3), mem.total / (1024**3),
                    mem.percent, mem.available / (1024**3))
            except ImportError:
                pass
            return engine

    @classmethod
    def _get_gpu_prefill(cls, model_path: str, device: torch.device):
        """Get or create the shared GpuPrefillManager singleton."""
        with cls._gpu_prefill_lock:
            if cls._shared_gpu_prefill is not None:
                return cls._shared_gpu_prefill

            from krasis.gpu_prefill import GpuPrefillManager

            # Read model config for GPU prefill parameters
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                raw = json.load(f)
            cfg = raw.get("text_config", raw)

            num_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 0))
            hidden_size = cfg["hidden_size"]
            intermediate_size = cfg["moe_intermediate_size"]
            n_shared = cfg.get("n_shared_experts", 0)
            routed_scale = cfg.get("routed_scaling_factor", 1.0)
            first_k = cls._first_k_dense or 0

            manager = GpuPrefillManager(
                model_path=model_path,
                device=device,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                params_dtype=torch.bfloat16,
                n_shared_experts=n_shared,
                routed_scaling_factor=routed_scale,
                first_k_dense=first_k,
                krasis_engine=cls._shared_engine,
            )
            cls._shared_gpu_prefill = manager
            logger.info("GPU prefill manager created (threshold=%d)", cls.gpu_prefill_threshold)
            return manager

    @staticmethod
    def _read_first_k_dense(model_path: str) -> int:
        """Read first_k_dense_replace from config.json."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw = json.load(f)
        cfg = raw.get("text_config", raw)

        if "first_k_dense_replace" in cfg:
            return cfg["first_k_dense_replace"]
        elif "decoder_sparse_step" in cfg:
            step = cfg["decoder_sparse_step"]
            return 0 if step <= 1 else step
        return 0

    # ── VRAM budget verification ─────────────────────────────────

    @classmethod
    def verify_vram_budget(cls, model_path: str):
        """Compare actual GPU memory usage to pre-computed VRAM budget estimate.

        Called once per rank after engine loading. Logs INFO with actual vs
        estimated, and WARNING if deviation exceeds 10%.
        """
        if cls._vram_verified:
            return
        cls._vram_verified = True

        if not torch.cuda.is_available():
            return

        try:
            from krasis.vram_budget import compute_vram_budget

            dev = torch.cuda.current_device()
            actual_bytes = torch.cuda.memory_allocated(dev)
            actual_mb = actual_bytes / (1024**2)

            # Determine PP partition and this rank's info
            partition_str = os.environ.get("SGLANG_PP_LAYER_PARTITION", "")
            if not partition_str:
                logger.info("VRAM verify: no PP partition set, skipping")
                return

            partition = [int(x.strip()) for x in partition_str.split(",")]

            # Infer rank
            rank = -1
            if cls._pp_first_layer_idx is not None:
                cumsum = 0
                for i, count in enumerate(partition):
                    if cumsum <= cls._pp_first_layer_idx < cumsum + count:
                        rank = i
                        break
                    cumsum += count

            if rank < 0:
                logger.info("VRAM verify: could not determine rank, skipping")
                return

            budget = compute_vram_budget(
                model_path=model_path,
                pp_partition=partition,
                kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "auto"),
                quantization="w8a8_int8",  # Currently always INT8
            )

            estimated_mb = budget["ranks"][rank]["weight_total_mb"]
            diff_pct = abs(actual_mb - estimated_mb) / estimated_mb * 100 if estimated_mb > 0 else 0

            if diff_pct > 10:
                logger.warning(
                    "VRAM budget: rank %d actual=%.0f MB vs estimated=%.0f MB "
                    "(%.1f%% deviation — budget may be inaccurate)",
                    rank, actual_mb, estimated_mb, diff_pct,
                )
            else:
                logger.info(
                    "VRAM budget: rank %d actual=%.0f MB vs estimated=%.0f MB "
                    "(%.1f%% deviation — OK)",
                    rank, actual_mb, estimated_mb, diff_pct,
                )

        except Exception as e:
            logger.info("VRAM verify: %s", e)

    # ── KTMoEWrapper interface ────────────────────────────────────

    def load_weights(self, physical_to_logical_map_cpu: Optional[torch.Tensor] = None):
        """Load weights into the shared engine.

        Called by SGLang's process_weights_after_loading for each layer.
        The actual load happens once (singleton). Records the first layer_idx
        seen for PP rank inference.

        Args:
            physical_to_logical_map_cpu: Expert ID remapping (ignored — Krasis
                uses original expert indices from the model).
        """
        # Record first layer index for PP rank detection
        if KrasisMoEWrapper._pp_first_layer_idx is None:
            KrasisMoEWrapper._pp_first_layer_idx = self.layer_idx
            logger.info("PP: first layer_idx on this rank = %d", self.layer_idx)

        # Log VRAM before Krasis engine load (which is CPU-only but helps track state)
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                free, total = torch.cuda.mem_get_info(dev)
                logger.info(
                    "GPU[%d] before Krasis load: allocated=%.2f GiB, free=%.2f/%.2f GiB",
                    dev, alloc, free / (1024**3), total / (1024**3))
            except Exception:
                pass

        self._engine = self._get_engine(self._model_path, self._cpuinfer_threads)

        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                free, total = torch.cuda.mem_get_info(dev)
                logger.info(
                    "GPU[%d] after Krasis load: allocated=%.2f GiB, free=%.2f/%.2f GiB",
                    dev, alloc, free / (1024**3), total / (1024**3))
            except Exception:
                pass

        # Verify VRAM usage against pre-computed budget (once per rank)
        self.verify_vram_budget(self._model_path)

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int,
    ) -> None:
        """Submit MoE forward computation.

        If batch_size >= gpu_prefill_threshold, uses GPU Marlin kernel (blocking).
        Otherwise, submits to CPU worker thread (non-blocking).

        Args:
            hidden_states: [batch_size, hidden_size] BF16 tensor on GPU
            topk_ids: [batch_size, topk] int32/int64 tensor on GPU
            topk_weights: [batch_size, topk] float32 tensor on GPU
            cuda_stream: Raw CUDA stream pointer (for DMA synchronization)
        """
        batch_size = hidden_states.shape[0]
        self._device = hidden_states.device
        self._batch_size = batch_size

        # GPU prefill path
        threshold = KrasisMoEWrapper.gpu_prefill_threshold
        if threshold > 0 and batch_size >= threshold and hidden_states.is_cuda:
            first_k = KrasisMoEWrapper._first_k_dense or 0
            moe_layer_idx = self.layer_idx - first_k

            manager = self._get_gpu_prefill(self._model_path, hidden_states.device)
            self._gpu_output = manager.forward(
                moe_layer_idx,
                hidden_states,
                topk_ids.to(torch.int32),
                topk_weights.float(),
            )
            return

        # CPU decode path
        self._gpu_output = None
        engine = self._engine or self._get_engine(self._model_path, self._cpuinfer_threads)

        # Ensure GPU writes are complete before CPU reads
        torch.cuda.current_stream(hidden_states.device).synchronize()

        # GPU → CPU copy
        act_cpu = hidden_states.detach().cpu().contiguous()
        ids_cpu = topk_ids.detach().cpu().to(torch.int32).contiguous()
        wts_cpu = topk_weights.detach().cpu().float().contiguous()

        # Mask GPU-handled experts to -1 so Krasis skips them
        if self.num_gpu_experts > 0:
            ids_cpu[ids_cpu < self.num_gpu_experts] = -1

        # Convert tensors to raw bytes for Rust engine
        # BF16 → bytes via uint16 view (numpy doesn't support bfloat16)
        act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
        ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
        wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()

        # Convert absolute layer index to MoE-relative index
        first_k = KrasisMoEWrapper._first_k_dense or 0
        moe_layer_idx = self.layer_idx - first_k

        # Remap for PP: subtract this rank's MoE layer offset
        if KrasisMoEWrapper.pp_start_moe_layer is not None:
            moe_layer_idx -= KrasisMoEWrapper.pp_start_moe_layer

        engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, batch_size)

    def sync_forward(
        self,
        hidden_states: torch.Tensor,
        cuda_stream: int,
    ) -> torch.Tensor:
        """Wait for computation and return result on GPU.

        Args:
            hidden_states: Reference tensor for shape/device/dtype info
            cuda_stream: Raw CUDA stream pointer

        Returns:
            [batch_size, hidden_size] BF16 tensor on same device as input
        """
        # GPU prefill path: result already computed
        if self._gpu_output is not None:
            result = self._gpu_output
            self._gpu_output = None
            return result

        # CPU decode path: wait for Rust engine
        engine = self._engine or self._get_engine(self._model_path, self._cpuinfer_threads)

        # Block until worker finishes and get BF16 output bytes
        output_bytes = engine.sync_forward()

        # Convert bytes → BF16 tensor
        output = torch.frombuffer(
            bytearray(output_bytes), dtype=torch.bfloat16
        ).reshape(self._batch_size, self.hidden_size)

        # CPU → GPU
        return output.to(self._device, non_blocking=True)

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Load weights from raw tensors (fallback path).

        Not needed for Krasis (loads directly from safetensors),
        but provided for interface compatibility.
        """
        logger.warning(
            "KrasisMoEWrapper.load_weights_from_tensors called but ignored — "
            "Krasis loads from HF safetensors directly"
        )

    @classmethod
    def reset(cls):
        """Reset the shared engine and GPU prefill manager (for testing)."""
        with cls._engine_lock:
            cls._shared_engine = None
            cls._first_k_dense = None
            cls._pp_first_layer_idx = None
            cls.pp_start_moe_layer = None
            cls.pp_num_moe_layers = None
        with cls._gpu_prefill_lock:
            cls._shared_gpu_prefill = None
        cls._vram_verified = False

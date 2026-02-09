"""Drop-in replacement for kt_kernel.KTMoEWrapper.

This module provides KrasisMoEWrapper which matches the interface that
SGLang's KTEPWrapperMethod expects from KTMoEWrapper. It handles:
- GPU ↔ CPU tensor transfers
- Async submit/sync pattern via Krasis Rust engine
- Expert ID masking (GPU experts = -1)

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
    """

    # ── Class-level configuration ─────────────────────────────────
    # Set before creating any wrapper instances.
    hf_model_path: Optional[str] = None
    num_threads: int = 16

    # Shared engine singleton (loaded once, used by all layer wrappers)
    _shared_engine: Optional[KrasisEngine] = None
    _engine_lock = threading.Lock()
    _first_k_dense: Optional[int] = None

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

        logger.info(
            "KrasisMoEWrapper: layer=%d, experts=%d, gpu_experts=%d, hidden=%d",
            layer_idx, num_experts, num_gpu_experts, hidden_size,
        )

    # ── Shared engine management ──────────────────────────────────

    @classmethod
    def _get_engine(cls, model_path: str, num_threads: int) -> KrasisEngine:
        """Get or create the shared KrasisEngine singleton."""
        with cls._engine_lock:
            if cls._shared_engine is not None:
                return cls._shared_engine

            logger.info("Loading Krasis engine from %s (%d threads)", model_path, num_threads)
            engine = KrasisEngine(parallel=True, num_threads=num_threads)
            engine.load(model_path)
            cls._shared_engine = engine

            # Parse first_k_dense_replace from config
            cls._first_k_dense = cls._read_first_k_dense(model_path)
            logger.info(
                "Krasis engine loaded: %d MoE layers, %d experts, hidden=%d, first_k_dense=%d",
                engine.num_moe_layers(), engine.num_experts(),
                engine.hidden_size(), cls._first_k_dense,
            )
            return engine

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

    # ── KTMoEWrapper interface ────────────────────────────────────

    def load_weights(self, physical_to_logical_map_cpu: Optional[torch.Tensor] = None):
        """Load weights into the shared engine.

        Called by SGLang's process_weights_after_loading for each layer.
        The actual load happens once (singleton).

        Args:
            physical_to_logical_map_cpu: Expert ID remapping (ignored — Krasis
                uses original expert indices from the model).
        """
        self._engine = self._get_engine(self._model_path, self._cpuinfer_threads)

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int,
    ) -> None:
        """Submit MoE forward computation to CPU worker thread (non-blocking).

        Args:
            hidden_states: [batch_size, hidden_size] BF16 tensor on GPU
            topk_ids: [batch_size, topk] int32/int64 tensor on GPU
            topk_weights: [batch_size, topk] float32 tensor on GPU
            cuda_stream: Raw CUDA stream pointer (for DMA synchronization)
        """
        engine = self._engine or self._get_engine(self._model_path, self._cpuinfer_threads)
        batch_size = hidden_states.shape[0]

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

        engine.submit_forward(moe_layer_idx, act_bytes, ids_bytes, wts_bytes, batch_size)

        # Save state for sync_forward
        self._device = hidden_states.device
        self._batch_size = batch_size

    def sync_forward(
        self,
        hidden_states: torch.Tensor,
        cuda_stream: int,
    ) -> torch.Tensor:
        """Wait for CPU computation and return result on GPU.

        Args:
            hidden_states: Reference tensor for shape/device/dtype info
            cuda_stream: Raw CUDA stream pointer

        Returns:
            [batch_size, hidden_size] BF16 tensor on same device as input
        """
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
        """Reset the shared engine (for testing)."""
        with cls._engine_lock:
            cls._shared_engine = None
            cls._first_k_dense = None

"""Full model: embedding → N transformer layers → LM head.

Single process, N GPUs (PP=N). Each GPU owns a slice of layers
plus its local KV cache. Hidden states transfer at PP boundaries.

Loading sequence:
  Phase 1: GPU weights (streaming BF16→INT8, one tensor at a time)
  Phase 2: CPU expert weights (Krasis Rust engine, INT4)
"""

import logging
import time
from typing import List, Optional

import torch

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


class KrasisModel:
    """Full model with pipeline-parallel GPU inference + CPU MoE experts."""

    def __init__(
        self,
        model_path: str,
        pp_partition: Optional[List[int]] = None,
        num_gpus: Optional[int] = None,
        devices: Optional[List[str]] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        krasis_threads: int = 16,
        gpu_prefill: bool = True,
        gpu_prefill_threshold: int = 300,
        attention_backend: str = "flashinfer",  # "flashinfer" or "trtllm"
        quant_cfg: QuantConfig = None,
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

        bits = self.quant_cfg.cpu_expert_bits
        engine = KrasisEngine(parallel=True, num_threads=self.krasis_threads)
        engine.load(self.cfg.model_path, num_bits=bits)
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
                )
                self.gpu_prefill_managers[dev_str] = manager
                logger.info("GPU prefill manager created for %s", dev_str)

        # Wire manager to MoE layers
        for layer in self.layers:
            if layer.is_moe:
                dev_str = str(layer.device)
                layer.gpu_prefill_manager = self.gpu_prefill_managers.get(dev_str)

        logger.info(
            "GPU prefill: %d managers, threshold=%d tokens",
            len(self.gpu_prefill_managers), self.gpu_prefill_threshold,
        )

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

        # Diagnostics: track first N forward calls (prefill + early decode)
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

                if diag and abs_layer_idx in (0, 1, 30, 60):
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
        logits = _linear(hidden, self.lm_head_data)
        logits = logits.float()

        if diag:
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

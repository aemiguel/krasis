"""Transformer layer: attention + MoE/dense MLP.

Each layer performs:
  1. fused_add_rmsnorm (residual + input norm)
  2. MLA attention
  3. fused_add_rmsnorm (residual + post-attention norm)
  4. MLP:
     - Dense (layer 0): gate_up → SiLU → down (INT8)
     - MoE (layers 1-60): gate → routing → shared expert (GPU) + routed experts (CPU)

For MoE layers, shared expert and routed experts overlap on GPU and CPU respectively.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import flashinfer

from krasis.config import ModelConfig
from krasis.kv_cache import PagedKVCache, SequenceKVState
from krasis.weight_loader import int8_linear

logger = logging.getLogger(__name__)

# Gate per-MoE-layer diagnostics behind env var to avoid GPU sync overhead
_DIAG_ENABLED = os.environ.get("KRASIS_DIAG", "") == "1"


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to INT8 or BF16 linear based on weight type.

    INT8 weights are stored as (weight_int8, scale) tuples.
    BF16 weights are stored as plain tensors.
    """
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)


class TransformerLayer:
    """One transformer layer: attention + MLP (dense or MoE)."""

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        weights: Dict,
        device: torch.device,
        krasis_engine=None,
        gpu_prefill_manager=None,
        gpu_prefill_threshold: int = 300,
        attention_backend: str = "flashinfer",
    ):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.device = device
        self.is_moe = weights["is_moe"]

        # Layer norms (BF16)
        self.input_norm_weight = weights["norms"]["input_layernorm"]
        self.post_attn_norm_weight = weights["norms"]["post_attention_layernorm"]

        # Attention — select backend based on attention type and backend preference
        if cfg.is_gqa:
            from krasis.attention import GQAAttention
            self.attention = GQAAttention(cfg, layer_idx, weights["attention"], device)
        elif attention_backend == "trtllm":
            from krasis.trtllm_attention import TRTLLMMLAAttention
            self.attention = TRTLLMMLAAttention(cfg, layer_idx, weights["attention"], device)
        else:
            from krasis.attention import MLAAttention
            self.attention = MLAAttention(cfg, layer_idx, weights["attention"], device)

        # MLP weights
        if self.is_moe:
            self.gate_weight = weights["gate"]["weight"]  # [n_experts, hidden]
            self.e_score_correction_bias = weights["gate"].get("e_score_correction_bias")
            self.shared_expert = weights.get("shared_expert")  # {gate_proj, up_proj, down_proj}
            self.dense_mlp = None
        else:
            self.dense_mlp = weights.get("dense_mlp")
            self.gate_weight = None
            self.shared_expert = None

        # Krasis CPU engine for routed experts
        self.krasis_engine = krasis_engine

        # GPU prefill manager for large batches (INT4 Marlin)
        self.gpu_prefill_manager = gpu_prefill_manager
        self.gpu_prefill_threshold = gpu_prefill_threshold

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_state: SequenceKVState,
        layer_offset: int,
        moe_layer_idx: Optional[int] = None,
        num_new_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one layer.

        Args:
            hidden: [M, hidden_size] BF16
            residual: [M, hidden_size] BF16 or None (first layer)
            positions: [M] int32 position indices
            kv_cache: Paged KV cache
            seq_state: Sequence KV state
            layer_offset: Index within this GPU's cache layers
            moe_layer_idx: MoE layer index for Krasis engine (0-based)
            num_new_tokens: Number of new tokens being processed (for kv_len correction)

        Returns:
            (hidden, residual) where hidden is input to next norm,
            residual is the running residual stream.
        """
        # ── Pre-attention norm ──
        if residual is None:
            residual = hidden
            hidden = flashinfer.norm.rmsnorm(
                hidden, self.input_norm_weight, self.cfg.rms_norm_eps
            )
        else:
            # fused_add_rmsnorm is IN-PLACE: residual += hidden, then hidden = rmsnorm(residual)
            flashinfer.norm.fused_add_rmsnorm(
                hidden, residual, self.input_norm_weight, self.cfg.rms_norm_eps
            )

        # ── Attention ──
        attn_out = self.attention.forward(
            hidden, positions, kv_cache, seq_state, layer_offset,
            num_new_tokens=num_new_tokens,
        )

        # ── Post-attention norm ──
        # fused_add_rmsnorm is IN-PLACE: residual += attn_out, then attn_out = rmsnorm(residual)
        flashinfer.norm.fused_add_rmsnorm(
            attn_out, residual, self.post_attn_norm_weight, self.cfg.rms_norm_eps
        )
        hidden = attn_out  # now contains normed value

        # ── MLP ──
        if self.is_moe:
            mlp_out = self._moe_forward(hidden, moe_layer_idx)
        else:
            mlp_out = self._dense_mlp_forward(hidden)

        return mlp_out, residual

    def _dense_mlp_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dense MLP: gate_proj + up_proj → SiLU(gate) * up → down_proj."""
        gate = _linear(hidden, self.dense_mlp["gate_proj"])
        up = _linear(hidden, self.dense_mlp["up_proj"])
        # SiLU(gate) * up
        activated = flashinfer.activation.silu_and_mul(
            torch.cat([gate, up], dim=-1)
        )
        del gate, up
        return _linear(activated, self.dense_mlp["down_proj"])

    def _shared_expert_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Shared expert MLP on GPU."""
        gate = _linear(hidden, self.shared_expert["gate_proj"])
        up = _linear(hidden, self.shared_expert["up_proj"])
        activated = flashinfer.activation.silu_and_mul(
            torch.cat([gate, up], dim=-1)
        )
        del gate, up
        return _linear(activated, self.shared_expert["down_proj"])

    def _moe_forward(
        self,
        hidden: torch.Tensor,
        moe_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """MoE forward: routing + shared expert (GPU) + routed experts (CPU).

        Kimi K2.5 specifics:
        - Scoring: sigmoid (not softmax)
        - e_score_correction_bias added before top-k selection
        - norm_topk_prob: divide by sum of selected weights
        - routed_scaling_factor: 2.827
        """
        M = hidden.shape[0]

        # ── Routing ──
        # gate: [M, hidden] @ [n_experts, hidden]^T → [M, n_experts]
        router_logits = torch.matmul(hidden.float(), self.gate_weight.float().t())

        # Sigmoid scoring (Kimi K2.5)
        if self.cfg.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        else:
            scores = torch.softmax(router_logits, dim=-1)

        # Add correction bias before top-k selection
        if self.e_score_correction_bias is not None:
            scores_for_selection = scores + self.e_score_correction_bias.float()
        else:
            scores_for_selection = scores

        # Top-k selection
        topk = self.cfg.num_experts_per_tok
        topk_weights, topk_ids = torch.topk(scores_for_selection, topk, dim=-1)

        # Use original scores (without bias) for the selected experts' weights
        topk_weights = scores.gather(1, topk_ids)

        # Normalize weights
        if self.cfg.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        # DIAG: per-MoE-layer logging for first N calls (gated behind KRASIS_DIAG=1)
        diag_moe = False
        if _DIAG_ENABLED:
            if not hasattr(self, '_moe_call_count'):
                self._moe_call_count = 0
            self._moe_call_count += 1
            diag_moe = self._moe_call_count <= 3  # first 3 forward passes
            if diag_moe and moe_layer_idx is not None and M == 1:
                h_rms = hidden.float().pow(2).mean().sqrt().item()
                ids_list = topk_ids[0].tolist()
                wts_list = [f"{w:.4f}" for w in topk_weights[0].tolist()]
                wt_sum = topk_weights[0].sum().item()
                logger.info(
                    "MOE-DIAG L%d call#%d: in_rms=%.4f ids=%s wts=[%s] wt_sum=%.4f",
                    moe_layer_idx, self._moe_call_count,
                    h_rms, ids_list, ",".join(wts_list), wt_sum,
                )

        # ── Dispatch: GPU prefill (large M) vs CPU decode (small M) ──
        if (
            M >= self.gpu_prefill_threshold
            and self.gpu_prefill_manager is not None
        ):
            # GPU path: INT4 Marlin kernel handles routing + shared expert
            output = self._gpu_prefill_forward(hidden, topk_ids, topk_weights, moe_layer_idx)
        else:
            # CPU path: Krasis engine handles routing + shared expert
            output = self._routed_expert_forward(hidden, topk_ids, topk_weights, moe_layer_idx)

        if _DIAG_ENABLED and diag_moe and moe_layer_idx is not None and M == 1:
            o_rms = output.float().pow(2).mean().sqrt().item()
            logger.info(
                "MOE-DIAG L%d call#%d: out_rms=%.4f",
                moe_layer_idx, self._moe_call_count, o_rms,
            )

        return output

    def _gpu_prefill_forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """GPU MoE forward using INT4 Marlin kernel (for prefill with large M).

        GpuPrefillManager handles: routing, shared expert, routed_scaling_factor.
        """
        if moe_layer_idx is None:
            raise RuntimeError("moe_layer_idx required for GPU prefill")

        return self.gpu_prefill_manager.forward(
            moe_layer_idx, hidden, topk_ids, topk_weights,
        )

    def _routed_expert_forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """Dispatch routed experts to Krasis CPU engine + shared expert on GPU.

        The Rust engine computes:
            routed_scaling_factor * Σ(w_i * expert_i(x)) + shared_expert(x)
        So the Python side just returns the Rust output directly.
        """
        if self.krasis_engine is None:
            raise RuntimeError("Krasis engine not set for MoE layer")

        M = hidden.shape[0]

        # GPU → CPU transfer (implicit sync via .cpu())
        act_cpu = hidden.detach().cpu().contiguous()
        ids_cpu = topk_ids.detach().cpu().contiguous()
        wts_cpu = topk_weights.detach().cpu().contiguous()

        # Convert to bytes for Rust engine
        act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
        ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
        wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()

        # Submit async
        self.krasis_engine.submit_forward(
            moe_layer_idx, act_bytes, ids_bytes, wts_bytes, M
        )

        # Sync — blocks until CPU experts finish
        output_bytes = self.krasis_engine.sync_forward()

        # Convert BF16 output bytes → tensor
        # Rust engine already applied: routed_scaling_factor * routed + shared_expert
        output = torch.frombuffer(
            bytearray(output_bytes), dtype=torch.bfloat16
        ).reshape(M, self.cfg.hidden_size)

        # CPU → GPU
        output = output.to(self.device)

        return output

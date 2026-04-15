"""Streaming BF16→INT8 weight loader for Krasis standalone server.

Loads non-expert weights from HF safetensors one tensor at a time,
quantizes to INT8 per-channel symmetric, and places on target GPU.
Peak transient RAM: ~50 MB (one tensor buffer at a time).

Expert weights are loaded by the Krasis Rust engine (INT4, CPU RAM).
"""

import json
import logging
import os
import struct
import time
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from krasis.config import ModelConfig, PPRankConfig, QuantConfig

logger = logging.getLogger(__name__)


def quantize_to_int8(
    weight_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight to INT8 per-channel symmetric.

    Args:
        weight_bf16: [out_features, in_features] BF16 tensor

    Returns:
        (weight_int8, scale) where:
        - weight_int8: [out_features, in_features] torch.int8
        - scale: [out_features] torch.bfloat16 (per-channel)
    """
    w = weight_bf16.float()
    # Per-channel (per-row) max absolute value
    amax = w.abs().amax(dim=1).clamp(min=1e-10)
    scale = amax / 127.0
    w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.to(torch.bfloat16)


def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A8 INT8 matmul with on-the-fly activation quantization.

    Both weights and activations are INT8, matmul accumulates in INT32 via
    torch._int_mm for maximum precision. Works well for MLP/shared_expert/lm_head
    but NOT suitable for attention projections (use attention="bf16" instead).

    Args:
        x: [..., in_features] BF16 input
        weight_int8: [out_features, in_features] INT8
        scale: [out_features] BF16 per-channel scale

    Returns:
        [..., out_features] BF16
    """
    orig_shape = x.shape[:-1]
    in_features = x.shape[-1]
    x_2d = x.reshape(-1, in_features)
    M = x_2d.shape[0]

    # Quantize activation to INT8 per-token
    x_float = x_2d.float()
    x_amax = x_float.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    x_scale = x_amax / 127.0
    x_int8 = (x_float / x_scale).round().clamp(-128, 127).to(torch.int8)

    # _int_mm requires M >= 17 and M to be a multiple of 8 on SM89
    padded_M = max(M, 17)
    padded_M = (padded_M + 7) & ~7  # round up to multiple of 8
    if padded_M != M:
        pad = padded_M - M
        x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, pad))

    # INT8 matmul: [M, K] @ [K, N] -> [M, N] INT32
    out_int32 = torch._int_mm(x_int8, weight_int8.t())

    if padded_M != M:
        out_int32 = out_int32[:M]
        x_scale = x_scale[:M]

    # Dequantize: multiply by x_scale * w_scale
    out = out_int32.float() * (x_scale * scale.float().unsqueeze(0))
    out = out.to(torch.bfloat16).reshape(*orig_shape, -1)

    if bias is not None:
        out = out + bias

    return out


class WeightLoader:
    """Streaming weight loader — loads one tensor at a time from safetensors.

    Uses safe_open with framework="pt" for random-access tensor reading.
    Each tensor is read, quantized to INT8, placed on GPU, then the CPU
    buffer is freed. Peak transient RAM: ~50 MB.
    """

    def __init__(self, cfg: ModelConfig, quant_cfg: QuantConfig = None):
        self.cfg = cfg
        self.quant_cfg = quant_cfg or QuantConfig()
        self.model_path = cfg.model_path

        # Load safetensors index
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self._weight_map: Dict[str, str] = index["weight_map"]
        self._handles: Dict[str, object] = {}

    def _get_handle(self, shard_name: str):
        """Get or open a safetensors file handle (cached)."""
        if shard_name not in self._handles:
            path = os.path.join(self.model_path, shard_name)
            self._handles[shard_name] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_name]

    def _read_tensor(self, name: str) -> torch.Tensor:
        """Read a single tensor from safetensors by name."""
        shard_name = self._weight_map[name]
        handle = self._get_handle(shard_name)
        return handle.get_tensor(name)

    def _read_and_dequant(self, name: str) -> torch.Tensor:
        """Read a tensor, dequanting FP8 to BF16 if needed.

        FP8 models (e.g. Mistral 4) store weights as float8_e4m3fn with a
        companion weight_scale_inv tensor.  Dequant: bf16 = fp8.to(bf16) * scale_inv.
        Non-FP8 tensors are returned as-is converted to BF16.
        """
        w = self._read_tensor(name)
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                        torch.float8_e5m2, torch.float8_e5m2fnuz):
            # Look for companion scale_inv tensor
            scale_name = f"{name}_scale_inv"
            if scale_name not in self._weight_map:
                # Try without .weight suffix: "foo.weight" → "foo.weight_scale_inv"
                # already handled above since name includes ".weight"
                logger.warning("FP8 tensor %s has no scale_inv — raw conversion only", name)
                return w.to(torch.bfloat16)
            scale_inv = self._read_tensor(scale_name).float()
            w = w.to(torch.float32) * scale_inv
            return w.to(torch.bfloat16)
        return w.to(torch.bfloat16)

    def _load_and_quantize(
        self, name: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a weight tensor, quantize to INT8, place on GPU.

        Returns (weight_int8, scale) both on device.
        """
        w = self._read_and_dequant(name)
        w_int8, scale = quantize_to_int8(w)
        del w
        return w_int8.to(device), scale.to(device)

    def _load_bf16(self, name: str, device: torch.device) -> torch.Tensor:
        """Read a weight tensor and place on GPU as BF16."""
        w = self._read_and_dequant(name)
        return w.to(device)



    def load_embedding(self, device: torch.device) -> torch.Tensor:
        """Load embedding table (BF16, ~2.2 GB for Kimi K2.5)."""
        # Nemotron uses "embeddings" instead of "embed_tokens"
        name = f"{self.cfg.layers_prefix}.embed_tokens.weight"
        if name not in self._weight_map:
            name = f"{self.cfg.layers_prefix}.embeddings.weight"
        logger.info("Loading embedding: %s", name)
        return self._load_bf16(name, device)

    def load_final_norm(self, device: torch.device) -> torch.Tensor:
        """Load final RMSNorm weight (BF16, tiny)."""
        # Nemotron uses "norm_f" instead of "norm"
        name = f"{self.cfg.layers_prefix}.norm.weight"
        if name not in self._weight_map:
            name = f"{self.cfg.layers_prefix}.norm_f.weight"
        logger.info("Loading final norm: %s", name)
        w = self._load_bf16(name, device)
        return self._maybe_apply_delta_rms_bias(w, name)

    def _norm_uses_delta_rms_bias(self, label: str) -> bool:
        """Return whether a norm tensor uses the Qwen delta-RMS convention.

        Qwen3-Next and Qwen3.5 mix two norm contracts:
        - outer RMS norms store deltas and apply (1 + w) * x
        - linear-attention inner norm stores direct scales and applies w * x

        Use the tensor role rather than value heuristics so the rule stays
        stable across models within the family.
        """
        delta_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        if label.endswith(delta_suffixes):
            return True
        final_norm_suffixes = (
            ".norm.weight",
            ".norm_f.weight",
        )
        if label.endswith(final_norm_suffixes) and ".layers." not in label:
            return True
        return False

    def _maybe_apply_delta_rms_bias(self, weight: torch.Tensor, label: str) -> torch.Tensor:
        """Apply +1 only to norm roles that use the delta-RMS contract."""
        if not self.cfg.norm_bias_one:
            return weight

        if self._norm_uses_delta_rms_bias(label):
            logger.debug(
                "Applying +1 RMSNorm bias for %s (delta-RMS role)",
                label,
            )
            return weight + 1.0

        logger.debug(
            "Leaving RMSNorm weight unchanged for %s (direct-scale role)",
            label,
        )
        return weight

    def load_lm_head(self, device: torch.device):
        """Load LM head weight.

        Returns (weight_int8, scale) if INT8, or plain BF16 tensor if BF16.
        """
        # lm_head location depends on model — try multiple naming conventions
        name = "lm_head.weight"
        if name not in self._weight_map:
            if self.cfg.layers_prefix != "model":
                prefix = self.cfg.layers_prefix.rsplit(".", 1)[0]
                name = f"{prefix}.lm_head.weight"
        logger.info("Loading LM head: %s (precision=%s)", name, self.quant_cfg.lm_head)
        if self.quant_cfg.lm_head == "bf16":
            return self._load_bf16(name, device)
        return self._load_and_quantize(name, device)

    def load_attention_weights(
        self, layer_idx: int, device: torch.device,
        proj_device: torch.device = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load attention weights for one layer (MLA or GQA).

        Args:
            device: Device for all weights by default (norms, biases, etc).
            proj_device: Device for large projection weights (q/k/v/o_proj).
                         Defaults to device. Pass CPU for AWQ mode where
                         projections stay on CPU for quantization, while
                         small weights (norms, biases) go directly to GPU.
        """
        if proj_device is None:
            proj_device = device
        if self.cfg.is_gqa:
            return self._load_gqa_attention(layer_idx, device, proj_device)
        return self._load_mla_attention(layer_idx, device)

    def _load_mla_attention(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load MLA attention weights for one layer.

        Handles both:
        - q_lora path (Kimi K2.5): q_a_proj + q_a_layernorm + q_b_proj
        - direct q path (V2-Lite): q_proj only

        kv_b_proj kept BF16 (split into w_kc, w_vc for quality).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        if self.cfg.has_q_lora:
            for proj in ["q_a_proj", "q_b_proj"]:
                weights[proj] = load_proj(f"{prefix}.{proj}.weight", device)
            weights["q_a_layernorm"] = self._load_bf16(f"{prefix}.q_a_layernorm.weight", device)
        else:
            weights["q_proj"] = load_proj(f"{prefix}.q_proj.weight", device)

        weights["kv_a_proj_with_mqa"] = load_proj(
            f"{prefix}.kv_a_proj_with_mqa.weight", device)
        weights["o_proj"] = load_proj(f"{prefix}.o_proj.weight", device)
        weights["kv_a_layernorm"] = self._load_bf16(f"{prefix}.kv_a_layernorm.weight", device)

        kv_b = self._load_bf16(f"{prefix}.kv_b_proj.weight", device)
        n_heads = self.cfg.num_attention_heads
        qk_nope = self.cfg.qk_nope_head_dim
        v_head = self.cfg.v_head_dim

        kv_b = kv_b.reshape(n_heads, qk_nope + v_head, self.cfg.kv_lora_rank)
        weights["w_kc"] = kv_b[:, :qk_nope, :].contiguous()
        weights["w_vc"] = kv_b[:, qk_nope:, :].contiguous()
        del kv_b

        return weights

    def _load_gqa_attention(
        self, layer_idx: int, device: torch.device,
        proj_device: torch.device = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load GQA attention weights for one layer (Qwen3, GLM-4.7).

        Loads: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
        Also loads bias tensors when present (GLM-4.7 has attention_bias=true).

        Args:
            device: Device for small weights (norms, biases) — always GPU.
            proj_device: Device for large projection weights. Defaults to device.
                         CPU when AWQ will quantize them before GPU upload.
        """
        if proj_device is None:
            proj_device = device
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[proj] = load_proj(f"{prefix}.{proj}.weight", proj_device)
            # Load bias if present (GLM-4.7) — biases are small, always on GPU
            bias_name = f"{prefix}.{proj}.bias"
            if bias_name in self._weight_map:
                weights[f"{proj}_bias"] = self._load_bf16(bias_name, device)

        # QK-Norm (Qwen3 and GLM-4.7 use RMSNorm on Q and K)
        q_norm_name = f"{prefix}.q_norm.weight"
        k_norm_name = f"{prefix}.k_norm.weight"
        if q_norm_name in self._weight_map:
            w = self._load_bf16(q_norm_name, device)
            w = self._maybe_apply_delta_rms_bias(w, q_norm_name)
            weights["q_norm"] = w
        if k_norm_name in self._weight_map:
            w = self._load_bf16(k_norm_name, device)
            w = self._maybe_apply_delta_rms_bias(w, k_norm_name)
            weights["k_norm"] = w

        # Attention sinks (GPT OSS: learnable logits for attention normalization)
        sinks_name = f"{prefix}.sinks"
        if sinks_name in self._weight_map:
            weights["sinks"] = self._load_bf16(sinks_name, device)

        return weights

    def load_layer_norms(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load input_layernorm and post_attention_layernorm (BF16)."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}"
        norms = {
            "input_layernorm": self._load_bf16(
                f"{prefix}.input_layernorm.weight", device),
            "post_attention_layernorm": self._load_bf16(
                f"{prefix}.post_attention_layernorm.weight", device),
        }
        for key in norms:
            norms[key] = self._maybe_apply_delta_rms_bias(
                norms[key],
                f"{prefix}.{key}.weight",
            )
        return norms

    def load_dense_mlp(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load dense MLP weights for a non-MoE layer.

        Uses quant_cfg.dense_mlp ("int8" or "bf16").
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp"
        load = self._load_and_quantize if self.quant_cfg.dense_mlp == "int8" else self._load_bf16
        return {
            "gate_proj": load(f"{prefix}.gate_proj.weight", device),
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

    def load_moe_gate(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load MoE router gate weight and optional biases (BF16).

        Supports both naming conventions:
        - "mlp.gate" (DeepSeek/Kimi/Qwen3)
        - "mlp.router" (GPT OSS)
        """
        # Detect naming: "gate" vs "router"
        prefix_gate = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.gate"
        prefix_router = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.router"
        if f"{prefix_gate}.weight" in self._weight_map:
            prefix = prefix_gate
        else:
            prefix = prefix_router
        result = {
            "weight": self._load_bf16(f"{prefix}.weight", device),
        }
        # Router bias (GPT OSS)
        bias_name = f"{prefix}.bias"
        if bias_name in self._weight_map:
            result["bias"] = self._load_bf16(bias_name, device)
        # e_score_correction_bias (Kimi K2.5)
        corr_name = f"{prefix}.e_score_correction_bias"
        if corr_name in self._weight_map:
            result["e_score_correction_bias"] = self._load_bf16(corr_name, device)
        return result

    def load_shared_expert(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load shared expert MLP weights for a MoE layer.

        Uses quant_cfg.shared_expert ("int8" or "bf16").
        Handles both naming conventions:
        - "shared_experts" (plural, DeepSeek/Kimi)
        - "shared_expert" (singular, Qwen3-Next)
        """
        # Detect naming convention from weight map
        prefix_plural = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_experts"
        prefix_singular = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert"
        if f"{prefix_plural}.gate_proj.weight" in self._weight_map:
            prefix = prefix_plural
        else:
            prefix = prefix_singular

        load = self._load_and_quantize if self.quant_cfg.shared_expert == "int8" else self._load_bf16
        result = {
            "gate_proj": load(f"{prefix}.gate_proj.weight", device),
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

        # Shared expert gate (Qwen3-Next): sigmoid gate on shared expert output
        # Weight: [1, hidden_size] — projects hidden → scalar per token
        gate_name = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        if gate_name in self._weight_map:
            result["shared_expert_gate"] = self._load_bf16(gate_name, device)

        return result

    def load_linear_attention_weights(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load Gated DeltaNet linear attention weights for one layer.

        Loads: in_proj_qkvz, in_proj_ba, out_proj (quantizable),
               conv1d.weight, A_log, dt_bias, norm.weight (always BF16).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.linear_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        # Quantizable projections — handle fused (QCN) or separate (Qwen3.5) format
        fused_qkvz = f"{prefix}.in_proj_qkvz.weight"
        if fused_qkvz in self._weight_map:
            # Fused format: in_proj_qkvz = [Q, K, V, Z], in_proj_ba = [B, A]
            weights["in_proj_qkvz"] = load_proj(fused_qkvz, device)
            weights["in_proj_ba"] = load_proj(f"{prefix}.in_proj_ba.weight", device)
        else:
            # Separate format (Qwen3.5): rearrange into fused interleaved format
            # QKV is flat [Q_all, K_all, V_all], Z/B/A are flat per-head
            # Must interleave per key-head group to match QCN's fused layout
            qkv_raw = self._load_bf16(f"{prefix}.in_proj_qkv.weight", device)
            z_raw = self._load_bf16(f"{prefix}.in_proj_z.weight", device)
            b_raw = self._load_bf16(f"{prefix}.in_proj_b.weight", device)
            a_raw = self._load_bf16(f"{prefix}.in_proj_a.weight", device)

            nk = self.cfg.linear_num_key_heads    # 16
            dk = self.cfg.linear_key_head_dim      # 128
            hr = self.cfg.linear_num_value_heads // nk  # 2
            dv = self.cfg.linear_value_head_dim    # 128
            key_dim = nk * dk   # 2048
            val_dim = self.cfg.linear_num_value_heads * dv  # 4096

            # Interleave QKVZ per key-head group: [q_i, k_i, v_i, z_i] for each group i
            qkvz_parts = []
            for i in range(nk):
                qkvz_parts.append(qkv_raw[i * dk : (i + 1) * dk])              # q group i
                qkvz_parts.append(qkv_raw[key_dim + i * dk : key_dim + (i + 1) * dk])  # k group i
                qkvz_parts.append(qkv_raw[key_dim * 2 + i * hr * dv : key_dim * 2 + (i + 1) * hr * dv])  # v group i
                qkvz_parts.append(z_raw[i * hr * dv : (i + 1) * hr * dv])      # z group i
            weights["in_proj_qkvz"] = torch.cat(qkvz_parts, dim=0)

            # Interleave BA per key-head group: [b_i, a_i] for each group i
            ba_parts = []
            for i in range(nk):
                ba_parts.append(b_raw[i * hr : (i + 1) * hr])
                ba_parts.append(a_raw[i * hr : (i + 1) * hr])
            weights["in_proj_ba"] = torch.cat(ba_parts, dim=0)
        weights["out_proj"] = load_proj(f"{prefix}.out_proj.weight", device)

        # Small/critical weights — always BF16
        weights["conv1d_weight"] = self._load_bf16(f"{prefix}.conv1d.weight", device)
        weights["A_log"] = self._load_bf16(f"{prefix}.A_log", device)
        weights["dt_bias"] = self._load_bf16(f"{prefix}.dt_bias", device)
        norm_weight = self._load_bf16(f"{prefix}.norm.weight", device)
        weights["norm_weight"] = self._maybe_apply_delta_rms_bias(
            norm_weight,
            f"{prefix}.norm.weight",
        )

        return weights

    def load_mamba2_weights(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load Mamba2 SSM weights for one layer (Nemotron-H).

        Loads: in_proj, out_proj (quantizable projections),
               conv1d.weight, conv1d.bias, A_log, D, dt_bias, norm.weight (always BF16/FP32).
        """
        # Nemotron-H uses "mixer" instead of "self_attn"
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer"
        weights = {}

        # Large projections (quantizable via Marlin for decode)
        weights["in_proj"] = self._load_bf16(f"{prefix}.in_proj.weight", device)
        weights["out_proj"] = self._load_bf16(f"{prefix}.out_proj.weight", device)

        # Small SSM parameters — always FP32 for numerical precision
        weights["conv1d_weight"] = self._load_bf16(f"{prefix}.conv1d.weight", device)
        conv_bias_name = f"{prefix}.conv1d.bias"
        if conv_bias_name in self._weight_map:
            weights["conv1d_bias"] = self._load_bf16(conv_bias_name, device)
        weights["A_log"] = self._load_bf16(f"{prefix}.A_log", device)
        weights["D"] = self._load_bf16(f"{prefix}.D", device)
        weights["dt_bias"] = self._load_bf16(f"{prefix}.dt_bias", device)
        weights["norm_weight"] = self._load_bf16(f"{prefix}.norm.weight", device)

        return weights

    def load_nemotron_attention_weights(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load GQA attention weights for a Nemotron-H attention layer.

        Same structure as standard GQA but uses 'mixer' prefix instead of 'self_attn'.
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer"
        weights = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[proj] = self._load_bf16(f"{prefix}.{proj}.weight", device)
        return weights

    def load_latent_moe_projections(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load LatentMoE latent projections for a Nemotron-H MoE layer.

        Returns fc1_latent_proj (hidden -> latent) and fc2_latent_proj (latent -> hidden).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer"
        return {
            "fc1_latent_proj": self._load_bf16(f"{prefix}.fc1_latent_proj.weight", device),
            "fc2_latent_proj": self._load_bf16(f"{prefix}.fc2_latent_proj.weight", device),
        }

    def load_nemotron_moe_gate(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load MoE gate for a Nemotron-H layer (uses 'mixer.gate' prefix)."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer.gate"
        result = {
            "weight": self._load_bf16(f"{prefix}.weight", device),
        }
        corr_name = f"{prefix}.e_score_correction_bias"
        if corr_name in self._weight_map:
            result["e_score_correction_bias"] = self._load_bf16(corr_name, device)
        return result

    def load_nemotron_shared_expert(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load shared expert for Nemotron-H (relu2 activation, no gate_proj)."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mixer.shared_experts"
        load = self._load_and_quantize if self.quant_cfg.shared_expert == "int8" else self._load_bf16
        return {
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

    def load_nemotron_layer_norm(
        self, layer_idx: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Load layer norm for Nemotron-H (single norm per layer, no post_attn_norm).

        Nemotron blocks have exactly one mixer (Mamba2/MoE/Attention) with pre-norm only.
        post_attention_layernorm is set to None to signal the decode loop to skip it.
        """
        name = f"{self.cfg.layers_prefix}.layers.{layer_idx}.norm.weight"
        return {
            "input_layernorm": self._load_bf16(name, device),
            "post_attention_layernorm": None,  # signal: skip post_attn_norm
        }

    def load_layer(
        self, layer_idx: int, device: torch.device,
        attn_device: torch.device = None,
    ) -> Dict[str, any]:
        """Load all GPU weights for one layer.

        Args:
            device: Device for norms, gate, shared expert, dense MLP.
            attn_device: Device for attention weights. Defaults to device.
                         Pass torch.device('cpu') to keep attention in system RAM.

        Returns a dict with:
        - "norms": {input_layernorm, post_attention_layernorm}
        - "attention" or "linear_attention" or "mamba2": attention/SSM weights
        - "layer_type": "linear_attention" | "full_attention" | "mamba2" | "moe"
        - "is_moe": bool
        """
        if attn_device is None:
            attn_device = device
        start = time.perf_counter()

        # Nemotron-H: dispatch by hybrid_override_pattern layer type
        if self.cfg.is_nemotron_h:
            return self._load_nemotron_layer(layer_idx, device, start)

        is_linear = self.cfg.is_linear_attention_layer(layer_idx)
        if is_linear:
            layer_type = "linear_attention"
        elif self.cfg.is_sliding_attention_layer(layer_idx):
            layer_type = "sliding_attention"
        else:
            layer_type = "full_attention"

        result = {
            "norms": self.load_layer_norms(layer_idx, device),
            "is_moe": self.cfg.is_moe_layer(layer_idx),
            "layer_type": layer_type,
        }

        # Load attention weights based on layer type
        # Linear attention is NOT affected by AWQ (AWQ only applies to GQA layers),
        # so linear attention always loads to the primary GPU device.
        if is_linear:
            result["linear_attention"] = self.load_linear_attention_weights(layer_idx, device)
        else:
            result["attention"] = self.load_attention_weights(
                layer_idx, device, proj_device=attn_device)

        if result["is_moe"]:
            result["gate"] = self.load_moe_gate(layer_idx, device)
            if self.cfg.n_shared_experts > 0:
                result["shared_expert"] = self.load_shared_expert(layer_idx, device)
        else:
            result["dense_mlp"] = self.load_dense_mlp(layer_idx, device)

        elapsed = time.perf_counter() - start
        alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
        logger.info(
            "Layer %d loaded in %.1fs (GPU alloc: %.0f MB, moe=%s, type=%s)",
            layer_idx, elapsed, alloc_mb, result["is_moe"], layer_type,
        )
        return result

    def _load_nemotron_layer(
        self, layer_idx: int, device: torch.device, start: float,
    ) -> Dict[str, any]:
        """Load a Nemotron-H layer based on the hybrid pattern type."""
        layer_type = self.cfg.layer_types[layer_idx]
        norms = self.load_nemotron_layer_norm(layer_idx, device)

        if layer_type == "mamba2":
            result = {
                "norms": norms,
                "is_moe": False,
                "layer_type": "mamba2",
                "mamba2": self.load_mamba2_weights(layer_idx, device),
            }
        elif layer_type == "moe":
            result = {
                "norms": norms,
                "is_moe": True,
                "layer_type": "moe",
                "gate": self.load_nemotron_moe_gate(layer_idx, device),
            }
            # LatentMoE projections: only if moe_latent_size > 0 (Ultra models)
            if self.cfg.moe_latent_size > 0:
                result["latent_proj"] = self.load_latent_moe_projections(layer_idx, device)
            if self.cfg.n_shared_experts > 0:
                result["shared_expert"] = self.load_nemotron_shared_expert(layer_idx, device)
        elif layer_type == "full_attention":
            # Nemotron attention layers are JUST attention (no MoE).
            # Each block has exactly one mixer: Mamba2, MoE, or Attention.
            result = {
                "norms": norms,
                "is_moe": False,
                "layer_type": "full_attention",
                "attention": self.load_nemotron_attention_weights(layer_idx, device),
            }
        else:
            raise ValueError(f"Unknown Nemotron layer type: {layer_type}")

        elapsed = time.perf_counter() - start
        alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
        logger.info(
            "Layer %d loaded in %.1fs (GPU alloc: %.0f MB, type=%s)",
            layer_idx, elapsed, alloc_mb, layer_type,
        )
        return result

    def close(self):
        """Close all safetensors handles."""
        self._handles.clear()

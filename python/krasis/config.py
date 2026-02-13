"""Model config parsing and PP partition for Krasis standalone server."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuantConfig:
    """Per-component quantization config for GPU weights.

    Components NOT configurable (always BF16): embedding, kv_b_proj/w_kc/w_vc,
    layernorms, gate weight. These are either too quality-critical or too small.
    """
    lm_head: str = "int8"          # "bf16" or "int8"
    attention: str = "bf16"        # "bf16" or "int8" (q_a, q_b, kv_a, o_proj)
    shared_expert: str = "int8"    # "bf16" or "int8"
    dense_mlp: str = "int8"        # "bf16" or "int8"
    gpu_expert_bits: int = 4       # 4 or 8 for Marlin kernel
    cpu_expert_bits: int = 4       # 4 or 8 for CPU expert quantization


@dataclass
class ModelConfig:
    """Parsed model configuration for MLA (Kimi/DeepSeek) and GQA (Qwen3) models."""

    model_path: str
    hidden_size: int
    intermediate_size: int       # dense MLP intermediate (layer 0)
    moe_intermediate_size: int   # per-expert intermediate
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int

    # MLA dimensions (all None for GQA models)
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # GQA dimensions (None for MLA models)
    gqa_head_dim: Optional[int] = None    # per-head dim (e.g. 128 for Qwen3)

    # Hybrid model: linear attention (Gated DeltaNet) + full attention
    full_attention_interval: int = 0   # 0 = all full attention; N = every Nth layer is full
    layer_types: Optional[List[str]] = None  # computed: ["linear_attention", ..., "full_attention", ...]
    linear_conv_kernel_dim: int = 4    # conv1d kernel size for linear attention
    linear_key_head_dim: int = 128     # per-head dim for keys in linear attention
    linear_num_key_heads: int = 16     # number of key heads in linear attention
    linear_value_head_dim: int = 128   # per-head dim for values in linear attention
    linear_num_value_heads: int = 32   # number of value heads in linear attention

    # MoE
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    n_shared_experts: int = 0
    shared_expert_intermediate_size: int = 0  # explicit shared expert size (Qwen3-Next)
    first_k_dense_replace: int = 0
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"         # "sigmoid" or "softmax"
    topk_method: str = "greedy"           # "noaux_tc"
    norm_topk_prob: bool = False

    # Norm / activation
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"              # "silu"

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = field(default_factory=dict)
    max_position_embeddings: int = 262144
    partial_rotary_factor: float = 1.0  # GLM-4.7 uses 0.5 (only half of head_dim gets RoPE)

    # Attention
    attention_bias: bool = False       # GLM-4.7 has bias on Q/K/V projections

    # Norm convention: Qwen3NextRMSNorm uses (1 + weight) * x, stored weights are ~0
    # Standard RMSNorm uses weight * x, stored weights are ~1
    norm_bias_one: bool = False  # True for qwen3_next models

    # Misc
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0

    # Tensor prefix in safetensors (auto-detected)
    layers_prefix: str = "language_model.model"

    @classmethod
    def from_model_path(cls, model_path: str) -> "ModelConfig":
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw = json.load(f)

        # Kimi K2.5 nests under text_config; other models are flat
        cfg = raw.get("text_config", raw)

        # tie_word_embeddings may be at top level
        tie = cfg.get("tie_word_embeddings",
                       raw.get("tie_word_embeddings", True))

        # Detect attention type: MLA has kv_lora_rank, GQA does not
        is_mla = "kv_lora_rank" in cfg

        # Handle first_k_dense_replace from either field or decoder_sparse_step
        if "first_k_dense_replace" in cfg:
            first_k_dense = cfg["first_k_dense_replace"]
        elif "decoder_sparse_step" in cfg:
            step = cfg["decoder_sparse_step"]
            first_k_dense = 0 if step <= 1 else step
        else:
            first_k_dense = 0

        # Hybrid model: compute layer_types from full_attention_interval
        full_attn_interval = cfg.get("full_attention_interval", 0)
        num_layers = cfg["num_hidden_layers"]
        layer_types = None
        if full_attn_interval > 0:
            layer_types = [
                "full_attention" if (i + 1) % full_attn_interval == 0
                else "linear_attention"
                for i in range(num_layers)
            ]

        # Norm convention: Qwen3NextRMSNorm uses (1 + weight) * x
        # with weight initialized to zeros, while standard models use weight * x
        # with weight initialized to ones. We add 1.0 to stored weights at load time.
        arch = cfg.get("model_type", "")
        norm_bias_one = arch in ("qwen3_next",)

        # Shared experts: n_shared_experts or infer from shared_expert_intermediate_size
        n_shared = cfg.get("n_shared_experts", 0)
        shared_inter = cfg.get("shared_expert_intermediate_size", 0)
        if n_shared == 0 and shared_inter > 0:
            n_shared = 1  # infer single shared expert

        return cls(
            model_path=model_path,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 0)),
            moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
            num_hidden_layers=num_layers,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            vocab_size=cfg["vocab_size"],
            # MLA fields (None for GQA)
            q_lora_rank=cfg.get("q_lora_rank") if is_mla else None,
            kv_lora_rank=cfg.get("kv_lora_rank") if is_mla else None,
            qk_nope_head_dim=cfg.get("qk_nope_head_dim") if is_mla else None,
            qk_rope_head_dim=cfg.get("qk_rope_head_dim") if is_mla else None,
            v_head_dim=cfg.get("v_head_dim") if is_mla else None,
            # GQA fields (None for MLA)
            gqa_head_dim=cfg.get("head_dim") if not is_mla else None,
            # Hybrid model
            full_attention_interval=full_attn_interval,
            layer_types=layer_types,
            linear_conv_kernel_dim=cfg.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=cfg.get("linear_key_head_dim", 128),
            linear_num_key_heads=cfg.get("linear_num_key_heads", 16),
            linear_value_head_dim=cfg.get("linear_value_head_dim", 128),
            linear_num_value_heads=cfg.get("linear_num_value_heads", 32),
            # MoE
            n_routed_experts=cfg.get("n_routed_experts", cfg.get("num_experts", 0)),
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0),
            n_shared_experts=n_shared,
            shared_expert_intermediate_size=shared_inter,
            first_k_dense_replace=first_k_dense,
            routed_scaling_factor=cfg.get("routed_scaling_factor", 1.0),
            scoring_func=cfg.get("scoring_func", "softmax"),
            topk_method=cfg.get("topk_method", "greedy"),
            norm_topk_prob=cfg.get("norm_topk_prob", False),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            hidden_act=cfg.get("hidden_act", "silu"),
            rope_theta=cfg.get("rope_theta", 10000.0),
            rope_scaling=cfg.get("rope_scaling") or {},
            max_position_embeddings=cfg.get("max_position_embeddings", 131072),
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            attention_bias=cfg.get("attention_bias", False),
            norm_bias_one=norm_bias_one,
            tie_word_embeddings=tie,
            bos_token_id=raw.get("bos_token_id", cfg.get("bos_token_id", 0)),
            eos_token_id=raw.get("eos_token_id", cfg.get("eos_token_id", 0)),
            layers_prefix="language_model.model" if "text_config" in raw else "model",
        )

    @property
    def attention_type(self) -> str:
        """'mla' for MLA (DeepSeek/Kimi), 'gqa' for GQA (Qwen3)."""
        return "mla" if self.kv_lora_rank is not None else "gqa"

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def is_gqa(self) -> bool:
        return self.kv_lora_rank is None

    @property
    def num_moe_layers(self) -> int:
        return self.num_hidden_layers - self.first_k_dense_replace

    @property
    def head_dim(self) -> int:
        """Full head dim for q: nope + rope (MLA) or gqa_head_dim (GQA)."""
        if self.is_mla:
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.gqa_head_dim

    @property
    def q_head_dim(self) -> int:
        """Total query head dim."""
        return self.head_dim

    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions that get RoPE (partial_rotary_factor * head_dim)."""
        if self.is_mla:
            return self.qk_rope_head_dim
        return int(self.gqa_head_dim * self.partial_rotary_factor)

    @property
    def kv_compressed_dim(self) -> int:
        """Compressed KV dimension stored in cache (MLA only)."""
        assert self.is_mla, "kv_compressed_dim only valid for MLA models"
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def has_q_lora(self) -> bool:
        """Whether this model uses q_a_proj + q_b_proj (True) or direct q_proj (False)."""
        return self.q_lora_rank is not None and self.q_lora_rank > 0

    @property
    def is_hybrid(self) -> bool:
        """True if model has a mix of linear attention and full attention layers."""
        return self.layer_types is not None

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses linear attention (Gated DeltaNet)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "linear_attention"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses standard full attention (GQA/MLA)."""
        if self.layer_types is None:
            return True
        return self.layer_types[layer_idx] == "full_attention"

    @property
    def num_full_attention_layers(self) -> int:
        """Number of layers that use full attention (need KV cache)."""
        if self.layer_types is None:
            return self.num_hidden_layers
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def effective_shared_expert_intermediate(self) -> int:
        """Shared expert intermediate size, handling both naming conventions."""
        if self.shared_expert_intermediate_size > 0:
            return self.shared_expert_intermediate_size
        if self.n_shared_experts > 0:
            return self.n_shared_experts * self.moe_intermediate_size
        return 0

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace


def compute_pp_partition(
    num_layers: int,
    num_gpus: int,
) -> List[int]:
    """Compute a balanced PP partition.

    Distributes layers as evenly as possible, with earlier ranks getting
    extra layers (they also hold embedding).
    """
    base = num_layers // num_gpus
    remainder = num_layers % num_gpus
    partition = []
    for i in range(num_gpus):
        partition.append(base + (1 if i < remainder else 0))
    return partition


@dataclass
class PPRankConfig:
    """Per-rank configuration for pipeline parallelism."""
    rank: int
    device: str                  # e.g. "cuda:0"
    layer_start: int             # first layer index (absolute)
    layer_end: int               # exclusive end
    num_layers: int
    has_embedding: bool
    has_lm_head: bool

    @property
    def layer_range(self) -> range:
        return range(self.layer_start, self.layer_end)


def build_pp_ranks(
    cfg: ModelConfig,
    pp_partition: List[int],
    devices: Optional[List[str]] = None,
) -> List[PPRankConfig]:
    """Build per-rank configs from PP partition."""
    num_ranks = len(pp_partition)
    if devices is None:
        devices = [f"cuda:{i}" for i in range(num_ranks)]

    ranks = []
    offset = 0
    for i, count in enumerate(pp_partition):
        ranks.append(PPRankConfig(
            rank=i,
            device=devices[i],
            layer_start=offset,
            layer_end=offset + count,
            num_layers=count,
            has_embedding=(i == 0),
            has_lm_head=(i == num_ranks - 1),
        ))
        offset += count
    return ranks

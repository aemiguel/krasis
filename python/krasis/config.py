"""Model config parsing and PP partition for Krasis standalone server."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


ATTENTION_QUANT_CHOICES = ("bf16", "hqq4", "hqq46", "hqq46_auto", "hqq6", "hqq68_auto", "hqq8")
DEPRECATED_ATTENTION_QUANT_CHOICES = ("awq",)
KV_CACHE_FORMAT_CHOICES = ("fp8", "fp8_e4m3", "bf16", "bfloat16", "k8v4", "k8v6", "k7v4", "k6v6", "k6v4", "k4v4", "tq4")
DEPRECATED_KV_CACHE_FORMAT_CHOICES = ("polar4",)
GPU_EXPERT_INT4_CALIB_CHOICES = ("amax", "search_rmse")
HQQ_CACHE_PROFILE_BASELINE = "baseline"
HQQ_CACHE_PROFILE_SELFCAL_V1 = "selfcal_v1"
HQQ_CACHE_PROFILE_CHOICES = (HQQ_CACHE_PROFILE_BASELINE, HQQ_CACHE_PROFILE_SELFCAL_V1)
HQQ_ATTENTION_GROUP_SIZE_CHOICES = (32, 64, 128)
HQQ_ATTENTION_DEFAULT_GROUP_SIZE = 128


def cache_dir_for_model(model_path: str) -> str:
    """Return the cache directory for a model: ~/.krasis/cache/<model_folder_name>/.

    On first call, migrates any existing .krasis_cache/ from the model directory
    to the new location, renaming GPU Marlin files to include explicit quantization.
    """
    model_name = os.path.basename(os.path.normpath(model_path))
    home = os.path.expanduser("~")
    new_dir = os.path.join(home, ".krasis", "cache", model_name)
    old_dir = os.path.join(model_path, ".krasis_cache")
    if os.path.isdir(old_dir) and not os.path.isdir(new_dir):
        os.makedirs(new_dir, exist_ok=True)
        import shutil
        for name in os.listdir(old_dir):
            # Rename GPU Marlin files to include explicit int4/int8
            new_name = name
            if name.startswith("experts_marlin_") and "int" not in name:
                if "_8b_" in name:
                    new_name = name.replace("experts_marlin_8b_", "experts_marlin_int8_")
                else:
                    new_name = name.replace("experts_marlin_", "experts_marlin_int4_")
            src = os.path.join(old_dir, name)
            dst = os.path.join(new_dir, new_name)
            shutil.move(src, dst)
        os.rmdir(old_dir)
        print(f"\033[33m▸ Migrated cache: {old_dir} → {new_dir}\033[0m", flush=True)
    return new_dir


def marlin_cache_basename(gpu_bits: int, group_size: Union[int, str], gpu_expert_int4_calib: str = "amax") -> str:
    calib_suffix = ""
    if gpu_bits == 4:
        calib_suffix = f"_cal{gpu_expert_int4_calib.replace('_', '')}"
    return f"experts_marlin_int{gpu_bits}_g{group_size}{calib_suffix}.bin"


def _collect_eos_ids(raw: dict, cfg: dict, gen_cfg: dict) -> list:
    """Collect all unique EOS token IDs from config.json and generation_config.json.

    generation_config.json is authoritative for stop tokens (often has the full
    list while config.json only has one).  Merge both, preserving order.
    """
    ids = []
    seen = set()
    # generation_config.json first (authoritative for generation)
    for source in (gen_cfg, raw, cfg):
        eos = source.get("eos_token_id")
        if eos is None:
            continue
        items = eos if isinstance(eos, list) else [eos]
        for v in items:
            if isinstance(v, int) and v not in seen:
                ids.append(v)
                seen.add(v)
    return ids if ids else [0]


def _parse_eos_token_id(raw: dict, cfg: dict, gen_cfg: dict) -> int:
    """Primary EOS token ID (first from merged list)."""
    return _collect_eos_ids(raw, cfg, gen_cfg)[0]


def _parse_extra_stop_ids(raw: dict, cfg: dict, gen_cfg: dict) -> tuple:
    """Additional stop token IDs beyond the primary EOS."""
    ids = _collect_eos_ids(raw, cfg, gen_cfg)
    return tuple(ids[1:]) if len(ids) > 1 else ()


def _infer_from_weights(model_path: str, cfg: dict) -> dict:
    """Infer missing config fields from safetensors weight shapes.

    Some VL models (DeepSeek-VL2) have incomplete language_config that's
    missing num_hidden_layers, num_attention_heads, MLA dims, etc.
    We infer these from the actual weight tensor shapes.
    """
    needed = {"num_hidden_layers", "num_attention_heads"}
    # Also need MLA dims if model has kv_a_proj (MLA) but no kv_lora_rank in config
    if all(k in cfg for k in needed) and "kv_lora_rank" in cfg:
        return cfg  # nothing missing

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return cfg

    with open(index_path) as f:
        index = json.load(f)
    wmap = index.get("weight_map", {})

    # Detect prefix from weights — find the one with attention layers (not projector/vision)
    prefix = None
    for key in wmap:
        pos = key.find(".layers.")
        if pos > 0 and "self_attn" in key:
            prefix = key[:pos]
            break
    if not prefix:
        return cfg

    # Infer num_hidden_layers by counting layer indices
    if "num_hidden_layers" not in cfg:
        layers = set()
        for k in wmap:
            if k.startswith(f"{prefix}.layers."):
                rest = k[len(prefix) + 8:]  # skip ".layers."
                try:
                    layers.add(int(rest.split(".")[0]))
                except ValueError:
                    pass
        if layers:
            cfg = dict(cfg)
            cfg["num_hidden_layers"] = max(layers) + 1

    # Infer MLA dims from weight shapes if kv_a_proj_with_mqa exists
    kv_a_key = f"{prefix}.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    if kv_a_key in wmap and "kv_lora_rank" not in cfg:
        import struct as _struct
        # Read shapes from safetensors header
        shapes = {}
        _header_cache = {}
        def _get_shape(tensor_name):
            shard = wmap.get(tensor_name)
            if not shard:
                return None
            if shard not in _header_cache:
                fpath = os.path.join(model_path, shard)
                with open(fpath, "rb") as f:
                    hlen = _struct.unpack("<Q", f.read(8))[0]
                    _header_cache[shard] = json.loads(f.read(hlen))
            info = _header_cache[shard].get(tensor_name)
            return info["shape"] if info else None

        cfg = dict(cfg)

        # kv_a_proj_with_mqa: [kv_lora_rank + qk_rope_head_dim, hidden_size]
        # kv_a_layernorm: [kv_lora_rank]
        # kv_b_proj: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # o_proj: [hidden_size, n_heads * v_head_dim]
        # q_proj: [n_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden_size]
        ln_shape = _get_shape(f"{prefix}.layers.0.self_attn.kv_a_layernorm.weight")
        kv_a_shape = _get_shape(kv_a_key)
        kv_b_shape = _get_shape(f"{prefix}.layers.0.self_attn.kv_b_proj.weight")
        o_shape = _get_shape(f"{prefix}.layers.0.self_attn.o_proj.weight")
        q_shape = _get_shape(f"{prefix}.layers.0.self_attn.q_proj.weight")

        if ln_shape and kv_a_shape and kv_b_shape and o_shape and q_shape:
            kv_lora_rank = ln_shape[0]
            qk_rope_head_dim = kv_a_shape[0] - kv_lora_rank
            hidden_size = cfg.get("hidden_size", o_shape[0])

            # o_proj: [hidden, n_heads * v_head_dim]
            total_v = o_shape[1]
            # kv_b: [n_heads * (nope + v), kv_lora_rank]
            total_kv_b = kv_b_shape[0]
            # q_proj: [n_heads * (nope + rope), hidden]
            total_q = q_shape[0]

            # Solve: n_heads * v_head_dim = total_v
            #        n_heads * (nope + v) = total_kv_b
            #        n_heads * (nope + rope) = total_q
            # From q: n_heads * nope = total_q - n_heads * rope
            # From kv_b: n_heads * nope + total_v = total_kv_b
            #   → total_q - n_heads * rope + total_v = total_kv_b
            #   → n_heads = (total_q + total_v - total_kv_b) / rope  ... doesn't simplify easily
            # Try common head dims: 128
            for v_head in (128, 64, 96, 256):
                if total_v % v_head == 0:
                    n_heads = total_v // v_head
                    nope_plus_v = total_kv_b // n_heads if total_kv_b % n_heads == 0 else 0
                    nope = nope_plus_v - v_head
                    if nope > 0 and total_q == n_heads * (nope + qk_rope_head_dim):
                        cfg.setdefault("kv_lora_rank", kv_lora_rank)
                        cfg.setdefault("qk_nope_head_dim", nope)
                        cfg.setdefault("qk_rope_head_dim", qk_rope_head_dim)
                        cfg.setdefault("v_head_dim", v_head)
                        cfg.setdefault("num_attention_heads", n_heads)
                        cfg.setdefault("num_key_value_heads", n_heads)
                        break

    return cfg


def _detect_layers_prefix(model_path: str) -> str:
    """Auto-detect the tensor prefix by scanning the safetensors index.

    Looks for '.layers.' in weight names to determine the prefix:
      - "model.layers.0..." → "model"
      - "model.language_model.layers.0..." → "model.language_model"
      - "language_model.model.layers.0..." → "language_model.model"
      - "language.model.layers.0..." → "language.model"
    Falls back to heuristic if no index file found.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        # Prefer keys with self_attn to disambiguate from projector/vision layers
        for key in index.get("weight_map", {}):
            pos = key.find(".layers.")
            if pos > 0 and "self_attn" in key:
                return key[:pos]
        # Fallback: any .layers. key
        for key in index.get("weight_map", {}):
            pos = key.find(".layers.")
            if pos > 0:
                return key[:pos]
    # Fallback: check for text_config in config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw = json.load(f)
        if "text_config" in raw:
            return "language_model.model"
    return "model"


@dataclass
class QuantConfig:
    """Per-component quantization config for GPU weights.

    Components NOT configurable (always BF16): embedding, kv_b_proj/w_kc/w_vc,
    layernorms, gate weight. These are either too quality-critical or too small.
    """
    lm_head: str = "int8"          # "bf16" or "int8" ("bf16" remains an unvalidated debug path)
    attention: str = "bf16" # "bf16", "hqq4", "hqq46", "hqq46_auto", "hqq6", "hqq68_auto", or "hqq8" (native HQQ attention; "bf16" is debug-oriented, not an oracle)
    shared_expert: str = "int8"    # "bf16" or "int8" ("bf16" remains an unvalidated debug path)
    dense_mlp: str = "int8"        # "bf16" or "int8" ("bf16" remains an unvalidated debug path)
    gpu_expert_bits: int = 4       # 4, 8 (Marlin), or 16 (UNVALIDATED BF16 debug-only path; do not use for validation)
    expert_group_size: int = 128   # routed expert quantization group size; 32 matches Q8_0-style block scale granularity
    gpu_expert_int4_calib: str = "amax"  # "amax" or "search_rmse" for routed-expert GPU INT4 cache build
    cpu_expert_bits: int = 4       # 4 or 8 for CPU expert quantization
    kv_cache_format: str = "k6v6"  # Quality default; public modes are "k6v6", "k4v4", and "bf16"
    hqq_cache_profile: str = HQQ_CACHE_PROFILE_BASELINE  # "baseline" or an explicit calibrated HQQ profile
    hqq_group_size: int = HQQ_ATTENTION_DEFAULT_GROUP_SIZE  # HQQ attention quantization group size
    hqq_auto_budget_pct: Optional[float] = None  # auto promotion budget as % of base-to-target attention span
    hqq46_auto_budget_mib: Optional[int] = None  # legacy HQQ4/6 auto promotion budget in MiB
    hqq_sidecar_manifest: Optional[str] = None  # explicit HQQ4-only sidecar manifest for switchable correction

    def __post_init__(self):
        kv_aliases = {
            "fp8": "fp8",
            "fp8_e4m3": "fp8",
            "bf16": "bf16",
            "bfloat16": "bf16",
            "k8v4": "k8v4",
            "k8v6": "k8v6",
            "k7v4": "k7v4",
            "k6v6": "k6v6",
            "k6v4": "k6v4",
            "k4v4": "k4v4",
            "tq4": "tq4",
        }
        if self.kv_cache_format in DEPRECATED_KV_CACHE_FORMAT_CHOICES:
            raise ValueError(
                f"kv_cache_format='{self.kv_cache_format}' is deprecated and disabled. "
                "Use 'k6v6' for quality, 'k4v4' for compact KV, or 'bf16' for full precision."
            )
        if self.kv_cache_format not in kv_aliases:
            raise ValueError(
                f"Unsupported kv_cache_format '{self.kv_cache_format}'. "
                "Use public modes 'k6v6', 'k4v4', or 'bf16'; internal modes include "
                "'fp8_e4m3', 'k8v4', 'k8v6', 'k7v4', 'k6v4', and 'tq4'."
            )
        self.kv_cache_format = kv_aliases[self.kv_cache_format]

        if self.attention in ("int4", "int8"):
            raise ValueError(
                f"Unsupported attention quant '{self.attention}'. "
                "Naive int4/int8 attention has been removed; use 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', 'hqq8', or 'bf16'."
            )
        if self.attention in DEPRECATED_ATTENTION_QUANT_CHOICES:
            raise ValueError(
                f"attention='{self.attention}' is deprecated and disabled. "
                "Use HQQ attention modes: 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', or 'hqq8'."
            )
        if self.attention not in ATTENTION_QUANT_CHOICES:
            raise ValueError(
                f"Unsupported attention quant '{self.attention}'. "
                f"Use one of: {', '.join(ATTENTION_QUANT_CHOICES)}."
            )
        self.hqq_cache_profile = str(self.hqq_cache_profile or HQQ_CACHE_PROFILE_BASELINE).strip().lower()
        if self.hqq_cache_profile not in HQQ_CACHE_PROFILE_CHOICES:
            raise ValueError(
                f"Unsupported hqq_cache_profile '{self.hqq_cache_profile}'. "
                f"Use one of: {', '.join(HQQ_CACHE_PROFILE_CHOICES)}."
            )
        try:
            self.hqq_group_size = int(self.hqq_group_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"hqq_group_size must be an integer, got {self.hqq_group_size!r}"
            ) from exc
        if self.hqq_group_size not in HQQ_ATTENTION_GROUP_SIZE_CHOICES:
            raise ValueError(
                f"Unsupported hqq_group_size={self.hqq_group_size}. "
                "Use 32, 64, or 128."
            )
        if not self.attention.startswith("hqq") and self.hqq_cache_profile != HQQ_CACHE_PROFILE_BASELINE:
            raise ValueError(
                f"hqq_cache_profile={self.hqq_cache_profile} requires attention='hqq4', attention='hqq46', attention='hqq46_auto', attention='hqq6', attention='hqq68_auto', or attention='hqq8'. "
                "Non-HQQ attention backends must use hqq_cache_profile='baseline'."
            )
        if not self.attention.startswith("hqq") and self.hqq_group_size != HQQ_ATTENTION_DEFAULT_GROUP_SIZE:
            raise ValueError(
                f"hqq_group_size={self.hqq_group_size} requires attention='hqq4', attention='hqq46', attention='hqq46_auto', attention='hqq6', attention='hqq68_auto', or attention='hqq8'. "
                "Non-HQQ attention backends must use the default HQQ group size."
            )
        if isinstance(self.hqq_auto_budget_pct, str) and not self.hqq_auto_budget_pct.strip():
            self.hqq_auto_budget_pct = None
        if self.hqq_auto_budget_pct is not None:
            try:
                self.hqq_auto_budget_pct = float(self.hqq_auto_budget_pct)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"hqq_auto_budget_pct must be a numeric percentage, got {self.hqq_auto_budget_pct!r}"
                ) from exc
            if self.hqq_auto_budget_pct < 0.0 or self.hqq_auto_budget_pct > 100.0:
                raise ValueError(
                    f"hqq_auto_budget_pct must satisfy 0 <= pct <= 100, got {self.hqq_auto_budget_pct!r}"
                )
        if isinstance(self.hqq46_auto_budget_mib, str) and not self.hqq46_auto_budget_mib.strip():
            self.hqq46_auto_budget_mib = None
        if self.hqq46_auto_budget_mib is not None:
            try:
                self.hqq46_auto_budget_mib = int(self.hqq46_auto_budget_mib)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"hqq46_auto_budget_mib must be an integer MiB budget, got {self.hqq46_auto_budget_mib!r}"
                ) from exc
        if self.attention == "hqq46_auto":
            if self.hqq_auto_budget_pct is None and (
                self.hqq46_auto_budget_mib is None or self.hqq46_auto_budget_mib <= 0
            ):
                raise ValueError(
                    "attention='hqq46_auto' requires hqq_auto_budget_pct. "
                    "Legacy hqq46_auto_budget_mib remains accepted only for existing configs."
                )
        elif self.attention == "hqq68_auto":
            if self.hqq_auto_budget_pct is None:
                raise ValueError(
                    "attention='hqq68_auto' requires hqq_auto_budget_pct. "
                    "Auto planner budgets are percentages of the HQQ6-to-HQQ8 promotion span."
                )
            if self.hqq46_auto_budget_mib not in (None, 0):
                raise ValueError("hqq46_auto_budget_mib is not valid with attention='hqq68_auto'.")
        elif self.hqq46_auto_budget_mib not in (None, 0):
            raise ValueError("hqq46_auto_budget_mib is only valid with attention='hqq46_auto'.")
        if self.attention not in ("hqq46_auto", "hqq68_auto") and self.hqq_auto_budget_pct is not None:
            raise ValueError("hqq_auto_budget_pct is only valid with attention='hqq46_auto' or attention='hqq68_auto'.")
        if self.hqq_sidecar_manifest is not None:
            sidecar_path = str(self.hqq_sidecar_manifest).strip()
            self.hqq_sidecar_manifest = os.path.expanduser(sidecar_path) if sidecar_path else None
        if self.hqq_sidecar_manifest is not None and self.attention != "hqq4":
            raise ValueError(
                "hqq_sidecar_manifest requires attention='hqq4'. "
                "HQQ4/6, HQQ6, HQQ6/8, and HQQ8 are clean higher-precision attention modes and do not support sidecar/self-correction."
            )
        if self.hqq_sidecar_manifest is not None and self.hqq_group_size != HQQ_ATTENTION_DEFAULT_GROUP_SIZE:
            raise ValueError(
                "hqq_sidecar_manifest currently requires hqq_group_size=128 because sidecar manifests "
                "are tied to source HQQ group boundaries."
            )
        self.gpu_expert_int4_calib = self.gpu_expert_int4_calib.lower()
        if self.gpu_expert_int4_calib not in GPU_EXPERT_INT4_CALIB_CHOICES:
            raise ValueError(
                f"Unsupported gpu_expert_int4_calib '{self.gpu_expert_int4_calib}'. "
                f"Use one of: {', '.join(GPU_EXPERT_INT4_CALIB_CHOICES)}."
            )
        try:
            self.expert_group_size = int(self.expert_group_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"expert_group_size must be an integer, got {self.expert_group_size!r}"
            ) from exc
        if self.expert_group_size not in (32, 64, 128):
            raise ValueError(
                f"Unsupported expert_group_size={self.expert_group_size}. "
                "Use 32, 64, or 128."
            )


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
    norm_topk_prob: bool = True

    # Norm / activation
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"              # "silu"

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = field(default_factory=dict)
    max_position_embeddings: int = 262144
    partial_rotary_factor: float = 1.0  # GLM-4.7 uses 0.5 (only half of head_dim gets RoPE)
    rope_interleave: bool = True  # MLA models: True means q_pe/k_pe need de-interleaving before RoPE

    # Attention
    attention_bias: bool = False       # GLM-4.7, GPT OSS have bias on Q/K/V projections
    sliding_window: int = 0            # GPT OSS: 128 tokens for sliding_attention layers

    # Pre-quantized experts
    expert_quant_method: str = ""      # "mxfp4" for GPT OSS, "" for standard BF16
    swiglu_limit: float = 0.0         # GPT OSS: 7.0 — clamp SwiGLU output to [-limit, limit]

    # Nemotron-H (hybrid Mamba2 + MoE + Attention)
    model_type: str = ""               # e.g. "nemotron_h", "qwen3_next", etc.
    hybrid_override_pattern: str = ""  # e.g. "MEMEMEM*E..." — M=Mamba2, E=MoE, *=Attention
    mamba_num_heads: int = 0           # Mamba2 SSM heads (e.g. 128 for Super, 64 for Nano)
    mamba_head_dim: int = 0            # Mamba2 SSM per-head dim (e.g. 64)
    ssm_state_size: int = 0            # Mamba2 SSM state size (e.g. 128)
    mamba_expand: int = 0              # Mamba2 expansion factor (e.g. 2)
    mamba_conv_kernel: int = 0         # Mamba2 conv1d kernel size (e.g. 4)
    mamba_n_groups: int = 1            # Mamba2 number of groups for B/C (e.g. 8)
    mamba_chunk_size: int = 128        # Mamba2 SSD chunk size (must be power of 2)
    moe_latent_size: int = 0           # LatentMoE: latent projection dim (e.g. 1024)
    moe_shared_expert_intermediate_size: int = 0  # LatentMoE shared expert intermediate
    mlp_hidden_act: str = "silu"       # MLP activation: "silu" or "relu2"

    # GQA output gating: q_proj outputs [query, gate], apply sigmoid(gate) before o_proj
    # Qwen3.5 calls this attn_output_gate, Qwen3Next uses it implicitly
    gated_attention: bool = False

    # Norm convention: Qwen3NextRMSNorm uses (1 + weight) * x, stored weights are ~0
    # Standard RMSNorm uses weight * x, stored weights are ~1
    norm_bias_one: bool = False  # True for qwen3_next models

    # Misc
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    extra_stop_token_ids: tuple = ()  # additional stop tokens (e.g. from array eos_token_id)

    # Tensor prefix in safetensors (auto-detected)
    layers_prefix: str = "language_model.model"

    @classmethod
    def from_model_path(cls, model_path: str) -> "ModelConfig":
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw = json.load(f)

        # generation_config.json has the authoritative eos_token_id list
        gen_cfg_path = os.path.join(model_path, "generation_config.json")
        gen_cfg = {}
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path) as f:
                gen_cfg = json.load(f)

        # Some models nest config: Kimi K2.5 → text_config, DeepSeek-VL2 → language_config
        cfg = raw.get("text_config", raw.get("language_config", raw))

        # Infer missing fields from weight shapes (VL models with incomplete config)
        cfg = _infer_from_weights(model_path, cfg)

        # tie_word_embeddings may be at top level; infer from weight presence if not set
        tie_default = True
        if "tie_word_embeddings" not in cfg and "tie_word_embeddings" not in raw:
            # Check if lm_head.weight exists in safetensors index — if so, not tied
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    idx_data = json.load(f)
                for k in idx_data.get("weight_map", {}):
                    if "lm_head.weight" in k:
                        tie_default = False
                        break
        tie = cfg.get("tie_word_embeddings",
                       raw.get("tie_word_embeddings", tie_default))

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

        # Model architecture type
        arch = cfg.get("model_type", "")

        # Hybrid model: compute layer_types
        full_attn_interval = cfg.get("full_attention_interval", 0)
        num_layers = cfg["num_hidden_layers"]
        layer_types = None
        hybrid_pattern = cfg.get("hybrid_override_pattern", "")
        if hybrid_pattern and arch == "nemotron_h":
            # Nemotron-H: parse M=mamba2, E=moe, *=attention from pattern
            type_map = {"M": "mamba2", "E": "moe", "*": "full_attention"}
            layer_types = [type_map.get(c, "full_attention") for c in hybrid_pattern]
            assert len(layer_types) == num_layers, (
                f"hybrid_override_pattern length {len(layer_types)} != num_hidden_layers {num_layers}")
        elif "layer_types" in cfg:
            # GPT OSS: explicit layer_types array (sliding_attention / full_attention)
            layer_types = cfg["layer_types"]
        elif full_attn_interval > 0:
            # Qwen3-Next: compute from full_attention_interval
            layer_types = [
                "full_attention" if (i + 1) % full_attn_interval == 0
                else "linear_attention"
                for i in range(num_layers)
            ]

        # Norm convention: Qwen3NextRMSNorm uses (1 + weight) * x
        # with weight initialized to zeros, while standard models use weight * x
        # with weight initialized to ones. We add 1.0 to stored weights at load time.
        norm_bias_one = arch in ("qwen3_next", "qwen3_5_moe_text")

        # Gated attention: q_proj outputs [query, gate], apply sigmoid(gate) before o_proj
        # Qwen3.5 uses explicit attn_output_gate flag; Qwen3Next always uses it
        gated_attention = cfg.get("attn_output_gate", arch in ("qwen3_next", "qwen3_5_moe_text"))

        # Nemotron-H shared expert intermediate: separate field name
        nemotron_shared_inter = cfg.get("moe_shared_expert_intermediate_size", 0)

        # Shared experts: n_shared_experts or infer from shared_expert_intermediate_size
        n_shared = cfg.get("n_shared_experts", 0)
        shared_inter = cfg.get("shared_expert_intermediate_size", nemotron_shared_inter)
        if n_shared == 0 and shared_inter > 0:
            n_shared = 1  # infer single shared expert

        # Expert count: n_routed_experts (DeepSeek) / num_experts (Qwen3) / num_local_experts (GPT OSS)
        n_experts = cfg.get("n_routed_experts",
                           cfg.get("num_experts",
                                  cfg.get("num_local_experts", 0)))
        # Experts per token: num_experts_per_tok (DeepSeek/Qwen3) / experts_per_token (GPT OSS)
        experts_per_tok = cfg.get("num_experts_per_tok",
                                 cfg.get("experts_per_token", 0))

        # MoE intermediate size: moe_intermediate_size (Qwen3) / intermediate_size (GPT OSS)
        moe_inter = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 0))

        # Sliding window (GPT OSS: 128 tokens for sliding_attention layers)
        sliding_window = cfg.get("sliding_window", 0)

        # Pre-quantized expert format (GPT OSS uses MXFP4)
        quant_config = cfg.get("quantization_config", {})
        expert_quant_method = quant_config.get("quant_method", "")

        # SwiGLU activation limit (GPT OSS clamps SwiGLU output)
        swiglu_limit = cfg.get("swiglu_limit", 0.0)

        # RoPE: some models (Qwen3.5) nest rope_theta/partial_rotary_factor inside rope_parameters
        rope_params = cfg.get("rope_parameters", {}) or {}
        rope_theta = cfg.get("rope_theta", rope_params.get("rope_theta", 10000.0))
        partial_rotary = cfg.get("partial_rotary_factor",
                                 rope_params.get("partial_rotary_factor", 1.0))

        return cls(
            model_path=model_path,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 0)),
            moe_intermediate_size=moe_inter,
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
            n_routed_experts=n_experts,
            num_experts_per_tok=experts_per_tok,
            n_shared_experts=n_shared,
            shared_expert_intermediate_size=shared_inter,
            first_k_dense_replace=first_k_dense,
            routed_scaling_factor=cfg.get("routed_scaling_factor", 1.0),
            scoring_func=cfg.get("scoring_func",
                                 "sigmoid" if arch == "nemotron_h" else "softmax"),
            topk_method=cfg.get("topk_method", "greedy"),
            norm_topk_prob=cfg.get("norm_topk_prob", True),
            rms_norm_eps=cfg.get("rms_norm_eps", cfg.get("norm_eps", cfg.get("layer_norm_epsilon", 1e-6))),
            hidden_act=cfg.get("hidden_act", "silu"),
            rope_theta=rope_theta,
            rope_scaling=cfg.get("rope_scaling") or rope_params or {},
            max_position_embeddings=cfg.get("max_position_embeddings", 131072),
            partial_rotary_factor=partial_rotary,
            rope_interleave=cfg.get("rope_interleave", True),
            attention_bias=cfg.get("attention_bias", False),
            sliding_window=sliding_window,
            expert_quant_method=expert_quant_method,
            swiglu_limit=swiglu_limit,
            # Nemotron-H fields
            model_type=arch,
            hybrid_override_pattern=hybrid_pattern,
            mamba_num_heads=cfg.get("mamba_num_heads", 0),
            mamba_head_dim=cfg.get("mamba_head_dim", 0),
            ssm_state_size=cfg.get("ssm_state_size", 0),
            mamba_expand=cfg.get("expand", 0),
            mamba_conv_kernel=cfg.get("conv_kernel", 0),
            mamba_n_groups=cfg.get("n_groups", 1),
            mamba_chunk_size=cfg.get("chunk_size", 128),
            moe_latent_size=cfg.get("moe_latent_size", 0),
            moe_shared_expert_intermediate_size=nemotron_shared_inter,
            mlp_hidden_act=cfg.get("mlp_hidden_act", cfg.get("hidden_act", "silu")),
            gated_attention=gated_attention,
            norm_bias_one=norm_bias_one,
            tie_word_embeddings=tie,
            bos_token_id=raw.get("bos_token_id", cfg.get("bos_token_id", 0)),
            eos_token_id=_parse_eos_token_id(raw, cfg, gen_cfg),
            extra_stop_token_ids=_parse_extra_stop_ids(raw, cfg, gen_cfg),
            layers_prefix=_detect_layers_prefix(model_path),
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
        if self.hybrid_override_pattern:
            return self.hybrid_override_pattern.count('E')
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
    def is_nemotron_h(self) -> bool:
        """True for Nemotron-H hybrid models (Mamba2 + MoE + Attention)."""
        return self.model_type == "nemotron_h"

    @property
    def is_hybrid(self) -> bool:
        """True if model has a mix of layer types (LA+GQA, or Mamba2+MoE+Attention)."""
        return self.layer_types is not None

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses linear attention (Gated DeltaNet)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "linear_attention"

    def is_mamba2_layer(self, layer_idx: int) -> bool:
        """True if this layer uses Mamba2 SSM (Nemotron-H)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "mamba2"

    def is_moe_only_layer(self, layer_idx: int) -> bool:
        """True if this layer is MoE-only (no attention, Nemotron-H 'E' layers)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "moe"

    def is_sliding_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses sliding window attention (GPT OSS)."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "sliding_attention"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """True if this layer uses standard full attention (GQA/MLA)."""
        if self.layer_types is None:
            return True
        return self.layer_types[layer_idx] in ("full_attention", "sliding_attention")

    @property
    def num_full_attention_layers(self) -> int:
        """Number of layers that need KV cache (full + sliding attention)."""
        if self.layer_types is None:
            return self.num_hidden_layers
        return sum(1 for t in self.layer_types if t in ("full_attention", "sliding_attention"))

    @property
    def effective_shared_expert_intermediate(self) -> int:
        """Shared expert intermediate size, handling both naming conventions."""
        if self.shared_expert_intermediate_size > 0:
            return self.shared_expert_intermediate_size
        if self.n_shared_experts > 0:
            return self.n_shared_experts * self.moe_intermediate_size
        return 0

    @property
    def mamba_d_inner(self) -> int:
        """Mamba2 inner dimension = num_heads * head_dim."""
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        """Mamba2 conv1d dimension = d_inner + 2 * n_groups * state_size."""
        return self.mamba_d_inner + 2 * self.mamba_n_groups * self.ssm_state_size

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.is_nemotron_h:
            # Nemotron: only MoE layers have experts. Attention and Mamba2 layers do not.
            return self.layer_types[layer_idx] == "moe"
        if self.n_routed_experts <= 0:
            return False
        return layer_idx >= self.first_k_dense_replace


def compute_pp_partition(
    num_layers: int,
    num_gpus: int,
) -> List[int]:
    """Compute PP partition — always PP=1 (all layers on primary GPU).

    Multi-GPU uses Expert Parallelism (EP) instead of Pipeline Parallelism.
    PP>1 (splitting layers across GPUs) is not supported as it provides
    zero parallelism (sequential pipeline with idle GPUs).
    """
    return [num_layers]


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

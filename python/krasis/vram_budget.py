"""VRAM budget calculator for Krasis + SGLang.

The user passes a context length hint. Krasis calculates the actual weight
footprint per PP rank, determines free VRAM, and allocates the LOWER of:
  (A) KV cache needed for the requested context, or
  (B) Maximum KV cache that fits minus headroom.

This way we never waste VRAM on KV cache we don't need, and always leave
room for GPU prefill workspace + temporary buffers.

Usage (CLI):
    python -m krasis.vram_budget \\
        --model-path /path/to/Kimi-K2.5 \\
        --pp-partition 20,21,20 \\
        --kv-cache-dtype fp8_e4m3 \\
        --quantization w8a8_int8 \\
        --context-length 65536

Usage (Python):
    from krasis.vram_budget import compute_vram_budget
    budget = compute_vram_budget(
        model_path="/path/to/Kimi-K2.5",
        pp_partition=[20, 21, 20],
        kv_cache_dtype="fp8_e4m3",
        quantization="w8a8_int8",
        requested_context=65536,
    )
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fixed overhead per GPU that SGLang/PyTorch/CUDA/NCCL consume beyond model
# weights. Measured empirically on RTX 2000 Ada + CUDA 12.6:
#   PP0: 1716 MB, PP1: 1743 MB, PP2: 1912 MB overhead.
# Using 2000 MB as a conservative upper bound.
SGLANG_OVERHEAD_MB = 2000


def _read_model_config(model_path: str) -> Dict[str, Any]:
    """Read and normalize model config.json."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    # Kimi K2.5 nests under text_config; Qwen3 is flat
    cfg = raw.get("text_config", raw)

    # Normalize tie_word_embeddings (may be at top level)
    if "tie_word_embeddings" not in cfg:
        cfg["tie_word_embeddings"] = raw.get("tie_word_embeddings", True)

    return cfg


def _detect_gpu_vram_bytes() -> int:
    """Auto-detect per-GPU VRAM in bytes via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0].strip())
            return mb * 1024 * 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    logger.warning("Could not detect GPU VRAM, assuming 16 GB")
    return 16 * 1024**3


def _is_mla(cfg: Dict[str, Any]) -> bool:
    return "kv_lora_rank" in cfg and cfg["kv_lora_rank"] > 0


def _weight_bytes(params: int, quantization: str) -> int:
    if quantization == "w8a8_int8":
        return params  # 1 byte per param
    else:
        return params * 2  # BF16


def _mla_attention_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    hidden = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    n_heads = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]

    q_a_params = hidden * q_lora
    q_b_params = q_lora * (n_heads * (qk_nope + qk_rope))
    kv_a_params = hidden * (kv_lora + qk_rope)
    kv_b_params = kv_lora * (n_heads * (qk_nope + v_head))
    o_params = (n_heads * v_head) * hidden

    total_params = q_a_params + q_b_params + kv_a_params + kv_b_params + o_params
    return _weight_bytes(total_params, quantization)


def _gqa_attention_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // n_heads)

    q_params = hidden * (n_heads * head_dim)
    k_params = hidden * (n_kv_heads * head_dim)
    v_params = hidden * (n_kv_heads * head_dim)
    o_params = (n_heads * head_dim) * hidden

    total_params = q_params + k_params + v_params + o_params
    return _weight_bytes(total_params, quantization)


def _dense_mlp_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    hidden = cfg["hidden_size"]
    intermediate = cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 0))
    total_params = 3 * hidden * intermediate  # gate + up + down
    return _weight_bytes(total_params, quantization)


def _gate_bytes_per_moe_layer(cfg: Dict[str, Any]) -> int:
    hidden = cfg["hidden_size"]
    n_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 0))
    return hidden * n_experts * 2  # BF16


def _shared_expert_bytes_per_moe_layer(cfg: Dict[str, Any], quantization: str) -> int:
    n_shared = cfg.get("n_shared_experts", 0)
    if n_shared == 0:
        return 0
    hidden = cfg["hidden_size"]
    intermediate = cfg.get("moe_intermediate_size", 0) * n_shared
    total_params = 3 * hidden * intermediate
    return _weight_bytes(total_params, quantization)


def _layernorm_bytes_per_layer(cfg: Dict[str, Any]) -> int:
    hidden = cfg["hidden_size"]
    return hidden * 2 * 2  # 2 norms × hidden_size × 2 bytes (BF16)


def _embedding_bytes(cfg: Dict[str, Any]) -> int:
    return cfg["vocab_size"] * cfg["hidden_size"] * 2


def _lm_head_bytes(cfg: Dict[str, Any]) -> int:
    if cfg.get("tie_word_embeddings", True):
        return 0
    return cfg["vocab_size"] * cfg["hidden_size"] * 2


def _kv_dtype_bytes(kv_cache_dtype: str) -> int:
    dtypes = {
        "fp8_e4m3": 1, "fp8_e5m2": 1, "fp8": 1,
        "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
        "auto": 2,
    }
    return dtypes.get(kv_cache_dtype, 2)


def _kv_bytes_per_token_per_layer(cfg: Dict[str, Any], kv_cache_dtype: str) -> int:
    dtype_bytes = _kv_dtype_bytes(kv_cache_dtype)
    if _is_mla(cfg):
        kv_lora = cfg["kv_lora_rank"]
        qk_rope = cfg["qk_rope_head_dim"]
        return (kv_lora + qk_rope) * dtype_bytes
    else:
        n_kv_heads = cfg["num_key_value_heads"]
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        return 2 * n_kv_heads * head_dim * dtype_bytes


def _is_hybrid(cfg: Dict[str, Any]) -> bool:
    """True if model has a mix of linear and full attention layers."""
    return cfg.get("full_attention_interval", 0) > 0


def _num_full_attention_layers(cfg: Dict[str, Any]) -> int:
    """Number of layers that use full attention (need KV cache)."""
    interval = cfg.get("full_attention_interval", 0)
    if interval <= 0:
        return cfg["num_hidden_layers"]
    return sum(
        1 for i in range(cfg["num_hidden_layers"])
        if (i + 1) % interval == 0
    )


def _linear_attention_bytes_per_layer(cfg: Dict[str, Any], quantization: str) -> int:
    """Weight bytes for one linear attention (Gated DeltaNet) layer."""
    hidden = cfg["hidden_size"]
    nk = cfg.get("linear_num_key_heads", 16)
    nv = cfg.get("linear_num_value_heads", 32)
    dk = cfg.get("linear_key_head_dim", 128)
    dv = cfg.get("linear_value_head_dim", 128)
    kernel = cfg.get("linear_conv_kernel_dim", 4)

    q_dim = nk * dk
    k_dim = nk * dk
    v_dim = nv * dv
    z_dim = nv * dv
    conv_dim = q_dim + k_dim + v_dim

    # Quantizable: in_proj_qkvz, in_proj_ba, out_proj
    qkvz_params = hidden * (q_dim + k_dim + v_dim + z_dim)
    ba_params = hidden * (nv + nv)  # beta + alpha, one per value head each
    out_params = (nv * dv) * hidden
    quantizable_params = qkvz_params + ba_params + out_params

    # Always BF16: conv1d.weight, A_log, dt_bias, norm.weight
    bf16_params = conv_dim * kernel + nv + nv + dv

    return _weight_bytes(quantizable_params, quantization) + bf16_params * 2


def _expert_bytes_per_expert(cfg: Dict[str, Any], bits: int = 4) -> int:
    """Expert buffer size for Marlin INT4/INT8 on GPU."""
    hidden = cfg["hidden_size"]
    intermediate = cfg.get("moe_intermediate_size", 0)
    total_params = 3 * hidden * intermediate
    if bits == 4:
        packed = total_params // 2
        scales = (total_params // 128) * 2
    else:  # INT8
        packed = total_params
        scales = (total_params // 128) * 2
    return packed + scales


def _component_weight_bytes(params: int, quant: str) -> int:
    """Weight bytes for per-component quant ("int8" or "bf16")."""
    if quant == "int8":
        return params  # 1 byte per param
    return params * 2  # BF16


def _cpu_expert_bytes_per_expert(cfg: Dict[str, Any], bits: int = 4) -> int:
    """CPU expert size (INT4 or INT8 with group scales)."""
    hidden = cfg["hidden_size"]
    intermediate = cfg.get("moe_intermediate_size", 0)
    total_params = 3 * hidden * intermediate
    if bits == 4:
        return total_params // 2 + (total_params // 128) * 2
    else:  # INT8
        return total_params + (total_params // 128) * 2


def _detect_total_ram_gb() -> int:
    """Auto-detect total system RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb // (1024 * 1024)
    except (FileNotFoundError, ValueError):
        pass
    return 0


def compute_launcher_budget(
    model_path: str,
    pp_partition: List[int],
    expert_divisor: int = 1,
    kv_dtype: str = "fp8_e4m3",
    gpu_expert_bits: int = 4,
    cpu_expert_bits: int = 4,
    attention_quant: str = "int8",
    shared_expert_quant: str = "int8",
    dense_mlp_quant: str = "int8",
    lm_head_quant: str = "int8",
    gpu_vram_mb: int = 0,
    total_ram_gb: int = 0,
) -> Dict[str, Any]:
    """Compute VRAM + RAM budget for the launcher TUI.

    Supports per-component quantization and expert_divisor modes.
    Returns a dict with worst-case rank breakdown for display.
    """
    import math

    cfg = _read_model_config(model_path)

    if gpu_vram_mb <= 0:
        gpu_vram_mb = _detect_gpu_vram_bytes() // (1024 * 1024)
    if total_ram_gb <= 0:
        total_ram_gb = _detect_total_ram_gb()

    num_ranks = len(pp_partition)
    total_layers = cfg["num_hidden_layers"]
    is_mla = _is_mla(cfg)
    hybrid = _is_hybrid(cfg)
    n_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 0))

    if "first_k_dense_replace" in cfg:
        first_k_dense = cfg["first_k_dense_replace"]
    elif "decoder_sparse_step" in cfg:
        step = cfg["decoder_sparse_step"]
        first_k_dense = 0 if step <= 1 else step
    else:
        first_k_dense = 0

    # Pre-compute hybrid layer types
    full_attn_interval = cfg.get("full_attention_interval", 0)

    # Linear attention weight bytes per layer (hybrid only)
    linear_attn_bpl = _linear_attention_bytes_per_layer(cfg, attention_quant) if hybrid else 0

    # Attention params per layer
    if is_mla:
        q_lora = cfg.get("q_lora_rank", 0)
        kv_lora = cfg["kv_lora_rank"]
        n_heads = cfg["num_attention_heads"]
        qk_nope = cfg["qk_nope_head_dim"]
        qk_rope = cfg["qk_rope_head_dim"]
        v_head = cfg["v_head_dim"]
        if q_lora:
            q_params = cfg["hidden_size"] * q_lora + q_lora * n_heads * (qk_nope + qk_rope)
        else:
            q_params = cfg["hidden_size"] * n_heads * (qk_nope + qk_rope)
        kv_a_params = cfg["hidden_size"] * (kv_lora + qk_rope)
        kv_b_params = kv_lora * n_heads * (qk_nope + v_head)
        o_params = n_heads * v_head * cfg["hidden_size"]
        attn_params_per_layer = q_params + kv_a_params + kv_b_params + o_params
    else:
        hidden = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        n_kv_heads = cfg.get("num_key_value_heads", n_heads)
        head_dim = cfg.get("head_dim", hidden // n_heads)
        attn_params_per_layer = (
            hidden * n_heads * head_dim +          # Q
            hidden * n_kv_heads * head_dim +        # K
            hidden * n_kv_heads * head_dim +        # V
            n_heads * head_dim * hidden             # O
        )

    # Expert buffer bytes per expert (GPU Marlin format)
    expert_buf_bytes = _expert_bytes_per_expert(cfg, gpu_expert_bits) if n_experts > 0 else 0

    # Shared expert params per MoE layer
    hidden = cfg["hidden_size"]
    n_shared = cfg.get("n_shared_experts", 0)
    shared_inter = cfg.get("shared_expert_intermediate_size", 0)
    if not shared_inter and n_shared:
        shared_inter = n_shared * cfg.get("moe_intermediate_size", 0)
    shared_params_per_moe = 3 * hidden * shared_inter if shared_inter else 0

    # Dense MLP params per dense layer
    dense_inter = cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 0))
    dense_mlp_params_per_layer = 3 * hidden * dense_inter

    # KV bytes per token per layer
    kv_b = _kv_dtype_bytes(kv_dtype)
    if is_mla:
        kv_ptl = (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * kv_b
    else:
        n_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        kv_ptl = 2 * n_kv_heads * head_dim * kv_b

    cuda_overhead = SGLANG_OVERHEAD_MB

    # Per-rank budget
    ranks = []
    total_model_layers = cfg["num_hidden_layers"]
    
    # Base model footprint: attention and norms for ALL layers, replicated on every GPU
    if hybrid:
        total_full_attn = sum(1 for i in range(total_model_layers) if (i + 1) % full_attn_interval == 0)
        total_linear_attn = total_model_layers - total_full_attn
    else:
        total_full_attn = total_model_layers
        total_linear_attn = 0

    base_attn_bytes = _component_weight_bytes(attn_params_per_layer, attention_quant) * total_full_attn
    base_attn_bytes += linear_attn_bpl * total_linear_attn
    base_norm_bytes = 2 * hidden * 2 * total_model_layers  # always BF16
    base_embed_bytes = cfg["vocab_size"] * hidden * 2      # always BF16
    
    # LM head footprint (only on last rank or replicated? Plan says "Base Model Footprint (N x weights)")
    # We replicate everything to make every GPU a fully functional compute unit.
    base_lmhead_bytes = 0
    if not cfg.get("tie_word_embeddings", True):
        base_lmhead_bytes = _component_weight_bytes(cfg["vocab_size"] * hidden, lm_head_quant)

    base_model_footprint = base_attn_bytes + base_norm_bytes + base_embed_bytes + base_lmhead_bytes

    for rank_idx in range(num_ranks):
        # PP partition still determines which expert slices this rank is RESPONSIBLE for in prefill
        # or which it holds in its HCS cache.
        rank_start = sum(pp_partition[:rank_idx])
        rank_end = rank_start + pp_partition[rank_idx]
        n_layers = pp_partition[rank_idx]

        # Expert related components (these are NOT replicated across all GPUs)
        dn = max(0, min(rank_end, first_k_dense) - rank_start)
        mn = n_layers - dn
        
        shared_bytes = _component_weight_bytes(shared_params_per_moe, shared_expert_quant) * mn if shared_params_per_moe else 0
        dense_mlp_bytes = _component_weight_bytes(dense_mlp_params_per_layer, dense_mlp_quant) * dn
        gate_bytes = hidden * n_experts * 2 * mn  # always BF16 (routing weights)

        # Expert buffers (GPU side) - these fit into the remaining space
        if mn > 0 and n_experts > 0:
            if expert_divisor == 1:
                # Persistent: all experts for all MoE layers in rank
                ebuf_bytes = expert_buf_bytes * n_experts * mn
                emode = "persistent"
            elif expert_divisor >= 2:
                # Layer-grouped: only one group at a time
                group_size = math.ceil(mn / expert_divisor)
                ebuf_bytes = expert_buf_bytes * n_experts * group_size
                emode = f"grouped({expert_divisor})"
            else:
                # Chunked (divisor=0): one layer at a time
                ebuf_bytes = expert_buf_bytes * n_experts
                emode = "chunked"
        else:
            ebuf_bytes = 0
            emode = "n/a"

        total_bytes = (
            base_model_footprint + shared_bytes + dense_mlp_bytes +
            gate_bytes +
            ebuf_bytes + cuda_overhead * 1024 * 1024
        )
        free_bytes = gpu_vram_mb * 1024 * 1024 - total_bytes
        # In EP/HCS mode, KV cache is only on GPU 0? No, plan doesn't specify.
        # But if we want 10k token prefill, we probably need KV on all?
        # Actually, attention runs on all GPUs for their subset of tokens.
        # So every GPU needs KV cache for ITS tokens.
        kv_layers_in_rank = total_full_attn  # All GPUs have all layers
        kv_per_rank = kv_ptl * kv_layers_in_rank
        kv_tokens = max(0, int(free_bytes // kv_per_rank)) if kv_per_rank > 0 and free_bytes > 0 else 0

        MB = 1024 * 1024
        ranks.append({
            "rank": rank_idx,
            "n_layers": n_layers, # this rank is still 'responsible' for this many layers in HCS/EP
            "moe_layers": mn,
            "dense_layers": dn,
            "attention_mb": base_attn_bytes / MB,
            "shared_expert_mb": shared_bytes / MB,
            "dense_mlp_mb": dense_mlp_bytes / MB,
            "expert_buffer_mb": ebuf_bytes / MB,
            "expert_mode": emode,
            "embed_lmhead_mb": (base_embed_bytes + base_lmhead_bytes) / MB,
            "norms_gates_mb": (gate_bytes + base_norm_bytes) / MB,
            "cuda_overhead_mb": cuda_overhead,
            "total_mb": total_bytes / MB,
            "free_mb": free_bytes / MB,
            "kv_tokens": kv_tokens,
        })

    # Find worst-case rank
    worst = max(ranks, key=lambda r: r["total_mb"])
    over_budget = worst["total_mb"] > gpu_vram_mb

    # CPU expert RAM
    cpu_expert_bytes_each = _cpu_expert_bytes_per_expert(cfg, cpu_expert_bits) if n_experts > 0 else 0
    total_moe_layers = total_layers - first_k_dense
    cpu_expert_gb = (cpu_expert_bytes_each * n_experts * total_moe_layers) / (1024 ** 3)

    arch = "MLA" if is_mla else "GQA"
    if hybrid:
        arch += "+DeltaNet"

    return {
        "model_type": cfg.get("model_type", "unknown"),
        "architecture": arch,
        "num_layers": total_layers,
        "n_experts": n_experts,
        "n_shared_experts": n_shared,
        "first_k_dense": first_k_dense,
        "gpu_vram_mb": gpu_vram_mb,
        "total_ram_gb": total_ram_gb,
        "ranks": ranks,
        "worst_rank": worst["rank"],
        "over_budget": over_budget,
        "cpu_expert_gb": cpu_expert_gb,
        "kv_dtype": kv_dtype,
        "hybrid": hybrid,
        "num_full_attention_layers": _num_full_attention_layers(cfg) if hybrid else total_layers,
    }


def compute_vram_budget(
    model_path: str,
    pp_partition: List[int],
    kv_cache_dtype: str = "auto",
    quantization: str = "none",
    gpu_vram_bytes: Optional[int] = None,
    headroom_mb: int = 500,
    num_gpu_experts: int = 0,
    requested_context: int = 65536,
) -> Dict[str, Any]:
    """Compute VRAM budget and recommended SGLang parameters.

    The user passes requested_context as a hint. We allocate the lower of:
      (A) KV cache for the requested context, or
      (B) Maximum KV cache that fits minus headroom.

    Args:
        model_path: Path to HF model directory.
        pp_partition: Layer counts per PP rank (e.g. [20, 21, 20]).
        kv_cache_dtype: KV cache dtype ("fp8_e4m3", "bf16", etc.).
        quantization: Non-expert weight quantization ("w8a8_int8" or "none").
        gpu_vram_bytes: Per-GPU total VRAM in bytes (auto-detected if None).
        headroom_mb: Reserved VRAM for GPU prefill workspace + temporaries.
        num_gpu_experts: Number of pinned experts on GPU (default 0).
        requested_context: Context length hint from user (tokens).
    """
    cfg = _read_model_config(model_path)

    if gpu_vram_bytes is None:
        gpu_vram_bytes = _detect_gpu_vram_bytes()

    num_ranks = len(pp_partition)
    total_layers = cfg["num_hidden_layers"]
    is_mla = _is_mla(cfg)
    hybrid = _is_hybrid(cfg)
    full_attn_interval = cfg.get("full_attention_interval", 0)

    if "first_k_dense_replace" in cfg:
        first_k_dense = cfg["first_k_dense_replace"]
    elif "decoder_sparse_step" in cfg:
        step = cfg["decoder_sparse_step"]
        first_k_dense = 0 if step <= 1 else step
    else:
        first_k_dense = 0

    max_position_embeddings = cfg.get("max_position_embeddings", 131072)
    headroom_bytes = headroom_mb * 1024 * 1024
    overhead_bytes = SGLANG_OVERHEAD_MB * 1024 * 1024

    expert_bytes = _expert_bytes_per_expert(cfg) if num_gpu_experts > 0 else 0
    linear_attn_bpl = _linear_attention_bytes_per_layer(cfg, quantization) if hybrid else 0

    # Base model footprint: attention and norms for ALL layers, replicated on every GPU
    total_model_layers = cfg["num_hidden_layers"]
    if is_mla:
        attn_per_layer = _mla_attention_bytes_per_layer(cfg, quantization)
    else:
        attn_per_layer = _gqa_attention_bytes_per_layer(cfg, quantization)

    if hybrid:
        total_full_attn = sum(1 for i in range(total_model_layers) if (i + 1) % full_attn_interval == 0)
        total_linear_attn = total_model_layers - total_full_attn
    else:
        total_full_attn = total_model_layers
        total_linear_attn = 0

    base_attn_total = attn_per_layer * total_full_attn + linear_attn_bpl * total_linear_attn
    base_norm_total = _layernorm_bytes_per_layer(cfg) * total_model_layers
    base_embed_total = _embedding_bytes(cfg)
    base_lm_head_total = _lm_head_bytes(cfg)
    
    base_model_footprint = base_attn_total + base_norm_total + base_embed_total + base_lm_head_total

    ranks = []
    for rank_idx in range(num_ranks):
        rank_start = sum(pp_partition[:rank_idx])
        rank_end = rank_start + pp_partition[rank_idx]
        num_layers = pp_partition[rank_idx]

        dense_layers_in_rank = max(0, min(rank_end, first_k_dense) - rank_start)
        moe_layers_in_rank = num_layers - dense_layers_in_rank

        dense_mlp_total = _dense_mlp_bytes_per_layer(cfg, quantization) * dense_layers_in_rank
        shared_expert_total = _shared_expert_bytes_per_moe_layer(cfg, quantization) * moe_layers_in_rank
        gate_total = _gate_bytes_per_moe_layer(cfg) * moe_layers_in_rank
        pinned_expert_total = expert_bytes * num_gpu_experts if num_gpu_experts > 0 else 0

        weight_total = (
            base_model_footprint + dense_mlp_total + shared_expert_total +
            gate_total + pinned_expert_total
        )

        # Only full attention layers need KV cache
        kv_per_token = _kv_bytes_per_token_per_layer(cfg, kv_cache_dtype) * total_full_attn

        # Free VRAM = total - weights - SGLang overhead - headroom
        free_bytes = gpu_vram_bytes - weight_total - overhead_bytes - headroom_bytes
        if free_bytes <= 0:
            max_tokens = 0
        else:
            max_tokens = free_bytes // kv_per_token if kv_per_token > 0 else 0

        ranks.append({
            "rank": rank_idx,
            "layers": f"{rank_start}-{rank_end - 1}",
            "num_layers": num_layers,
            "dense_layers": dense_layers_in_rank,
            "moe_layers": moe_layers_in_rank,
            "attention_mb": base_attn_total / (1024**2),
            "dense_mlp_mb": dense_mlp_total / (1024**2),
            "shared_expert_mb": shared_expert_total / (1024**2),
            "gate_mb": gate_total / (1024**2),
            "norm_mb": base_norm_total / (1024**2),
            "embedding_mb": base_embed_total / (1024**2),
            "lm_head_mb": base_lm_head_total / (1024**2),
            "pinned_experts_mb": pinned_expert_total / (1024**2),
            "weight_total_mb": weight_total / (1024**2),
            "weight_total_bytes": weight_total,
            "kv_bytes_per_token": kv_per_token,
            "free_for_kv_mb": max(0, free_bytes) / (1024**2),
            "max_tokens": max_tokens,
        })

    # Bottleneck rank determines max capacity
    bottleneck_rank = min(ranks, key=lambda r: r["max_tokens"])
    bottleneck_max = bottleneck_rank["max_tokens"]

    # Final context = min(user request, bottleneck capacity, model max)
    context_length = min(requested_context, bottleneck_max, max_position_embeddings)

    # Was the user request satisfied?
    if requested_context <= bottleneck_max:
        context_note = "user request fits"
    else:
        context_note = f"limited by rank {bottleneck_rank['rank']}, requested {requested_context:,d}"

    # Compute mem_fraction_static from actual weights + chosen KV allocation
    # Find the rank that needs the most VRAM (weights + its KV share)
    max_used_bytes = 0
    for r in ranks:
        kv_for_context = context_length * r["kv_bytes_per_token"]
        used = r["weight_total_bytes"] + overhead_bytes + kv_for_context
        if used > max_used_bytes:
            max_used_bytes = used
    mem_fraction = round(max_used_bytes / gpu_vram_bytes, 4)
    # Clamp to [0.1, 0.95] — never reserve more than 95%
    mem_fraction = max(0.1, min(0.95, mem_fraction))

    arch = "MLA" if is_mla else "GQA"
    if hybrid:
        arch += "+DeltaNet"

    return {
        "model_path": model_path,
        "model_type": cfg.get("model_type", "unknown"),
        "architecture": arch,
        "num_hidden_layers": total_layers,
        "first_k_dense": first_k_dense,
        "pp_partition": pp_partition,
        "gpu_vram_mb": gpu_vram_bytes / (1024**2),
        "headroom_mb": headroom_mb,
        "overhead_mb": SGLANG_OVERHEAD_MB,
        "quantization": quantization,
        "kv_cache_dtype": kv_cache_dtype,
        "max_position_embeddings": max_position_embeddings,
        "num_gpu_experts": num_gpu_experts,
        "requested_context": requested_context,
        "ranks": ranks,
        "bottleneck_rank": bottleneck_rank["rank"],
        "bottleneck_max_tokens": bottleneck_max,
        "context_length": context_length,
        "context_note": context_note,
        "mem_fraction": mem_fraction,
    }


def print_budget_summary(budget: Dict[str, Any], file=sys.stderr) -> None:
    """Print a human-readable budget summary to stderr."""
    print(f"\n=== VRAM Budget: {budget['model_type']} ({budget['architecture']}) ===", file=file)
    print(f"PP partition: {budget['pp_partition']} ({budget['num_hidden_layers']} layers)", file=file)
    print(f"Quantization: {budget['quantization']}, KV dtype: {budget['kv_cache_dtype']}", file=file)
    print(f"GPU VRAM: {budget['gpu_vram_mb']:.0f} MB", file=file)
    print(f"Headroom: {budget['headroom_mb']} MB (GPU prefill + temporaries)", file=file)
    print(f"SGLang overhead: {budget['overhead_mb']} MB (PyTorch/CUDA/NCCL)", file=file)
    print(f"Requested context: {budget['requested_context']:,d} tokens", file=file)
    if budget['num_gpu_experts'] > 0:
        print(f"GPU-pinned experts: {budget['num_gpu_experts']}", file=file)
    print(file=file)

    headers = [f"PP{r['rank']}" for r in budget["ranks"]]
    print(f"{'':>20s}  " + "  ".join(f"{h:>10s}" for h in headers), file=file)
    print(f"{'':>20s}  " + "  ".join(f"{'─'*10:>10s}" for _ in headers), file=file)

    fields = [
        ("Layers", "layers", None),
        ("Attention", "attention_mb", "MB"),
        ("Dense MLP", "dense_mlp_mb", "MB"),
        ("Shared expert", "shared_expert_mb", "MB"),
        ("Gate+norms", None, "MB"),
        ("Embedding", "embedding_mb", "MB"),
        ("LM head", "lm_head_mb", "MB"),
        ("Pinned experts", "pinned_experts_mb", "MB"),
        ("Weight total", "weight_total_mb", "MB"),
        ("Free for KV", "free_for_kv_mb", "MB"),
        ("KV/token", "kv_bytes_per_token", "B"),
        ("Max tokens", "max_tokens", None),
    ]

    for label, key, unit in fields:
        vals = []
        for r in budget["ranks"]:
            if key is None and label == "Gate+norms":
                v = r["gate_mb"] + r["norm_mb"]
            elif key is not None:
                v = r[key]
            else:
                v = r.get(key, 0)

            if isinstance(v, str):
                vals.append(f"{v:>10s}")
            elif unit == "MB":
                vals.append(f"{v:>8.0f} MB")
            elif unit == "B":
                vals.append(f"{v:>8,d} B")
            else:
                if isinstance(v, int) and v > 1000:
                    vals.append(f"{v:>10,d}")
                else:
                    vals.append(f"{v!s:>10s}")

        print(f"{label:>20s}  {'  '.join(vals)}", file=file)

    print(file=file)
    bn = budget["bottleneck_rank"]
    print(f"Bottleneck: rank {bn} (max {budget['bottleneck_max_tokens']:,d} tokens)", file=file)
    print(f"Estimated context: {budget['context_length']:,d} tokens ({budget['context_note']})", file=file)
    print(f"mem_fraction_static: {budget['mem_fraction']}", file=file)
    print(file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Compute VRAM budget for Krasis + SGLang",
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to HF model directory")
    parser.add_argument("--pp-partition", required=True,
                        help="Comma-separated layer counts per rank (e.g. 20,21,20)")
    parser.add_argument("--kv-cache-dtype", default="auto",
                        help="KV cache dtype (fp8_e4m3, bf16, etc.)")
    parser.add_argument("--quantization", default="none",
                        help="Non-expert quantization (w8a8_int8 or none)")
    parser.add_argument("--gpu-vram-mb", type=int, default=None,
                        help="Per-GPU VRAM in MB (auto-detected if omitted)")
    parser.add_argument("--headroom-mb", type=int, default=500,
                        help="Reserved VRAM for GPU prefill + temporaries (default: 500)")
    parser.add_argument("--gpu-experts", type=int, default=0,
                        help="Number of pinned experts on GPU")
    parser.add_argument("--context-length", type=int, default=65536,
                        help="Requested context length hint in tokens (default: 65536)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress summary to stderr")
    args = parser.parse_args()

    partition = [int(x.strip()) for x in args.pp_partition.split(",")]
    gpu_vram_bytes = args.gpu_vram_mb * 1024 * 1024 if args.gpu_vram_mb else None

    budget = compute_vram_budget(
        model_path=args.model_path,
        pp_partition=partition,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        gpu_vram_bytes=gpu_vram_bytes,
        headroom_mb=args.headroom_mb,
        num_gpu_experts=args.gpu_experts,
        requested_context=args.context_length,
    )

    if not args.quiet:
        print_budget_summary(budget)

    # Output JSON to stdout for script consumption
    json.dump(budget, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()

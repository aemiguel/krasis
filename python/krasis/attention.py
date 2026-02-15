"""Attention implementations: MLA (DeepSeek/Kimi) and GQA (Qwen3).

MLA (Multi-head Latent Attention) flow:
  1. Compressed KV projection: hidden → [kv_compressed, k_pe]
  2. LayerNorm on q_compressed and kv_compressed
  3. q_b_proj: q_compressed → q_nope + q_pe (per-head)
  4. RoPE on q_pe and k_pe
  5. Absorb w_kc into query: q_nope_absorbed = q_nope @ w_kc
  6. Append kv to paged cache
  7. FlashInfer MLA attention
  8. Post-attention: output @ w_vc^T → o_proj

GQA (Grouped Query Attention) flow:
  1. Q/K/V projections
  2. QKNorm (per-head RMSNorm on Q and K)
  3. RoPE on all Q and K heads
  4. Append K, V to paged cache
  5. FlashInfer standard paged attention
  6. O projection
"""

import logging
import math
from typing import Optional, Tuple

import torch
import flashinfer

from krasis.config import ModelConfig
from krasis.kv_cache import PagedKVCache, SequenceKVState
from krasis.weight_loader import int8_linear

logger = logging.getLogger(__name__)


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to INT8 or BF16 linear based on weight type."""
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)


class MLAAttention:
    """Multi-head Latent Attention for one transformer layer."""

    # Shared workspace buffer per device (created once, shared across all layers)
    _workspace_bufs: dict = {}

    @classmethod
    def _get_workspace(cls, device: torch.device) -> torch.Tensor:
        """Get or create a shared 128MB FlashInfer workspace for this device."""
        key = str(device)
        if key not in cls._workspace_bufs:
            cls._workspace_bufs[key] = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        return cls._workspace_bufs[key]

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        weights: dict,
        device: torch.device,
    ):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.device = device
        self.num_heads = cfg.num_attention_heads
        self.qk_nope_dim = cfg.qk_nope_head_dim  # 128
        self.qk_rope_dim = cfg.qk_rope_head_dim   # 64
        self.v_head_dim = cfg.v_head_dim           # 128
        self.kv_lora_rank = cfg.kv_lora_rank       # 512
        self.q_lora_rank = cfg.q_lora_rank
        self.has_q_lora = cfg.has_q_lora
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim  # 192

        # Scale factor: 1/sqrt(head_dim) * mscale^2 (YaRN attention scaling)
        # When q_head_dim != v_head_dim (always true for MLA), apply yarn mscale squared
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        rope_cfg = cfg.rope_scaling
        if rope_cfg:
            factor = rope_cfg.get("factor", 1.0)
            if factor > 1.0:
                mscale_all_dim = rope_cfg.get("mscale_all_dim", 0)
                mscale = 0.1 * mscale_all_dim * math.log(factor) + 1.0
                self.sm_scale *= mscale * mscale

        # INT8 weights: (weight_int8, scale)
        if self.has_q_lora:
            self.q_a_proj = weights["q_a_proj"]       # [q_lora_rank, hidden]
            self.q_b_proj = weights["q_b_proj"]       # [heads*(nope+rope), q_lora_rank]
            self.q_a_norm_weight = weights["q_a_layernorm"]  # [q_lora_rank]
        else:
            self.q_proj = weights["q_proj"]           # [heads*(nope+rope), hidden]

        self.kv_a_proj = weights["kv_a_proj_with_mqa"] # [kv_lora_rank+rope, hidden]
        self.o_proj = weights["o_proj"]               # [hidden, heads*v_head]

        # BF16 layernorms
        self.kv_a_norm_weight = weights["kv_a_layernorm"] # [kv_lora_rank]

        # BF16 kv_b_proj split weights (kept full precision for quality)
        self.w_kc = weights["w_kc"]  # [heads, qk_nope, kv_lora_rank]
        self.w_vc = weights["w_vc"]  # [heads, v_head, kv_lora_rank]

        # RoPE parameters
        self.rope_theta = cfg.rope_theta
        self._rope_cos_sin = None

        # FlashInfer attention wrapper (shared workspace across layers on same device)
        workspace_buf = MLAAttention._get_workspace(device)
        self._attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace_buf,
        )

    def _get_rope_cos_sin(self, max_len: int):
        """Compute or retrieve cached RoPE cos/sin tables."""
        if self._rope_cos_sin is not None and self._rope_cos_sin[0].shape[0] >= max_len:
            return self._rope_cos_sin

        # YaRN-adjusted theta
        rope_cfg = self.cfg.rope_scaling
        factor = rope_cfg.get("factor", 1.0)
        original_max = rope_cfg.get("original_max_position_embeddings", 4096)

        dim = self.qk_rope_dim
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=self.device).float() / dim))

        # YaRN frequency interpolation (matching HF DeepseekV2YarnRotaryEmbedding)
        if factor > 1.0:
            beta_fast = rope_cfg.get("beta_fast", 32.0)
            beta_slow = rope_cfg.get("beta_slow", 1.0)

            low = math.floor(dim * math.log(original_max / (beta_fast * 2 * math.pi)) /
                             (2 * math.log(self.rope_theta)))
            high = math.ceil(dim * math.log(original_max / (beta_slow * 2 * math.pi)) /
                             (2 * math.log(self.rope_theta)))
            low = max(low, 0)
            high = min(high, dim // 2 - 1)

            # HF convention:
            #   i < low  → keep original (high-freq, fast rotation, extrapolated)
            #   i > high → interpolate (low-freq, slow rotation, divided by factor)
            #   between  → smooth blend from original to interpolated
            freq_extra = freqs.clone()  # original frequencies
            freq_inter = freqs / factor  # interpolated frequencies
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=self.device).float() - low) / max(high - low, 0.001),
                0, 1,
            )
            inv_freq_mask = 1.0 - ramp  # 1 for i<low (original), 0 for i>high (interpolated)
            freqs = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = torch.arange(max_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # [max_len, dim//2]
        cos = freqs.cos().to(torch.bfloat16)
        sin = freqs.sin().to(torch.bfloat16)
        self._rope_cos_sin = (cos, sin)
        return cos, sin

    @staticmethod
    def _deinterleave(x: torch.Tensor) -> torch.Tensor:
        """De-interleave from [re0, im0, re1, im1, ...] to [re0, re1, ..., im0, im1, ...].

        HF DeepSeek V2 stores q_pe/k_pe in interleaved format from the projection weights.
        Before applying RoPE (which operates on half-split format), we must de-interleave.
        """
        d = x.shape[-1]
        return x.view(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(x.shape)

    def _apply_rope(
        self,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to q_pe and k_pe.

        Args:
            q_pe: [M, num_heads, qk_rope_dim]
            k_pe: [M, 1, qk_rope_dim] (shared across heads for MLA)
            positions: [M] position indices
        """
        # De-interleave: HF weights store rope dims as [re0, im0, re1, im1, ...]
        # RoPE rotation requires half-split: [re0, re1, ..., im0, im1, ...]
        q_pe = self._deinterleave(q_pe)
        k_pe = self._deinterleave(k_pe)

        max_pos = positions.max().item() + 1
        cos, sin = self._get_rope_cos_sin(max_pos)

        # Gather cos/sin for these positions
        pos_cos = cos[positions]  # [M, dim//2]
        pos_sin = sin[positions]  # [M, dim//2]

        def rotate(x, c, s):
            # x: [..., dim], c/s: [M, dim//2]
            d2 = x.shape[-1] // 2
            # Reshape cos/sin to broadcast with x
            while c.dim() < x.dim():
                c = c.unsqueeze(1)
                s = s.unsqueeze(1)
            x1, x2 = x[..., :d2], x[..., d2:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

        q_pe = rotate(q_pe, pos_cos, pos_sin)
        k_pe = rotate(k_pe, pos_cos, pos_sin)
        return q_pe, k_pe

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_state: SequenceKVState,
        layer_offset: int,
        num_new_tokens: int = 0,
    ) -> torch.Tensor:
        """Forward pass for one layer's MLA attention.

        Args:
            hidden: [M, hidden_size] BF16
            positions: [M] int32 position indices
            kv_cache: The paged KV cache
            seq_state: Sequence state (page allocation, seq_len)
            layer_offset: This layer's index within its GPU's cache
            num_new_tokens: Number of new tokens being processed

        Returns:
            [M, hidden_size] BF16 attention output
        """
        M = hidden.shape[0]
        H = self.num_heads

        # ── Step 1: KV compressed projection (common to both paths) ──
        # kv_a_proj_with_mqa: [M, hidden] → [M, kv_lora_rank + qk_rope_dim]
        kv_out = _linear(hidden, self.kv_a_proj)
        kv_compressed = kv_out[:, :self.kv_lora_rank]     # [M, 512]
        k_pe = kv_out[:, self.kv_lora_rank:]              # [M, 64]

        # ── Step 2: KV LayerNorm ──
        kv_compressed = flashinfer.norm.rmsnorm(
            kv_compressed, self.kv_a_norm_weight, self.cfg.rms_norm_eps
        )

        # ── Step 3: Query projection (two paths) ──
        if self.has_q_lora:
            # Kimi K2.5: q_a_proj → layernorm → q_b_proj
            q_compressed = _linear(hidden, self.q_a_proj)
            q_compressed = flashinfer.norm.rmsnorm(
                q_compressed, self.q_a_norm_weight, self.cfg.rms_norm_eps
            )
            q_full = _linear(q_compressed, self.q_b_proj)
            del q_compressed
        else:
            # V2-Lite: direct q_proj
            q_full = _linear(hidden, self.q_proj)

        # Reshape to [M, heads, nope + rope]
        q_full = q_full.reshape(M, H, self.head_dim)
        q_nope = q_full[:, :, :self.qk_nope_dim]   # [M, H, 128]
        q_pe = q_full[:, :, self.qk_nope_dim:]      # [M, H, 64]
        del q_full

        # ── Step 4: RoPE ──
        k_pe_heads = k_pe.unsqueeze(1)  # [M, 1, 64] — shared across heads
        q_pe, k_pe_heads = self._apply_rope(q_pe, k_pe_heads, positions)
        k_pe = k_pe_heads.squeeze(1)    # [M, 64]

        # ── Step 5: Absorb w_kc into query ──
        # q_nope: [M, H, 128] @ w_kc: [H, 128, 512] → [M, H, 512]
        q_nope_absorbed = torch.einsum("mhi,hid->mhd", q_nope.float(), self.w_kc.float()).to(torch.bfloat16)
        del q_nope

        # ── Step 6: Append to KV cache ──
        # NOTE: ensure_capacity() is called by model.forward once per rank.
        # advance() is called AFTER all layers, so seq_len is NOT yet updated.
        ckv_layer, kpe_layer = kv_cache.get_layer_caches(layer_offset)

        kv_indices = seq_state.kv_indices(self.device)
        kv_indptr = seq_state.kv_indptr(self.device)

        # append positions: [M] batch_indices all 0 (single sequence)
        batch_indices = torch.zeros(M, dtype=torch.int32, device=self.device)
        # positions within the page/sequence
        append_positions = positions.to(torch.int32)

        # last_page_len BEFORE append (seq_len not yet advanced)
        last_page_len_tensor = torch.tensor(
            [seq_state.last_page_len() if seq_state.seq_len > 0 else 0],
            dtype=torch.int32, device=self.device,
        )

        # Cast to cache dtype for append
        ckv_append = kv_compressed.to(kv_cache.kv_dtype)  # [M, 512]
        kpe_append = k_pe.to(kv_cache.kv_dtype)           # [M, 64]

        flashinfer.page.append_paged_mla_kv_cache(
            ckv_append, kpe_append,
            batch_indices, append_positions,
            ckv_layer, kpe_layer,
            kv_indices, kv_indptr, last_page_len_tensor,
        )

        # ── Step 7: FlashInfer MLA attention ──
        kv_indices = seq_state.kv_indices(self.device)
        kv_indptr = seq_state.kv_indptr(self.device)
        # kv_len must include the newly appended tokens (seq_len not yet advanced)
        effective_kv_len = seq_state.seq_len + num_new_tokens
        kv_len_arr = torch.tensor([effective_kv_len], dtype=torch.int32, device=self.device)

        # qo_indptr: [0, M] for single sequence
        qo_indptr = torch.tensor([0, M], dtype=torch.int32, device=self.device)

        # FP8 KV cache: MLA kernel requires 16-bit floats. Upcast only the
        # pages actually in use (via kv_indices) instead of the entire cache.
        if ckv_layer.dtype != torch.bfloat16:
            used_pages = kv_indices  # [num_used_pages]
            attn_ckv = ckv_layer[used_pages].to(torch.bfloat16)
            attn_kpe = kpe_layer[used_pages].to(torch.bfloat16)
            # Remap page indices to compact range [0, N)
            compact_kv_indices = torch.arange(
                len(used_pages), dtype=torch.int32, device=self.device
            )
        else:
            attn_ckv = ckv_layer
            attn_kpe = kpe_layer
            compact_kv_indices = kv_indices

        self._attn_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=compact_kv_indices,
            kv_len_arr=kv_len_arr,
            num_heads=self.num_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_dim,
            page_size=kv_cache.page_size,
            causal=True,
            sm_scale=self.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        # attn_out: [M, H, kv_lora_rank=512]
        attn_out = self._attn_wrapper.run(
            q_nope=q_nope_absorbed,
            q_pe=q_pe,
            ckv_cache=attn_ckv,
            kpe_cache=attn_kpe,
        )

        # ── Step 8: Post-attention projection ──
        # attn_out [M, H, 512] @ w_vc^T [H, 512, 128] → [M, H, 128]
        attn_projected = torch.einsum(
            "mhd,hod->mho", attn_out.float(), self.w_vc.float()
        ).to(torch.bfloat16)

        # Reshape to [M, H*v_head_dim] = [M, 8192]
        attn_flat = attn_projected.reshape(M, H * self.v_head_dim)

        # o_proj: [M, 8192] → [M, 7168]
        output = _linear(attn_flat, self.o_proj)

        return output


class GQAAttention:
    """Grouped Query Attention for one transformer layer (Qwen3)."""

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        weights: dict,
        device: torch.device,
    ):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.device = device
        self.num_heads = cfg.num_attention_heads     # 64 for Qwen3
        self.num_kv_heads = cfg.num_key_value_heads  # 4 for Qwen3
        self.head_dim = cfg.gqa_head_dim             # 128 for Qwen3
        self.num_groups = self.num_heads // self.num_kv_heads  # 16 for Qwen3

        # Scale factor: 1/sqrt(head_dim)
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        # Detect gated attention: q_proj outputs 2x (query + gate)
        # Qwen3-Coder-Next: q_proj is [num_heads * head_dim * 2, hidden]
        q_weight = weights["q_proj"]
        q_out_dim = q_weight[0].shape[0] if isinstance(q_weight, tuple) else q_weight.shape[0]
        expected_q_dim = self.num_heads * self.head_dim
        self.gated_attention = (q_out_dim == 2 * expected_q_dim)
        if self.gated_attention:
            logger.info("Layer %d: gated attention detected (q_proj dim=%d, expected=%d)",
                        layer_idx, q_out_dim, expected_q_dim)

        # Projections (INT8 or BF16)
        self.q_proj = weights["q_proj"]  # [num_heads * head_dim (* 2 if gated), hidden]
        self.k_proj = weights["k_proj"]  # [num_kv_heads * head_dim, hidden]
        self.v_proj = weights["v_proj"]  # [num_kv_heads * head_dim, hidden]
        self.o_proj = weights["o_proj"]  # [hidden, num_heads * head_dim]

        # Attention biases (optional, GLM-4.7 / GPT OSS have these)
        self.q_proj_bias = weights.get("q_proj_bias")  # [num_heads * head_dim]
        self.k_proj_bias = weights.get("k_proj_bias")  # [num_kv_heads * head_dim]
        self.v_proj_bias = weights.get("v_proj_bias")  # [num_kv_heads * head_dim]
        self.o_proj_bias = weights.get("o_proj_bias")  # [hidden]

        # Attention sinks (GPT OSS: per-head learnable logit for softmax normalization)
        self.sinks = weights.get("sinks")  # [num_heads]

        # Sliding window (GPT OSS: 128 for sliding_attention layers)
        self.sliding_window = None
        if cfg.is_sliding_attention_layer(layer_idx):
            self.sliding_window = cfg.sliding_window

        # QKNorm (per-head RMSNorm, optional)
        self.q_norm = weights.get("q_norm")  # [head_dim]
        self.k_norm = weights.get("k_norm")  # [head_dim]

        # RoPE parameters
        self.rope_theta = cfg.rope_theta
        self.rotary_dim = cfg.rotary_dim  # may be < head_dim (GLM-4.7: 64 of 128)
        self._rope_cos_sin = None

        # FlashInfer attention wrapper (shared workspace across layers on same device)
        workspace_buf = MLAAttention._get_workspace(device)
        self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buf,
        )

    def _get_rope_cos_sin(self, max_len: int):
        """Compute or retrieve cached RoPE cos/sin tables."""
        if self._rope_cos_sin is not None and self._rope_cos_sin[0].shape[0] >= max_len:
            return self._rope_cos_sin

        dim = self.rotary_dim  # may be < head_dim for partial RoPE
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=self.device).float() / dim))

        t = torch.arange(max_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # [max_len, rotary_dim//2]
        cos = freqs.cos().to(torch.bfloat16)
        sin = freqs.sin().to(torch.bfloat16)
        self._rope_cos_sin = (cos, sin)
        return cos, sin

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K heads.

        Supports partial RoPE: when rotary_dim < head_dim, only the first
        rotary_dim dimensions are rotated; the rest pass through unchanged.

        Args:
            q: [M, num_heads, head_dim]
            k: [M, num_kv_heads, head_dim]
            positions: [M] position indices
        """
        max_pos = positions.max().item() + 1
        cos, sin = self._get_rope_cos_sin(max_pos)

        pos_cos = cos[positions]  # [M, rotary_dim//2]
        pos_sin = sin[positions]  # [M, rotary_dim//2]

        def rotate(x, c, s):
            d2 = c.shape[-1]  # rotary_dim // 2
            while c.dim() < x.dim():
                c = c.unsqueeze(1)
                s = s.unsqueeze(1)
            x1, x2 = x[..., :d2], x[..., d2:2*d2]
            rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
            if 2 * d2 < x.shape[-1]:
                # Partial RoPE: append the non-rotated passthrough dims
                rotated = torch.cat([rotated, x[..., 2*d2:]], dim=-1)
            return rotated

        q = rotate(q, pos_cos, pos_sin)
        k = rotate(k, pos_cos, pos_sin)
        return q, k

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_state: SequenceKVState,
        layer_offset: int,
        num_new_tokens: int = 0,
    ) -> torch.Tensor:
        """Forward pass for one layer's GQA attention.

        Args:
            hidden: [M, hidden_size] BF16
            positions: [M] int32 position indices
            kv_cache: The paged KV cache (GQA format)
            seq_state: Sequence state
            layer_offset: This layer's index within its GPU's cache
            num_new_tokens: Number of new tokens being processed

        Returns:
            [M, hidden_size] BF16 attention output
        """
        M = hidden.shape[0]

        # ── Step 1: Q/K/V projections ──
        q_raw = _linear(hidden, self.q_proj)  # [M, num_heads * head_dim (* 2 if gated)]
        k = _linear(hidden, self.k_proj)  # [M, num_kv_heads * head_dim]
        v = _linear(hidden, self.v_proj)  # [M, num_kv_heads * head_dim]

        # Gated attention: split q_proj output into query + gate
        if self.gated_attention:
            # Reshape to [M, num_heads, head_dim * 2], then chunk
            q_raw = q_raw.view(M, self.num_heads, self.head_dim * 2)
            q, attn_gate = q_raw.chunk(2, dim=-1)  # each [M, num_heads, head_dim]
            attn_gate = attn_gate.reshape(M, self.num_heads * self.head_dim)  # [M, num_heads * head_dim]
        else:
            q = q_raw

        # Apply attention biases if present (GLM-4.7)
        if not self.gated_attention:
            if self.q_proj_bias is not None:
                q = q + self.q_proj_bias
        if self.k_proj_bias is not None:
            k = k + self.k_proj_bias
        if self.v_proj_bias is not None:
            v = v + self.v_proj_bias

        # Reshape to per-head
        if not self.gated_attention:
            q = q.reshape(M, self.num_heads, self.head_dim)
        k = k.reshape(M, self.num_kv_heads, self.head_dim)
        v = v.reshape(M, self.num_kv_heads, self.head_dim)

        # ── Step 2: QKNorm ──
        if self.q_norm is not None:
            # Per-head RMSNorm: apply same norm weight to each head
            q = flashinfer.norm.rmsnorm(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = flashinfer.norm.rmsnorm(k, self.k_norm, self.cfg.rms_norm_eps)

        # ── Step 3: RoPE ──
        q, k = self._apply_rope(q, k, positions)

        # ── Step 4: Append K, V to paged cache ──
        k_layer, v_layer = kv_cache.get_gqa_layer_caches(layer_offset)

        kv_indices = seq_state.kv_indices(self.device)
        kv_indptr = seq_state.kv_indptr(self.device)

        batch_indices = torch.zeros(M, dtype=torch.int32, device=self.device)
        append_positions = positions.to(torch.int32)
        last_page_len_tensor = torch.tensor(
            [seq_state.last_page_len() if seq_state.seq_len > 0 else 0],
            dtype=torch.int32, device=self.device,
        )

        # Cast to cache dtype
        k_append = k.to(kv_cache.kv_dtype)  # [M, num_kv_heads, head_dim]
        v_append = v.to(kv_cache.kv_dtype)  # [M, num_kv_heads, head_dim]

        flashinfer.page.append_paged_kv_cache(
            k_append, v_append,
            batch_indices, append_positions,
            (k_layer, v_layer),
            kv_indices, kv_indptr, last_page_len_tensor,
        )

        # ── Step 5: FlashInfer paged attention ──
        kv_indices = seq_state.kv_indices(self.device)
        kv_indptr = seq_state.kv_indptr(self.device)
        # kv_len must include the newly appended tokens (seq_len not yet advanced)
        effective_kv_len = seq_state.seq_len + num_new_tokens
        kv_len_arr = torch.tensor([effective_kv_len], dtype=torch.int32, device=self.device)
        qo_indptr = torch.tensor([0, M], dtype=torch.int32, device=self.device)

        # Compute last_page_len from effective seq_len (after append)
        eff_rem = effective_kv_len % kv_cache.page_size
        eff_last_page_len = eff_rem if eff_rem > 0 else kv_cache.page_size
        eff_last_page_len_tensor = torch.tensor(
            [eff_last_page_len], dtype=torch.int32, device=self.device,
        )

        # Pass KV cache directly to FlashInfer — FP8 E4M3 is natively supported
        # on SM89+ (Ada/Hopper). This avoids upcasting the entire paged cache layer
        # (hundreds of MB) when only a few pages are accessed.
        kv_dtype = k_layer.dtype

        plan_kwargs = dict(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=eff_last_page_len_tensor,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=kv_cache.page_size,
            causal=True,
            sm_scale=self.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
        )
        if self.sliding_window is not None:
            plan_kwargs["window_left"] = self.sliding_window - 1
        self._prefill_wrapper.plan(**plan_kwargs)

        # Request LSE (log-sum-exp) if sinks are present — needed for post-correction
        run_kwargs = dict(
            q=q.to(torch.bfloat16),
            paged_kv_cache=(k_layer, v_layer),
        )
        if self.sinks is not None:
            run_kwargs["return_lse"] = True

        run_result = self._prefill_wrapper.run(**run_kwargs)

        if self.sinks is not None:
            # Sinks: learnable logit per head concatenated to attention weights.
            # After FlashInfer computes output with LSE, we adjust:
            #   adjusted_output = output * sigmoid(lse - sink)
            # This is mathematically equivalent to adding a sink logit to softmax.
            attn_out, lse = run_result  # lse: [M, num_heads]
            sink = self.sinks.view(1, -1)  # [1, num_heads]
            scale = torch.sigmoid(lse - sink)  # [M, num_heads]
            attn_out = attn_out * scale.unsqueeze(-1)  # [M, num_heads, head_dim]
        else:
            # attn_out: [M, num_heads, head_dim]
            attn_out = run_result

        # ── Step 6: O projection ──
        attn_flat = attn_out.reshape(M, self.num_heads * self.head_dim)

        # Gated attention: apply sigmoid(gate) to attention output before o_proj
        if self.gated_attention:
            attn_flat = attn_flat * torch.sigmoid(attn_gate)

        # Ensure BF16 for linear (sinks path can produce float32 via sigmoid)
        if attn_flat.dtype != torch.bfloat16:
            attn_flat = attn_flat.to(torch.bfloat16)

        output = _linear(attn_flat, self.o_proj)
        if self.o_proj_bias is not None:
            output = output + self.o_proj_bias

        return output

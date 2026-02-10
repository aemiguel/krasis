"""TRTLLM MLA attention backend using FlashInfer TRTLLM kernels.

Two paths:
  Prefill (M > 1): trtllm_ragged_attention_deepseek
    - Non-absorbed: Q [M, H, 192], K [M, H, 192], V [M, H, 128]
    - Decompresses KV via w_kc/w_vc BEFORE attention
    - Output: [M, H, v_head=128] — no w_vc post-multiply needed

  Decode (M = 1): trtllm_batch_decode_with_kv_cache_mla
    - Absorbed: Q [1, H, 576], KV cache [pages, page_size, 576]
    - Output: [M, H, kv_lora_rank=512] — needs w_vc post-multiply

Both paths store compressed KV (576 dim) in the paged cache.

Adapted from SGLang's trtllm_mla_backend.py.
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

WORKSPACE_SIZE_MB = 150


class TRTLLMMLAAttention:
    """Multi-head Latent Attention using TRTLLM kernels for one layer."""

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

        # head_dim for non-absorbed path (prefill ragged): qk_nope + qk_rope = 192
        self.q_head_dim = self.qk_nope_dim + self.qk_rope_dim  # 192
        # head_dim for absorbed path (decode): kv_lora_rank + qk_rope = 576
        self.absorbed_head_dim = self.kv_lora_rank + self.qk_rope_dim  # 576
        self.kv_cache_dim = self.absorbed_head_dim  # 576

        # Scale factor: 1/sqrt(q_head_dim)
        self.sm_scale = 1.0 / math.sqrt(self.q_head_dim)

        # INT8 weights
        if self.has_q_lora:
            self.q_a_proj = weights["q_a_proj"]
            self.q_b_proj = weights["q_b_proj"]
            self.q_a_norm_weight = weights["q_a_layernorm"]
        else:
            self.q_proj = weights["q_proj"]

        self.kv_a_proj = weights["kv_a_proj_with_mqa"]
        self.o_proj = weights["o_proj"]
        self.kv_a_norm_weight = weights["kv_a_layernorm"]

        # BF16 kv_b_proj split weights
        self.w_kc = weights["w_kc"]  # [heads, qk_nope, kv_lora_rank]
        self.w_vc = weights["w_vc"]  # [heads, v_head, kv_lora_rank]

        # RoPE
        self.rope_theta = cfg.rope_theta
        self._rope_cos_sin = None

        # Workspace
        self._workspace = torch.zeros(
            WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8, device=device,
        )

    def _get_rope_cos_sin(self, max_len: int):
        """Compute or retrieve cached RoPE cos/sin tables."""
        if self._rope_cos_sin is not None and self._rope_cos_sin[0].shape[0] >= max_len:
            return self._rope_cos_sin

        rope_cfg = self.cfg.rope_scaling
        factor = rope_cfg.get("factor", 1.0)
        original_max = rope_cfg.get("original_max_position_embeddings", 4096)

        dim = self.qk_rope_dim
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=self.device).float() / dim))

        if factor > 1.0:
            beta_fast = rope_cfg.get("beta_fast", 32.0)
            beta_slow = rope_cfg.get("beta_slow", 1.0)
            low = math.floor(dim * math.log(original_max / (beta_fast * 2 * math.pi)) /
                             (2 * math.log(self.rope_theta)))
            high = math.ceil(dim * math.log(original_max / (beta_slow * 2 * math.pi)) /
                             (2 * math.log(self.rope_theta)))
            low = max(low, 0)
            high = min(high, dim // 2 - 1)
            for i in range(dim // 2):
                if i < low:
                    freqs[i] = freqs[i] / factor
                elif i > high:
                    pass
                else:
                    smooth = (i - low) / (high - low)
                    freqs[i] = (1 - smooth) * (freqs[i] / factor) + smooth * freqs[i]

        t = torch.arange(max_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        cos = freqs.cos().to(torch.bfloat16)
        sin = freqs.sin().to(torch.bfloat16)
        self._rope_cos_sin = (cos, sin)
        return cos, sin

    def _apply_rope(self, q_pe, k_pe, positions):
        """Apply rotary position embeddings."""
        max_pos = positions.max().item() + 1
        cos, sin = self._get_rope_cos_sin(max_pos)
        pos_cos = cos[positions]
        pos_sin = sin[positions]

        def rotate(x, c, s):
            d2 = x.shape[-1] // 2
            while c.dim() < x.dim():
                c = c.unsqueeze(1)
                s = s.unsqueeze(1)
            x1, x2 = x[..., :d2], x[..., d2:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

        return rotate(q_pe, pos_cos, pos_sin), rotate(k_pe, pos_cos, pos_sin)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_state: SequenceKVState,
        layer_offset: int,
    ) -> torch.Tensor:
        """Forward pass dispatching to prefill or decode."""
        M = hidden.shape[0]
        H = self.num_heads

        # ── KV compressed projection ──
        kv_out = _linear(hidden, self.kv_a_proj)
        kv_compressed = kv_out[:, :self.kv_lora_rank]   # [M, 512]
        k_pe = kv_out[:, self.kv_lora_rank:]            # [M, 64]

        # ── KV LayerNorm ──
        kv_compressed = flashinfer.norm.rmsnorm(
            kv_compressed, self.kv_a_norm_weight, self.cfg.rms_norm_eps
        )

        # ── Query projection ──
        if self.has_q_lora:
            q_compressed = _linear(hidden, self.q_a_proj)
            q_compressed = flashinfer.norm.rmsnorm(
                q_compressed, self.q_a_norm_weight, self.cfg.rms_norm_eps
            )
            q_full = _linear(q_compressed, self.q_b_proj)
            del q_compressed
        else:
            q_full = _linear(hidden, self.q_proj)

        q_full = q_full.reshape(M, H, self.qk_nope_dim + self.qk_rope_dim)
        q_nope = q_full[:, :, :self.qk_nope_dim]   # [M, H, 128]
        q_pe = q_full[:, :, self.qk_nope_dim:]      # [M, H, 64]
        del q_full

        # ── RoPE ──
        k_pe_heads = k_pe.unsqueeze(1)  # [M, 1, 64]
        q_pe, k_pe_heads = self._apply_rope(q_pe, k_pe_heads, positions)
        k_pe_rope = k_pe_heads.squeeze(1)  # [M, 64]

        # ── Store compressed KV in paged cache ──
        kv_combined = torch.cat([kv_compressed, k_pe_rope], dim=-1)  # [M, 576]
        cache_positions = positions.to(torch.long)
        seq_state.store_kv_combined(layer_offset, kv_combined, cache_positions)

        if M > 1:
            # ── PREFILL: non-absorbed path with ragged attention ──
            # Decompress K and V from compressed KV
            # k_nope: [M, 512] @ [H, 128, 512]^T → [M, H, 128]
            k_nope = torch.einsum(
                "md,hid->mhi", kv_compressed.float(), self.w_kc.float()
            ).to(torch.bfloat16)

            # v: [M, 512] @ [H, 128, 512]^T → [M, H, 128]
            v = torch.einsum(
                "md,hod->mho", kv_compressed.float(), self.w_vc.float()
            ).to(torch.bfloat16)

            # K: cat(k_nope, k_pe_broadcast) → [M, H, 192]
            k_pe_broadcast = k_pe_rope.unsqueeze(1).expand(-1, H, -1)  # [M, H, 64]
            key = torch.cat([k_nope, k_pe_broadcast], dim=-1)  # [M, H, 192]

            # Q: cat(q_nope, q_pe) → [M, H, 192] (NOT absorbed)
            query = torch.cat([q_nope, q_pe], dim=-1)

            # Ragged attention
            cum_seq_lens = torch.tensor([0, M], dtype=torch.int32, device=self.device)
            seq_lens = torch.tensor([M], dtype=torch.int32, device=self.device)

            out = torch.empty(M, H, self.v_head_dim, device=self.device, dtype=query.dtype)

            attn_out = flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=query,
                key=key,
                value=v,
                workspace_buffer=self._workspace,
                seq_lens=seq_lens,
                max_q_len=M,
                max_kv_len=M,
                bmm1_scale=self.sm_scale,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=1,
                window_left=-1,
                cum_seq_lens_q=cum_seq_lens,
                cum_seq_lens_kv=cum_seq_lens,
                enable_pdl=False,
                is_causal=True,
                return_lse=False,
                out=out,
            )

            # Prefill output: [M, H, v_head=128] — already decompressed, no w_vc needed
            attn_flat = attn_out.reshape(M, H * self.v_head_dim)
            return _linear(attn_flat, self.o_proj)

        else:
            # ── DECODE: absorbed path with paged cache ──
            # Absorb w_kc into query
            q_nope_absorbed = torch.einsum(
                "mhi,hid->mhd", q_nope.float(), self.w_kc.float()
            ).to(torch.bfloat16)

            # Merged query: [M, H, 576]
            query = torch.cat([q_nope_absorbed, q_pe], dim=-1)
            # Shape for decode kernel: [bs=1, seq=M, H, head_dim]
            q = query.unsqueeze(0)

            # KV cache: [max_pages, 1, page_size, kv_cache_dim]
            cache_layer = kv_cache.kv_cache[layer_offset]  # [max_pages, page_size, 576]
            kv = cache_layer.unsqueeze(1)  # [max_pages, 1, page_size, 576]

            block_tables = seq_state.block_tables(self.device)
            total_len = seq_state.seq_len
            seq_lens = torch.tensor([total_len], dtype=torch.int32, device=self.device)

            raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv,
                workspace_buffer=self._workspace,
                qk_nope_head_dim=self.qk_nope_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_dim,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=total_len,
                bmm1_scale=self.sm_scale,
            )

            # Decode output: [1, M, H, kv_lora_rank=512] → needs w_vc post-multiply
            attn_out = raw_out.squeeze(0)  # [M, H, 512]
            attn_projected = torch.einsum(
                "mhd,hod->mho", attn_out.float(), self.w_vc.float()
            ).to(torch.bfloat16)

            attn_flat = attn_projected.reshape(M, H * self.v_head_dim)
            return _linear(attn_flat, self.o_proj)

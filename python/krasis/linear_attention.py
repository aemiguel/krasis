"""Gated DeltaNet (linear attention) for hybrid Transformer-Mamba models.

Used by Qwen3-Coder-Next where 36/48 layers use linear attention and 12/48 use
standard GQA. Each linear attention layer maintains a small recurrent state
(~1 MB) and conv state (~48 KB) on GPU — no KV cache needed.

Algorithm based on "Gated Delta Networks with Softmax Attention" (Yang et al.).
Decode uses the recurrent formulation; prefill uses the chunked parallel form.

All computation runs on GPU (weights are tiny: ~35 MB INT8 per layer).

Ported from HF transformers Qwen3NextGatedDeltaNet — fixes:
  - QKVZ un-interleaving (fix_query_key_value_ordering)
  - BA un-interleaving
  - Conv1d state update order
  - Query scale factor (1/sqrt(head_dim))
  - l2norm matching FLA library
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from krasis.config import ModelConfig
from krasis.weight_loader import int8_linear

logger = logging.getLogger(__name__)


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to INT8 or BF16 linear based on weight type."""
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize matching FLA library implementation."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class GatedDeltaNetAttention:
    """Gated DeltaNet linear attention for one layer.

    Maintains internal recurrent state and conv state across forward calls.
    State is lazily initialized on first forward call.

    Weight names (from HF safetensors):
        linear_attn.in_proj_qkvz.weight  [q+k+v+z, hidden] (interleaved per key-head group)
        linear_attn.in_proj_ba.weight    [b+a, hidden] (interleaved per key-head group)
        linear_attn.conv1d.weight        [conv_dim, 1, kernel_dim]
        linear_attn.out_proj.weight      [hidden, v_total]
        linear_attn.A_log                [num_value_heads]
        linear_attn.dt_bias              [num_value_heads]
        linear_attn.norm.weight          [value_head_dim]
    """

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

        # Dimensions from config
        self.num_k_heads = cfg.linear_num_key_heads      # 16
        self.num_v_heads = cfg.linear_num_value_heads     # 32
        self.k_head_dim = cfg.linear_key_head_dim         # 128
        self.v_head_dim = cfg.linear_value_head_dim       # 128
        self.hidden_size = cfg.hidden_size                # 2048
        self.kernel_dim = cfg.linear_conv_kernel_dim      # 4

        # Derived dimensions
        self.key_dim = self.num_k_heads * self.k_head_dim   # 2048
        self.value_dim = self.num_v_heads * self.v_head_dim  # 4096
        self.conv_dim = self.key_dim * 2 + self.value_dim    # 8192

        # Head group ratio: how many value heads per key head
        self.head_ratio = self.num_v_heads // self.num_k_heads  # 2

        # Query scale factor (applied after L2 norm)
        self.scale = 1.0 / (self.k_head_dim ** 0.5)

        # Weights (INT8 or BF16)
        self.in_proj_qkvz = weights["in_proj_qkvz"]  # [q+k+v+z, hidden]
        self.in_proj_ba = weights["in_proj_ba"]       # [b+a, hidden]
        self.out_proj = weights["out_proj"]           # [hidden, v_total]

        # Small weights — always BF16
        self.conv1d_weight = weights["conv1d_weight"]  # [conv_dim, 1, kernel_dim]
        self.A_log = weights["A_log"]                  # [num_value_heads]
        self.dt_bias = weights["dt_bias"]              # [num_value_heads]
        self.norm_weight = weights["norm_weight"]      # [value_head_dim]

        # State (lazy-initialized on first forward)
        # Conv state stores last kernel_dim tokens (matching HF convention)
        self._conv_state: Optional[torch.Tensor] = None    # [1, conv_dim, kernel_dim]
        self._recurrent_state: Optional[torch.Tensor] = None  # [1, num_v_heads, k_head_dim, v_head_dim]

    def reset_state(self):
        """Reset recurrent and conv state (e.g. between sequences)."""
        self._conv_state = None
        self._recurrent_state = None

    def _init_state(self, batch_size: int = 1):
        """Initialize conv and recurrent states to zero."""
        if self._conv_state is None:
            # Match HF: store kernel_dim tokens in conv state
            self._conv_state = torch.zeros(
                batch_size, self.conv_dim, self.kernel_dim,
                dtype=torch.bfloat16, device=self.device,
            )
        if self._recurrent_state is None:
            self._recurrent_state = torch.zeros(
                batch_size, self.num_v_heads, self.k_head_dim, self.v_head_dim,
                dtype=torch.float32, device=self.device,
            )

    def _fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Un-interleave QKVZ and BA projections.

        The weight matrices are structured so that the output is interleaved
        per key-head group. This function undoes that interleaving.

        Args:
            mixed_qkvz: [M, q+k+v+z] from in_proj_qkvz
            mixed_ba: [M, b+a] from in_proj_ba

        Returns:
            (q, k, v, z, b, a) with proper head grouping
        """
        M = mixed_qkvz.shape[0]

        # QKVZ: reshape to [M, num_k_heads, group_dim] then split
        # group_dim = 2*k_head_dim + 2*v_head_dim*(num_v_heads//num_k_heads)
        group_dim = 2 * self.k_head_dim + 2 * self.v_head_dim * self.head_ratio
        qkvz_grouped = mixed_qkvz.view(M, self.num_k_heads, group_dim)

        # Split within each group
        split_sizes = [
            self.k_head_dim,                          # q: 128
            self.k_head_dim,                          # k: 128
            self.head_ratio * self.v_head_dim,        # v: 256
            self.head_ratio * self.v_head_dim,        # z: 256
        ]
        q, k, v, z = torch.split(qkvz_grouped, split_sizes, dim=2)
        # q: [M, nk, dk], k: [M, nk, dk]
        # v: [M, nk, ratio*dv] → reshape to [M, nv, dv]
        # z: [M, nk, ratio*dv] → reshape to [M, nv, dv]
        v = v.reshape(M, self.num_v_heads, self.v_head_dim)
        z = z.reshape(M, self.num_v_heads, self.v_head_dim)

        # BA: reshape to [M, num_k_heads, 2*ratio] then split
        ba_group_dim = 2 * self.head_ratio
        ba_grouped = mixed_ba.view(M, self.num_k_heads, ba_group_dim)
        b, a = torch.split(ba_grouped, [self.head_ratio, self.head_ratio], dim=2)
        # b: [M, nk, ratio] → [M, nv]
        # a: [M, nk, ratio] → [M, nv]
        b = b.reshape(M, self.num_v_heads)
        a = a.reshape(M, self.num_v_heads)

        return q, k, v, z, b, a

    def forward(
        self,
        hidden: torch.Tensor,
        is_decode: bool,
    ) -> torch.Tensor:
        """Forward pass for Gated DeltaNet linear attention.

        Args:
            hidden: [M, hidden_size] BF16
            is_decode: True for single-token decode, False for prefill

        Returns:
            [M, hidden_size] BF16 output
        """
        self._init_state()

        if is_decode:
            return self._forward_recurrent(hidden)
        else:
            return self._forward_chunked(hidden)

    def _forward_recurrent(self, hidden: torch.Tensor) -> torch.Tensor:
        """Recurrent decode: process tokens using recurrent formulation.

        Matches HF torch_recurrent_gated_delta_rule exactly.
        """
        M = hidden.shape[0]

        # Project to qkvz and ba
        qkvz = _linear(hidden, self.in_proj_qkvz)  # [M, q+k+v+z]
        ba = _linear(hidden, self.in_proj_ba)       # [M, b+a]

        # Un-interleave
        q, k, v, z, b, a = self._fix_query_key_value_ordering(qkvz, ba)
        # q: [M, nk, dk], k: [M, nk, dk], v: [M, nv, dv], z: [M, nv, dv]
        # b: [M, nv], a: [M, nv]

        # Flatten q, k, v for conv1d: [M, key_dim], [M, key_dim], [M, value_dim]
        q_flat = q.reshape(M, self.key_dim)
        k_flat = k.reshape(M, self.key_dim)
        v_flat = v.reshape(M, self.value_dim)

        # Causal conv1d update (token by token)
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [M, conv_dim]
        mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1, 2)  # [1, conv_dim, M]

        # Conv state update: cat state + new, then update state to last kernel_dim
        conv_weight = self.conv1d_weight.squeeze(1)  # [conv_dim, kernel_dim]
        conv_input = torch.cat([self._conv_state, mixed_qkv], dim=-1)  # [1, conv_dim, kernel_dim + M]
        self._conv_state = conv_input[:, :, -self.kernel_dim:].clone()

        # Apply depthwise conv1d
        conv_out = F.conv1d(
            conv_input.to(conv_weight.dtype),
            self.conv1d_weight,
            bias=None,
            padding=0,
            groups=self.conv_dim,
        )  # [1, conv_dim, M]

        # SiLU activation
        conv_out = F.silu(conv_out[:, :, -M:])
        conv_out = conv_out.to(hidden.dtype)
        conv_out = conv_out.transpose(1, 2).squeeze(0)  # [M, conv_dim]

        # Split back to q, k, v and reshape to heads
        q_out = conv_out[:, :self.key_dim].reshape(M, self.num_k_heads, self.k_head_dim)
        k_out = conv_out[:, self.key_dim:self.key_dim * 2].reshape(M, self.num_k_heads, self.k_head_dim)
        v_out = conv_out[:, self.key_dim * 2:].reshape(M, self.num_v_heads, self.v_head_dim)

        # Compute gating parameters
        beta = torch.sigmoid(b)                                   # [M, nv]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [M, nv]

        # Repeat-interleave k to match value heads
        if self.head_ratio > 1:
            q_out = q_out.repeat_interleave(self.head_ratio, dim=1)  # [M, nv, dk]
            k_out = k_out.repeat_interleave(self.head_ratio, dim=1)  # [M, nv, dk]

        # L2 normalize q and k
        q_out = _l2norm(q_out, dim=-1)
        k_out = _l2norm(k_out, dim=-1)

        # Scale query
        q_out = q_out * self.scale

        # Recurrent delta rule (matches HF torch_recurrent_gated_delta_rule)
        outputs = []
        for t in range(M):
            q_t = q_out[t]           # [nv, dk]
            k_t = k_out[t]           # [nv, dk]
            v_t = v_out[t]           # [nv, dv]
            g_t = g[t].exp()         # [nv]
            beta_t = beta[t]         # [nv]

            # Decay state
            self._recurrent_state = self._recurrent_state * g_t.unsqueeze(-1).unsqueeze(-1)

            # Delta update: state @ k → kv_mem, then delta = beta * (v - kv_mem)
            kv_mem = (self._recurrent_state.squeeze(0) * k_t.unsqueeze(-1)).sum(dim=-2)  # [nv, dv]
            delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # [nv, dv]
            self._recurrent_state = self._recurrent_state + k_t.unsqueeze(-1).unsqueeze(0) * delta.unsqueeze(-2).unsqueeze(0)

            # Output: state @ q
            out_t = (self._recurrent_state.squeeze(0) * q_t.unsqueeze(-1)).sum(dim=-2)  # [nv, dv]
            outputs.append(out_t)

        # Stack outputs: [M, nv, dv]
        attn_out = torch.stack(outputs, dim=0)

        # Gated RMSNorm with z gate
        attn_out = self._gated_rmsnorm(attn_out, z)  # [M, nv, dv]

        # Flatten and project (cast back to BF16 for out_proj)
        attn_flat = attn_out.reshape(M, self.num_v_heads * self.v_head_dim).to(torch.bfloat16)
        return _linear(attn_flat, self.out_proj)  # [M, hidden]

    def _forward_chunked(self, hidden: torch.Tensor) -> torch.Tensor:
        """Chunked prefill: matches HF torch_chunk_gated_delta_rule.

        Uses the parallel-within-chunk, recurrent-across-chunks formulation.
        """
        M = hidden.shape[0]
        chunk_size = 64

        # Project all tokens at once
        qkvz = _linear(hidden, self.in_proj_qkvz)  # [M, q+k+v+z]
        ba = _linear(hidden, self.in_proj_ba)       # [M, b+a]

        # Un-interleave
        q, k, v, z, b, a = self._fix_query_key_value_ordering(qkvz, ba)

        # Flatten q, k, v for conv1d
        q_flat = q.reshape(M, self.key_dim)
        k_flat = k.reshape(M, self.key_dim)
        v_flat = v.reshape(M, self.value_dim)

        # Apply causal conv1d over the full sequence
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [M, conv_dim]
        mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1, 2)  # [1, conv_dim, M]

        # Pad with conv state on the left
        conv_input = torch.cat([self._conv_state, mixed_qkv], dim=-1)

        # Update conv state with last kernel_dim tokens (for subsequent decode)
        self._conv_state = conv_input[:, :, -self.kernel_dim:].clone()

        # Depthwise conv1d
        conv_out = F.conv1d(
            conv_input.to(self.conv1d_weight.dtype),
            self.conv1d_weight,
            bias=None,
            padding=0,
            groups=self.conv_dim,
        )  # [1, conv_dim, M]

        # SiLU activation + take last M outputs
        conv_out = F.silu(conv_out[:, :, -M:])
        conv_out = conv_out.to(hidden.dtype)
        conv_out = conv_out.transpose(1, 2).squeeze(0)  # [M, conv_dim]

        # Split to q, k, v and reshape to heads
        q_all = conv_out[:, :self.key_dim].reshape(M, self.num_k_heads, self.k_head_dim)
        k_all = conv_out[:, self.key_dim:self.key_dim * 2].reshape(M, self.num_k_heads, self.k_head_dim)
        v_all = conv_out[:, self.key_dim * 2:].reshape(M, self.num_v_heads, self.v_head_dim)

        # Compute gating parameters
        beta_all = torch.sigmoid(b)  # [M, nv]
        a_float = a.float()
        g_all = -self.A_log.float().exp() * F.softplus(a_float + self.dt_bias)  # [M, nv]

        # Repeat-interleave k to match value heads
        if self.head_ratio > 1:
            q_all = q_all.repeat_interleave(self.head_ratio, dim=1)  # [M, nv, dk]
            k_all = k_all.repeat_interleave(self.head_ratio, dim=1)  # [M, nv, dk]

        # L2 normalize q and k
        q_all = _l2norm(q_all, dim=-1)
        k_all = _l2norm(k_all, dim=-1)

        # Scale query
        q_all = q_all * self.scale

        # Convert to float32 for numerical stability (matching HF)
        q_all = q_all.float()
        k_all = k_all.float()
        v_all = v_all.float()
        beta_all = beta_all.float()

        # Pad to multiple of chunk_size
        pad_size = (chunk_size - M % chunk_size) % chunk_size
        if pad_size > 0:
            q_all = F.pad(q_all, (0, 0, 0, 0, 0, pad_size))
            k_all = F.pad(k_all, (0, 0, 0, 0, 0, pad_size))
            v_all = F.pad(v_all, (0, 0, 0, 0, 0, pad_size))
            beta_all = F.pad(beta_all, (0, 0, 0, pad_size))
            g_all = F.pad(g_all, (0, 0, 0, pad_size))
        total_len = M + pad_size

        # Add batch and head dims: [1, nv, total_len, dim]
        q_4d = q_all.unsqueeze(0).transpose(1, 2)    # [1, nv, total_len, dk]
        k_4d = k_all.unsqueeze(0).transpose(1, 2)    # [1, nv, total_len, dk]
        v_4d = v_all.unsqueeze(0).transpose(1, 2)    # [1, nv, total_len, dv]
        beta_3d = beta_all.unsqueeze(0).transpose(1, 2)  # [1, nv, total_len]
        g_3d = g_all.unsqueeze(0).transpose(1, 2)        # [1, nv, total_len]

        # Pre-compute beta-scaled values
        v_beta = v_4d * beta_3d.unsqueeze(-1)          # [1, nv, total_len, dv]
        k_beta = k_4d * beta_3d.unsqueeze(-1)          # [1, nv, total_len, dk]

        # Reshape to chunks: [1, nv, num_chunks, chunk_size, dim]
        num_chunks = total_len // chunk_size
        q_c = q_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        k_c = k_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        v_c = v_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.v_head_dim)
        k_beta_c = k_beta.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        v_beta_c = v_beta.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.v_head_dim)
        g_c = g_3d.reshape(1, self.num_v_heads, num_chunks, chunk_size)

        # Masks
        mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=self.device), diagonal=0)
        mask_strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=self.device), diagonal=1)

        # Cumulative g for decay within chunks
        g_cum = g_c.cumsum(dim=-1)  # [1, nv, num_chunks, chunk_size]

        # Intra-chunk decay matrix
        decay_mask = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()
        # [1, nv, num_chunks, chunk_size, chunk_size]

        # Intra-chunk attention matrix (lower triangular)
        # attn[i,j] = -(k_beta[j] @ k[j]^T) * decay[i,j]
        attn = -(k_beta_c @ k_c.transpose(-1, -2)) * decay_mask
        attn = attn.masked_fill(mask_upper, 0)

        # Nilpotent correction (resolvent series)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=self.device)

        # Corrected values and keys
        value_corrected = attn @ v_beta_c
        k_cumdecay = attn @ (k_beta_c * g_cum.exp().unsqueeze(-1))

        # Initialize output
        core_attn_out = torch.zeros_like(value_corrected)

        # Process chunks with recurrent state
        for i in range(num_chunks):
            q_i = q_c[:, :, i]      # [1, nv, chunk_size, dk]
            k_i = k_c[:, :, i]      # [1, nv, chunk_size, dk]
            v_i = value_corrected[:, :, i]  # [1, nv, chunk_size, dv]
            g_i = g_cum[:, :, i]     # [1, nv, chunk_size]

            # Intra-chunk attention
            attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]
            attn_intra = attn_intra.masked_fill_(mask_strict_upper, 0)

            # Cross-chunk contribution from recurrent state
            v_prime = k_cumdecay[:, :, i] @ self._recurrent_state  # [1, nv, chunk_size, dv]
            v_new = v_i - v_prime

            attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ self._recurrent_state  # [1, nv, chunk_size, dv]
            core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new

            # Update recurrent state
            g_last = g_i[:, :, -1]  # [1, nv] — cumulative g at end of chunk
            self._recurrent_state = (
                self._recurrent_state * g_last.unsqueeze(-1).unsqueeze(-1).exp()
                + (k_i * (g_last.unsqueeze(-1) - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
            )

        # Reshape output: [1, nv, num_chunks, chunk_size, dv] → [1, nv, total_len, dv]
        core_attn_out = core_attn_out.reshape(1, self.num_v_heads, -1, self.v_head_dim)
        # Trim padding and transpose back
        core_attn_out = core_attn_out[:, :, :M, :]  # [1, nv, M, dv]
        core_attn_out = core_attn_out.transpose(1, 2).squeeze(0)  # [M, nv, dv]
        core_attn_out = core_attn_out.to(hidden.dtype)

        # Gated RMSNorm with z gate
        attn_out = self._gated_rmsnorm(core_attn_out, z)  # [M, nv, dv]

        # Flatten and project (cast back to BF16 for out_proj)
        attn_flat = attn_out.reshape(M, self.num_v_heads * self.v_head_dim).to(torch.bfloat16)
        return _linear(attn_flat, self.out_proj)  # [M, hidden]

    def _gated_rmsnorm(
        self,
        x: torch.Tensor,    # [..., nv, dv]
        gate: torch.Tensor,  # [..., nv, dv]
    ) -> torch.Tensor:
        """Gated RMSNorm: rmsnorm(x) * silu(gate).

        Matches HF Qwen3NextRMSNormGated: norm first, then multiply by silu(gate).
        """
        input_dtype = x.dtype
        # RMSNorm per head
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.cfg.rms_norm_eps)
        x_normed = (self.norm_weight.float() * x_normed).to(input_dtype)

        # Gate with SiLU (gate in float32 for numerical stability)
        return x_normed * F.silu(gate.float()).to(input_dtype)

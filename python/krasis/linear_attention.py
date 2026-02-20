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
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from krasis.config import ModelConfig
from krasis.timing import TIMING
from krasis.weight_loader import int8_linear

logger = logging.getLogger(__name__)

# Fused linear attention: solve_triangular + torch.compile
# Disable with KRASIS_FUSED_LINEAR_ATTN=0 to fall back to original Python loops
_FUSED_LINEAR_ATTN = os.environ.get("KRASIS_FUSED_LINEAR_ATTN", "1") != "0"

# Compiled chunk step (lazily created on first use)
_compiled_chunk_step = None


def _chunk_step(q_i, k_i, v_i, g_i, k_cumd_i, decay_mask_i,
                mask_strict_upper, state):
    """Single recurrent chunk step: intra-chunk attention + cross-chunk state update."""
    attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask_i
    attn_intra = attn_intra.masked_fill(mask_strict_upper, 0)

    v_prime = k_cumd_i @ state
    v_new = v_i - v_prime

    attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ state
    output = attn_inter + attn_intra @ v_new

    g_last = g_i[:, :, -1]
    g_last_exp = g_last.unsqueeze(-1).unsqueeze(-1).exp()
    k_decay = (g_last.unsqueeze(-1) - g_i).exp().unsqueeze(-1)
    new_state = state * g_last_exp + (k_i * k_decay).transpose(-1, -2) @ v_new

    return output, new_state


def _get_chunk_step():
    """Get the chunk step function (compiled if enabled, eager otherwise)."""
    global _compiled_chunk_step
    if not _FUSED_LINEAR_ATTN:
        return _chunk_step
    if _compiled_chunk_step is not None:
        return _compiled_chunk_step
    try:
        # Disable Inductor CUDA graphs: the recurrent loop feeds output→input
        # (state from iteration N → input to iteration N+1), which CUDA graphs
        # can't handle (buffer overwrite error). Inductor JIT fusion still works.
        torch._inductor.config.triton.cudagraphs = False
        _compiled_chunk_step = torch.compile(
            _chunk_step, mode="default", dynamic=False)
        logger.info("Compiled linear attention chunk step (default/Inductor mode)")
    except Exception as e:
        logger.warning("torch.compile failed for chunk step, using eager: %s", e)
        _compiled_chunk_step = _chunk_step
    return _compiled_chunk_step


def warmup_compiled_chunk_step(device, nv, dk, dv, chunk_size=64):
    """Warm up torch.compile for the linear attention chunk step function."""
    if not _FUSED_LINEAR_ATTN:
        return
    step_fn = _get_chunk_step()
    if step_fn is _chunk_step:
        return  # compilation failed, using eager — no warmup needed
    try:
        s = torch.zeros(1, nv, dk, dv, dtype=torch.float32, device=device)
        q = torch.zeros(1, nv, chunk_size, dk, dtype=torch.float32, device=device)
        k = torch.zeros(1, nv, chunk_size, dk, dtype=torch.float32, device=device)
        v = torch.zeros(1, nv, chunk_size, dv, dtype=torch.float32, device=device)
        g = torch.zeros(1, nv, chunk_size, dtype=torch.float32, device=device)
        dm = torch.zeros(1, nv, chunk_size, chunk_size, dtype=torch.float32, device=device)
        m = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=1)
        for _ in range(3):
            step_fn(q, k, v, g, k, dm, m, s)
        torch.cuda.synchronize(device)
        logger.info("Linear attention torch.compile warmup complete on %s", device)
    except Exception as e:
        logger.warning("Linear attention torch.compile warmup on %s failed: %s", device, e)


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

        # CUDA graph for M=1 recurrent decode
        self._la_graph: Optional[torch.cuda.CUDAGraph] = None
        self._la_input: Optional[torch.Tensor] = None    # [1, hidden_size] static input
        self._la_output: Optional[torch.Tensor] = None   # [1, hidden_size] static output
        self._la_stream = torch.cuda.Stream(device=device)  # non-default stream for graph capture

    def reset_state(self):
        """Reset recurrent and conv state (e.g. between sequences)."""
        self._conv_state = None
        self._recurrent_state = None
        # Invalidate CUDA graph — it references old state tensor addresses
        self._la_graph = None
        self._la_input = None
        self._la_output = None

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

    def _capture_la_graph(self):
        """Capture CUDA graph for entire M=1 linear attention recurrent forward.

        Eliminates ~60+ kernel launches (conv1d, INT8 quantize/pad/matmul/dequant,
        element-wise ops) with a single graph replay. First called on second M=1
        forward (after one warmup iteration to settle allocations).

        State is saved before warmup/capture and restored after, so the graph
        capture doesn't corrupt the model's recurrent state.
        """
        self._la_input = torch.empty(1, self.hidden_size, dtype=torch.bfloat16, device=self.device)
        self._la_output = torch.empty(1, self.hidden_size, dtype=torch.bfloat16, device=self.device)

        # Save state before warmup (warmup iterations would corrupt it)
        conv_saved = self._conv_state.clone()
        recur_saved = self._recurrent_state.clone()

        # Must capture on a non-default stream (CUDA graph requirement)
        stream = self._la_stream

        # Warmup iterations on the capture stream (let CUDA allocator settle)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for _ in range(3):
                out = self._forward_recurrent_inplace(self._la_input)
                self._la_output.copy_(out)
        torch.cuda.synchronize()

        # Capture the graph on the same stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            out = self._forward_recurrent_inplace(self._la_input)
            self._la_output.copy_(out)

        # Restore state after capture
        self._conv_state.copy_(conv_saved)
        self._recurrent_state.copy_(recur_saved)

        self._la_graph = g
        logger.debug("Captured linear attention CUDA graph for layer %d", self.layer_idx)

    def _forward_recurrent_inplace(self, hidden: torch.Tensor) -> torch.Tensor:
        """Graph-compatible M=1 recurrent forward using in-place state updates.

        Same computation as _forward_recurrent but:
        - _conv_state updated via copy_ (not reassignment)
        - _recurrent_state updated via mul_/add_ (not reassignment)
        This makes the operation compatible with CUDA graph replay since
        state tensors keep their memory addresses.
        """
        M = hidden.shape[0]

        # Project to qkvz and ba
        qkvz = _linear(hidden, self.in_proj_qkvz)
        ba = _linear(hidden, self.in_proj_ba)

        # Un-interleave
        q, k, v, z, b, a = self._fix_query_key_value_ordering(qkvz, ba)

        q_flat = q.reshape(M, self.key_dim)
        k_flat = k.reshape(M, self.key_dim)
        v_flat = v.reshape(M, self.value_dim)

        # Conv1d: cat state + new token, update state in-place
        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1).unsqueeze(0).transpose(1, 2)
        conv_input = torch.cat([self._conv_state, mixed_qkv], dim=-1)
        self._conv_state.copy_(conv_input[:, :, -self.kernel_dim:])

        conv_out = F.conv1d(
            conv_input.to(self.conv1d_weight.dtype),
            self.conv1d_weight, bias=None, padding=0, groups=self.conv_dim,
        )
        conv_out = F.silu(conv_out[:, :, -M:]).to(hidden.dtype)
        conv_out = conv_out.transpose(1, 2).squeeze(0)

        # Split to q, k, v heads
        q_out = conv_out[:, :self.key_dim].reshape(M, self.num_k_heads, self.k_head_dim)
        k_out = conv_out[:, self.key_dim:self.key_dim * 2].reshape(M, self.num_k_heads, self.k_head_dim)
        v_out = conv_out[:, self.key_dim * 2:].reshape(M, self.num_v_heads, self.v_head_dim)

        # Gating parameters
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Repeat-interleave k heads to match value heads
        if self.head_ratio > 1:
            q_out = q_out.repeat_interleave(self.head_ratio, dim=1)
            k_out = k_out.repeat_interleave(self.head_ratio, dim=1)

        q_out = _l2norm(q_out, dim=-1) * self.scale
        k_out = _l2norm(k_out, dim=-1)

        # Recurrent update (M=1, single step) — ALL state updates in-place
        q_t = q_out[0]       # [nv, dk]
        k_t = k_out[0]       # [nv, dk]
        v_t = v_out[0]       # [nv, dv]
        g_t = g[0].exp()     # [nv]
        beta_t = beta[0]     # [nv]

        # In-place decay: state *= g_t
        self._recurrent_state.mul_(g_t.unsqueeze(-1).unsqueeze(-1))

        # Delta update
        kv_mem = (self._recurrent_state.squeeze(0) * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)
        self._recurrent_state.add_(
            k_t.unsqueeze(-1).unsqueeze(0) * delta.unsqueeze(-2).unsqueeze(0)
        )

        # Output: state @ q
        out_t = (self._recurrent_state.squeeze(0) * q_t.unsqueeze(-1)).sum(dim=-2)

        # Gated RMSNorm + output projection
        attn_out = self._gated_rmsnorm(out_t.unsqueeze(0), z)
        attn_flat = attn_out.reshape(1, self.num_v_heads * self.v_head_dim).to(torch.bfloat16)
        return _linear(attn_flat, self.out_proj)

    def _fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_k_heads: Optional[int] = None,
        num_v_heads: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Un-interleave QKVZ and BA projections.

        The weight matrices are structured so that the output is interleaved
        per key-head group. This function undoes that interleaving.

        Args:
            mixed_qkvz: [M, q+k+v+z] from in_proj_qkvz
            mixed_ba: [M, b+a] from in_proj_ba
            num_k_heads: override for split-attention (default: self.num_k_heads)
            num_v_heads: override for split-attention (default: self.num_v_heads)

        Returns:
            (q, k, v, z, b, a) with proper head grouping
        """
        M = mixed_qkvz.shape[0]
        nk = num_k_heads if num_k_heads is not None else self.num_k_heads
        nv = num_v_heads if num_v_heads is not None else self.num_v_heads

        # QKVZ: reshape to [M, num_k_heads, group_dim] then split
        # group_dim = 2*k_head_dim + 2*v_head_dim*(num_v_heads//num_k_heads)
        group_dim = 2 * self.k_head_dim + 2 * self.v_head_dim * self.head_ratio
        qkvz_grouped = mixed_qkvz.view(M, nk, group_dim)

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
        v = v.reshape(M, nv, self.v_head_dim)
        z = z.reshape(M, nv, self.v_head_dim)

        # BA: reshape to [M, num_k_heads, 2*ratio] then split
        ba_group_dim = 2 * self.head_ratio
        ba_grouped = mixed_ba.view(M, nk, ba_group_dim)
        b, a = torch.split(ba_grouped, [self.head_ratio, self.head_ratio], dim=2)
        # b: [M, nk, ratio] → [M, nv]
        # a: [M, nk, ratio] → [M, nv]
        b = b.reshape(M, nv)
        a = a.reshape(M, nv)

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
            # CUDA graph path for M=1: eliminates ~60+ kernel launches
            if self._la_graph is not None:
                # Sync: ensure hidden is ready, then replay on capture stream
                self._la_stream.wait_stream(torch.cuda.current_stream(self.device))
                self._la_input.copy_(hidden)
                self._la_graph.replay()
                # Sync: wait for graph to finish before returning output
                torch.cuda.current_stream(self.device).wait_stream(self._la_stream)
                return self._la_output.clone()
            elif self._la_input is None and not (TIMING.decode or TIMING.prefill):
                # First M=1 call: run normally (warmup), capture graph next call
                self._la_input = "pending"  # sentinel: capture on next call
                return self._forward_recurrent(hidden)
            elif self._la_input == "pending":
                # Second M=1 call: capture the graph (requires ~20-30 MB free VRAM)
                free_mb = torch.cuda.mem_get_info(self.device)[0] / (1024 * 1024)
                if free_mb < 50:
                    logger.debug("Skipping LA graph capture for layer %d: only %.0f MB free (need ~50 MB)", self.layer_idx, free_mb)
                    self._la_input = None  # disable graph attempts
                    return self._forward_recurrent(hidden)
                try:
                    self._capture_la_graph()
                    self._la_input.copy_(hidden)
                    self._la_graph.replay()
                    return self._la_output.clone()
                except Exception as e:
                    logger.warning("CUDA graph capture failed for LA layer %d: %s", self.layer_idx, e)
                    self._la_input = None  # disable graph attempts
                    return self._forward_recurrent(hidden)
            else:
                return self._forward_recurrent(hidden)
        else:
            # Invalidate graph state tracking when switching to prefill
            # (conv_state and recurrent_state get modified by chunked path)
            if self._la_graph is not None:
                self._la_graph = None
                self._la_input = None
                self._la_output = None
            return self._forward_chunked(hidden)

    def _forward_recurrent(self, hidden: torch.Tensor) -> torch.Tensor:
        """Recurrent decode: process tokens using recurrent formulation.

        Matches HF torch_recurrent_gated_delta_rule exactly.
        """
        M = hidden.shape[0]
        timing = TIMING.decode and M == 1

        if timing:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # Project to qkvz and ba
        qkvz = _linear(hidden, self.in_proj_qkvz)  # [M, q+k+v+z]
        ba = _linear(hidden, self.in_proj_ba)       # [M, b+a]

        if timing:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

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

        if timing:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()

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

        if timing:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()

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

        if timing:
            torch.cuda.synchronize()
            _t4 = time.perf_counter()

        # Stack outputs: [M, nv, dv]
        attn_out = torch.stack(outputs, dim=0)

        # Gated RMSNorm with z gate
        attn_out = self._gated_rmsnorm(attn_out, z)  # [M, nv, dv]

        # Flatten and project (cast back to BF16 for out_proj)
        attn_flat = attn_out.reshape(M, self.num_v_heads * self.v_head_dim).to(torch.bfloat16)
        result = _linear(attn_flat, self.out_proj)  # [M, hidden]

        if timing:
            torch.cuda.synchronize()
            _t5 = time.perf_counter()
            logger.info(
                "  LA-DETAIL L%d: proj=%.2fms conv=%.2fms prep=%.2fms recur=%.2fms norm+out=%.2fms total=%.2fms",
                self.layer_idx,
                (_t1 - _t0) * 1000,
                (_t2 - _t1) * 1000,
                (_t3 - _t2) * 1000,
                (_t4 - _t3) * 1000,
                (_t5 - _t4) * 1000,
                (_t5 - _t0) * 1000,
            )

        return result

    def _chunked_inner(
        self,
        recurrent_state: torch.Tensor,
        q_c: torch.Tensor,
        k_c: torch.Tensor,
        v_beta_c: torch.Tensor,
        k_beta_c: torch.Tensor,
        g_c: torch.Tensor,
        chunk_size: int,
        num_chunks: int,
    ) -> tuple:
        """Nilpotent correction + recurrent chunk loop (shared by all chunked paths).

        Replaces ~600 kernel launches (63-iter nilpotent loop + 16-chunk recurrent loop)
        with ~18 (2 triangular solves + 16 compiled chunk steps) when fused mode is on.

        Args:
            recurrent_state: [1, nv, dk, dv] current recurrent state
            q_c: [1, nv, num_chunks, chunk_size, dk]
            k_c: [1, nv, num_chunks, chunk_size, dk]
            v_beta_c: [1, nv, num_chunks, chunk_size, dv] beta-scaled values
            k_beta_c: [1, nv, num_chunks, chunk_size, dk] beta-scaled keys
            g_c: [1, nv, num_chunks, chunk_size] gating values
            chunk_size: size of each chunk (64)
            num_chunks: number of chunks

        Returns:
            (core_attn_out, updated_recurrent_state)
        """
        device = q_c.device

        mask_upper = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=0)
        mask_strict_upper = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=1)

        # Cumulative g for decay within chunks
        g_cum = g_c.cumsum(dim=-1)

        # Intra-chunk decay matrix
        decay_mask = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()

        _timing = TIMING.prefill

        # Intra-chunk attention matrix (strictly lower triangular, nilpotent)
        attn = -(k_beta_c @ k_c.transpose(-1, -2)) * decay_mask
        attn = attn.masked_fill(mask_upper, 0)

        if _timing:
            torch.cuda.synchronize(device)
            _ci_t0 = time.perf_counter()

        if _FUSED_LINEAR_ATTN:
            # Solve (I - A) @ x = b via triangular solve (2 cuBLAS calls vs 63-iter loop).
            # A = attn (strictly lower triangular). (I - A) is unitriangular lower.
            # Pass -attn with unitriangular=True: diagonal treated as 1, giving I - A.
            neg_attn = -attn
            value_corrected = torch.linalg.solve_triangular(
                neg_attn, v_beta_c, upper=False, unitriangular=True)
            k_cumdecay = torch.linalg.solve_triangular(
                neg_attn, k_beta_c * g_cum.exp().unsqueeze(-1),
                upper=False, unitriangular=True)
        else:
            # Original nilpotent correction (resolvent series, 63 iterations)
            for i in range(1, chunk_size):
                row = attn[..., i, :i].clone()
                sub = attn[..., :i, :i].clone()
                attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=device)
            value_corrected = attn @ v_beta_c
            k_cumdecay = attn @ (k_beta_c * g_cum.exp().unsqueeze(-1))

        if _timing:
            torch.cuda.synchronize(device)
            _ci_t1 = time.perf_counter()

        # Recurrent chunk loop
        core_attn_out = torch.zeros_like(value_corrected)
        step_fn = _get_chunk_step()

        for i in range(num_chunks):
            output, recurrent_state = step_fn(
                q_c[:, :, i], k_c[:, :, i], value_corrected[:, :, i],
                g_cum[:, :, i], k_cumdecay[:, :, i], decay_mask[:, :, i],
                mask_strict_upper, recurrent_state)
            core_attn_out[:, :, i] = output

        if _timing:
            torch.cuda.synchronize(device)
            _ci_t2 = time.perf_counter()
            logger.info(
                "  LA-INNER chunks=%d: nilpotent=%.1fms loop=%.1fms (%.2fms/chunk)",
                num_chunks, (_ci_t1 - _ci_t0) * 1000, (_ci_t2 - _ci_t1) * 1000,
                (_ci_t2 - _ci_t1) * 1000 / max(num_chunks, 1))

        return core_attn_out, recurrent_state

    def _forward_chunked(self, hidden: torch.Tensor) -> torch.Tensor:
        """Chunked prefill: matches HF torch_chunk_gated_delta_rule.

        Uses the parallel-within-chunk, recurrent-across-chunks formulation.
        """
        _timing = TIMING.prefill
        M = hidden.shape[0]
        chunk_size = 64

        if _timing:
            torch.cuda.synchronize(hidden.device)
            _t0 = time.perf_counter()

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

        if _timing:
            torch.cuda.synchronize(hidden.device)
            _t1 = time.perf_counter()

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

        if _timing:
            torch.cuda.synchronize(hidden.device)
            _t2 = time.perf_counter()

        # Reshape to chunks: [1, nv, num_chunks, chunk_size, dim]
        num_chunks = total_len // chunk_size
        q_c = q_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        k_c = k_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        v_c = v_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.v_head_dim)
        k_beta_c = k_beta.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        v_beta_c = v_beta.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.v_head_dim)
        g_c = g_3d.reshape(1, self.num_v_heads, num_chunks, chunk_size)

        # Nilpotent correction + recurrent chunk loop (shared implementation)
        core_attn_out, self._recurrent_state = self._chunked_inner(
            self._recurrent_state, q_c, k_c, v_beta_c, k_beta_c, g_c,
            chunk_size, num_chunks)

        if _timing:
            torch.cuda.synchronize(hidden.device)
            _t3 = time.perf_counter()

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
        result = _linear(attn_flat, self.out_proj)  # [M, hidden]

        if _timing:
            torch.cuda.synchronize(hidden.device)
            _t4 = time.perf_counter()
            logger.info(
                "LA-PREFILL L%s(%s) M=%d chunks=%d: proj+conv=%.1fms prep=%.1fms "
                "inner(nilpotent+loop)=%.1fms norm+outproj=%.1fms total=%.1fms",
                self.layer_idx, hidden.device, M, num_chunks,
                (_t1 - _t0) * 1000, (_t2 - _t1) * 1000,
                (_t3 - _t2) * 1000, (_t4 - _t3) * 1000,
                (_t4 - _t0) * 1000)

        return result

    def _forward_recurrent_no_outproj(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full recurrent forward without out_proj. Returns flat [M, nv*dv]."""
        M = hidden.shape[0]

        qkvz = _linear(hidden, self.in_proj_qkvz)
        ba = _linear(hidden, self.in_proj_ba)
        q, k, v, z, b, a = self._fix_query_key_value_ordering(qkvz, ba)

        q_flat = q.reshape(M, self.key_dim)
        k_flat = k.reshape(M, self.key_dim)
        v_flat = v.reshape(M, self.value_dim)

        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)
        mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1, 2)

        conv_input = torch.cat([self._conv_state, mixed_qkv], dim=-1)
        self._conv_state = conv_input[:, :, -self.kernel_dim:].clone()

        conv_out = F.conv1d(
            conv_input.to(self.conv1d_weight.dtype),
            self.conv1d_weight, bias=None, padding=0, groups=self.conv_dim,
        )
        conv_out = F.silu(conv_out[:, :, -M:]).to(hidden.dtype)
        conv_out = conv_out.transpose(1, 2).squeeze(0)

        q_out = conv_out[:, :self.key_dim].reshape(M, self.num_k_heads, self.k_head_dim)
        k_out = conv_out[:, self.key_dim:self.key_dim * 2].reshape(M, self.num_k_heads, self.k_head_dim)
        v_out = conv_out[:, self.key_dim * 2:].reshape(M, self.num_v_heads, self.v_head_dim)

        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.head_ratio > 1:
            q_out = q_out.repeat_interleave(self.head_ratio, dim=1)
            k_out = k_out.repeat_interleave(self.head_ratio, dim=1)

        q_out = _l2norm(q_out, dim=-1) * self.scale
        k_out = _l2norm(k_out, dim=-1)

        outputs = []
        for t in range(M):
            q_t = q_out[t]
            k_t = k_out[t]
            v_t = v_out[t]
            g_t = g[t].exp()
            beta_t = beta[t]

            self._recurrent_state = self._recurrent_state * g_t.unsqueeze(-1).unsqueeze(-1)
            kv_mem = (self._recurrent_state.squeeze(0) * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)
            self._recurrent_state = self._recurrent_state + k_t.unsqueeze(-1).unsqueeze(0) * delta.unsqueeze(-2).unsqueeze(0)
            out_t = (self._recurrent_state.squeeze(0) * q_t.unsqueeze(-1)).sum(dim=-2)
            outputs.append(out_t)

        attn_out = torch.stack(outputs, dim=0)
        attn_out = self._gated_rmsnorm(attn_out, z)
        return attn_out.reshape(M, self.num_v_heads * self.v_head_dim).to(torch.bfloat16)

    def _forward_chunked_no_outproj(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full chunked forward without out_proj. Returns flat [M, nv*dv]."""
        M = hidden.shape[0]
        chunk_size = 64

        qkvz = _linear(hidden, self.in_proj_qkvz)
        ba = _linear(hidden, self.in_proj_ba)
        q, k, v, z, b, a = self._fix_query_key_value_ordering(qkvz, ba)

        q_flat = q.reshape(M, self.key_dim)
        k_flat = k.reshape(M, self.key_dim)
        v_flat = v.reshape(M, self.value_dim)

        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)
        mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1, 2)

        conv_input = torch.cat([self._conv_state, mixed_qkv], dim=-1)
        self._conv_state = conv_input[:, :, -self.kernel_dim:].clone()

        conv_out = F.conv1d(
            conv_input.to(self.conv1d_weight.dtype),
            self.conv1d_weight, bias=None, padding=0, groups=self.conv_dim,
        )
        conv_out = F.silu(conv_out[:, :, -M:]).to(hidden.dtype)
        conv_out = conv_out.transpose(1, 2).squeeze(0)

        q_all = conv_out[:, :self.key_dim].reshape(M, self.num_k_heads, self.k_head_dim)
        k_all = conv_out[:, self.key_dim:self.key_dim * 2].reshape(M, self.num_k_heads, self.k_head_dim)
        v_all = conv_out[:, self.key_dim * 2:].reshape(M, self.num_v_heads, self.v_head_dim)

        beta_all = torch.sigmoid(b)
        g_all = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.head_ratio > 1:
            q_all = q_all.repeat_interleave(self.head_ratio, dim=1)
            k_all = k_all.repeat_interleave(self.head_ratio, dim=1)

        q_all = _l2norm(q_all, dim=-1) * self.scale
        k_all = _l2norm(k_all, dim=-1)

        q_all = q_all.float()
        k_all = k_all.float()
        v_all = v_all.float()
        beta_all = beta_all.float()

        pad_size = (chunk_size - M % chunk_size) % chunk_size
        if pad_size > 0:
            q_all = F.pad(q_all, (0, 0, 0, 0, 0, pad_size))
            k_all = F.pad(k_all, (0, 0, 0, 0, 0, pad_size))
            v_all = F.pad(v_all, (0, 0, 0, 0, 0, pad_size))
            beta_all = F.pad(beta_all, (0, 0, 0, pad_size))
            g_all = F.pad(g_all, (0, 0, 0, pad_size))
        total_len = M + pad_size

        q_4d = q_all.unsqueeze(0).transpose(1, 2)
        k_4d = k_all.unsqueeze(0).transpose(1, 2)
        v_4d = v_all.unsqueeze(0).transpose(1, 2)
        beta_3d = beta_all.unsqueeze(0).transpose(1, 2)
        g_3d = g_all.unsqueeze(0).transpose(1, 2)

        v_beta = v_4d * beta_3d.unsqueeze(-1)
        k_beta = k_4d * beta_3d.unsqueeze(-1)

        num_chunks = total_len // chunk_size
        q_c = q_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        k_c = k_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        v_c = v_4d.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.v_head_dim)
        k_beta_c = k_beta.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.k_head_dim)
        v_beta_c = v_beta.reshape(1, self.num_v_heads, num_chunks, chunk_size, self.v_head_dim)
        g_c = g_3d.reshape(1, self.num_v_heads, num_chunks, chunk_size)

        core_attn_out, self._recurrent_state = self._chunked_inner(
            self._recurrent_state, q_c, k_c, v_beta_c, k_beta_c, g_c,
            chunk_size, num_chunks)

        core_attn_out = core_attn_out.reshape(1, self.num_v_heads, -1, self.v_head_dim)
        core_attn_out = core_attn_out[:, :, :M, :]
        core_attn_out = core_attn_out.transpose(1, 2).squeeze(0)
        core_attn_out = core_attn_out.to(hidden.dtype)

        attn_out = self._gated_rmsnorm(core_attn_out, z)
        return attn_out.reshape(M, self.num_v_heads * self.v_head_dim).to(torch.bfloat16)

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

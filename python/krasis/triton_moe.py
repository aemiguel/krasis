"""Triton GEMV kernel for M=1 MoE decode on auxiliary GPUs.

The original sglang moe_wna16_marlin_gemm crashed on non-primary GPUs because
its CUDA dispatch defaults to device 0 without calling cudaSetDevice().
This module provides a Triton-based alternative that works on any device.

Weight format: standard GPTQ-packed INT4
  - weights: [N_out, K_in // 8] int32  (8 INT4 values per int32, packed along K)
  - scales:  [N_out, K_in // group_size] bf16  (one scale per group per output)

At HCS init, Marlin-packed weights from the engine are converted to this
standard format via inverse_marlin_repack() / inverse_scale_permute().
"""

import torch
import triton
import triton.language as tl

# ─── Marlin permutation tables ───────────────────────────────────────────────


def _generate_weight_perm_int4() -> list[int]:
    """Generate 1024-element Marlin weight permutation table for INT4.

    Matches vLLM's get_weight_perm(num_bits=4) and Rust generate_weight_perm_int4().
    """
    perm: list[int] = []
    for i in range(32):
        col = i // 4
        perm1: list[int] = []
        for block in range(2):
            for row_offset in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row_offset + col + 8 * block)
        for j in range(4):
            for p in perm1:
                perm.append(p + 256 * j)

    interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    result = [0] * 1024
    for group in range(1024 // 8):
        for dest, src in enumerate(interleave):
            result[group * 8 + dest] = perm[group * 8 + src]
    return result


def _generate_scale_perms() -> tuple[list[int], list[int]]:
    """Generate scale permutation tables.

    Returns (scale_perm_64, scale_perm_single_32).
    """
    scale_perm = [0] * 64
    for i in range(8):
        for j in range(8):
            scale_perm[i * 8 + j] = i + 8 * j

    offsets = [0, 1, 8, 9, 16, 17, 24, 25]
    scale_perm_single = [0] * 32
    for i in range(4):
        for j, off in enumerate(offsets):
            scale_perm_single[i * 8 + j] = 2 * i + off

    return scale_perm, scale_perm_single


# ─── Inverse Marlin repack ───────────────────────────────────────────────────


def inverse_marlin_repack(
    w_marlin: torch.Tensor,
    K: int,
    N: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Convert Marlin-packed INT4 weights to standard GPTQ packed format.

    Args:
        w_marlin: [..., K//16, 2*N] int32 (Marlin format, any leading batch dims)
        K: input dimension (reduction dim)
        N: output dimension
        num_bits: quantization bits (4)

    Returns:
        [..., N, K//pack_factor] int32 (standard GPTQ packed)
    """
    assert num_bits == 4, f"Only INT4 supported, got {num_bits}"
    pack_factor = 8  # 32 // 4
    k_tiles = K // 16
    n_tiles = N // 16
    row_len = N * 16  # values per tiled row
    out_cols = 2 * N  # Marlin packed columns = row_len // pack_factor

    batch_shape = w_marlin.shape[:-2]
    device = w_marlin.device
    w = w_marlin.reshape(-1, k_tiles, out_cols)  # [B, K//16, 2*N]
    B = w.shape[0]

    # Step 1: Unpack int32 → individual INT4 values (unsigned 0-15)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4  # [8]
    expanded = ((w.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    # [B, k_tiles, out_cols, 8] → [B, k_tiles, out_cols * 8] = [B, k_tiles, row_len]
    perm_applied = expanded.reshape(B, k_tiles, row_len)

    # Step 2: Inverse weight permutation (per 1024-element chunk)
    weight_perm = _generate_weight_perm_int4()
    inv_perm = [0] * 1024
    for i, p in enumerate(weight_perm):
        inv_perm[p] = i
    inv_perm_t = torch.tensor(inv_perm, dtype=torch.long, device=device)

    num_chunks = row_len // 1024
    perm_applied_chunked = perm_applied.reshape(B, k_tiles, num_chunks, 1024)
    permuted = perm_applied_chunked[:, :, :, inv_perm_t]
    permuted = permuted.reshape(B, k_tiles, row_len)

    # Step 3: Inverse tile transpose
    # permuted is [B, k_tiles, n_tiles * 16 * 16] → reshape to [B, kt, nt, tk=16, tn=16]
    # where permuted[b, kt, nt, tk, tn] = original[b, nt*16+tn, kt*16+tk]
    tiled = permuted.reshape(B, k_tiles, n_tiles, 16, 16)
    # Recover original[b, n, k] = tiled[b, n//16, n%16, k//16, k%16]
    # Permute: (b, kt, nt, tk, tn) → (b, nt, tn, kt, tk) → reshape (b, N, K)
    original = tiled.permute(0, 2, 4, 1, 3).reshape(B, N, K)

    # Step 4: Repack to standard [N, K//8] int32
    original_groups = original.reshape(B, N, K // pack_factor, pack_factor).to(torch.int32)
    shifts_pack = torch.arange(pack_factor, device=device, dtype=torch.int32) * 4
    packed = (original_groups << shifts_pack).sum(dim=-1, dtype=torch.int32)

    return packed.reshape(*batch_shape, N, K // pack_factor)


def inverse_scale_permute(
    s_marlin: torch.Tensor,
    K: int,
    N: int,
    group_size: int,
) -> torch.Tensor:
    """Un-permute Marlin scales to standard GPTQ format.

    Args:
        s_marlin: [..., K//gs, N] bf16 (Marlin permuted + transposed scales)

    Returns:
        [..., N, K//gs] bf16 (standard format: output dim first)
    """
    num_groups_k = K // group_size
    batch_shape = s_marlin.shape[:-2]
    device = s_marlin.device
    s = s_marlin.reshape(-1, num_groups_k, N)  # [B, K//gs, N]
    B = s.shape[0]

    # Step 1: Inverse scale permutation
    scale_perm, scale_perm_single = _generate_scale_perms()
    is_grouped = group_size < K
    sperm = scale_perm if is_grouped else scale_perm_single
    perm_len = len(sperm)

    inv_sperm = [0] * perm_len
    for i, p in enumerate(sperm):
        inv_sperm[p] = i
    inv_sperm_t = torch.tensor(inv_sperm, dtype=torch.long, device=device)

    s_flat = s.reshape(B, -1)  # [B, num_groups_k * N]
    total = s_flat.shape[1]
    num_chunks = total // perm_len
    s_chunked = s_flat.reshape(B, num_chunks, perm_len)
    s_unpermuted = s_chunked[:, :, inv_sperm_t]
    s_unpermuted = s_unpermuted.reshape(B, num_groups_k, N)

    # Step 2: Transpose [K//gs, N] → [N, K//gs]
    s_standard = s_unpermuted.permute(0, 2, 1).contiguous()

    return s_standard.reshape(*batch_shape, N, num_groups_k)


# ─── Triton GEMV kernel ──────────────────────────────────────────────────────


@triton.jit
def _int4_gemv_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    N,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT4 dequant + GEMV for M=1.

    Computes out[n] = sum_k x[k] * dequant(W[n, k]) for a block of N outputs.
    W is [N, K//8] int32 standard packed, scales are [N, K//group_size] bf16.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load x values: [BLOCK_K]
        x_vals = tl.load(x_ptr + k_offsets).to(tl.float32)

        # Load packed INT4 weights: need w[n, k//8] for each (n, k)
        pack_idx = k_offsets // 8  # [BLOCK_K]
        bit_shift = ((k_offsets % 8) * 4).to(tl.int32)  # [BLOCK_K]
        w_offsets = n_offsets[:, None] * K_PACKED + pack_idx[None, :]  # [BLOCK_N, BLOCK_K]
        w_packed = tl.load(w_ptr + w_offsets, mask=n_mask[:, None], other=0)

        # Extract INT4, convert to signed float
        w_uint4 = (w_packed >> bit_shift[None, :]) & 0xF
        w_float = (w_uint4 - 8).to(tl.float32)

        # Load scale: one per group per output element
        # BLOCK_K == GROUP_SIZE, so one scale per k_start
        group_idx = k_start // GROUP_SIZE
        scale_offsets = n_offsets * K_GROUPS + group_idx
        scales = tl.load(scale_ptr + scale_offsets, mask=n_mask, other=1.0).to(tl.float32)

        # Dequantize and dot product
        w_dequant = w_float * scales[:, None]
        acc += tl.sum(w_dequant * x_vals[None, :], axis=1)

    tl.store(out_ptr + n_offsets, acc.to(tl.bfloat16), mask=n_mask)


def triton_gemv_int4(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """GEMV with INT4 dequantization: y = x @ W^T + bias.

    Args:
        x: [1, K] bf16 input
        w_packed: [N, K//8] int32 standard packed INT4
        scale: [N, K//group_size] bf16
        group_size: quantization group size (default 128)
        bias: [N] bf16 optional

    Returns:
        [1, N] bf16
    """
    K = x.shape[1]
    N = w_packed.shape[0]
    K_packed = K // 8
    K_groups = K // group_size

    assert w_packed.shape == (N, K_packed), f"w_packed shape {w_packed.shape} != ({N}, {K_packed})"
    assert scale.shape == (N, K_groups), f"scale shape {scale.shape} != ({N}, {K_groups})"

    out = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    x_flat = x.view(-1)  # [K]

    BLOCK_N = 128
    BLOCK_K = group_size  # match group_size for clean scale loading
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _int4_gemv_kernel[grid](
        x_flat, w_packed, scale, out,
        N, K, K_packed, group_size, K_groups,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    if bias is not None:
        out = out + bias

    return out.unsqueeze(0)  # [1, N]


# ─── Per-expert forward ──────────────────────────────────────────────────────


def triton_expert_forward(
    x: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    swiglu_limit: float,
    activation_alpha: float,
    group_size: int = 128,
) -> torch.Tensor:
    """Single expert forward: GEMM1 → activation → GEMM2.

    Args:
        x: [1, K] bf16
        w13_packed: [2*N, K//8] int32 (gate+up projection, standard packed)
        w13_scale: [2*N, K//group_size] bf16
        w2_packed: [K, N//8] int32 (down projection, standard packed)
        w2_scale: [K, N//group_size] bf16
        gate_up_bias: [2*N] bf16 or None
        down_bias: [K] bf16 or None
        swiglu_limit: GPT OSS clamping limit (0.0 for standard SiLU)
        activation_alpha: GPT OSS sigmoid alpha (1.702)
        group_size: quantization group size

    Returns:
        [1, K] bf16 expert output (NOT weighted by routing weight)
    """
    # GEMM1: x @ w13^T → [1, 2*N]
    gate_up = triton_gemv_int4(x, w13_packed, w13_scale, group_size, gate_up_bias)

    # Split gate and up projections
    N2 = gate_up.shape[1]
    N = N2 // 2
    gate = gate_up[:, :N]
    up = gate_up[:, N:]

    # Activation
    if swiglu_limit > 0:
        # GPT OSS: gate * sigmoid(gate * alpha) * (up + 1) with clamping
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        activated = gate * torch.sigmoid(gate * activation_alpha) * (up + 1.0)
    else:
        # Standard SwiGLU: silu(gate) * up
        activated = torch.nn.functional.silu(gate) * up

    # GEMM2: activated @ w2^T → [1, K]
    output = triton_gemv_int4(activated, w2_packed, w2_scale, group_size, down_bias)

    return output


def triton_moe_decode(
    x: torch.Tensor,
    expert_ids: list[int],
    expert_weights: list[float],
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    lookup: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    swiglu_limit: float,
    activation_alpha: float,
    group_size: int = 128,
) -> torch.Tensor:
    """Full MoE decode for one layer on a single device using Triton.

    Loops over activated experts, runs each through triton_expert_forward,
    weights and sums the results.

    Args:
        x: [1, K] bf16
        expert_ids: list of global expert IDs activated for this token
        expert_weights: list of routing weights for each expert
        w13_packed: [n_pinned, 2*N, K//8] int32
        w13_scale: [n_pinned, 2*N, K//gs] bf16
        w2_packed: [n_pinned, K, N//8] int32
        w2_scale: [n_pinned, K, N//gs] bf16
        lookup: [num_experts] int32 tensor mapping global ID → local slot (-1 = not here)
        gate_up_bias: [n_pinned, 2*N] bf16 or None
        down_bias: [n_pinned, K] bf16 or None
        swiglu_limit: GPT OSS clamping limit
        activation_alpha: GPT OSS sigmoid alpha
        group_size: quantization group size

    Returns:
        [1, K] bf16 weighted sum of expert outputs
    """
    K = x.shape[1]
    result = torch.zeros(1, K, dtype=x.dtype, device=x.device)

    for eid, weight in zip(expert_ids, expert_weights):
        if weight == 0.0:
            continue
        slot = int(lookup[eid])
        if slot < 0:
            continue

        gu_bias = gate_up_bias[slot] if gate_up_bias is not None else None
        dn_bias = down_bias[slot] if down_bias is not None else None

        expert_out = triton_expert_forward(
            x,
            w13_packed[slot],
            w13_scale[slot],
            w2_packed[slot],
            w2_scale[slot],
            gu_bias,
            dn_bias,
            swiglu_limit,
            activation_alpha,
            group_size,
        )
        result += expert_out * weight

    return result

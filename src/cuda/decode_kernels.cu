// Krasis GPU decode kernels — compiled via NVRTC at runtime.
// All kernels operate on BF16 hidden states with FP32 intermediates.
// Target: compute_89 (Ada Lovelace), also works on sm_80+ (Ampere).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ── Helpers ────────────────────────────────────────────────────────────

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ── Embedding Lookup ───────────────────────────────────────────────────

// Copy one row from embedding table [vocab, hidden] BF16 into hidden state BF16.
extern "C" __global__ void embedding_lookup(
    __nv_bfloat16* __restrict__ output,      // [hidden_size]
    const __nv_bfloat16* __restrict__ table,  // [vocab_size, hidden_size]
    int token_id,
    int hidden_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        output[i] = table[token_id * hidden_size + i];
    }
}

// ── RMSNorm ────────────────────────────────────────────────────────────

// Fused residual add + RMSNorm.
// If first_layer: residual = hidden; hidden = RMSNorm(hidden, weight)
// Else: hidden += residual; residual = hidden; hidden = RMSNorm(hidden, weight)
//
// Uses warp reduction for the sum-of-squares. One block per call.
// hidden_size must be <= 8192 (fits in shared memory for one block).
extern "C" __global__ void fused_add_rmsnorm(
    __nv_bfloat16* __restrict__ hidden,     // [hidden_size] in/out
    __nv_bfloat16* __restrict__ residual,   // [hidden_size] in/out
    const __nv_bfloat16* __restrict__ weight, // [hidden_size]
    float eps,
    int hidden_size,
    int first_layer  // 1 = first layer (no add), 0 = add residual
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Step 1: Load hidden into shared mem as FP32, optionally add residual
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += num_threads) {
        float h = bf16_to_f32(hidden[i]);
        if (!first_layer) {
            float r = bf16_to_f32(residual[i]);
            h += r;
        }
        smem[i] = h;
        sum_sq += h * h;
    }

    // Warp reduction for sum_sq
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block reduction across warps using shared memory
    __shared__ float warp_sums[32]; // max 32 warps per block
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) total += warp_sums[w];
        // Store the RMS scale factor
        warp_sums[0] = rsqrtf(total / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms_scale = warp_sums[0];

    // Step 2: Write residual = pre-norm value, hidden = normalized * weight
    for (int i = tid; i < hidden_size; i += num_threads) {
        float h = smem[i];
        residual[i] = f32_to_bf16(h);  // save pre-norm value
        float w = bf16_to_f32(weight[i]);
        hidden[i] = f32_to_bf16(h * rms_scale * w);
    }
}

// Simple RMSNorm (no residual, no fused add). Used for final norm.
extern "C" __global__ void rmsnorm(
    __nv_bfloat16* __restrict__ output,      // [hidden_size]
    const __nv_bfloat16* __restrict__ input,  // [hidden_size]
    const __nv_bfloat16* __restrict__ weight, // [hidden_size]
    float eps,
    int hidden_size
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += num_threads) {
        float x = bf16_to_f32(input[i]);
        smem[i] = x;
        sum_sq += x * x;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) total += warp_sums[w];
        warp_sums[0] = rsqrtf(total / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms_scale = warp_sums[0];

    for (int i = tid; i < hidden_size; i += num_threads) {
        float w = bf16_to_f32(weight[i]);
        output[i] = f32_to_bf16(smem[i] * rms_scale * w);
    }
}

// ── SiLU * Mul (gate activation) ──────────────────────────────────────

// Fused gate_proj * silu(gate_proj) * up_proj operation for MLP.
// Input: gate_up[2 * intermediate_size] = concat(gate, up) from fused matmul.
// Output: result[intermediate_size] = silu(gate[i]) * up[i]
extern "C" __global__ void silu_mul(
    __nv_bfloat16* __restrict__ output,        // [intermediate_size]
    const __nv_bfloat16* __restrict__ gate_up,  // [2 * intermediate_size]
    int intermediate_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate_size) {
        float g = bf16_to_f32(gate_up[i]);
        float u = bf16_to_f32(gate_up[intermediate_size + i]);
        output[i] = f32_to_bf16(silu(g) * u);
    }
}

// ── Sigmoid + TopK for MoE Routing ────────────────────────────────────

// Apply sigmoid to gate logits, optionally add bias and e_score_correction,
// then find topk expert indices and weights.
// This is a single-block kernel since num_experts is small (64-512).
extern "C" __global__ void sigmoid_topk(
    const float* __restrict__ logits,           // [num_experts]
    const float* __restrict__ bias,             // [num_experts] or NULL
    const float* __restrict__ e_score_corr,     // [num_experts] or NULL
    int* __restrict__ topk_indices,             // [topk]
    float* __restrict__ topk_weights,           // [topk]
    int num_experts,
    int topk
) {
    // Single-threaded for simplicity (num_experts <= 512, topk <= 16)
    // This is NOT the bottleneck — gate matmul is.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Compute sigmoid scores
    extern __shared__ float scores[];
    for (int i = 0; i < num_experts; i++) {
        float x = logits[i];
        if (bias) x += bias[i];
        scores[i] = 1.0f / (1.0f + expf(-x));
        if (e_score_corr) scores[i] += e_score_corr[i];
    }

    // Simple selection sort for topk (small k)
    for (int t = 0; t < topk; t++) {
        int best_idx = -1;
        float best_val = -1e30f;
        for (int i = 0; i < num_experts; i++) {
            if (scores[i] > best_val) {
                best_val = scores[i];
                best_idx = i;
            }
        }
        topk_indices[t] = best_idx;
        topk_weights[t] = best_val;
        scores[best_idx] = -1e30f; // mask out selected
    }

    // Normalize weights
    float sum = 0.0f;
    for (int t = 0; t < topk; t++) sum += topk_weights[t];
    if (sum > 0.0f) {
        for (int t = 0; t < topk; t++) topk_weights[t] /= sum;
    }
}

// Softmax + TopK for models that use softmax routing (e.g. DeepSeek)
extern "C" __global__ void softmax_topk(
    const float* __restrict__ logits,       // [num_experts]
    int* __restrict__ topk_indices,         // [topk]
    float* __restrict__ topk_weights,       // [topk]
    int num_experts,
    int topk
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    extern __shared__ float scores[];

    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < num_experts; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < num_experts; i++) {
        scores[i] = expf(logits[i] - max_val);
        sum_exp += scores[i];
    }
    for (int i = 0; i < num_experts; i++) {
        scores[i] /= sum_exp;
    }

    // TopK selection
    for (int t = 0; t < topk; t++) {
        int best_idx = -1;
        float best_val = -1e30f;
        for (int i = 0; i < num_experts; i++) {
            if (scores[i] > best_val) {
                best_val = scores[i];
                best_idx = i;
            }
        }
        topk_indices[t] = best_idx;
        topk_weights[t] = best_val;
        scores[best_idx] = -1e30f;
    }
}

// ── Vector Operations ──────────────────────────────────────────────────

// Weighted add: output += weight * input (for accumulating expert outputs)
extern "C" __global__ void weighted_add_bf16(
    __nv_bfloat16* __restrict__ output,      // [size]
    const __nv_bfloat16* __restrict__ input,  // [size]
    float weight,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float o = bf16_to_f32(output[i]);
        float x = bf16_to_f32(input[i]);
        output[i] = f32_to_bf16(o + weight * x);
    }
}

// Zero a BF16 buffer
extern "C" __global__ void zero_bf16(
    __nv_bfloat16* __restrict__ buf,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        buf[i] = f32_to_bf16(0.0f);
    }
}

// Add two BF16 vectors: output = a + b
extern "C" __global__ void add_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
    }
}

// Multiply BF16 by sigmoid gate: output = input * sigmoid(gate)
// Used for shared expert gating: output = shared_expert_out * sigmoid(gate_weight @ hidden)
extern "C" __global__ void sigmoid_gate_bf16(
    __nv_bfloat16* __restrict__ output,       // [size]
    const __nv_bfloat16* __restrict__ input,   // [size]
    float gate_value,  // scalar sigmoid(gate_logit) pre-computed
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = f32_to_bf16(bf16_to_f32(input[i]) * gate_value);
    }
}

// Scale BF16 vector by a scalar: output[i] = input[i] * scale
extern "C" __global__ void scale_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    float scale,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = f32_to_bf16(bf16_to_f32(input[i]) * scale);
    }
}

// ── Linear Attention Convolution ───────────────────────────────────────

// 1D causal convolution for linear attention (Mamba-style).
// Shifts conv_state, inserts new input, computes conv output.
// conv_state: [conv_dim, kernel_dim] (each of conv_dim channels has kernel_dim history)
extern "C" __global__ void la_conv1d(
    float* __restrict__ conv_state,          // [conv_dim, kernel_dim]
    const float* __restrict__ input,          // [conv_dim] (new input from projection)
    float* __restrict__ output,               // [conv_dim]
    const float* __restrict__ conv_weight,    // [conv_dim, kernel_dim]
    int conv_dim,
    int kernel_dim
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < conv_dim) {
        // Shift state left by 1
        for (int k = 0; k < kernel_dim - 1; k++) {
            conv_state[c * kernel_dim + k] = conv_state[c * kernel_dim + k + 1];
        }
        // Insert new input at the end
        conv_state[c * kernel_dim + (kernel_dim - 1)] = input[c];

        // Compute convolution output (dot product of state with weight)
        float out = 0.0f;
        for (int k = 0; k < kernel_dim; k++) {
            out += conv_state[c * kernel_dim + k] * conv_weight[c * kernel_dim + k];
        }
        output[c] = out;
    }
}

// ── Linear Attention Recurrence (SSM state update) ─────────────────────

// Per-head recurrent state update for gated delta net:
//   state[i,j] = gate * state[i,j] + beta * k[i] * v[j]
//   output[j] = sum_i(q[i] * state[i,j])
// One block per head, threads iterate over the [dk, dv] matrix.
extern "C" __global__ void la_recurrence(
    float* __restrict__ state,   // [nv, dk, dv]
    const float* __restrict__ q, // [nv * dk]
    const float* __restrict__ k, // [nv * dk]
    const float* __restrict__ v, // [nv * dv]
    const float* __restrict__ gate, // [nv] (per-head gate)
    const float* __restrict__ beta, // [nv] (per-head beta)
    float* __restrict__ output,  // [nv * dv]
    int nv, int dk, int dv
) {
    int head = blockIdx.x;
    if (head >= nv) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float g = gate[head];
    float b = beta[head];

    float* head_state = state + head * dk * dv;
    const float* head_q = q + head * dk;
    const float* head_k = k + head * dk;
    const float* head_v = v + head * dv;
    float* head_out = output + head * dv;

    // Each thread handles a subset of the [dk, dv] elements
    int total = dk * dv;
    for (int idx = tid; idx < total; idx += num_threads) {
        int i = idx / dv;  // dk dimension
        int j = idx % dv;  // dv dimension
        head_state[idx] = g * head_state[idx] + b * head_k[i] * head_v[j];
    }
    __syncthreads();

    // Compute output: output[j] = sum_i(q[i] * state[i, j])
    for (int j = tid; j < dv; j += num_threads) {
        float acc = 0.0f;
        for (int i = 0; i < dk; i++) {
            acc += head_q[i] * head_state[i * dv + j];
        }
        head_out[j] = acc;
    }
}

// ── Gated RMSNorm + SiLU ──────────────────────────────────────────────

// For linear attention output: output = silu(z) * rmsnorm(recurrence_out, weight)
// z and recurrence_out have different sizes: z is [nv*dv], recurrence_out is [nv*dv]
extern "C" __global__ void gated_rmsnorm_silu(
    float* __restrict__ output,              // [nv * dv]
    const float* __restrict__ recur_out,      // [nv * dv]
    const float* __restrict__ z,              // [nv * dv]
    const float* __restrict__ norm_weight,    // [dv] (shared across heads)
    float eps,
    int nv, int dv
) {
    int head = blockIdx.x;
    if (head >= nv) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* r = recur_out + head * dv;
    const float* gate = z + head * dv;
    float* out = output + head * dv;

    // RMSNorm over dv elements for this head
    extern __shared__ float smem[];
    float sum_sq = 0.0f;
    for (int i = tid; i < dv; i += num_threads) {
        float x = r[i];
        smem[i] = x;
        sum_sq += x * x;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) total += warp_sums[w];
        warp_sums[0] = rsqrtf(total / (float)dv + eps);
    }
    __syncthreads();
    float rms_scale = warp_sums[0];

    // Apply: output = silu(z) * rmsnorm(recur_out)
    for (int i = tid; i < dv; i += num_threads) {
        float normed = smem[i] * rms_scale * norm_weight[i];
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * normed;
    }
}

// ── GQA Attention Helpers ──────────────────────────────────────────────

// Per-head RMSNorm for Q/K (QK norm before RoPE)
extern "C" __global__ void per_head_rmsnorm(
    float* __restrict__ data,        // [num_heads * head_dim]
    const float* __restrict__ weight, // [head_dim] or [num_heads * head_dim]
    float eps,
    int num_heads,
    int head_dim,
    int weight_per_head  // 1 if weight is [num_heads * head_dim], 0 if [head_dim]
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float* h = data + head * head_dim;
    const float* w = weight_per_head ? (weight + head * head_dim) : weight;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += num_threads) {
        sum_sq += h[i] * h[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) total += warp_sums[w];
        warp_sums[0] = rsqrtf(total / (float)head_dim + eps);
    }
    __syncthreads();
    float rms_scale = warp_sums[0];

    for (int i = tid; i < head_dim; i += num_threads) {
        h[i] = h[i] * rms_scale * w[i];
    }
}

// RoPE (rotary position encoding) applied to Q and K
extern "C" __global__ void apply_rope(
    float* __restrict__ q,          // [num_q_heads * head_dim]
    float* __restrict__ k,          // [num_kv_heads * head_dim]
    const float* __restrict__ cos_table,  // [max_seq * half_dim]
    const float* __restrict__ sin_table,  // [max_seq * half_dim]
    int position,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim   // head_dim / 2 (rotary dim)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = num_q_heads + num_kv_heads;
    int total_work = total_heads * half_dim;
    if (tid >= total_work) return;

    int head = tid / half_dim;
    int i = tid % half_dim;

    float cos_val = cos_table[position * half_dim + i];
    float sin_val = sin_table[position * half_dim + i];

    float* data;
    if (head < num_q_heads) {
        data = q + head * head_dim;
    } else {
        data = k + (head - num_q_heads) * head_dim;
    }

    float x1 = data[i];
    float x2 = data[half_dim + i];
    data[i] = x1 * cos_val - x2 * sin_val;
    data[half_dim + i] = x2 * cos_val + x1 * sin_val;
}

// Write K,V to FP16 KV cache at given position
extern "C" __global__ void kv_cache_write(
    __half* __restrict__ k_cache,   // [max_seq, kv_stride]
    __half* __restrict__ v_cache,   // [max_seq, kv_stride]
    const float* __restrict__ k,     // [kv_stride]
    const float* __restrict__ v,     // [kv_stride]
    int position,
    int kv_stride   // num_kv_heads * head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_stride) {
        k_cache[position * kv_stride + i] = __float2half(k[i]);
        v_cache[position * kv_stride + i] = __float2half(v[i]);
    }
}

// Single-query GQA attention: scores over all cached K, softmax, weighted V sum
// One block per Q head. Uses shared memory for scores.
extern "C" __global__ void gqa_attention(
    float* __restrict__ output,          // [num_q_heads * head_dim]
    const float* __restrict__ q,          // [num_q_heads * head_dim]
    const __half* __restrict__ k_cache,   // [max_seq, kv_stride]
    const __half* __restrict__ v_cache,   // [max_seq, kv_stride]
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,     // number of valid positions (position + 1)
    int max_seq
) {
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Which KV head does this Q head attend to?
    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int kv_stride = num_kv_heads * head_dim;

    const float* q_head = q + qh * head_dim;

    // Step 1: Compute attention scores (Q @ K^T) for all positions
    extern __shared__ float smem[];  // [seq_len] for scores, then reused
    // Each thread computes scores for a subset of positions
    float max_score = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        const __half* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            score += q_head[d] * __half2float(k_vec[d]);
        }
        score *= sm_scale;
        smem[pos] = score;
        if (score > max_score) max_score = score;
    }

    // Block-wide max reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_score, offset);
        if (other > max_score) max_score = other;
    }
    __shared__ float warp_maxes[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) warp_maxes[warp_id] = max_score;
    __syncthreads();
    if (tid == 0) {
        float gmax = warp_maxes[0];
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 1; w < num_warps; w++) {
            if (warp_maxes[w] > gmax) gmax = warp_maxes[w];
        }
        warp_maxes[0] = gmax;
    }
    __syncthreads();
    max_score = warp_maxes[0];

    // Step 2: Softmax (exp and normalize)
    float sum_exp = 0.0f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float val = expf(smem[pos] - max_score);
        smem[pos] = val;
        sum_exp += val;
    }

    // Block-wide sum reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    if (lane_id == 0) warp_maxes[warp_id] = sum_exp;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) total += warp_maxes[w];
        warp_maxes[0] = total;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_maxes[0];

    for (int pos = tid; pos < seq_len; pos += num_threads) {
        smem[pos] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V vectors
    float* out_head = output + qh * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            acc += smem[pos] * __half2float(v_cache[pos * kv_stride + kv_head * head_dim + d]);
        }
        out_head[d] = acc;
    }
}

// ── Type Conversions ───────────────────────────────────────────────────

// BF16 -> FP32
extern "C" __global__ void bf16_to_fp32(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = bf16_to_f32(input[i]);
    }
}

// FP32 -> BF16
extern "C" __global__ void fp32_to_bf16(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = f32_to_bf16(input[i]);
    }
}

// ── Marlin INT4 GEMV (GPU decode expert compute) ─────────────────────
//
// Computes output[N] = dequant(marlin_packed[K/16, 2*N], scales[K/gs, N]) @ input[K]
//
// Marlin format stores INT4 weights with tile permutation + weight perm
// optimized for warp-level GEMM. This kernel inverts the permutations
// on-the-fly for GEMV (M=1 decode).
//
// Launch: grid=(N/TILE_N, 1, 1), block=(TILE_N * K_SLICES, 1, 1)
// where TILE_N=16 (Marlin tile size), K_SLICES=16 (parallelism over K).
//
// Each block computes TILE_N=16 output elements.
// Within a block, K_SLICES=16 thread groups each handle K/(16*K_SLICES)
// k-tiles, then reduce via shared memory.

// Marlin INT4 GEMV kernel.
// Inverse perm tables are passed as arguments (uploaded from Rust).
extern "C" __global__ void marlin_gemv_int4(
    const unsigned int* __restrict__ packed,   // [K/16, 2*N] Marlin tile-permuted INT4
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    unsigned short* __restrict__ output,       // [N] BF16
    const int* __restrict__ inv_weight_perm,   // [1024] inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size
) {
    // Thread mapping: 256 threads per block
    // k_slice = threadIdx.x & 15  (0..15, which slice of K tiles)
    // tn = threadIdx.x >> 4       (0..15, which output in this tile)
    int k_slice = threadIdx.x & 15;
    int tn = threadIdx.x >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;    // K / 16
    int out_cols = N << 1;         // 2 * N (u32 columns in packed)
    int row_len = N << 4;          // N * 16 (INT4 values per k-tile row)

    // Distribute k_tiles across 16 slices
    int tiles_per_slice = k_tiles_total >> 4;  // k_tiles_total / 16
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Load input values for our k range into registers
    // (input is small enough that L2 handles repeated access well)

    float acc = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        // Process 16 k positions within this tile
        for (int tk = 0; tk < 16; tk++) {
            int k = (kt << 4) + tk;

            // --- Find weight value W[n, k] in Marlin packed format ---
            // Step 1: tile position in pre-perm array
            int tile_pos = (n_tile << 8) + (tk << 4) + tn;  // n_tile*256 + tk*16 + tn

            // Step 2: apply inverse weight perm (within 1024-element chunk)
            int chunk = tile_pos >> 10;        // tile_pos / 1024
            int local = tile_pos & 1023;       // tile_pos % 1024
            int perm_pos = (chunk << 10) + inv_weight_perm[local];

            // Step 3: extract INT4 from packed u32
            int u32_col = perm_pos >> 3;       // perm_pos / 8
            int nibble = perm_pos & 7;         // perm_pos % 8
            unsigned int word = packed[kt * out_cols + u32_col];
            int raw = (word >> (nibble << 2)) & 0xF;
            float w_val = (float)(raw - 8);    // symmetric INT4: [0,15] → [-8, 7]

            // --- Find scale for this (n, k) position ---
            int sg = k / group_size;           // scale group index
            int scale_flat = sg * N + n;       // position in [K/gs, N] array
            int schunk = scale_flat >> 6;      // / 64
            int slocal = scale_flat & 63;      // % 64
            int sperm_pos = (schunk << 6) + inv_scale_perm[slocal];

            // Read BF16 scale, convert to FP32
            unsigned short scale_bits = scales[sperm_pos];
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));

            // Read BF16 input, convert to FP32
            unsigned short x_bits = input[k];
            float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x_bits));

            acc += w_val * scale * x;
        }
    }

    // --- Reduce across 16 k_slices for this output element ---
    __shared__ float reduce[256];
    reduce[threadIdx.x] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = acc;  // Already have our own contribution
        int base = tn << 4;  // tn * 16
        for (int s = 1; s < 16; s++) {
            sum += reduce[base + s];
        }
        // Write BF16 output
        __nv_bfloat16 result = __float2bfloat16(sum);
        output[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

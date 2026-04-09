// Krasis GPU decode kernels — compiled via NVRTC at runtime.
// All kernels operate on BF16 hidden states with FP32 intermediates.
// Target: compute_89 (Ada Lovelace), also works on sm_80+ (Ampere).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ── Helpers ────────────────────────────────────────────────────────────

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_e4m3 x) {
    return float(x);
}

__device__ __forceinline__ __nv_fp8_e4m3 f32_to_fp8e4m3(float x) {
    return __nv_fp8_e4m3(x);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
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
        scores[i] = 1.0f / (1.0f + __expf(-x));
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
    int topk,
    int norm_topk_prob                      // 1=renormalize weights to sum to 1
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
        scores[i] = __expf(logits[i] - max_val);
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

    // Normalize top-K weights to sum to 1.0 (only when norm_topk_prob is set)
    if (norm_topk_prob) {
    float topk_sum = 0.0f;
    for (int t = 0; t < topk; t++) {
        topk_sum += topk_weights[t];
    }
    if (topk_sum > 0.0f) {
        float inv_sum = 1.0f / topk_sum;
        for (int t = 0; t < topk; t++) {
            topk_weights[t] *= inv_sum;
        }
    }
    } // end norm_topk_prob
}

// ── Expert Classify + Batch Prepare ───────────────────────────────────
//
// GPU-side route sync: after topk, check the device-side HCS lookup table
// and populate d_batch_upload directly for cached experts.
// Cold (uncached) expert IDs are written to mapped host memory for CPU to DMA.
// This eliminates cuStreamSynchronize and CPU-side classification.
//
// d_expert_ptrs layout: [num_layers * num_experts * 4] u64 values.
//   For expert (layer, eid): base = (layer * num_experts + eid) * 4
//     [base+0] = w13_packed VRAM ptr (0 if not cached)
//     [base+1] = w13_scales VRAM ptr
//     [base+2] = w2_packed VRAM ptr
//     [base+3] = w2_scales VRAM ptr
//
// d_batch_upload layout: [max_ept * 8] * 4 pointer arrays + [max_ept * 4] weights
//   Stride between arrays = max_ept * 8 bytes
//   Array 0: w13_packed ptrs [max_ept]
//   Array 1: w13_scales ptrs [max_ept]
//   Array 2: w2_packed ptrs  [max_ept]
//   Array 3: w2_scales ptrs  [max_ept]
//   Array 4: weights (f32)   [max_ept] at offset max_ept*8*4
//
// mapped_cold_buf layout (host-visible via PCIe BAR):
//   [0]:       cold_count (int32) — number of cold experts
//   [1]:       ready_flag (int32) — set to 1 when classification is done
//   [2..2+topk]: cold expert IDs (int32)
//   [2+topk..2+2*topk]: cold topk positions (int32) — slot index in batch
//
// Launch: grid=(1,1,1), block=(1,1,1), smem=0
// Single-threaded: topk <= 16, trivial work.

extern "C" __global__ void expert_classify_prepare(
    const int* __restrict__ topk_ids,           // [topk] from topk kernel
    const float* __restrict__ topk_weights,     // [topk] from topk kernel
    const unsigned long long* __restrict__ d_expert_ptrs, // [num_layers * num_experts * 4]
    unsigned long long* __restrict__ d_batch_upload,      // batch pointer table
    volatile int* __restrict__ mapped_cold_buf, // host-visible mapped memory
    int layer_idx,
    int num_experts,
    int topk,
    int max_ept,
    unsigned long long dummy_ptr,  // dummy expert pointer for unfilled slots (same for all 4)
    volatile int* __restrict__ mapped_activations, // [num_moe * topk] for post-hoc recording (NULL to skip)
    int moe_seq_idx   // sequential index of this MoE layer (0, 1, 2, ...)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Pointer strides in d_batch_upload (in u64 units)
    // Array layout: [w13p_ptrs | w13s_ptrs | w2p_ptrs | w2s_ptrs | weights(f32)]
    unsigned long long* w13p_arr = d_batch_upload;
    unsigned long long* w13s_arr = d_batch_upload + max_ept;
    unsigned long long* w2p_arr  = d_batch_upload + max_ept * 2;
    unsigned long long* w2s_arr  = d_batch_upload + max_ept * 3;
    float* wts_arr = (float*)(d_batch_upload + max_ept * 4);

    int batch_count = 0;
    int cold_count = 0;
    int layer_base = layer_idx * num_experts * 4;

    for (int i = 0; i < topk && batch_count < max_ept; i++) {
        int eid = topk_ids[i];
        if (eid < 0 || eid >= num_experts) continue;

        int ptr_base = layer_base + eid * 4;
        unsigned long long w13p = d_expert_ptrs[ptr_base + 0];

        if (w13p != 0) {
            // HCS hit (or mapped host pointer) — write pointers directly to batch upload
            w13p_arr[batch_count] = w13p;
            w13s_arr[batch_count] = d_expert_ptrs[ptr_base + 1];
            w2p_arr[batch_count]  = d_expert_ptrs[ptr_base + 2];
            w2s_arr[batch_count]  = d_expert_ptrs[ptr_base + 3];
            wts_arr[batch_count]  = topk_weights[i];
            batch_count++;
        } else {
            // Cold miss — record for CPU DMA
            mapped_cold_buf[2 + cold_count] = eid;
            mapped_cold_buf[2 + topk + cold_count] = batch_count;
            wts_arr[batch_count] = topk_weights[i];
            batch_count++;
            cold_count++;
        }
    }

    // Fill remaining slots with dummy expert (zero weight)
    for (int i = batch_count; i < topk && i < max_ept; i++) {
        w13p_arr[i] = dummy_ptr;
        w13s_arr[i] = dummy_ptr;
        w2p_arr[i]  = dummy_ptr;
        w2s_arr[i]  = dummy_ptr;
        wts_arr[i]  = 0.0f;
    }

    // Write topk IDs to mapped activations buffer for post-hoc HCS recording
    if (mapped_activations != NULL) {
        int act_base = moe_seq_idx * topk;
        for (int i = 0; i < topk; i++) {
            mapped_activations[act_base + i] = topk_ids[i];
        }
        __threadfence_system();  // ensure visible to CPU after final sync
    }

    // Write cold count and ready flag (CPU polls these)
    // Use __threadfence_system to ensure all writes are visible to CPU
    mapped_cold_buf[0] = cold_count;
    __threadfence_system();
    mapped_cold_buf[1] = 1;  // ready flag — CPU can now read
}

// ── Fused Gate GEMV + TopK ─────────────────────────────────────────────
//
// Replaces: bf16_to_fp32 + cuBLAS gate GEMV + sigmoid_topk/softmax_topk
// in a single kernel launch. Saves ~20us per layer from eliminated launches.
//
// Launch: grid=(1,1,1), block=(num_experts,1,1), smem = hidden_size * 2 bytes
// Constraints: num_experts <= 1024, hidden_size <= 16384
//
// scoring_func: 0 = softmax, 1 = sigmoid

extern "C" __global__ void fused_gate_topk(
    const __nv_bfloat16* __restrict__ hidden,     // [hidden_size] bf16
    const float* __restrict__ gate,               // [num_experts, hidden_size] fp32
    const float* __restrict__ gate_bias,          // [num_experts] or NULL
    const float* __restrict__ e_score_corr,       // [num_experts] or NULL
    int* __restrict__ topk_indices,               // [topk]
    float* __restrict__ topk_weights,             // [topk]
    int num_experts,
    int hidden_size,
    int topk,
    int scoring_func,                             // 0=softmax, 1=sigmoid
    float routed_scaling_factor,                  // for sigmoid normalization
    int norm_topk_prob                            // 1=renormalize weights to sum to 1
) {
    const int eid = threadIdx.x;
    if (eid >= num_experts) return;

    // Load hidden state into shared memory as fp32 (all threads cooperate)
    extern __shared__ char smem_raw[];
    float* s_hidden = (float*)smem_raw;
    // After hidden: scores array for topK selection (float, num_experts)
    float* s_scores = s_hidden + hidden_size;

    for (int j = eid; j < hidden_size; j += num_experts) {
        s_hidden[j] = bf16_to_f32(hidden[j]);
    }
    __syncthreads();

    // Each thread computes dot product for one expert
    const float* gate_row = gate + (long long)eid * hidden_size;
    float acc = 0.0f;
    for (int j = 0; j < hidden_size; j++) {
        acc += gate_row[j] * s_hidden[j];
    }

    // Apply bias if present
    if (gate_bias) acc += gate_bias[eid];

    // Apply scoring function
    float score;
    if (scoring_func == 1) {
        // Sigmoid
        score = 1.0f / (1.0f + __expf(-acc));
        if (e_score_corr) score += e_score_corr[eid];
    } else {
        // For softmax, store raw logit first (need cooperative reduction)
        score = acc;
    }

    s_scores[eid] = score;
    __syncthreads();

    // Softmax normalization (if scoring_func == 0)
    if (scoring_func == 0 && eid == 0) {
        // Find max for numerical stability
        float max_val = s_scores[0];
        for (int i = 1; i < num_experts; i++) {
            if (s_scores[i] > max_val) max_val = s_scores[i];
        }
        // exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_experts; i++) {
            s_scores[i] = __expf(s_scores[i] - max_val);
            sum_exp += s_scores[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < num_experts; i++) {
            s_scores[i] *= inv_sum;
        }
    }
    __syncthreads();

    // TopK selection + normalize (single thread)
    if (eid == 0) {
        for (int t = 0; t < topk; t++) {
            int best_idx = -1;
            float best_val = -1e30f;
            for (int i = 0; i < num_experts; i++) {
                if (s_scores[i] > best_val) {
                    best_val = s_scores[i];
                    best_idx = i;
                }
            }
            topk_indices[t] = best_idx;
            topk_weights[t] = best_val;
            s_scores[best_idx] = -1e30f;
        }

        // Normalize weights (only when norm_topk_prob is set)
        if (norm_topk_prob) {
            float sum = 0.0f;
            for (int t = 0; t < topk; t++) sum += topk_weights[t];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int t = 0; t < topk; t++) topk_weights[t] *= inv;
            }
        }
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

// In-place sigmoid gate using BF16 gate logit from device memory (e.g. cuBLAS GEMV output).
// Reads one BF16 value from gate_logit_ptr, applies sigmoid, scales output in-place.
// Used after shared expert GEMV when gate_weight is registered as a BF16 weight.
extern "C" __global__ void sigmoid_gate_inplace_bf16(
    __nv_bfloat16* __restrict__ data,              // [size], modified in-place
    const __nv_bfloat16* __restrict__ gate_logit_ptr, // [1], single BF16 on device
    int size
) {
    float gate_value = 1.0f / (1.0f + __expf(-bf16_to_f32(gate_logit_ptr[0])));
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = f32_to_bf16(bf16_to_f32(data[i]) * gate_value);
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

// ── MLA (Multi-head Latent Attention) Decode Kernels ──────────────────

// Write compressed KV + rope PE to FP8 caches at the current position.
// ckv_cache: [max_seq, kv_lora_rank], kpe_cache: [max_seq, qk_rope_dim]
// Input ckv [kv_lora_rank] and kpe [qk_rope_dim] are FP32.
extern "C" __global__ void mla_kv_cache_write_g(
    __nv_fp8_e4m3* __restrict__ ckv_cache,
    __nv_fp8_e4m3* __restrict__ kpe_cache,
    const float* __restrict__ ckv,       // [kv_lora_rank] FP32
    const float* __restrict__ kpe,       // [qk_rope_dim] FP32
    const int* __restrict__ d_position,
    int kv_lora_rank,
    int ckv_cache_dim,
    int qk_rope_dim
) {
    int position = *d_position;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ckv_cache_dim) {
        float val = (i < kv_lora_rank) ? ckv[i] : 0.0f;
        ckv_cache[position * ckv_cache_dim + i] = f32_to_fp8e4m3(val);
    }
    if (i < qk_rope_dim) {
        kpe_cache[position * qk_rope_dim + i] = f32_to_fp8e4m3(kpe[i]);
    }
}

// MLA KV cache write (non-graphed: position as immediate)
// ckv_cache_dim >= kv_lora_rank: elements [kv_lora_rank, ckv_cache_dim) are zero-padded.
extern "C" __global__ void mla_kv_cache_write(
    __nv_fp8_e4m3* __restrict__ ckv_cache,
    __nv_fp8_e4m3* __restrict__ kpe_cache,
    const float* __restrict__ ckv,
    const float* __restrict__ kpe,
    int position,
    int kv_lora_rank,
    int ckv_cache_dim,
    int qk_rope_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ckv_cache_dim) {
        float val = (i < kv_lora_rank) ? ckv[i] : 0.0f;
        ckv_cache[position * ckv_cache_dim + i] = f32_to_fp8e4m3(val);
    }
    if (i < qk_rope_dim) {
        kpe_cache[position * qk_rope_dim + i] = f32_to_fp8e4m3(kpe[i]);
    }
}

// MLA attention decode kernel (graphable variant, reads seq_len from GPU pointer).
//
// Computes single-token MLA attention for all query heads simultaneously.
// Score = dot(q_absorbed[h], ckv_cache[pos]) + dot(q_pe[h], kpe_cache[pos])
// Output = softmax-weighted sum of ckv_cache values (kv_lora_rank dims per head).
//
// Grid: (num_heads, 1, 1)   Block: (256, 1, 1)
// Shared memory: (kv_lora_rank + qk_rope_dim + num_warps + MLA_TILE_SIZE) * sizeof(float)
#define MLA_TILE_SIZE 4096

extern "C" __global__ void mla_attention_g(
    float* __restrict__ output,              // [num_heads * kv_lora_rank] FP32
    const float* __restrict__ q_absorbed,    // [num_heads * kv_lora_rank] FP32
    const float* __restrict__ q_pe,          // [num_heads * qk_rope_dim] FP32
    const __nv_fp8_e4m3* __restrict__ ckv_cache,  // [max_seq * kv_lora_rank] FP8
    const __nv_fp8_e4m3* __restrict__ kpe_cache,  // [max_seq * qk_rope_dim] FP8
    float sm_scale,
    int num_heads,
    int kv_lora_rank,
    int qk_rope_dim,
    const int* __restrict__ d_seq_len,
    int max_seq
) {
    int seq_len = *d_seq_len;
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    // Shared memory: [kv_lora_rank] q_abs + [qk_rope_dim] q_pe + [num_warps] reduce + [MLA_TILE_SIZE] weights
    extern __shared__ float smem[];
    float* s_q_abs = smem;
    float* s_q_pe = smem + kv_lora_rank;
    float* smem_reduce = s_q_pe + qk_rope_dim;
    float* smem_weights = smem_reduce + num_warps;

    // Preload absorbed query and rope PE query into shared memory
    const float* qa_head = q_absorbed + h * kv_lora_rank;
    const float* qp_head = q_pe + h * qk_rope_dim;
    for (int i = tid; i < kv_lora_rank; i += num_threads) {
        s_q_abs[i] = qa_head[i];
    }
    for (int i = tid; i < qk_rope_dim; i += num_threads) {
        s_q_pe[i] = qp_head[i];
    }
    __syncthreads();

    // Pass 1: find max score
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        // Compressed KV dot product (kv_lora_rank dims)
        const __nv_fp8_e4m3* ckv_vec = ckv_cache + pos * kv_lora_rank;
        for (int d = 0; d < kv_lora_rank; d += 16) {
            int remaining = kv_lora_rank - d;
            int count = remaining < 16 ? remaining : 16;
            if (count == 16) {
                uint4 packed = *reinterpret_cast<const uint4*>(ckv_vec + d);
                const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q_abs[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                }
            } else {
                for (int j = 0; j < count; j++) {
                    score += s_q_abs[d + j] * fp8e4m3_to_f32(ckv_vec[d + j]);
                }
            }
        }
        // Rope PE dot product (qk_rope_dim dims)
        const __nv_fp8_e4m3* kpe_vec = kpe_cache + pos * qk_rope_dim;
        for (int d = 0; d < qk_rope_dim; d += 16) {
            int remaining = qk_rope_dim - d;
            int count = remaining < 16 ? remaining : 16;
            if (count == 16) {
                uint4 packed = *reinterpret_cast<const uint4*>(kpe_vec + d);
                const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q_pe[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                }
            } else {
                for (int j = 0; j < count; j++) {
                    score += s_q_pe[d + j] * fp8e4m3_to_f32(kpe_vec[d + j]);
                }
            }
        }
        score *= sm_scale;
        local_max = fmaxf(local_max, score);
    }
    // Warp reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) m = fmaxf(m, smem_reduce[w]);
        smem_reduce[0] = m;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    // Pass 2 (tiled): compute weights, accumulate weighted ckv
    float local_sum = 0.0f;
    // Per-thread V accumulators (kv_lora_rank dims, 4 per thread stride)
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_start = 0; tile_start < seq_len; tile_start += MLA_TILE_SIZE) {
        int tile_end = tile_start + MLA_TILE_SIZE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Phase A: compute exp(score - max) for each position
        for (int ti = tid; ti < tile_len; ti += num_threads) {
            int pos = tile_start + ti;
            float score = 0.0f;
            const __nv_fp8_e4m3* ckv_vec = ckv_cache + pos * kv_lora_rank;
            for (int d = 0; d < kv_lora_rank; d += 16) {
                int remaining = kv_lora_rank - d;
                int count = remaining < 16 ? remaining : 16;
                if (count == 16) {
                    uint4 packed = *reinterpret_cast<const uint4*>(ckv_vec + d);
                    const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        score += s_q_abs[d + j] * fp8e4m3_to_f32(
                            *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                    }
                } else {
                    for (int j = 0; j < count; j++) {
                        score += s_q_abs[d + j] * fp8e4m3_to_f32(ckv_vec[d + j]);
                    }
                }
            }
            const __nv_fp8_e4m3* kpe_vec = kpe_cache + pos * qk_rope_dim;
            for (int d = 0; d < qk_rope_dim; d += 16) {
                int remaining = qk_rope_dim - d;
                int count = remaining < 16 ? remaining : 16;
                if (count == 16) {
                    uint4 packed = *reinterpret_cast<const uint4*>(kpe_vec + d);
                    const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        score += s_q_pe[d + j] * fp8e4m3_to_f32(
                            *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                    }
                } else {
                    for (int j = 0; j < count; j++) {
                        score += s_q_pe[d + j] * fp8e4m3_to_f32(kpe_vec[d + j]);
                    }
                }
            }
            float w = __expf(score * sm_scale - global_max);
            smem_weights[ti] = w;
            local_sum += w;
        }
        __syncthreads();

        // Phase B: accumulate weighted ckv using stored weights
        // Output dim is kv_lora_rank (not head_dim)
        const __nv_fp8_e4m3* v_base = ckv_cache + tile_start * kv_lora_rank;
        int di = 0;
        for (int d = tid; d < kv_lora_rank; d += num_threads) {
            float acc = 0.0f;
            for (int ti = 0; ti < tile_len; ti++) {
                acc += smem_weights[ti] * fp8e4m3_to_f32(v_base[ti * kv_lora_rank + d]);
            }
            v_acc[di++] += acc;
        }
        __syncthreads();
    }

    // Warp reduce sum_exp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps; w++) s += smem_reduce[w];
        smem_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem_reduce[0];

    // Write normalized output
    float* out_head = output + h * kv_lora_rank;
    int di = 0;
    for (int d = tid; d < kv_lora_rank; d += num_threads) {
        out_head[d] = v_acc[di++] * inv_sum;
    }
}

// MLA attention decode kernel (non-graphed: seq_len as immediate)
extern "C" __global__ void mla_attention(
    float* __restrict__ output,
    const float* __restrict__ q_absorbed,
    const float* __restrict__ q_pe,
    const __nv_fp8_e4m3* __restrict__ ckv_cache,
    const __nv_fp8_e4m3* __restrict__ kpe_cache,
    float sm_scale,
    int num_heads,
    int kv_lora_rank,
    int qk_rope_dim,
    int seq_len,
    int max_seq
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    extern __shared__ float smem[];
    float* s_q_abs = smem;
    float* s_q_pe = smem + kv_lora_rank;
    float* smem_reduce = s_q_pe + qk_rope_dim;
    float* smem_weights = smem_reduce + num_warps;

    const float* qa_head = q_absorbed + h * kv_lora_rank;
    const float* qp_head = q_pe + h * qk_rope_dim;
    for (int i = tid; i < kv_lora_rank; i += num_threads) {
        s_q_abs[i] = qa_head[i];
    }
    for (int i = tid; i < qk_rope_dim; i += num_threads) {
        s_q_pe[i] = qp_head[i];
    }
    __syncthreads();

    // Pass 1: find max score
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        const __nv_fp8_e4m3* ckv_vec = ckv_cache + pos * kv_lora_rank;
        for (int d = 0; d < kv_lora_rank; d += 16) {
            int remaining = kv_lora_rank - d;
            int count = remaining < 16 ? remaining : 16;
            if (count == 16) {
                uint4 packed = *reinterpret_cast<const uint4*>(ckv_vec + d);
                const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q_abs[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                }
            } else {
                for (int j = 0; j < count; j++) {
                    score += s_q_abs[d + j] * fp8e4m3_to_f32(ckv_vec[d + j]);
                }
            }
        }
        const __nv_fp8_e4m3* kpe_vec = kpe_cache + pos * qk_rope_dim;
        for (int d = 0; d < qk_rope_dim; d += 16) {
            int remaining = qk_rope_dim - d;
            int count = remaining < 16 ? remaining : 16;
            if (count == 16) {
                uint4 packed = *reinterpret_cast<const uint4*>(kpe_vec + d);
                const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q_pe[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                }
            } else {
                for (int j = 0; j < count; j++) {
                    score += s_q_pe[d + j] * fp8e4m3_to_f32(kpe_vec[d + j]);
                }
            }
        }
        score *= sm_scale;
        local_max = fmaxf(local_max, score);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) m = fmaxf(m, smem_reduce[w]);
        smem_reduce[0] = m;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    // Pass 2 (tiled): compute weights, accumulate weighted ckv
    float local_sum = 0.0f;
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_start = 0; tile_start < seq_len; tile_start += MLA_TILE_SIZE) {
        int tile_end = tile_start + MLA_TILE_SIZE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        for (int ti = tid; ti < tile_len; ti += num_threads) {
            int pos = tile_start + ti;
            float score = 0.0f;
            const __nv_fp8_e4m3* ckv_vec = ckv_cache + pos * kv_lora_rank;
            for (int d = 0; d < kv_lora_rank; d += 16) {
                int remaining = kv_lora_rank - d;
                int count = remaining < 16 ? remaining : 16;
                if (count == 16) {
                    uint4 packed = *reinterpret_cast<const uint4*>(ckv_vec + d);
                    const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        score += s_q_abs[d + j] * fp8e4m3_to_f32(
                            *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                    }
                } else {
                    for (int j = 0; j < count; j++) {
                        score += s_q_abs[d + j] * fp8e4m3_to_f32(ckv_vec[d + j]);
                    }
                }
            }
            const __nv_fp8_e4m3* kpe_vec = kpe_cache + pos * qk_rope_dim;
            for (int d = 0; d < qk_rope_dim; d += 16) {
                int remaining = qk_rope_dim - d;
                int count = remaining < 16 ? remaining : 16;
                if (count == 16) {
                    uint4 packed = *reinterpret_cast<const uint4*>(kpe_vec + d);
                    const unsigned char* pb = reinterpret_cast<const unsigned char*>(&packed);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        score += s_q_pe[d + j] * fp8e4m3_to_f32(
                            *reinterpret_cast<const __nv_fp8_e4m3*>(&pb[j]));
                    }
                } else {
                    for (int j = 0; j < count; j++) {
                        score += s_q_pe[d + j] * fp8e4m3_to_f32(kpe_vec[d + j]);
                    }
                }
            }
            float w = __expf(score * sm_scale - global_max);
            smem_weights[ti] = w;
            local_sum += w;
        }
        __syncthreads();

        const __nv_fp8_e4m3* v_base = ckv_cache + tile_start * kv_lora_rank;
        int di = 0;
        for (int d = tid; d < kv_lora_rank; d += num_threads) {
            float acc = 0.0f;
            for (int ti = 0; ti < tile_len; ti++) {
                acc += smem_weights[ti] * fp8e4m3_to_f32(v_base[ti * kv_lora_rank + d]);
            }
            v_acc[di++] += acc;
        }
        __syncthreads();
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps; w++) s += smem_reduce[w];
        smem_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem_reduce[0];

    float* out_head = output + h * kv_lora_rank;
    int di = 0;
    for (int d = tid; d < kv_lora_rank; d += num_threads) {
        out_head[d] = v_acc[di++] * inv_sum;
    }
}

// MLA de-interleave kernel: [re0, im0, re1, im1, ...] → [re0, re1, ..., im0, im1, ...]
// Applied to q_pe and k_pe before RoPE when rope_interleave=true.
extern "C" __global__ void mla_deinterleave(
    float* __restrict__ data,    // [num_heads * dim] or [dim] in-place
    int total_elements,          // num_heads * dim
    int dim                      // rope dimension per head (must be even)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;
    int head_offset = (i / dim) * dim;
    int local_i = i % dim;
    int half = dim / 2;
    // Only process the first half to avoid double-swap
    if (local_i >= half) return;
    // Swap: re at even indices, im at odd indices → re block then im block
    float re = data[head_offset + local_i * 2];
    float im = data[head_offset + local_i * 2 + 1];
    data[head_offset + local_i] = re;
    data[head_offset + half + local_i] = im;
}

// MLA split Q: [h0_nope|h0_rope|h1_nope|h1_rope|...] → contiguous q_nope [nh*nope] and q_pe [nh*rope]
extern "C" __global__ void mla_split_q(
    float* __restrict__ q_nope_out,  // [num_heads * nope_dim]
    float* __restrict__ q_pe_out,    // [num_heads * rope_dim]
    const float* __restrict__ q_in,  // [num_heads * (nope_dim + rope_dim)]
    int num_heads,
    int nope_dim,
    int rope_dim
) {
    int stride = nope_dim + rope_dim;
    int total = num_heads * stride;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int h = i / stride;
    int local = i % stride;
    float val = q_in[i];
    if (local < nope_dim) {
        q_nope_out[h * nope_dim + local] = val;
    } else {
        q_pe_out[h * rope_dim + (local - nope_dim)] = val;
    }
}

// MLA w_kc absorption: q_nope[h, nope_dim] @ w_kc[h, nope_dim, kv_lora_rank] → q_absorbed[h, kv_lora_rank]
// One block per head. Each thread computes a subset of the kv_lora_rank output dims.
extern "C" __global__ void mla_absorb_wkc(
    float* __restrict__ q_absorbed,    // [num_heads * kv_lora_rank] FP32 output
    const float* __restrict__ q_nope,  // [num_heads * nope_dim] FP32 input
    const __nv_bfloat16* __restrict__ w_kc,  // [num_heads, nope_dim, kv_lora_rank] BF16
    int num_heads,
    int nope_dim,
    int kv_lora_rank
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* q_h = q_nope + h * nope_dim;
    const __nv_bfloat16* w_h = w_kc + h * nope_dim * kv_lora_rank;
    float* out_h = q_absorbed + h * kv_lora_rank;

    // Each thread computes one or more output dimensions
    for (int d = tid; d < kv_lora_rank; d += num_threads) {
        float acc = 0.0f;
        for (int k = 0; k < nope_dim; k++) {
            acc += q_h[k] * bf16_to_f32(w_h[k * kv_lora_rank + d]);
        }
        out_h[d] = acc;
    }
}

// MLA w_vc post-attention: attn_out[h, kv_lora_rank] @ w_vc[h, v_head_dim, kv_lora_rank]^T → out[h, v_head_dim]
// Then converts to BF16 for o_proj input.
// One block per head.
extern "C" __global__ void mla_apply_wvc(
    __nv_bfloat16* __restrict__ output,       // [num_heads * v_head_dim] BF16
    const float* __restrict__ attn_out,       // [num_heads * kv_lora_rank] FP32
    const __nv_bfloat16* __restrict__ w_vc,   // [num_heads, v_head_dim, kv_lora_rank] BF16
    int num_heads,
    int v_head_dim,
    int kv_lora_rank
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* a_h = attn_out + h * kv_lora_rank;
    const __nv_bfloat16* w_h = w_vc + h * v_head_dim * kv_lora_rank;
    __nv_bfloat16* out_h = output + h * v_head_dim;

    for (int d = tid; d < v_head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int k = 0; k < kv_lora_rank; k++) {
            acc += a_h[k] * bf16_to_f32(w_h[d * kv_lora_rank + k]);
        }
        out_h[d] = f32_to_bf16(acc);
    }
}

// ── Linear Attention Convolution ───────────────────────────────────────

// 1D causal convolution for linear attention (Mamba-style).
// Shifts conv_state, inserts new input, computes conv output.
// conv_state: [conv_dim, kernel_dim] (each of conv_dim channels has kernel_dim history)
// Conv1d with SiLU activation for linear attention decode.
// Input is UN-INTERLEAVED [q_flat, k_flat, v_flat] = [conv_dim].
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
        // Apply SiLU activation
        output[c] = out / (1.0f + __expf(-out));
    }
}

// ── Linear Attention Helper Kernels ────────────────────────────────────

// Un-interleave QKVZ projection output.
// Input layout (interleaved per key-head group):
//   [h0_q(dk), h0_k(dk), h0_v(ratio*dv), h0_z(ratio*dv), h1_q(dk), h1_k(dk), ...]
// Output: separate contiguous arrays for conv_input [q_flat, k_flat, v_flat] and z_flat.
// conv_input: [q(nk*dk), k(nk*dk), v(nv*dv)] = [conv_dim]
// z_out: [nv*dv]
extern "C" __global__ void uninterleave_qkvz(
    float* __restrict__ conv_input,   // [conv_dim] = [q_flat, k_flat, v_flat]
    float* __restrict__ z_out,        // [nv * dv]
    const float* __restrict__ qkvz,   // [nk * group_dim] interleaved
    int nk,       // num key heads
    int dk,       // key head dim
    int ratio,    // head_ratio = nv / nk
    int dv        // value head dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int group_dim = 2 * dk + 2 * ratio * dv;
    int key_dim = nk * dk;
    int val_dim = nk * ratio * dv;  // = nv * dv
    int total = nk * group_dim;
    if (idx >= total) return;

    int head = idx / group_dim;
    int offset = idx % group_dim;

    if (offset < dk) {
        // Q element
        conv_input[head * dk + offset] = qkvz[idx];
    } else if (offset < 2 * dk) {
        // K element
        int k_elem = offset - dk;
        conv_input[key_dim + head * dk + k_elem] = qkvz[idx];
    } else if (offset < 2 * dk + ratio * dv) {
        // V element
        int v_elem = offset - 2 * dk;
        conv_input[2 * key_dim + head * ratio * dv + v_elem] = qkvz[idx];
    } else {
        // Z element
        int z_elem = offset - (2 * dk + ratio * dv);
        z_out[head * ratio * dv + z_elem] = qkvz[idx];
    }
}

// Compute gate and beta for gated delta net from BA projection output.
// beta = sigmoid(b), gate = exp(-A_log.exp() * softplus(a + dt_bias))
extern "C" __global__ void compute_gate_beta(
    float* __restrict__ gate_out,     // [nv]
    float* __restrict__ beta_out,     // [nv]
    const float* __restrict__ ba,     // [nk * 2 * ratio] interleaved: [h0_b(ratio), h0_a(ratio), h1_b(ratio), ...]
    const float* __restrict__ A_log,  // [nv]
    const float* __restrict__ dt_bias, // [nv]
    int nv,
    int ratio                         // head_ratio = nv / nk
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nv) {
        // Un-interleave BA: group = i / ratio, pos = i % ratio, group_dim = 2 * ratio
        int group = i / ratio;
        int pos = i % ratio;
        int group_dim = 2 * ratio;
        float b = ba[group * group_dim + pos];
        float a = ba[group * group_dim + ratio + pos];
        beta_out[i] = 1.0f / (1.0f + __expf(-b));  // sigmoid(b)
        float al = A_log[i];
        float sp = logf(1.0f + __expf(a + dt_bias[i]));  // softplus(a + dt_bias)
        gate_out[i] = __expf(-__expf(al) * sp);  // exp(-exp(A_log) * softplus(a + dt_bias))
    }
}

// Repeat-interleave heads: expand [nk, dim] to [nv, dim] where nv = nk * ratio.
// Each key head is repeated 'ratio' times consecutively.
extern "C" __global__ void repeat_interleave_heads(
    float* __restrict__ output,   // [nv * dim]
    const float* __restrict__ input,   // [nk * dim]
    int nk,
    int dim,
    int ratio
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nv = nk * ratio;
    int total = nv * dim;
    if (idx < total) {
        int out_head = idx / dim;
        int elem = idx % dim;
        int in_head = out_head / ratio;
        output[idx] = input[in_head * dim + elem];
    }
}

// L2 normalize per head, with optional scale factor.
// data[head * dim .. head * dim + dim] is normalized in-place.
extern "C" __global__ void l2norm_scale_per_head(
    float* __restrict__ data,   // [num_heads * dim]
    float scale,
    int num_heads,
    int dim
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    float* h = data + head * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += num_threads) {
        sum_sq += h[i] * h[i];
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
        float inv_norm = rsqrtf(total + 1e-12f);
        warp_sums[0] = inv_norm * scale;
    }
    __syncthreads();
    float norm_scale = warp_sums[0];

    for (int i = tid; i < dim; i += num_threads) {
        h[i] = h[i] * norm_scale;
    }
}

// Gated delta net recurrence step (one token).
// Implements the full delta rule used by Qwen3-Coder-Next's linear attention:
//   state *= gate  (per-head decay)
//   kv_mem = (state * k.unsqueeze(-1)).sum(-2)  (memory recall)
//   delta = (v - kv_mem) * beta  (error-correcting delta)
//   state += k.unsqueeze(-1) * delta.unsqueeze(-2)  (update)
//   output = (state * q.unsqueeze(-1)).sum(-2)  (readout)
//
// One block per head. state is [nv, dk, dv].
extern "C" __global__ void gated_delta_net_step(
    float* __restrict__ state,   // [nv, dk, dv] in/out
    const float* __restrict__ q, // [nv * dk]
    const float* __restrict__ k, // [nv * dk]
    const float* __restrict__ v, // [nv * dv]
    const float* __restrict__ gate, // [nv]
    const float* __restrict__ beta, // [nv]
    float* __restrict__ output,  // [nv * dv]
    int nv, int dk, int dv
) {
    int head = blockIdx.x;
    if (head >= nv) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float g = gate[head];
    float b = beta[head];
    int mat_size = dk * dv;

    float* S = state + head * mat_size;
    const float* q_h = q + head * dk;
    const float* k_h = k + head * dk;
    const float* v_h = v + head * dv;
    float* out_h = output + head * dv;

    // Step 1: Decay state: S *= gate
    for (int idx = tid; idx < mat_size; idx += num_threads) {
        S[idx] *= g;
    }
    __syncthreads();

    // Step 2: Memory recall: kv_mem[j] = sum_i(S[i,j] * k[i])
    // Step 3: Delta: delta[j] = (v[j] - kv_mem[j]) * beta
    // Step 4: State update: S[i,j] += k[i] * delta[j]
    // These are fused to avoid extra shared memory.
    for (int j = tid; j < dv; j += num_threads) {
        // kv_mem[j]
        float kv_mem = 0.0f;
        for (int i = 0; i < dk; i++) {
            kv_mem += S[i * dv + j] * k_h[i];
        }
        float delta = (v_h[j] - kv_mem) * b;
        // Update state column j
        for (int i = 0; i < dk; i++) {
            S[i * dv + j] += k_h[i] * delta;
        }
    }
    __syncthreads();

    // Step 5: Output: out[j] = sum_i(q[i] * S[i,j])
    for (int j = tid; j < dv; j += num_threads) {
        float acc = 0.0f;
        for (int i = 0; i < dk; i++) {
            acc += q_h[i] * S[i * dv + j];
        }
        out_h[j] = acc;
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
        float silu_g = g / (1.0f + __expf(-g));
        out[i] = silu_g * normed;
    }
}

// Same as gated_rmsnorm_silu but outputs BF16 directly (eliminates fp32_to_bf16 kernel)
extern "C" __global__ void gated_rmsnorm_silu_bf16(
    unsigned short* __restrict__ output,         // [nv * dv] BF16
    const float* __restrict__ recur_out,         // [nv * dv]
    const float* __restrict__ z,                 // [nv * dv]
    const float* __restrict__ norm_weight,       // [dv]
    float eps,
    int nv, int dv
) {
    int head = blockIdx.x;
    if (head >= nv) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* r = recur_out + head * dv;
    const float* gate = z + head * dv;
    unsigned short* out = output + head * dv;

    extern __shared__ float smem[];
    float sum_sq = 0.0f;
    for (int i = tid; i < dv; i += num_threads) {
        float x = r[i];
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
        warp_sums[0] = rsqrtf(total / (float)dv + eps);
    }
    __syncthreads();
    float rms_scale = warp_sums[0];

    for (int i = tid; i < dv; i += num_threads) {
        float normed = smem[i] * rms_scale * norm_weight[i];
        float g = gate[i];
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 result = __float2bfloat16(silu_g * normed);
        out[i] = *reinterpret_cast<unsigned short*>(&result);
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

// Write K,V to FP8 E4M3 KV cache at given position
extern "C" __global__ void kv_cache_write(
    __nv_fp8_e4m3* __restrict__ k_cache,   // [max_seq, kv_stride] FP8
    __nv_fp8_e4m3* __restrict__ v_cache,   // [max_seq, kv_stride] FP8
    const float* __restrict__ k,     // [kv_stride]
    const float* __restrict__ v,     // [kv_stride]
    int position,
    int kv_stride   // num_kv_heads * head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_stride) {
        k_cache[position * kv_stride + i] = f32_to_fp8e4m3(k[i]);
        v_cache[position * kv_stride + i] = f32_to_fp8e4m3(v[i]);
    }
}

// Single-query GQA attention: scores over all cached K, softmax, weighted V sum.
// One block per Q head. KV cache is FP8 E4M3 — dequantized to FP32 on the fly.
//
// Q is preloaded into shared memory for fast reuse across KV positions.
// FP8 K values are loaded 16 at a time via uint4 (16-byte vectorized loads).
//
// Uses dynamic shared memory to store attention scores when seq_len fits.
// For longer sequences, falls back to 2-pass approach that recomputes K dot
// products (slower but uses O(1) shared memory for scores).
extern "C" __global__ void gqa_attention(
    float* __restrict__ output,          // [num_q_heads * head_dim]
    const float* __restrict__ q,          // [num_q_heads * head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache,   // [max_seq, kv_stride] FP8
    const __nv_fp8_e4m3* __restrict__ v_cache,   // [max_seq, kv_stride] FP8
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,     // number of valid positions (position + 1)
    int max_seq,
    int use_smem     // 1 = shared memory path, 0 = 2-pass fallback
) {
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int kv_stride = num_kv_heads * head_dim;

    const float* q_head = q + qh * head_dim;

    // Dynamic shared memory layout:
    //   s_q[0..head_dim-1]:          Q vector preloaded (float)
    //   smem_scores[0..seq_len-1]:   attention scores (only when use_smem=1)
    //   smem_reduce[0..31]:          warp reduction scratch (always)
    extern __shared__ float smem[];
    float* s_q = smem;
    float* smem_scores = smem + head_dim;
    float* smem_reduce = smem_scores + (use_smem ? seq_len : 0);

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    // Preload Q into shared memory (reused across all KV positions)
    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    if (use_smem) {
        // ══════ FAST PATH: shared memory for scores ══════

        // Step 1: Compute all attention scores (vectorized FP8 loads)
        for (int pos = tid; pos < seq_len; pos += num_threads) {
            float score = 0.0f;
            const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
            for (int d = 0; d < head_dim; d += 16) {
                uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
                const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
                }
            }
            smem_scores[pos] = score * sm_scale;
        }
        __syncthreads();

        // Step 2: Find max (parallel reduction)
        float local_max = -1e30f;
        for (int pos = tid; pos < seq_len; pos += num_threads) {
            local_max = fmaxf(local_max, smem_scores[pos]);
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (lane_id == 0) smem_reduce[warp_id] = local_max;
        __syncthreads();
        if (tid == 0) {
            float gmax = smem_reduce[0];
            for (int w = 1; w < num_warps; w++) gmax = fmaxf(gmax, smem_reduce[w]);
            smem_reduce[0] = gmax;
        }
        __syncthreads();
        float global_max = smem_reduce[0];

        // Step 3: Compute softmax weights in-place
        float local_sum = 0.0f;
        for (int pos = tid; pos < seq_len; pos += num_threads) {
            float w = __expf(smem_scores[pos] - global_max);
            smem_scores[pos] = w;
            local_sum += w;
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane_id == 0) smem_reduce[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float gsum = 0.0f;
            for (int w = 0; w < num_warps; w++) gsum += smem_reduce[w];
            smem_reduce[0] = gsum;
        }
        __syncthreads();
        float inv_sum = 1.0f / smem_reduce[0];

        // Normalize weights
        for (int pos = tid; pos < seq_len; pos += num_threads) {
            smem_scores[pos] *= inv_sum;
        }
        __syncthreads();

        // Step 4: Weighted V sum (each thread handles subset of output dims)
        float* out_head = output + qh * head_dim;
        for (int d = tid; d < head_dim; d += num_threads) {
            float acc = 0.0f;
            for (int pos = 0; pos < seq_len; pos++) {
                acc += smem_scores[pos] * fp8e4m3_to_f32(v_cache[pos * kv_stride + kv_head * head_dim + d]);
            }
            out_head[d] = acc;
        }

    } else {
        // ══════ SLOW PATH: 2-pass, no score storage ══════

        // Pass 1: Online softmax — find max and sum_exp (vectorized FP8 K loads)
        float local_max = -1e30f;
        float local_sum = 0.0f;
        for (int pos = tid; pos < seq_len; pos += num_threads) {
            float score = 0.0f;
            const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
            for (int d = 0; d < head_dim; d += 16) {
                uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
                const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
                }
            }
            score *= sm_scale;
            if (score > local_max) {
                local_sum *= __expf(local_max - score);
                local_max = score;
            }
            local_sum += __expf(score - local_max);
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
            float new_max = fmaxf(local_max, other_max);
            local_sum = local_sum * __expf(local_max - new_max) + other_sum * __expf(other_max - new_max);
            local_max = new_max;
        }
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_max;
            smem_reduce[warp_id + 16] = local_sum;
        }
        __syncthreads();
        if (tid == 0) {
            float gmax = smem_reduce[0];
            float gsum = smem_reduce[16];
            for (int w = 1; w < num_warps; w++) {
                float wm = smem_reduce[w];
                float ws = smem_reduce[w + 16];
                float new_max = fmaxf(gmax, wm);
                gsum = gsum * __expf(gmax - new_max) + ws * __expf(wm - new_max);
                gmax = new_max;
            }
            smem_reduce[0] = gmax;
            smem_reduce[16] = gsum;
        }
        __syncthreads();
        float global_max = smem_reduce[0];
        float inv_sum = 1.0f / smem_reduce[16];

        // Pass 2: Weighted V sum (recompute scores, vectorized FP8 K loads)
        float* out_head = output + qh * head_dim;
        for (int d = tid; d < head_dim; d += num_threads) {
            float acc = 0.0f;
            for (int pos = 0; pos < seq_len; pos++) {
                float score = 0.0f;
                const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
                for (int dd = 0; dd < head_dim; dd += 16) {
                    uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + dd);
                    const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        score += s_q[dd + j] * fp8e4m3_to_f32(
                            *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
                    }
                }
                float weight = __expf(score * sm_scale - global_max) * inv_sum;
                acc += weight * fp8e4m3_to_f32(v_cache[pos * kv_stride + kv_head * head_dim + d]);
            }
            out_head[d] = acc;
        }
    }
}

// ── FlashDecoding-style tiled GQA attention ───────────────────────────
//
// Splits the sequence dimension across grid.y blocks for massive SM
// utilization on large GPUs.  Each block computes attention for one Q
// head over a contiguous tile of KV positions.
//
// Output per tile: unnormalised weighted V (FP32), local max score, and
// local sum of exp(score - max).  A lightweight reduce kernel merges
// tiles using the log-sum-exp identity.
//
// Q is preloaded into shared memory for fast reuse across KV positions.
// FP8 K values are loaded 16 at a time via uint4 (16-byte vectorized loads).
//
// Grid: (num_q_heads, num_tiles, 1)   Block: (256, 1, 1)
// Shared memory: (head_dim + tile_size) * 4 + 128 bytes

extern "C" __global__ void gqa_attention_tiled(
    float* __restrict__ partial_o,     // [num_q_heads, num_tiles, head_dim]
    float* __restrict__ partial_lse,   // [num_q_heads, num_tiles, 2] (max, sum_exp)
    const float* __restrict__ q,       // [num_q_heads * head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache,  // [max_seq, kv_stride] FP8
    const __nv_fp8_e4m3* __restrict__ v_cache,  // [max_seq, kv_stride] FP8
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int tile_size
) {
    int qh = blockIdx.x;
    int tile_idx = blockIdx.y;
    int num_tiles = (seq_len + tile_size - 1) / tile_size;
    if (qh >= num_q_heads || tile_idx >= num_tiles) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int kv_stride = num_kv_heads * head_dim;

    const float* q_head = q + qh * head_dim;

    int tile_start = tile_idx * tile_size;
    int tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;
    int tile_len = tile_end - tile_start;

    // Dynamic shared memory layout:
    //   s_q[0..head_dim-1]:          Q vector preloaded (float)
    //   smem_scores[0..tile_size-1]: attention scores (float)
    //   smem_reduce[0..31]:          warp scratch (float)
    extern __shared__ float smem[];
    float* s_q = smem;
    float* smem_scores = smem + head_dim;
    float* smem_reduce = smem_scores + tile_size;

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    // Preload Q into shared memory (reused across all KV positions)
    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    // ── Step 1: Q·K dot products for this tile (vectorized FP8 loads) ──
    for (int i = tid; i < tile_len; i += num_threads) {
        int pos = tile_start + i;
        float score = 0.0f;
        const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        // Load 16 FP8 K values at once via uint4 (16 bytes per transaction)
        for (int d = 0; d < head_dim; d += 16) {
            uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
            const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                score += s_q[d + j] * fp8e4m3_to_f32(
                    *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
            }
        }
        smem_scores[i] = score * sm_scale;
    }
    __syncthreads();

    // ── Step 2: Local max (parallel reduction) ──
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += num_threads) {
        local_max = fmaxf(local_max, smem_scores[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float gmax = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) gmax = fmaxf(gmax, smem_reduce[w]);
        smem_reduce[0] = gmax;
    }
    __syncthreads();
    float tile_max = smem_reduce[0];

    // ── Step 3: exp(score - max) in-place, compute local sum ──
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += num_threads) {
        float w = __expf(smem_scores[i] - tile_max);
        smem_scores[i] = w;
        local_sum += w;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float gsum = 0.0f;
        for (int w = 0; w < num_warps; w++) gsum += smem_reduce[w];
        smem_reduce[0] = gsum;
    }
    __syncthreads();
    float tile_sum = smem_reduce[0];

    // ── Step 4: Unnormalised weighted V sum ──
    float* out_partial = partial_o + (qh * num_tiles + tile_idx) * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int i = 0; i < tile_len; i++) {
            acc += smem_scores[i] * fp8e4m3_to_f32(
                v_cache[(tile_start + i) * kv_stride + kv_head * head_dim + d]);
        }
        out_partial[d] = acc;
    }

    // ── Step 5: Store tile statistics ──
    if (tid == 0) {
        float* lse = partial_lse + (qh * num_tiles + tile_idx) * 2;
        lse[0] = tile_max;
        lse[1] = tile_sum;
    }
}

// Merge tiled attention partials using log-sum-exp rescaling.
// One block per Q head, 256 threads cooperate on head_dim output dims.
//
// Grid: (num_q_heads, 1, 1)   Block: (256, 1, 1)
// Shared memory: num_tiles * 4 bytes (correction weights)

extern "C" __global__ void gqa_attention_reduce(
    float* __restrict__ output,            // [num_q_heads * head_dim]
    const float* __restrict__ partial_o,   // [num_q_heads, num_tiles, head_dim]
    const float* __restrict__ partial_lse, // [num_q_heads, num_tiles, 2]
    int num_q_heads,
    int head_dim,
    int num_tiles
) {
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float smem[];
    // smem[0..num_tiles-1] = per-tile normalised correction weight

    const float* lse = partial_lse + qh * num_tiles * 2;

    // Thread 0 computes correction weights (num_tiles is small, ~O(100))
    if (tid == 0) {
        float global_max = -1e30f;
        for (int t = 0; t < num_tiles; t++) {
            global_max = fmaxf(global_max, lse[t * 2]);
        }
        // global_sum = sum_t(exp(tile_max_t - global_max) * tile_sum_t)
        //            = sum_all(exp(score_i - global_max))
        float global_sum = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            float correction = __expf(lse[t * 2] - global_max);
            global_sum += correction * lse[t * 2 + 1];
            smem[t] = correction;  // just the rescaling factor, NOT multiplied by tile_sum
        }
        // Normalize: smem[t] = exp(tile_max_t - global_max) / global_sum
        float inv_sum = 1.0f / global_sum;
        for (int t = 0; t < num_tiles; t++) {
            smem[t] *= inv_sum;
        }
    }
    __syncthreads();

    // All threads cooperate on output dimensions
    const float* po = partial_o + qh * num_tiles * head_dim;
    float* out = output + qh * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            acc += smem[t] * po[t * head_dim + d];
        }
        out[d] = acc;
    }
}

// ── Gated Attention Helpers ────────────────────────────────────────────

// Split gated Q projection output into Q and gate.
// Input: qg_in[nh * hd * 2] with layout [head0_q(hd), head0_gate(hd), head1_q(hd), ...]
// Output: q_out[nh * hd] with layout [head0_q(hd), head1_q(hd), ...]
//         gate_out[nh * hd] with layout [head0_gate(hd), head1_gate(hd), ...]
// Safe to have q_out == qg_in (in-place for Q part) since each output idx <= input idx.
extern "C" __global__ void split_gated_q(
    float* __restrict__ q_out,
    float* __restrict__ gate_out,
    const float* __restrict__ qg_in,
    int nh,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * hd;
    if (idx < total) {
        int head = idx / hd;
        int elem = idx % hd;
        int src_base = head * (hd * 2);
        // Gate must be read BEFORE Q is written (in-place safety)
        float gate_val = qg_in[src_base + hd + elem];
        float q_val = qg_in[src_base + elem];
        q_out[idx] = q_val;
        gate_out[idx] = gate_val;
    }
}

// Apply sigmoid gate to attention output: out[i] *= sigmoid(gate[i])
extern "C" __global__ void apply_gated_attn(
    float* __restrict__ attn_out,
    const float* __restrict__ gate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = 1.0f / (1.0f + __expf(-gate[idx]));
        attn_out[idx] *= g;
    }
}

// Fused gated attention + BF16 conversion (eliminates separate fp32_to_bf16 kernel)
extern "C" __global__ void apply_gated_attn_bf16(
    unsigned short* __restrict__ output,    // [n] BF16
    const float* __restrict__ attn_out,     // [n] FP32
    const float* __restrict__ gate,         // [n] FP32
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = 1.0f / (1.0f + __expf(-gate[idx]));
        __nv_bfloat16 result = __float2bfloat16(attn_out[idx] * g);
        output[idx] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// FP32 to BF16 conversion
extern "C" __global__ void fp32_to_bf16_simple(
    unsigned short* __restrict__ output,    // [n] BF16
    const float* __restrict__ input,        // [n] FP32
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __nv_bfloat16 result = __float2bfloat16(input[idx]);
        output[idx] = *reinterpret_cast<unsigned short*>(&result);
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
// k-tiles, then reduce via warp shuffle.
//
// Optimizations (Pass 4):
//   1. Input vector preloaded into shared memory (one read per block)
//   2. Inv perm tables preloaded into shared memory (no global reads in hot loop)
//   3. Scale cached per group (recomputed only at group_size boundaries)
//   4. Warp shuffle reduction (width=16, eliminates shared memory reduce array)
//
// Shared memory layout (dynamic, passed at launch):
//   [0 .. K*2)                 : input BF16 (K unsigned shorts)
//   [K*2 .. K*2 + 4096)       : inv_weight_perm (1024 ints)
//   [K*2 + 4096 .. K*2 + 4352): inv_scale_perm (64 ints)

extern "C" __global__ void marlin_gemv_int4(
    const unsigned int* __restrict__ packed,   // [K/16, 2*N] Marlin tile-permuted INT4
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    unsigned short* __restrict__ output,       // [N] BF16
    const int* __restrict__ inv_weight_perm,   // [1024] inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // ── Cooperative preload: input, inv_weight_perm, inv_scale_perm ──
    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    // Thread mapping: 256 threads per block
    // k_slice = tid & 15  (0..15, which slice of K tiles)
    // tn = tid >> 4       (0..15, which output in this tile)
    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;    // K / 16
    int out_cols = N << 1;         // 2 * N (u32 columns in packed)

    // Distribute k_tiles across 16 slices
    int tiles_per_slice = k_tiles_total >> 4;  // k_tiles_total / 16
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // ── Warp shuffle reduction across 16 k_slices ──
    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        output[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── Marlin INT4 GEMV with FP32 output (for attention projections) ────
//
// Same as marlin_gemv_int4 but outputs FP32 (needed by attention score paths).

extern "C" __global__ void marlin_gemv_int4_f32(
    const unsigned int* __restrict__ packed,   // [K/16, 2*N] Marlin tile-permuted INT4
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    float* __restrict__ output,                // [N] FP32
    const int* __restrict__ inv_weight_perm,   // [1024] inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;

    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        output[n] = acc;
    }
}

// ── Fused silu_mul + w2 GEMV + weighted_add (Pass 5) ─────────────────
//
// Replaces 3 separate kernel launches per expert with 1:
//   1. Reads gate_up[2*K] (output of w13 GEMV)
//   2. Applies silu_mul during shared memory preload: scratch[i] = silu(gate[i]) * up[i]
//   3. Runs w2 Marlin GEMV: output = w2 @ scratch
//   4. Weighted accumulation: accum[n] += weight * output[n]
//
// Launch: grid=(N/16, 1, 1), block=(256, 1, 1)
// where K = intermediate_size, N = hidden_size
// Shared memory: K*2 + 1024*4 + 64*4 bytes (same layout as standard GEMV)

extern "C" __global__ void marlin_gemv_int4_fused_silu_accum(
    const unsigned int* __restrict__ packed,    // [K/16, 2*N] w2 Marlin-packed INT4
    const unsigned short* __restrict__ w2_scales, // [K/gs, N] w2 scales BF16
    const unsigned short* __restrict__ gate_up, // [2*K] BF16 (gate_up output from w13)
    unsigned short* __restrict__ accum,         // [N] BF16 (moe_out accumulator, read-modify-write)
    const int* __restrict__ inv_weight_perm,    // [1024]
    const int* __restrict__ inv_scale_perm,     // [64]
    int K,              // intermediate_size
    int N,              // hidden_size
    int group_size,
    float weight,       // routing weight for this expert
    const float* weight_ptr  // optional: if non-NULL, weight = sigmoid(*weight_ptr)
) {
    // If weight_ptr is non-NULL, read the gate logit and apply sigmoid on GPU
    if (weight_ptr != NULL) {
        weight = 1.0f / (1.0f + __expf(-(*weight_ptr)));
    }

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // ── Cooperative preload: apply silu_mul while loading gate_up into shared mem ──
    for (int i = tid; i < K; i += 256) {
        float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[i]));
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[K + i]));
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 val = __float2bfloat16(silu_g * u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    // ── Standard Marlin GEMV with cached optimizations ──
    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;

    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // Warp shuffle reduction
    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    // Weighted accumulate: accum[n] += weight * gemv_result
    if (k_slice == 0) {
        float existing = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&accum[n]));
        __nv_bfloat16 result = __float2bfloat16(existing + weight * acc);
        accum[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── INT8 version of fused silu + w2 GEMV + weighted accumulate ────────
extern "C" __global__ void marlin_gemv_int8_fused_silu_accum(
    const unsigned int* __restrict__ packed,
    const unsigned short* __restrict__ w2_scales,
    const unsigned short* __restrict__ gate_up,
    unsigned short* __restrict__ accum,
    const int* __restrict__ inv_weight_perm,
    const int* __restrict__ inv_scale_perm,
    int K, int N, int group_size,
    float weight,
    const float* weight_ptr
) {
    if (weight_ptr != NULL) {
        weight = 1.0f / (1.0f + __expf(-(*weight_ptr)));
    }

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[i]));
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[K + i]));
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 val = __float2bfloat16(silu_g * u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;
    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;  // INT8: 4 values per u32
    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        float existing = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&accum[n]));
        __nv_bfloat16 result = __float2bfloat16(existing + weight * acc);
        accum[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── Marlin INT4 GEMV v2: K-split + coalesced thread mapping ───────────
//
// Splits K dimension across gridDim.y blocks for better SM utilization
// on large GPUs (e.g., 5090 with 170 SMs vs 64 blocks for QCN w13).
//
// Key differences from v1:
//   1. Thread mapping swapped: tn = tid & 15, k_slice = tid >> 4
//      → consecutive threads in a half-warp share same k_slice (same row)
//      → better memory coalescing for weight and scale reads
//   2. gridDim.y = k_splits: K tiles distributed across grid blocks
//   3. Output is FP32 partial sums [k_splits, N] (reduced by separate kernel)
//   4. Shared memory reduction replaces warp shuffle width=16
//
// Grid: (n_tiles, k_splits, 1),  Block: (256, 1, 1)
// Shared memory: K*2 + 4096 + 256 + 1024 bytes

extern "C" __global__ void marlin_gemv_int4_v2(
    const unsigned int* __restrict__ packed,   // [K/16, 2*N] Marlin tile-permuted INT4
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    float* __restrict__ partial_out,           // [k_splits * N] FP32 partial sums
    const int* __restrict__ inv_weight_perm,   // [1024]
    const int* __restrict__ inv_scale_perm,    // [64]
    int K, int N, int group_size, int k_splits
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    // ── Cooperative preload ──
    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    // Swapped thread mapping: consecutive threads → consecutive N positions
    int tn = tid & 15;       // output element within tile
    int k_slice = tid >> 4;  // K-parallelism (16 slices per block)
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;

    // K-split range for this grid block
    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;

    // Within this split, distribute among 16 k_slices
    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // Shared memory reduction across 16 k_slices
    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) {
            sum += s_reduce[ks * 16 + tn];
        }
        partial_out[ksplit * N + n] = sum;
    }
}

// Reduce K-split partial sums to BF16 output
extern "C" __global__ void reduce_ksplits_bf16(
    unsigned short* __restrict__ output,  // [N] BF16
    const float* __restrict__ partial,    // [k_splits * N] FP32
    int N, int k_splits
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float sum = 0.0f;
    for (int ks = 0; ks < k_splits; ks++) {
        sum += partial[ks * N + n];
    }
    __nv_bfloat16 result = __float2bfloat16(sum);
    output[n] = *reinterpret_cast<unsigned short*>(&result);
}

// Reduce K-split partial sums to FP32 output (for attention score paths).
extern "C" __global__ void reduce_ksplits_f32(
    float* __restrict__ output,           // [N] FP32
    const float* __restrict__ partial,    // [k_splits * N] FP32
    int N, int k_splits
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float sum = 0.0f;
    for (int ks = 0; ks < k_splits; ks++) {
        sum += partial[ks * N + n];
    }
    output[n] = sum;
}

// ── Marlin INT4 GEMV v2 with inline atomic reduction to FP32 output ───
//
// Same as marlin_gemv_int4_v2 but eliminates the separate reduce kernel.
// Instead of writing to a partial buffer, each k_split block atomicAdds
// its result directly to the output. Output must be zeroed before launch.
//
// Grid: (n_tiles, k_splits, 1),  Block: (256, 1, 1)

extern "C" __global__ void marlin_gemv_int4_v2_fused_f32(
    const unsigned int* __restrict__ packed,   // [K/16, 2*N] Marlin tile-permuted INT4
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    float* __restrict__ output,                // [N] FP32, MUST be zeroed before launch
    const int* __restrict__ inv_weight_perm,   // [1024]
    const int* __restrict__ inv_scale_perm,    // [64]
    int K, int N, int group_size, int k_splits
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;

    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;

    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // Shared memory reduction across 16 k_slices
    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) {
            sum += s_reduce[ks * 16 + tn];
        }
        // Inline reduction: atomicAdd directly to output (no separate reduce kernel)
        atomicAdd(&output[n], sum);
    }
}

// ── Marlin INT8 GEMV v2 with K-splitting (FP32 partial output) ────────
//
// INT8 version of marlin_gemv_int4_v2. Provides K-splitting for better
// SM utilization on small-N projections (e.g. k_proj [512, 7168]).
// Without this, only n_tiles blocks run = poor occupancy on large GPUs.
//
// Outputs FP32 partial sums; caller reduces with reduce_ksplits_bf16/f32.
// Grid: (n_tiles, k_splits, 1),  Block: (256, 1, 1)

extern "C" __global__ void marlin_gemv_int8_v2(
    const unsigned int* __restrict__ packed,   // [K/16, 4*N] Marlin tile-permuted INT8
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    float* __restrict__ partial_out,           // [k_splits * N] FP32 partial sums
    const int* __restrict__ inv_weight_perm,   // [1024] INT8 inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size, int k_splits
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;

    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;

    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) {
            sum += s_reduce[ks * 16 + tn];
        }
        partial_out[ksplit * N + n] = sum;
    }
}

// ── Marlin INT8 GEMV v2 with inline atomic reduction to FP32 output ───
//
// Same as marlin_gemv_int8_v2 but atomicAdds to output instead of
// writing to a partial buffer. Output must be zeroed before launch.
// Grid: (n_tiles, k_splits, 1),  Block: (256, 1, 1)

extern "C" __global__ void marlin_gemv_int8_v2_fused_f32(
    const unsigned int* __restrict__ packed,   // [K/16, 4*N] Marlin tile-permuted INT8
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    float* __restrict__ output,                // [N] FP32, MUST be zeroed before launch
    const int* __restrict__ inv_weight_perm,   // [1024] INT8 inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size, int k_splits
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;

    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;

    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) {
            sum += s_reduce[ks * 16 + tn];
        }
        atomicAdd(&output[n], sum);
    }
}

// ── Fused silu_mul + w2 GEMV + weighted_add v2 (K-split) ─────────────
//
// Same fusion as v1 but with K-splitting for better GPU occupancy.
// Outputs FP32 partial sums; caller reduces and accumulates.
//
// Grid: (n_tiles, k_splits, 1),  Block: (256, 1, 1)

extern "C" __global__ void marlin_gemv_int4_fused_silu_accum_v2(
    const unsigned int* __restrict__ packed,    // [K/16, 2*N] w2 Marlin-packed INT4
    const unsigned short* __restrict__ w2_scales, // [K/gs, N] w2 scales BF16
    const unsigned short* __restrict__ gate_up, // [2*K] BF16 (gate_up output from w13)
    float* __restrict__ partial_out,            // [k_splits * N] FP32
    const int* __restrict__ inv_weight_perm,    // [1024]
    const int* __restrict__ inv_scale_perm,     // [64]
    int K, int N, int group_size, int k_splits
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    // Preload with silu_mul applied
    for (int i = tid; i < K; i += 256) {
        float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[i]));
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[K + i]));
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 val = __float2bfloat16(silu_g * u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;

    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;

    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) {
            sum += s_reduce[ks * 16 + tn];
        }
        partial_out[ksplit * N + n] = sum;
    }
}

// ── INT8 version of fused silu + w2 GEMV v2 (K-split) ────────────────
extern "C" __global__ void marlin_gemv_int8_fused_silu_accum_v2(
    const unsigned int* __restrict__ packed,
    const unsigned short* __restrict__ w2_scales,
    const unsigned short* __restrict__ gate_up,
    float* __restrict__ partial_out,
    const int* __restrict__ inv_weight_perm,
    const int* __restrict__ inv_scale_perm,
    int K, int N, int group_size, int k_splits
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[i]));
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[K + i]));
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 val = __float2bfloat16(silu_g * u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;
    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;  // INT8
    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;
    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) sum += s_reduce[ks * 16 + tn];
        partial_out[ksplit * N + n] = sum;
    }
}

// Reduce K-split partial sums with weighted accumulation to BF16
extern "C" __global__ void reduce_ksplits_weighted_accum_bf16(
    unsigned short* __restrict__ accum,   // [N] BF16 read-modify-write
    const float* __restrict__ partial,    // [k_splits * N] FP32
    int N, int k_splits, float weight
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float sum = 0.0f;
    for (int ks = 0; ks < k_splits; ks++) {
        sum += partial[ks * N + n];
    }
    float existing = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&accum[n]));
    __nv_bfloat16 result = __float2bfloat16(existing + weight * sum);
    accum[n] = *reinterpret_cast<unsigned short*>(&result);
}

// ── Batched Expert Kernels ─────────────────────────────────────────────
//
// Process up to MAX_BATCH_EXPERTS experts in a single kernel launch.
// blockIdx.z selects the expert. Each expert's weight pointers are read
// from device arrays passed as kernel arguments.
//
// This eliminates per-expert launch overhead (~5us × 30 kernels/layer
// = 150us/layer → ~30us/layer with batched launches).

#define MAX_BATCH_EXPERTS 10

// Batched w13 GEMV v2 with k-splitting.
// Grid: (n_tiles, k_splits, num_experts), Block: (256, 1, 1)
// Each expert reads from its own packed/scales arrays, writes to a striped
// region of partial_out: expert i writes to partial_out + i * k_splits * N.
extern "C" __global__ void marlin_gemv_int4_v2_batched(
    const unsigned long long* __restrict__ packed_ptrs,  // [num_experts] device ptrs to packed weights
    const unsigned long long* __restrict__ scales_ptrs,  // [num_experts] device ptrs to scale weights
    const unsigned short* __restrict__ input,           // [K] BF16 (shared across experts)
    float* __restrict__ partial_out,                    // [num_experts * k_splits * N] FP32
    const int* __restrict__ inv_weight_perm,            // [1024]
    const int* __restrict__ inv_scale_perm,             // [64]
    int K, int N, int group_size, int k_splits,
    const float* __restrict__ weights                   // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    const unsigned int* packed = (const unsigned int*)packed_ptrs[expert_idx];
    const unsigned short* scales = (const unsigned short*)scales_ptrs[expert_idx];

    // Offset partial_out for this expert
    float* my_partial = partial_out + expert_idx * k_splits * N;

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) s_input[i] = input[i];
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;
    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;
    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) sum += s_reduce[ks * 16 + tn];
        my_partial[ksplit * N + n] = sum;
    }
}

// ── Batched Marlin INT8 GEMV v2 (analogous to INT4 batched above) ────
// Differences from INT4: out_cols = N*4, perm >>2 / <<3, mask 0xFF, offset -128
extern "C" __global__ void marlin_gemv_int8_v2_batched(
    const unsigned long long* __restrict__ packed_ptrs,  // [num_experts] device ptrs to packed weights
    const unsigned long long* __restrict__ scales_ptrs,  // [num_experts] device ptrs to scale weights
    const unsigned short* __restrict__ input,           // [K] BF16 (shared across experts)
    float* __restrict__ partial_out,                    // [num_experts * k_splits * N] FP32
    const int* __restrict__ inv_weight_perm,            // [1024] INT8 inverse weight perm
    const int* __restrict__ inv_scale_perm,             // [64]
    int K, int N, int group_size, int k_splits,
    const float* __restrict__ weights                   // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    const unsigned int* packed = (const unsigned int*)packed_ptrs[expert_idx];
    const unsigned short* scales = (const unsigned short*)scales_ptrs[expert_idx];

    // Offset partial_out for this expert
    float* my_partial = partial_out + expert_idx * k_splits * N;

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);
    float* s_reduce = (float*)(smem_raw + K * 2 + 1024 * 4 + 64 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) s_input[i] = input[i];
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int tn = tid & 15;
    int k_slice = tid >> 4;
    int n_tile = blockIdx.x;
    int ksplit = blockIdx.y;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;  // INT8: 4 values per u32
    int tiles_per_split = k_tiles_total / k_splits;
    int split_start = ksplit * tiles_per_split;
    int split_end = (ksplit == k_splits - 1) ? k_tiles_total : split_start + tiles_per_split;
    int split_tiles = split_end - split_start;
    int tiles_per_slice = split_tiles / 16;
    int kt_start = split_start + k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? split_end : kt_start + tiles_per_slice;

    // Pre-compute permutation indices: INT8 uses >>2, <<3 (4 values per u32)
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    s_reduce[k_slice * 16 + tn] = acc;
    __syncthreads();

    if (k_slice == 0) {
        float sum = 0.0f;
        for (int ks = 0; ks < 16; ks++) sum += s_reduce[ks * 16 + tn];
        my_partial[ksplit * N + n] = sum;
    }
}

// Batched reduce k-split partial sums → BF16 gate_up output.
// Grid: (ceil(N/256), 1, num_experts), Block: (256, 1, 1)
// Each expert reads from partial[expert * k_splits * N] and writes to output[expert * N].
extern "C" __global__ void reduce_ksplits_bf16_batched(
    unsigned short* __restrict__ output,  // [num_experts * N] BF16
    const float* __restrict__ partial,    // [num_experts * k_splits * N] FP32
    int N, int k_splits,
    const float* __restrict__ weights     // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* my_partial = partial + expert_idx * k_splits * N;
    float sum = 0.0f;
    for (int ks = 0; ks < k_splits; ks++) {
        sum += my_partial[ks * N + n];
    }
    __nv_bfloat16 result = __float2bfloat16(sum);
    output[expert_idx * N + n] = *reinterpret_cast<unsigned short*>(&result);
}

// Batched fused silu_mul + w2 GEMV, writing to per-expert output buffers.
// Grid: (n_tiles, 1, num_experts), Block: (256, 1, 1)
// Each expert reads its own gate_up and w2 weights, writes to expert_outs[expert * N].
// Does NOT accumulate into moe_out — that's done by multi_expert_weighted_add.
extern "C" __global__ void fused_silu_w2_batched(
    const unsigned long long* __restrict__ w2_packed_ptrs,  // [num_experts]
    const unsigned long long* __restrict__ w2_scales_ptrs,  // [num_experts]
    const unsigned short* __restrict__ gate_ups,            // [num_experts * 2*K] BF16
    unsigned short* __restrict__ expert_outs,               // [num_experts * N] BF16
    const int* __restrict__ inv_weight_perm,
    const int* __restrict__ inv_scale_perm,
    int K, int N, int group_size,
    const float* __restrict__ weights                       // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    const unsigned int* packed = (const unsigned int*)w2_packed_ptrs[expert_idx];
    const unsigned short* w2_scales = (const unsigned short*)w2_scales_ptrs[expert_idx];
    const unsigned short* gate_up = gate_ups + expert_idx * 2 * K;
    unsigned short* out = expert_outs + expert_idx * N;

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // Apply silu_mul while loading gate_up into shared memory
    for (int i = tid; i < K; i += 256) {
        float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[i]));
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[K + i]));
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 val = __float2bfloat16(silu_g * u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;
    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;
    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // Warp shuffle reduction across 16 k_slices
    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        out[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── Batched fused silu_mul + w2 GEMV for INT8 Marlin ──────────────────
// Analogous to fused_silu_w2_batched but with INT8 dequant logic.
// Grid: (n_tiles, 1, num_experts), Block: (256, 1, 1)
extern "C" __global__ void fused_silu_w2_int8_batched(
    const unsigned long long* __restrict__ w2_packed_ptrs,  // [num_experts]
    const unsigned long long* __restrict__ w2_scales_ptrs,  // [num_experts]
    const unsigned short* __restrict__ gate_ups,            // [num_experts * 2*K] BF16
    unsigned short* __restrict__ expert_outs,               // [num_experts * N] BF16
    const int* __restrict__ inv_weight_perm,
    const int* __restrict__ inv_scale_perm,
    int K, int N, int group_size,
    const float* __restrict__ weights                       // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    const unsigned int* packed = (const unsigned int*)w2_packed_ptrs[expert_idx];
    const unsigned short* w2_scales = (const unsigned short*)w2_scales_ptrs[expert_idx];
    const unsigned short* gate_up = gate_ups + expert_idx * 2 * K;
    unsigned short* out = expert_outs + expert_idx * N;

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // Apply silu_mul while loading gate_up into shared memory
    for (int i = tid; i < K; i += 256) {
        float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[i]));
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[K + i]));
        float silu_g = g / (1.0f + __expf(-g));
        __nv_bfloat16 val = __float2bfloat16(silu_g * u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;
    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;  // INT8: 4 values per u32
    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: INT8 uses >>2, <<3
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // Warp shuffle reduction across 16 k_slices
    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        out[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// Multi-expert weighted accumulation: moe_out += sum(weight[i] * expert_out[i])
// Grid: (ceil(N/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void multi_expert_weighted_add_bf16(
    unsigned short* __restrict__ accum,              // [N] BF16 moe_out
    const unsigned short* __restrict__ expert_outs,  // [num_experts * N] BF16
    const float* __restrict__ weights,               // [num_experts] FP32
    int N, int num_experts, int init_zero
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = init_zero ? 0.0f : __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&accum[n]));
    for (int e = 0; e < num_experts; e++) {
        float w = weights[e];
        if (w == 0.0f) continue;  // skip dummy experts (stale data could NaN)
        float val = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&expert_outs[e * N + n]));
        sum += w * val;
    }
    __nv_bfloat16 result = __float2bfloat16(sum);
    accum[n] = *reinterpret_cast<unsigned short*>(&result);
}

// ── Graphable Kernel Variants ─────────────────────────────────────────
//
// These variants read position/token_id from GPU pointers instead of
// immediate kernel parameters. This allows them to be captured in CUDA
// graphs — the pointer address is fixed, but the value at the pointer
// can be updated between graph replays.

extern "C" __global__ void embedding_lookup_g(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ d_token_id,   // read from GPU pointer
    int hidden_size
) {
    int token_id = *d_token_id;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        output[i] = table[token_id * hidden_size + i];
    }
}

extern "C" __global__ void apply_rope_g(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ d_position,   // read from GPU pointer
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim
) {
    int position = *d_position;
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

extern "C" __global__ void kv_cache_write_g(
    __nv_fp8_e4m3* __restrict__ k_cache,
    __nv_fp8_e4m3* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ d_position,   // read from GPU pointer
    int kv_stride
) {
    int position = *d_position;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_stride) {
        k_cache[position * kv_stride + i] = f32_to_fp8e4m3(k[i]);
        v_cache[position * kv_stride + i] = f32_to_fp8e4m3(v[i]);
    }
}

// GQA attention variant for CUDA graph capture.
// Tiled 3-pass: (1) find max, (2) compute weights + V accumulation in tiles.
// Weights are computed once per position and stored in shared memory,
// then reused across all V dimensions — eliminates redundant K·Q recomputation.
// Reads seq_len from a GPU pointer so it can change between graph replays.
#define GQA_TILE_SIZE 4096
extern "C" __global__ void gqa_attention_g(
    float* __restrict__ output,
    const float* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,    // read from GPU pointer
    int max_seq
) {
    int seq_len = *d_seq_len;
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int kv_stride = num_kv_heads * head_dim;

    const float* q_head = q + qh * head_dim;

    // Shared memory layout:
    //   s_q[0..head_dim-1]              — Q vector (float)
    //   smem_reduce[0..num_warps-1]     — warp reduction scratch (float)
    //   smem_weights[0..TILE_SIZE-1]    — attention weights tile (float)
    extern __shared__ float smem[];
    float* s_q = smem;

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    float* smem_reduce = smem + head_dim;
    float* smem_weights = smem_reduce + num_warps;

    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    // Pass 1: find max score (distribute positions across threads)
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        for (int d = 0; d < head_dim; d += 16) {
            uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
            const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                score += s_q[d + j] * fp8e4m3_to_f32(
                    *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
            }
        }
        score *= sm_scale;
        local_max = fmaxf(local_max, score);
    }
    // Warp reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) m = fmaxf(m, smem_reduce[w]);
        smem_reduce[0] = m;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    // Pass 2 (tiled): compute weights, store in smem, accumulate sum_exp and weighted V
    float local_sum = 0.0f;
    // Per-thread V accumulators (one per dimension this thread owns)
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // supports head_dim up to 4*num_threads

    for (int tile_start = 0; tile_start < seq_len; tile_start += GQA_TILE_SIZE) {
        int tile_end = tile_start + GQA_TILE_SIZE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Phase A: compute exp(score - max) for each position, store in smem_weights
        for (int ti = tid; ti < tile_len; ti += num_threads) {
            int pos = tile_start + ti;
            float score = 0.0f;
            const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
            for (int d = 0; d < head_dim; d += 16) {
                uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
                const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
                }
            }
            float w = __expf(score * sm_scale - global_max);
            smem_weights[ti] = w;
            local_sum += w;
        }
        __syncthreads();

        // Phase B: accumulate weighted V using stored weights
        const __nv_fp8_e4m3* v_base = v_cache + tile_start * kv_stride + kv_head * head_dim;
        int di = 0;
        for (int d = tid; d < head_dim; d += num_threads) {
            float acc = 0.0f;
            for (int ti = 0; ti < tile_len; ti++) {
                acc += smem_weights[ti] * fp8e4m3_to_f32(v_base[ti * kv_stride + d]);
            }
            v_acc[di++] += acc;
        }
        __syncthreads();
    }

    // Warp reduce sum_exp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps; w++) s += smem_reduce[w];
        smem_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = (smem_reduce[0] > 0.0f) ? 1.0f / smem_reduce[0] : 0.0f;

    // Write normalized output
    float* out = output + qh * head_dim;
    int di = 0;
    for (int d = tid; d < head_dim; d += num_threads) {
        out[d] = v_acc[di++] * inv_sum;
    }
}

// Same as gqa_attention_g but outputs BF16 directly (eliminates fp32_to_bf16 kernel)
extern "C" __global__ void gqa_attention_g_bf16(
    unsigned short* __restrict__ output,   // [num_q_heads * head_dim] BF16
    const float* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,
    int max_seq
) {
    int seq_len = *d_seq_len;
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int kv_stride = num_kv_heads * head_dim;

    const float* q_head = q + qh * head_dim;

    extern __shared__ float smem[];
    float* s_q = smem;

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    float* smem_reduce = smem + head_dim;
    float* smem_weights = smem_reduce + num_warps;

    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    // Pass 1: find max score
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        for (int d = 0; d < head_dim; d += 16) {
            uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
            const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                score += s_q[d + j] * fp8e4m3_to_f32(
                    *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
            }
        }
        score *= sm_scale;
        local_max = fmaxf(local_max, score);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) m = fmaxf(m, smem_reduce[w]);
        smem_reduce[0] = m;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    // Pass 2 (tiled): compute weights, store in smem, accumulate sum_exp and weighted V
    float local_sum = 0.0f;
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_start = 0; tile_start < seq_len; tile_start += GQA_TILE_SIZE) {
        int tile_end = tile_start + GQA_TILE_SIZE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Phase A: compute weights and store in smem
        for (int ti = tid; ti < tile_len; ti += num_threads) {
            int pos = tile_start + ti;
            float score = 0.0f;
            const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
            for (int d = 0; d < head_dim; d += 16) {
                uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
                const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    score += s_q[d + j] * fp8e4m3_to_f32(
                        *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
                }
            }
            float w = __expf(score * sm_scale - global_max);
            smem_weights[ti] = w;
            local_sum += w;
        }
        __syncthreads();

        // Phase B: accumulate weighted V using stored weights
        const __nv_fp8_e4m3* v_base = v_cache + tile_start * kv_stride + kv_head * head_dim;
        int di = 0;
        for (int d = tid; d < head_dim; d += num_threads) {
            float acc = 0.0f;
            for (int ti = 0; ti < tile_len; ti++) {
                acc += smem_weights[ti] * fp8e4m3_to_f32(v_base[ti * kv_stride + d]);
            }
            v_acc[di++] += acc;
        }
        __syncthreads();
    }

    // Warp reduce sum_exp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps; w++) s += smem_reduce[w];
        smem_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = (smem_reduce[0] > 0.0f) ? 1.0f / smem_reduce[0] : 0.0f;

    // Write normalized output as BF16
    unsigned short* out = output + qh * head_dim;
    int di = 0;
    for (int d = tid; d < head_dim; d += num_threads) {
        __nv_bfloat16 result = __float2bfloat16(v_acc[di++] * inv_sum);
        out[d] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── FlashDecoding-style tiled GQA for CUDA graph capture ─────────────
//
// Replaces the 2-pass gqa_attention_g with a tiled approach that:
//   1. Reads K cache ONCE per tile (not twice)
//   2. Uses grid.y = max_tiles for SM utilization (excess tiles early-exit)
//   3. Reads seq_len from GPU pointer for graph replay compatibility
//
// Each block handles one Q head × one KV tile.
// Output per tile: unnormalised weighted V (FP32), local max, local sum_exp.
// A reduce kernel merges tiles using log-sum-exp rescaling.
//
// Grid: (num_q_heads, max_tiles, 1)   Block: (256, 1, 1)
// Shared memory: (head_dim + tile_size) * 4 + 128 bytes

extern "C" __global__ void gqa_attention_tiled_g(
    float* __restrict__ partial_o,     // [num_q_heads, max_tiles, head_dim]
    float* __restrict__ partial_lse,   // [num_q_heads, max_tiles, 2] (max, sum_exp)
    const float* __restrict__ q,       // [num_q_heads * head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache,  // [max_seq, kv_stride] FP8
    const __nv_fp8_e4m3* __restrict__ v_cache,  // [max_seq, kv_stride] FP8
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,    // read from GPU pointer
    int tile_size,
    int max_tiles
) {
    int seq_len = *d_seq_len;
    int qh = blockIdx.x;
    int tile_idx = (int)blockIdx.y;
    int num_tiles = (seq_len + tile_size - 1) / tile_size;
    if (qh >= num_q_heads || tile_idx >= num_tiles) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int kv_stride = num_kv_heads * head_dim;

    const float* q_head = q + qh * head_dim;

    int tile_start = tile_idx * tile_size;
    int tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;
    int tile_len = tile_end - tile_start;

    // Dynamic shared memory layout:
    //   s_q[0..head_dim-1]:          Q vector preloaded (float)
    //   smem_scores[0..tile_size-1]: attention scores (float)
    //   smem_reduce[0..31]:          warp scratch (float)
    extern __shared__ float smem[];
    float* s_q = smem;
    float* smem_scores = smem + head_dim;
    float* smem_reduce = smem_scores + tile_size;

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    // Preload Q into shared memory
    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    // Step 1: Q·K dot products for this tile (vectorized FP8 loads)
    for (int i = tid; i < tile_len; i += num_threads) {
        int pos = tile_start + i;
        float score = 0.0f;
        const __nv_fp8_e4m3* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        for (int d = 0; d < head_dim; d += 16) {
            uint4 k_packed = *reinterpret_cast<const uint4*>(k_vec + d);
            const unsigned char* kb = reinterpret_cast<const unsigned char*>(&k_packed);
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                score += s_q[d + j] * fp8e4m3_to_f32(
                    *reinterpret_cast<const __nv_fp8_e4m3*>(&kb[j]));
            }
        }
        smem_scores[i] = score * sm_scale;
    }
    __syncthreads();

    // Step 2: Local max (parallel reduction)
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += num_threads) {
        local_max = fmaxf(local_max, smem_scores[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float gmax = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) gmax = fmaxf(gmax, smem_reduce[w]);
        smem_reduce[0] = gmax;
    }
    __syncthreads();
    float tile_max = smem_reduce[0];

    // Step 3: exp(score - max) in-place, compute local sum
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += num_threads) {
        float w = __expf(smem_scores[i] - tile_max);
        smem_scores[i] = w;
        local_sum += w;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float gsum = 0.0f;
        for (int w = 0; w < num_warps; w++) gsum += smem_reduce[w];
        smem_reduce[0] = gsum;
    }
    __syncthreads();
    float tile_sum = smem_reduce[0];

    // Step 4: Unnormalised weighted V sum
    float* out_partial = partial_o + (qh * max_tiles + tile_idx) * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int i = 0; i < tile_len; i++) {
            acc += smem_scores[i] * fp8e4m3_to_f32(
                v_cache[(tile_start + i) * kv_stride + kv_head * head_dim + d]);
        }
        out_partial[d] = acc;
    }

    // Step 5: Store tile statistics
    if (tid == 0) {
        float* lse = partial_lse + (qh * max_tiles + tile_idx) * 2;
        lse[0] = tile_max;
        lse[1] = tile_sum;
    }
}

// Merge tiled GQA attention partials for CUDA graph capture.
// Reads seq_len from GPU pointer, computes num_tiles at runtime.
// Excess blocks early-exit when qh >= num_q_heads.
//
// Grid: (num_q_heads, 1, 1)   Block: (256, 1, 1)
// Shared memory: max_tiles * 4 bytes

extern "C" __global__ void gqa_attention_reduce_g(
    float* __restrict__ output,            // [num_q_heads * head_dim]
    const float* __restrict__ partial_o,   // [num_q_heads, max_tiles, head_dim]
    const float* __restrict__ partial_lse, // [num_q_heads, max_tiles, 2]
    int num_q_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,
    int tile_size,
    int max_tiles
) {
    int seq_len = *d_seq_len;
    int num_tiles = (seq_len + tile_size - 1) / tile_size;
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float smem[];
    // smem[0..max_tiles-1] = per-tile normalised correction weight

    const float* lse = partial_lse + qh * max_tiles * 2;

    // Thread 0 computes correction weights
    if (tid == 0) {
        float global_max = -1e30f;
        for (int t = 0; t < num_tiles; t++) {
            global_max = fmaxf(global_max, lse[t * 2]);
        }
        float global_sum = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            float correction = __expf(lse[t * 2] - global_max);
            global_sum += correction * lse[t * 2 + 1];
            smem[t] = correction;
        }
        float inv_sum = 1.0f / global_sum;
        for (int t = 0; t < num_tiles; t++) {
            smem[t] *= inv_sum;
        }
    }
    __syncthreads();

    // All threads cooperate on output dimensions
    const float* po = partial_o + qh * max_tiles * head_dim;
    float* out = output + qh * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            acc += smem[t] * po[t * head_dim + d];
        }
        out[d] = acc;
    }
}

// ── Simple INT4 GEMV (for draft model) ───────────────────────────────
//
// Row-major packed INT4 weights: each byte holds 2 values (low nibble = even col, high nibble = odd col).
// Unsigned 0..15, subtract 8 to get signed -8..+7.
// Per-group FP32 scales: one scale per group_size elements per row.
//
// Launch: grid=(ceil(rows/8), 1, 1), block=(256, 1, 1)
// 8 warps per block, each warp handles one output row.
// Shared memory: cols * 2 bytes (BF16 input preloaded).

extern "C" __global__ void simple_int4_gemv_f32(
    const unsigned char* __restrict__ packed_w,  // [rows, cols/2] packed INT4
    const float* __restrict__ scales,            // [rows, cols/group_size] FP32
    const unsigned short* __restrict__ input,    // [cols] BF16
    float* __restrict__ output,                  // [rows] FP32
    int rows, int cols, int group_size
) {
    extern __shared__ unsigned short s_input[];
    int tid = threadIdx.x;

    // Cooperative load input to shared memory
    for (int i = tid; i < cols; i += 256) {
        s_input[i] = input[i];
    }
    __syncthreads();

    int warp_id = tid / 32;
    int lane = tid % 32;
    int row = blockIdx.x * 8 + warp_id;

    if (row >= rows) return;

    int half_cols = cols / 2;
    const unsigned char* w_row = packed_w + (long long)row * half_cols;
    const float* s_row = scales + (long long)row * (cols / group_size);

    float acc = 0.0f;

    // Vectorized path: read 4 bytes (8 weights) per lane per iteration via uint32.
    // 32 lanes × 8 weights = 256 weights per warp iteration (vs 64 in scalar path).
    // Each uint32 read is 128 bytes/warp — fully utilizes a cache line.
    // All 8 weights share one scale (group_size >= 8 guaranteed for INT4 AWQ).
    const unsigned int* w_row_u32 = (const unsigned int*)w_row;
    int packed_per_row = half_cols >> 2;  // number of uint32 per row

    for (int ki = lane; ki < packed_per_row; ki += 32) {
        unsigned int packed4 = __ldg(w_row_u32 + ki);
        int base_k = ki << 3;  // ki * 8
        float scale = s_row[base_k / group_size];

        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int byte_val = (packed4 >> (i * 8)) & 0xFF;
            int lo = (byte_val & 0xF) - 8;
            int hi = (byte_val >> 4) - 8;
            float x0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[base_k + i*2]));
            float x1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[base_k + i*2 + 1]));
            local_sum += (float)lo * x0 + (float)hi * x1;
        }
        acc += scale * local_sum;
    }

    // Handle remainder if cols not divisible by 8 (rare — all known models are aligned)
    int vec_cols = packed_per_row << 3;
    for (int k = vec_cols + lane * 2; k < cols; k += 64) {
        unsigned char packed = w_row[k / 2];
        int lo = (int)(packed & 0xF) - 8;
        int hi = (int)(packed >> 4) - 8;
        float scale = s_row[k / group_size];
        float x0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
        float x1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k + 1]));
        acc += scale * ((float)lo * x0 + (float)hi * x1);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        output[row] = acc;
    }
}

// Same as above but with BF16 output.
extern "C" __global__ void simple_int4_gemv_bf16(
    const unsigned char* __restrict__ packed_w,  // [rows, cols/2] packed INT4
    const float* __restrict__ scales,            // [rows, cols/group_size] FP32
    const unsigned short* __restrict__ input,    // [cols] BF16
    unsigned short* __restrict__ output,         // [rows] BF16
    int rows, int cols, int group_size
) {
    extern __shared__ unsigned short s_input[];
    int tid = threadIdx.x;

    for (int i = tid; i < cols; i += 256) {
        s_input[i] = input[i];
    }
    __syncthreads();

    int warp_id = tid / 32;
    int lane = tid % 32;
    int row = blockIdx.x * 8 + warp_id;

    if (row >= rows) return;

    int half_cols = cols / 2;
    const unsigned char* w_row = packed_w + (long long)row * half_cols;
    const float* s_row = scales + (long long)row * (cols / group_size);

    float acc = 0.0f;

    const unsigned int* w_row_u32 = (const unsigned int*)w_row;
    int packed_per_row = half_cols >> 2;

    for (int ki = lane; ki < packed_per_row; ki += 32) {
        unsigned int packed4 = __ldg(w_row_u32 + ki);
        int base_k = ki << 3;
        float scale = s_row[base_k / group_size];

        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int byte_val = (packed4 >> (i * 8)) & 0xFF;
            int lo = (byte_val & 0xF) - 8;
            int hi = (byte_val >> 4) - 8;
            float x0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[base_k + i*2]));
            float x1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[base_k + i*2 + 1]));
            local_sum += (float)lo * x0 + (float)hi * x1;
        }
        acc += scale * local_sum;
    }

    int vec_cols = packed_per_row << 3;
    for (int k = vec_cols + lane * 2; k < cols; k += 64) {
        unsigned char packed = w_row[k / 2];
        int lo = (int)(packed & 0xF) - 8;
        int hi = (int)(packed >> 4) - 8;
        float scale = s_row[k / group_size];
        float x0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
        float x1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k + 1]));
        acc += scale * ((float)lo * x0 + (float)hi * x1);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        output[row] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── BF16 → FP8 E4M3 conversion ──────────────────────────────────────────
// ── Marlin INT8 GEMV (W8A16 attention decode) ────────────────────────
//
// Computes output[N] = dequant(marlin_packed[K/16, 4*N], scales[K/gs, N]) @ input[K]
//
// Same structure as marlin_gemv_int4 but:
//   - 4 bytes per u32 (PACK_FACTOR_INT8 = 4, vs 8 nibbles for INT4)
//   - out_cols = 4*N (vs 2*N)
//   - Unsigned offset: raw - 128 (vs raw - 8)
//   - INT8 weight permutation (interleave [0,2,1,3] groups of 4)
//
// Launch: grid=(N/16, 1, 1), block=(256, 1, 1)
// Shared memory: K*2 + 1024*4 + 64*4 bytes

extern "C" __global__ void marlin_gemv_int8(
    const unsigned int* __restrict__ packed,   // [K/16, 4*N] Marlin tile-permuted INT8
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    unsigned short* __restrict__ output,       // [N] BF16
    const int* __restrict__ inv_weight_perm,   // [1024] INT8 inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // ── Cooperative preload: input, inv_weight_perm, inv_scale_perm ──
    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;    // K / 16
    int out_cols = N * 4;          // 4*N (4 bytes per u32 for INT8)

    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        output[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── Marlin INT8 GEMV with FP32 output (for attention projections) ────
//
// Same as marlin_gemv_int8 but outputs FP32 (needed by attention score paths).

extern "C" __global__ void marlin_gemv_int8_f32(
    const unsigned int* __restrict__ packed,   // [K/16, 4*N] Marlin tile-permuted INT8
    const unsigned short* __restrict__ scales, // [K/gs, N] Marlin scale-permuted BF16
    const unsigned short* __restrict__ input,  // [K] BF16
    float* __restrict__ output,                // [N] FP32
    const int* __restrict__ inv_weight_perm,   // [1024] INT8 inverse weight perm
    const int* __restrict__ inv_scale_perm,    // [64] inverse scale perm
    int K, int N, int group_size
) {
    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    for (int i = tid; i < K; i += 256) {
        s_input[i] = input[i];
    }
    for (int i = tid; i < 1024; i += 256) {
        s_inv_wperm[i] = inv_weight_perm[i];
    }
    if (tid < 64) {
        s_inv_sperm[tid] = inv_scale_perm[tid];
    }
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;

    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N * 4;

    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    // Pre-compute permutation indices: constant across all k_tiles
    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 2;
        perm_shift[tk] = (perm_pos & 3) << 3;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xFF;
                float w_val = (float)(raw - 128);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        output[n] = acc;
    }
}

// ── Fused LA Post-Projection Kernel ─────────────────────────────────────
//
// Combines 6 separate kernels into one per-head launch:
//   1. Repeat-interleave Q and K from [nk, dk] to [nv, dk]
//   2. L2 normalize Q (with scale) and K (scale=1.0)
//   3. Gated delta net step (state decay, recall, update, readout)
//   4. Gated RMSNorm + SiLU → BF16 output
//
// One block per head (gridDim.x = nv). Eliminates 5 kernel launches per LA layer.
// Requires shared memory: dk * 2 (Q and K) + dv + 32 (for warp reduction) floats.
//
extern "C" __global__ void la_fused_post_proj(
    float* __restrict__ state,             // [nv, dk, dv] in/out (recurrence state)
    const float* __restrict__ q_in,        // [nk * dk] (from conv1d output: SiLU(conv(proj_q)))
    const float* __restrict__ k_in,        // [nk * dk] (from conv1d output: SiLU(conv(proj_k)))
    const float* __restrict__ v_in,        // [nv * dv] (from conv1d output: SiLU(conv(proj_v)))
    const float* __restrict__ gate,        // [nv] (from compute_gate_beta)
    const float* __restrict__ beta,        // [nv] (from compute_gate_beta)
    const float* __restrict__ z,           // [nv * dv] (saved from uninterleave)
    const float* __restrict__ norm_weight, // [dv] (per-head norm weights)
    unsigned short* __restrict__ bf16_out, // [nv * dv] BF16 output
    float q_scale,                         // L2 norm scale for Q
    float eps,                             // RMSNorm epsilon
    long long dims_packed                  // (nv_dk << 32) | dv_ratio, each 16-bit packed
) {
    // Unpack: high 32 bits = (nv << 16) | dk, low 32 bits = (dv << 16) | ratio
    int nv_dk = (int)(dims_packed >> 32);
    int dv_ratio = (int)(dims_packed & 0xFFFFFFFF);
    int nv = (nv_dk >> 16) & 0xFFFF;
    int dk = nv_dk & 0xFFFF;
    int dv = (dv_ratio >> 16) & 0xFFFF;
    int ratio = dv_ratio & 0xFFFF;

    int head = blockIdx.x;
    if (head >= nv) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory layout: [dk] Q + [dk] K + [dv + 32] scratch
    extern __shared__ float smem[];
    float* s_q = smem;              // [dk]
    float* s_k = smem + dk;         // [dk]
    float* s_scratch = smem + 2 * dk; // [dv + 32]

    // ── Step 1: Repeat-interleave Q and K ──
    // Map output head -> input head: in_head = head / ratio
    int in_head = head / ratio;
    for (int i = tid; i < dk; i += num_threads) {
        s_q[i] = q_in[in_head * dk + i];
        s_k[i] = k_in[in_head * dk + i];
    }
    __syncthreads();

    // ── Step 2: L2 normalize Q (with scale) and K (scale=1.0) ──
    float sum_sq_q = 0.0f;
    float sum_sq_k = 0.0f;
    for (int i = tid; i < dk; i += num_threads) {
        sum_sq_q += s_q[i] * s_q[i];
        sum_sq_k += s_k[i] * s_k[i];
    }

    // Warp reduction for both
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq_q += __shfl_down_sync(0xffffffff, sum_sq_q, offset);
        sum_sq_k += __shfl_down_sync(0xffffffff, sum_sq_k, offset);
    }

    __shared__ float warp_sums_q[32];
    __shared__ float warp_sums_k[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) {
        warp_sums_q[warp_id] = sum_sq_q;
        warp_sums_k[warp_id] = sum_sq_k;
    }
    __syncthreads();

    if (tid == 0) {
        float total_q = 0.0f, total_k = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) {
            total_q += warp_sums_q[w];
            total_k += warp_sums_k[w];
        }
        warp_sums_q[0] = rsqrtf(total_q + 1e-12f) * q_scale;
        warp_sums_k[0] = rsqrtf(total_k + 1e-12f);
    }
    __syncthreads();

    float norm_q = warp_sums_q[0];
    float norm_k = warp_sums_k[0];
    for (int i = tid; i < dk; i += num_threads) {
        s_q[i] *= norm_q;
        s_k[i] *= norm_k;
    }
    __syncthreads();

    // ── Step 3: Gated delta net step ──
    float g = gate[head];
    float b = beta[head];
    int mat_size = dk * dv;
    float* S = state + head * mat_size;
    const float* v_h = v_in + head * dv;

    // Decay state
    for (int idx = tid; idx < mat_size; idx += num_threads) {
        S[idx] *= g;
    }
    __syncthreads();

    // Memory recall + delta + state update + readout
    // Store recurrence output in s_scratch[0..dv]
    for (int j = tid; j < dv; j += num_threads) {
        float kv_mem = 0.0f;
        for (int i = 0; i < dk; i++) {
            kv_mem += S[i * dv + j] * s_k[i];
        }
        float delta = (v_h[j] - kv_mem) * b;
        for (int i = 0; i < dk; i++) {
            S[i * dv + j] += s_k[i] * delta;
        }
        // Readout
        float acc = 0.0f;
        for (int i = 0; i < dk; i++) {
            acc += s_q[i] * S[i * dv + j];
        }
        s_scratch[j] = acc;
    }
    __syncthreads();

    // ── Step 4: Gated RMSNorm + SiLU → BF16 ──
    // RMSNorm on s_scratch[0..dv]
    float sum_sq = 0.0f;
    for (int i = tid; i < dv; i += num_threads) {
        sum_sq += s_scratch[i] * s_scratch[i];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane_id == 0) warp_sums_q[warp_id] = sum_sq;  // reuse warp_sums_q
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (num_threads + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) total += warp_sums_q[w];
        warp_sums_q[0] = rsqrtf(total / (float)dv + eps);
    }
    __syncthreads();
    float rms_scale = warp_sums_q[0];

    const float* z_h = z + head * dv;
    unsigned short* out_h = bf16_out + head * dv;
    for (int i = tid; i < dv; i += num_threads) {
        float normed = s_scratch[i] * rms_scale * norm_weight[i];
        float zv = z_h[i];
        float silu_z = zv / (1.0f + __expf(-zv));
        __nv_bfloat16 result = __float2bfloat16(silu_z * normed);
        out_h[i] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════
// Mamba2 SSM kernels (Nemotron-H hybrid models)
// ═══════════════════════════════════════════════════════════════════════

// Conv1d state update: shift conv_state left, insert new input, apply conv weights + SiLU.
// conv_state: [conv_dim, conv_kernel] FP32 — updated in-place (shift left, insert x_input)
// x_input: [conv_dim] FP32 (new input to shift in)
// conv_weight: [conv_dim, 1, conv_kernel] BF16 (original HF shape, flattened to [conv_dim, conv_kernel])
// conv_bias: [conv_dim] BF16 (or null)
// output: [conv_dim] FP32 (convolved + SiLU activated)
// Grid: (ceil(conv_dim/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void mamba2_conv1d(
    float* __restrict__ conv_state,
    const float* __restrict__ x_input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    int conv_dim, int conv_kernel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_dim) return;

    // Shift conv_state left: state[idx][j] = state[idx][j+1] for j < kernel-1
    float* state_row = conv_state + idx * conv_kernel;
    for (int j = 0; j < conv_kernel - 1; j++) {
        state_row[j] = state_row[j + 1];
    }
    // Insert new input at the right end
    state_row[conv_kernel - 1] = x_input[idx];

    // Convolve: dot product of state with conv_weight
    const float* weight_row = conv_weight + idx * conv_kernel;
    float acc = 0.0f;
    for (int j = 0; j < conv_kernel; j++) {
        acc += state_row[j] * weight_row[j];
    }
    // Add bias if present
    if (conv_bias != nullptr) {
        acc += conv_bias[idx];
    }

    // SiLU activation: x * sigmoid(x)
    float sigmoid_val = 1.0f / (1.0f + __expf(-acc));
    output[idx] = acc * sigmoid_val;
}

// Discretize: compute dt, A_bar, B_bar from continuous parameters.
// For Mamba2 with n_groups: B and C are [n_groups, state_size], shared across heads in each group.
// delta: [num_heads] FP32 (from in_proj split)
// dt_bias: [num_heads] FP32
// A_log: [num_heads] FP32 (log-space A parameter, NOT exp'd yet)
// B: [n_groups * state_size] FP32 (from in_proj split, n_groups groups)
// dt_out: [num_heads] FP32 = softplus(delta + dt_bias)
// A_bar: [num_heads] FP32 = exp(A * dt) where A = -exp(A_log)
// B_bar: [num_heads * state_size] FP32 = dt[h] * B[group(h)]
// Grid: (1, 1, 1), Block: (max(num_heads, 256), 1, 1) — num_heads is typically 64-128
extern "C" __global__ void mamba2_discretize(
    const float* __restrict__ delta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ B,
    float* __restrict__ dt_out,
    float* __restrict__ A_bar,
    float* __restrict__ B_bar,
    int num_heads, int state_size, int n_groups
) {
    int h = threadIdx.x;
    if (h >= num_heads) return;

    // dt = softplus(delta + dt_bias) = log(1 + exp(x))
    float x = delta[h] + dt_bias[h];
    float dt;
    if (x > 20.0f) {
        dt = x;  // softplus(x) ≈ x for large x
    } else {
        dt = logf(1.0f + __expf(x));
    }
    dt_out[h] = dt;

    // A = -exp(A_log), A_bar = exp(A * dt)
    float A_val = -__expf(A_log[h]);
    A_bar[h] = __expf(A_val * dt);

    // B_bar: each head maps to a group. B_bar[h, s] = dt[h] * B[group(h), s]
    int heads_per_group = num_heads / n_groups;
    int group = h / heads_per_group;
    const float* B_group = B + group * state_size;
    float* B_bar_h = B_bar + h * state_size;
    for (int s = 0; s < state_size; s++) {
        B_bar_h[s] = dt * B_group[s];
    }
}

// SSM recurrence step: h = A_bar * h + outer(B_bar, x), y = C · h
// This is the core Mamba2 decode operation — O(1) per token.
//
// ssm_state: [num_heads, head_dim, state_size] FP32 (updated in-place)
// x: [num_heads * head_dim] FP32 (convolved input, split from in_proj)
// A_bar: [num_heads] FP32 (per-head decay)
// B_bar: [num_heads * state_size] FP32 (per-head input gate)
// C: [n_groups * state_size] FP32 (output gate, grouped)
// y: [num_heads * head_dim] FP32 (output)
//
// Grid: (num_heads, 1, 1), Block: (min(head_dim, 256), 1, 1)
// Each block handles one head: loops over head_dim elements.
extern "C" __global__ void mamba2_ssm_step(
    float* __restrict__ ssm_state,
    const float* __restrict__ x,
    const float* __restrict__ A_bar,
    const float* __restrict__ B_bar,
    const float* __restrict__ C,
    float* __restrict__ y,
    int num_heads, int head_dim, int state_size, int n_groups
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    float a = A_bar[h];
    const float* b_h = B_bar + h * state_size;
    int heads_per_group = num_heads / n_groups;
    int group = h / heads_per_group;
    const float* c_group = C + group * state_size;
    float x_val = x[h * head_dim + d];

    // State pointer for this head and dim: ssm_state[h, d, :] = [state_size] FP32
    float* state = ssm_state + (h * head_dim + d) * state_size;

    // h = A_bar * h + B_bar * x, y = C @ h
    float y_val = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float new_state = a * state[s] + b_h[s] * x_val;
        state[s] = new_state;
        y_val += c_group[s] * new_state;
    }

    y[h * head_dim + d] = y_val;
}

// Gate + skip + per-group RMSNorm: output = norm_weight * groupRMSNorm(silu(z) * (y + D*x)) -> BF16
// Matches Mamba2 reference: gate is applied BEFORE normalization, norm is per-group.
//
// z: [d_inner] FP32 (gate values from in_proj)
// y: [d_inner] FP32 (SSM output, does NOT include D*x)
// D: [num_heads] FP32 (skip connection per head)
// x_skip: [d_inner] FP32 (post-conv x, same x that went into SSM)
// norm_weight: [d_inner] FP32 (RMSNorm weight)
// output: [d_inner] BF16
//
// Order: gated = (y + D*x) * silu(z), then per-group RMSNorm(gated) * norm_weight
// Grid: (n_groups, 1, 1), Block: (256, 1, 1) — one block per group
extern "C" __global__ void mamba2_gate_output(
    const float* __restrict__ z,
    const float* __restrict__ y,
    const float* __restrict__ D,
    const float* __restrict__ x_skip,
    const float* __restrict__ norm_weight,
    unsigned short* __restrict__ output,
    int d_inner, int head_dim, int n_groups, float eps
) {
    int group = blockIdx.x;
    int group_size = d_inner / n_groups;
    int group_offset = group * group_size;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory layout: [0..num_threads-1] for reduction, [num_threads..num_threads+group_size-1] for gated values
    extern __shared__ float smem[];
    float* s_gated = smem + num_threads;

    // Pass 1: compute gated = (y + D*x) * silu(z), store in shared mem, accumulate sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < group_size; i += num_threads) {
        int idx = group_offset + i;
        int head = idx / head_dim;
        float y_d = y[idx] + D[head] * x_skip[idx];
        float z_val = z[idx];
        float silu_z = z_val / (1.0f + __expf(-z_val));
        float gated = y_d * silu_z;
        s_gated[i] = gated;
        local_ss += gated * gated;
    }
    smem[tid] = local_ss;
    __syncthreads();

    // Block reduction for sum of squares
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / (float)group_size + eps);

    // Pass 2: apply per-group RMSNorm * weight -> BF16 output
    for (int i = tid; i < group_size; i += num_threads) {
        int idx = group_offset + i;
        float normed = s_gated[i] * rms_inv * norm_weight[idx];
        __nv_bfloat16 bf16_val = __float2bfloat16(normed);
        output[idx] = *reinterpret_cast<unsigned short*>(&bf16_val);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// relu2 expert kernels (Nemotron LatentMoE)
// Analogous to fused_silu_w2_batched but with relu^2 activation
// and no gate projection (input is just up_proj output, not gate+up).
// ═══════════════════════════════════════════════════════════════════════

// INT4 Marlin variant — relu(up_out)^2 then Marlin INT4 GEMV for down_proj
// Grid: (n_tiles, 1, num_experts), Block: (256, 1, 1)
// up_outs: [num_experts * K] BF16 — just up_proj output, NO gate
extern "C" __global__ void relu2_w2_batched(
    const unsigned long long* __restrict__ w2_packed_ptrs,
    const unsigned long long* __restrict__ w2_scales_ptrs,
    const unsigned short* __restrict__ up_outs,
    unsigned short* __restrict__ expert_outs,
    const int* __restrict__ inv_weight_perm,
    const int* __restrict__ inv_scale_perm,
    int K, int N, int group_size,
    const float* __restrict__ weights                       // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    const unsigned int* packed = (const unsigned int*)w2_packed_ptrs[expert_idx];
    const unsigned short* w2_scales = (const unsigned short*)w2_scales_ptrs[expert_idx];
    const unsigned short* up_out = up_outs + expert_idx * K;  // just K, not 2*K
    unsigned short* out = expert_outs + expert_idx * N;

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // Apply relu^2 while loading up_out into shared memory
    for (int i = tid; i < K; i += 256) {
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&up_out[i]));
        float relu_u = fmaxf(u, 0.0f);
        float relu2_u = relu_u * relu_u;
        __nv_bfloat16 val = __float2bfloat16(relu2_u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;
    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int out_cols = N << 1;
    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    int perm_shift[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        int perm_pos = (chunk << 10) + s_inv_wperm[local_idx];
        perm_u32_col[tk] = perm_pos >> 3;
        perm_shift[tk] = (perm_pos & 7) << 2;
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * out_cols;
        int sg_start = k_base / group_size;
        int sg_end = (k_base + 15) / group_size;

        if (sg_start != cur_scale_group) {
            cur_scale_group = sg_start;
            int scale_flat = sg_start * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        if (sg_start == sg_end) {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
                acc += w_val * cached_scale * x;
            }
        } else {
            #pragma unroll
            for (int tk = 0; tk < 16; tk++) {
                int k = k_base + tk;
                int sg = k / group_size;
                if (sg != cur_scale_group) {
                    cur_scale_group = sg;
                    int scale_flat = sg * N + n;
                    int schunk = scale_flat >> 6;
                    int slocal = scale_flat & 63;
                    int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
                    unsigned short scale_bits = w2_scales[sperm_pos];
                    cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
                }
                unsigned int word = packed[row_base + perm_u32_col[tk]];
                int raw = (word >> perm_shift[tk]) & 0xF;
                float w_val = (float)(raw - 8);
                float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k]));
                acc += w_val * cached_scale * x;
            }
        }
    }

    // Warp shuffle reduction across 16 k_slices
    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        out[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// INT8 Marlin variant — relu(up_out)^2 then Marlin INT8 GEMV for down_proj
// Grid: (n_tiles, 1, num_experts), Block: (256, 1, 1)
extern "C" __global__ void relu2_w2_int8_batched(
    const unsigned long long* __restrict__ w2_packed_ptrs,
    const unsigned long long* __restrict__ w2_scales_ptrs,
    const unsigned short* __restrict__ up_outs,
    unsigned short* __restrict__ expert_outs,
    const int* __restrict__ inv_weight_perm,
    const int* __restrict__ inv_scale_perm,
    int K, int N, int group_size,
    const float* __restrict__ weights                       // [num_experts] — skip if zero
) {
    int expert_idx = blockIdx.z;
    if (weights[expert_idx] == 0.0f) return;
    const signed char* packed = (const signed char*)w2_packed_ptrs[expert_idx];
    const unsigned short* w2_scales = (const unsigned short*)w2_scales_ptrs[expert_idx];
    const unsigned short* up_out = up_outs + expert_idx * K;
    unsigned short* out = expert_outs + expert_idx * N;

    extern __shared__ char smem_raw[];
    unsigned short* s_input = (unsigned short*)smem_raw;
    int* s_inv_wperm = (int*)(smem_raw + K * 2);
    int* s_inv_sperm = (int*)(smem_raw + K * 2 + 1024 * 4);

    int tid = threadIdx.x;

    // Apply relu^2 while loading up_out into shared memory
    for (int i = tid; i < K; i += 256) {
        float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&up_out[i]));
        float relu_u = fmaxf(u, 0.0f);
        float relu2_u = relu_u * relu_u;
        __nv_bfloat16 val = __float2bfloat16(relu2_u);
        s_input[i] = *reinterpret_cast<unsigned short*>(&val);
    }
    for (int i = tid; i < 1024; i += 256) s_inv_wperm[i] = inv_weight_perm[i];
    if (tid < 64) s_inv_sperm[tid] = inv_scale_perm[tid];
    __syncthreads();

    int k_slice = tid & 15;
    int tn = tid >> 4;
    int n_tile = blockIdx.x;
    int n = n_tile * 16 + tn;
    if (n >= N) return;

    int k_tiles_total = K >> 4;
    int tiles_per_slice = k_tiles_total >> 4;
    int kt_start = k_slice * tiles_per_slice;
    int kt_end = (k_slice == 15) ? k_tiles_total : (kt_start + tiles_per_slice);

    int tile_base = (n_tile << 8) + tn;
    int perm_u32_col[16];
    #pragma unroll
    for (int tk = 0; tk < 16; tk++) {
        int tile_pos = tile_base + (tk << 4);
        int chunk = tile_pos >> 10;
        int local_idx = tile_pos & 1023;
        perm_u32_col[tk] = (chunk << 10) + s_inv_wperm[local_idx];
    }

    float acc = 0.0f;
    int cur_scale_group = -1;
    float cached_scale = 0.0f;

    for (int kt = kt_start; kt < kt_end; kt++) {
        int k_base = kt << 4;
        int row_base = kt * N;
        int sg = k_base / group_size;

        if (sg != cur_scale_group) {
            cur_scale_group = sg;
            int scale_flat = sg * N + n;
            int schunk = scale_flat >> 6;
            int slocal = scale_flat & 63;
            int sperm_pos = (schunk << 6) + s_inv_sperm[slocal];
            unsigned short scale_bits = w2_scales[sperm_pos];
            cached_scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&scale_bits));
        }

        #pragma unroll
        for (int tk = 0; tk < 16; tk++) {
            float w_val = (float)packed[row_base + perm_u32_col[tk]];
            float x = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&s_input[k_base + tk]));
            acc += w_val * cached_scale * x;
        }
    }

    // Warp shuffle reduction
    for (int offset = 8; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
    }

    if (k_slice == 0) {
        __nv_bfloat16 result = __float2bfloat16(acc);
        out[n] = *reinterpret_cast<unsigned short*>(&result);
    }
}

// ── 4-bit PolarQuant KV Cache ──────────────────────────────────────────

// 16-level codebook for quantized angles (normalized components).
// Computed via Lloyd-Max quantization of a Gaussian distribution, scaled by 1/4.
__device__ __constant__ float polar4_codebook[16] = {
    -0.6892f, -0.5241f, -0.4115f, -0.3206f, -0.2412f, -0.1685f, -0.0997f, -0.0330f,
     0.0330f,  0.0997f,  0.1685f,  0.2412f,  0.3206f,  0.4115f,  0.5241f,  0.6892f
};

// Fixed sign flip for SRR (16 elements)
__device__ __constant__ float polar4_signs[16] = {
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// Fast Hadamard Transform for 16 elements (in-place)
__device__ inline void fht16(float* x) {
    float a, b;
    // Stage 1
    #pragma unroll
    for (int i = 0; i < 8; i++) { a = x[i]; b = x[i+8]; x[i] = a + b; x[i+8] = a - b; }
    // Stage 2
    #pragma unroll
    for (int i = 0; i < 4; i++) { a = x[i]; b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
    #pragma unroll
    for (int i = 8; i < 12; i++) { a = x[i]; b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
    // Stage 3
    #pragma unroll
    for (int i = 0; i < 16; i += 4) {
        a = x[i]; b = x[i+2]; x[i] = a + b; x[i+2] = a - b;
        a = x[i+1]; b = x[i+3]; x[i+1] = a + b; x[i+3] = a - b;
    }
    // Stage 4
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        a = x[i]; b = x[i+1]; x[i] = a + b; x[i+1] = a - b;
    }
}

// Find nearest codebook index for a value
__device__ inline int quantize_polar4(float val) {
    int best_idx = 0;
    float min_diff = fabsf(val - polar4_codebook[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float diff = fabsf(val - polar4_codebook[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

// Write K,V to Polar 4-bit cache
extern "C" __global__ void kv_cache_write_polar4(
    unsigned short* __restrict__ k_radius_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ k_angles_cache,
    unsigned char* __restrict__ v_angles_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int position,
    int kv_stride
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = kv_stride / 16;
    if (block_idx >= num_blocks) return;

    int offset = block_idx * 16;
    float k_local[16];
    float v_local[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        k_local[i] = k[offset + i] * polar4_signs[i];
        v_local[i] = v[offset + i] * polar4_signs[i];
    }

    fht16(k_local);
    fht16(v_local);

    // Normalize FHT (scale 1/4)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        k_local[i] *= 0.25f;
        v_local[i] *= 0.25f;
    }

    // Compute Radii
    float k_r = 0.0f, v_r = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        k_r += k_local[i] * k_local[i];
        v_r += v_local[i] * v_local[i];
    }
    k_r = sqrtf(k_r + 1e-12f);
    v_r = sqrtf(v_r + 1e-12f);

    // Store Radii (BF16)
    __nv_bfloat16 k_rb = __float2bfloat16(k_r);
    __nv_bfloat16 v_rb = __float2bfloat16(v_r);
    k_radius_cache[position * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_rb);
    v_radius_cache[position * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);

    // Quantize and Store Angles (4-bit)
    float inv_k_r = 1.0f / k_r;
    float inv_v_r = 1.0f / v_r;
    unsigned char* k_ang = k_angles_cache + (position * num_blocks + block_idx) * 8;
    unsigned char* v_ang = v_angles_cache + (position * num_blocks + block_idx) * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int k0 = quantize_polar4(k_local[i*2] * inv_k_r);
        int k1 = quantize_polar4(k_local[i*2+1] * inv_k_r);
        k_ang[i] = (unsigned char)((k1 << 4) | k0);

        int v0 = quantize_polar4(v_local[i*2] * inv_v_r);
        int v1 = quantize_polar4(v_local[i*2+1] * inv_v_r);
        v_ang[i] = (unsigned char)((v1 << 4) | v0);
    }
}

// Simple GQA attention with Polar 4-bit KV cache
extern "C" __global__ void gqa_attention_polar4(
    float* __restrict__ output,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_radius_cache,
    const unsigned short* __restrict__ v_radius_cache,
    const unsigned char* __restrict__ k_angles_cache,
    const unsigned char* __restrict__ v_angles_cache,
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int max_seq
) {
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = num_threads >> 5;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int num_blocks = (num_kv_heads * head_dim) / 16;
    int block_offset_in_head = (kv_head * head_dim) / 16;

    const float* q_head = q + qh * head_dim;

    extern __shared__ float smem[];
    float* s_q = smem; // head_dim
    float* smem_reduce = smem + head_dim; // 2 * num_warps
    float* s_partial_v = smem_reduce + 2 * num_warps; // num_warps * head_dim

    // Load Q
    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    // Rotate Q in shared memory (Hadamard-16 blocks)
    for (int b = 0; b < head_dim / 16; b++) {
        // Simple sequential rotation per block in shared memory
        if (tid == b) {
            float q_local[16];
            for (int i = 0; i < 16; i++) q_local[i] = s_q[b*16+i] * polar4_signs[i];
            fht16(q_local);
            for (int i = 0; i < 16; i++) s_q[b*16+i] = q_local[i] * 0.25f;
        }
    }
    __syncthreads();

    // Pass 1: max score
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        for (int b = 0; b < head_dim / 16; b++) {
            int block_idx = block_offset_in_head + b;
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = k_angles_cache + (pos * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char p = angs[i];
                score += s_q[b*16 + i*2] * r * polar4_codebook[p & 0xF];
                score += s_q[b*16 + i*2+1] * r * polar4_codebook[p >> 4];
            }
        }
        local_max = fmaxf(local_max, score * sm_scale);
    }
    // Simple block reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if ((tid % 32) == 0) smem_reduce[tid/32] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smem_reduce[0];
        for (int i = 1; i < num_threads/32; i++) m = fmaxf(m, smem_reduce[i]);
        smem_reduce[0] = m;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    // Pass 2: shard positions across warps, then reduce per-warp partial V sums.
    float local_sum_exp = 0.0f;
    float v_acc_lane[8];
    int lane_dims[8];
    int lane_dim_count = 0;
    for (int d = lane_id; d < head_dim; d += 32) {
        lane_dims[lane_dim_count] = d;
        v_acc_lane[lane_dim_count] = 0.0f;
        lane_dim_count++;
    }

    for (int pos = warp_id; pos < seq_len; pos += num_warps) {
        float score_partial = 0.0f;
        for (int b = lane_id; b < head_dim / 16; b += 32) {
            int block_idx = block_offset_in_head + b;
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = k_angles_cache + (pos * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char p = angs[i];
                score_partial += s_q[b*16 + i*2] * r * polar4_codebook[p & 0xF];
                score_partial += s_q[b*16 + i*2+1] * r * polar4_codebook[p >> 4];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
        }
        float w = __expf(__shfl_sync(0xffffffff, score_partial, 0) * sm_scale - global_max);
        local_sum_exp += w;

        for (int idx = 0; idx < lane_dim_count; idx++) {
            int d = lane_dims[idx];
            int block_idx = block_offset_in_head + (d >> 4);
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = v_angles_cache + (pos * num_blocks + block_idx) * 8;
            unsigned char p = angs[(d & 15) >> 1];
            float v_val = ((d & 1) == 0)
                ? r * polar4_codebook[p & 0xF]
                : r * polar4_codebook[p >> 4];
            v_acc_lane[idx] += w * v_val;
        }
    }

    if (lane_id == 0) smem_reduce[warp_id] = local_sum_exp;
    for (int idx = 0; idx < lane_dim_count; idx++) {
        s_partial_v[warp_id * head_dim + lane_dims[idx]] = v_acc_lane[idx];
    }
    __syncthreads();

    if (tid == 0) {
        float gsum = 0.0f;
        for (int w = 0; w < num_warps; w++) gsum += smem_reduce[w];
        smem_reduce[0] = gsum;
    }
    __syncthreads();

    float inv_sum = 1.0f / (smem_reduce[0] + 1e-12f);
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int w = 0; w < num_warps; w++) acc += s_partial_v[w * head_dim + d];
        s_q[d] = acc * inv_sum;
    }
    __syncthreads();

    // Apply inverse SRR to final sum and write back
    // Use one thread per block to un-rotate
    for (int b = 0; b < head_dim / 16; b++) {
        if (tid == b) {
            float o_local[16];
            for (int i = 0; i < 16; i++) o_local[i] = s_q[b*16+i];
            fht16(o_local);
            for (int i = 0; i < 16; i++) {
                output[qh * head_dim + b*16 + i] = o_local[i] * 0.25f * polar4_signs[i];
            }
        }
    }
}

// ── Graph-compatible Polar4 kernels ──────────────────────────────────
// These read position/seq_len from device pointers so values can change
// between CUDA graph replays without recapturing.

extern "C" __global__ void kv_cache_write_polar4_g(
    unsigned short* __restrict__ k_radius_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ k_angles_cache,
    unsigned char* __restrict__ v_angles_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ d_position,   // read from GPU pointer
    int kv_stride
) {
    int position = *d_position;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = kv_stride / 16;
    if (block_idx >= num_blocks) return;

    int offset = block_idx * 16;
    float k_local[16];
    float v_local[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        k_local[i] = k[offset + i] * polar4_signs[i];
        v_local[i] = v[offset + i] * polar4_signs[i];
    }

    fht16(k_local);
    fht16(v_local);

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        k_local[i] *= 0.25f;
        v_local[i] *= 0.25f;
    }

    float k_r = 0.0f, v_r = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        k_r += k_local[i] * k_local[i];
        v_r += v_local[i] * v_local[i];
    }
    k_r = sqrtf(k_r + 1e-12f);
    v_r = sqrtf(v_r + 1e-12f);

    __nv_bfloat16 k_rb = __float2bfloat16(k_r);
    __nv_bfloat16 v_rb = __float2bfloat16(v_r);
    k_radius_cache[position * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_rb);
    v_radius_cache[position * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);

    float inv_k_r = 1.0f / k_r;
    float inv_v_r = 1.0f / v_r;
    unsigned char* k_ang = k_angles_cache + (position * num_blocks + block_idx) * 8;
    unsigned char* v_ang = v_angles_cache + (position * num_blocks + block_idx) * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int k0 = quantize_polar4(k_local[i*2] * inv_k_r);
        int k1 = quantize_polar4(k_local[i*2+1] * inv_k_r);
        k_ang[i] = (unsigned char)((k1 << 4) | k0);

        int v0 = quantize_polar4(v_local[i*2] * inv_v_r);
        int v1 = quantize_polar4(v_local[i*2+1] * inv_v_r);
        v_ang[i] = (unsigned char)((v1 << 4) | v0);
    }
}

extern "C" __global__ void gqa_attention_polar4_g(
    float* __restrict__ output,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_radius_cache,
    const unsigned short* __restrict__ v_radius_cache,
    const unsigned char* __restrict__ k_angles_cache,
    const unsigned char* __restrict__ v_angles_cache,
    float sm_scale,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,    // read from GPU pointer
    int max_seq
) {
    int seq_len = *d_seq_len;
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = num_threads >> 5;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int num_blocks = (num_kv_heads * head_dim) / 16;
    int block_offset_in_head = (kv_head * head_dim) / 16;

    const float* q_head = q + qh * head_dim;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* smem_reduce = smem + head_dim;
    float* s_partial_v = smem_reduce + 2 * num_warps;

    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    for (int b = 0; b < head_dim / 16; b++) {
        if (tid == b) {
            float q_local[16];
            for (int i = 0; i < 16; i++) q_local[i] = s_q[b*16+i] * polar4_signs[i];
            fht16(q_local);
            for (int i = 0; i < 16; i++) s_q[b*16+i] = q_local[i] * 0.25f;
        }
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += num_threads) {
        float score = 0.0f;
        for (int b = 0; b < head_dim / 16; b++) {
            int block_idx = block_offset_in_head + b;
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = k_angles_cache + (pos * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char p = angs[i];
                score += s_q[b*16 + i*2] * r * polar4_codebook[p & 0xF];
                score += s_q[b*16 + i*2+1] * r * polar4_codebook[p >> 4];
            }
        }
        local_max = fmaxf(local_max, score * sm_scale);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if ((tid % 32) == 0) smem_reduce[tid/32] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smem_reduce[0];
        for (int i = 1; i < num_threads/32; i++) m = fmaxf(m, smem_reduce[i]);
        smem_reduce[0] = m;
    }
    __syncthreads();
    float global_max = smem_reduce[0];

    float local_sum_exp = 0.0f;
    float v_acc_lane[8];
    int lane_dims[8];
    int lane_dim_count = 0;
    for (int d = lane_id; d < head_dim; d += 32) {
        lane_dims[lane_dim_count] = d;
        v_acc_lane[lane_dim_count] = 0.0f;
        lane_dim_count++;
    }

    for (int pos = warp_id; pos < seq_len; pos += num_warps) {
        float score_partial = 0.0f;
        for (int b = lane_id; b < head_dim / 16; b += 32) {
            int block_idx = block_offset_in_head + b;
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = k_angles_cache + (pos * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char p = angs[i];
                score_partial += s_q[b*16 + i*2] * r * polar4_codebook[p & 0xF];
                score_partial += s_q[b*16 + i*2+1] * r * polar4_codebook[p >> 4];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
        }
        float w = __expf(__shfl_sync(0xffffffff, score_partial, 0) * sm_scale - global_max);
        local_sum_exp += w;

        for (int idx = 0; idx < lane_dim_count; idx++) {
            int d = lane_dims[idx];
            int block_idx = block_offset_in_head + (d >> 4);
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = v_angles_cache + (pos * num_blocks + block_idx) * 8;
            unsigned char p = angs[(d & 15) >> 1];
            float v_val = ((d & 1) == 0)
                ? r * polar4_codebook[p & 0xF]
                : r * polar4_codebook[p >> 4];
            v_acc_lane[idx] += w * v_val;
        }
    }

    if (lane_id == 0) smem_reduce[warp_id] = local_sum_exp;
    for (int idx = 0; idx < lane_dim_count; idx++) {
        s_partial_v[warp_id * head_dim + lane_dims[idx]] = v_acc_lane[idx];
    }
    __syncthreads();

    if (tid == 0) {
        float gsum = 0.0f;
        for (int w = 0; w < num_warps; w++) gsum += smem_reduce[w];
        smem_reduce[0] = gsum;
    }
    __syncthreads();

    float inv_sum = 1.0f / (smem_reduce[0] + 1e-12f);
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int w = 0; w < num_warps; w++) acc += s_partial_v[w * head_dim + d];
        s_q[d] = acc * inv_sum;
    }
    __syncthreads();

    for (int b = 0; b < head_dim / 16; b++) {
        if (tid == b) {
            float o_local[16];
            for (int i = 0; i < 16; i++) o_local[i] = s_q[b*16+i];
            fht16(o_local);
            for (int i = 0; i < 16; i++) {
                output[qh * head_dim + b*16 + i] = o_local[i] * 0.25f * polar4_signs[i];
            }
        }
    }
}

// ── Tiled Polar4 GQA attention (graph-compatible) ─────────────────────
//
// Splits KV positions into tiles across grid.y, parallelising across SMs
// just like gqa_attention_tiled_g does for FP8.  Each block computes a
// partial softmax-weighted V sum (in the rotated Hadamard domain) for one
// Q head × one tile.  A separate reduce kernel merges tiles and applies
// the inverse Hadamard transform.
//
// Grid:  (num_q_heads, max_tiles, 1)   Block: (256, 1, 1)
// Shared: (head_dim + tile_size + 32) * 4 bytes

extern "C" __global__ void gqa_attention_polar4_tiled_g(
    float* __restrict__ partial_o,     // [num_q_heads, max_tiles, head_dim]
    float* __restrict__ partial_lse,   // [num_q_heads, max_tiles, 2] (max, sum_exp)
    const float* __restrict__ q,       // [num_q_heads * head_dim]
    const unsigned short* __restrict__ k_radius_cache,
    const unsigned short* __restrict__ v_radius_cache,
    const unsigned char* __restrict__ k_angles_cache,
    const unsigned char* __restrict__ v_angles_cache,
    float sm_scale,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,
    int tile_size
) {
    // num_q_heads and max_tiles derived from grid dimensions
    int num_q_heads = gridDim.x;
    int max_tiles = gridDim.y;

    int seq_len = *d_seq_len;
    int qh = blockIdx.x;
    int tile_idx = (int)blockIdx.y;
    int num_tiles = (seq_len + tile_size - 1) / tile_size;
    if (tile_idx >= num_tiles) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = qh / heads_per_kv;
    int num_blocks = (num_kv_heads * head_dim) / 16;
    int block_offset_in_head = (kv_head * head_dim) / 16;
    int blocks_per_head = head_dim / 16;

    int tile_start = tile_idx * tile_size;
    int tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;
    int tile_len = tile_end - tile_start;

    // Shared memory layout: s_q | smem_scores | smem_reduce
    extern __shared__ float smem[];
    float* s_q = smem;
    float* smem_scores = smem + head_dim;
    float* smem_reduce = smem_scores + tile_size;

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    // Preload Q into shared memory
    const float* q_head = q + qh * head_dim;
    for (int i = tid; i < head_dim; i += num_threads) {
        s_q[i] = q_head[i];
    }
    __syncthreads();

    // Apply Hadamard transform to Q in shared memory (all blocks in parallel)
    if (tid < blocks_per_head) {
        int b = tid;
        float q_local[16];
        for (int i = 0; i < 16; i++) q_local[i] = s_q[b*16+i] * polar4_signs[i];
        fht16(q_local);
        for (int i = 0; i < 16; i++) s_q[b*16+i] = q_local[i] * 0.25f;
    }
    __syncthreads();

    // Step 1: Q·K dot products for this tile
    for (int i = tid; i < tile_len; i += num_threads) {
        int pos = tile_start + i;
        float score = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            int block_idx = block_offset_in_head + b;
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_radius_cache[pos * num_blocks + block_idx]));
            const unsigned char* angs = k_angles_cache + (pos * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned char p = angs[j];
                score += s_q[b*16 + j*2]   * r * polar4_codebook[p & 0xF];
                score += s_q[b*16 + j*2+1] * r * polar4_codebook[p >> 4];
            }
        }
        smem_scores[i] = score * sm_scale;
    }
    __syncthreads();

    // Step 2: Tile max (parallel reduction)
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += num_threads) {
        local_max = fmaxf(local_max, smem_scores[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float gmax = smem_reduce[0];
        for (int w = 1; w < num_warps; w++) gmax = fmaxf(gmax, smem_reduce[w]);
        smem_reduce[0] = gmax;
    }
    __syncthreads();
    float tile_max = smem_reduce[0];

    // Step 3: exp(score - max) in-place, compute sum
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += num_threads) {
        float w = __expf(smem_scores[i] - tile_max);
        smem_scores[i] = w;
        local_sum += w;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float gsum = 0.0f;
        for (int w = 0; w < num_warps; w++) gsum += smem_reduce[w];
        smem_reduce[0] = gsum;
    }
    __syncthreads();
    float tile_sum = smem_reduce[0];

    // Step 4: Unnormalised weighted V sum (in rotated Hadamard domain)
    float* out_partial = partial_o + (qh * max_tiles + tile_idx) * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        int b = d >> 4;  // d / 16
        int block_idx = block_offset_in_head + b;
        int sub_idx = d & 15;
        int byte_idx = sub_idx >> 1;
        bool is_high = (sub_idx & 1) != 0;
        float acc = 0.0f;
        for (int i = 0; i < tile_len; i++) {
            int pos = tile_start + i;
            float r = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &v_radius_cache[pos * num_blocks + block_idx]));
            unsigned char p = v_angles_cache[(pos * num_blocks + block_idx) * 8 + byte_idx];
            float v_val = is_high
                ? r * polar4_codebook[p >> 4]
                : r * polar4_codebook[p & 0xF];
            acc += smem_scores[i] * v_val;
        }
        out_partial[d] = acc;
    }

    // Step 5: Store tile statistics
    if (tid == 0) {
        float* lse = partial_lse + (qh * max_tiles + tile_idx) * 2;
        lse[0] = tile_max;
        lse[1] = tile_sum;
    }
}

// ── Polar4 tiled attention reduce (graph-compatible) ──────────────────
//
// Merges per-tile partial outputs using log-sum-exp correction, then
// applies the inverse Hadamard transform + sign flip to produce the
// final output in the original (non-rotated) domain.
//
// Grid: (num_q_heads, 1, 1)   Block: (256, 1, 1)
// Shared: (max_tiles + head_dim) * 4 bytes

extern "C" __global__ void gqa_attention_polar4_reduce_g(
    float* __restrict__ output,            // [num_q_heads * head_dim]
    const float* __restrict__ partial_o,   // [num_q_heads, max_tiles, head_dim]
    const float* __restrict__ partial_lse, // [num_q_heads, max_tiles, 2]
    int num_q_heads,
    int head_dim,
    const int* __restrict__ d_seq_len,
    int tile_size,
    int max_tiles
) {
    int seq_len = *d_seq_len;
    int num_tiles = (seq_len + tile_size - 1) / tile_size;
    int qh = blockIdx.x;
    if (qh >= num_q_heads) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory: tile_weights[max_tiles] | s_merged[head_dim]
    extern __shared__ float smem[];
    float* tile_weights = smem;
    float* s_merged = smem + max_tiles;

    const float* lse = partial_lse + qh * max_tiles * 2;

    // Thread 0 computes normalised correction weights
    if (tid == 0) {
        float global_max = -1e30f;
        for (int t = 0; t < num_tiles; t++) {
            global_max = fmaxf(global_max, lse[t * 2]);
        }
        float global_sum = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            float correction = __expf(lse[t * 2] - global_max);
            global_sum += correction * lse[t * 2 + 1];
            tile_weights[t] = correction;
        }
        float inv_sum = 1.0f / (global_sum + 1e-12f);
        for (int t = 0; t < num_tiles; t++) {
            tile_weights[t] *= inv_sum;
        }
    }
    __syncthreads();

    // All threads merge tile partials into rotated-domain output
    const float* po = partial_o + qh * max_tiles * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            acc += tile_weights[t] * po[t * head_dim + d];
        }
        s_merged[d] = acc;
    }
    __syncthreads();

    // Inverse Hadamard transform + sign flip (all blocks in parallel)
    int blocks_per_head = head_dim / 16;
    if (tid < blocks_per_head) {
        int b = tid;
        float o_local[16];
        for (int i = 0; i < 16; i++) o_local[i] = s_merged[b*16+i];
        fht16(o_local);
        for (int i = 0; i < 16; i++) {
            output[qh * head_dim + b*16 + i] = o_local[i] * 0.25f * polar4_signs[i];
        }
    }
}

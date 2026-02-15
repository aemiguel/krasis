# Krasis Research & Performance Analysis

**Hardware**: AMD EPYC 7742 (64 cores, AVX2, 995 GB DDR4) + 3x NVIDIA RTX 2000 Ada (16 GB each, SM89)

## Benchmarks

### DeepSeek-V2-Lite (27 layers, 64 experts, PP=1, 1 GPU)

**Config**: INT4 GPU (Marlin) + INT4 CPU (optimized), BF16 attention, expert_divisor=1 (persistent)

| Metric | Value |
|--------|:---:|
| GPU prefill (512 tok) | 3,294 tok/s |
| GPU prefill (2K tok) | 4,074 tok/s |
| GPU prefill (5K tok) | 2,494 tok/s |
| GPU prefill (10K tok) | 2,409 tok/s |
| CPU decode | 5.8 tok/s (172ms avg, P50=172.9, P90=175.6) |
| Model load (cached) | 14.2s |
| GPU VRAM | 10,746 MB (2,924 weights + 7,654 persistent experts) |
| KV capacity | 93K tokens |

Decode is flat across context lengths: 5.9 tok/s at 512 tokens, 5.8 tok/s at 8K tokens.

### GGUF Input (V2-Lite, Q4_K_M)

| Source | Prefill (5K tok) | Decode | Cache Load | Cache Size |
|--------|:-:|:-:|:-:|:-:|
| BF16 safetensors → INT4 CPU | 2,494 tok/s | **5.8 tok/s** | 6.0s | 7.7 GB |
| **GGUF Q4_K_M → AVX2** | 2,388 tok/s | **4.77 tok/s** | 5.9s | 10.1 GB |
| GGUF-native (raw blocks) | 156 tok/s | 1.83 tok/s | 11.5s | N/A |

The GGUF→AVX2 pipeline: dequant GGUF to f32, requant to transposed AVX2 format, disk-cache. Mixed precision: gate/up=INT4 (from Q4_K), down=INT8 (from Q5_0/Q8_0). The 4.77 vs 5.8 decode gap is from INT8 down projections (2x more bytes).

### Kimi K2.5 (61 layers, 384 experts, PP=2, 2 GPUs)

| Config | Decode | GPU0 VRAM | GPU1 VRAM |
|--------|:---:|:-:|:-:|
| BF16 weights, BF16 KV | 1.55-1.87 tok/s | 12,063 MB | 11,105 MB |
| INT8 weights, BF16 KV | 1.28-1.41 tok/s | 7,654 MB | 6,044 MB |
| INT8 weights, FP8 KV | 1.21-1.28 tok/s | 7,654+4,032 KV | 6,044+4,839 KV |
| KTransformers PP=2 (baseline) | 4.0 tok/s | ~7.6 GB | ~7.6 GB |

Retired: 0.55 tok/s with Marlin-only CPU decode was unacceptable. Led to the dual-format architecture.

### Qwen3-235B-A22B (94 layers, 128 experts, PP=3)

| System | Decode | GPU Prefill |
|--------|:---:|:---:|
| KTransformers + expert pinning | 4.21 tok/s | N/A (CPU only) |
| KTransformers baseline | 3.88 tok/s | N/A |

---

## GPU Prefill Analysis

### Expert DMA is the Bottleneck

The fundamental bottleneck was expert weight transfers, not compute. Per forward call on V2-Lite: 7.3 GB of expert DMA vs 8 MB of prompt data (900:1 ratio). This is why 512 tokens and 2K tokens both took ~11s — the same 7.3 GB transfer dominated.

### Persistent Expert Buffers

Pre-load all experts into VRAM once. During forward: zero DMA.

| Tokens | Chunked (DMA/layer) | Persistent (zero DMA) | Speedup |
|-------:|:---:|:---:|:---:|
| 512 | 11.17s (45.8 tok/s) | 0.155s (3,294 tok/s) | 72x |
| 2,048 | 11.01s (184 tok/s) | 0.498s (4,074 tok/s) | 22x |
| 10K | 57.26s (173 tok/s) | 4.110s (2,409 tok/s) | 14x |

Speedup varies because we removed a fixed cost — biggest relative improvement when compute is smallest (short prompts).

### Layer-Grouped Prefill

For models where experts don't all fit in VRAM, layer-grouped mode cycles expert groups through VRAM:

```
Group 1: DMA layers 0-6 → process ALL tokens → free
Group 2: DMA layers 7-12 → process ALL tokens → free
...
```

Key: reversed loop nesting. Instead of "for each token, for each layer" (constant DMA), it's "for each group, DMA once, then all tokens." DMA cost is fixed regardless of prompt length.

### Comparison: What if Experts Don't Fit?

When only 25% of expert data fits in VRAM (e.g., Qwen3-235B with 160 experts × 62 layers):

| Approach | Prefill (5K tok) | Why |
|----------|:-:|-----|
| llama.cpp (layer split) | ~30 tok/s | 75% of layers on CPU |
| KTransformers | ~25 tok/s | All experts always on CPU |
| **Krasis layer-grouped** | **~400-600 tok/s** | All compute on GPU, 4 DMA round-trips |
| *Krasis persistent (100% fits)* | *2,388 tok/s* | *Zero DMA* |

Krasis with only 25% VRAM is still 15-20x faster than alternatives at prefill.

---

## CPU Weight Format Analysis

### Why Marlin is Bad for CPU

Marlin format uses tile permutation (16x16 tiles with bit interleaving) designed for GPU warp parallelism. On CPU at M=1 decode:
- Random access pattern defeats hardware prefetcher
- MarlinTileMap indirection adds dependent load in hot loop
- Stride-8 packing doesn't align with CPU SIMD patterns
- Cache line waste from scattered tile accesses

Result: **0.55 tok/s** (Marlin INT4) vs **1.55 tok/s** (BF16 native) on Kimi K2.5.

### Why Sequential Row-Major is Optimal

The CPU-optimized format uses INT4 packed `[K/8, 2*N]` layout:
- Data contiguous along output dimension N — sequential cache line reads
- `_mm256_madd_epi16` kernel: 3-cycle latency, 1 CPI on Zen 2, 16 MACs/instruction
- Combined w13 (gate+up): one matrix instead of two, 2 matmuls per expert instead of 3
- Hardware prefetcher friendly — `_mm_prefetch` with `_MM_HINT_NTA` brings next expert to L3

### Why Not FP Formats?

- **FP8 (E4M3/E5M2)**: AVX2 has zero native FP8 support. Upcast to FP32 + `_mm256_fmadd_ps` (5-cycle) is slower than INT8 `_mm256_madd_epi16` (3-cycle).
- **FP4**: No hardware support. Custom unpack + FP32 upcast required.
- **INT4**: Bandwidth-optimal (half bytes of INT8), feeds directly into fast integer MAD pipeline.
- **INT8**: Simpler (no nibble extraction), 2x bandwidth. Best when precision matters more than speed.

---

## Decode Timing Analysis

### M=1 Decode is CPU-Bound

Per-token timing (V2-Lite, estimated per-layer):
- MLA attention (GPU): ~2ms
- Norms (GPU): ~0.5ms
- Routing (GPU): ~0.2ms
- Shared expert (GPU): ~0.5ms
- Routed experts (CPU): ~4ms (6 experts × INT4 AVX2)
- GPU-CPU sync: ~0.1ms
- **Total per layer**: ~6.3ms × 26 MoE layers + ~3ms dense = ~167ms (measured: 172ms)

CPU expert computation dominates. GPU work finishes while CPU is still running — GPU-CPU overlap saves nothing at M=1.

### Marlin Kernel Launch Overhead

At M=1, `fused_marlin_moe` has a ~1.5ms floor regardless of expert count or precision. Pure kernel launch overhead. CUDA graphs could help but are incompatible with dynamic expert routing.

### Expert Pinning

- Qwen3-235B with 1444 pinned experts: 4.21 tok/s (+8.5% vs baseline)
- Kimi K2.5: pinning hurts — experts 2.3x larger (22.7 MB vs 9.7 MB), fewer fit

---

## Memory Architecture

### DDR4 Bandwidth is the Ceiling

- Theoretical: ~76 GB/s (4 channels × 3200 MT/s)
- Practical: ~50-55 GB/s
- 8 experts × ~7.2 MB each = ~57.6 MB/token → ~1ms at 50 GB/s (matches measured timings)

### NUMA Effects

- Single socket (EPYC 7742): no cross-socket penalty, but CCX boundaries exist
- NUMA-aware placement via `mbind(MPOL_MF_MOVE)` ensures pages are local
- Thread pinning prevents cross-CCX cache coherency traffic

### Hyperthreading is Harmful

128 threads (HT on): 1.9 tok/s. 48 threads (HT off): 5.9 tok/s. SMT siblings compete for execution units and cache — pure overhead for bandwidth-bound workloads.

---

## FP8 KV Cache

- Store as FP8 E4M3, upcast to BF16 before FlashInfer kernel
- 2x VRAM savings: 576 bytes/token/layer vs 1152
- Negligible precision loss: cosine similarity 1.0000 (FP8 vs BF16)
- 4x more context capacity
- Slight decode penalty: 1.21-1.28 tok/s vs 1.28-1.41 tok/s (INT8 weights, Kimi K2.5)

---

## INT8 Non-Expert Weights

- Per-channel symmetric: `scale = max(abs(weight)) / 127.0`
- 4-5 GB VRAM savings per GPU (12 GB → 7.6 GB on Kimi K2.5)
- Slight M=1 decode slowdown: `torch._int_mm` overhead > bandwidth savings
- Quality identical to BF16

---

## Build Optimization: Fused Transpose

First-run Marlin cache build for Kimi K2.5 was 3+ hours. Bottleneck: separate transpose step (3.1s per expert repack). Fixed by fusing transpose into tile permutation — read source data with swapped indices. Result: 60 layers in 25 min.

---

## Codebase Comparison

| Component | Lines |
|-----------|:---:|
| Rust (CPU MoE engine) | ~5,400 |
| Python (standalone server) | ~4,200 |
| **Total** | **~9,600** |

For comparison: SGLang ~200K lines, KTransformers ~50K lines. Krasis replaces both with under 10K lines for hybrid MoE inference.

---

## Strategy Analysis

Auto-optimiser results comparing prefill/decode strategies across models. Each strategy tested with 3 runs (10K prefill, 64 decode tokens).

### DeepSeek-V2-Lite — 1 GPU

Winner: persistent (1,757 tok/s) + pure_cpu (6.98 tok/s)

| Prefill | tok/s | Decode | tok/s |
|---------|-------|--------|-------|
| persistent | 1,757 | pure_cpu | 6.98 |
| layer_grouped_2 | 1,746 | hcs_hybrid | 4.88 |
| chunked | 792 | compact | 3.31 |
| active_only | 303 | lru | 3.31 |

### Qwen3-Coder-Next — 2 GPUs

Winner: layer_grouped_4 (619 tok/s) + pure_cpu (7.26 tok/s)

| Prefill | tok/s | Decode | tok/s |
|---------|-------|--------|-------|
| layer_grouped_4 | 619 | pure_cpu | 7.26 |
| hcs_prefill | 603 | hcs_hybrid | 7.02 |
| chunked | 155 | compact | 4.99 |
| persistent | OOM | lru | 4.90 |

### Qwen3-235B-A22B — 2 GPUs

Winner: hcs_prefill (211 tok/s) + pure_cpu (1.81 tok/s)

| Prefill | tok/s | Decode | tok/s |
|---------|-------|--------|-------|
| hcs_prefill | 211 | pure_cpu | 1.81 |
| chunked | 56 | hcs_hybrid | 1.77 |
| active_only | 47 | compact | 0.23 |
| persistent | OOM | lru | 0.23 |

**Universal finding**: pure_cpu decode wins on all models — GPU Marlin kernel launch overhead at M=1 exceeds any benefit from having experts on GPU.

---

## What Would Help Most

1. **DDR5 / Zen 4**: 2x memory bandwidth → ~8 tok/s decode
2. **Continuous batching**: Batch N decode tokens → amortize GPU overhead
3. **Speculative decoding**: Draft model on spare GPU → 2-3x effective tok/s
4. **CUDA graphs**: Eliminate kernel launch overhead in decode path (if compatible with dynamic routing)

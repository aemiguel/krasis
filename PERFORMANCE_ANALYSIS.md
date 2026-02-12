# Krasis Performance Analysis — DeepSeek-V2-Lite

**Date:** 2026-02-12
**Model:** DeepSeek-V2-Lite (27 layers, 64 experts, top-6)
**Hardware:** 1x RTX 2000 Ada (16GB) + AMD EPYC 7742 (64 cores, AVX2)
**Config:** INT4 dual-format (GPU Marlin + CPU optimized), BF16 KV, 16 threads

## GPU Prefill Performance

### Throughput Scaling

| Tokens | Wall Time | tok/s | ms/tok |
|:---:|:---:|:---:|:---:|
| 100 | 11.13s | 9.0 | 111.3 |
| 500 | 10.61s | 47.1 | 21.2 |
| 1,000 | 10.69s | 93.6 | 10.7 |
| 2,000 | 10.96s | 182.5 | 5.5 |
| 5,000 | 11.97s | 417.9 | 2.4 |
| 9,903 | 14.28s | 693.3 | 1.4 |

### Analysis

**Fixed overhead: ~10.5s per forward pass.** This is the dominant cost at all prompt lengths.

The overhead is per-layer DMA + kernel setup, repeated 26 times (one per MoE layer):
- Each layer: DMA copy 64 experts (4.5 MB each = 288 MB) from RAM → GPU
- Plus: attention forward, RMSnorms, routing gate, LM head
- Total DMA per forward: ~7.3 GB (26 layers × 288 MB)

**Marginal GPU compute: ~3,000 tok/s.** Subtracting the fixed overhead:
- 100→9,903 tokens adds only 3.15s of wall time for 9,803 extra tokens
- Marginal throughput: **~3,100 tok/s** (pure Marlin MoE compute)
- This is the actual fused_marlin_moe kernel performance

**Implication:** For large prompts (>5K tokens), throughput approaches 700+ tok/s.
For short prompts (<500 tokens), the fixed overhead dominates and throughput drops.

### Optimization Opportunities

1. **Layer-level DMA pipelining** — Overlap DMA copy of layer N+1 with GPU compute of layer N.
   Could reduce the ~10.5s overhead significantly (currently sequential).
2. **Expert subset DMA** — Only copy the experts that will actually be activated for this prompt.
   At top-6 of 64, only ~6/64 = 9.4% of experts are active per token. But expert selection varies per token, so the union over all tokens in the prompt approaches 100% for large prompts.
3. **Persistent GPU expert cache** — Pin hottest experts in GPU VRAM permanently (like ExpertPinningManager in SGLang). Eliminates DMA for frequently-used experts.

## CPU Decode Performance

### Latency by Context Length

| Context | Avg (ms) | p50 (ms) | Min (ms) | Max (ms) | tok/s |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 177.0 | 175.3 | 162.0 | 227.8 | 5.6 |
| 1,000 | 177.0 | 175.8 | 142.8 | 192.1 | 5.7 |
| 10,000 | 202.2 | 200.8 | 152.2 | 224.8 | 4.9 |

### Analysis

**CPU decode is consistent at ~177ms for contexts up to 1K tokens**, rising to ~202ms at 10K.

The per-token decode consists of:
1. **GPU attention** — MLA absorbed attention via FlashInfer. Scales with context length (KV cache lookback). At 10K context, this accounts for the 25ms increase.
2. **GPU norms + routing** — RMSnorm, gate softmax, top-k selection. Fixed cost ~2ms.
3. **CPU MoE experts** — Krasis Rust INT4 kernel. 6 active experts × (gate+up matmul + SiLU + down matmul). Fixed cost per token.
4. **GPU→CPU→GPU transfer** — Hidden state transfer for MoE dispatch. Fixed cost ~1ms.

**Bottleneck:** CPU expert MoE computation. At M=1, the GPU is idle for most of the decode step waiting for CPU experts. The Marlin MoE kernel has a ~1.5ms launch floor (launch-overhead-bound at M=1), so GPU prefill has no benefit at batch size 1.

### Context Length Impact

The 25ms increase from 1K→10K context is entirely GPU attention (FlashInfer MLA).
At 10K tokens with MLA, the KV cache is 10K × 576 bytes/token/layer = 5.6 MB per layer.
FlashInfer must attend over this for each decode step.

## Time-to-First-Token (TTFT)

For a typical IDE query (~10K tokens):
- **GPU prefill TTFT: 14.3s** (model loading not included — 14.2s from cache)
- **First model load from cache: 14.2s** (GPU 0.9s + CPU 9.8s + prefill init 3.3s)
- **Total cold start: 28.5s** (load + first prefill)
- **Warm TTFT: 14.3s** (subsequent queries, model already loaded)

For comparison, CPU-only prefill at 17.7 tok/s would take **565 seconds** for 10K tokens.
**GPU prefill provides 37x speedup for TTFT on 10K prompts.**

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Peak GPU prefill throughput | **693 tok/s** (10K tokens) |
| Marginal GPU compute rate | **~3,100 tok/s** |
| GPU prefill fixed overhead | ~10.5s (DMA + per-layer setup) |
| CPU decode (short context) | **5.6 tok/s** (177ms) |
| CPU decode (10K context) | **4.9 tok/s** (202ms) |
| TTFT for 10K prompt | **14.3s** (warm) |
| Model load from cache | **14.2s** |
| Dual-format cache size | 14.2 GB (7.2 GB GPU Marlin + 7.0 GB CPU INT4) |

## Comparison: Krasis vs KTransformers (Qwen3-235B baseline)

| Metric | Krasis (V2-Lite) | KTransformers (Qwen3-235B) |
|--------|:-:|:-:|
| GPU prefill | 693 tok/s | 57-443 tok/s (INT4 Marlin) |
| CPU decode | 5.1-5.6 tok/s | 4.21 tok/s |
| Architecture | Standalone | SGLang + KTransformers |
| Weight format | Dual (GPU Marlin + CPU INT4) | Single (GGUF for CPU) |

Note: V2-Lite is much smaller (16B params, 64 experts) vs Qwen3-235B (235B, 128 experts),
so direct comparison is not meaningful. The value is validating the dual-format architecture
before scaling to larger models.

# Krasis Performance Analysis — DeepSeek-V2-Lite

**Date**: 2026-02-12
**Hardware**: AMD EPYC 7742 (64c, AVX2) + 1x RTX 2000 Ada (16 GB)
**Config**: INT4 GPU (Marlin) + INT4 CPU (optimized), BF16 attention, PP=1

## GGUF → AVX2 CPU Cache (2026-02-12)

| Source | Prefill (5K tok) | Decode | Load Time | Cache Size |
|--------|:-:|:-:|:-:|:-:|
| BF16 safetensors → INT4 CPU | 2,494 tok/s | **5.8 tok/s** | 6.0s (cached) | 7.7 GB |
| **GGUF Q4_K_M → AVX2 (new)** | 2,388 tok/s | **4.77 tok/s** | 5.9s (cached) | 10.1 GB |
| GGUF-native (raw blocks) | 156 tok/s | 1.83 tok/s | 11.5s | N/A (no cache) |

Prefill uses GPU Marlin persistent mode (expert_divisor=1, all experts in VRAM). Prefill speed is the same regardless of CPU source since it uses the Marlin cache from safetensors.

The GGUF→AVX2 pipeline dequants GGUF to f32, requants to our transposed AVX2 format, and disk-caches. Mixed precision: gate/up=INT4 (from Q4_K), down=INT8 (from Q5_0/Q8_0). The 4.77 vs 5.8 tok/s decode gap is from INT8 down projection (2× more bytes than INT4 from safetensors).

## Summary

| Metric | Chunked (before) | Persistent (after) |
|--------|:-:|:-:|
| GPU prefill (512 tokens) | 45.8 tok/s | **3,294 tok/s** (72x) |
| GPU prefill (2K tokens) | 184 tok/s | **4,074 tok/s** (22x) |
| GPU prefill (10K tokens) | 173 tok/s | **2,409 tok/s** (14x) |
| CPU decode | 5.9 tok/s | 5.8 tok/s (unchanged) |
| VRAM usage | 2,924 + 286 MB buffer | 2,924 + 7,654 MB persistent |
| KV cache capacity | 212K tokens | 93K tokens |

## The Root Cause: Expert DMA Dominates Everything

The fundamental bottleneck was **expert weight transfers**, not compute, not attention, not prompt size. The numbers prove this conclusively:

**Per forward call, the data transferred was:**

| What | Size | Frequency | Total |
|------|------|-----------|-------|
| Expert weights (26 MoE layers × 64 experts × 4.5 MB) | **7.3 GB** | Every forward call | 7.3 GB |
| Hidden states (2048 tokens × 2048 dim × 2 bytes BF16) | **8 MB** | Once at start | 8 MB |

That's a **900:1 ratio**. The expert DMA was three orders of magnitude larger than the prompt data. This is why:

- 512 tokens took 11.17s — because 7.3 GB of expert DMA dominated
- 2048 tokens took 11.01s — because the same 7.3 GB of expert DMA still dominated
- Token count was irrelevant to wall time

The Marlin MoE kernel itself, once the weights were on GPU, was fast. We were spending ~10.5s moving data and ~0.5s computing.

## The Fix: Persistent Expert Buffers (`expert_divisor=1`)

Pre-load all 26 layers × 64 experts = 7,654 MB of Marlin-format weights into GPU VRAM once at startup. During forward: zero DMA, just index into the pre-loaded buffers.

### Before vs After

| Tokens | Chunked (DMA/layer) | Persistent (zero DMA) | Speedup |
|-------:|--------------------:|----------------------:|--------:|
| 512 | 11.17s (45.8 tok/s) | 0.155s (3,294 tok/s) | **72x** |
| 1,024 | 10.71s (95.7 tok/s) | 0.250s (4,090 tok/s) | **43x** |
| 2,031 | 11.01s (184 tok/s) | 0.498s (4,074 tok/s) | **22x** |
| 4,047 | 22.12s (183 tok/s) | 1.167s (3,468 tok/s) | **19x** |
| 9,903 | 57.26s (173 tok/s) | 4.110s (2,409 tok/s) | **14x** |

### Why Speedup Varies With Token Count

At small prompts (512 tokens), nearly all the old wall time was DMA — so eliminating it gives 72x. At larger prompts the actual compute (Marlin kernel + attention) becomes a larger fraction of the total, so the speedup converges to ~14x. This is the expected behavior: we removed a fixed cost, so relative improvement is greatest when the fixed cost was dominant.

### 10K Prefill Breakdown (Persistent Mode)

With DMA eliminated, the per-chunk time now scales with token count as expected — attention computation grows with sequence length:

```
Chunk 0: pos    0..2048  (2048 tok) = 0.505s = 4,057 tok/s
Chunk 1: pos 2048..4096  (2048 tok) = 0.682s = 3,005 tok/s
Chunk 2: pos 4096..6144  (2048 tok) = 0.862s = 2,375 tok/s
Chunk 3: pos 6144..8192  (2048 tok) = 1.044s = 1,962 tok/s
Chunk 4: pos 8192..9903  (1711 tok) = 1.016s = 1,684 tok/s
TOTAL: 9,903 tokens in 4.110s = 2,409 tok/s
```

Each subsequent chunk is slower because attention must attend over a longer KV cache — this is real compute scaling, not a bottleneck we can remove.

### VRAM Trade-off

| Mode | Expert VRAM | KV Cache | Total | Max Context |
|------|------------|----------|-------|-------------|
| Chunked (divisor=0) | 286 MB (buffer) | 6,596 MB | ~10.0 GB | 212K tokens |
| Persistent (divisor=1) | 7,654 MB (all layers) | 2,891 MB | ~13.5 GB | 93K tokens |

Persistent mode trades KV cache capacity for prefill speed. 93K tokens is still more than sufficient for IDE use cases (typically 10-20K).

## GPU Prefill Modes

The `expert_divisor` parameter controls the trade-off:

- **`divisor=0`** (chunked): 286 MB VRAM, DMA 7.3 GB every forward call. Baseline.
- **`divisor=1`** (persistent): 7,654 MB VRAM, zero DMA. 14-72x faster prefill.
- **`divisor=2`** (selective): 3,705 MB VRAM, DMA only active experts (~30-40 of 64). Middle ground.
- OOM fallback: persistent → selective(2) automatically if VRAM is insufficient.

## CPU Decode Analysis

### Timing Distribution

100 tokens decoded after 2K prefill (persistent mode):

```
Avg:  172.4ms (5.8 tok/s)
P50:  172.9ms
P90:  175.6ms
P99:  182.3ms
Min:  146.0ms
Max:  182.3ms
```

Decode is completely unaffected by the prefill mode — expert weights are on CPU for decode regardless.

### Per-Token Breakdown (estimated from ~170ms total)

Each decode step processes 1 token through 27 layers:
- **MLA attention** (GPU): ~2ms (embedding + Q/K projection + FlashInfer paged decode + output projection)
- **Norms** (GPU): ~0.5ms (2 fused_add_rmsnorm per layer × 27)
- **Routing** (GPU): ~0.2ms per MoE layer (gate matmul + topk)
- **Shared expert** (GPU): ~0.5ms per MoE layer (gate+up → SiLU → down)
- **Routed experts** (CPU): ~4ms per MoE layer (6 experts × INT4 AVX2 matmul via Krasis)
- **GPU-CPU sync**: ~0.1ms per MoE layer

Estimated per-layer: ~6.3ms × 26 MoE layers + ~3ms dense layer 0 = ~167ms. Close to measured 172ms.

### Context Length Impact

| Context | Avg | Difference |
|---------|-----|-----------|
| 512 | 170.7ms | baseline |
| 2,031 | 174.8ms | +2.4% |
| 8,079 | 166.3ms | -2.6% |

MLA attention's KV cache lookup is O(context × kv_lora_rank), not O(context × heads × head_dim), so context scaling is minimal. This is a major advantage of MLA over standard GQA.

## Real-World Impact

For an IDE use case (e.g., OpenCode with 10K token prompts):

| Metric | Chunked | Persistent |
|--------|---------|------------|
| 10K prefill | 57s | **4.1s** |
| First token latency | ~57s | **~4.1s** |
| Time to 100 generated tokens | ~74s | **~21s** |

The difference between 57s and 4s for first-token latency makes IDE integration practical.

## Key Takeaways

1. **Expert DMA was the bottleneck** — 7.3 GB per forward call vs 8 MB of prompt data (900:1 ratio)
2. **Persistent expert buffers eliminate it** — 14-72x prefill speedup depending on prompt size
3. **The VRAM trade-off is acceptable** — 93K token KV cache is more than enough for IDE use
4. **CPU decode is unaffected** — stable at 5.8 tok/s regardless of prefill mode
5. **Attention compute is now the scaling factor** — longer prompts = more attention work (expected, not a bottleneck)
6. **GPU prefill is now practical for IDE use** — 10K prompt in 4s instead of 57s

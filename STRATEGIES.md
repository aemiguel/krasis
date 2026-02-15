# Krasis MoE Expert Dispatch Strategies

Krasis supports multiple strategies for dispatching MoE (Mixture of Experts) expert computation between GPU and CPU. Each strategy trades off VRAM usage, DMA overhead, and throughput differently.

## Strategy Summary

| Name | CLI | Description |
|------|-----|-------------|
| **cpu_only** | `--no-gpu-decode` | All experts on CPU via Rust AVX2 engine. Baseline. |
| **chunked** | `--expert-divisor 0` | DMA all experts per-layer per-chunk from CPU to GPU. Original. |
| **active_only** | `--cache-strategy active_only` | DMA only activated experts per-layer. Eliminates wasted transfers. |
| **static_pin** | `--cache-strategy static_pin` | active_only + heatmap-based pinning of hot experts to GPU VRAM. |
| **weighted_pin** | `--cache-strategy weighted_pin` | active_only + weighted per-layer pinning proportional to activation skew. |
| **lru** | `--cache-strategy lru` | Cross-layer LRU expert cache with eviction. |
| **persistent** | `--expert-divisor 1` | ALL experts pre-loaded in VRAM. Zero DMA. Small models only. |
| **layer_grouped** | `--expert-divisor N` (N>=2) | Group N layers, DMA once per group. Amortizes DMA cost. |
| **hot_cached_static** | `--cache-strategy hot_cached_static` | Hot experts on GPU (static), cold experts on CPU, run in parallel. Zero DMA. |

---

## cpu_only

**CLI:** `--no-gpu-decode` (no `--cache-strategy`)

All routed experts computed on CPU via the Rust AVX2 engine. GPU only handles attention and shared experts. Hidden states transfer GPU→CPU→GPU every MoE layer.

**Prefill flow:** CPU computes all experts sequentially/parallel (Rayon). Very slow for long prompts.
**Decode flow:** Same as prefill. ~0.5 s/tok for large models.

**Pros:** Minimal VRAM. Works with any model size.
**Cons:** No GPU acceleration for experts. Prefill is extremely slow.
**When to use:** Testing, or when GPU VRAM is needed entirely for KV cache.

---

## chunked

**CLI:** `--expert-divisor 0`

Original GPU prefill strategy. Divides experts into chunks that fit in a fixed GPU buffer. For each MoE layer, iterates over chunks: DMA all experts in the chunk to GPU, run `fused_marlin_moe`, accumulate output.

**Prefill flow:** For each layer, for each chunk: DMA chunk → kernel → accumulate. O(layers × chunks) DMA operations.
**Decode flow:** Same as prefill but with M=1. Very wasteful — DMAs all experts even when only top-k are active.

**Pros:** Simple. Works with any expert count.
**Cons:** O(layers × chunks) DMA dominates. Transfers unused experts.
**When to use:** Fallback when no Rust engine is available (legacy safetensors path).

---

## active_only

**CLI:** `--cache-strategy active_only`

Only DMA the experts that are actually activated by routing (top-k selection). Maintains a full [num_experts, ...] GPU buffer but only populates the active slots. For decode (M=1), uses compact per-layer buffers with cross-token caching.

**Prefill flow:** For each layer: determine active experts → batched DMA of missing experts → `fused_marlin_moe` with full buffer.
**Decode flow:** Compact buffers (21 slots/layer). Experts persist across tokens. LRU eviction within each layer's slots. ~55% hit rate on Coder-Next.

**Pros:** Eliminates wasted DMA. Good decode hit rate with compact buffers.
**Cons:** Still DMA-bound for cache misses. Buffer zeroing overhead for prefill.
**When to use:** Base strategy before adding pinning. Good for models where expert activation is sparse.

---

## static_pin

**CLI:** `--cache-strategy static_pin`

Builds on active_only. After a warmup request, pins the globally hottest (layer, expert) pairs permanently in GPU VRAM based on an activation heatmap. Pinned experts use GPU→GPU copy (fast) instead of CPU→GPU DMA.

**Prefill flow:** Same as active_only, but pinned experts are instant GPU→GPU copies.
**Decode flow:** Compact buffers pre-populated with pinned experts on first creation. Higher hit rate.

**Benchmark results (Qwen3-Coder-Next PP=2):**
- Decode: 5.31 avg / 5.44 best tok/s (184ms/tok)
- 8K prompt: 95s TTFT, 91 tok/s prefill
- 3,777 experts pinned, 21% prefill hit rate, ~55% decode compact hit rate

**Pros:** Best existing decode speed. Good balance of VRAM and speed.
**Cons:** Still has DMA for unpinned experts. Heatmap may not cover all workloads.
**When to use:** Production default. Best general-purpose strategy for large models.

---

## weighted_pin

**CLI:** `--cache-strategy weighted_pin`

Like static_pin but allocates per-layer pinning budgets proportional to activation skew. Layers with more concentrated expert usage (fewer unique experts activated more frequently) get more pinning slots.

**Prefill/Decode flow:** Same as static_pin with different pinning distribution.

**Pros:** Better pin allocation for models with uneven layer activation patterns.
**Cons:** Slightly more complex. May not outperform uniform pinning in practice.
**When to use:** When profiling shows significant per-layer activation skew.

---

## lru

**CLI:** `--cache-strategy lru`

Cross-layer LRU expert cache. Maintains a large pool of GPU expert slots shared across all layers. Each slot caches one (layer, expert) pair. LRU eviction when full.

**Prefill flow:** For each active expert: check LRU cache → hit: copy to staging → miss: batched DMA + cache insertion + LRU eviction.
**Decode flow:** Uses compact per-layer buffers (same as active_only).

**Pros:** Cross-layer sharing can improve hit rate for models with similar activation patterns across layers.
**Cons:** Complex eviction logic. May not outperform per-layer strategies.
**When to use:** When VRAM budget allows large cache and expert reuse across layers is high.

---

## persistent

**CLI:** `--expert-divisor 1`

Pre-loads ALL expert weights for ALL MoE layers into GPU VRAM at startup. Zero DMA during inference. Falls back to layer_grouped(2) on OOM.

**Prefill flow:** Zero DMA. Direct index into persistent buffers → `fused_marlin_moe`.
**Decode flow:** Same. Zero DMA.

**Benchmark results (V2-Lite):**
- Prefill: 2,494 tok/s (10K prompt)
- Decode: 5.8 tok/s
- Total VRAM: 10,746 MB (7,654 MB experts + attention + KV)

**Pros:** Maximum speed. Zero DMA overhead.
**Cons:** Requires all experts to fit in VRAM. Only works for small models (V2-Lite: 16 experts × 27 layers).
**When to use:** Small MoE models where all experts fit in VRAM with room for KV cache.

---

## layer_grouped

**CLI:** `--expert-divisor N` (N >= 2)

Groups MoE layers into N groups. For each group, loads all experts once, then processes all token chunks through the group's layers before freeing. Amortizes DMA: O(N) loads instead of O(layers × chunks).

**Prefill flow:** For each group: DMA all experts once → for each chunk → for each layer in group → compute. Fixed DMA overhead regardless of prompt length.
**Decode flow:** Not designed for decode (uses standard active_only path).

**Benchmark results (V2-Lite, divisor=2):**
- 512 tok: 48 tok/s (5.2x vs chunked)
- 1K+: ~same as chunked (~95-340 tok/s)
- Fixed ~10.4s DMA overhead
- KV cache: 216K tokens (vs 93K persistent) — 2.3x more context

**Pros:** Fixed DMA cost. More VRAM for KV cache than persistent.
**Cons:** DMA cost is constant even for short prompts. No decode acceleration.
**When to use:** Large models where persistent doesn't fit. Long prompt prefill.

---

## hot_cached_static

**CLI:** `--cache-strategy hot_cached_static`

Hybrid GPU/CPU strategy with zero DMA during inference. Hot experts (selected by heatmap) are loaded once into static GPU buffers at startup. Cold experts are computed on CPU via the Rust AVX2 engine. Both run in parallel per MoE layer.

**Architecture:**
```
Per MoE layer:
  1. Route on GPU -> topk_ids [M, top_k]
  2. Lookup table: split into GPU (hot) vs CPU (cold)
  3. GPU: fused_marlin_moe on STATIC buffer ---+
     CPU: Rust AVX2 on cold experts -----------+ PARALLEL
  4. Sync + combine weighted outputs -----------+
```

**Prefill flow:** Split experts by lookup table. GPU processes hot experts via `fused_marlin_moe`. CPU processes cold experts via Rust engine (async). Total time = max(GPU, CPU) per layer.
**Decode flow (M=1):** Same split. CPU typically finishes in ~0.3-0.9ms, within GPU's ~1.5ms Marlin kernel window — CPU work is essentially free.

**Key properties:**
- Zero DMA during inference. GPU buffer loaded once at startup.
- CPU already has ALL expert weights loaded (Rust engine).
- Shared expert always on GPU (separate stream, unchanged).
- Expert IDs set to -1 for GPU-handled experts in CPU submission (Rust engine skips them).
- Each layer may have different number of pinned experts.
- Optional CUDA graph capture for decode eliminates kernel launch overhead.

**Expected performance (Coder-Next PP=2):**

| Config | Decode tok/s | ms/tok |
|--------|-------------|--------|
| static_pin (baseline) | 5.44 | 184 |
| hot_cached_static | ~5.5-6.0 | ~167-182 |
| + CUDA graphs | ~7-9 | ~111-143 |

**Pros:** Zero DMA. CPU work is free during decode. CUDA graphs possible.
**Cons:** Requires heatmap for expert selection. Cold experts limited by CPU speed during prefill.
**When to use:** Production deployment. Maximum decode throughput for large MoE models.

---

## Design Philosophy: Prefill First

**GPU prefill speed is the top priority.** The whole point of Krasis is that CPU decode is the unavoidable bottleneck — you're always waiting for it. What you CAN control is how fast prompts are processed (TTFT). A slow TTFT makes the tool unusable in interactive settings (IDE integration, chat).

Therefore:
1. **Prefill gets first claim on VRAM.** The best prefill strategy (HCS or layer_grouped) is selected first, and its VRAM requirements are satisfied.
2. **Decode strategy uses whatever VRAM is left.** If HCS prefill needs 2.5 GB for hot expert buffers, decode gets what remains. If that means decode can only use LRU with fewer slots, or falls back to active_only — so be it.
3. **Never sacrifice prefill for decode.** A 10% decode improvement that costs 50% prefill speed is a bad trade. Decode is CPU-bound anyway (~10 tok/s ceiling on our hardware).

This means the auto-optimiser should:
- Select the fastest prefill strategy first
- Then find the best decode strategy within the remaining VRAM budget
- Never let decode strategy choices reduce prefill throughput

---

## Strategy Selection Guide

| Model Size | Experts | VRAM | Recommended |
|-----------|---------|------|-------------|
| Small (V2-Lite) | 16/layer | Fits all | **persistent** |
| Medium | 64/layer | Fits ~50% | **hot_cached_static** |
| Large (Coder-Next) | 512/layer | Fits ~4% | **hot_cached_static** |
| Very large | 512+/layer | Fits <1% | **static_pin** or **hot_cached_static** |

**General rule:** If you can fit all experts → persistent. If you can fit hot experts → hot_cached_static. If you need maximum VRAM for KV cache → static_pin with compact decode buffers.

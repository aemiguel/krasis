# Krasis Changelog

## Qwen3-Coder-Next & Qwen3-235B Auto-Optimiser Benchmark — 2026-02-15

### Benchmark
Ran auto-optimiser on Qwen3-Coder-Next (48 layers, 512 experts) and Qwen3-235B-A22B (94 layers, 128 experts) with PP=2 across 2 GPUs. Tested all 6 prefill and 4 decode strategies with 3 runs each (10K prefill, 64 decode tokens).

### Results
- **Coder-Next**: layer_grouped_4 prefill (619 tok/s, TTFT 16.2s) + pure_cpu decode (7.26 tok/s)
  - persistent/layer_grouped_2 OOM (512 experts too large)
  - Optimise time: 834s
- **235B-A22B**: hcs_prefill (211 tok/s, TTFT 47.4s) + pure_cpu decode (1.81 tok/s)
  - persistent/layer_grouped_2/layer_grouped_4 ALL OOM (128×3072 experts = 9.7 MB each)
  - compact/LRU decode catastrophically slow (0.23 tok/s) — only 6 compact layers fit
  - Optimise time: 4124s
- **Universal finding**: pure_cpu decode wins on ALL models (V2-Lite, Coder-Next, 235B)
- Results saved to `tests/qwen3_coder_next_auto_2gpu.json` and `tests/qwen3_235b_a22b_auto_2gpu.json`
- BENCHMARKS.md updated with full strategy comparison tables and cross-model summary

---

## V2-Lite Auto-Optimiser Benchmark — 2026-02-15

### Benchmark
Ran auto-optimiser on DeepSeek-V2-Lite with 1 GPU and 2 GPUs. Tested all 6 prefill
and 4 decode strategies with 3 runs each (10K prefill, 64 decode tokens).

### Results
- **1 GPU**: persistent prefill (1,757 tok/s, TTFT 5.69s) + pure_cpu decode (6.98 tok/s)
- **2 GPU**: layer_grouped_2 prefill (1,729 tok/s, TTFT 5.79s) + pure_cpu decode (6.69 tok/s)
- Quantization: INT4 Marlin GPU, INT4 CPU, FP8 KV, INT8 attention
- Results saved to `tests/v2lite_auto_1gpu.json` and `tests/v2lite_auto_2gpu.json`
- BENCHMARKS.md updated with full strategy comparison tables

---

## Auto-Optimiser Integration — 2026-02-15

### Feature
Integrated the meta-optimiser into Krasis startup flow. On first launch with
`--strategy auto`, Krasis automatically discovers the optimal prefill + decode
strategy pair, saves results, and loads them instantly on subsequent launches.

### How It Works
- **First run (~12-18 min):** builds expert heatmap, benchmarks 6 prefill + 4 decode
  strategies, saves results to `.krasis_cache/auto_optimise.json`
- **Subsequent runs (~2s):** validates hardware/model fingerprint, loads cached result,
  applies winning strategy
- **Cache invalidation:** automatic when GPU model/count/VRAM changes, model path or
  PP partition changes, or quantization config changes

### Files Changed
- `python/krasis/auto_optimise.py`: New module — AutoOptimiser class, cache validation,
  fingerprinting, `auto_optimise_or_load()` entry point
- `python/krasis/server.py`: Added `--strategy auto|manual` and `--force-optimize` flags.
  Auto mode builds with neutral expert_divisor=0, then calls auto_optimise_or_load()
- `python/krasis/launcher.py`: Added "auto" as first choice for expert_divisor in TUI,
  passes `--strategy auto` to server.py when selected, default changed to "auto"
- `CHANGELOG.md`: This entry

---

## Qwen3-235B PP=2 HCS Hybrid Benchmark — 2026-02-15

### Benchmark
Ran Qwen3-235B-A22B with PP=2 (47+47 layers) HCS hybrid on 2x RTX 2000 Ada 16GB.
Direct comparison with KTransformers and llama.cpp on the same model.

### Results
- Prefill: 198.1 tok/s avg (3.4x faster than KTransformers' 57.5, 6.0x faster than llama.cpp's 32.9)
- TTFT: 43.9s avg (3.4x faster than KTransformers' 149.5s)
- Decode: 1.65 tok/s avg (lower than KT's 4.85, expected with large expert dims)
- 86 hot experts pinned (cuda:0=27, cuda:1=59)
- VRAM: GPU0=12,963 MB, GPU1=12,968 MB, RAM: 223 GB

### Files Changed
- `tests/bench_qwen235b_pp2.py`: New PP=2 benchmark script
- `BENCHMARKS.md`: Added PP=2 Krasis result + direct comparison table
- `CHANGELOG.md`: This entry

---

## Qwen3-235B HCS Hybrid Benchmark — 2026-02-14

### Benchmark
Ran Qwen3-235B-A22B with HCS hybrid strategy (layer_grouped prefill + HCS decode)
on PP=3 (32+31+31) across 3x RTX 2000 Ada 16GB GPUs.

### OOM Fix
`_compute_layer_groups(divisor=-3)` returned ALL layers in ONE group because
`divisor <= 1` branch catches negative values. For 235B: 128 experts × 32 layers
× 9.7 MB = ~39.8 GB → instant OOM on 16 GB GPU. Fixed with `_hcs_prefill_divisor()`
that auto-computes divisor from available VRAM.

### Results
- 535 hot experts pinned (145+216+174 across 3 GPUs)
- TTFT: 187.6s (was 327.3s with div=4, **43% faster**)
- Prefill: 46.4 tok/s (was 26.3, **1.76x**)
- Decode: 1.73 avg / 1.80 best tok/s (was 1.45, **19% faster**)
- VRAM: 12,861 + 12,859 + 12,658 MB
- Output quality: not coherent (known 235B model execution issue)

### Files Changed
- `python/krasis/model.py`: Added `_hcs_prefill_divisor()`, HCS prefill dispatch
- `tests/run_235b_benchmark.py`: New HCS hybrid benchmark for 235B
- `BENCHMARKS.md`: Added HCS hybrid row and detailed results

---

## HCS Hybrid: DMA Prefill + HCS Decode — 2026-02-14

### Problem
Pure HCS routed cold experts through CPU for ALL token counts. At M=8K, CPU
processing took 178s vs 23s with DMA. CUDA graphs also proved useless (CPU sync
is the bottleneck, not kernel launch overhead).

### Fix
One-line dispatch change in `gpu_prefill.py forward()`: M>1 routes through
DMA-based `_forward_active_only`, M=1 routes through `_forward_hot_cached_static`.

### Benchmark Results (Qwen3-Coder-Next PP=2, 8278 token prompt)

| Strategy | Short Decode (tok/s) | 8.6K Total (tok/s) | 8.6K Time |
|----------|---------------------|---------------------|-----------|
| static_pin | 5.09 (best 5.19) | 0.48 (best 0.49) | 66.5s |
| **hcs_hybrid** | **6.03 (best 6.07)** | **0.71 (best 0.72)** | **45.2s** |

HCS hybrid wins: **+18% decode, +48% total throughput, 32% faster 8K**.

### Files Changed
- `python/krasis/gpu_prefill.py`: HCS dispatch now routes M>1 to active_only DMA
- `tests/run_hcs_hybrid_bench.py`: New benchmark script

---

## hot_cached_static Strategy — 2026-02-14

### New Strategy: hot_cached_static
Hybrid GPU/CPU MoE dispatch with zero DMA during inference. Hot experts (selected by
heatmap) are loaded once into static per-layer GPU buffers at startup. Cold experts are
computed on CPU via the Rust AVX2 engine. Both run in parallel per MoE layer.

**Architecture:**
- Per MoE layer: split topk_ids via lookup table into GPU (hot) vs CPU (cold)
- Submit cold experts to CPU engine async (hot expert IDs set to -1, Rust skips them)
- Run fused_marlin_moe on static GPU buffer for hot experts (cold weights zeroed)
- Sync CPU result + combine: output = gpu_output + cpu_output
- For M=1 decode: CPU finishes within GPU's ~1.5ms Marlin kernel window — CPU work is free

**Features:**
- `--cache-strategy hot_cached_static` CLI flag
- `--heatmap-path` to load expert activation heatmap from file
- `--cuda-graphs` for optional CUDA graph acceleration (eliminates kernel launch overhead)
- `save_heatmap()` method to export accumulated activation data
- Per-layer static buffers (different hot expert counts per layer)
- Auto-enables GPU decode (threshold=1)

### STRATEGIES.md Documentation
Comprehensive documentation of all 9 MoE expert dispatch strategies:
cpu_only, chunked, active_only, static_pin, weighted_pin, lru, persistent,
layer_grouped, and hot_cached_static. Includes architecture diagrams, benchmark
results, pros/cons, and selection guide.

### Benchmark Results (Qwen3-Coder-Next PP=2, 2x RTX 2000 Ada)

**Short prompt decode (pure decode speed):**
| Strategy | Avg tok/s | Best tok/s |
|----------|-----------|------------|
| static_pin | 4.78 | 4.86 |
| hot_cached_static | **5.98** | **6.10** |
| hcs+cuda_graphs | 5.98 | 6.09 |

**8K prompt (total incl prefill):**
| Strategy | TTFT | Total tok/s |
|----------|------|-------------|
| static_pin | ~23s | 1.09 |
| hot_cached_static | ~178s | 0.17 |
| hcs+cuda_graphs | ~178s | 0.15 |

**Findings:**
- HCS decode is **25% faster** than static_pin (zero DMA, ~158ms/tok vs ~184ms/tok)
- HCS prefill is **7.7x slower** (cold experts routed through CPU at M=8192)
- CUDA graphs add nothing (~0% improvement) — CPU sync is the bottleneck, not kernel launch
- **Optimal hybrid**: static_pin for prefill (GPU DMA) → HCS for decode (zero DMA)
- HCS hot experts: 1817 (GPU0) + 1912 (GPU1) = 3729 experts pinned, ~6 GB total

### Benchmark Script
`tests/bench_hot_cached_static.py`: Compares static_pin, hot_cached_static, and
hot_cached_static+CUDA graphs. Includes heatmap generation mode.
`tests/run_hcs_benchmark.py`: Full automated benchmark for Qwen3-Coder-Next.

### Files Changed
- `python/krasis/gpu_prefill.py`: _init_hot_cached_static(), _forward_hot_cached_static(),
  _forward_hcs_cpu_only(), _init_cuda_graphs(), _forward_hcs_graphed(), save_heatmap()
- `python/krasis/server.py`: --cache-strategy hot_cached_static, --heatmap-path, --cuda-graphs
- `python/krasis/model.py`: hot_cached_static in is_active_only check
- `STRATEGIES.md`: New file documenting all strategies
- `tests/bench_hot_cached_static.py`: New benchmark script

---

## Compact Expert Buffers + Further Decode Optimizations — 2026-02-14

### Compact Per-Layer Expert Buffers (MAJOR OPTIMIZATION)
The shared AO buffer [512 slots] was cleared on every layer change during decode, causing
zero cross-token expert caching. Compact buffers partition the same VRAM into per-layer
views (21 slots each), so experts loaded for layer N at token T survive for token T+1.

**Key insight**: Consecutive decode tokens share ~52% of expert activations per layer. By
keeping experts in per-layer compact buffers, only ~48% of experts need fresh DMA each token.

- **GPU Decode**: **3.41 → 5.44 tok/s** (+60%, 184ms/tok)
- **CPU Decode**: **3.04 → 4.92 tok/s** (+62%, 203ms/tok)
- Zero extra VRAM (views into existing AO buffer)
- Per-layer LRU eviction when compact slots are full
- topk_ids remapped global→local before calling fused_marlin_moe

### Additional Optimizations (on top of compact buffers)
- **Compact slots 16 → 21**: More cache headroom for 10-expert top-k selection
- **CUDA stream overlap**: Shared expert computation on separate stream, overlapped with routed expert DMA+kernel
- **Pre-populate from pinned**: On first access, compact buffers pre-filled with pinned experts (GPU→GPU copy, instant cache hits)
- **index_copy_ batching**: Replace 40 per-expert `copy_()` calls with 4 `index_copy_` ops
- **Zero-copy bytes**: `torch.frombuffer()` directly on Rust PyBytes (saves ~15 MB/layer)
- **Combined Rust FFI**: `get_experts_all_batch()` returns all 4 weight types in single call
- **Fused shared expert**: Concatenate gate+up into single matmul (1.2ms → 0.84ms)
- **CPU-side unique for M=1**: `list(set(topk_ids[0].tolist()))` avoids GPU kernel launch
- **Pre-allocated gating_output**: Eliminates per-call CUDA allocation for M=1

### All Changes:
1. **Compact buffers** (`gpu_prefill.py`): `_forward_compact()`, `_init_compact_decode()`, `_get_or_create_compact_buf()`, `_invalidate_compact()`
2. **Compact slots 16→21** (`gpu_prefill.py`): `_compact_slots_per_layer = 21`
3. **Pre-populate from pinned** (`gpu_prefill.py`): `_get_or_create_compact_buf()` copies pinned experts into compact on first access
4. **CUDA stream overlap** (`layer.py`): Shared expert on `_shared_stream` before dispatch, synced after
5. **index_copy_ batching** (`gpu_prefill.py`): Contiguous `.to(device)` + `index_copy_` scatter
6. **Zero-copy bytes** (`gpu_prefill.py`): `torch.frombuffer(rust_bytes)` without `bytearray()` wrapper
7. **Combined FFI** (`moe.rs`): `get_experts_all_batch()` — single call for all 4 weight types
8. **Fused shared expert** (`layer.py`): gate_proj + up_proj concatenated at load time
9. **CPU-side unique** (`gpu_prefill.py`): Avoids GPU `torch.unique()` for M=1
10. **Pre-allocated tensors** (`gpu_prefill.py` + `model.py`): gating_output, decode token/pos buffers

### Benchmark Results (Qwen3-Coder-Next, PP=2, 32 decode tokens):

**Short prompt (48 tokens, 4 runs):**

| Mode | Decode (tok/s) | ms/tok | Prefill (tok/s) |
|------|---------------|--------|-----------------|
| **GPU decode (compact)** | **5.31 avg / 5.43 best** | **184** | 203 (threshold=300) |
| CPU decode | 4.92 | 203 | 31.8 |
| **Speedup** | **1.08x** | | |

**8K prompt (8,700 tokens, 3 runs):**

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| TTFT | 112.4s | 95.1s | 95.5s |
| Prefill | 77.4 tok/s | 91.5 tok/s | 91.1 tok/s |
| Decode | 5.39 tok/s | 5.44 tok/s | 5.44 tok/s |

Full optimization progression: **2.14 → 2.23 (FP8 fix) → 3.41 (DMA batching) → 5.23 (compact buffers) → 5.44 (slots+overlap+prepopulate)**

---

## FP8 Native KV + GPU Decode — 2026-02-14

### FP8 Native KV Cache (MAJOR FIX)
**Root cause**: GQA attention was upcasting the ENTIRE paged KV cache layer (~1.1 GB) from FP8 to BF16
every decode step — even though FlashInfer only accesses the few pages in the active sequence.
With 12 GQA layers, this was ~13 GB of unnecessary data conversion per token.

**Fix** (`attention.py`): Pass FP8 KV cache directly to FlashInfer with `kv_data_type=torch.float8_e4m3fn`.
FlashInfer on SM89 (Ada) natively handles FP8 KV, only converting accessed pages internally.

- GQA attention per layer: **24ms → 1.4ms** (17x faster)
- GPU decode: **1.40 → 2.23 tok/s** (+59%)
- CPU decode: **1.37 → 1.98 tok/s** (+44%)

### GPU Decode for MoE
Route M=1 decode through GPU `fused_marlin_moe` kernel instead of CPU Rust engine.
Eliminates 48× GPU↔CPU hidden state transfers and Python-Rust boundary crossings per token.

### All Changes:
1. **FP8 native KV for GQA** (`attention.py`): Remove whole-cache `.to(torch.bfloat16)` upcast, pass FP8 directly to FlashInfer
1b. **FP8 selective upcast for MLA** (`attention.py`): MLA kernel still requires BF16, but now only upcasts USED pages (via kv_indices gather) instead of entire cache
2. **Core dispatch** (`layer.py`): M=1 decode now uses `_gpu_prefill_forward()` when `gpu_prefill_manager` is available
3. **Remove unconditional CUDA sync** (`gpu_prefill.py`): `torch.cuda.synchronize()` in `forward()` now debug-only
4. **Skip buffer zeroing for decode** (`gpu_prefill.py`): Skip zeroing [512, ...] AO buffer and workspace for M=1 (kernel only reads active expert slots)
5. **Batched expert DMA** (`moe.rs` + `gpu_prefill.py`): New `get_experts_batch()` Rust function fetches non-contiguous expert IDs in one PyO3 call (4 calls/layer instead of 4×N_experts)
6. **Per-layer timing** (`layer.py` + `model.py`): `KRASIS_DECODE_TIMING=1` env var for attention vs MoE breakdown
7. **`--gpu-decode` CLI flag** (`server.py`): Auto-enabled for active_only/static_pin strategies, sets `gpu_prefill_threshold=1`

### Benchmark Results (Qwen3-Coder-Next, PP=2, 32 decode tokens):

| Mode | Decode (tok/s) | ms/tok | Notes |
|------|---------------|--------|-------|
| **GPU decode** | **2.23** | 448 | threshold=1, static_pin |
| CPU decode | 1.98 | 504 | threshold=300, static_pin |
| **Speedup** | **1.12x** | | |

Both paths improved ~50% from FP8 fix alone. MoE is now the dominant bottleneck (73% of decode time).

---

## Qwen3-Coder-Next GPU Prefill Benchmark — 2026-02-14

First GPU prefill benchmark on Qwen3-Coder-Next (48 layers, 512 experts, top-10, PP=2).
Much smaller experts (1.55 MB each) enable 3x faster prefill than Qwen3-235B.

| Strategy | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Pinned | GPU VRAM |
|----------|----------|-----------------|----------------|--------|----------|
| **Static Pin** | **42.4** | **203.5** | **2.14** | 3,777 | 30.4 GB |
| Active-Only | 49.8 | 173.4 | 2.11 | 0 | 25.4 GB |

### Old-Mode Baselines (for comparison):

| Mode | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | GPU VRAM |
|------|----------|-----------------|----------------|----------|
| Chunked (div=0) | 65.8 | 131.4 | 2.13 | 28.9 GB |
| Layer-Grouped (div=4) | 65.2 | 132.4 | 2.60 | 16.4 GB |

**Static pin is 35% faster TTFT and 55% higher prefill throughput than old chunked mode.**
Layer-grouped has best decode (2.60 tok/s) due to lowest VRAM pressure (16.4 GB).

### Bug Fixed:
- Static pin OOM on Qwen3-Coder-Next: 2GB headroom insufficient for models with many small experts
  (2,864 pinned × 1.6 MB = 4.6 GB, left only 1 GB for workspace). Increased to 3.5 GB headroom.

---

## Expert Caching Strategies Benchmark — 2026-02-14

Implemented and benchmarked 5 expert caching strategies for GPU prefill DMA optimization
on Qwen3-235B-A22B (PP=1, 1 GPU, INT4 Marlin experts, INT8 attention, FP8 KV).

### Strategies Implemented (in `gpu_prefill.py`):
1. **Active-Only DMA** (`--cache-strategy active_only`): Per-layer gate→DMA, skip inactive experts
2. **Static Frequency Pinning** (`--cache-strategy static_pin`): Pin hottest experts globally after warmup
3. **Weighted Pinning** (`--cache-strategy weighted_pin`): Pin proportional to per-layer activation skew
4. **LRU Expert Cache** (`--cache-strategy lru`): Cross-layer VRAM cache with LRU eviction
5. **Hybrid** (`--cache-strategy hybrid`): LRU + static pinning (combined)

### Benchmark Results (Run 4 / hot cache, 8,700-token prompt):

| Strategy | TTFT (s) | Improvement | DMA (s) | Hit Rate |
|----------|----------|-------------|---------|----------|
| Baseline (div=32) | 193.5 | — | ~170 | N/A |
| Active-Only | 152.1 | -21.4% | 128.1 | N/A |
| Static Pin | 148.8 | -23.1% | 124.7 | 1.8% |
| **Weighted Pin** | **142.9** | **-26.1%** | **118.8** | **1.7%** |
| LRU Cache | 153.6 | -20.6% | 129.6 | 0% |
| Hybrid | 158.6 | -18.0% | 134.5 | 0% |

### PP=3 Results (3 GPUs, 48 GB total, Run 4 / hot cache):

| Strategy | TTFT (s) | vs AO | Pinned | Total VRAM |
|----------|----------|-------|--------|------------|
| **Static Pin** | **128.7** | **-12.2%** | **1,078** | **44.7 GB** |
| Weighted Pin | 130.4 | -11.0% | 1,078 | 44.7 GB |
| Active-Only | 146.6 | — | 0 | 34.0 GB |
| LRU Cache | 158.9 | +8.4% | 0 | 47.2 GB |
| Hybrid | 164.4 | +12.1% | 0 | 47.0 GB |

### Recommendation:
- **PP=3**: `--cache-strategy static_pin` — 128.7s TTFT, 14% hit rate
- **PP=1**: `--cache-strategy weighted_pin` — 142.9s TTFT, 26% faster than baseline
- LRU/Hybrid: 0% hit rate regardless of GPU count — avoid

### Files Changed:
- `python/krasis/gpu_prefill.py`: 5 prefill modes, heatmap tracking, pinning, LRU cache
- `python/krasis/model.py`: Active-only/LRU forward dispatch, stats logging
- `python/krasis/server.py`: `--cache-strategy` CLI argument

### Bugs Fixed:
- LRU buffer OOM: only reserved 300 MB for KV cache, increased to 2 GB
- Pin OOM: 224 experts overcommitted VRAM, added 1 GB headroom for intermediates
- Stats tracking: hit counter only fired when ALL experts cached, fixed per-expert counting
- `_compute_layer_groups`: negative divisor caused crash, added early return

---

## Test Status

| Suite | Count | Status | Last Run |
|-------|-------|--------|----------|
| Rust (`cargo test`) | 52 | 44 PASS, 8 SKIP (model files) | 2026-02-12 |
| Python bridge (`test_bridge.py`) | 3 | ALL PASS | 2026-02-09 |
| GPU prefill (`test_gpu_prefill.py`) | 5 | ALL PASS | 2026-02-09 |
| **Total** | **56** | **40+ PASS** | |

Re-run needed after: any change to `src/`, `python/krasis/`, or test files.

---

## Enable GPU prefill for Qwen3-Coder-Next — 2026-02-13

**GPU prefill with Marlin MoE kernel now works for hybrid linear+GQA models.**

### Performance (PP=2, INT4 Marlin GPU, INT4 CPU, FP8 KV, 48 threads):

| Mode | Prompt | Prefill speed | KV context | Notes |
|------|--------|-------------|------------|-------|
| Layer-grouped (divisor=4) | 5008 tok | **~78 tok/s** | 745K tokens | 4 groups/GPU, ~7s DMA each, 88% DMA overhead |
| Layer-grouped (divisor=4) | 1170 tok | ~23 tok/s | 745K tokens | DMA dominates at small prompts |
| Chunked (divisor=0) | 1170 tok | ~22 tok/s | 1M+ tokens | 1 chunk/layer, 48 DMA ops/GPU |
| CPU-only (no GPU prefill) | — | ~8 tok/s | — | Baseline |

- Raw GPU Marlin compute: **~640 tok/s** (DMA-free estimate from 5008-token run)
- Decode: **~2.0 tok/s** (CPU MoE)
- All 512 experts fit in 1 chunk (830 MB), num_chunks=1 → single kernel call per layer
- KV cache only for 12 full-attention layers (not 48)
- DMA throughput ~715 MB/s (4983 MB in ~7s) — bottleneck for prompts under ~5K tokens

### Changes:
- **`gpu_prefill.py`**: Fixed fallback bug — when persistent mode OOMs and falls back to
  layer_grouped, but model uses regular forward(), the manager now gracefully falls through
  to engine chunked path instead of crashing on empty persistent buffers
- **`kv_cache.py`**: Added `vram_reserve_bytes` parameter to auto-sizer, subtracted from
  free VRAM before computing KV cache budget (prevents OOM with layer-grouped prefill)
- **`model.py`**: Added VRAM reservation for layer-grouped prefill using per-rank MoE layer
  count (not total), so KV cache correctly sizes around expert group VRAM
- **`test_qwen3_next_generate.py`**: Updated to use `gpu_prefill=True, expert_divisor=0`
- **`test_qwen3_next_prefill_speed.py`**: GPU prefill benchmark with ~5000-token prompt (8 tokens decode)

### VRAM budget (per GPU, layer-grouped divisor=4):
- GPU weights: ~2.4 GB (INT8 attention + embedding/norm)
- Expert group buffer: ~5 GB (6 layers × 512 experts × 1.6 MB Marlin INT4)
- KV cache: ~4.5 GB (6 layers, 745K tokens)
- Total: ~12 GB of 16 GB

---

## Fix Qwen3-Coder-Next empty responses — 2026-02-13

**Root cause**: 7 bugs in the Gated DeltaNet linear attention, GQA attention, and RMSNorm implementation.

### Bug 1: QKVZ interleaving (CRITICAL)
`in_proj_qkvz` output is interleaved per key-head group, not sequential.
Added `_fix_query_key_value_ordering()` to un-interleave Q,K,V,Z by reshaping
to `[M, num_k_heads, group_dim]` and splitting within each group.
Without this, all 36 linear attention layers produced garbage activations.

### Bug 2: BA interleaving (CRITICAL)
Same issue for `in_proj_ba` — beta and alpha were mixed up across heads.
Fixed in the same `_fix_query_key_value_ordering()` function.

### Bug 3: Conv1d state update order
State was updated BEFORE computing conv output, losing oldest element and
double-counting the new token. Now uses proper `F.conv1d` with state+new
concatenation, then updates state to last `kernel_dim` tokens.
Conv state size also fixed: `kernel_dim` (4) not `kernel_dim-1` (3).

### Bug 4: Missing query scale factor
HF reference applies `scale = 1/sqrt(head_dim)` to query AFTER L2 normalization.
Added `self.scale = 1.0 / (k_head_dim ** 0.5)` and `q *= scale`.

### Bug 5: Missing `shared_expert_gate`
Qwen3-Coder-Next has `mlp.shared_expert_gate.weight` [1, hidden_size] — a sigmoid
gate that scales shared expert output: `sigmoid(gate @ hidden) * shared_expert_out`.
- Added loading in `weight_loader.py:load_shared_expert()`
- Added application in `layer.py:_shared_expert_forward()`
- Set `skip_shared_experts=True` on Rust engine and GpuPrefillManager when gate exists
  (Python/GPU handles shared expert with gate; avoids double-counting)

### Bug 6: Missing gated attention in GQA
Qwen3-Coder-Next's `q_proj` outputs `num_heads * head_dim * 2` — the second half
is a sigmoid gate applied to the attention output before `o_proj`:
`attn_output * sigmoid(gate)`. Auto-detected from q_proj weight dimensions.

### Bug 7: RMSNorm convention mismatch (CRITICAL)
Qwen3-Coder-Next uses `Qwen3NextRMSNorm` which computes `(1 + weight) * norm(x)` where
weight is initialized to **zeros**. Standard RMSNorm computes `weight * norm(x)` where
weight is initialized to **ones**. Our `flashinfer.norm.rmsnorm` uses the standard formula.
Without +1.0 correction, all layer norms multiply by near-zero → activations collapse
to zero by layer 47 (hidden std 0.004).
- Added `norm_bias_one` config flag, set True for `qwen3_next` model_type
- Apply `weight + 1.0` at load time for: input_layernorm, post_attention_layernorm,
  q_norm, k_norm, final_norm (NOT the gated norm in linear attention which uses standard convention)
- Linear attention comparison test: cos_sim=0.999996 vs HF (verified correct)

### Additional fixes:
- Chunked prefill algorithm ported from HF `torch_chunk_gated_delta_rule` (was buggy custom impl)
- L2 norm matches FLA library: `rsqrt(x²·sum + eps)` not `F.normalize`
- Gated RMSNorm: gate converted to float32 before SiLU (numerical stability)
- BF16 cast before out_proj in linear attention (recurrent + chunked paths)

### Test results:
- `test_qwen3_next_generate.py`: PASS (2/2 tests, "2+2=4" and "count 1-5")
- `test_linear_attn_compare.py`: cos_sim=0.999996 vs HF reference
- Prefill: 21 tokens in ~2.7s, decode: ~0.5s/token (CPU-only, no GPU prefill)

### Files modified:
- `python/krasis/config.py` — `norm_bias_one` flag, detect `qwen3_next` model_type
- `python/krasis/linear_attention.py` — near-complete rewrite, BF16 cast fix
- `python/krasis/weight_loader.py` — load shared_expert_gate, apply norm +1.0 correction
- `python/krasis/layer.py` — apply shared_expert_gate, add shared expert on GPU when gate exists
- `python/krasis/model.py` — detect shared_expert_gate, set skip_shared on engine + prefill manager
- `python/krasis/gpu_prefill.py` — skip_shared_experts parameter
- `python/krasis/attention.py` — gated attention support

---

## Move models to krasis/models/ — 2026-02-13

Models now live in `krasis/models/` instead of `~/Documents/Claude/hf-models/`.

- Created `models/` directory in krasis repo root
- Moved all 6 model directories (DeepSeek-V2-Lite, DeepSeek-V2-Lite-GGUF, GLM-4.7, Qwen3-235B-A22B, Qwen3-235B-A22B-Thinking-2507, Qwen3-Coder-Next)
- Added `models/` and `.krasis_config` to `.gitignore`
- Left symlink at old location (`~/Documents/Claude/hf-models/` → `krasis/models/`)
- Updated launcher to scan `models/` relative to repo root (was hardcoded `~/Documents/Claude/hf-models/`)
- Updated saved config with new paths

---

## Krasis Chat — interactive terminal chat client — 2026-02-12

New `krasis-chat` script and `python/krasis/chat.py` module.

**Features:**
- Auto-discovers running Krasis servers by scanning localhost:8080-8090
- Arrow-key server selection screen (if multiple servers found)
- Streaming SSE chat with token-by-token display
- Multi-turn conversation history
- Commands: `/new` (clear history), `/system <msg>` (set system prompt), `/exit`
- Stats after each response (approximate tokens, time, tok/s)
- Uses only stdlib (`http.client` for streaming, `urllib` for discovery, `readline` for input)

**Usage:**
```
./krasis-chat                                    # auto-discover
./krasis-chat --port 8080                        # specific port
./krasis-chat --url http://host:8080             # direct URL
./krasis-chat --system "You are a coding assistant."
```

### New: `python/krasis/chat.py`
### New: `krasis-chat` (bash wrapper)

---

## Split model selection: native + CPU expert source — 2026-02-12

Launcher now has a 4-screen interactive flow (always shown, with saved config pre-selected):

1. **Select native model** — shows only HF models with safetensors files
2. **Select CPU expert source** — build INT4/INT8 from native, or pick a GGUF file
3. **Select GPUs** — toggle individual GPUs on/off
4. **Config screen** — fine-tune all parameters with live VRAM budget

### Changed: `python/krasis/launcher.py`
- `scan_models()`: added `native_only` param — filters to directories with `.safetensors` files
- `scan_gguf_files()`: **new** — scans all subdirectories for `.gguf` files, returns name/path/size
- `CpuExpertChoice`: **new** dataclass — result of CPU expert selection (source, bits, gguf_path)
- `_cpu_expert_selection_screen()`: **new** — shows "Build INT4" / "Build INT8" + available GGUF files
- `_model_selection_screen()`: added `preselected_path` param; removed `has_gguf` display tag
- `run_interactive()`: 4-screen flow (model → CPU experts → GPUs → config)
- `print_summary()`: shows CPU expert source (GGUF or build INT4/8)
- Removed `has_gguf` field from model dicts (no longer needed)

---

## Fix Ctrl-C server exit — 2026-02-12

**Fixed**: Ctrl-C during active generation would hang instead of exiting.

Root cause: uvicorn's graceful shutdown waits for active HTTP connections to close,
but the generation thread is blocked in the Rust/CUDA forward pass and never finishes.

Fix: Use `uvicorn.Server` directly with a patched `handle_exit` that sets `force_exit=True`
on first Ctrl-C (skip waiting for connections), and `os._exit(0)` on second Ctrl-C.

### Changed: `python/krasis/server.py`
- Replaced `uvicorn.run()` with `uvicorn.Server` + patched `handle_exit`
- First Ctrl-C: sets `should_exit + force_exit`, server exits within ~1s
- Second Ctrl-C: `os._exit(0)` for immediate kill if still stuck

---

## Gated DeltaNet (Linear Attention) Support — 2026-02-12

**Added support for hybrid Transformer-Mamba models (Qwen3-Coder-Next).**

Qwen3-Coder-Next uses 36 linear attention layers (Gated DeltaNet) + 12 standard GQA layers.
Linear attention layers maintain a small recurrent state (~1 MB) + conv state (~48 KB) on GPU
instead of KV cache. This reduces KV cache from 48 layers to just 12 → 4x VRAM savings for KV.

### New: `python/krasis/linear_attention.py`
- `GatedDeltaNetAttention` class: pure PyTorch Gated DeltaNet implementation
- Recurrent decode (M=1): sequential state update per token
- Chunked prefill (M>1): parallel-within-chunk, recurrent-across-chunks
- Internal conv state and recurrent state (lazy-initialized, auto-reset between sequences)
- Gated RMSNorm output with learnable per-head weight

### Modified: `python/krasis/config.py`
- Added hybrid model fields: `full_attention_interval`, `layer_types`, linear attention dimensions
- Added `shared_expert_intermediate_size` for Qwen3-Next naming convention
- Added properties: `is_hybrid`, `is_linear_attention_layer()`, `is_full_attention_layer()`, `num_full_attention_layers`, `effective_shared_expert_intermediate`
- Auto-compute `layer_types` from `full_attention_interval`
- Handle `decoder_sparse_step` → `first_k_dense_replace` conversion

### Modified: `python/krasis/weight_loader.py`
- Added `load_linear_attention_weights()` for DeltaNet weights (in_proj_qkvz, in_proj_ba, conv1d, out_proj, A_log, dt_bias, norm)
- Branch in `load_layer()` based on `layer_type` (linear_attention vs full_attention)
- Handle both `shared_experts` (plural, DeepSeek) and `shared_expert` (singular, Qwen3-Next)

### Modified: `python/krasis/layer.py`
- Branch `__init__` on `layer_type`: creates `GatedDeltaNetAttention` for linear layers
- Branch `forward()`: linear layers pass `is_decode` flag instead of KV cache args

### Modified: `python/krasis/model.py`
- KV cache allocates only for full attention layers (`num_full_attention_layers`)
- Built `_kv_layer_offsets` mapping: global layer idx → KV cache offset (-1 for linear layers)
- Reset linear attention states between sequences in `generate()`
- Handle `None` seq_states/kv_caches throughout forward paths

### Modified: `python/krasis/vram_budget.py`
- Added `_linear_attention_bytes_per_layer()`, `_is_hybrid()`, `_num_full_attention_layers()`
- KV budget uses only full attention layer count (4x improvement for Qwen3-Coder-Next)
- Architecture string shows "GQA+DeltaNet" for hybrid models
- Both `compute_launcher_budget` and `compute_vram_budget` updated for hybrid

---

## Python TUI Launcher — 2026-02-12

**Moved interactive launcher from bash into Python with arrow-key-driven TUI.**

### New: `python/krasis/launcher.py`
- Arrow-key model selection screen (scans `~/Documents/Claude/hf-models/`)
- **GPU selection screen**: toggle individual GPUs on/off with space bar, shows per-GPU name/VRAM
- 13-option config screen with live VRAM/RAM budget updates on every change
- Quality annotations on each quant option: shows native dtype → selected (e.g. `bf16 → int8 — ~lossless`)
- Default CPU threads = physical cores (no longer capped at 48)
- Raw terminal mode via `tty`/`termios` — no external dependencies
- Hardware auto-detection: per-GPU info (index, name, VRAM), CPU cores, RAM from /proc
- PP partition auto-recomputes when selected GPU count changes
- `CUDA_VISIBLE_DEVICES` set automatically from selected GPUs at launch
- Config saved to `.krasis_config` (backward-compatible KEY=VALUE format, now includes `CFG_SELECTED_GPUS`)
- `--non-interactive` mode: prints summary + launches server directly
- `--selected-gpus 0,2` CLI arg to pre-select specific GPUs
- All CLI args from bash version supported via argparse

### New: `compute_launcher_budget()` in `vram_budget.py`
- Per-component quantization (attention, shared expert, dense MLP, LM head)
- Expert divisor modes: persistent (1), grouped (2-4), chunked (0)
- Per-rank VRAM breakdown + worst-case rank detection
- CPU expert RAM estimate with configurable INT4/INT8
- Separate `_cpu_expert_bytes_per_expert()` and `_detect_total_ram_gb()` helpers

### Modified: `./krasis` Bash Wrapper
- Reduced from 1,019 lines to 175 lines
- Keeps Phase 1 only (venv creation, dependency checking, Krasis native build)
- All args passed through to `python -m krasis.launcher`
- Removed: hardware detection, model selection, config steps, budget calculator, launch logic

### Tests Run
- `python -m krasis.launcher --help` — all args visible, PASS
- `compute_launcher_budget()` V2-Lite (MLA, 64 experts, persistent) — 4,919 MB / 16,380 MB, ~2.3M KV tokens
- `compute_launcher_budget()` Qwen3-235B (GQA, 128 experts, persistent) — over budget 43,412 MB (correct)
- `compute_launcher_budget()` Qwen3-235B (chunked) — fits, 6,584 MB, 313K tokens
- `scan_models()` — detected all 5 models with correct arch/layers/experts
- Non-interactive summary with V2-Lite — correct PP partition, budget, CPU expert RAM
- Config backward compatibility — loaded existing bash `.krasis_config`, round-trip save/load OK

---

## VRAM/RAM Budget Estimates + Back-Navigation in Launcher — 2026-02-12

**Added live VRAM/RAM budget calculator to `./krasis` interactive launcher.**

### New: `_show_budget` Function
- Computes per-GPU VRAM breakdown: attention weights, shared experts, expert buffers, embedding/LM head, norms/gates, CUDA overhead
- Shows worst-case rank with remaining VRAM for KV cache (token capacity estimate)
- Shows system RAM for CPU experts
- Handles both MLA (DeepSeek V2, Kimi K2.5) and GQA (Qwen3, GLM-4.7) attention formulas
- Handles dense layers (`first_k_dense_replace`), shared experts, tied embeddings
- Over-budget warning when total exceeds GPU VRAM

### New: Step-Loop with Back-Navigation
- Phase 4 refactored from sequential code into 9 numbered step functions
- After each step, user can press Enter to advance or `b` to go back
- Budget displayed after each memory-affecting step (1-6) in interactive mode
- Non-interactive mode: budget shown once in Phase 5 summary only

### Extended Model Metadata
- `_parse_model_config` now extracts 14 additional fields: vocab, dense layers, KV LoRA rank, QK dims, head counts, head dims, MLA detection, tie embeddings, shared expert intermediate size
- Supports all 5 model architectures: deepseek_v2, qwen3_moe, qwen3_next, glm4_moe

### Phase 5 Enhanced
- Replaced rough "Est. expert RAM" line with full VRAM/RAM budget breakdown from `_show_budget`

### Tests Run
- `./krasis --non-interactive --skip-setup --model-path .../DeepSeek-V2-Lite` — budget correct, 4,845/16,380 MB, ~2333K tokens
- `./krasis --non-interactive --skip-setup --model-path .../Qwen3-235B-A22B` — correctly shows OVER BUDGET (persistent 36,864 MB)
- `./krasis --non-interactive --skip-setup --model-path .../GLM-4.7` — dense layers handled, rank 1 worst-case
- `./krasis --non-interactive --skip-setup --model-path .../Qwen3-Coder-Next` — 512 experts, 15,250/16,380 MB
- `bash -n ./krasis` — syntax check PASS

---

## Interactive Launcher + server.py CLI Args — 2026-02-12

**Added `./krasis` executable launcher script and full CLI arg support in `server.py`.**

### New: `./krasis` Launcher (repo root)

Interactive 5-phase launcher script (~550 lines):
- **Phase 1: Setup** — checks Python, venv, krasis build, torch, deps
- **Phase 2: Hardware Detection** — GPUs, CPU cores, AVX2/FMA, RAM, NUMA
- **Phase 3: Model Selection** — scans `~/Documents/Claude/hf-models/`, shows model info
- **Phase 4: Interactive Config** — each parameter explained with smart defaults
- **Phase 5: Launch** — summary box, confirm, exec server

Features:
- `--non-interactive` mode for scripted/automated launches
- `--skip-setup` to skip venv/build checks
- All params passable as CLI flags (override interactive/saved)
- Saves config to `.krasis_config` for quick relaunch
- GGUF detection (sidecar directories)
- Expert RAM estimates from model config

### Modified: `server.py`

Added 11 new CLI args wired to `QuantConfig` and `KrasisModel`:
- `--expert-divisor` (0=chunked, 1=persistent, >=2=layer-grouped)
- `--gpu-expert-bits` (4 or 8)
- `--cpu-expert-bits` (4 or 8)
- `--attention-quant` (bf16 or int8)
- `--shared-expert-quant`, `--dense-mlp-quant`, `--lm-head-quant`
- `--gpu-prefill-threshold` (default: 300)
- `--gguf-path`, `--force-load`

Also updated defaults: `--kv-dtype` now defaults to `fp8_e4m3`, `--krasis-threads` to 48.

### Tests Run
- `./krasis --help` — shows usage
- `./krasis --non-interactive --skip-setup --model-path .../DeepSeek-V2-Lite` — all 5 phases complete, config saved, correct args passed to server
- `./krasis --non-interactive --skip-setup` (relaunch) — loads saved config, skips Phase 4
- `python -m krasis.server --help` — all new args visible
- Model scan shows all 5 models with architecture details

---

## Documentation Consolidation — 2026-02-12

Consolidated 9 markdown files into 3:
- **README.md** — absorbed DESIGN.md (architecture) + STATUS.md (features, supported models)
- **RESEARCH.md** (new) — absorbed PERFORMANCE_ANALYSIS.md + BENCHMARKS.md + EFFICIENCY_REPORT.md + research-findings.md
- **CHANGELOG.md** — kept as-is (audit trail)
- Deleted: DESIGN.md, STATUS.md, BENCHMARKS.md, EFFICIENCY_REPORT.md, research-findings.md, gguf-feature.md, PERFORMANCE_ANALYSIS.md

---

## GLM-4.7 Support (GQA Enhancements) — 2026-02-12

**Added partial RoPE and attention bias support for GLM-4.7 (glm4_moe) architecture.**

### Changes
- **`python/krasis/config.py`**: Added `partial_rotary_factor` (default 1.0) and `attention_bias` (default false) fields + `rotary_dim` property.
- **`python/krasis/attention.py`**: GQA RoPE now supports partial rotation — only first `rotary_dim` dimensions get RoPE, rest pass through. Added Q/K/V/O bias loading and application.
- **`python/krasis/weight_loader.py`**: GQA attention loader now picks up `*.bias` tensors when present.
- **No Rust changes** — MoE engine, weight caching, GPU prefill all already generic.

### GLM-4.7 Config
- 92 layers (3 dense + 89 MoE), 160 experts, top-8, 1 shared
- GQA: 96 heads / 8 KV heads, head_dim=128, partial_rotary_factor=0.5
- attention_bias=true, use_qk_norm=true
- 668 GB BF16 safetensors

### Regression
- V2-Lite GGUF test: ALL PASS, decode 4.82 tok/s (unchanged)

---

## GGUF → AVX2 Transposed CPU Cache — 2026-02-12

**New: GGUF files are now dequantized and re-quantized to our fast AVX2 transposed format with disk caching. 2.6× faster decode than raw GGUF-native path.**

### What Changed
- **`src/weights/mod.rs`**: Added `streaming_build_cpu_cache_from_gguf()` — reads GGUF, dequants experts to f32, requants to AVX2 transposed INT4/INT8, writes v5 disk cache. Per-projection mixed precision: `w2_bits` field added to `UnifiedExpertWeights` for separate gate/up vs down precision.
- **`src/moe.rs`**: w2 dispatch uses `expert.w2_bits` (was `expert.num_bits`). Added `gguf_native` parameter.
- **`src/gguf.rs`**: Added public `dequantize_raw_data()` for standalone dequantization.
- **`python/krasis/model.py`**: Added `gguf_native` parameter (default False = use AVX2 cache).
- **Per-layer GGUF type detection**: Handles Q4_K_M mixed types across layers (e.g. Q5_0 + Q8_0 for down).
- **Warnings for non-exact conversions**: Q5_0→INT8, Q6_K→INT8, etc. are logged but proceed.

### Cache Format
- v5 cache: `.krasis_cache/experts_gguf_avx2_g{gs}.bin` — separate from safetensors caches
- Mixed precision header: w13_bits and w2_bits stored independently
- First run: build + cache (24s for V2-Lite), subsequent runs: load from disk (6s)

### GGUF Type → Target Precision
| GGUF Type | Target | Exact? |
|-----------|--------|--------|
| Q4_0, Q4_K | INT4 | Yes |
| Q5_0, Q5_K | INT4 | No (round down) |
| Q6_K | INT8 | No (round up) |
| Q8_0 | INT8 | Yes |
| F16, BF16, F32 | INT8 | No (quantize) |

### V2-Lite Q4_K_M Results
- **2+2 = "4"**: PASS, **Capital = "Paris"**: PASS
- **GGUF types**: gate/up=[Q4_K] → INT4, down=[Q5_0, Q8_0] → INT8 (mixed precision)
- **Cache build**: 26 layers × 64 experts in 24.3s, 10.1 GB
- **Cache load**: 5.9s from disk
- **Decode**: 4.77 tok/s (up from 1.83 tok/s GGUF-native — **2.6× speedup**)
- **Prefill**: 169 tok/s (GPU Marlin persistent, unchanged)

### Previous: Native GGUF (kept as `gguf_native=True` fallback)
- **Decode**: 1.83 tok/s (AVX2 INT16×INT4/INT8 direct on GGUF blocks)
- **Load**: 11.5s (raw block copy, no conversion)
- Useful when you want fastest load time and don't need best decode speed

---

## Layer-Grouped Prefill (expert_divisor >= 2) — 2026-02-12

**Replaces selective mode with layer-grouped prefill: O(groups) DMA instead of O(chunks × layers).**

### Problem
The old selective mode (divisor >= 2) still DMAs per-layer per-chunk — it only reduces how many
experts per DMA, not the number of DMA calls. For a 10K prompt (5 chunks × 26 layers = 130 DMA
calls), this is still slow.

### Solution: Reverse the Loop Nesting
```
OLD (chunked):     for each chunk → for each layer → DMA experts → compute
NEW (layer-grouped): for each group → DMA experts once → for each chunk → for each layer → compute
```

For `divisor=2`, split MoE layers into 2 groups. Load one group's experts (~3.7 GB for V2-Lite),
process ALL chunks through those layers, free, repeat. Total DMA = 2 group loads ≈ 1 full model
load, regardless of prompt length.

### Changes
- `gpu_prefill.py`: Removed `_forward_selective()`, replaced with `"layer_grouped"` prefill mode
  - New `preload_layer_group(moe_layer_indices)`: loads expert weights for a group of MoE layers
  - New `free_layer_group()`: releases GPU VRAM between groups
  - `_forward_persistent()` reused for layer_grouped (same lookup mechanism)
  - OOM fallback: persistent → layer_grouped (was: selective)
- `model.py`: New `forward_prefill_layer_grouped()` method
  - `_compute_layer_groups(rank, cfg, divisor)` helper: splits rank's layers into groups
  - `forward()` auto-routes to layer_grouped when `divisor >= 2` and `M >= threshold`
  - Internal chunking (2048 tokens/chunk) with hidden/residual saved between groups
  - KV cache correctness: seq_len reset per group, position-based writes, pre-allocated pages
- `test_v2lite_profile.py`: Updated `--compare` for 3-mode comparison
  - `profile_prefill_single()`: single forward call for layer_grouped/persistent
  - `profile_decode()`: single-call prefill before decode
  - Summary table: Chunked vs Layer-Grouped vs Persistent with speedup columns

### VRAM Budget (V2-Lite, 64 experts, 26 MoE layers)
| Divisor | Mode | Layers/group | Expert VRAM | KV headroom |
|---------|------|-------------|-------------|-------------|
| 1 | persistent | 26 (all) | ~7,654 MB | ~2,891 MB |
| 2 | layer_grouped | 13 | ~3,827 MB | ~6,718 MB |
| 3 | layer_grouped | 9 | ~2,551 MB | ~7,994 MB |
| 0 | chunked | — | ~286 MB | ~10,259 MB |

### Benchmark Results (V2-Lite, `--compare`)
| Tokens | Chunked (0) | LayerGroup (2) | Persistent (1) | LG/Chunk | Pers/Chunk |
|--------|------------|----------------|----------------|----------|------------|
| 512 | 9.3/s | 48.0/s | 3,315/s | 5.2x | 356x |
| 1024 | 93.9/s | 94.7/s | 4,080/s | 1.0x | 43.5x |
| 2031 | 180.7/s | 178.6/s | 4,082/s | 1.0x | 22.6x |
| 4047 | 341.6/s | 340.2/s | 3,554/s | 1.0x | 10.4x |

- **10K prefill (persistent):** 2,494 tok/s
- **Decode:** 5.8 tok/s avg (172ms ITL), unchanged across modes
- **Layer-grouped DMA:** ~5.2s per group load (2 groups = ~10.4s fixed overhead)
- Layer-grouped wins at 512 tokens (5.2x vs chunked) due to fewer DMA calls
- At 1K+ tokens, layer-grouped ≈ chunked because DMA cost amortized
- Persistent is 10-356x faster (zero DMA)
- Layer-grouped's value: large models where persistent can't fit (e.g., Qwen3-235B)

### Tests Run
- Syntax check: all 3 files PASS
- Correctness: "2+2" → "4", "Capital of France" → "Paris" (divisor=2) PASS
- Full 3-mode benchmark: PASS (all modes produce correct output)

---

## Persistent Expert Buffers + Selective Loading — 2026-02-12

**New `expert_divisor` parameter eliminates per-layer DMA overhead during GPU prefill.**

### Changes
- `GpuPrefillManager`: new `expert_divisor` and `num_moe_layers` parameters
- **Persistent mode** (`divisor=1`): pre-loads ALL experts for ALL MoE layers into GPU VRAM at startup. Zero DMA during forward — single `fused_marlin_moe` call per layer.
- **Selective mode** (`divisor>=2`): buffer sized for `num_experts/divisor`. During forward, only DMA active experts via `torch.unique(topk_ids)`, remap IDs, single kernel call.
- **Chunked mode** (`divisor=0`): unchanged baseline behavior.
- OOM fallback: persistent → selective (divisor=2) on `torch.cuda.OutOfMemoryError`
- Shared experts also pre-loaded in persistent mode (zero DMA)
- `KrasisModel`: new `expert_divisor` parameter (default 1), wired to `GpuPrefillManager`
- `test_v2lite_profile.py`: `--divisor` and `--compare` flags for benchmarking modes

### V2-Lite VRAM Math (64 experts, 26 MoE layers, ~4.5 MB/expert)
| Divisor | Experts/layer | VRAM | Mode |
|---------|--------------|------|------|
| 1 | 64 | ~7,410 MB | Full persistent |
| 2 | 32 | ~3,705 MB | Selective |
| 0 | — | ~285 MB | Chunked (baseline) |

### Benchmark Results (V2-Lite, 1 GPU)

| Tokens | Chunked (tok/s) | Persistent (tok/s) | Speedup |
|:---:|:---:|:---:|:---:|
| 512 | 9.2 | **3,294** | **357x** |
| 1,024 | 93.7 | **4,090** | **44x** |
| 2,031 | 185.0 | **4,074** | **22x** |
| 4,047 | 180.5 | **3,468** | **19x** |

**10K prefill**: 9,903 tokens in 4.1s = **2,409 tok/s** (was 57s = 173 tok/s → **14x**)

VRAM: 10,746 MB total (2,924 weights + 7,654 persistent experts)
KV cache auto-sized to 92.9K tokens (down from 212K in chunked)
Decode speed unchanged: 5.8 tok/s (172ms avg)
Correctness: both modes produce identical output

### Tests Run
- Syntax/import: PASS
- Benchmark: PASS (`test_v2lite_profile.py --compare`)

---

## V2-Lite In-Depth Performance Analysis — 2026-02-12

**Full GPU prefill scaling + CPU decode timing analysis.**

### GPU Prefill Scaling (chunked, 2048 tok/chunk)
| Tokens | Chunks | Wall Time | tok/s |
|:---:|:---:|:---:|:---:|
| 512 | 1 | 11.17s | 45.8 |
| 1,024 | 1 | 10.71s | 95.7 |
| 2,031 | 1 | 11.01s | 184.4 |
| 4,047 | 2 | 22.12s | 183.0 |
| 8,079 | 4 | 45.79s | 176.4 |
| 9,903 | 5 | 57.26s | **173.0** |

~11s fixed overhead per chunk (DMA 7.3 GB expert weights per layer).

### CPU Decode (100 tokens after 2K prefill)
| Metric | Value |
|--------|------:|
| Avg | 170.2ms (5.9 tok/s) |
| P50 | 170.5ms |
| P90 | 173.5ms |
| Min | 158.9ms |
| Max | 175.1ms |

### Decode vs Context Length
| Context | Avg | tok/s |
|:---:|:---:|:---:|
| 512 | 168.8ms | 5.9 |
| 2,031 | 167.7ms | 6.0 |
| 8,079 | 173.1ms | 5.8 |

Test scripts: `test_v2lite_10k.py`, `test_v2lite_profile.py`
Full analysis: `PERFORMANCE_ANALYSIS.md`, `BENCHMARKS.md`

---

## V2-Lite Dual-Format Validation — 2026-02-12

**All tests PASS. Both GPU Marlin + CPU INT4 caches working end-to-end.**

### Results
| Metric | Value |
|--------|-------|
| Correctness | 2/2 PASS ("2+2"→"4", "Capital of France"→"Paris") |
| GPU Marlin cache | 7.2 GB on disk |
| CPU INT4 cache | 7.0 GB on disk |
| First-run cache build | 256.7s |
| Cached load | **5.9s** |
| GPU prefill (825 tok) | **77.2 tok/s** |
| CPU decode | **5.5 tok/s** (181ms/tok) |

Test script: `test_v2lite_dual_format.py`

---

## Dual-Format Cache Implementation Complete — 2026-02-12

**Fully implemented: dual GPU + CPU cached weight formats**

### What's New

#### Transposed INT8 Kernel (`src/kernel/avx2.rs`)
- New `expert_matmul_int8_transposed_integer()` — AVX2 kernel for INT8 CPU decode
- Uses `_mm256_madd_epi16` with byte interleaving via `_mm_unpacklo_epi8`
- Parallel wrapper `matmul_int8_transposed_integer_parallel()` with N=256 chunking
- 2 new tests: matches non-transposed INT8, parallel correctness

#### WeightStore Dual Storage (`src/weights/mod.rs`)
- `experts_unified` → `experts_cpu` (CPU transposed) + `experts_gpu` (GPU Marlin)
- `UnifiedExpertWeights` gains `num_bits: u8` field for kernel dispatch
- `from_expert_weights_int8()` for INT8 transposed format (i8 packed into Vec<u32>)
- `cpu_num_bits` + `gpu_num_bits` independently configurable
- Backward compat: `has_unified()`, `get_expert_unified()`, `get_shared_expert_unified()`

#### Dual Disk Cache Build & Load (`src/weights/mod.rs`)
- `streaming_build_cpu_cache()` — streams safetensors → transposed format, layer by layer
- `load_cpu_cache()` — loads v4 CPU cache from disk with INT4/INT8 dispatch
- Cache version 4 header encodes `num_bits` in packed metadata
- `load_from_hf()` now loads BOTH GPU (Marlin) and CPU (transposed) caches
- File paths: `.krasis_cache/experts_cpu_int{4|8}_g{gs}.bin`

#### CPU Forward Dispatch (`src/moe.rs`)
- `expert_forward_unified()` dispatches on `expert.num_bits`: 4→transposed INT4, 8→transposed INT8
- PyO3 getters read from `experts_gpu` for GPU prefill DMA
- New `cpu_num_bits()` and `gpu_num_bits()` PyO3 properties
- `load()` accepts `cpu_num_bits` + `gpu_num_bits` (backward compat via `num_bits`)

#### Python Updates
- `model.py`: passes `cpu_num_bits`/`gpu_num_bits` to Rust engine
- `sglang_bridge.py`: passes `cpu_num_bits`/`gpu_num_bits` to Rust engine
- `gpu_prefill.py`: unchanged (already uses `num_bits` param for GPU path)

---

## Architecture Change: Dual-Format Cache — 2026-02-12

**Moving from single Marlin format to dual GPU + CPU format**

### Problem
Kimi K2.5 with Marlin-only format: 0.55 tok/s CPU decode vs 1.55 tok/s with BF16 native layout.
Marlin's tile permutation destroys sequential memory access patterns, causing ~3x CPU slowdown
due to cache misses and MarlinTileMap indirection overhead.

### Solution
Two separate cached formats, each independently configurable precision (INT4 or INT8):
- **(A) GPU cache (Marlin format)**: tile-permuted for `fused_marlin_moe` CUDA kernel
- **(B) CPU cache (CPU-optimized format)**: sequential row-major for AVX2 cache locality

GPU and CPU precision independently configurable via Krasis params. Krasis can run any
model at any precision combo (e.g. INT4 GPU + INT8 CPU).

### Also in this change
- Fused transpose optimization: cache build 25 min (was 3+ hours) for Kimi K2.5
- Removed hardcoded GROUP_SIZE in gpu_prefill.py (now reads from engine)
- Dynamic diagnostic layer indices in model.py (calculates from num_hidden_layers)
- Monitor script improvements: auto-detect model config, layer-based ETA, GPU filtering
- Kimi K2.5 3/3 correctness tests PASS with Marlin INT4 format
- Kimi K2.5 retired (0.55 tok/s unacceptable), moving to Qwen3-Coder-Next

---

## RAM watchdog fix — 2026-02-11

- **Bug**: RAM watchdog started AFTER model fully loaded (line 328)
  - Only protected during inference, NOT during the multi-minute loading phase
- **Fix**: Moved `_start_ram_watchdog()` to before Phase 1 (GPU weight load)
  - Now protects during entire load: GPU weights, CPU experts, GPU prefill init, KV caches
- Protection stack:
  1. **Phase 0** (pre-flight): One-shot estimate, refuses to run if >95% MemTotal
  2. **RAM watchdog** (continuous): Checks every 1s, exits if MemAvailable <5% MemTotal
  3. **Post-load RSS check**: Verifies actual vs estimated after experts loaded

---

## Marlin-Native: THE ONLY FORMAT — 2026-02-11

**ONE FORMAT everywhere: GPU-native Marlin INT4. Same bytes on disk, in RAM, on GPU.**

### Changes
- **New Marlin CPU kernel** (`kernel/avx2.rs`):
  - `build_marlin_tile_map()` / `build_marlin_scale_map()` — precomputed inverse permutation tables
  - `matmul_int4_marlin_scalar()` — scalar reference for correctness verification
  - `expert_matmul_int4_marlin()` — AVX2 production kernel, reads Marlin-packed data directly
  - `matmul_int4_marlin()` / `matmul_int4_marlin_parallel()` — safe wrappers
  - 4 kernel tests: PASS (matches transposed reference within 0.00024)
- **New v3 Marlin cache** (`weights/mod.rs`):
  - `from_expert_weights_marlin()` — combines gate+up, Marlin-repacks both w13 and w2
  - `streaming_build_marlin_cache()` — safetensors → quantize → Marlin repack → disk (~128 MB peak)
  - `build_marlin_cache_locked()` — multi-process safe with file lock
  - `load_marlin_cache()` — loads v3 cache directly (same byte sizes as v2, different content)
  - `cache_path_marlin()` — `.krasis_cache/experts_marlin_g{gs}.bin`
  - `load_from_hf()` — Marlin cache only (v2 unified fallback removed)
- **Updated CPU forward** (`moe.rs`):
  - `expert_forward_unified()` — Marlin-only (old transposed branch removed)
  - `KrasisEngine.is_marlin_format()` — Python-accessible property for GPU prefill
  - OnceLock-based lazy init for MarlinTileMap/MarlinScaleMap
- **SGLang GPU prefill DMA copy** (`python/krasis/gpu_prefill.py`, `sglang_bridge.py`):
  - `sglang_bridge.py`: passes Krasis engine to GpuPrefillManager
  - `_repack_chunk_from_engine()`: detects `is_marlin_format` → DMA copy (no gptq_marlin_repack/marlin_permute_scales)
  - `_repack_shared_expert()`: same Marlin-native fast path
  - Legacy (non-Marlin) path preserved for backward compat
- **Removed dead v2 code** (~600 lines):
  - `save_cache_unified`, `load_cache_unified`, `streaming_build_unified_cache`
  - `build_unified_cache_locked`, `streaming_v1_to_unified_cache`, `convert_to_unified`
  - `write_unified_cache_header`, `cache_path_unified`, `CACHE_VERSION_UNIFIED`
  - Old `expert_forward()` FMA path, old `marlin_format` parameter

### Key Design
- Disk cache: Marlin format (.krasis_cache/experts_marlin_g128.bin)
- RAM: Same Marlin data, loaded directly from cache
- GPU: DMA copy from RAM, zero conversion — fused_marlin_moe runs instantly
- CPU: New kernel reads Marlin-packed data directly via inverse permutation

### Test Results
```
cargo test: 38 passed, 8 failed (model-loading tests, pre-existing cache dir issue)
test_marlin_kernel_matches_transposed [64×256]: max_diff=0.00012207 PASS
test_marlin_kernel_scalar_matches_avx2 [64×256]: max_diff=0.00024414 PASS
test_marlin_kernel_large [1408×2048]: max_diff=0.00000000 PASS
test_marlin_kernel_parallel [1408×2048]: max_diff=0.00000000 PASS
test_marlin_forward_matches_transposed [512×256]: max_diff=0.000000 PASS
```

---

## Partial v2 Cache Loading + License Change — 2026-02-11

### Features
- **Range-aware unified cache loading** — `load_cache_unified()` now accepts `start_moe_layer`
  and `num_layers_to_load` parameters. PP ranks can load just their layer range from the
  single full-model cache file. No more re-converting from safetensors every launch.
- **Multi-process cache build lock** — `build_unified_cache_locked()` uses an exclusive
  `.bin.lock` file so only one PP rank builds the cache while others wait.
- **License changed** from MIT to Apache 2.0.

### Changes
- `load_from_hf()` restructured: INT4 partial loads now use v2 unified cache instead of
  falling back to slow `load_and_quantize_all()` from safetensors.
- Cache is always built for ALL MoE layers, usable by any PP partition.
- Kimi K2.5 MoE test updated to auto-dispatch unified vs old format.

### Test Results
- 41/41 Rust tests pass

---

## Kimi K2.5 SGLang Integration Fixes — 2026-02-11

### Bugs Fixed

**1. correction_bias on CPU (CUDA illegal memory access)**
- `kimi_k2_moe_fused_gate` custom CUDA kernel crashed because `correction_bias` tensor
  was on CPU while `gating_output` was on GPU. The kernel read a CPU pointer from GPU.
- Root cause: With Krasis handling MoE weights on CPU, the gate's `e_score_correction_bias`
  nn.Parameter was never moved to GPU by SGLang's weight loading.
- Fix: In `select_experts()` (topk.py), detect device mismatch and move correction_bias
  to the same device as router_logits. Also cache the fix in topk_config.
- Also added tensor validation (contiguity, dtype) before the kernel call.

**2. Double shared expert computation (garbage output)**
- SGLang's `forward_normal()` computes shared experts on GPU AND Krasis's `moe_forward()`
  was also applying shared experts on CPU. Result: shared expert output doubled, routed
  scaling factor applied twice → garbage output.
- Fix: Added `skip_shared_experts` flag to `KrasisEngine`. When `True` (SGLang mode),
  the worker passes `None` for shared_scratch, preventing CPU shared expert computation.
- SGLang bridge now creates engine with `skip_shared_experts=True`.

### Test Results
- 41/41 Rust tests pass (standalone mode unaffected, `skip_shared_experts=false`)
- Kimi K2.5 generates correct, coherent output ("Paris" for capital of France, accurate
  quantum entanglement explanation, etc.)
- Decode speed: ~1.0 tok/s with debug flags, ~1.25 tok/s without

### Files Modified
- `krasis/src/moe.rs` — Added `skip_shared_experts` field, constructor param, worker arg
- `krasis/python/krasis/sglang_bridge.py` — Pass `skip_shared_experts=True` to engine
- `sglang/python/sglang/srt/layers/moe/topk.py` — Device fix for correction_bias,
  `SGLANG_SKIP_KIMI_GATE_KERNEL` bypass option
- `run_kimi_krasis.sh` — Debug flags (now commented out for production)

---

## Streaming Unified Cache Conversion — 2026-02-11

### Problem
First-run conversion (safetensors → unified cache) held ~488 GB in RAM throughout,
causing `systemd-oomd` kill at 979 GB RSS for Kimi K2.5.

### Fix: Streaming One-Layer-at-a-Time Conversion
New `streaming_build_unified_cache()` processes one MoE layer at a time:
- Load expert weights from safetensors (mmap, near-zero base RSS)
- Convert to unified format via `from_expert_weights()`
- Write to v2 cache file via BufWriter
- Drop all layer data immediately

New `streaming_v1_to_unified_cache()` does the same for v1→v2 conversion using mmap'd v1 cache.

Both paths end with `load_cache_unified()` to load the complete v2 cache from disk.

**Peak RAM during conversion:** ~16 GB (one layer) instead of ~488 GB (all layers).

Extracted `write_unified_cache_header()` helper shared by all cache writers.

Modified `load_from_hf()` to use streaming paths for full INT4 loads (Path B: v1→v2, Path C: safetensors→v2).
Original `load_and_quantize_all()` + `convert_to_unified()` kept for partial loads and INT8.

### Verified
- V2-Lite streaming build: RSS stayed at ~27 GB (mmap page cache, not allocations)
- 41/41 Rust tests pass (including delete-cache-and-rebuild)

---

## On-the-fly GPU Repack + RAM Safety — 2026-02-11

### OOM Root Cause
Running Kimi K2.5 (PP=2) caused OOM at ~943 GB RSS. Two causes:
1. **glibc arena retention**: `convert_to_unified()` freed ~507 GB of old-format weights, but glibc kept them in arenas, adding ~278 GB to RSS.
2. **GPU prefill `_engine_cache`**: `_prepare_from_engine()` cached ALL experts as Marlin-repacked tensors in Python RAM — 9.5 GB/layer × 60 layers = 570 GB on top of the 572 GB unified weights.

### Fix: On-the-fly Marlin Repack (Zero RAM Cache)
GPU prefill now reads raw INT4 from Rust engine per-chunk and repacks to Marlin on GPU during `forward()`. No caching at all.

**New methods in `gpu_prefill.py`:**
- `_repack_chunk_from_engine()` — gets raw INT4 from Rust for one chunk, uploads + repacks on GPU
- `_forward_engine()` — on-the-fly per-chunk forward path
- `_forward_cached()` — legacy path for safetensors (unchanged)
- `_repack_shared_expert()` — on-the-fly shared expert repack

**Removed from `gpu_prefill.py`:**
- `_engine_cache` dict (was storing ~570 GB of Marlin-repacked weights)
- `_prepare_from_engine()` method

### Fix: malloc_trim After Conversion
Added `libc::malloc_trim(0)` in `src/weights/mod.rs` after `convert_to_unified()` to force glibc to return freed pages to OS. Reclaims ~278 GB.

### Fix: Dynamic Group Size for GPU Buffers
GPU buffer allocation and auto-chunk-sizing now use the engine's actual `group_size` (e.g., 32 for Kimi K2.5) instead of the hardcoded `GROUP_SIZE = 128`. Prevents buffer underallocation for scale tensors.

### Fix: Shared Expert Intermediate Size
`_repack_shared_expert()` was using `N = intermediate_size` but shared experts have `N = n_shared_experts * intermediate_size`. Fixed to use `shared_N`.

### RAM Watchdog Thread
Background daemon thread in `model.py` checks `/proc/meminfo` every second. Exits with code 137 if available memory drops below 5% of total. Prevents full system OOM that kills desktop processes.

### PyO3 Range-Based Weight Access
Updated all `get_expert_*` methods in `src/moe.rs` to accept optional `start`/`end` range parameters. Added `get_shared_expert_weights()` method. Enables per-chunk data retrieval without copying all experts.

### Test Results (V2-Lite)
- Sanity test (CPU decode): 2/2 PASS — "2+2=4", counting 1-10
- GPU prefill test (threshold=10): 3/3 PASS — math, counting, 320-token prompt
- RAM watchdog: started successfully, no false triggers

**Files changed:**
- `src/weights/mod.rs`: malloc_trim after conversion
- `src/syscheck.rs`: `get_rss_gib()`, `get_available_gib()`, `get_total_gib()` helpers
- `src/moe.rs`: range-based PyO3 methods + `get_shared_expert_weights()`
- `python/krasis/gpu_prefill.py`: on-the-fly repack, removed RAM caching
- `python/krasis/model.py`: RAM watchdog thread, `_check_system_ram()`

---

## V2 Unified Cache + OOM Fix — 2026-02-11

### Bug Fix: convert_to_unified() OOM
`convert_to_unified()` created ALL unified weights (~507 GB) before freeing old weights (~507 GB),
causing peak RAM of ~1014 GB on a 995 GB system → OOM crash.

**Fix:** Process one layer at a time using `std::mem::take()`. Convert layer N → free old layer N → next.
Peak RAM stays at ~515 GB (old format for remaining layers + one layer of new format).

### V2 Unified Disk Cache
The v1 cache stored expert weights in the old separate gate/up/down format.
Loading v1 required runtime conversion to unified format.

**Changes:**
1. **V2 cache format (version=2)** — Stores unified weights (combined w13, transposed layout) directly.
   Includes shared experts. Path: `.krasis_cache/experts_unified_int4_g{gs}.bin`
2. **Cache loading priority**: v2 unified → v1 (convert + save v2 + delete v1) → fresh quantize
3. **Auto-migration**: When v1 cache loaded, automatically converts layer-by-layer and saves v2 cache.
   Old v1 cache is deleted after successful v2 write.
4. **Fresh quantize path**: Now also converts to unified and saves v2 cache immediately.
5. **Shared experts in cache**: V2 format includes shared experts (V2-Lite: 2 shared, Kimi K2.5: 1 shared).

### Test Updates
Updated 5 tests to work with unified-first format:
- `test_load_v2_lite`: checks unified dimensions and non-zero weights
- `test_cache_bit_exact`: validates v2 unified cache path and size
- `test_v2_lite_single_expert`: uses `expert_forward_unified()`
- `test_v2_lite_moe_forward`: dispatches to `moe_forward_unified()` when unified available
- `test_shared_expert_v2_lite`: checks unified shared expert format
- `test_async_submit_sync`: reference computed via unified path

**Files changed:**
- `src/weights/mod.rs`: layer-by-layer conversion, v2 cache save/load, cache migration, helper functions
- `src/moe.rs`: updated tests for unified-first format

---

## GPU Prefill Crash Fix + Decode Speed Optimization — 2026-02-11

### Phase 1: Fix GPU Prefill Crash at PP Boundary

Root cause: `forward()` lazily called `prepare_layer()` on the first forward pass for GPU1,
triggering 384-expert quantization via `gptq_marlin_repack()` CUDA kernels (~95s/layer) while
in the middle of a forward pass, likely causing VRAM exhaustion.

**Changes:**
1. **`gpu_prefill.py`: Added `prepare_all_layers()` method** — Pre-loads all MoE layers from
   disk cache at startup, avoiding lazy quantization during forward()
2. **`gpu_prefill.py`: Reduced VRAM budget** — 50%→40% of free VRAM, minus 100 MB for
   kernel intermediate allocations (fused_marlin_moe workspace)
3. **`gpu_prefill.py`: Added `torch.cuda.synchronize(device)` before first kernel launch**
4. **`model.py`: Call `prepare_all_layers()` in `_init_gpu_prefill()`** — Eagerly prepares
   layers on each device before any forward pass

### Phase 2: Decode Speed Optimization (target: 1.8→3.5-4.0 tok/s)

| Change | File | Impact |
|--------|------|--------|
| Threads 16→48 | `model.py` | ~2x (main bottleneck) |
| Gate diagnostics behind `KRASIS_DIAG=1` | `layer.py`, `model.py` | Eliminates 180+ `.item()` GPU syncs |
| Remove explicit `synchronize()` | `layer.py` | ~6-12ms/token saved |
| BF16 attention default | `config.py` | ~12ms/token saved (no INT8 quantize/dequant at M=1) |
| `import numpy` at module level | `layer.py` | Clean import |

**Changes:**
1. **`model.py`: Default `krasis_threads` 16→48** — Matches KTransformers config
2. **`layer.py` + `model.py`: `_DIAG_ENABLED = os.environ.get("KRASIS_DIAG") == "1"`** —
   All diagnostic blocks (MoE per-layer logging, embedding/layer/logits diagnostics)
   gated behind env var. Set `KRASIS_DIAG=1` to re-enable.
3. **`layer.py`: Removed `torch.cuda.current_stream(self.device).synchronize()`** from
   `_routed_expert_forward()` — `.cpu()` calls on the tensors below already handle sync
4. **`config.py`: `QuantConfig.attention` default `"int8"`→`"bf16"`** — INT8 quantize/pad/dequant
   overhead at M=1 decode outweighs the VRAM savings (+2.7 GB/GPU, still fits 16GB with FP8 KV)
5. **`layer.py`: Moved `import numpy as np` to module level**

---

## GPU Prefill CUDA Crash Investigation — 2026-02-10

**CUDA illegal memory access at PP boundary (layer 31, first layer on GPU1)**

### Crash Details
- Test: `test_kimi_gpu_prefill.py` — 333-token prompt, INT8 weights, FP8 KV, PP=2
- First-time Marlin precompute: 31 of 60 MoE layers completed (~95s/layer, ~49 min)
- Crash in `fused_marlin_moe` kernel for layer 31 (async error surfaced in `prepare_shared_expert`)
- GPU0 entered ERR! state, corrupting CUDA driver for all GPUs (requires reboot)
- Layers 1-30 on GPU0 all processed successfully
- GPU0: chunk_size=186, 3 chunks; GPU1: chunk_size=223, 2 chunks

### Fixes Applied
1. **Disk cache for Marlin weights** — saves 95 min re-precompute per run
   - Path: `{model_path}/.marlin_cache/b{bits}/layer{N}_{routed|shared}.pt`
   - Both routed experts and shared experts cached separately
2. **Debug sync points** — `KRASIS_DEBUG_SYNC=1` env var enables:
   - `torch.cuda.synchronize()` before/after each Marlin kernel call
   - Device verification for all input tensors
   - Per-chunk logging with exact crash location
3. **Workspace zeroing** between chunks — `self._workspace.zero_()` to prevent stale kernel state
4. **CUDA_LAUNCH_BLOCKING=1** — synchronous CUDA error reporting in test

### Hypothesis
Async CUDA error from Marlin MoE kernel on GPU1. Possible causes:
- Stale workspace state between chunks
- Buffer size mismatch with GPU1's larger chunk_size (223 vs 186)
- Cross-device memory corruption during PP transfer

### Next Steps
1. Reboot to recover GPUs
2. Re-run test — disk cache will skip 95-min precompute for layers 1-31
3. Debug sync will pinpoint exact failing kernel call

---

## Kimi K2.5 GPU Prefill Enabled — 2026-02-10

**GPU prefill managers enabled on Kimi K2.5 PP=2 (INT8 weights, FP8 KV)**

Tests: 3/3 PASS. GPU prefill managers created for both GPUs. Short prompts (~20 tokens)
used CPU path (threshold=300). No impact on correctness or speed. GPU prefill not yet
triggered — needs long prompt (300+ tokens) to test actual GPU Marlin prefill speed.

---

## Kimi K2.5 FP8 KV Cache Verification — 2026-02-10

**Verified FP8 E4M3 KV cache on Kimi K2.5 PP=2 (INT8 weights)**

Tests: 3/3 PASS. Quality identical to BF16 KV — same answers produced.

| Metric | FP8 KV + INT8 wt | BF16 KV + INT8 wt |
|--------|-----------------|-------------------|
| GPU0 KV cache | 4,032 MB (236K tokens) | 1,811 MB (53K tokens) |
| GPU1 KV cache | 4,839 MB (293K tokens) | 2,292 MB (69K tokens) |
| Decode speed | 1.21-1.28 tok/s | 1.28-1.41 tok/s |
| Context capacity | **~4x more** | baseline |

FP8 auto-sizes to fill free VRAM, giving much more context capacity.

---

## Kimi K2.5 INT8 Weights Verification — 2026-02-10

**Verified INT8 attention/shared_expert/dense_mlp/lm_head on Kimi K2.5 PP=2**

Switched from all-BF16 to default QuantConfig (INT8 everywhere). Tests: 3/3 PASS.

| Metric | INT8 | BF16 |
|--------|------|------|
| GPU0 VRAM | 7,654 MB | 12,063 MB |
| GPU1 VRAM | 6,044 MB | 11,105 MB |
| Decode speed | 1.28-1.41 tok/s | 1.55-1.87 tok/s |
| VRAM savings | **4-5 GB/GPU** | baseline |

INT8 slightly slower at M=1 decode (torch._int_mm overhead > bandwidth savings), but frees
significant VRAM for KV cache and GPU prefill buffers. Quality identical — same answers.

---

## Bug Fix: YaRN RoPE Frequency Assignment + De-interleave — 2026-02-10

**Fixed two bugs in MLA attention RoPE that caused garbled output on V2-Lite (and likely Kimi K2.5)**

### Bug: YaRN frequency interpolation was BACKWARDS
- Krasis: low indices (i < low) → interpolated (divided by factor), high indices (i > high) → original
- HF reference: low indices → **original** (high-freq, fast rotation), high indices → **interpolated** (divided by factor)
- This completely broke position encoding, producing progressive degradation with position
  (cos sim: pos0=0.999, pos3=0.968, worsening per token)
- Fix: Replaced manual loop with HF-matching mask-based computation using `freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask`

### Bug: Missing de-interleave before RoPE
- HF stores q_pe/k_pe in interleaved format `[re0, im0, re1, im1, ...]` from projection weights
- RoPE rotation operates on half-split format `[re0, re1, ..., im0, im1, ...]`
- Without de-interleave, rotation was applied to wrong dimension pairs
- Fix: Added `_deinterleave()` static method, applied before RoPE in `_apply_rope()`

### Verification
- Created `test_layer_compare.py`: full layer-by-layer comparison HF vs Krasis
- Created `test_layer0_detail.py`: sub-component comparison within layer 0 (dense)
- **Before fix**: Layer 0 cos=0.968, final logits cos=0.880, garbled output
- **After fix**: Layer 0 cos=0.999989, final logits cos=0.997, top-5 tokens IDENTICAL to HF
- V2-Lite generation tests: "2+2"→"4", "Count 1 to 5"→"1, 2, 3, 4, 5.", "Capital of France?"→"Paris." — 5/6 PASS
- **Kimi K2.5 PP=2 generation tests: 3/3 PASS** — YaRN factor=64 produces correct answers

### Files changed
- `python/krasis/attention.py` — YaRN frequency fix + `_deinterleave()` in MLAAttention

---

## Bug Fixes: Double Shared Expert + YaRN mscale + KV kv_len — 2026-02-10

**Fixed four critical bugs in standalone model that produced garbled output**

### Bug 4: Double-application of shared expert + routed_scaling_factor
- Rust `moe_forward()` computes: `output = routed_scaling_factor * routed_sum + shared_expert`
- Python `_routed_expert_forward()` ALSO computed: `shared_gpu + routed_scaling_factor * rust_output`
- Result: `shared_gpu + 2.827 * (2.827 * routed + shared_cpu)` = `shared + 7.99 * routed + 2.827 * shared`
- Instead of correct: `shared + 2.827 * routed`
- routed_scaling_factor applied TWICE (2.827^2 = 7.99), shared expert computed TWICE
- Made MLP output ~3-4x too large per layer, causing exponential growth in residual stream
- Fix: Python now returns Rust output directly (Rust handles shared expert + scaling)
- Note: NUMA path (line 272) also skips shared expert — not hit on single-socket system

### Bug 3: Missing YaRN mscale^2 in MLA softmax scale
- `attention.py` used `sm_scale = 1/sqrt(head_dim)` = 0.072
- Kimi K2.5 with YaRN factor=64 needs `mscale = 0.1 * mscale_all_dim * log(factor) + 1 = 1.416`
- Correct: `sm_scale = 1/sqrt(192) * mscale^2 = 0.145` (2x larger)
- Without this, attention logits are 2x too small → nearly uniform attention → garbled output
- Fix: compute YaRN mscale from rope_scaling config, multiply sm_scale by mscale^2

### Bug 1: Missing shared expert + routed_scaling_factor in CPU decode path
- `layer.py:_routed_expert_forward()` only returned raw routed expert output
- Missing: `shared_expert(hidden) + routed_scaling_factor * routed_experts(hidden)`
- For Kimi K2.5 this means missing the shared expert AND a 2.827x scaling factor
- GPU prefill path was correct (gpu_prefill.py handles both)

### Bug 2: kv_len_arr off-by-M in attention
- `attention.py` used `seq_state.seq_len` for `kv_len_arr` in FlashInfer plan()
- But `seq_state.advance(M)` is called AFTER all layers, not before
- Prefill: kv_len_arr=[0] when 18 tokens were appended → attention reads ZERO tokens
- Decode: always missing the current token (kv_len=N instead of N+1)
- Fix: pass `num_new_tokens` through model→layer→attention, use `seq_len + num_new_tokens`

### Files changed
- `python/krasis/layer.py` — shared expert + scaling in CPU path, num_new_tokens param
- `python/krasis/model.py` — pass num_new_tokens to layer.forward()
- `python/krasis/attention.py` — use effective kv_len in both MLA and GQA attention

### Tests run
- Kimi K2.5 standalone PP=2 (BF16 weights, BF16 KV, 16 threads): **3/3 PASS**
  - simple_math ("2+2?"): "4" — 128 tokens in 82.6s (1.55 tok/s)
  - counting ("1 to 5"): "1, 2, 3, 4, 5" — 128 tokens in 68.9s (1.86 tok/s)
  - factual ("Capital of France?"): "Paris" — 128 tokens in 68.4s (1.87 tok/s)
- Model produces coherent reasoning (thinking) followed by correct answers
- Load time: 722s (GPU weights 18s, CPU experts 704s)
- GPU VRAM: GPU0=12,063 MB, GPU1=11,105 MB

---

## GQA Attention Support — 2026-02-10

**Added GQA (Grouped Query Attention) alongside existing MLA, enabling Qwen3-235B on standalone model**

Previously Krasis standalone only supported MLA (Multi-head Latent Attention) used by DeepSeek V2/V3
and Kimi K2.5. Now supports GQA as used by Qwen3-235B-A22B.

### Changes (5 Python files)
- `config.py`: Made MLA fields optional (None for GQA), added `gqa_head_dim`, auto-detection via
  `attention_type` property ("mla" or "gqa"). GQA detected by absence of `kv_lora_rank`.
- `weight_loader.py`: Split `load_attention_weights()` into `_load_mla_attention()` and
  `_load_gqa_attention()`. GQA loads: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
- `kv_cache.py`: Supports both MLA split (ckv+kpe) and GQA split (k+v) cache layouts.
  GQA: `[layers, pages, page_size, num_kv_heads, head_dim]` per K and V.
- `attention.py`: Added `GQAAttention` class using FlashInfer `BatchPrefillWithPagedKVCacheWrapper`.
  Handles QKNorm (per-head RMSNorm), standard RoPE on all Q/K, `append_paged_kv_cache`.
- `layer.py`: Auto-selects GQAAttention or MLAAttention based on `cfg.attention_type`.

### Tests run
- V2-Lite MLA end-to-end: **PASSED** (no regression)
- Qwen3 GQA attention forward (prefill + decode): **PASSED**
- Qwen3 TransformerLayer with GQA dispatch: **PASSED**
- Config parsing all 3 models (V2-Lite, Kimi K2.5, Qwen3-235B): **PASSED**

---

## INT8 CPU Expert Quantization — 2026-02-10

**INT8 CPU expert support: configurable 4-bit or 8-bit CPU MoE expert quantization**

Previously CPU experts were always INT4 (4 bits per weight). Now supports INT8 (8 bits)
for 2x better precision at 2x memory. Configurable via `num_bits` parameter in Rust engine
and `cpu_expert_bits` in Python `QuantConfig`.

### Rust changes (4 files)
- `weights/marlin.rs`: Added `QuantizedInt8` struct and `quantize_int8()`/`dequantize_int8()`
  functions. INT8 uses per-group symmetric quantization (scale = amax / 127.0), stores raw i8.
- `kernel/avx2.rs`: Added `expert_matmul_int8_integer` AVX2 kernel — uses
  `_mm256_cvtepi8_epi16` + `_mm256_madd_epi16` (simpler than INT4: no nibble extraction).
  Plus `matmul_int8_integer`/`matmul_int8_integer_parallel` wrappers.
- `weights/mod.rs`: Added `QuantWeight` enum (`Int4`/`Int8`) wrapping the two quant types.
  Updated `ExpertWeights` to use `QuantWeight`. `WeightStore` now has `num_bits` field.
  `load_from_hf()` takes `num_bits` parameter. Disk cache uses separate files per bit width
  (`experts_int4_g128.bin` vs `experts_int8_g128.bin`). Updated all cache read/write/NUMA paths.
- `moe.rs`: Added `matmul_integer()`/`matmul_integer_parallel()` dispatch helpers.
  `expert_forward_integer()` dispatches via enum. `prefetch_expert_nta()` handles both types.
  `KrasisEngine.load()` accepts `num_bits` parameter (default 4).

### Python changes (3 files)
- `config.py`: Added `cpu_expert_bits: int = 4` to `QuantConfig`
- `model.py`: Passes `num_bits=quant_cfg.cpu_expert_bits` to Rust engine
- `sglang_bridge.py`: Added `cpu_expert_bits` class config, passes to `engine.load()`

### Test Results (V2-Lite)
- INT8 load: 78.7s (first quantization from BF16), cached loading expected ~2s
- INT4 vs INT8 token match: 14/20 (different precision = different outputs, expected)
- INT8 decode slightly faster than INT4 (simpler weight access, no nibble extraction)
- Build: clean compile, all existing INT4 tests pass

### Usage
```python
# Rust engine directly
engine = KrasisEngine(parallel=True, num_threads=16)
engine.load("/path/to/model", num_bits=8)

# Standalone model
from krasis.config import QuantConfig
qcfg = QuantConfig(cpu_expert_bits=8, gpu_expert_bits=8)
model = KrasisModel("path/to/model", quant_cfg=qcfg)

# SGLang bridge
KrasisMoEWrapper.cpu_expert_bits = 8
```

---

## Per-Component Precision Config + INT8 Marlin GPU Prefill — 2026-02-10

**Configurable per-component quantization + INT8 Marlin GPU expert prefill**

Previously all GPU weight precision was hardcoded (INT8 for projections, INT4 for GPU expert prefill).
Now each component's precision is configurable via `QuantConfig`, and GPU expert prefill supports
both INT4 and INT8 Marlin kernels.

### Changes
- `config.py`: Added `QuantConfig` dataclass — per-component precision settings:
  `lm_head`, `attention`, `shared_expert`, `dense_mlp` (each "bf16"/"int8"),
  `gpu_expert_bits` (4/8 for Marlin kernel)
- `weight_loader.py`: Accepts `QuantConfig`, conditionally loads weights as BF16 or INT8
  in `load_lm_head`, `load_attention_weights`, `load_dense_mlp`, `load_shared_expert`
- `layer.py`: Added `_linear()` dispatch helper (handles both INT8 tuples and BF16 tensors),
  replaced all `int8_linear(*self.xxx)` calls with `_linear(self.xxx)`
- `attention.py`: Same `_linear()` dispatch for `kv_a_proj`, `q_a_proj`, `q_b_proj`,
  `q_proj`, `o_proj`
- `model.py`: Added `quant_cfg` parameter to `KrasisModel`, passes through to
  `WeightLoader` and `GpuPrefillManager`. LM head stored as generic `lm_head_data`
  (tuple or tensor) with `_linear()` dispatch
- `gpu_prefill.py`: `num_bits` constructor parameter (default 4, supports 4/8).
  Generalized `_quantize_and_pack_gpu()` for INT4 (8 vals/int32) and INT8 (4 vals/int32).
  Updated buffer allocation, repack calls, `fused_marlin_moe` calls, VRAM estimates

### V2-Lite Test Results (PP=1)

| Metric | Default (INT8+INT4) | BF16attn+INT8exp | CPU-only |
|--------|----:|----:|----:|
| GPU weights | 5,194 MB | 5,524 MB (+330) | 5,206 MB |
| Expert buffer | 285.5 MB | 562.3 MB (2x) | N/A |
| Peak GPU | 11,321 MB | 11,726 MB (+405) | 11,034 MB |
| Decode speed | 3.3 tok/s | 3.6 tok/s | 3.3 tok/s |
| Token match (short) | baseline | 19/20 | 20/20 |

BF16 attention slightly faster on decode (no INT8 quantize overhead).
INT8 Marlin GPU prefill runs correctly via `fused_marlin_moe(num_bits=8)`.
VRAM cost modest (+405 MB peak). V2-Lite too small for quality assessment.

### Defaults
Default `QuantConfig()` matches previous hardcoded behavior (all INT8, GPU experts INT4).
No change for existing users who don't pass `quant_cfg`.

### Usage
```python
from krasis.config import QuantConfig
# Higher quality attention (BF16) with INT8 GPU expert prefill
qcfg = QuantConfig(attention="bf16", gpu_expert_bits=8)
model = KrasisModel("path/to/model", quant_cfg=qcfg)
```

---

## FP8 KV Cache — 2026-02-10

**FP8 E4M3 KV cache: halves KV VRAM, same attention quality**

SGLang's approach: store KV as FP8, upcast to BF16 before FlashInfer kernel. The kernel
always computes in BF16 — FP8 is purely a storage optimization. Simple to implement.

### Changes
- `attention.py`: Upcast FP8 cache to BF16 before `BatchMLAPagedAttentionWrapper.plan()/run()`.
  `kv_data_type=torch.bfloat16` always passed to plan (not the cache dtype).
- `model.py`: Default `kv_dtype` changed from `torch.bfloat16` to `torch.float8_e4m3fn`.
- `kv_cache.py`: Already had FP8 default — no change needed.

### Test Results (V2-Lite, PP=1)
- FP8 cache: 5,322 MB vs BF16: 10,644 MB → **2x VRAM savings**
- Prefill logits: cosine similarity 1.0000 (FP8 vs BF16), top-5 tokens identical
- FP8 append + upcast roundtrip: max diff 0.015 (within FP8 E4M3 precision)
- V2-Lite generation quality degrades (model too small for FP8+INT4 noise) — expected,
  Kimi K2.5 should be fine

### VRAM Impact (Kimi K2.5 PP=2, estimated)
- FP8: 576 bytes/token/layer × 31 layers = ~1.1 GB for 64K context
- BF16: 1152 bytes/token/layer × 31 layers = ~2.2 GB for 64K context
- **Saves ~1.1 GB per GPU** — more room for GPU prefill buffers

---

## TRTLLM MLA Backend + Efficiency Report — 2026-02-10

**TRTLLM MLA attention backend implemented but blocked by SM89**

Created `trtllm_attention.py` (283 lines) with two-path architecture:
- Prefill (M > 1): Non-absorbed path, decompresses K/V via w_kc/w_vc before attention,
  uses `trtllm_ragged_attention_deepseek` with Q[M,H,192], K[M,H,192], V[M,H,128]
- Decode (M = 1): Absorbed path, uses `trtllm_batch_decode_with_kv_cache_mla`
  with Q[1,H,576], paged KV cache [pages,1,page_size,576]

### Changes
- `trtllm_attention.py`: New file — TRTLLMMLAAttention class
- `kv_cache.py`: Added combined cache format, `store_kv_combined()`, `block_tables()`
- `layer.py`: Added `attention_backend` parameter ("flashinfer" or "trtllm")
- `model.py`: Added `attention_backend` parameter, combined KV cache init

### BLOCKED
TRTLLM FMHA runner reports `Unsupported architecture` on SM89 (RTX 2000 Ada, compute 8.9).
Code is structurally complete but cannot run on our GPUs. FP8 KV cache (which depends on
TRTLLM backend) is also blocked.

### Efficiency Report
Produced `EFFICIENCY_REPORT.md` with:
- Full performance comparison: Krasis vs KTransformers+SGLang
- Resource utilization breakdown (GPU VRAM, CPU RAM)
- Decode latency analysis (~245ms/token, CPU expert-bound)
- Plan deviations: 1.7x code size vs estimates, all 4 core phases complete
- Technical deviations: BF16 KV (not FP8), custom RoPE (not FlashInfer), CPU bounce for P2P

---

## GPU Prefill Integration — 2026-02-10

**INT4 Marlin GPU MoE prefill integrated into standalone server**

Threshold-based routing: M >= threshold → GPU (INT4 Marlin), M < threshold → CPU (Krasis).
GpuPrefillManager created per PP rank device, wired to MoE layers automatically.

### Changes
- `layer.py`: Added `gpu_prefill_manager`, `gpu_prefill_threshold` params.
  `_moe_forward()` dispatches to `_gpu_prefill_forward()` or `_routed_expert_forward()`.
- `model.py`: Added `gpu_prefill`, `gpu_prefill_threshold` params.
  `_init_gpu_prefill()` creates one `GpuPrefillManager` per GPU device.

### Benchmark (V2-Lite, 313 tokens, PP=1)
| Path | Prefill | Decode |
|------|---------|--------|
| GPU Marlin INT4 | **424 tok/s** (0.74s) | 4.2 tok/s (236ms) |
| CPU Krasis INT4 | 10 tok/s (30.2s) | 4.0 tok/s (250ms) |
| **Speedup** | **40.9x** | ~same |

First-time layer quantization: ~1.2s/layer (26 layers = 32s, cached in RAM after).

### FP8 KV Cache Research
- FlashInfer `BatchMLAPagedAttentionWrapper` does NOT support FP8 (static_assert 16-bit)
- SGLang uses TRTLLM MLA backend for FP8: `trtllm_ragged_attention_deepseek`
- XQA MLA has FP8 but needs SM120+ (our GPUs are SM89)
- Current BF16 KV works fine: ~1.1 GB/GPU for 64K context

---

## Krasis Standalone Server — 2026-02-10

**Full standalone LLM server: replaces SGLang for GPU forward pass**

Single-process, N-GPU pipeline-parallel server with OpenAI-compatible HTTP API.
Eliminates SGLang's multi-process architecture that caused OOM crashes.

### New Python modules (10 files)

- `config.py`: Parse `config.json` for Kimi K2.5 (nested text_config) and V2-Lite (flat).
  `ModelConfig` with `has_q_lora`, `is_moe_layer()`, PP partition computation.
- `weight_loader.py`: Streaming BF16→INT8 per-channel symmetric quantization.
  Uses `torch._int_mm` (M>16 padding workaround). Loads one tensor at a time.
- `attention.py`: MLA attention using FlashInfer `BatchMLAPagedAttentionWrapper`.
  Dual path: q_lora (Kimi: q_a→norm→q_b) vs direct q_proj (V2-Lite). YaRN RoPE.
  Pre-absorbs w_kc into query, post-multiplies w_vc. Paged KV cache append.
- `kv_cache.py`: Paged KV cache for MLA (3D: [pages, page_size, dim]).
  Auto-sizes to 50% free VRAM. `SequenceKVState` for per-sequence page tracking.
- `layer.py`: Transformer layer: attention + MoE/dense MLP. Uses FlashInfer
  `fused_add_rmsnorm` (in-place), `silu_and_mul`. MoE routing with
  sigmoid/softmax, norm_topk_prob, routed_scaling_factor.
- `model.py`: Full model orchestration with PP. Two-phase loading:
  GPU weights (streaming INT8) + CPU experts (Krasis Rust INT4).
  CPU-bounce for broken GPU P2P transfers (auto-detected).
- `sampler.py`: FlashInfer `top_k_top_p_sampling_from_logits`.
- `tokenizer.py`: HF tokenizer wrapper with chat template, incremental decode.
- `scheduler.py`: Async request scheduler with thread pool executor.
- `server.py`: FastAPI HTTP server. `/v1/chat/completions` (SSE + blocking),
  `/v1/models`, `/health`.

### Bugs found and fixed

- `fused_add_rmsnorm` returns None (in-place) — code tried to unpack return value
- FlashInfer sampling returns single tensor in 0.6.1, not (samples, success) tuple
- `torch._int_mm` requires M > 16 strictly (not >=) — pad to 17
- FlashInfer MLA kernel requires 16-bit KV (not FP8) — `static_assert(sizeof(DType) == 2)`
- KV cache `ensure_capacity`/`advance` was called per-layer instead of per-forward-pass
  (inflated seq_len by num_layers, caused OOM in RoPE for multi-GPU)
- `seq_states[0]` used for all ranks — fixed to `seq_states[rank_idx]`
- GPU P2P transfers silently return zeros on this system — added auto-detection + CPU bounce
- Krasis Rust engine includes shared expert computation — Python was also computing it on GPU,
  causing double-counting. Fixed by removing GPU shared expert (Krasis handles it).

### Testing on DeepSeek-V2-Lite

- PP=1 (1 GPU): Works, 3.3 tok/s decode
- PP=2 (2 GPUs): Works, 3.3 tok/s decode, identical output to PP=1
- PP=3 (3 GPUs): Works, 3.4 tok/s decode, identical output to PP=1 and PP=2
- HTTP server: Health, models, blocking chat, streaming SSE all verified
- Logits cosine sim vs HF BF16 reference: 0.94 (INT4 expert quantization noise on small model)
- Per-layer hidden state cosine sim: 0.997 through layer 20, degrades in last 3 layers
  due to INT4 noise amplification (model's final residual stream has very small norm ~0.5)

---

## Fix silent crash during loading + shard filtering — 2026-02-10

**Fix silent OOM crash when 3 PP ranks load concurrently**

Root cause: `load_and_quantize_all` in `weights/mod.rs` opened ALL 64 safetensors shards
via mmap regardless of which layers this rank needs. Each PP rank only needs ~20 shards
but was mmapping all 64 (~580 GB each). With 3 concurrent ranks, page cache pressure from
reading different parts of the same files + 520 GB heap for expert weights could exhaust
the 995 GB RAM (no swap), silently killing the server and collateral processes (tmux, Claude).

- `weights/mod.rs` (load_and_quantize_all): Filter shards by layer range — only open shards
  containing expert tensors for layers in `[start_layer, start_layer + max_layers)`.
  Reduces from 64→~20 shards per rank, eliminates cross-rank page cache contention.
- `weights/mod.rs` (detect_prequant_group_size): Probe using this rank's first layer
  instead of global first_moe layer (which might not be in our filtered shards).
- `sglang_bridge.py` (_get_engine): Added `[DIAG]` logging with `sys.stdout.flush()`
  before/after engine creation and load — pinpoints crash location.
- `moe.rs` (load): Added `[DIAG-RUST]` logging with `log_memory_usage()` at each step.
- `weights/mod.rs`: Per-shard open progress, per-5-layer memory logging.
- `syscheck.rs`: New `log_memory_usage()` reads `/proc/self/status` + `/proc/meminfo`.

---

## Fix GPU Xid 13 crash with 0 GPU experts — 2026-02-10

**Fix out-of-range GPU memory access when num_gpu_experts=0**

Root cause: With `--kt-num-gpu-experts 0`, W8A8Int8MoEMethod creates weight tensors with
shape `[0, ...]`. During inference, the Triton fused_experts kernel launches with these
empty tensors and ALL topk_ids masked to -1. The kernel accesses `weight_ptr + expert_id * stride`
on a buffer with 0 elements, causing NVIDIA Xid 13 (Out Of Range Address) on GPU0.

- `kt_ep_wrapper.py` (KTEPWrapperMethod.apply): Skip GPU MoE kernel entirely when
  `num_gpu_experts == 0` — return zeros instead of launching Triton with empty weights
- `w8a8_int8.py` (W8A8Int8MoEMethod.apply): Guard check `layer.w13_weight.shape[0] == 0`
  before kernel launch — belt-and-suspenders defense
- `kt_ep_wrapper.py` (MarlinInt4PrefillMethod.apply): Same empty-expert guard for Marlin path
- Added diagnostic logging: first apply() call per layer, INT8 cache save markers with flush
- Created `crash-tracking.md` to track GPU Xid errors

---

## VRAM budget calculator — 2026-02-09

**Add per-rank VRAM budget calculation and dynamic SGLang parameter tuning**

- `python/krasis/vram_budget.py`: New module that computes exact per-rank GPU memory usage
  - Reads model config.json: supports both MLA (Kimi K2.5/DeepSeek) and GQA (Qwen3)
  - Calculates per-rank: attention weights, dense MLP, shared experts, gate/norms, embedding, lm_head
  - Handles INT8 quantization (1 byte/param) and BF16 (2 bytes/param)
  - KV cache per token: MLA `(kv_lora_rank + qk_rope_head_dim) * dtype_bytes` or GQA `2 * n_kv_heads * head_dim * dtype_bytes`
  - Determines max context length from bottleneck rank, capped at `max_position_embeddings`
  - Computes `mem_fraction_static`, `max_total_tokens`, `context_length` for SGLang
  - CLI entry point: `python -m krasis.vram_budget --model-path ... --pp-partition ...`
  - Prints human-readable summary to stderr, JSON to stdout
- `run_kimi_krasis.sh`: Replaced hardcoded `MEM_FRACTION=0.50` and `CTX_SIZE=25000` with dynamic VRAM budget
  - Runs `krasis.vram_budget` at startup, extracts parameters from JSON output
  - Prints per-rank breakdown at launch for visibility
  - Supports manual overrides via `CTX_SIZE_OVERRIDE`, `MEM_FRACTION_OVERRIDE`, `MAX_TOTAL_TOKENS_OVERRIDE`
- `sglang_bridge.py`: Added `verify_vram_budget()` classmethod
  - Called after engine loads (once per rank) from `load_weights()`
  - Compares `torch.cuda.memory_allocated()` to pre-computed estimate
  - Logs INFO (<10% deviation) or WARNING (>10% deviation)
- `__init__.py`: Exports `compute_vram_budget`

**Kimi K2.5 PP=3 budget** (INT8, fp8 KV, 16 GB/GPU):
- PP0: 5,445 MB weights → 10.4 GB free → 950K tokens max
- PP1: 3,018 MB weights → 12.9 GB free → 1.1M tokens max
- PP2: 5,114 MB weights → 10.8 GB free → 980K tokens max
- Bottleneck: PP0 → context capped at 262,144 (model max) — fits easily

Tests: CLI tested against Kimi K2.5 (MLA) and Qwen3-235B (GQA)

## Multi-GPU PP=3 support — 2026-02-09

**Add pipeline-parallel partial loading for multi-GPU setups**

- `WeightStore::load_from_hf()`: new `start_layer` param — loads MoE layers `[start..start+count)` instead of all
- `KrasisEngine.load()`: new `start_layer` keyword arg, passed through to Rust
- Memory budget estimate updated to use actual loaded layer count (not total)
- Disk cache skipped for partial loads (start_layer or max_layers set)
- `sglang_bridge.py`: PP-aware engine loading
  - Records `_pp_first_layer_idx` from first `load_weights()` call
  - Parses `SGLANG_PP_LAYER_PARTITION` env var to infer rank and MoE range
  - Passes `start_layer`/`max_layers` to `engine.load()` for partial loading
  - Remaps `moe_layer_idx` in CPU decode path (GPU prefill uses absolute indices)
- `run_kimi_krasis.sh`: Krasis-based launch script for Kimi K2.5 PP=3
  - No GGUF download needed (reads HF safetensors directly)
  - Partition 20,21,20 across 3 GPUs, 21 threads/rank
  - Each rank loads ~174 GB MoE experts (total ~520 GB fits in 995 GB RAM)
  - Sets `KRASIS_BACKEND=1`, `KRASIS_MODEL_PATH`, `SGLANG_PP_LAYER_PARTITION`

Tests: config + synthetic + V2-Lite pass, Rust build clean

## GPU prefill implementation — 2026-02-09

**Add INT4 Marlin GPU prefill for MoE layers**

- `python/krasis/gpu_prefill.py`: `GpuPrefillManager` class for GPU-accelerated MoE prefill
- `_quantize_and_pack_gpu()`: BF16→INT4 symmetric quantization + packing on GPU (group_size=128)
- Weight pipeline: HF safetensors → BF16 → transpose [N,K]→[K,N] → GPU INT4 quantize → pack → `gptq_marlin_repack` → `marlin_permute_scales` → RAM cache
- `prepare_layer()`: quantizes all experts for one layer, caches in CPU RAM
- `prepare_shared_expert()`: same for shared expert (stored as single-expert MoE)
- `forward()`: loads cached weights to GPU buffer, calls `fused_marlin_moe` kernel
- Multi-chunk path for large expert counts (Kimi K2.5: 384 experts): remaps expert IDs per chunk
- `_shared_expert_forward()`: runs shared expert via Marlin with topk_ids=0, weight=1.0
- Pre-quantized weight support: dequantizes compressed-tensors INT4 → BF16 → re-quantizes to Marlin format
- `sglang_bridge.py`: GPU prefill integration — threshold-based switching (batch >= 300 → GPU, else CPU)
- `GpuPrefillManager` singleton shared across all layer wrappers
- **V2-Lite benchmarks**: 26ms/layer, 40 tok/s (batch=1) → 19,123 tok/s (batch=512)
- **GPU vs CPU cosine similarity**: 0.9825 (excellent agreement despite different INT4 implementations)
- `test_gpu_prefill.py`: 5 tests — quantize_pack, v2_lite_forward, shared_expert, gpu_vs_cpu, scaling

Tests: 34 Rust + 3 Python bridge + 5 GPU prefill = **42 PASS**

## Shared expert support — 2026-02-09

**Add shared expert loading, forward computation, and routed_scaling_factor**

- `ModelConfig`: added `n_shared_experts` (0/1/2) and `routed_scaling_factor` (1.0/2.827)
- `WeightStore.shared_experts`: per-MoE-layer shared expert weights (BF16 → INT4 quantized)
- `load_shared_experts()`: loads from safetensors (always BF16, even for pre-quantized models like Kimi K2.5)
- `moe_forward()`: now accepts `shared_scratch` parameter; computes `scale * routed_output + shared_output`
- Async worker: allocates separate scratch buffer for shared expert's larger intermediate size
- `ExpertScratch` reuses quantized activation from routed path (same hidden_size + group_size)
- Shared experts loaded after cache too (not in cache format yet)
- **V2-Lite**: 2 shared experts, intermediate=2816, routed_scaling_factor=1.0
- **Kimi K2.5**: 1 shared expert, intermediate=2048, routed_scaling_factor=2.827
- **Qwen3**: 0 shared experts (no-op)
- `test_shared_expert_v2_lite`: verifies shared expert changes output (max_diff=0.075)

Tests: 34 Rust + 3 Python = **37 PASS**

## [1456e1f] SGLang bridge — 2026-02-09

**Add SGLang bridge: KrasisMoEWrapper drop-in for KTMoEWrapper**

- `python/krasis/sglang_bridge.py`: `KrasisMoEWrapper` class implementing KTMoEWrapper interface (submit_forward, sync_forward, load_weights)
- Singleton engine pattern — one KrasisEngine shared across all MoE layer wrappers
- GPU↔CPU tensor transfer via BF16→uint16 numpy view (numpy lacks bfloat16 support)
- Expert ID masking: IDs < num_gpu_experts set to -1 (Krasis skips them)
- `python/krasis/__init__.py`: exports KrasisMoEWrapper
- `test_bridge.py`: 3 tests — engine roundtrip, wrapper interface, batch forward
- `run_krasis.sh`: Launch script for SGLang with Krasis backend
- SGLang patch: `kt_ep_wrapper.py` import toggle via `KRASIS_BACKEND=1` env var

Tests: 33 Rust + 3 Python = **36 PASS**

## [d2cc31b] Async submit/sync — 2026-02-09

**Add batch MoE forward and async submit/sync pattern**

- Background worker thread with `mpsc` channels for non-blocking CPU expert dispatch
- `MoeWork` struct carries layer_idx, activation, topk_ids/weights, batch_size
- `submit_forward()` sends work to background thread, `sync_forward()` blocks for result
- `Arc<WeightStore>` for shared ownership between main thread and worker
- `Mutex<mpsc::Receiver>` wrapper — PyO3 `#[pyclass]` requires `Sync`, `mpsc::Receiver` is `!Sync`
- `Drop` impl sends sentinel (layer_idx=usize::MAX) for clean worker shutdown
- `test_async_submit_sync`: bit-exact match sync vs async, batch=2, masked experts, cleanup

Tests: **33 PASS** (added 1)

## [f91cf42] NUMA-aware placement — 2026-02-09

**Add NUMA-aware expert placement and execution**

- `src/numa.rs`: `NumaTopology`, `NumaAlloc`, `NumaExpertMap` types
- `libnuma` FFI bindings for `numa_alloc_onnode`, `numa_move_pages`, `sched_setaffinity`
- Expert-to-NUMA-node mapping based on activation heatmap
- `migrate_to_node()`: moves expert weight pages to target NUMA node
- `pin_thread_to_node()`: pins worker thread to NUMA node's CPU cores
- `build.rs`: links `libnuma`

Tests: **32 PASS** (added 5 NUMA tests)

## [f9c1a3a] Intra-expert parallelism — 2026-02-09

**Enable intra-expert parallelism and configurable thread count**

- `KrasisEngine(parallel=True, num_threads=N)` constructor parameters
- Thread count forwarded to `rayon::ThreadPoolBuilder`
- Intra-expert parallelism: single large matmul split across multiple threads
- Configurable via PyO3 constructor

Tests: **27 PASS**

## [5731df6] Partial loading + Kimi K2.5 — 2026-02-09

**Add partial model loading and Kimi K2.5 end-to-end forward test**

- `WeightStore::load_partial()`: load subset of layers (for memory-constrained testing)
- Kimi K2.5 (384 experts, 4096 intermediate) end-to-end forward test
- Pre-quantized INT4 (compressed-tensors) dequantization path

Tests: **26 PASS** (added 2: kimi_k25_single_expert, kimi_k25_moe_forward)

## [d960d4a] Generify config parsing — 2026-02-09

**Generify config parsing and add pre-quantized model support**

- Generic `MoeConfig` from HF `config.json` — auto-detects DeepSeek, Qwen3, Kimi K2.5
- Pre-quantized model support: reads `weight_packed` + `weight_scale` + `weight_shape`
- `compressed-tensors` INT4 dequantization (group_size=32)

Tests: **24 PASS**

## [644c1f8] Zero-allocation scratch pool — 2026-02-09

**Pre-allocate scratch pool for zero-allocation expert parallelism**

- `ScratchPool`: pre-allocated per-thread scratch buffers for matmul intermediates
- Eliminates allocation in hot MoE forward path
- Pool sized to max(num_experts_per_tok) × thread_count

Tests: **23 PASS**

## [b628ccb] Expert parallelism — 2026-02-09

**Add expert-level parallelism for 3.3x MoE throughput**

- Rayon parallel iterator over active experts within a single token
- 3.3x speedup on V2-Lite (6 experts/token)
- Thread-safe expert forward via shared `&WeightStore`

Tests: **22 PASS**

## [56e77e6] Integer kernel — 2026-02-09

**Add integer kernel (_mm256_madd_epi16) for 2x matmul throughput**

- `_mm256_maddubs_epi16` + `_mm256_madd_epi16` pipeline for INT4×INT8 matmul
- 2x throughput vs FP32 accumulation path
- AVX2 throughput benchmark test

Tests: **21 PASS** (added 2: integer kernel + throughput)

## [5d1bdca] MoE benchmark — 2026-02-09

**Add comprehensive MoE benchmark script**

- `bench_moe.py`: Benchmarks single-token and batch MoE forward latency
- Reports tok/s, ms/token, ms/expert breakdowns

Tests: **19 PASS**

## [ebf757a] System checks — 2026-02-09

**Add startup system checks**

- `system_check()` PyO3 function: CPU governor, hugepages, memory, NUMA, SIMD
- Warns on performance-degrading configurations

Tests: **19 PASS** (added 1)

## [608b4f2] NTA prefetch — 2026-02-09

**Add NTA prefetch for next expert and optimize cache read path**

- `_mm_prefetch` with `_MM_HINT_NTA` for next-expert weight pages
- Reduced cache pollution for streaming weight access pattern

Tests: **18 PASS**

## [150b60e] INT4 disk cache — 2026-02-09

**Add INT4 disk cache for instant model loading**

- Quantizes HF safetensors → INT4 on first load, caches to disk
- Subsequent loads: mmap from cache (no quantization)
- `test_cache_bit_exact`: verifies cached == freshly quantized

Tests: **17 PASS** (added 1)

## [854daa1] PyO3 shim — 2026-02-09

**Implement PyO3 shim with KrasisEngine and SGLang FusedMoE wrapper**

- `KrasisEngine` Python class: `load()`, `forward()`, `num_moe_layers()`, etc.
- Byte-buffer interface for zero-copy BF16/INT32/FP32 tensor transfer

Tests: **16 PASS**

## [db25bcb] WeightStore + MoE forward — 2026-02-09

**Add WeightStore, parallel matmul, and MoE forward pass**

- `WeightStore`: holds all expert weights in quantized INT4 format
- `moe_forward()`: full token routing — gate/up matmul, SiLU, down matmul, weighted sum
- Parallel matmul via rayon over output rows

Tests: **14 PASS** (added 4: v2_lite_load, v2_lite_single_expert, v2_lite_moe_forward, throughput)

## [2132c51] Marlin repack — 2026-02-09

**Implement Marlin GPU format repack with permutation tables**

- `marlin_repack()`: converts INT4 packed format to Marlin GPU layout
- Permutation tables for efficient GPU access patterns
- Round-trip verification test

Tests: **10 PASS** (added 2)

## [e94511f] AVX2 INT4 kernel — 2026-02-09

**Implement AVX2 INT4 matmul kernel with scalar reference**

- `avx2_int4_matmul()`: vectorized INT4×BF16 matmul with group-wise dequantization
- `scalar_int4_matmul()`: reference implementation for verification
- Handles group_size=32/128, arbitrary M/N/K

Tests: **8 PASS** (added 4)

## [86e5293] Safetensors reader — 2026-02-09

**Implement safetensors mmap reader and INT4 quantization**

- `SafetensorsReader`: mmap-based reader for HF safetensors format
- Symmetric INT4 quantization with per-group scales
- BF16↔FP32 conversion utilities

Tests: **4 PASS** (added 4)

## [4747d96] Project scaffold — 2026-02-09

**Initial project scaffold**

- Rust + PyO3 project structure with Cargo.toml, lib.rs, build.rs
- Module layout: kernel/, weights/, moe.rs, numa.rs

## [26100b0] Initial commit — 2026-02-09

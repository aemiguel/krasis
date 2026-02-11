# Krasis — Feature & Optimization Status

## Architecture

Rust + PyO3 hybrid LLM MoE runtime. Replaces KTransformers CPU expert dispatch
for SGLang. Targets AMD EPYC (AVX2) + NVIDIA GPUs.

## Completed Features

### Core Engine
- [x] **Safetensors mmap reader** — zero-copy tensor access from HF model shards
- [x] **INT4 symmetric quantization** — BF16 → INT4 with per-group scales
- [x] **AVX2 INT4 matmul kernel** — FMA-based dequant + accumulate
- [x] **Integer kernel** (`_mm256_madd_epi16`) — 2x throughput over FMA path
- [x] **Marlin GPU format repack** — permutation tables for GPU prefill compatibility
- [x] **WeightStore** — manages all expert weights across layers
- [x] **MoE forward pass** — gate/up → SiLU → down with weighted expert sum
- [x] **PyO3 shim** — `KrasisEngine` Python class with byte-buffer interface

### Performance Optimizations
- [x] **Expert-level parallelism** — rayon parallel iter over active experts (3.3x speedup)
- [x] **Intra-expert parallelism** — single large matmul split across threads
- [x] **Zero-allocation scratch pool** — pre-allocated per-expert buffers
- [x] **NTA prefetch** — prefetch next expert's weights into L3 during current compute
- [x] **INT4 disk cache** — quantize once, mmap from cache on subsequent loads
- [x] **NUMA-aware placement** — migrate expert pages to local NUMA nodes, pin threads
- [x] **Configurable thread count** — rayon pool sized to hardware

### Model Support
- [x] **DeepSeek V2-Lite** — BF16 weights, 64 experts, top-6 (test model)
- [x] **Kimi K2.5** — pre-quantized INT4 (compressed-tensors), 384 experts, top-8
- [x] **Qwen3-235B-A22B** — BF16 weights, 128 experts, top-8
- [x] **Generic config parsing** — auto-detects DeepSeek, Kimi, Qwen3 from config.json
- [x] **Pre-quantized weight loading** — compressed-tensors INT4 (weight_packed + weight_scale + weight_shape)
- [x] **Auto-detect expert prefix** — handles `model.layers.*.mlp.experts.*` and `language_model.model.layers.*.mlp.experts.*`
- [x] **Partial model loading** — load subset of layers for testing/memory-constrained runs

### SGLang Integration
- [x] **KrasisMoEWrapper** — drop-in replacement for KTMoEWrapper
- [x] **Async submit/sync** — background worker thread with mpsc channels
- [x] **Expert ID masking** — skip GPU-handled experts (id=-1)
- [x] **Singleton engine** — one KrasisEngine shared across all layer wrappers
- [x] **SGLang import toggle** — `KRASIS_BACKEND=1` env var in kt_ep_wrapper.py
- [x] **Launch script** — `run_krasis.sh` for DeepSeek-V2-Lite testing

### Shared Experts
- [x] **Shared expert loading** — loads `shared_experts.{gate,up,down}_proj` BF16 weights, quantizes to INT4
- [x] **Shared expert forward** — always-active MLP added to routed expert output
- [x] **routed_scaling_factor** — scale routed output before adding shared (V2-Lite: 1.0, Kimi K2.5: 2.827)
- [x] **Model support**: V2-Lite (2 shared), Kimi K2.5 (1 shared), Qwen3 (0 shared, no-op)

### Unified Weight Format
- [x] **Combined w13 (gate+up)** — single matrix `[K/8, 2*N]` transposed layout, eliminates one matmul per expert
- [x] **Transposed AVX2 integer kernel** — `_mm256_mullo_epi32` SIMD across output dim (no horizontal sum)
- [x] **Transposed AVX2 FMA kernel** — `_mm256_fmadd_ps` variant for verification/fallback
- [x] **Parallel transposed kernels** — rayon split across N dimension, chunk_n=256
- [x] **UnifiedExpertWeights** — in-place conversion from gate/up/down, drops old format after
- [x] **Layer-by-layer conversion** — converts one layer at a time, frees old format immediately (avoids 2x RAM peak)
- [x] **V2 unified disk cache** — stores unified weights directly (`.krasis_cache/experts_unified_int4_g{gs}.bin`), includes shared experts
- [x] **V1→V2 cache migration** — loads v1 cache, converts layer-by-layer, saves v2, deletes v1 automatically
- [x] **moe_forward_unified** — full MoE forward with unified weights (parallel, NUMA, shared experts)
- [x] **NTA prefetch for unified** — prefetch w13+w2 packed+scales into L3
- [x] **Auto-dispatch** — engine.load() converts to unified, moe_forward auto-dispatches
- [x] **Verified correct** — V2-Lite real weights max_abs_diff=0.000001 unified vs original

### Safety & Monitoring
- [x] **System RAM budget check** — estimates RAM before loading, refuses if >95% MemTotal
- [x] **Post-load RSS check** — warns if actual RSS deviates >10% from estimate
- [x] **force_load parameter** — override RAM check for testing

### Infrastructure
- [x] **System checks** — CPU governor, hugepages, memory budget, NUMA, SIMD
- [x] **MoE benchmark script** — `bench_moe.py` for latency profiling
- [x] **41 Rust tests** — unit + integration, all passing
- [x] **3 Python bridge tests** — engine roundtrip, wrapper interface, batch forward

### GPU Prefill
- [x] **INT4 Marlin prefill kernel** — GPU-accelerated MoE via `fused_marlin_moe` kernel
- [x] **CPU/GPU prefill switching** — GPU for prompts > threshold (300 tokens), CPU for short
- [x] **Expert buffer management** — GPU quantize BF16→INT4, Marlin repack, RAM cache, DMA to GPU
- [x] **Chunked expert processing** — handles models with many experts (e.g. 384) in VRAM-sized chunks
- [x] **Shared expert GPU path** — shared expert forward via Marlin kernel with weight=1.0
- [x] **Pre-quantized weight support** — dequantize compressed-tensors INT4 before re-quantizing to Marlin format
- [x] **PyO3 weight exposure** — `get_expert_w13_packed/scales/w2_packed/scales` methods on KrasisEngine
- [x] **Engine-backed GPU prefill** — GpuPrefillManager reads weights from Rust engine, repacks to Marlin on GPU. Eliminates ~438 GB Python RAM cache for Kimi K2.5

### Multi-GPU
- [x] **Pipeline parallelism** — PP=3 verified on Kimi K2.5 (GPU0: 20 layers, GPU1: 21 layers, GPU2: 20 layers)
- [x] **PP communication** — CPU bounce for cross-GPU transfer (GPU P2P broken on RTX 2000 Ada)
- [x] **Skip shared experts** — `skip_shared_experts` flag prevents double computation when host (SGLang) handles shared experts on GPU

### Advanced Optimizations
- [ ] **CUDA graphs** — reduce kernel launch overhead (if compatible with dynamic routing)
- [ ] **Speculative decoding** — draft model on spare GPU
- [ ] **Dynamic expert offloading** — move cold experts to disk, hot to RAM/GPU
- [ ] **Token batching** — batch multiple decode tokens for higher throughput

### Standalone Model (replaces SGLang entirely)
- [x] **MLA attention** — FlashInfer BatchMLAPagedAttentionWrapper, YaRN RoPE, FP8 KV cache
- [x] **GQA attention** — FlashInfer BatchPrefillWithPagedKVCacheWrapper, QKNorm, standard RoPE
- [x] **Per-component quantization** — configurable BF16/INT8 per weight type via QuantConfig
- [x] **INT8 Marlin GPU prefill** — fused_marlin_moe supports num_bits=4 and num_bits=8
- [x] **FP8 KV cache** — store FP8 E4M3, upcast to BF16 for FlashInfer kernel
- [x] **HTTP server** — FastAPI, SSE streaming, /v1/chat/completions
- [x] **VRAM budget calculator** — auto-sizes KV cache and context length

## Performance Results

| Model | Config | Decode | GPU0 VRAM | GPU1 VRAM | Notes |
|-------|--------|--------|-----------|-----------|-------|
| V2-Lite | Standalone PP=1, INT4 GPU prefill | 3.3 tok/s | 424 tok/s prefill | — | Test model, 5/6 gen tests pass |
| Kimi K2.5 | PP=2, BF16 wt, BF16 KV | 1.55-1.87 tok/s | 12,063 MB | 11,105 MB | **3/3 PASS**, diag ON |
| Kimi K2.5 | PP=2, INT8 wt, BF16 KV | 1.28-1.41 tok/s | 7,654 MB | 6,044 MB | **3/3 PASS** |
| Kimi K2.5 | PP=2, INT8 wt, FP8 KV | 1.21-1.28 tok/s | 7,654+4,032 KV | 6,044+4,839 KV | **3/3 PASS**, ~4x context |
| Kimi K2.5 | SGLang PP=3, INT8 wt, FP8 KV, CPU decode | ~1.0 tok/s (debug ON) | ~6.8 GB | ~4.4 GB | **WORKING** — correct output verified |
| Kimi K2.5 | KTransformers PP=2 | 4.0 tok/s | ~7.6 GB | ~7.6 GB | Production baseline |
| Qwen3-235B | KTransformers PP=3 | 4.21 tok/s | — | — | With expert pinning |

### Current Status
- **Kimi K2.5 WORKING** on SGLang PP=3 with Krasis CPU decode
- Correct output verified (factual Q&A, reasoning, creative writing)
- Decode speed ~1.0 tok/s with debug flags, testing without
- GPU prefill not yet tested (CPU-only for short prompts)

### Known Issues
- **Decode speed gap** — ~1.0 tok/s vs 4.0 tok/s KTransformers baseline. Contributing factors: debug flags, non-unified format (partial load), CUDA_LAUNCH_BLOCKING, PP overhead
- **Partial load skips v2 cache** — PP ranks use `load_and_quantize_all()` from safetensors instead of the fast v2 unified disk cache. Loading takes ~8 min vs ~2 min. Needs per-rank cache loading.
- **3 copies of expert weights** — each PP rank process loads its own portion from safetensors. Total ~570 GB RSS for 3 ranks vs ~530 GB if shared.

### Resolved Blockers
- ~~**VRAM budget calculator broken**~~ — FIXED: redesigned with proper overhead estimation (2000 MB), context-length-as-hint approach
- ~~**CUDA illegal memory access**~~ — FIXED: `correction_bias` was on CPU, moved to GPU in `select_experts()`
- ~~**Garbage output**~~ — FIXED: shared experts were applied twice (CPU + GPU). Added `skip_shared_experts` flag.
- ~~**GPU prefill was disabled**~~ — FIXED: GPU_PREFILL_THRESHOLD restored to 300
- ~~**OOM from dual weight copies**~~ — FIXED: unified weight format eliminates ~438 GB duplicate
- ~~**OOM from conversion peak**~~ — FIXED: streaming conversion processes one layer at a time
- ~~**Stale v1 disk cache**~~ — FIXED: auto-migrates v1→v2 cache

### Implemented: VRAM Budget Calculator
Context length is now a **hint**. The calculator computes per-rank weight footprint, free VRAM,
KV bytes/token, and allocates the minimum of (requested, max that fits). Measured overhead:
1716-1912 MB per rank → uses 2000 MB conservative estimate.
Result: 65K context on 16GB GPUs with INT8 weights + FP8 KV.

## Target Architecture

```
Krasis Standalone (single process, replaces SGLang + KTransformers)
    ├── GPU: attention (MLA/GQA), norms, routing, shared expert
    │   ├── INT8 or BF16 weights (per-component configurable)
    │   ├── FlashInfer MLA/GQA attention
    │   └── INT4/INT8 Marlin GPU prefill for MoE (large batches)
    ├── CPU: routed expert MoE (Rust AVX2 kernel)
    │   ├── INT4 or INT8 expert weights
    │   ├── Expert-level parallelism (rayon)
    │   ├── NUMA-aware weight placement
    │   └── Async worker thread
    └── HTTP: FastAPI /v1/chat/completions (SSE streaming)
```

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

### Infrastructure
- [x] **System checks** — CPU governor, hugepages, memory budget, NUMA, SIMD
- [x] **MoE benchmark script** — `bench_moe.py` for latency profiling
- [x] **34 Rust tests** — unit + integration, all passing
- [x] **3 Python bridge tests** — engine roundtrip, wrapper interface, batch forward

### GPU Prefill
- [x] **INT4 Marlin prefill kernel** — GPU-accelerated MoE via `fused_marlin_moe` kernel
- [x] **CPU/GPU prefill switching** — GPU for prompts > threshold (300 tokens), CPU for short
- [x] **Expert buffer management** — GPU quantize BF16→INT4, Marlin repack, RAM cache, DMA to GPU
- [x] **Chunked expert processing** — handles models with many experts (e.g. 384) in VRAM-sized chunks
- [x] **Shared expert GPU path** — shared expert forward via Marlin kernel with weight=1.0
- [x] **Pre-quantized weight support** — dequantize compressed-tensors INT4 before re-quantizing to Marlin format

### Multi-GPU
- [x] **Pipeline parallelism** — PP=2 verified on Kimi K2.5 (GPU0: 31 layers, GPU1: 30 layers)
- [x] **PP communication** — CPU bounce for cross-GPU transfer (GPU P2P broken on RTX 2000 Ada)

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
| Kimi K2.5 | PP=2, INT8 wt, FP8 KV, GPU prefill | **CRASH** | — | — | CUDA illegal addr at L31 (PP boundary) |
| Kimi K2.5 | KTransformers PP=2 | 4.0 tok/s | ~7.6 GB | ~7.6 GB | Production baseline |
| Qwen3-235B | KTransformers PP=3 | 4.21 tok/s | — | — | With expert pinning |

### Current Blockers
- **GPU prefill crashes at PP boundary** — CUDA illegal memory access when Marlin kernel runs on GPU1 for first time (layer 31). Debugging with sync points + CUDA_LAUNCH_BLOCKING. See CHANGELOG for details.
- **GPUs need reboot** — GPU0 in error state from crash, corrupted CUDA driver for all GPUs
- **Decode speed gap** — 1.2-1.9 tok/s vs 4.0 tok/s baseline (BF16 weights + diag overhead, not yet optimized)

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

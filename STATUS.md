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

### Infrastructure
- [x] **System checks** — CPU governor, hugepages, memory budget, NUMA, SIMD
- [x] **MoE benchmark script** — `bench_moe.py` for latency profiling
- [x] **33 Rust tests** — unit + integration, all passing
- [x] **3 Python bridge tests** — engine roundtrip, wrapper interface, batch forward

## Not Yet Implemented

### Shared Experts (next up)
- [ ] **Shared expert loading** — load `shared_experts.{gate,up,down}_proj` weights
- [ ] **Shared expert forward** — always-active MLP added to routed expert output
- [ ] **routed_scaling_factor** — scale routed output before adding shared (Kimi K2.5: 2.827)
- [ ] **Models**: V2-Lite (2 shared), Kimi K2.5 (1 shared), Qwen3 (0 shared)

### GPU Expert Pinning
- [ ] **Heatmap-based pinning** — pin hottest experts as INT4 Marlin on GPU VRAM
- [ ] **Hybrid dispatch** — GPU kernel for pinned experts, CPU for rest, per-layer
- [ ] **VRAM budget** — auto-fit experts within available GPU memory

### GPU Prefill
- [ ] **INT4 Marlin prefill kernel** — GPU-accelerated prefill for long prompts
- [ ] **CPU/GPU prefill switching** — GPU for prompts > threshold, CPU for short
- [ ] **Expert buffer management** — DMA cached INT4 weights to GPU

### Multi-GPU
- [ ] **Pipeline parallelism** — split layers across GPUs (PP=3 for 3x RTX 2000 Ada)
- [ ] **PP communication** — NCCL/Gloo inter-GPU hidden state transfer

### Advanced Optimizations
- [ ] **CUDA graphs** — reduce kernel launch overhead (if compatible with dynamic routing)
- [ ] **Speculative decoding** — draft model on spare GPU
- [ ] **Dynamic expert offloading** — move cold experts to disk, hot to RAM/GPU
- [ ] **Token batching** — batch multiple decode tokens for higher throughput

## Performance Baselines

| Model | System | Decode | Prefill | Notes |
|-------|--------|--------|---------|-------|
| V2-Lite | Krasis (16 threads) | ~1.7ms/token | — | Test model, single GPU |
| Kimi K2.5 | KTransformers PP=3 | 3.52 tok/s | 19 tok/s | Current production |
| Qwen3-235B | KTransformers PP=3 | 4.21 tok/s | 25 tok/s CPU, 443 tok/s GPU | With expert pinning |

## Target Architecture

```
SGLang (GPU: attention, routing, norms)
    ↕ submit_forward / sync_forward
Krasis (CPU: expert MoE computation)
    ├── AVX2 INT4 matmul kernel
    ├── Expert-level parallelism (rayon)
    ├── NUMA-aware weight placement
    └── Async worker thread
```

# Krasis

Rust + PyO3 MoE runtime for large mixture-of-experts LLMs. Runs 350B+ parameter models on commodity hardware with full GPU prefill and efficient CPU decode.

## What It Does

Runs MoE models (DeepSeek V2, Kimi K2.5, Qwen3-235B, GLM-4.7) on 3x 16GB GPUs + 1TB system RAM. GPUs handle attention, norms, routing, and expert prefill; Krasis handles MoE expert decode on CPU with AVX2 INT4/INT8 kernels.

**Key idea**: GPU prefill is always ON. Prompts process at hundreds-to-thousands of tokens per second on GPU, then CPU handles token-by-token decode. This makes IDE integration practical (10K prompt in 4s instead of 57s).

## Architecture

```
Krasis Standalone (single process, replaces SGLang + KTransformers)
├── GPU: attention (MLA/GQA), norms, routing, shared expert
│   ├── INT8 or BF16 weights (per-component configurable via QuantConfig)
│   ├── FlashInfer MLA attention (DeepSeek/Kimi) or GQA (Qwen3/GLM-4.7)
│   ├── FP8 E4M3 KV cache (2x VRAM savings, upcast to BF16 for kernel)
│   └── INT4/INT8 Marlin GPU prefill for MoE (fused_marlin_moe kernel)
├── CPU: routed expert MoE (Rust AVX2 kernel)
│   ├── INT4 or INT8 expert weights (CPU-optimized sequential layout)
│   ├── Expert-level + intra-expert parallelism (rayon)
│   ├── NUMA-aware weight placement + thread pinning
│   ├── Zero-allocation scratch pool, NTA prefetch
│   └── Async worker thread with mpsc channels
├── Dual weight format:
│   ├── (A) GPU cache: Marlin tile-permuted format → DMA to GPU, zero conversion
│   └── (B) CPU cache: sequential row-major → AVX2 cache-friendly decode
└── HTTP: FastAPI /v1/chat/completions (SSE streaming)
```

### Weight Format

Two separate disk caches, independently configurable precision (INT4 or INT8):

- **(A) GPU cache** (`experts_marlin_g{gs}.bin`): Marlin tile-permuted layout for `fused_marlin_moe` CUDA kernel. DMA copy from RAM to GPU with zero conversion.
- **(B) CPU cache** (`experts_cpu_{bits}_g{gs}.bin`): Sequential row-major layout optimized for AVX2 cache locality. Combined w13 (gate+up) eliminates one matmul per expert.

First run: BF16 safetensors → quantize → write both caches. Every run: load both from disk.

This dual format replaced an earlier single-format (Marlin everywhere) after testing showed Marlin's tile permutation destroyed CPU cache locality (0.55 tok/s vs 1.55 tok/s on Kimi K2.5).

### GPU Prefill Modes (`expert_divisor`)

| Mode | VRAM | Prefill Speed | KV Capacity |
|------|------|:---:|:---:|
| `divisor=0` (chunked) | 286 MB buffer | 173 tok/s (10K) | 212K tokens |
| `divisor=1` (persistent) | 7,654 MB all experts | 2,409 tok/s (10K) | 93K tokens |
| `divisor=2` (layer-grouped) | ~3,827 MB/group | ~400-600 tok/s | 216K tokens |

OOM fallback: persistent → layer-grouped(2) automatically if VRAM insufficient.

## Supported Models

| Model | Architecture | Experts | Attention | Status |
|-------|-------------|---------|-----------|--------|
| **DeepSeek V2-Lite** | deepseek_v2 | 64 + 2 shared, top-6 | MLA | Working (test model, 5.8 tok/s) |
| **Kimi K2.5** | kimi_k2 | 384 + 1 shared, top-8 | MLA | Retired (too slow on our HW) |
| **Qwen3-235B-A22B** | qwen3_moe | 128 routed, top-8 | GQA | Working (KTransformers, 4.21 tok/s) |
| **GLM-4.7** | glm4_moe | 160 + 1 shared, top-8 | GQA (partial RoPE, bias) | Config parses, untested |
| **Qwen3-Coder-Next** | qwen3_moe | 160 routed, top-8 | GQA | Next target |

### Input Formats

- **BF16 safetensors** (default): builds both GPU Marlin + CPU optimized caches
- **GGUF** (Q4_K_M, Q5_K, Q8_0, etc.): dequant → AVX2 transposed cache, with per-projection mixed precision

## Hardware Requirements

- **CPU**: x86-64 with AVX2+FMA (AMD EPYC, Intel Xeon). AVX512 not required.
- **GPU**: NVIDIA compute 8.0+ (Ampere/Ada). 16GB+ VRAM per GPU.
- **RAM**: ~500-600 GB for 350B+ models (expert weights in system RAM).

## Building

```bash
# Build Rust library
cargo build --release

# Install Python package
pip install -e .

# Or build + install manually (for AMD Zen 2 — NATIVE adds -mfma)
CPUINFER_CPU_INSTRUCT=NATIVE ./install.sh build --manual
```

## Usage

### Standalone Server

```bash
python -m krasis.server \
    --model-path /path/to/model \
    --pp-partition 1 \
    --gpu-expert-bits 4 \
    --cpu-expert-bits 4 \
    --expert-divisor 1
```

### With SGLang (legacy)

```bash
export KRASIS_BACKEND=1
python -m sglang.launch_server \
    --model /path/to/model \
    --pp-size 3 \
    --quantization w8a8_int8 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-cuda-graph
```

## Features

- **Dual weight format** — separate GPU (Marlin) and CPU-optimized caches, each independently configurable as INT4 or INT8
- **Persistent expert buffers** — pre-load all experts in VRAM for 14-72x prefill speedup
- **Layer-grouped prefill** — cycles expert groups through VRAM when they don't all fit
- **GGUF input** — accepts GGUF files, converts to AVX2 transposed format, disk-caches
- **FP8 KV cache** — 2x VRAM savings with negligible precision loss
- **INT8 non-expert weights** — halves attention VRAM via per-channel quantization
- **Per-component quantization** — QuantConfig controls BF16/INT8 per weight type
- **VRAM budget calculator** — auto-sizes KV cache and context length to available VRAM
- **System checks** — CPU governor, hugepages, NUMA topology, SIMD capability
- **MLA + GQA attention** — FlashInfer backends with YaRN RoPE, partial RoPE, attention bias
- **Pipeline parallelism** — PP=1/2/3 with CPU bounce for cross-GPU transfer

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — Detailed change history with test results
- [RESEARCH.md](RESEARCH.md) — Performance analysis, benchmarks, and research findings

## License

Apache 2.0

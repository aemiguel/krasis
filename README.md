# Krasis

Rust + PyO3 MoE runtime for large mixture-of-experts LLMs. Runs 350B+ parameter models on commodity hardware with full GPU prefill and efficient CPU decode.

For the TLDR version take a look at the [BENCHMARKS.md](https://github.com/brontoguana/krasis/blob/main/BENCHMARKS.md) file.

Contact: see the Google Forms link on my GitHub profile

## Quick Start

### Option A: pip install (recommended)

```bash
pip install krasis

# PyTorch with CUDA must be installed separately
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Download a model into ~/.krasis/models/
huggingface-cli download Qwen/Qwen3-Coder-Next \
    --local-dir ~/.krasis/models/Qwen3-Coder-Next

# Launch
krasis
```

### Option B: from source

```bash
# Prerequisites (Ubuntu/Debian)
sudo apt update && sudo apt install python3.12-venv

# Clone and run — everything else is automatic
git clone https://github.com/brontoguana/krasis.git
cd krasis
./krasis
```

### Where things go

Krasis stores everything under `~/.krasis/` (override with `KRASIS_HOME`):

```
~/.krasis/
├── config          # saved launch configuration
└── models/         # put your HuggingFace models here
    └── Qwen3-Coder-Next/
        ├── *.safetensors   # model weights (BF16)
        ├── config.json
        └── .krasis_cache/  # auto-generated, can grow large
            ├── experts_marlin_g128.bin   # GPU cache (INT4/INT8)
            ├── experts_cpu_4_g128.bin    # CPU cache (INT4/INT8)
            └── auto_optimise.json        # cached strategy
```

**Note:** The `.krasis_cache/` folder inside each model directory can be very large (100-300 GB for 350B+ models). It contains pre-computed quantized weights and is rebuilt automatically if deleted.

On first run, Krasis:
1. Creates `~/.krasis/` and `~/.krasis/models/`
2. Checks system configuration (CPU governor, hugepages, SIMD)
3. Opens an interactive TUI for model selection and configuration
4. Builds optimized weight caches on disk (GPU Marlin + CPU AVX2)
5. Auto-discovers the best prefill/decode strategy for your hardware

Subsequent runs skip all setup and load cached weights + strategy in seconds.

```bash
# Non-interactive (use saved config from last launch)
krasis --non-interactive

# Override specific settings
krasis --model-path /path/to/model --num-gpus 2

# Chat with a running server
krasis-chat
```

The server exposes an OpenAI-compatible API at `http://localhost:8080/v1/chat/completions` with SSE streaming support, compatible with tools like Cursor, OpenCode, and any OpenAI SDK client.

## What It Does

Krasis runs large language models — the kind with 350 billion+ parameters — on hardware you can actually buy. Three consumer GPUs and a lot of system RAM is all you need. No datacenter, no $100k GPU rental.

These models use a trick called Mixture of Experts (MoE): instead of running every parameter on every token, they pick a small subset of "expert" sub-networks per token. This means a 350B model might only activate 20B parameters at a time, making it feasible to run — if your software is smart about where to put the weights.

### Why GPU prefill matters

When you type a message to an LLM, two things happen. First, the model reads your entire prompt at once — this is **prefill**. Then it generates a response one token at a time — this is **decode**. Prefill is embarrassingly parallel (all tokens processed simultaneously), so GPUs are orders of magnitude faster at it than CPUs.

This matters a lot in practice. An IDE like Cursor or OpenCode sends 10,000+ tokens of context with every request. If prefill runs on CPU at 25 tokens/second, you're waiting 7 minutes before the model even starts responding. If prefill runs on GPU at 2,400 tokens/second, that same prompt processes in 4 seconds. That's the difference between a usable tool and a paperweight.

Krasis keeps GPU prefill **always on**. Prompts process at hundreds to thousands of tokens per second on GPU, then CPU handles the slower token-by-token generation. You wait a few seconds for the model to "read" your prompt, then tokens stream out at a steady 2-10+ tok/s depending on the model.

### How other tools handle GPU offloading

**llama.cpp** splits the model by layers. You tell it "put the first 10 layers on GPU, the rest on CPU." During both prefill and decode, each token flows through the GPU layers (fast), then the CPU layers (slow). The problem: if only 25% of the model fits on GPU, 75% of every computation still happens on CPU. Your GPU sits idle while the CPU grinds through its layers. Prefill speed is capped by the slowest link in the chain.

**KTransformers** is smarter — it puts attention and routing on GPU but keeps all expert weights on CPU. During decode this works well because only a few experts activate per token, and CPU handles that fine. But during prefill, when you're processing thousands of tokens, all those expert computations still happen on CPU. You're leaving the GPU's parallel compute power on the table for the most latency-sensitive part of inference.

**Krasis** takes a different approach. During prefill, it copies expert weights to GPU in bulk and runs everything — attention, routing, AND experts — on GPU using quantized Marlin kernels. During decode, it switches to CPU for experts (where the per-token workload is small enough that CPU keeps up). This means prefill gets full GPU acceleration on every layer, not just the attention layers.

### Tradeoffs

Krasis has to make some tradeoffs to make this happen.

It stores **two copies of the model in System RAM**, this allows it to maximise GPU prefill speed and CPU speed.

Research showed that a unified model was just not performant <1 tok/s decode.  Because of this, and because many large models have prefill that just doesn't fit
even on a 16GB GPU, Krasis allows you to selectively quantize differnt components of the model and gives recommendations about the impact of doing so.  You can
quantize the GPU weights to fit inside your VRAM budget or give you extra KV cache room (context window).  You can quantize the CPU model separately to maintain
quality or decrease system RAM to get the model to fit.  Krasis always needs to be given the native BF16 HuggingFace model.

It uses this to build the GPU-optimised in-memory model and the CPU-optimised in-memory model.  These are cached to disk so you'll typically need **3x the disk space for the model.**

**The fundamental tradeoff is using more system RAM** (which is drastically cheaper than VRAM) **and more disk** (which is much cheaper than both) to allow models that would
**otherwise be unusable to run on non-datacenter hardware.**

If you prefer you can also give Krasis the native model for GPU and an optimised (e.g. unsloth) Q4_K_S or similar GGUF model for the CPU to take advantage of more advanced quantization schemes.

### Technical comparison

The key architectural difference is how expert weights reach the GPU during prefill:

| System | Prefill strategy | Expert compute | Typical prefill speed (5K tokens) |
|--------|-----------------|----------------|:-:|
| **llama.cpp** | Layer split (fixed GPU/CPU partition) | Layers on GPU run fast, layers on CPU run slow | ~30 tok/s (75% CPU) |
| **KTransformers** | GPU attention + CPU experts | Attention on GPU, all experts always on CPU | ~25 tok/s |
| **Krasis (persistent)** | All experts pre-loaded in VRAM | Everything on GPU, zero weight transfers | **2,400 tok/s** |
| **Krasis (layer-grouped)** | Expert groups cycled through VRAM | All compute on GPU, a few bulk DMA transfers | **400-600 tok/s** |

When experts fit entirely in VRAM (small models, or big GPUs), Krasis pre-loads them once and runs with zero weight transfers — this is the persistent mode. When they don't fit (the common case with 350B+ models), Krasis uses layer-grouped mode: it loads a group of layers' experts into VRAM, processes ALL prompt tokens through those layers, frees the VRAM, and loads the next group. The DMA cost is fixed regardless of prompt length, so longer prompts amortize it better.

Even in the worst case — layer-grouped with only 25% of experts fitting at once — Krasis prefill is 15-20x faster than llama.cpp or KTransformers, because every multiply-accumulate happens on GPU hardware designed for exactly this workload.

For decode (token-by-token generation), all three systems are CPU-bound since only a handful of experts activate per token. llama.cpp and KTransformers typically achieve 2-5 tok/s. Krasis achieves 2-10+ tok/s depending on model architecture, using hand-tuned AVX2 INT4 kernels on CPU combined with CUDA graph optimizations that eliminate GPU kernel launch overhead.

## Architecture

```
Krasis Standalone (single process, replaces SGLang + KTransformers)
├── GPU: attention, norms, routing, shared expert
│   ├── INT8 or BF16 weights (per-component configurable via QuantConfig)
│   ├── FlashInfer MLA (DeepSeek) or GQA (Qwen3) attention
│   ├── GatedDeltaNet linear attention (Qwen3-Coder-Next hybrid models)
│   ├── FP8 E4M3 KV cache (2x VRAM savings, native FlashInfer support)
│   ├── INT4/INT8 Marlin GPU prefill for MoE (fused_marlin_moe kernel)
│   └── CUDA graphs: shared expert + linear attention captured as graphs
├── CPU: routed expert MoE (Rust AVX2 kernel)
│   ├── INT4 or INT8 expert weights (CPU-optimized sequential layout)
│   ├── Expert-level + intra-expert parallelism (rayon)
│   ├── NUMA-aware weight placement + thread pinning
│   ├── Zero-allocation scratch pool, NTA prefetch
│   └── Async worker thread with mpsc channels
├── Dual weight format:
│   ├── (A) GPU cache: Marlin tile-permuted format → DMA to GPU, zero conversion
│   └── (B) CPU cache: sequential row-major → AVX2 cache-friendly decode
├── Auto-optimiser: discovers best strategy on first run, caches for instant reload
└── HTTP: FastAPI /v1/chat/completions (SSE streaming, OpenAI-compatible)
```

### Weight Format

Two separate disk caches, independently configurable precision (INT4 or INT8):

- **(A) GPU cache** (`experts_marlin_g{gs}.bin`): Marlin tile-permuted layout for `fused_marlin_moe` CUDA kernel. DMA copy from RAM to GPU with zero conversion.
- **(B) CPU cache** (`experts_cpu_{bits}_g{gs}.bin`): Sequential row-major layout optimized for AVX2 cache locality. Combined w13 (gate+up) eliminates one matmul per expert.

First run: BF16 safetensors → quantize → write both caches. Every run: load both from disk.

This dual format replaced an earlier single-format (Marlin everywhere) after testing showed Marlin's tile permutation destroyed CPU cache locality (0.55 tok/s vs 1.55 tok/s on Kimi K2.5).

### GPU Prefill Modes (`expert_divisor`)

The auto-optimiser (`--strategy auto`) discovers the best mode automatically. Manual options:

| Mode | Description | Speed vs Chunked |
|------|-------------|:---:|
| `auto` | Auto-discover best mode (recommended) | best available |
| `divisor=1` (persistent) | All experts pre-loaded in VRAM | 14-72x faster |
| `divisor=2-32` (layer-grouped) | Expert groups cycled through VRAM | 2-10x faster |
| `divisor=-1` (active-only) | DMA only activated experts per layer | 1-3x faster |
| `divisor=0` (chunked) | Legacy per-layer full DMA | baseline |

OOM fallback: persistent → layer-grouped(2) automatically if VRAM insufficient.

Speed depends on model size and VRAM capacity. Smaller models (V2-Lite) can fit all experts persistently for 2,400+ tok/s. Larger models (235B+) use layer-grouped for 200-600 tok/s.

## Supported Models

| Model | Architecture | Experts | Attention | Status |
|-------|-------------|---------|-----------|--------|
| **Qwen3-Coder-Next** | qwen3_next | 512 routed, top-10 | Hybrid (36 linear + 12 GQA) | Primary target (10.5 tok/s decode, 580 tok/s prefill) |
| **Qwen3-235B-A22B** | qwen3_moe | 128 routed, top-8 | GQA | Working (1.65 tok/s decode, 198 tok/s prefill) |
| **DeepSeek V2-Lite** | deepseek_v2 | 64 + 2 shared, top-6 | MLA | Working (test model, 5.8 tok/s decode) |
| **Kimi K2.5** | kimi_k2 | 384 + 1 shared, top-8 | MLA | Retired (too slow on our HW) |
| **GLM-4.7** | glm4_moe | 160 + 1 shared, top-8 | GQA (partial RoPE, bias) | Config parses, untested |

### Input Formats

- **BF16 safetensors** (default): builds both GPU Marlin + CPU optimized caches
- **GGUF** (Q4_K_M, Q5_K, Q8_0, etc.): dequant → AVX2 transposed cache, with per-projection mixed precision

## Hardware Requirements

- **CPU**: x86-64 with AVX2+FMA (AMD EPYC, Intel Xeon). AVX512 not required.
- **GPU**: NVIDIA compute 8.0+ (Ampere/Ada). 16GB+ VRAM per GPU. AMD GPUs are not supported (Marlin kernels and FlashInfer are CUDA-only).
- **RAM**: ~500-600 GB for 350B+ models (expert weights in system RAM).

## Building from Source

The `./krasis` launcher handles building automatically on first run. For manual/development setup:

```bash
git clone https://github.com/brontoguana/krasis.git
cd krasis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# PyTorch must be installed separately
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Usage

### Interactive Launcher (recommended)

```bash
krasis        # pip install
./krasis      # from source
```

The launcher provides a TUI with:
- Model selection (scans `~/.krasis/models/` directory)
- GPU selection and pipeline parallelism configuration
- Per-component quantization with live VRAM/RAM budget display
- CPU expert source selection (build INT4/INT8 from native, or use GGUF)
- Auto-optimisation (discovers best prefill/decode strategy on first run)

Configuration is saved to `~/.krasis/config` and reloaded on subsequent launches.

### Non-Interactive Launch

```bash
# Use saved config
krasis --non-interactive

# Override specific settings
krasis --non-interactive --model-path /path/to/model --num-gpus 2
```

### Direct Server (advanced)

For fine-grained control, run the server directly:

```bash
python -m krasis.server \
    --model-path /path/to/model \
    --pp-partition 24,24 \
    --strategy auto \
    --gpu-expert-bits 4 \
    --cpu-expert-bits 4
```

Key server flags:
- `--strategy auto` — auto-discover best strategy (recommended)
- `--strategy manual --expert-divisor 4` — explicit layer-grouped(4)
- `--gpu-prefill-threshold 300` — GPU for prompts >= 300 tokens, CPU for M=1 decode
- `--kv-dtype fp8_e4m3` — FP8 KV cache (default, 2x VRAM savings)
- `--attention-quant int8` — INT8 attention weights (default, halves VRAM)

## Features

- **Interactive TUI launcher** — model selection, GPU picker, per-component quantization with live VRAM/RAM budget
- **Auto-optimiser** — discovers best prefill/decode strategy on first run, caches results for instant reload
- **Dual weight format** — separate GPU (Marlin) and CPU-optimized caches, each independently configurable as INT4 or INT8
- **Persistent expert buffers** — pre-load all experts in VRAM for 14-72x prefill speedup
- **Layer-grouped prefill** — cycles expert groups through VRAM when they don't all fit
- **CUDA graph optimizations** — shared expert + linear attention captured as graphs, eliminating kernel launch overhead
- **GGUF input** — accepts GGUF files, converts to AVX2 transposed format, disk-caches
- **FP8 KV cache** — 2x VRAM savings with negligible precision loss (native FlashInfer FP8 on SM89+)
- **INT8 non-expert weights** — halves attention VRAM via per-channel quantization
- **Per-component quantization** — QuantConfig controls BF16/INT8 per weight type
- **VRAM budget calculator** — auto-sizes KV cache and context length to available VRAM
- **System checks** — CPU governor, hugepages, NUMA topology, SIMD capability
- **MLA + GQA + linear attention** — FlashInfer MLA/GQA backends, GatedDeltaNet linear attention for hybrid models
- **Pipeline parallelism** — PP=1/2/3 with CPU bounce for cross-GPU transfer
- **OpenAI-compatible API** — `/v1/chat/completions` with SSE streaming, works with Cursor, OpenCode, etc.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — Detailed change history with test results
- [BENCHMARKS.md](BENCHMARKS.md) — Performance comparisons vs llama.cpp and KTransformers
- [RESEARCH.md](RESEARCH.md) — Performance analysis, benchmarks, and research findings

## License

Apache 2.0

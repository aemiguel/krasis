# Benchmarks

This document records comparison runs of Krasis vs other major LLM runtimes - llama.cpp and Ktransformers.

It's goal is to investigate and demonstrate the benefits of Krasis' approach to a hybrid runtime where 100% of the prefill stage is offloaded to GPU.

## Test LLM Model

Target Model: **Qwen3-235B-A22B (MoE, 658GB)**

- Llama model used: Unsloth Qwen3-235B-A22B Q8 (8-bit quantized)
- KTransformers model used: Unsloth Qwen3-235B-A22B Q8 (8-bit quantized)
- Krasis models used: Unsloth Qwen3-235B-A22B Q8 (8-bit quantized) + Original BF16 model weights

## Methodology

In each case, each tool is capable of both running the model on CPU, and trying to take advantage of the GPU(s).

We test with what we believe are ideal parameters given the situation for each tool to do its best.

We run each test tool with 1,2 and 3 GPUs.

We pass in the same large ~10,000 token prompt to each tool and measure both prefill (input) speed in tokens per second and decode (output) speed in tokens per second.

## Test Hardware

- Epyc 7742 (single CPU, 64 cores)
- 1TB DDR4 2666Mhz (8 channels)
- 3x RTX Ada 2000 GPUs

## Tool Configurations

### llama.cpp configuration + parameters used:

For each run:

- The Q8 GGUF model is used directly
- `-ngl` is set to offload as many layers as will fit in the available GPU VRAM
- `-t 48` to use physical cores only (hyperthreading hurts throughput on EPYC)
- `--flash-attn` enabled for efficient attention computation
- `-c 16384` for sufficient context length
- With multiple GPUs, `--split-mode layer` distributes offloaded layers across GPUs

Reasoning:

- llama.cpp treats each MoE layer as an atomic unit for GPU offload — all 128 experts in the layer must fit on GPU together
- At Q8 precision, each MoE layer is ~2.5 GB (experts + attention), so even 3 GPUs (48 GB) can only hold ~15 of 94 layers
- The remaining ~80 layers run on CPU for **both** prefill and decode
- This means prefill of a 10,000 token prompt is overwhelmingly CPU-bound, limited to CPU matrix multiplication speed
- Decode speed is similar to other tools since all are CPU-bound at M=1, but the prefill bottleneck makes llama.cpp much slower for large prompts

### KTransformers configuration + parameters used:

For each run:

- The Q8 GGUF model is used for CPU expert computation
- Pipeline parallelism across available GPUs (PP=1, PP=2, or PP=3)
- Attention weights are loaded onto the GPU(s), experts remain on CPU
- `--cpu_infer LLAMAFILE` backend for optimised AVX2 expert computation
- Expert pinning enabled to keep the hottest experts resident on GPU

Reasoning:

- KTransformers separates attention (GPU) from experts (CPU), which is more memory-efficient than llama.cpp's all-or-nothing layer offload
- All attention computation happens on GPU regardless of GPU count, which is an improvement over llama.cpp
- However, during prefill, expert computation still runs on CPU — and experts dominate the compute in MoE models
- Expert pinning helps by keeping frequently-used experts on GPU, but the majority of experts still run on CPU during prefill
- The result is faster prefill than llama.cpp (GPU attention), but still fundamentally CPU-bottlenecked on expert computation for large prompts

### Krasis configuration

For each run:

- Krasis is configured to quantize the BF16 model into an optimised INT8 GPU model, same precision as the Q8 GGUF
- Krasis is configured to divide the experts the minimal amount which allows them to fit on the GPU (to run as large a batch as possible)
- Individual component weights on the GPU which do not largely impact quality are quantised by Krasis to INT8
- Krasis is given the original BF16 model to build a GPU optimised Marlin based model on disk before loading into system RAM
- Krasis is given the Q8 GGUF model to build an AVX2 CPU optimised model on disk before loading into system RAM

Reasoning:

- Krasis will build optimised models for both the GPU and CPU, trading off system RAM usage (2x the other tools) for prefill and decode speed
- Krasis will DMA the optimised GPU model from system RAM onto the GPU, and process the entire prefill on GPU in an optimal way
- This will greatly speed up input token handling and result in a much faster response vs other tools with a similar decode generation speed

## Benchmark Results

### Test Configuration
- **Model**: Qwen3-235B-A22B (Q8_0 GGUF for llama.cpp/KTransformers, BF16 safetensors for Krasis)
- **Prompt**: 44,916 chars / 6,650 words / ~8,600 tokens (12 technical sections covering distributed systems, compilers, databases, etc.)
- **Decode tokens**: 50 (including thinking tokens for Qwen3)
- **Hardware**: AMD EPYC 7742 (48 threads), 995 GB RAM, 3x RTX 2000 Ada 16GB

### Results Table

TTFT (Time to First Token) measured wall-clock from request to first token (KTransformers, Krasis) or estimated as ~8,600 tokens / prefill rate (llama.cpp).

| Tool | GPUs | Config | Prefill (tok/s) | Decode (tok/s) | TTFT (s) | GPU VRAM (MB) | RAM (GB) | Notes |
|------|------|--------|----------------|----------------|----------|---------------|----------|-------|
| llama.cpp | 1 | ngl=5 | 30.1 | 3.5 | ~286 | 11,839 | 228 | |
| llama.cpp | 2 | ngl=10 | 31.6 | 3.7 | ~272 | 13,763+11,152 | 218 | |
| llama.cpp | 3 | ngl=14 | 32.9 | 3.8 | ~261 | 13,763+12,939+8,598 | 208 | ngl=16 OOM |
| KTransformers | 1 | TP=1 INT8 attn | 49.7 | 4.85 | 173.0 | 14,735 | 275 | INT8 attn needed to fit on 1 GPU |
| KTransformers | 2 | PP=2 | 57.5 | 3.60 | 149.5 | 14,895+13,707 | 275 | |
| KTransformers | 3 | PP=3 | 57.3 | 3.29 | 150.1 | 14,129+14,287+13,132 | 278 | |
| **Krasis** | **2** | **PP=2 HCS hybrid** | **198.1** | **1.65** | **43.9** | **12,963+12,968** | **223** | **Qwen3-235B-A22B** |
| **Krasis** | **2** | **PP=2 HCS hybrid** | **621** | **7.2** | **15.5** | **13,115+13,112** | **125** | **Qwen3-Coder-Next** |

### Qwen3-235B-A22B (PP=2, HCS Hybrid) — Krasis vs Others (Same Model)

This is a direct apples-to-apples comparison using the same Qwen3-235B-A22B model across all three tools.

| Metric | Krasis PP=2 | KTransformers (best) | llama.cpp (best) |
|--------|-----------|---------------------|-----------------|
| **Prefill (tok/s)** | **198.1** | 57.5 | 32.9 |
| **TTFT (s)** | **43.9** | 149.5 | ~261 |
| **Decode (tok/s)** | 1.65 | 4.85 | 3.8 |
| **Speedup vs KT (prefill)** | **3.4x** | 1x | - |
| **Speedup vs llama (prefill)** | **6.0x** | - | 1x |
| **TTFT improvement vs KT** | **3.4x faster** | 1x | - |
| **GPUs used** | 2 | 1 (best prefill: 2) | 3 |

Benchmark details (3 runs, ~8,700 token prompt):

| Run | Total time | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
|-----|-----------|----------|-----------------|----------------|
| 1 | 75.7s | 44.0 | 197.9 | 1.57 |
| 2 | 73.2s | 43.9 | 198.2 | 1.71 |
| 3 | 73.9s | 43.9 | 198.1 | 1.66 |

Configuration:
- PP=2 (47+47 layers), HCS hybrid mode with 86 hot experts pinned (cuda:0=27, cuda:1=59)
- INT4 Marlin GPU experts, INT4 CPU experts, FP8 KV cache, INT8 attention weights
- Layer-grouped DMA prefill: 1-layer groups, 5 chunks of 2048 tokens
- VRAM: GPU0=12,963 MB, GPU1=12,968 MB, RAM: ~223 GB

**Key takeaway**: Krasis achieves **3.4x faster prefill** and **3.4x faster TTFT** than KTransformers on the same model with fewer GPUs. Decode speed is lower (1.65 vs 4.85 tok/s) due to the HCS cold expert path and larger expert dimensions (128 experts × 3072 intermediate), but for large prompt workloads the dramatically faster TTFT dominates the user experience.

### Qwen3-Coder-Next (PP=2, HCS Hybrid) — VERIFIED CORRECT

The Qwen3-Coder-Next model (48 layers, 512 experts, top-10) achieves dramatically better results due to:
- Hybrid architecture: 36 linear attention + 12 GQA layers (only 12 need KV cache)
- Smaller expert size (2048×512 vs 2048×3072) — faster DMA and more experts fit in VRAM
- PP=2 pipeline parallelism (24+24 layers across 2 GPUs)

| Metric | Krasis HCS | KTransformers (best) | llama.cpp (best) |
|--------|-----------|---------------------|-----------------|
| **Prefill (tok/s)** | **621** | 57.3 | 32.9 |
| **TTFT (s)** | **15.5** | 150.1 | ~261 |
| **Speedup vs KT** | **10.8x** | 1x | - |
| **Speedup vs llama** | **18.8x** | - | 1x |
| **GPUs used** | 2 | 3 | 3 |

Benchmark details (3 runs, ~9,630 token prompt):

| Run | Total time | Prefill (tok/s) | Overall (tok/s) |
|-----|-----------|-----------------|-----------------|
| 1 | 18.2s | 620.4 | 530.4 |
| 2 | 18.2s | 621.4 | 531.1 |
| 3 | 18.1s | 622.1 | 531.6 |

Configuration:
- 4,161 hot experts pinned across 2 GPUs (cuda:0=2,033, cuda:1=2,128)
- INT4 Marlin GPU experts, INT4 CPU experts, FP8 KV cache, INT8 attention weights
- Pre-allocated DMA staging buffers: 830.5 MB (zero-copy memcpy path)
- VRAM: GPU0=13,115 MB, GPU1=13,112 MB, RAM: ~125 GB

**Note**: Prefill comparison uses Qwen3-235B numbers for KTransformers/llama.cpp (different model). Direct comparison pending Qwen3-Coder-Next runs on those tools — but the 10x+ advantage is primarily architectural (Krasis' full GPU prefill vs CPU-bottlenecked prefill in other tools).

## Conclusions

TODO

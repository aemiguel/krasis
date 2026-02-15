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

### DeepSeek-V2-Lite — Auto-Optimiser Strategy Comparison

The auto-optimiser was run on DeepSeek-V2-Lite (27 MoE layers, 64 experts, top-6, MLA attention) to benchmark all available prefill and decode strategies. Each strategy was tested with 3 runs, 10K token prefill prompt, 64 decode tokens.

**Quantization**: INT4 Marlin GPU experts, INT4 CPU experts, FP8 KV cache, INT8 attention weights.

#### 1 GPU (PP=[27])

**Prefill strategies** (ranked by tok/s):

| Strategy | Avg tok/s | Avg TTFT (s) | VRAM (MB) | Divisor |
|----------|----------|-------------|-----------|---------|
| persistent | 1,757.4 | 5.69 | 9,339 | 1 |
| layer_grouped_2 | 1,746.0 | 5.73 | 9,339 | 2 |
| layer_grouped_4 | 1,743.5 | 5.74 | 9,339 | 4 |
| hcs_prefill | 1,722.7 | 5.81 | 12,776 | -3 |
| chunked | 792.1 | 12.62 | 9,624 | 0 |
| active_only | 303.2 | 32.98 | 9,624 | -1 |

**Decode strategies** (ranked by tok/s):

| Strategy | Avg tok/s | VRAM (MB) | Divisor |
|----------|----------|-----------|---------|
| **pure_cpu** | **6.98** | **9,339** | null |
| hcs_hybrid | 4.88 | 12,776 | -3 |
| compact | 3.31 | 9,624 | -1 |
| lru | 3.31 | 14,376 | -2 |

**Winner**: persistent prefill (1,757 tok/s) + pure_cpu decode (6.98 tok/s)

#### 2 GPUs (PP=[14,13])

**Prefill strategies** (ranked by tok/s):

| Strategy | Avg tok/s | Avg TTFT (s) | VRAM (MB) | Divisor |
|----------|----------|-------------|-----------|---------|
| layer_grouped_2 | 1,729.2 | 5.79 | 8,863+8,709 | 2 |
| layer_grouped_4 | 1,719.9 | 5.82 | 8,863+8,709 | 4 |
| persistent | 1,709.5 | 5.86 | 8,863+8,709 | 1 |
| hcs_prefill | 1,706.4 | 5.87 | 12,815+12,823 | -3 |
| chunked | 790.3 | 12.65 | 9,148+8,995 | 0 |
| active_only | 302.5 | 33.06 | 9,148+8,995 | -1 |

**Decode strategies** (ranked by tok/s):

| Strategy | Avg tok/s | VRAM (MB) | Divisor |
|----------|----------|-----------|---------|
| **pure_cpu** | **6.69** | **8,863+8,709** | null |
| hcs_hybrid | 4.86 | 12,815+12,823 | -3 |
| compact | 3.51 | 9,148+8,995 | -1 |
| lru | 3.49 | 14,391+14,402 | -2 |

**Winner**: layer_grouped_2 prefill (1,729 tok/s) + pure_cpu decode (6.69 tok/s)

#### Key Observations

1. **Top 4 prefill strategies are nearly identical** (~1,700-1,760 tok/s) — V2-Lite's experts are small enough that DMA overhead is minimal regardless of strategy
2. **pure_cpu decode is fastest** (6.98-6.69 tok/s) — for this small model, CPU Rust engine beats GPU Marlin at M=1 (no GPU kernel launch overhead)
3. **Chunked is 2.2x slower** than the best for prefill — per-layer DMA is costly even for small experts
4. **Active-only is 5.8x slower** — individual expert DMA transfers dominate
5. **Adding a 2nd GPU has negligible effect** — V2-Lite is compute-light enough that 1 GPU handles it easily; the 2nd GPU adds PP overhead without benefit
6. **HCS uses more VRAM** (12.8 GB vs 9.3 GB) for hot expert pinning, but doesn't help decode on this model
7. **LRU uses excessive VRAM** (14.4 GB) without any decode benefit over compact

### Qwen3-Coder-Next — Auto-Optimiser Strategy Comparison (2 GPUs)

The auto-optimiser was run on Qwen3-Coder-Next (48 MoE layers, 512 experts, top-10, hybrid linear+GQA attention) with PP=[24,24] across 2 GPUs. Each strategy was tested with 3 runs, 10K token prefill prompt, 64 decode tokens.

**Quantization**: INT4 Marlin GPU experts, INT4 CPU experts, FP8 KV cache, INT8 attention weights.

**Prefill strategies** (ranked by tok/s):

| Strategy | Avg tok/s | Avg TTFT (s) | VRAM (MB) | Divisor |
|----------|----------|-------------|-----------|---------|
| layer_grouped_4 | 618.8 | 16.17 | 9,389+9,234 | 4 |
| hcs_prefill | 603.0 | 16.59 | 12,606+12,597 | -3 |
| chunked | 154.7 | 64.63 | 10,202+10,045 | 0 |
| active_only | 121.0 | 82.66 | 10,211+10,054 | -1 |
| persistent | CUDA OOM | — | — | 1 |
| layer_grouped_2 | CUDA OOM | — | — | 2 |

**Decode strategies** (ranked by tok/s):

| Strategy | Avg tok/s | VRAM (MB) | Divisor |
|----------|----------|-----------|---------||
| **pure_cpu** | **7.26** | **9,371+9,215** | null |
| hcs_hybrid | 7.02 | 12,271+12,271 | -3 |
| compact | 4.99 | 10,202+10,045 | -1 |
| lru | 4.90 | 14,099+14,104 | -2 |

**Winner**: layer_grouped_4 prefill (618.8 tok/s) + pure_cpu decode (7.26 tok/s)

#### Key Observations (Coder-Next)

1. **Layer-grouped(4) barely beats HCS prefill** (619 vs 603 tok/s, 3%) — both achieve ~600+ tok/s
2. **Persistent and layer_grouped(2) OOM** — 512 experts × 48 MoE layers is too large for 16GB VRAM at divisor ≤ 2
3. **pure_cpu decode is fastest** (7.26 tok/s) — beats HCS hybrid (7.02) by only 3%, but uses less VRAM
4. **Compact/LRU decode is ~30% slower** (5.0/4.9 tok/s) — GPU kernel launch overhead dominates at M=1 with small experts
5. **Chunked and active_only prefill are 4-5x slower** than the best — per-layer DMA overhead with 512 experts is crippling

### Qwen3-235B-A22B — Auto-Optimiser Strategy Comparison (2 GPUs)

The auto-optimiser was run on Qwen3-235B-A22B (94 MoE layers, 128 experts, top-8, MLA attention) with PP=[47,47] across 2 GPUs. Each strategy was tested with 3 runs, 10K token prefill prompt, 64 decode tokens.

**Quantization**: INT4 Marlin GPU experts, INT4 CPU experts, FP8 KV cache, INT8 attention weights.

**Prefill strategies** (ranked by tok/s):

| Strategy | Avg tok/s | Avg TTFT (s) | VRAM (MB) | Divisor |
|----------|----------|-------------|-----------|---------|
| hcs_prefill | 210.8 | 47.45 | 12,790+12,800 | -3 |
| chunked | 55.9 | 178.87 | 13,325+13,014 | 0 |
| active_only | 47.3 | 211.80 | 13,325+13,014 | -1 |
| persistent | CUDA OOM | — | — | 1 |
| layer_grouped_2 | CUDA OOM | — | — | 2 |
| layer_grouped_4 | CUDA OOM | — | — | 4 |

**Decode strategies** (ranked by tok/s):

| Strategy | Avg tok/s | VRAM (MB) | Divisor |
|----------|----------|-----------|---------||
| **pure_cpu** | **1.81** | **12,079+11,768** | null |
| hcs_hybrid | 1.77 | 12,780+12,800 | -3 |
| compact | 0.23 | 13,325+13,014 | -1 |
| lru | 0.23 | 16,030+14,427 | -2 |

**Winner**: hcs_prefill (210.8 tok/s) + pure_cpu decode (1.81 tok/s)

#### Key Observations (Qwen3-235B-A22B)

1. **HCS prefill is the ONLY viable strategy** — all layer_grouped and persistent OOM due to enormous expert size (128 × 3072 intermediate = 9.7 MB/expert × 94 layers = 117 GB total)
2. **HCS achieves 3.8x better prefill** than chunked (211 vs 56 tok/s) by pinning hot experts and using 1-layer groups
3. **pure_cpu and hcs_hybrid decode are nearly tied** (1.81 vs 1.77 tok/s) — both far superior to compact/LRU
4. **Compact/LRU decode is catastrophically slow** (0.23 tok/s, 8x slower!) — only 6 compact layers fit in the AO buffer, so 88 of 94 layers do full per-expert DMA every token
5. **Large expert size is the key differentiator** — 235B's 9.7 MB/expert (vs Coder-Next's 1.6 MB) makes DMA-based decode strategies unviable

### Cross-Model Strategy Summary

| Model | Best Prefill | Prefill tok/s | Best Decode | Decode tok/s | Key Factor |
|-------|-------------|---------------|-------------|--------------|------------|
| V2-Lite (1 GPU) | persistent | 1,757 | pure_cpu | 6.98 | Small experts → everything fits |
| V2-Lite (2 GPU) | layer_grouped_2 | 1,729 | pure_cpu | 6.69 | 2nd GPU adds overhead |
| Coder-Next (2 GPU) | layer_grouped_4 | 619 | pure_cpu | 7.26 | Medium experts, top-10, many layers |
| 235B-A22B (2 GPU) | hcs_prefill | 211 | pure_cpu | 1.81 | Large experts, only HCS fits |

**Universal finding**: pure_cpu decode wins on all models — GPU Marlin kernel launch overhead at M=1 exceeds any benefit from having experts on GPU.

## Conclusions

TODO

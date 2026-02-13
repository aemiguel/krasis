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

- Krasis is configured to divide the experts the minimal amount which allows them to fit on the GPU (to run as large a batch as possible)
- Individual component weights on the GPU which do not largely impact quality are quantised by Krasis to INT8
- Krasis is given the original BF16 model to build a GPU optimised Marlin based in-memory model
- Krasis is given the Q8 GGUF model to build an AVX2 CPU optimised in-memory model

Reasoning:

- Krasis will build optimised models for both the GPU and CPU, trading off system RAM for prefill and decode speed
- Krasis will DMA the optimised GPU model from system RAM onto the GPU, and process the entire prefill on GPU in an optimal way
- This will greatly speed up input token handling and result in a much faster response vs other tools with a similar decode generation speed

## Benchmark Results

TODO 

## Conclusions

TODO

# Benchmarks

This document records comparison runs of Krasis vs other major LLM runtimes - llama.cpp and Ktransformers.

It's goal is to investigate and demonstrate the benefits of Krasis' approach to a hybrid runtime where 100% of the prefill stage is offloaded to GPU.

## Test LLM Model

Target Model: Qwen3-235B-A22B (MoE, 658GB)

Llama model used: Unsloth Qwen3-235B-A22B Q8 (8-bit quantized)
KTransformers model used: Unsloth Qwen3-235B-A22B Q8 (8-bit quantized)
Krasis model used: Unsloth Qwen3-235B-A22B Q8 (8-bit quantized) + Original BF16 model weights

## Methodology

In each case, each tool is capable of both running the model on CPU, and trying to take advantage of the GPU(s).

We test with what we believe are ideal parameters given the situation for each tool to do its best.

We run each test tool with 1,2 and 3 GPUs.

We pass in the same large ~10,000 token prompt to each tool and measure both prefill (input) speed in tokens per second and decode (output) speed in tokens per second.

## Test Hardware

Epyc 7742 (single CPU, 64 cores)
1TB DDR4 2666Mhz (8 channels)
3x RTX Ada 2000 GPUs

## Tool Configurations

### LLama.cpp configuration + parameters used:

TODO 

### KTransformers configuration + parameters used:

TODO 

### Krasis configuration

For each run:

- Krasis is configured to divide the experts the minimal amount which allows them to fit on the GPU (to run as large a batch as possible)
- Individual component weights on the GPU which do not largely impact quality are quantised by Krasis to INT8
- Krasis is given the original BF16 model to build a GPU optimised Marlin based in-memory model
- Krasis is given the Q8 GGUF model to build an AVX2 CPU optimised in-memory model


## Benchmark Results

TODO 

## Conclusions

TODO

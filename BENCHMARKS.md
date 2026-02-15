# Benchmarks

Standardized benchmark results for Krasis. Full logs in [`benchmarks/`](benchmarks/).

## Hardware

- **CPU**: AMD EPYC 7742 (64 cores, 1 socket, AVX2)
- **RAM**: 995 GB DDR4 2666MHz (8 channels)
- **GPUs**: 3x NVIDIA RTX 2000 Ada (16 GB each, 48 GB total)
- **PCIe**: 4.0 x8 per GPU (~16 GB/s each)

## Methodology

- 10,000 token prefill prompt (technical content), 64 decode tokens, 3 runs averaged
- TTFT measured wall-clock, all instrumentation disabled during timing
- Filename: `<model>_<source>_<Ngpu>_<gpu_quant>_<cpu_quant>.log`

## Krasis Results

### DeepSeek-V2-Lite (27 layers, 64 experts, top-6, MLA)

| GPUs | GPU Quant | CPU Quant | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Log |
|------|-----------|-----------|-----------------|----------|----------------|--------|-----|
| 1 | INT4 | INT4 | 2,171 | 4.62 | 5.68 | 177 | [log](benchmarks/DeepSeek-V2-Lite_native_1gpu_int4gpu_int4cpu.log) |
| 1 | INT4 | INT8 | 2,173 | 4.61 | 4.57 | 219 | [log](benchmarks/DeepSeek-V2-Lite_native_1gpu_int4gpu_int8cpu.log) |

Config: PP=[27], persistent prefill, pure_cpu decode, FP8 KV, INT8 attention

### Qwen3-Coder-Next (48 layers, 512 experts, top-10, hybrid 12 GQA + 36 linear)

| GPUs | GPU Quant | CPU Quant | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Log |
|------|-----------|-----------|-----------------|----------|----------------|--------|-----|
| 2 | INT4 | INT4 | 546 | 18.6 | 10.03 | 100 | [log](benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu.log) |
| 2 | INT4 | INT8 | 546 | 18.6 | 6.32 | 160 | [log](benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int8cpu.log) |

Config: PP=[24,24], layer_grouped(4) prefill, pure_cpu decode, FP8 KV, INT8 attention

### Qwen3-235B-A22B (94 layers, 128 experts, top-8, MLA)

| GPUs | GPU Quant | CPU Quant | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Log |
|------|-----------|-----------|-----------------|----------|----------------|--------|-----|
| 2 | INT4 | INT4 | 196 | 51.1 | 1.37 | 733 | [log](benchmarks/Qwen3-235B-A22B_native_2gpu_int4gpu_int4cpu.log) |

Config: PP=[47,47], HCS prefill (1-layer groups), pure_cpu decode, FP8 KV, INT8 attention

## Comparison vs Other Tools (Qwen3-235B-A22B, ~8,600 token prompt)

| Tool | GPUs | Prefill (tok/s) | Decode (tok/s) | TTFT (s) |
|------|------|-----------------|----------------|----------|
| llama.cpp | 1 | 30.1 | 3.5 | ~286 |
| llama.cpp | 2 | 31.6 | 3.7 | ~272 |
| llama.cpp | 3 | 32.9 | 3.8 | ~261 |
| KTransformers | 1 | 49.7 | 4.85 | 173.0 |
| KTransformers | 2 | 57.5 | 3.60 | 149.5 |
| KTransformers | 3 | 57.3 | 3.29 | 150.1 |
| **Krasis** | **2** | **198.1** | **1.65** | **43.9** |

---

## Historical Strategy Analysis

Auto-optimiser results comparing prefill/decode strategies. Internal development benchmarks.

### DeepSeek-V2-Lite — 1 GPU

Winner: persistent (1,757 tok/s) + pure_cpu (6.98 tok/s)

| Prefill | tok/s | Decode | tok/s |
|---------|-------|--------|-------|
| persistent | 1,757 | pure_cpu | 6.98 |
| layer_grouped_2 | 1,746 | hcs_hybrid | 4.88 |
| chunked | 792 | compact | 3.31 |
| active_only | 303 | lru | 3.31 |

### Qwen3-Coder-Next — 2 GPUs

Winner: layer_grouped_4 (619 tok/s) + pure_cpu (7.26 tok/s)

| Prefill | tok/s | Decode | tok/s |
|---------|-------|--------|-------|
| layer_grouped_4 | 619 | pure_cpu | 7.26 |
| hcs_prefill | 603 | hcs_hybrid | 7.02 |
| chunked | 155 | compact | 4.99 |
| persistent | OOM | lru | 4.90 |

### Qwen3-235B-A22B — 2 GPUs

Winner: hcs_prefill (211 tok/s) + pure_cpu (1.81 tok/s)

| Prefill | tok/s | Decode | tok/s |
|---------|-------|--------|-------|
| hcs_prefill | 211 | pure_cpu | 1.81 |
| chunked | 56 | hcs_hybrid | 1.77 |
| active_only | 47 | compact | 0.23 |
| persistent | OOM | lru | 0.23 |

**Universal finding**: pure_cpu decode wins on all models — GPU Marlin kernel launch overhead at M=1 exceeds any benefit from having experts on GPU.

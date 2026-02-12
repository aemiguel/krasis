# Krasis Benchmarks

## Environment
- **CPU**: AMD EPYC 7742 (64 cores, AVX2 only, 995 GB RAM)
- **GPU**: 1x NVIDIA RTX 2000 Ada (16 GB, compute 8.9)
- **CUDA**: 12.6, driver 12.8
- **Krasis**: INT4 GPU (Marlin) + INT4 CPU (optimized), BF16 attention/lm_head

## DeepSeek-V2-Lite (27 layers, 64 experts, PP=1, 1 GPU)

### GPU Prefill Scaling (2026-02-12)

| Prompt Tokens | Wall Time | tok/s | ms/tok | Notes |
|:---:|:---:|:---:|:---:|-------|
| 100 | 11.13s | 9.0 | 111.3 | Fixed overhead dominates |
| 500 | 10.61s | 47.1 | 21.2 | |
| 825 | 10.69s | 77.2 | 13.0 | Dual-format validation run |
| 1,000 | 10.69s | 93.6 | 10.7 | |
| 2,000 | 10.96s | 182.5 | 5.5 | |
| 5,000 | 11.97s | 417.9 | 2.4 | |
| 9,903 | 14.28s | **693.3** | 1.4 | Production-scale prompt |

- Fixed overhead: ~10.5s (DMA copy 7.3 GB + attention + norms per 26 layers)
- Marginal GPU compute: ~3,100 tok/s (pure fused_marlin_moe throughput)

### CPU Decode (2026-02-12)

| Context Length | Avg (ms) | P50 (ms) | Min (ms) | Max (ms) | tok/s |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 177.0 | 175.3 | 162.0 | 227.8 | 5.6 |
| 1,000 | 177.0 | 175.8 | 142.8 | 192.1 | 5.7 |
| 10,000 | 202.2 | 200.8 | 152.2 | 224.8 | 4.9 |

- Consistent ~177ms up to 1K context, +25ms at 10K (MLA attention scaling)
- Decode speed: 4.9-5.7 tok/s

### Model Loading

| Metric | Value |
|--------|-------|
| First-run cache build | 256.7s |
| Model load (from cache) | 14.2s |
| GPU weights load | 0.9s |
| CPU expert load | 9.8s |
| GPU prefill init | 3.3s |
| GPU Marlin cache | 7.2 GB |
| CPU INT4 cache | 7.0 GB |
| GPU VRAM (weights) | 2,924 MB |
| GPU VRAM (prefill buffers) | 285.5 MB |
| KV cache (auto-sized) | 6,290 MB (212K tokens) |

### Correctness
- "2+2" → "4" PASS
- "Capital of France" → "Paris" PASS
- 10K-token prompt → coherent summary PASS

## Raw Data
See `benchmarks.jsonl` and `timing_analysis.json` for machine-readable results.

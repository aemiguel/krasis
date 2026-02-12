# Krasis Benchmarks

## Environment
- **CPU**: AMD EPYC 7742 (64 cores, AVX2 only, 995 GB RAM)
- **GPU**: 1x NVIDIA RTX 2000 Ada (16 GB, compute 8.9)
- **CUDA**: 12.6, driver 12.8
- **Krasis**: INT4 GPU (Marlin) + INT4 CPU (optimized), BF16 attention/lm_head

## DeepSeek-V2-Lite (27 layers, 64 experts, PP=1, 1 GPU)

### GPU Prefill (2026-02-12)

| Prompt Tokens | Chunks | Wall Time | tok/s | Notes |
|:---:|:---:|:---:|:---:|-------|
| 512 | 1 | 11.17s | 45.8 | |
| 1,024 | 1 | 10.71s | 95.7 | |
| 2,048 | 1 | 11.01s | 184.4 | Max single-pass on 16GB |
| 4,047 | 2 | 22.12s | 183.0 | |
| 8,079 | 4 | 45.79s | 176.4 | |
| 9,903 | 5 | 57.26s | 173.0 | |

- Per-chunk overhead: ~11.2s (DMA 288 MB × 26 layers = 7.3 GB per chunk)
- Marginal compute: ~5,000 tok/s (pure Marlin kernel throughput)

### CPU Decode (2026-02-12)

| Context | Avg (ms) | P50 (ms) | P90 (ms) | Min (ms) | Max (ms) | tok/s |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 512 | 168.8 | — | — | — | — | 5.9 |
| 2,048 | 170.2 | 170.5 | 173.5 | 158.9 | 175.1 | 5.9 |
| 8,192 | 173.1 | — | — | — | — | 5.8 |

- Very stable across context lengths (+4ms from 512→8K)
- P90/P50 ratio: 1.02x (no jitter)

### Model Loading

| Metric | Value |
|--------|-------|
| First-run cache build | 256.7s |
| Model load (from cache) | 14.2s |
| GPU weights | 0.9s |
| CPU experts | 9.8s |
| GPU prefill init | 3.3s |
| GPU Marlin cache | 7.2 GB |
| CPU INT4 cache | 7.0 GB |
| GPU VRAM (weights) | 2,924 MB |
| GPU VRAM (prefill buffers) | 285.5 MB |

### Correctness
- "2+2" → "4" PASS
- "Capital of France" → "Paris" PASS
- 10K-token prompt → coherent summary PASS

## Raw Data
- `benchmarks.jsonl` — per-run results
- `profile_results.json` — detailed timing analysis
- `timing_analysis.json` — scaling data

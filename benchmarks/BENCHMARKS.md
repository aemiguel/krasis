# Krasis Benchmark Results

## GPU Decode Benchmark — 2026-03-02 (5090, 40% HCS, pinned memory)

**Hardware:** EPYC 7742, 995 GB RAM, 1x RTX 5090 32 GB, PCIe 4.0 x16 (27 GB/s peak).

**Config:** QCN (Qwen3-Coder-Next), INT4 GPU/CPU, BF16 attention, LGS=2, GPU decode (Rust, zero GIL), HCS 40.2% (9,869/24,576 experts), pinned expert memory for async DMA, no debug/timing instrumentation.

### Decode Speed

| Prompt | Tokens | Decode Time | Decode Speed | TTFT |
|--------|-------:|------------:|-------------:|-----:|
| Short (math) | 199 | 7.04s | 28.1 tok/s | 5.31s |
| Medium (caches) | 499 | 16.43s | 30.3 tok/s | 5.33s |
| Code (BST) | 499 | 17.29s | 28.8 tok/s | 5.30s |
| Long (essay) | 799 | 23.16s | 34.5 tok/s | 5.30s |
| **Average** | | | **30.4 tok/s** | **5.31s** |

### Prefill Speed (from server log, includes ~5.3s layer streaming overhead)

| Input Tokens | TTFT | Prefill Compute (est) | Prefill Speed (est) |
|-------------:|-----:|----------------------:|--------------------:|
| 64 | 5.29s | ~0.0s | n/a (streaming dominated) |
| 576 | 5.31s | ~0.01s | n/a |
| 1,126 | 5.28s | ~0.0s | n/a |
| 2,236 | 5.29s | ~0.0s | n/a |
| 4,456 | 5.32s | ~0.02s | ~838 tok/s |
| 8,906 | 5.32s | ~0.02s | ~1,674 tok/s |
| 17,796 | 5.69s | ~0.39s | ~3,129 tok/s |
| 27,796 | 7.64s | ~2.34s | ~3,636 tok/s |

### Notes
- TTFT is dominated by layer group streaming (~5.3s constant overhead for lgs=2, 48 layers, 24 groups)
- Actual prefill compute only becomes visible above ~8K tokens
- Peak prefill throughput: ~3,636 tok/s at 28K tokens
- Decode speed varies 28-35 tok/s, higher on longer outputs (routing stabilizes)
- VRAM: 24,923 MB used, 7,196 MB free during decode, lowest watermark 2,888 MB (during 28K prefill)
- Rust KV cache limited to 8,192 tokens (prompts >8K skip decode)

Full log: [20260302-gpu-decode-5090-qcn-40pct-hcs.log](../logs/benchmarks/20260302-gpu-decode-5090-qcn-40pct-hcs.log)

---

## Standard Benchmarks — 2026-02-27 (Rust server, unified timing)

**Hardware:** EPYC 7742 (64 cores, 4 NUMA nodes), DDR4-2666 8-channel, 1x RTX 2000 Ada 16 GB, PCIe 4.0 x8.

Config: 20K–50K token prompts, FP8 KV cache, BF16 attention, INT8 shared_expert/dense_mlp/lm_head, 40 CPU threads, NUMA thread pinning + interleaved allocation, LGS=2, pure CPU decode, Rust HTTP server with ring buffer SSE.

| Model | GPUs | GPU/CPU bits | Engine Prefill | Engine Decode | Network Prefill | Network Decode | Overhead | Log |
|-------|-----:|-------------:|---------------:|--------------:|----------------:|---------------:|---------:|-----|
| Qwen3-Coder-Next | 1 | INT4/INT4 | 1,003 tok/s | 12.97 tok/s | 932 tok/s | 12.13 tok/s | 7.1% / 6.5% | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |

### Key changes from previous benchmarks

- **Rust-internal timing**: Both engine and network decode use Rust `Instant` timers. Previous Python timing included `torch.cuda.synchronize()` overhead, making engine decode appear 33% slower than network (impossible).
- **Ring buffer SSE**: Decode loop pushes to mpsc channel, writer thread flushes every 100ms. First token flushed immediately for accurate TTFT.
- **Unified tokenization**: Both paths use `apply_chat_template(enable_thinking=False)`. Network sends text (not pre-tokenized IDs).
- **Model warmup before benchmarks**: Full generate cycle runs before any measurement, paying all cold-start costs.

---

## Standard Benchmarks — 2026-02-25 (NUMA-optimized, 1 GPU)

**Hardware:** EPYC 7742 (64 cores, 4 NUMA nodes), DDR4-2666 8-channel, 1x RTX 2000 Ada 16 GB, PCIe 4.0 x8.

Config: 10K–50K token prompts, FP8 KV cache, BF16 attention, INT8 shared_expert/dense_mlp/lm_head, 40 CPU threads, NUMA thread pinning + interleaved allocation, LGS=2, pure CPU decode.

| Model | GPUs | GPU/CPU bits | Prefill (tok/s) | TTFT @ 20K | Decode (tok/s) | ms/tok | Log |
|-------|-----:|-------------:|----------------:|:----------:|:--------------:|:------:|-----|
| Qwen3-Coder-Next | 1 | INT4/INT4 | 1,056.6 | 18.9s | 15.81 | 63.6 | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 1 | INT8/INT8 | 873.2 | 40.1s | 12.41 | 80.6 | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int8gpu_int8cpu_stream_lgs2.log) |
| DeepSeek-V2-Lite | 1 | INT4/INT4 | 1,476.5 | 13.6s | 20.18 | 49.7 | [log](../logs/benchmarks/DeepSeek-V2-Lite_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| DeepSeek-V2-Lite | 1 | INT8/INT8 | 1,316.9 | 15.2s | 17.84 | 56.2 | [log](../logs/benchmarks/DeepSeek-V2-Lite_native_1gpu_int8gpu_int8cpu_stream_lgs2.log) |

### Key improvements over previous benchmarks

- **NUMA-aware thread pinning**: rayon threads pinned round-robin across 4 NUMA nodes via sched_setaffinity. Eliminates cross-node memory traffic.
- **MPOL_INTERLEAVE**: Weight mmap pages spread across all memory controllers. 4x aggregate DRAM bandwidth.
- **MLA AVX2 kernels**: w_kc/w_vc absorption and attention vectorized with parallel head dispatch.
- **Combined effect**: QCN decode 7.89 → 15.81 tok/s (+100%), V2-Lite decode 6.22 → 20.18 tok/s (+224%).
- **Note**: Decode numbers from Feb 25 used Python timing that included cuda.synchronize() — may be slightly pessimistic. Feb 27 numbers use Rust-internal timing.

---

## Previous Benchmarks — 2026-02-22 (pre-NUMA, multi-GPU)

**Hardware:** EPYC 7742, DDR4-2666 8-channel, 3x RTX 2000 Ada 16 GB, 1 NUMA node (NPS1), 48 CPU threads.

Config: 10K token prompt, FP8 KV cache, INT8 attention/shared_expert/dense_mlp/lm_head.
Default: pure CPU MoE decode (no HCS), streamed attention with double buffering.

| Model | GPUs | GPU/CPU bits | LGS | HCS | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Status | Log |
|-------|-----:|-------------:|----:|-----|----------------:|---------:|---------------:|-------:|--------|-----|
| DeepSeek-V2-Lite | 1 | INT8/INT8 | 2 | ON | 1882.8 | 5.32 | 3.04 | 328.8 | PASS | [log](../logs/benchmarks/) |
| DeepSeek-V2-Lite | 2 | INT4/INT4 | 2 | ON | 1623.1 | 6.16 | 6.22 | 160.9 | PASS | [log](../logs/benchmarks/) |
| Qwen3-Coder-Next | 1 | INT8/INT8 | 2 | ON | 696.4 | 14.36 | 5.93 | 168.6 | PASS | [log](../logs/benchmarks/) |
| Qwen3-Coder-Next | 1 | INT4/INT4 | 2 | ON | 979.6 | 10.21 | 7.89 | 126.8 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 1 | INT4/INT4 | 2 | OFF | 1097.4 | 18.23 | 8.12 | 123.4 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | ON | 880.2 | 11.36 | 8.15 | 122.8 | FAIL* | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | multi | 806.8 | 12.39 | 9.14 | 109.4 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_multigpu_hcs.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | ON | 859.6 | 11.63 | 7.21 | 138.8 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 4 | ON | 845.2 | 11.83 | 7.21 | 138.7 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_stream_lgs4.log) |
| gpt-oss-120b | 1 | INT8/INT8 | 2 | ON | 516.1 | 19.38 | 3.59 | 278.7 | PASS | [log](../logs/benchmarks/) |
| gpt-oss-120b | 2 | INT4/INT4 | 2 | ON | 825.7 | 12.11 | 5.17 | 193.6 | PASS | [log](../logs/benchmarks/) |
| Qwen3-235B-A22B | 1 | INT4/INT4 | 2 | OFF | 369.7 | 27.05 | 1.58 | 632.1 | PASS | [log](../logs/benchmarks/Qwen3-235B-A22B_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-235B-A22B | 2 | INT4/INT4 | 2 | OFF | 214.2 | 46.69 | 1.58 | 635.3 | PASS | [log](../logs/benchmarks/Qwen3-235B-A22B_native_2gpu_int4gpu_int4cpu_stream_lgs2.log) |

### Column Legend

- **LGS**: Layer Group Size — number of layers streamed through GPU at a time (double-buffered). Lower = less VRAM, more DMA rounds.
- **HCS**: Hot-Cache Strategy — ON = GPU-cached experts for decode, OFF = pure CPU decode, multi = HCS on all GPUs.

### Notes

- **Pure CPU decode** (HCS OFF) is now default. QCN pure CPU decode (7.82 tok/s) beats HCS ON (7.21 tok/s) because GPU Marlin M=1 overhead exceeds CPU AVX2 INT4 cost for QCN's tiny experts (intermediate=512).
- **Heatmap overhead fix**: Disabling heatmap collection during normal inference improved QCN decode from 7.38 to 7.82 tok/s (+6%). Heatmap accumulation called torch.unique() per MoE layer per token — unnecessary when HCS is off.
- **Qwen3-235B-A22B** now runs on 1 GPU thanks to streaming attention (94 MLA layers streamed through ~136 MB double buffers instead of 6.5 GB persistent). Previously OOM'd.
- **Qwen3-235B-A22B 1 GPU vs 2 GPU**: Decode identical (1.58 tok/s, all CPU). Prefill 73% faster on 1 GPU (369.7 vs 214.2 tok/s) — second GPU adds cross-device DMA overhead with no benefit.
- **QCN 2gpu INT4/INT4 FAIL***: Prefill and decode speeds are valid, but decode output is garbage (cross-GPU HCS expert corruption).
- **QCN 2gpu multi-HCS**: HCS experts on both GPUs (11,279 total). Decode 9.14 tok/s (slower than pure CPU 10.57 due to CPU bounce overhead).
- **QCN 2gpu stream lgs=2 vs lgs=4**: Nearly identical performance. lgs=2 slightly better for VRAM headroom.

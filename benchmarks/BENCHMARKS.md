# Krasis Benchmark Results

## Standard Benchmarks — 2026-04-03 (QCN Polar4 AWQ speed-test rerun on c35d9b0)

Hardware: EPYC 7742, 995 GB RAM, 1x RTX 5090 32 GB used for benchmark, 2x RTX 5090 present.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| c35d9b0 speed-test rerun | FAIL before timed benchmark | 84.8 tok/s short calibration only | not reached | 3262 MB during short prefill probe | [log](20260403_174355_qcn_polar4_awq_5090_failed_illegal_address.log) |

Notes:
- Run executed via `./dev speed-test` on detached `c35d9b0`.
- Load, warmup, decode-store setup, and short calibration passed.
- Failure occurred in long VRAM calibration at 39,920 prompt tokens inside `gpu_store.rust_prefill_tokens(...)`.
- Error: `CUDA_ERROR_ILLEGAL_ADDRESS (grid=(39920, 1, 1), block=(1024, 1, 1), smem=4096, nparams=6)`.
- Cleanup also hit a Rust destructor panic while tearing down `GpuDecodeStore` after the illegal address.

---

## Standard Benchmarks — 2026-04-01 (QCN Polar4 AWQ padding rewrite)

Hardware: EPYC 7742, 995 GB RAM, 1x RTX 5090 32 GB used for benchmark, 2x RTX 5090 present.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| Intermediate rewrite (cached dummy ptrs + real-expert alias padding) | 7,398.3 | 90.99 | 17010/24576 (69.2%) | 686 MB | [report](../logs/dev-benchmark_20260401_083831/benchmark_report.log) |
| Final rewrite (cached dummy ptrs + dummy-only zero-weight padding) | 7,769.5 | 96.43 | 17010/24576 (69.2%) | 686 MB | [report](20260401_084451_qcn_polar4_awq_5090_padding_rewrite.log) |

Notes:
- The first rewrite removed only the per-step `cuMemcpyDtoH` and did not recover the regression.
- The final rewrite shows the remaining loss came from aliasing zero-weight slots onto a real expert during replay.
- Standard benchmark log archived at `benchmarks/20260401_084451_qcn_polar4_awq_5090_padding_rewrite.log`.

---

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

## Standard Speed Benchmark — 2026-04-03 (resolution, BF-01 host ptr-table base import)

Hardware: 1x RTX 5090

Config: Qwen3-Coder-Next, INT4 experts, AWQ attention, Polar4 KV, standard command `./dev speed-test`, timing instrumentation off.

| Date | Commit | Change | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Status | Log |
|------|--------|--------|----------------:|---------------:|-------------------:|-----|--------------:|--------|-----|
| 2026-04-03 18:53 | 83dd3b0 + local BF-01 | Pointer-table fused MoE host base set to null in ptr-table mode; BF-02 already present | 7,584.7 | 100.38 | 138.95 | 16929/24576 (68.9%) | 738 MB | PASS | [log](20260403_185345_qcn_polar4_awq_5090_bf01_host_null_base.log) |
| 2026-04-03 19:20 | 83dd3b0 + local BF-01 + BF-03 cache | Cache fused MoE `C_tmp` floor calculation once per model/device config and reuse in hot path | FAIL before timed benchmark | 80.6 tok/s short calibration only | not reached | not reached | 3248 MB during short prefill probe | FAIL | [log](20260403_192025_qcn_polar4_awq_5090_bf03_cached_ctmp_failed_illegal_address.log) |
| 2026-04-03 19:26 | 83dd3b0 + local BF-01, BF-03 cache reverted | Revert the one-time `C_tmp` cache follow-up and rerun standard speed test | FAIL before timed benchmark | 80.6 tok/s short calibration only | not reached | not reached | 3248 MB during short prefill probe | FAIL | [log](20260403_192634_qcn_polar4_awq_5090_bf03_revert_failed_illegal_address.log) |
| 2026-04-03 19:38 | 83dd3b0 + local BF-01 + BF-03 cache | Post-reboot rerun with BF-03 one-time `C_tmp` cache reapplied | 7,844.1 | 99.29 | 182.51 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_193344_qcn_polar4_awq_5090_bf03_reapplied_post_reboot.log) |
| 2026-04-03 19:51 | 3b36240 + local BF-04 clean import | Replace drifting ptr-table fused-MoE `B` progression with explicit expert base + signed slice rebasing on expert/block transitions | 7,513.1 | 98.68 | 127.13 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_195132_qcn_polar4_awq_5090_bf04_clean_rebase.log) |
| 2026-04-03 20:11 | fb49b0f + local BF-05 clean import | Keep ptr-table `B` fetch source indices signed through slice rewinds and guard the hazard path with `cp_async4_pred` | 7,515.0 | 97.82 | 128.37 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_201155_qcn_polar4_awq_5090_bf05_signed_fetch_guard.log) |
| 2026-04-03 20:36 | 4aa3bee + local no-valid-block guard | Exit `update_next_moe_block_data()` cleanly when invalid-block scanning reaches the padded tail without finding another valid expert block | 7,606.7 | 99.88 | 137.25 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_203640_qcn_polar4_awq_5090_no_valid_block_guard.log) |

Notes:
- The BF-03 cache edit built cleanly through `./dev build`.
- This benchmark failed in long VRAM calibration at `39,920` prompt tokens before the timed benchmark section.
- Failure remained `CUDA_ERROR_ILLEGAL_ADDRESS (grid=(39920, 1, 1), block=(1024, 1, 1), smem=4096, nparams=6)`.
- Reverting the BF-03 cache follow-up did not restore a successful run on this attempt; the same long-calibration illegal-address fault reproduced with the same short-calibration numbers.
- After reboot, the same BF-03 cache state completed the full standard benchmark cleanly and produced the best prefill result in this local series.
- The BF-04 clean import also completed the full standard benchmark cleanly on the same branch state, but did not improve throughput versus the earlier post-reboot BF-03 pass.
- The BF-05 clean import also completed the full standard benchmark cleanly; throughput stayed effectively flat versus BF-04 while preserving the signed rewind-safe pointer-table fetch path.
- The no-valid-block guard also completed the full standard benchmark cleanly; it is a cheap control-flow correctness fix and did not regress standard prefill or decode throughput.

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

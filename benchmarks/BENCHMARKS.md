# Krasis Benchmark Results

## Standard Benchmarks — 2026-02-22 (10K prompt, pure CPU decode default)

Config: 10K token prompt, FP8 KV cache, INT8 attention/shared_expert/dense_mlp/lm_head, 48 CPU threads.
Default: pure CPU MoE decode (no HCS), streamed attention with double buffering.

| Model | GPUs | GPU/CPU bits | LGS | HCS | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Status | Log |
|-------|-----:|-------------:|----:|-----|----------------:|---------:|---------------:|-------:|--------|-----|
| DeepSeek-V2-Lite | 1 | INT8/INT8 | 2 | ON | 1882.8 | 5.32 | 3.04 | 328.8 | PASS | [log](suite_logs/) |
| DeepSeek-V2-Lite | 2 | INT4/INT4 | 2 | ON | 1623.1 | 6.16 | 6.22 | 160.9 | PASS | [log](suite_logs/) |
| Qwen3-Coder-Next | 1 | INT8/INT8 | 2 | ON | 696.4 | 14.36 | 5.93 | 168.6 | PASS | [log](suite_logs/) |
| Qwen3-Coder-Next | 1 | INT4/INT4 | 2 | ON | 979.6 | 10.21 | 7.89 | 126.8 | PASS | [log](Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 1 | INT4/INT4 | 2 | OFF | 1024.4 | 9.78 | 7.82 | 127.9 | PASS | [log](Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | ON | 880.2 | 11.36 | 8.15 | 122.8 | FAIL* | [log](Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | multi | 806.8 | 12.39 | 9.14 | 109.4 | PASS | [log](Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_multigpu_hcs.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | ON | 859.6 | 11.63 | 7.21 | 138.8 | PASS | [log](Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 4 | ON | 845.2 | 11.83 | 7.21 | 138.7 | PASS | [log](Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_stream_lgs4.log) |
| gpt-oss-120b | 1 | INT8/INT8 | 2 | ON | 516.1 | 19.38 | 3.59 | 278.7 | PASS | [log](suite_logs/) |
| gpt-oss-120b | 2 | INT4/INT4 | 2 | ON | 825.7 | 12.11 | 5.17 | 193.6 | PASS | [log](suite_logs/) |
| Qwen3-235B-A22B | 1 | INT4/INT4 | 2 | OFF | 369.7 | 27.05 | 1.58 | 632.1 | PASS | [log](Qwen3-235B-A22B_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-235B-A22B | 2 | INT4/INT4 | 2 | OFF | 214.2 | 46.69 | 1.58 | 635.3 | PASS | [log](Qwen3-235B-A22B_native_2gpu_int4gpu_int4cpu_stream_lgs2.log) |

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

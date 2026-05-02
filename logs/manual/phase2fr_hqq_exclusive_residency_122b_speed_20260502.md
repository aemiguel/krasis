# Phase 2FR: HQQ Exclusive Residency 122B Speed Test

Date: 2026-05-02

## Goal

Fix the remaining HQQ6/k4v4 VRAM issues found after Phase 2FQ, then rerun the
122B speed benchmark.

## Fixes

- HQQ attention is now registered once for the primary runtime path, not once
  for the decode store and again for the prefill engine.
- HQQ modes skip the normal BF16/Marlin attention registration branches, so
  HQQ-managed projections are not also registered as BF16 attention weights.
- After HQQ runtime descriptors are built, replaced BF16 attention projection
  tensors are released from Python layer objects and PyTorch cached blocks are
  returned to the driver.
- HQQ RoPE table setup is registered explicitly because HQQ modes bypass the
  normal GQA attention registration branch.

## Memory Validation

Bounded diagnostic:

- log: `logs/manual/phase2fr_q122b_hqq6_exclusive_residency_diag_cleanup_20260502.log`
- one HQQ runtime staging line at `device_mb=3696.47`
- released `108` BF16 attention projection tensors, `6075.00 MB`
- post-calibration free VRAM: `19904 MB`
- reclaimable HCS budget: `19292 MB`

Full benchmark startup:

- one HQQ runtime staging line at `device_mb=3696.47`
- released `108` BF16 attention projection tensors, `6075.00 MB`
- long calibration post-cleanup free VRAM: `19942 MB`
- reclaimable HCS budget: `19330 MB`
- startup HCS: `3888/12288 (31.6%)`, `18154.1 MB`

The HQQ6/k4v4 startup HCS is now above the previous BF16/fp8 control
(`3537/12288`, `28.8%`), so the main HQQ residency bug is fixed for this
surface.

## 122B Benchmark Result

Command:

```bash
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Logs:

- `logs/manual/phase2fr_q122b_hqq6_exclusive_residency_benchmark_20260502.log`
- `benchmarks/20260502_phase2fr_q122b_k4v4_hqq6_exclusive_residency_benchmark.log`
- `logs/dev-benchmark_20260502_180904/benchmark_report.log`

Config:

- model: `Qwen3.5-122B-A10B`
- GPU experts: INT4 Marlin
- CPU experts: INT4
- attention: HQQ6
- KV: k4v4
- shared/dense/lm-head: INT8
- GPU: one RTX 5090 32 GB selected

Results:

| Metric | Result |
| --- | ---: |
| Best prefill | `2070.2 tok/s` |
| Best internal decode | `9.03 tok/s` |
| Best network round trip | `14.62 tok/s` |
| Startup HCS | `3888/12288 (31.6%)` |
| Final HCS | `3456/12288 (28.1%)` |
| Decode min free VRAM | `672 MB` |
| Lowest prefill free VRAM warning | `518 MB` |

Timed prefill rows:

| Prompt | Speed | Time |
| ---: | ---: | ---: |
| 1,000 | `231.6 tok/s` | `4317.6 ms` |
| 5,000 | `1155.7 tok/s` | `4326.2 ms` |
| 10,000 | `1884.0 tok/s` | `5307.9 ms` |
| 20,000 | `2070.2 tok/s` | `9661.0 ms` |
| 35,000 | `1819.0 tok/s` | `19241.1 ms` |
| 50,000 | `1820.9 tok/s` | `27459.5 ms` |

Decode rows:

| Target | Speed |
| ---: | ---: |
| 50 | `8.70 tok/s` |
| 100 | `8.88 tok/s` |
| 250 | `9.03 tok/s` |

Round-trip rows:

| Target | Speed |
| ---: | ---: |
| 50 | `14.62 tok/s` |
| 100 | `11.05 tok/s` |
| 250 | `9.64 tok/s` |

## Comparison

| Surface | Prefill | Decode | HCS |
| --- | ---: | ---: | ---: |
| Phase 2FM broken HQQ6/k4v4 | `541.9 tok/s` | `17.66 tok/s` | `2187/12288 (17.8%)` |
| Phase 2FQ stage-exact/chunkfix | `1377.3 tok/s` | `16.41 tok/s` | `1755/12288 (14.3%)` |
| Phase 2FR exclusive HQQ residency | `2070.2 tok/s` | `9.03 tok/s` | `3456/12288 (28.1%)` |
| BF16/fp8 control | `2765.6 tok/s` | `23.20 tok/s` | `3510/12288 (28.6%)` |

## Interpretation

The HQQ residency issue is fixed for this 122B HQQ6/k4v4 surface:

- duplicate HQQ runtime residency is gone
- BF16 attention projection residency is released
- reclaimable HCS budget is now `19330 MB`, above the BF16/fp8 control
- final HCS is effectively back to the BF16/fp8 coverage class

Prefill improved substantially but is still below the old BF16/fp8 surface.
Decode is now worse despite restored HCS coverage, so the next bottleneck is
not raw HCS capacity. The likely next diagnostic target is decode-time behavior
after HQQ exclusive residency: soft reload policy, HQQ decode swap state, graph
capture, or k4v4 decode path interaction.

VRAM headroom remains tight on the 2 GB k4v4 config. The run completed, but
prefill dipped to `518 MB` free, below the configured `600 MB` safety margin.

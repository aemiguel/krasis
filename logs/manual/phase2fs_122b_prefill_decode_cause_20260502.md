# Phase 2FS 122B Prefill/Decode Cause Investigation

Date: 2026-05-02

## Scope

Investigate why 122B HQQ6/k4v4 remains below the previous BF16/fp8 control:

- Prefill: `2070.2 tok/s` current vs `2765.6 tok/s` BF16/fp8 control
- Decode: `9.03 tok/s` current vs `23.20 tok/s` BF16/fp8 control

This report uses instrumentation runs only for attribution. Speed comparisons
should continue to use timing-free benchmark runs.

## Inputs

- Current timing run:
  `logs/manual/phase2fs_q122b_hqq6_k4v4_prefill_decode_timing_20260502.log`
- Graph/HCS validation run:
  `logs/manual/phase2fs_q122b_hcs_graph_validation_20260502.log`
- Prefill component timing run:
  `logs/manual/phase2fs_q122b_prefill_component_timing_20260502.log`
- BF16/fp8 control:
  `logs/manual/phase2ev_q122b_post_calibration_benchmark_20260501.log`

## Prefill Finding

The current HQQ6/k4v4 prefill is no longer chunk-cap broken. The bad 128-token
cap is gone:

- 39,920-token calibration: `chunk_size=7984`, `num_chunks=5`
- 25K benchmark warmups: `chunk_size=8334`, `num_chunks=3`

The remaining one-chunk gap is mostly HQQ projection/dequantization cost.

10K prefill component timing:

| Metric | HQQ6/k4v4 |
| --- | ---: |
| Total | `4143.2 ms` |
| Attention | `1310.3 ms` |
| MoE | `2494.8 ms` |
| HQQ projection internals | `930.8 ms` |

HQQ projection internals:

| Component | Time |
| --- | ---: |
| `marlin_float_zp` | `838.4 ms` |
| `group_sums` | `8.8 ms` |
| `correction_gemm` | `29.7 ms` |
| `correction_add` | `53.9 ms` |

The BF16/fp8 10K control row was `3251.9 ms`. The HQQ projection/dequant work
therefore accounts for almost the whole one-chunk prefill delta.

Long rows also lose throughput from smaller chunks:

| Prompt | HQQ6/k4v4 chunking | BF16/fp8 chunking |
| --- | ---: | ---: |
| 35K | `7000 x 5` | `8750 x 4` |
| 50K | `7143 x 7` | `8334 x 6` |

## Decode Finding

Decode capacity is not the bottleneck anymore. HQQ6/k4v4 has comparable or
better HCS capacity than the BF16/fp8 control, but the selected experts are not
the ones used by benchmark decode.

Timing-enabled decode comparison:

| Run | Cached experts | HCS experts/tok | Cold experts/tok |
| --- | ---: | ---: | ---: |
| HQQ6/k4v4 50 tok | `3267` | `103.4` | `280.6` |
| HQQ6/k4v4 100 tok | `3402` | `116.8` | `267.2` |
| HQQ6/k4v4 250 tok | `3456` | `119.7` | `264.3` |
| BF16/fp8 50 tok | `3510` | `250.0` | `134.0` |
| BF16/fp8 100 tok | `3510` | `235.1` | `148.9` |
| BF16/fp8 250 tok | `3510` | `219.7` | `164.3` |

State validation confirms the same pattern:

| Decode prompt | Resident experts | Cold routed uses |
| --- | ---: | ---: |
| `prompt61` | `3510` | `13434` |
| `prompt66` | `3537` | `26404` |
| `prompt61` | `3564` | `67301` |

For 50/100/250-token decode this is roughly `264-269` cold experts/token.

The top cold experts have low heatmap scores, so the pool is loaded correctly
from a ranking that is not predictive for the benchmark decode distribution.
Examples:

- `L13E98`: cold `246`, heatmap `29`
- `L9E198`: cold `245`, heatmap `45`
- `L27E74`: cold `243`, heatmap `11`
- `L15E185`: cold `243`, heatmap `6`

Graph capture also recurs per decode request:

- `14` decode requests
- `14` per-layer graph captures
- `13` pointer invalidations

Capture succeeds, but HCS pointer changes cause graph invalidation and
recapture on the next decode request.

## Conclusion

Prefill cause:

- HQQ6 prefill projection/dequantization is adding about `0.93s` on a 10K
  one-chunk prefill, which matches the observed gap to BF16/fp8.
- Long prompts are additionally slowed by smaller chunk sizes on the HQQ6/k4v4
  memory surface.

Decode cause:

- HCS capacity is sufficient, but HCS ranking quality is poor for the actual
  benchmark decode prompts under HQQ6/k4v4.
- Per-request graph pointer invalidation adds graph recapture overhead after
  HCS reloads.

Next fix targets:

1. Prefill: avoid per-prefill HQQ projection/dequant overhead by adding native
   compact HQQ projection kernels or keeping a stage-owned prefill-ready HQQ
   format without duplicating steady decode residency.
2. Decode: improve HCS ranking for the actual decode distribution and make
   graph pointer stability survive HCS reloads when slot semantics are stable.

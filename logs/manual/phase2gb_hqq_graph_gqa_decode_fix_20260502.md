# Phase 2GB - HQQ Graph GQA Decode Fix

Date: 2026-05-02

## Problem

Current HQQ decode collapsed after the first generated token on multiple
surfaces:

- QCN HQQ8/BF16-KV: exact `28/653`, containment `28/653`
- QCN HQQ8+k6v6: exact `28/653`, containment `28/653`
- Q122B HQQ6+k4v4: exact `28/361`, containment `28/361`

Compaction was ruled out by running QCN HQQ8/BF16-KV with
`KRASIS_HQQ_STAGE_COMPACT=0`; it still failed at exact/containment `27/653`.

## Root Cause

The recent exclusive-HQQ residency path stopped registering normal BF16
attention weights and released BF16 projection tensors, which is correct for
VRAM. However, graph-captured GQA decode still used normal `GpuAttnConfig`
projection IDs. In exclusive HQQ mode those fallback IDs can be zero.

Ungraphed token 1 used HQQ descriptors and matched witness. Graph token 2 could
fall through to the normal GQA path and read wrong/fallback metadata, matching
the observed `run=2` collapse.

## Fix

Changed `src/gpu_decode.rs` only:

- `allocate_batch_buffers()` now sizes HQQ LA/GQA projection scratch from HQQ
  descriptors instead of normal projection IDs.
- Graph-captured GQA decode now dispatches HQQ fused/split Q/K/V descriptors
  and HQQ O projection when `layer.hqq_exec` is present.
- No BF16 attention projection tensors are restored.
- No BF16 fallback was added.
- HQQ remains the only resident attention weight path.

## Validation

Build:

- `./dev build` passed.

QCN HQQ8/BF16-KV:

- Command:
  `./dev witness-compare tests/qcn-bf16kv-hqq8-accuracy.conf --profile phase2bn_qcn_64tok --startup-timeout 1200`
- Run dir: `logs/reference-test_20260502_230815`
- Result: exact `494/653`, containment `531/653`, full exact `7/14`,
  first token `14/14`, avg run `35.3`

QCN HQQ8+k6v6:

- Command:
  `./dev witness-compare tests/qcn-k6v6-hqq8-accuracy.conf --profile phase2bn_qcn_64tok --startup-timeout 1200`
- Run dir: `logs/reference-test_20260502_231544`
- Result: exact `463/653`, containment `510/653`, full exact `7/14`,
  first token `14/14`, avg run `33.1`

Q122B HQQ6+k4v4:

- Command:
  `./dev witness-compare tests/q122b-k4v4-hqq6-int4-benchmark.conf --profile llama_witness_q122b_seq32 --startup-timeout 1800`
- Run dir: `logs/reference-test_20260502_232250`
- Result: exact `258/361`, containment `284/361`, full exact `7/14`,
  first token `14/14`, avg run `18.4`

## Comparison

| Surface | Before fix | After fix |
| --- | ---: | ---: |
| QCN HQQ8/BF16-KV | `28/653` exact, `28/653` containment | `494/653`, `531/653` |
| QCN HQQ8+k6v6 | `28/653` exact, `28/653` containment | `463/653`, `510/653` |
| Q122B HQQ6+k4v4 | `28/361` exact, `28/361` containment | `258/361`, `284/361` |

## Notes

The catastrophic HQQ decode regression is fixed without increasing VRAM
residency. QCN HQQ8+k6v6 is still below the old strongest historical row
(`562/653`, `572/653`), so there may be a residual quality delta, but the
token-2 graph-path collapse is gone.

## Speed Check

Timing-free 122B benchmark after the fix:

- Command:
  `./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`
- Run dir: `logs/dev-benchmark_20260502_233155`
- Archived log:
  `benchmarks/20260502_phase2gb_q122b_k4v4_hqq6_graph_gqa_fix_benchmark.log`
- Prefill: `2071.5 tok/s`
- Internal decode: `22.54 tok/s`
- Round trip: `41.47 tok/s`
- HCS: `3483/12288 (28.3%)`
- Min free VRAM: `640 MB`

Compared with Phase 2FU on the same config, decode recovered from `8.85` to
`22.54 tok/s` while prefill stayed in the same range (`2030.4 -> 2071.5`).

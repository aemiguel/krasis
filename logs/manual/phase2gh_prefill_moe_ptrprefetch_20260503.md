# Phase 2GH - Prefill MoE Pointer-Table Prefetch

Date: 2026-05-03

## Goal

Reduce Q122B prefill MoE DMA wait after Phase 2GF showed materialized HQQ
prefill was dominated by MoE expert movement/wait rather than HQQ projection
dequant.

## Baseline

Phase 2GF 50K timing on `HQQ6+k4v4 + materialized HQQ prefill`:

- Total prefill: `18723.0 ms`
- MoE: `12661.9 ms`
- MoE DMA: `9774.2 ms`
- MoE DMA wait: `9244.9 ms`
- cold H2D enqueue/accounting: `512.9 ms`

The code path was serial for pointer-table prefill: build/stage current-layer
cold experts, record the copy-stream event, then wait immediately before MoE.
The old double-buffer overlap path was skipped for pointer-table mode.

## Implementation

- Added dense pointer-table prefetch for MoE prefill.
- Before layer attention, dense chunks build an all-expert pointer table and
  queue cold H2D copies on the copy stream.
- MoE consumes that prefetched table after attention, waiting on the existing
  DMA event only if needed.
- Sparse chunks stay on the exact active-expert current-layer pointer-table
  path.
- The dense predicate is runtime-derived:
  - use prescan active experts only when that full runtime chunk was prescanned
  - otherwise fall back to `(m * topk).min(n_experts)`
  - prefetch when predicted active density is at least 75%
- No extra cold staging allocation was added.
- No persistent expert or BF16 residency was added.
- Raw pointer-table buffers are freed after the consuming layer.

## Correctness

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
  ./dev witness-compare tests/q122b-k4v4-hqq6-int4-benchmark.conf \
    --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

Result:

- Run dir: `logs/reference-test_20260503_105211`
- Overall: PASS
- Prompts: `14 PASS`, `0 WARN`, `0 FAIL`
- First token: `14/14`
- Prefill top-10: `14/14`
- Generated exact: `270/361`
- Generated containment: `292/361`
- Full exact rows: `8/14`

## Timing Diagnostic

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PREFILL_DEBUG=1 \
  ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Result:

- Run dir: `logs/dev-benchmark_20260503_105923`
- 39,920-token calibration chunks `0/1/2/3` all used `[PTR-PREFETCH]`.
- 35K chunks `0/1/2/3` all used `[PTR-PREFETCH]`.
- 50K chunks `0/1/2/3/4` all used `[PTR-PREFETCH]`.
- Sparse short prompts continued to use exact `[PTR-TABLE]`.

Diagnostic speed:

- 20K: `4557.2 tok/s`
- 35K: `2888.0 tok/s`
- 50K: `3182.8 tok/s`

## Speed

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
  ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Result:

- Run dir: `logs/dev-benchmark_20260503_110832`
- Archived log: `benchmarks/20260503_phase2gh_q122b_k4v4_hqq6_ptrprefetch_benchmark.log`
- Prefill: `4689.8 tok/s`
- Decode internal: `24.80 tok/s`
- Round trip: `42.45 tok/s`
- HCS: `3780/12288 (30.8%)`
- Min decode free VRAM: `662 MB`

## Comparison

| Metric | Phase 2GE materialized HQQ prefill | Phase 2GH pointer-table prefetch |
| --- | ---: | ---: |
| Prefill | `3003.9 tok/s` | `4689.8 tok/s` |
| Decode internal | `24.28 tok/s` | `24.80 tok/s` |
| Round trip | `42.29 tok/s` | `42.45 tok/s` |
| HCS | `3780/12288 (30.8%)` | `3780/12288 (30.8%)` |
| Min free VRAM | `662 MB` | `662 MB` |

## Conclusion

Dense pointer-table prefetch is a real prefill win on Q122B HQQ6+k4v4. It
substantially reduces the practical cost of MoE cold staging in dense long
prompt chunks without increasing persistent VRAM residency or changing decode
behavior.

The remaining prefill work is likely attention/FLA/FA2 and the residual MoE
work that is not hidden by current-layer prefetch.

# Phase 2GE - Materialized HQQ Prefill 122B

Date: 2026-05-03

## Goal

Reduce Q122B HQQ6 prefill projection cost without adding persistent BF16
attention residency or changing compact HQQ decode.

## Implementation

- Added opt-in `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1`.
- Under the flag, HQQ prefill descriptors use row-major compact HQQ instead of
  the Marlin float-zp prefill format.
- Prefill dequantizes one HQQ projection at a time into a reusable BF16 scratch
  weight, applies existing HQQ sidecars, then runs existing cuBLAS BF16 GEMM.
- The scratch is included in `compute_scratch_vram`, so calibration/chunking
  accounts for the transient memory.
- Decode remains compact HQQ/VMM.

## Correctness

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev witness-compare tests/q122b-k4v4-hqq6-int4-benchmark.conf --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

Result:

- Run dir: `logs/reference-test_20260503_083900`
- Overall: PASS
- Prompts: `14 PASS`, `0 WARN`, `0 FAIL`
- First token: `14/14`
- Prefill argmax/top-10: `14/14`
- Generated exact: `280/361`
- Generated containment: `303/361`
- Full exact rows: `7/14`

## Component Timing

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
KRASIS_STARTUP_DIAG=1 KRASIS_STARTUP_EXIT_AFTER_CALIBRATION=1 \
KRASIS_STARTUP_CAL_LONG_TOKENS=10000 KRASIS_PREFILL_TIMING=1 \
KRASIS_PREFILL_DEBUG=1 \
  ./dev run tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Result:

- Log: `logs/manual/phase2ge_q122b_materialized_prefill_component_timing_20260503.log`
- 10K prefill block: `3600.9 ms`
- Old Phase 2FS 10K block: `4143.2 ms`
- GQA projection: `61.6 ms`
- LA projection: `236.6 ms`
- LA O projection: `84.7 ms`
- The old `marlin_float_zp` counter does not appear on the materialized path.

## Speed

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Result:

- Run dir: `logs/dev-benchmark_20260503_084927`
- Archived log: `benchmarks/20260503_phase2ge_q122b_k4v4_hqq6_materialized_prefill_benchmark.log`
- Prefill: `3003.9 tok/s`
- Decode internal: `24.28 tok/s`
- Round trip: `42.29 tok/s`
- HCS: `3780/12288 (30.8%)`
- Min decode free VRAM: `662 MB`

## Comparison

| Metric | Phase 2GC stable HQQ graphs | Phase 2GE materialized prefill |
| --- | ---: | ---: |
| Prefill | `2029.1 tok/s` | `3003.9 tok/s` |
| Decode internal | `24.49 tok/s` | `24.28 tok/s` |
| Round trip | `44.35 tok/s` | `42.29 tok/s` |
| HCS | `3780/12288 (30.8%)` | `3780/12288 (30.8%)` |
| Min free VRAM | `664 MB` | `662 MB` |

## Conclusion

The materialized HQQ prefill path fixes the measured prefill projection
bottleneck for Q122B HQQ6+k4v4. It restores prefill to the prior release-class
range while preserving decode behavior and compact HQQ residency.

# Phase 2BK - INT8 Exception Top-K Validation

## Decision

`top4_is_the_only_quality_positive_setting_across_qcn_and_q35_top8_regresses_top16_is_mixed_and_slower`

This was a bounded validation pass only. No runtime code, scoring logic, default
config, fallback, sidecar promotion, or quantized-prefill implementation was
changed.

## Witness Results

Lower selected-logprob sum is better. Witness is external validation only.

| Model | Variant | First-token | Selected sum | Delta vs base | Top-overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| QCN | HQQ-only | 8/8 | 0.1210486500 | 0.0000000000 | 72/80 |
| QCN | INT8 top-4 | 8/8 | 0.0936387312 | -0.0274099188 | 73/80 |
| QCN | INT8 top-8 | 8/8 | 0.1265897312 | +0.0055410812 | 74/80 |
| QCN | INT8 top-16 | 8/8 | 0.1127257312 | -0.0083229188 | 74/80 |
| Q35 | HQQ-only | 10/10 | 0.8856095677 | 0.0000000000 | 86/100 |
| Q35 | INT8 top-4 | 10/10 | 0.7995640518 | -0.0860455159 | 84/100 |
| Q35 | INT8 top-8 | 10/10 | 0.9006975677 | +0.0150880000 | 85/100 |
| Q35 | INT8 top-16 | 10/10 | 0.8659915677 | -0.0196180000 | 85/100 |

## Standard Benchmark Results

Timing instrumentation was off. Decode values are internal engine numbers, not
HTTP round-trip throughput.

| Model | Variant | Prefill tok/s | Decode tok/s | HCS | Min free VRAM |
| --- | ---: | ---: | ---: | --- | ---: |
| QCN | HQQ-only | 7,242.6 | 70.79 | 12960/24576 (52.7%) | 686 MB |
| QCN | INT8 top-4 | 6,936.2 | 71.14 | 12960/24576 (52.7%) | 692 MB |
| QCN | INT8 top-8 | 7,854.1 | 70.92 | 12960/24576 (52.7%) | 690 MB |
| QCN | INT8 top-16 | 7,833.9 | 68.59 | 12879/24576 (52.4%) | 758 MB |
| Q35 | HQQ-only | 7,146.8 | 114.01 | 10240/10240 (100.0%) | 5452 MB |
| Q35 | INT8 top-4 | 7,186.4 | 116.59 | 10240/10240 (100.0%) | 5428 MB |
| Q35 | INT8 top-8 | 7,138.4 | 114.16 | 10240/10240 (100.0%) | 5426 MB |
| Q35 | INT8 top-16 | 7,428.9 | 104.14 | 10240/10240 (100.0%) | 5396 MB |

## Interpretation

- Top-4 is the only variant that improves selected-logprob on both models.
- Top-8 should be rejected for now: it regresses selected-logprob on both QCN
  and Q35 despite passing first-token checks.
- Top-16 improves selected-logprob versus base on both models, but less than
  top-4 and with a meaningful decode-speed cost, especially Q35
  `114.01 -> 104.14 tok/s`.
- The quality effect is not monotonic with more exception groups. Selection
  policy matters more than simply increasing exception count.
- These results are enough to justify continuing INT8 exception implementation
  work, but not enough to default-enable or promote any manifest.

## Artifacts

Witness summaries:

- QCN base: `logs/reference-test_20260426_214138/reference_test_summary.json`
- QCN top-4: `logs/reference-test_20260426_214354/reference_test_summary.json`
- QCN top-8: `logs/reference-test_20260426_214616/reference_test_summary.json`
- QCN top-16: `logs/reference-test_20260426_214831/reference_test_summary.json`
- Q35 base: `logs/reference-test_20260426_215047/reference_test_summary.json`
- Q35 top-4: `logs/reference-test_20260426_215313/reference_test_summary.json`
- Q35 top-8: `logs/reference-test_20260426_215542/reference_test_summary.json`
- Q35 top-16: `logs/reference-test_20260426_215810/reference_test_summary.json`

Benchmark logs are archived in `benchmarks/` and summarized in
`benchmarks/BENCHMARKS.md`.

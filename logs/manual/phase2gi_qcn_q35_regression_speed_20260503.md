# Phase 2GI - QCN and 35B Accuracy/Speed Regression

Date: 2026-05-03

## Question

After Phase 2GH pointer-table prefill DMA overlap, verify the changes on QCN
and Qwen3.5-35B, then run timing-free speed benchmarks.

## Accuracy

| Model | Surface | Profile | Result | First | Prefill | Exact | Containment | Full |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| QCN | HQQ8 + k6v6 | `llama_witness_stage3_qcn_expanded` | PASS | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| Qwen3.5-35B | HQQ6 + fp8 KV | `llama_witness_qwen35_expanded_thinking_off` | PASS | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

Accuracy logs:
- QCN: `logs/reference-test_20260503_112838`,
  `logs/manual/phase2gi_qcn_hqq8_k6v6_accuracy_20260503.log`
- 35B: `logs/reference-test_20260503_114100`,
  `logs/manual/phase2gi_q35b_hqq6_accuracy_20260503.log`

## Speed

Timing/debug instrumentation was disabled.

| Model | Command | Prefill | Decode internal | Round trip | HCS | Min free VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| QCN | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev speed-test` | 8352.2 tok/s | 81.02 tok/s | 165.05 tok/s | 15147/24576 (61.6%) | 706 MB |
| Qwen3.5-35B | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q35b-4-4-hqq6-benchmark.conf` | 11074.0 tok/s | 113.32 tok/s | 230.62 tok/s | 10240/10240 (100.0%) | 8990 MB |

Archived benchmark logs:
- `benchmarks/20260503_phase2gi_qcn_hqq8_k4v4_speedtest.log`
- `benchmarks/20260503_phase2gi_q35b_hqq6_fp8_benchmark.log`

## Notes

- QCN accuracy used the compact `k6v6` HQQ8 witness surface; QCN speed used the
  required fixed `./dev speed-test` surface, which reports HQQ8/k4v4.
- 35B fits every expert into soft HCS residency on the selected 5090, which
  explains its much higher decode than Q122B's hybrid cold/HCS decode.
- No correctness regression was observed on either model.

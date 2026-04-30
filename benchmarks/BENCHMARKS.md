# Krasis Benchmark Results

## Standard Benchmarks — 2026-04-29 (Phase 2DK QCN INT4/HQQ8 KV comparison)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN HQQ8 attention with INT4 GPU/CPU experts, INT8 shared/dense/lm
head, graph replay enabled, timing instrumentation off. Only the KV format
differs. Both runs used the default HQQ8 prefill path with
`KRASIS_HQQ8_PREFILL_MODE` unset.

| Variant | KV bpe | Context @ 1GB KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-------:|-----------------:|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ8/k4v4, INT4 experts | 5.0 | 136,528 | 5,941.9 | 78.14 | 14256/24576 (58.0%) | 732 MB | [log](20260429_154355_qcn_k4v4_hqq8_int4_benchmark.log) |
| QCN HQQ8/k6v6, INT4 experts | 7.0 | 97,520 | 6,014.9 | 78.05 | 14256/24576 (58.0%) | 740 MB | [log](20260429_154908_qcn_k6v6_hqq8_int4_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- `k4v4` and `k6v6` are effectively tied on internal decode in this run
  (`78.14` vs `78.05 tok/s`); `k6v6` is slightly faster on best prefill
  (`6,014.9` vs `5,941.9 tok/s`).
- With a fixed `1000 MB` KV cache, `k4v4` provides a larger context window
  (`136,528` tokens) than `k6v6` (`97,520` tokens), matching the expected
  `5.0` versus `7.0` bpe footprint.
- Both runs emitted a prefill-time VRAM monitor warning below the configured
  `600 MB` safety margin (`514 MB` for `k4v4`, `522 MB` for `k6v6`). The table's
  min-free VRAM value is the benchmark summary's decode min-free value.
- A benchmark-report metadata bug was fixed during this pass: newer KV formats
  were previously displayed as `FP8 E4M3` in `benchmark_report.log` even when
  runtime logs correctly showed `Shared k4v4/k6v6 KV cache`.

---

## Standard Benchmarks — 2026-04-28 (Phase 2CT QCN k8v4 HQQ8 faster prefill mode)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: QCN HQQ8 attention with `k8v4` KV cache, graph replay enabled, INT4
GPU/CPU experts, timing instrumentation off. The run used
`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale`, which keeps the
two-scale HQQ8 Marlin prefill slope correction but removes the intercept
correction pass.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ8/k8v4, INT4 experts, two-scale no-intercept | 5,922.8 | 78.24 | 14256/24576 (58.0%) | 740 MB | [log](20260428_204438_qcn_k8v4_hqq8_twoscale_int4_benchmark.log) |

Notes:
- Accuracy gate before the benchmark: `PASS`, avg exact `34.07`, total exact
  `477/653`, containment `556/653` on `phase2bn_qcn_64tok`.
- Follow-up: this mode is now the default HQQ8 prefill path when
  `KRASIS_HQQ8_PREFILL_MODE` is unset.
- Compared with the previous k8v4 HQQ8 INT4 benchmark using
  `native-fused-marlin-twoscale-intercept`, prefill improved
  `5,245.5 -> 5,922.8 tok/s` and decode moved `76.74 -> 78.24 tok/s`.
- This mode still emitted a prefill-time VRAM monitor warning at `522 MB` free,
  below the configured `600 MB` safety margin; decode min-free was `740 MB`.
- Decode remains well below the old AWQ/Polar4 speed-test baseline
  (`91.77 tok/s`), but timing attribution shows the remaining decode gap is
  dominated by graph replay sync wait and cold expert DMA rather than HQQ
  attention math.
- Reduction: `logs/manual/phase2ct_qcn_hqq8_speed_followup_20260428.md`.

---

## Standard Benchmarks — 2026-04-28 (Phase 2CS QCN k8v4 HQQ8 expert-bit comparison)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN HQQ8 attention with `k8v4` KV cache, graph replay enabled, timing
instrumentation off. Both runs used
`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept`; only the
GPU/CPU expert bits differ.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ8/k8v4, INT8 experts | 4,238.1 | 35.00 | 7175/24576 (29.2%) | 752 MB | [log](20260428_194459_qcn_k8v4_hqq8_int8_benchmark.log) |
| QCN HQQ8/k8v4, INT4 experts | 5,245.5 | 76.74 | 14256/24576 (58.0%) | 708 MB | [log](20260428_195237_qcn_k8v4_hqq8_int4_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- INT4 experts are substantially faster on this surface because HCS soft
  coverage roughly doubles (`29.2% -> 58.0%`) and expert cache bandwidth is
  lower.
- The INT4 run emitted a prefill-time VRAM monitor warning at `564 MB` free,
  below the configured `600 MB` safety margin. The benchmark summary's min free
  VRAM row is the decode min-free value.
- Reduction: `logs/manual/phase2cs_qcn_k8v4_hqq8_benchmark_reduction_20260428.md`.

---

## Standard Benchmarks — 2026-04-27 (Phase 2BR QCN Polar4 HQQ speed-test variants)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: speed-test-equivalent QCN runs with the AWQ attention choice replaced
by HQQ4SC or HQQ8. The rest of the surface matches the QCN AWQ/Polar4 speed
test: INT4 GPU/CPU experts, Polar4 KV, INT8 shared/dense/lm-head, layer group
size 2, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN AWQ/Polar4 speed-test baseline, 2026-04-26 | 7,295.6 | 91.77 | 16848/24576 (68.6%) | 688 MB | [log](20260426_211755_qcn_polar4_awq_speed_regression_check.log) |
| QCN HQQ4SC/Polar4 speed-test variant | 466.2 | 79.17 | 14823/24576 (60.3%) | 706 MB | [log](20260427_201735_qcn_polar4_hqq4sc_speed_variant.log) |
| QCN HQQ8/Polar4 speed-test variant, scalar HQQ prefill | 486.8 | 75.16 | 14256/24576 (58.0%) | 734 MB | [log](20260427_202957_qcn_polar4_hqq8_speed_variant.log) |
| QCN HQQ8/Polar4 speed-test variant, Marlin prefill prototype | 5,011.6 | 79.80 | 14256/24576 (58.0%) | 734 MB | [log](20260427_215803_qcn_polar4_hqq8_fastprefill_speed.log) |
| QCN HQQ8/Polar4 speed-test variant, native fused Marlin experiment | 7,132.9 | 79.68 | 14256/24576 (58.0%) | 734 MB | [log](20260427_230455_qcn_polar4_hqq8_native_fused_speed.log) |
| QCN HQQ8/Polar4 speed-test variant, two-scale + intercept Marlin experiment | 5,340.1 | 83.45 | 14256/24576 (58.0%) | 702 MB | [log](20260428_063003_qcn_polar4_hqq8_twoscale_intercept_speed.log) |

Notes:
- Runs executed via `./dev benchmark tests/qcn-polar4-hqq4sc.conf` and
  `./dev benchmark tests/qcn-polar4-hqq8.conf` because `./dev speed-test` is
  still the fixed AWQ config.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- HQQ4SC/HQQ8 decode is usable but below AWQ on this exact Polar4 speed-test
  surface: HQQ4SC is about `86%` of AWQ decode, HQQ8 about `82%`.
- The first HQQ8/Polar4 row used the old scalar HQQ prefill kernel and was not
  acceptable: about `6-7%` of AWQ prefill.
- The Marlin prefill prototype replaces the scalar HQQ8 prefill GEMM with a
  two-pass Marlin U8B128 path plus grouped zero correction. It improves HQQ8
  prefill from `486.8` to `5,011.6 tok/s` (`10.3x`) on this surface, reaching
  about `68.7%` of the AWQ/Polar4 baseline prefill.
- The native fused Marlin experiment uses Marlin U8 with BF16 float zero-points
  as a single GEMM. It improves HQQ8 prefill from `486.8` to `7,132.9 tok/s`
  (`14.7x`) and reaches `97.8%` of the AWQ/Polar4 baseline prefill. Accuracy
  did not fully hold versus the residual Marlin HQQ8 prototype (`14.29` average
  exact prefix versus `15.79`), so this is a speed/architecture result rather
  than a production default.
- The two-scale + intercept Marlin experiment uses one U8 float-zp Marlin GEMM
  with a second BF16 scale plane plus a compact FP32 intercept correction. It
  improves the QCN HQQ8 64-token witness gate over both residual Marlin and
  native fused v1 (`18.14` average exact prefix, `326/653` decode containment),
  while prefill lands between those two speed points at `5,340.1 tok/s`.
- The Marlin prefill prototype passed the QCN 64-token witness gate before this
  speed run (`14/14` first-token, `15.79` average exact prefix), but selected
  first-token logprob delta remains worse than the previous fixed HQQ8 path; it
  should stay treated as a prototype until that residual quality gap is closed.
- VRAM monitor lows during timed prefill: HQQ4SC reached `532 MB` free; HQQ8
  scalar and Marlin-prefill runs reached `516 MB` free.
- Reduction: `logs/manual/phase2br_qcn_polar4_hqq_speed_reduction_20260427.md`.

---

## Standard Benchmarks — 2026-04-26 (Phase 2BK INT8 exception top-k validation)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN and Qwen3.5 HQQ4 baselines versus explicit top-4/top-8/top-16 INT8
exception manifests for layer-0 `in_proj_qkvz`. Timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ4 baseline | 7,242.6 | 70.79 | 12960/24576 (52.7%) | 686 MB | [log](20260426_220134_qcn_hqq4_phase2bk_base.log) |
| QCN HQQ4 + INT8 exceptions top-4 | 6,936.2 | 71.14 | 12960/24576 (52.7%) | 692 MB | [log](20260426_220448_qcn_hqq4_phase2bk_int8_top4.log) |
| QCN HQQ4 + INT8 exceptions top-8 | 7,854.1 | 70.92 | 12960/24576 (52.7%) | 690 MB | [log](20260426_220800_qcn_hqq4_phase2bk_int8_top8.log) |
| QCN HQQ4 + INT8 exceptions top-16 | 7,833.9 | 68.59 | 12879/24576 (52.4%) | 758 MB | [log](20260426_221107_qcn_hqq4_phase2bk_int8_top16.log) |
| Q35 HQQ4 baseline | 7,146.8 | 114.01 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_221419_q35_hqq4_phase2bk_base.log) |
| Q35 HQQ4 + INT8 exceptions top-4 | 7,186.4 | 116.59 | 10240/10240 (100.0%) | 5428 MB | [log](20260426_221755_q35_hqq4_phase2bk_int8_top4.log) |
| Q35 HQQ4 + INT8 exceptions top-8 | 7,138.4 | 114.16 | 10240/10240 (100.0%) | 5426 MB | [log](20260426_222218_q35_hqq4_phase2bk_int8_top8.log) |
| Q35 HQQ4 + INT8 exceptions top-16 | 7,428.9 | 104.14 | 10240/10240 (100.0%) | 5396 MB | [log](20260426_222643_q35_hqq4_phase2bk_int8_top16.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- Top-4 was the only variant that improved witness selected-logprob on both QCN
  and Q35 in the matching Phase 2BK witness set.
- Top-8 regressed witness selected-logprob on both models. Top-16 improved
  selected-logprob less than top-4 and had a meaningful decode-speed cost,
  especially on Q35.
- The associated quality reduction is
  `logs/manual/phase2bk_int8_exception_topk_validation_reduction_20260426.md`.

---

## Standard Benchmarks — 2026-04-26 (QCN AWQ/Polar4 speed regression check)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: standard `./dev speed-test` QCN AWQ/Polar4 path after Phase 2BJ decode INT8 exception work. Timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN AWQ/Polar4 speed-test check | 7,295.6 | 91.77 | 16848/24576 (68.6%) | 688 MB | [log](20260426_211755_qcn_polar4_awq_speed_regression_check.log) |

Notes:
- Run executed via `./dev speed-test`.
- This checks the historical `90+ tok/s` QCN path directly after Phase 2BJ.
- Result: standard QCN AWQ/Polar4 decode remains in the expected `90+ tok/s` class; the `71.30 tok/s` number belongs to the separate QCN HQQ4 + INT8 exception top-4 config.
- Decode values are the benchmark's internal engine numbers. Network round-trip numbers are present in the full log but are not used as decode speed.

---

## Standard Benchmarks — 2026-04-26 (Phase 2BJ INT8 exception prefill+decode top-4)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN and Qwen3.5 HQQ4 with explicit top-4 INT8 exception manifests after decode-side exception execution was implemented. Timing instrumentation off. Baselines are the same-day Phase 2BH HQQ4 baseline runs below.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ4 baseline | 7,912.5 | 73.97 | 12960/24576 (52.7%) | 686 MB | [log](20260426_193218_qcn_hqq4_int8_exception_phase2bh_baseline.log) |
| QCN HQQ4 + INT8 exceptions top-4 | 6,984.6 | 71.30 | 12960/24576 (52.7%) | 692 MB | [log](20260426_210438_qcn_hqq4_int8_exception_phase2bj_top4_prefill_decode.log) |
| Q35 HQQ4 baseline | 9,360.5 | 116.24 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_193851_q35_hqq4_int8_exception_phase2bh_baseline.log) |
| Q35 HQQ4 + INT8 exceptions top-4 | 9,523.2 | 111.88 | 10240/10240 (100.0%) | 5428 MB | [log](20260426_210751_q35_hqq4_int8_exception_phase2bj_top4_prefill_decode.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- QCN top-4 prefill+decode path had lower internal prefill throughput than the HQQ-only baseline (`-11.7%`) and lower decode throughput (`-3.6%`).
- Q35 top-4 prefill+decode path had slightly higher internal prefill throughput (`+1.7%`) but lower decode throughput (`-3.8%`).
- Decode values are the benchmark's internal engine numbers. Network round-trip numbers are present in the full logs but are not used as decode speed.
- The associated implementation and quality reduction is `logs/manual/phase2bj_int8_exception_decode_runtime_reduction_20260426.md`.

---

## Standard Benchmarks — 2026-04-26 (Phase 2BH INT8 exception prefill prototype)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN and Qwen3.5 HQQ4 baselines versus explicit single-block INT8 exception prefill manifests. Timing instrumentation off. These runs were used only to measure the opt-in Phase 2BH prefill path; no default/runtime promotion was made.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ4 baseline | 7,912.5 | 73.97 | 12960/24576 (52.7%) | 686 MB | [log](20260426_193218_qcn_hqq4_int8_exception_phase2bh_baseline.log) |
| QCN HQQ4 + INT8 exception group 12 | 7,832.5 | 71.95 | 12960/24576 (52.7%) | 682 MB | [log](20260426_193537_qcn_hqq4_int8_exception_phase2bh_g12.log) |
| Q35 HQQ4 baseline | 9,360.5 | 116.24 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_193851_q35_hqq4_int8_exception_phase2bh_baseline.log) |
| Q35 HQQ4 + INT8 exception group 14 | 9,332.0 | 116.06 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_194201_q35_hqq4_int8_exception_phase2bh_g14.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- QCN INT8 exception prefill overhead was about `-1.0%` internal prefill throughput and `-4 MB` measured min-free VRAM difference.
- Q35 INT8 exception prefill overhead was about `-0.3%` internal prefill throughput with unchanged measured min-free VRAM.
- Decode values are the benchmark's internal engine numbers. Network round-trip numbers are present in the full logs but are not used as decode speed.
- The associated quality reduction is `logs/manual/phase2bh_int8_exception_runtime_reduction_20260426.md`.

---

## Standard Benchmarks — 2026-04-16 (QCN Polar4 AWQ after QK FP32 decision rerun)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| post QK FP32 decision rerun | 7,308.0 | 95.06 | 16848/24576 (68.6%) | 682 MB | [log](20260416_065456_qcn_polar4_awq_5090_qk_fp32_policy_rerun.log) |

Notes:
- Run executed via `./dev speed-test` on branch `gpu-debug-trace` at `8e50e32`.
- Internal prefill runs:
  - `1K`: `524.9 tok/s`
  - `5K`: `2372.6 tok/s`
  - `10K`: `4166.5 tok/s`
  - `20K`: `5913.7 tok/s`
  - `35K`: `7308.0 tok/s`
  - `50K`: `6655.0 tok/s`
- Internal decode runs:
  - `50`: `91.84 tok/s`
  - `100`: `95.06 tok/s`
  - `250`: `81.76 tok/s`
- Round-trip HTTP runs:
  - `50`: `185.41 tok/s`
  - `100`: `111.49 tok/s`
  - `250`: `88.65 tok/s`
- Calibration summary:
  - short decode probe: `60.7 tok/s`
  - long decode probe: `49.2 tok/s`
  - transient deltas: short prefill `23678 MB`, long prefill `26206 MB`, short decode `50 MB`, long decode `2 MB`
  - worst-case prefill scratch reservation: `26743 MB` at `50000` tokens
- HCS load summary:
  - `16848/24576` experts loaded (`68.6%`)
  - soft HCS footprint `26199.4 MB`
- Standard benchmark log archived at `benchmarks/20260416_065456_qcn_polar4_awq_5090_qk_fp32_policy_rerun.log`.

---

## Standard Benchmarks — 2026-04-15 (QCN Polar4 AWQ linear-attention AWQ fold review)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| LA AWQ runtime disabled | 7,233.2 | 94.01 | 16848/24576 (68.6%) | 682 MB | [log](20260415_224343_qcn_polar4_awq_5090_la_awq_runtime_disabled.log) |
| LA AWQ fold restored | 7,241.5 | 94.65 | 16848/24576 (68.6%) | 682 MB | [log](20260415_224836_qcn_polar4_awq_5090_la_awq_fold_restored.log) |

Notes:
- Both runs used `./dev speed-test` on branch `gpu-debug-trace`.
- This pair was run specifically to evaluate whether linear-attention input projections should participate in the same AWQ input-scale-and-fold contract as calibration.
- Result:
  - prefill improved slightly: `7233.2 -> 7241.5 tok/s`
  - internal decode improved slightly: `94.01 -> 94.65 tok/s`
  - HCS coverage and minimum free VRAM were unchanged
- Internal prefill runs with fold restored:
  - `1K`: `520.9 tok/s`
  - `5K`: `2395.2 tok/s`
  - `10K`: `4070.0 tok/s`
  - `20K`: `5832.3 tok/s`
  - `35K`: `7241.5 tok/s`
  - `50K`: `6379.7 tok/s`
- Internal decode runs with fold restored:
  - `50`: `90.66 tok/s`
  - `100`: `94.65 tok/s`
  - `250`: `86.17 tok/s`
- Round-trip HTTP best with fold restored:
  - `167.68 tok/s` at `50` tokens

---

## Standard Benchmarks — 2026-04-13 (QCN Polar4 AWQ after BF16 policy / dead TRTLLM cleanup)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| post BF16 policy cleanup; decode benchmark EOS-early failure | 7,777.2 | FAILED (EOS at 2 tokens) | 16848/24576 (68.6%) | 696 MB | [log](20260413_171619_qcn_polar4_awq_5090_eos_early_decode_failure.log) |

Notes:
- Run executed via `./dev speed-test` on branch `gpu-debug-trace` after pushing `5e80acb`.
- Internal prefill runs:
  - `1K`: `549.1 tok/s`
  - `5K`: `2680.3 tok/s`
  - `10K`: `4194.3 tok/s`
  - `20K`: `6500.0 tok/s`
  - `35K`: `7777.2 tok/s`
  - `50K`: `7694.4 tok/s`
- Internal decode benchmark did not produce a valid throughput number:
  - `50`: failed, EOS at `2` tokens
  - `100`: failed, EOS at `2` tokens
  - `250`: failed, EOS at `2` tokens
- Round-trip HTTP benchmark also failed the same way:
  - `50`: failed, EOS at `2` tokens
  - `100`: failed, EOS at `2` tokens
  - `250`: failed, EOS at `2` tokens
- Standard benchmark log archived at `benchmarks/20260413_171619_qcn_polar4_awq_5090_eos_early_decode_failure.log`.

---

## Standard Benchmarks — 2026-04-13 (QCN Polar4 AWQ after capture-box and Nemotron reference fixes)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| post capture-box hardening and Nemotron compat fixes | 7,554.2 | 92.59 | 16848/24576 (68.6%) | 732 MB | [log](20260413_005327_qcn_polar4_awq_5090.log) |

Notes:
- Run executed via `./dev speed-test` on branch `gpu-debug-trace` after pushing `9cc7a91`.
- Internal prefill runs:
  - `1K`: `510.2 tok/s`
  - `5K`: `2446.1 tok/s`
  - `10K`: `4012.7 tok/s`
  - `20K`: `5810.8 tok/s`
  - `35K`: `7554.2 tok/s`
  - `50K`: `6579.1 tok/s`
- Internal decode runs:
  - `50`: `92.59 tok/s`
  - `100`: `87.11 tok/s`
  - `250`: `91.96 tok/s`
- Round-trip HTTP best:
  - `129.55 tok/s` at `50` tokens
- Standard benchmark log archived at `benchmarks/20260413_005327_qcn_polar4_awq_5090.log`.

---

## Standard Benchmarks — 2026-04-04 (QCN Polar4 AWQ after HCS async pointer lifetime fix)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| accuracy async ptr lifetime fix | 7,891.4 | 98.59 | 17010/24576 (69.2%) | 682 MB | [log](20260404_011629_qcn_polar4_awq_5090_accuracy_hcs_async_ptr_fix.log) |

Notes:
- Run executed via `./dev speed-test` on branch `accuracy`.
- Fix restored stable host backing for async `cuMemcpyHtoDAsync_v2` expert-pointer table uploads in `src/gpu_decode.rs`.
- This was run after a broken `release-test` on current main/accuracy had shown QCN collapsing into repeated `S` tokens and failing mini reference validation immediately.
- Post-fix QCN AWQ reference-test result on the same branch state:
  - `./dev reference-test qcn-a4`
  - `13/13` prompts PASS
  - first-token match `12/13`
  - prefill argmax match `249/273 (91%)`
  - prefill top-10 containment `273/273 (100%)`
  - report: `logs/reference-test_20260404_011152/reference_test.html`

---

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

## Standard Benchmarks — 2026-04-01 (QCN Polar4 AWQ after decode harness work)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, INT4 GPU experts, INT4 CPU experts, AWQ attention, Polar4 KV, layer group size 2, timing off.

| Model | GPUs | GPU/CPU bits | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|-------|-----:|-------------:|----------:|---:|----------------:|---------------:|----:|--------------:|-----|
| Qwen3-Coder-Next | 1 | INT4/INT4 | AWQ | Polar4 | 7645.2 | 96.10 | 17010/24576 (69.2%) | 686 MB | [log](20260401_101548_qcn_polar4_awq_5090_decode_harness.log) |

Notes:
- Internal decode results were 96.10 tok/s at 50 tokens, 94.35 tok/s at 100 tokens, and 93.84 tok/s at 250 tokens.
- Internal prefill peaked at 7645.2 tok/s on the 35K-token prompt.
- This confirms the decode padding rewrite and decode-harness changes did not knock QCN Polar4 AWQ out of its expected mid-90 tok/s decode class.

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
| 2026-04-03 20:54 | 7d09912 + local BF-09 | Feed the active decode-store CUDA ordinal into `PrefillModelConfig` so fused-MoE shared-memory capability queries stop assuming GPU 0 | 7,745.6 | 95.55 | 120.95 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_205423_qcn_polar4_awq_5090_bf09_device_ordinal.log) |
| 2026-04-03 21:23 | df6c259 + local BF-13 | Split BF16 shared-expert cuBLAS ownership so `shared_stream` uses a dedicated handle instead of retargeting the main prefill handle across streams | 7,498.6 | 100.62 | 137.97 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_212300_qcn_polar4_awq_5090_bf13_shared_cublas_handle.log) |
| 2026-04-03 22:03 | 68f1557 + local BF-10 | Split fused-MoE sorted scatter finalization into a second same-stream kernel so padding and `expert_ids` are written only after scatter completion | 7,510.0 | 100.18 | 135.55 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_220350_qcn_polar4_awq_5090_bf10_scatter_finalize_split.log) |
| 2026-04-03 22:33 | 1532389 + local FLA fail-closed | Fail startup for linear-attention models when vendored FLA cannot load, keeping `KRASIS_NO_FLA=1` as the only explicit opt-out to the slower custom LA path | 7,572.8 | 97.61 | 134.15 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_223342_qcn_polar4_awq_5090_fla_fail_closed.log) |
| 2026-04-03 23:00 | 1532389 + local FLA fail-closed + C-02 | Preserve raw `q`/`k` in canonical head-major layout before non-FLA `la_apply_beta`, and emit canonical `k_beta` directly from the beta kernel | 7,943.9 | 98.70 | 133.80 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_230008_qcn_polar4_awq_5090_c02_raw_k_canonical.log) |

Notes:
- The BF-03 cache edit built cleanly through `./dev build`.
- This benchmark failed in long VRAM calibration at `39,920` prompt tokens before the timed benchmark section.
- Failure remained `CUDA_ERROR_ILLEGAL_ADDRESS (grid=(39920, 1, 1), block=(1024, 1, 1), smem=4096, nparams=6)`.
- BF-13 cleared warmup, long calibration, HCS load, and the full timed benchmark on the standard QCN AWQ Polar4 path.
- BF-13 behaved like a correctness/stability fix with no obvious throughput regression; decode returned to ~100 tok/s while preserving async shared-expert overlap.
- Reverting the BF-03 cache follow-up did not restore a successful run on this attempt; the same long-calibration illegal-address fault reproduced with the same short-calibration numbers.
- After reboot, the same BF-03 cache state completed the full standard benchmark cleanly and produced the best prefill result in this local series.
- The BF-04 clean import also completed the full standard benchmark cleanly on the same branch state, but did not improve throughput versus the earlier post-reboot BF-03 pass.
- The BF-05 clean import also completed the full standard benchmark cleanly; throughput stayed effectively flat versus BF-04 while preserving the signed rewind-safe pointer-table fetch path.
- The no-valid-block guard also completed the full standard benchmark cleanly; it is a cheap control-flow correctness fix and did not regress standard prefill or decode throughput.
- The BF-09 actual-device-ordinal wiring also completed the full standard benchmark cleanly; on this single-GPU path it behaves like a correctness/generalization fix rather than a speed optimization.
- BF-10 also completed the full standard benchmark cleanly; the extra finalize launch did not materially change throughput on the standard QCN AWQ Polar4 path and removed the grid-wide race from the sorted scatter/finalize step.
- The FLA fail-closed change also completed the full standard benchmark cleanly; startup still succeeds on the shipped QCN path with vendored FLA present, but LA models will now fail visibly instead of silently degrading to the older custom LA kernels when FLA sidecar loading breaks.
- The C-02 layout-preservation change also completed the full standard benchmark cleanly on the combined branch state. This benchmark still exercises the shipped FLA path, so it confirms no regression in the standard product path but does not by itself prove non-FLA correctness.

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

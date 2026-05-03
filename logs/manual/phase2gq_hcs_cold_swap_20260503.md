# Phase 2GQ - HCS Cold Expert Swaps

Date: 2026-05-03

## Goal

Implement an explicit opt-in approximate decode mode that swaps lower-rank
selected cold experts to close same-layer HCS-resident substitutes, then test
Q122B accuracy and timing-free speed.

## Implementation

- Env flag: `KRASIS_HCS_COLD_SWAP=1`
- Default: off, exact decode unchanged.
- Incompatible with `KRASIS_GPU_ROUTE_SYNC=1`; the mode requires CPU route
  classification and fails closed if route-sync is enabled.
- Policy defaults:
  - protect top `75%` selected ranks
  - absolute router-score tolerance: `0.005`
  - relative router-score tolerance: `10%`
- Override knobs:
  - `KRASIS_ROUTE_SWAP_PROTECT_RANK_PCT`
  - `KRASIS_ROUTE_SWAP_ABS_TOL`
  - `KRASIS_ROUTE_SWAP_REL_TOL_PCT`
  - `KRASIS_HCS_COLD_SWAP_LOG=1` for per-swap verbose logging

The selected router weight is preserved. Only the selected expert ID is changed
for low-rank cold slots that match a same-layer HCS-resident substitute under
the policy.

## Accuracy

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_HCS_COLD_SWAP=1 ./dev witness-compare tests/q122b-k4v4-hqq6-int4-benchmark.conf --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

Run dir:
`logs/reference-test_20260503_200047`

Result:

| Metric | Result |
| --- | ---: |
| Overall | PASS |
| Prompts | `14/14` |
| First token | `14/14` |
| Prefill argmax | `14/14` |
| Prefill top-10 | `14/14` |
| Generated exact | `261/361` |
| Generated containment | `279/361` |
| Full exact prompts | `8/14` |
| First-token top-10 exact | `0/14` |
| First-token top-10 overlap | `130/140` |

Witness swap aggregate:

| Metric | Result |
| --- | ---: |
| Swaps | `7056` |
| Decode tokens | `347` |
| Swaps/token | `20.33` |
| Weighted cold after swaps | `143.69/tok` |
| Weighted original-cold saved | `12.6%` |
| Weighted score delta | `0.004335` |
| Weighted relative delta | `5.49%` |

Compared with the exact prompt-HCS default witness (`280/361` exact,
`303/361` containment), the swap mode is clearly less accurate even though it
passes the current seq32 gate.

## Speed

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_HCS_COLD_SWAP=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Run dir:
`logs/dev-benchmark_20260503_200905`

Timing instrumentation was disabled.

| Metric | Result |
| --- | ---: |
| Prefill best | `4280.2 tok/s` |
| Decode internal best | `27.30 tok/s` |
| Round trip best | `50.01 tok/s` |
| HCS | `3780/12288` |
| Min free VRAM | `662 MB` |

Official internal decode swap aggregate:

| Metric | Result |
| --- | ---: |
| Swaps | `7586` |
| Decode tokens | `397` |
| Swaps/token | `19.11` |
| Weighted cold after swaps | `109.03/tok` |
| Weighted original-cold saved | `14.9%` |
| Weighted score delta | `0.004325` |
| Weighted relative delta | `5.30%` |

## Interpretation

HCS cold swaps produce a real decode-speed improvement on Q122B:

- exact prompt-HCS default: `25.29 tok/s`
- HCS cold swaps: `27.30 tok/s`

That is about `+7.9%` internal decode. The tradeoff is accuracy: generated
exact dropped `280/361 -> 261/361`, and containment dropped `303/361 ->
279/361`. This should remain opt-in and approximate, not default.

Next useful tests would be stricter policies:

- protect `80/85/90%` ranks
- reduce relative tolerance below `10%`
- optionally require an expert-similarity matrix instead of router-score
  closeness alone

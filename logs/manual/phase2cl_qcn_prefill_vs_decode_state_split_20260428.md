# Phase 2CL QCN Prefill-vs-Decode State Split

Date: 2026-04-28

Reference surface: `phase2bn_qcn_64tok`, 14 QCN prompts, thinking disabled.

Question: does Krasis fail to score the BF16 next token when given the correct
BF16 prefix, or does the free-running decode trajectory drift into a different
history?

## Inputs

- Free-running decode run:
  `logs/reference-test_20260428_080508/reference_test_summary.json`
  - config: BF16 attention + BF16 KV
  - result: avg exact prefix `18.79`, total exact `263/653`
- Teacher-forced prefix run:
  `logs/reference-test_20260428_082353/reference_test_summary.json`
  - profile: `phase2cd_qcn_teacher_forced_prefixes`
  - each case input is original BF16 prompt tokens plus the BF16 witness prefix
    up to the prior Krasis divergence point
  - reference is the next BF16 token only

## Aggregate Split

| Measurement | Result |
| --- | ---: |
| Free-running divergent positions | `9/14` cases |
| BF16 token in free-running decode top-10 at divergence | `9/9` |
| BF16 token rank-1 in free-running decode at divergence | `0/9` |
| BF16 token rank 2 or 3 in free-running decode at divergence | `9/9` |
| Teacher-forced prefill-from-scratch top-1 at same BF16 prefixes | `12/14` |
| Teacher-forced prefill-from-scratch top-10 at same BF16 prefixes | `14/14` |
| Teacher-forced one-token decode from same BF16 prefixes | `13/14` |

## Per-Case Split

| Case | Prompt | Free run | BF16 rank at free divergence | TF prefill | TF decode |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Hi | `12/12` | full | `1/1` | match |
| 1 | What's your name? | `4/34` | `2` | `1/1` | match |
| 2 | Who trained you? | `17/64` | `3` | `0/1` | miss |
| 3 | What is 2+2? | `9/9` | full | `1/1` | match |
| 4 | Now multiply that by 10 | `11/11` | full | `1/1` | match |
| 5 | And divide the result by 5 | `11/11` | full | `1/1` | match |
| 6 | What is the largest animal in the world? | `1/64` | `2` | `1/1` | match |
| 7 | What is the largest body of water in the world? | `27/64` | `2` | `1/1` | match |
| 8 | Describe the binary chop algorithm in depth | `29/64` | `3` | `1/1` | match |
| 9 | Towels drying | `6/64` | `3` | `0/1` | match |
| 10 | Blue whale facts | `46/64` | `2` | `1/1` | match |
| 11 | Whales in general | `11/64` | `2` | `1/1` | match |
| 12 | Whale geography | `15/64` | `3` | `1/1` | match |
| 13 | Quicksort in Rust | `64/64` | full | `1/1` | match |

## Interpretation

The split points away from BF16 attention weights and KV precision as the main
cause. At the positions where free-running decode diverges, Krasis usually has
the BF16 token very close but not rank-1. When the correct BF16 prefix is
supplied and recomputed by prefill, Krasis recovers the BF16 next token in
top-1 for `12/14` cases and top-10 for `14/14`.

That means the low exact-prefix average is mostly accumulated trajectory drift:
small early rank swaps create a different generated history, and later tokens
are conditioned on that different history. The model is usually capable of
scoring the BF16 next token from the correct history.

## Remaining Visibility Gap

Current trace can emit LA state samples at decode start and selected tensor/logit
events, but it does not persist comparable LA recurrent/conv state checksums at
arbitrary generated positions. A stronger next instrumented run should add a
trace-gated LA state summary after each QCN linear-attention layer at selected
decode steps, then compare:

1. prefill-from-scratch state at BF16 prefix end
2. free-running decode state at the same generated position

Until that exists, the evidence is logit-level rather than direct state-checksum
level.

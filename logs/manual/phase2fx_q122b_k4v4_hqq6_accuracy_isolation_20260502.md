# Phase 2FX - Q122B k4v4/HQQ6 Accuracy Isolation

Date: 2026-05-02

## Question

The exact Q122B benchmark surface, HQQ6 attention plus k4v4 KV, failed seq32
witness badly. The question was whether this contradicts prior accurate k4v4
and HQQ results, and whether the failure is model-specific.

## New Isolation Run

```bash
./dev witness-compare tests/q122b-k4v4-a16-accuracy.conf \
  --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

Artifacts:

- Run dir: `logs/reference-test_20260502_210643`
- Log: `logs/manual/phase2fx_q122b_k4v4_a16_witness_seq32_20260502.log`
- Summary: `logs/reference-test_20260502_210643/reference_test_summary.json`

## Q122B seq32 Matrix

| Surface | Exact | Full | Containment | First token |
| --- | ---: | ---: | ---: | ---: |
| BF16/BF16 | `286/361` (`79.2%`) | `9/14` | `306/361` (`84.8%`) | `14/14` |
| HQQ6/BF16-KV | `283/361` (`78.4%`) | `8/14` | `305/361` (`84.5%`) | `14/14` |
| BF16-attn/k4v4 | `247/361` (`68.4%`) | `8/14` | `291/361` (`80.6%`) | `14/14` |
| HQQ6/k4v4 | `28/361` (`7.8%`) | `0/14` | `28/361` (`7.8%`) | `14/14` |

## Prior Combined-Surface Evidence

QCN had prior compact-KV plus HQQ runs:

| Surface | Exact | Full | Containment | Note |
| --- | ---: | ---: | ---: | --- |
| QCN HQQ8/k4v4 | `524/653` (`80.2%`) | `9/14` | `544/653` (`83.3%`) | good combined surface |
| QCN HQQ8/k8v4 | `450/653` (`68.9%`) | `8/14` | `504/653` (`77.2%`) | good enough combined surface |
| QCN HQQ6/k4v4 | `175/653` (`26.8%`) | `3/14` | `286/653` (`43.8%`) | already weak |

## Interpretation

This is not explained by k4v4 alone on Q122B. BF16 attention plus k4v4 passes
seq32 broadly, with only modest degradation versus BF16 KV.

This is also not explained by HQQ6 attention alone. HQQ6 with BF16 KV is about
as good as BF16/BF16 on the same seq32 artifact.

The failure appears when HQQ6 and k4v4 are combined. Older QCN data suggests
that HQQ6+k4v4 was already a weak combined surface, while HQQ8+k4v4 was good.
So the current Q122B result is probably not a purely model-specific bug; it is
more likely a combined HQQ6+k4v4 decode interaction that becomes catastrophic
on Q122B.

## Next Checks

1. Run Q122B HQQ8+k4v4 if cache/runtime support exists, because QCN HQQ8+k4v4
   previously passed and this would separate HQQ6-specific behavior from all
   HQQ compact-KV behavior.
2. Run Q122B HQQ6+k6v6 if available, to see whether k4v4-specific decode KV is
   part of the interaction.
3. Trace token 2 on Q122B HQQ6+k4v4 against HQQ6/BF16-KV and BF16-attn/k4v4,
   focusing on GQA KV dequant/read descriptors and HQQ6 decode projection state.
4. Tighten reference-test pass criteria so low seq32 decode containment cannot
   report overall `PASS`.

# Regression Review: 03f22e7 -> Current Tree

Generated: 2026-03-31
Repo: /home/main/Documents/Claude/krasis
Known fast commit: `03f22e7`
Current HEAD: `1c1e0d3`

## Why this file exists

We saw QCN 1-GPU Polar4/AWQ decode fall from the earlier `~95-97 tok/s` range to roughly `87-89 tok/s`.
This file reviews every code change between the known fast commit and the current tree, with a performance lens.

Important scope boundary:

- The committed git range `03f22e7..HEAD` changes only:
  - `python/krasis/model.py`
  - `src/gpu_decode.rs`
- The current uncommitted worktree also changes:
  - `src/gpu_prefill.rs`
  - `src/cuda/marlin/marlin_moe_vendor.cu`
  - `src/cuda/marlin/moe/marlin_template.h`

Those uncommitted files are the active fused-MoE prefill fixes. They are not part of the committed regression range from the known fast decode point.

## Benchmark context

Known fast point:

- `03f22e7`
- recorded QCN Polar4/AWQ decode around `96.67` and `95.35 tok/s`

Current measured runs:

- run 1: best internal decode `87.47 tok/s`
- run 2: best internal decode `89.33 tok/s`

So the observed drop is real enough to investigate, but the surviving benchmark data is noisy:

- one run had a weak `100-token` result relative to `50` and `250`
- the rerun had early EOS on `100` and `250`, so only one full decode point survived

That means the slowdown signal is plausible, but the benchmark evidence is still weaker than a clean 3/3 decode sweep would be.

## Commit-by-commit review

### 1. `b439f4c` Fix multi-GPU Polar4 KV cache: aux allocation and cross-GPU copy

Files:

- `python/krasis/model.py`
- `src/gpu_decode.rs`

What changed:

- Added aux-GPU Polar4 KV cache allocation and pointer registration in `model.py`
- Added `kv_format` validation and a Polar4-specific `copy_kv_to_aux()` path in `gpu_decode.rs`
- Polar4 copy path now copies 4 tensors per GQA layer:
  - `k_radius`
  - `v_radius`
  - `k_angles`
  - `v_angles`
- The copy is done as D2H then H2D through a host buffer, with GPU binds and syncs around the copies

Performance relevance:

- High relevance for multi-GPU prompt handoff / first-token latency
- Low relevance for single-GPU steady-state decode tok/s

Why:

- `copy_kv_to_aux()` is called after prefill to populate the aux GPU's KV cache
- That is request setup work, not the per-token single-GPU decode loop
- The single-GPU QCN benchmark uses `CFG_SELECTED_GPUS="0"`, so this path should not run in the benchmark that produced `87-89 tok/s`

Possible regression sources inside this commit:

1. Multi-GPU request startup regression
   - Polar4 copies 4 tensors instead of 2
   - It allocates a fresh host buffer per layer
   - It binds/synchronizes devices around copies
   - This can absolutely slow down multi-GPU handoff and first-token time

2. Multi-GPU scaling regression
   - No peer-to-peer path
   - Full D2H/H2D bounce through system memory
   - More CPU orchestration and synchronization than the fast path would want

Likelihood this explains the observed single-GPU decode drop:

- Very low

Conclusion:

- Real performance risk exists here, but for multi-GPU startup / cross-GPU KV transfer, not for the measured 1-GPU decode tok/s regression.

### 2. `a69090b` Fix multi-GPU GQA aux store: use config/dict instead of None attn object

Files:

- `python/krasis/model.py`

What changed:

- GQA aux-store setup now pulls weights from `layer.gqa_weights` instead of assuming an `attn` object
- Q/K norm pointers now come from that dict
- GQA metadata now comes from config values rather than `attn`
- Aux RoPE tables are rebuilt from config on the aux GPU

Performance relevance:

- Medium relevance for multi-GPU setup correctness
- Very low relevance for single-GPU decode tok/s

Why:

- This is aux-store construction logic
- It exists to make split-GPU GQA layers work when the attention object is `None`
- None of this is in the hot per-token single-GPU decode loop

Possible regression sources inside this commit:

1. Slightly higher aux setup work
   - Recomputing RoPE tables from config on aux GPU
   - Extra tensor copies to aux device

2. Slightly more aux-GPU memory pressure
   - Extra aux allocations for norms / tables / weights

Likelihood this explains the observed single-GPU decode drop:

- Extremely low

Conclusion:

- Correctness fix for multi-GPU GQA aux setup
- Not a credible explanation for lower 1-GPU decode throughput

### 3. `1c1e0d3` Cherry-pick Ampere fixes: RoPE `.to(device)` and safer expert padding

Files:

- `python/krasis/model.py`
- `src/gpu_decode.rs`

This is the only commit in the regression range that touches the single-GPU decode hot path directly.

#### 3a. RoPE `.to(device)` in `model.py`

What changed:

- RoPE cosine/sine tables are explicitly moved to the target device when created

Performance relevance:

- Startup/init only

Likelihood as decode regression source:

- Negligible

Conclusion:

- Necessary correctness / device-placement fix
- Not a steady-state decode throughput issue

#### 3b. Safer expert padding in `src/gpu_decode.rs`

What changed:

- Old behavior:
  - when `batch_count < topk`, padding pointers came from a synthetic dummy expert layout
  - code could fetch dummy pointer table contents from device to host
- New behavior:
  - if at least one real expert exists, zero-weight padding reuses that first real expert's packed/scales pointers
  - only falls back to dummy pointers if `batch_count == 0`

Why it was changed:

- To avoid graph replay depending on synthetic dummy layout details
- To make padding safer across architectures

Performance relevance:

- This is in the decode hot path
- It runs during batch expert pointer preparation
- So it is the one committed change that must be treated seriously for a single-GPU decode regression review

Expected performance direction from the code alone:

- Neutral to slightly positive

Why:

1. It removes a tiny D2H dependency in the common `batch_count > 0` case
   - old code could read `d_dummy_ptrs` back to host
   - new code skips that and reuses already-available host pointers

2. It does not increase the number of padded lanes
   - compute shape is unchanged
   - `topk` padding still exists either way

3. It does not introduce a new sync
   - the change is pointer selection only

Potential negative side effects worth considering:

1. Cache behavior could change
   - zero-weight padded lanes now point at a real expert's weights instead of a dummy expert buffer
   - if kernels still dereference those zero-weight lanes deeply enough, this could increase traffic against a real expert buffer

2. More overlap on one expert's memory region
   - all padded lanes collapse onto the same first-real-expert pointers
   - if there is a subtle memory-system effect here, it would show up only in some routed-expert patterns

But from the code alone, this still looks more likely to help or be neutral than to cause a roughly `7-9 tok/s` drop.

Likelihood this explains the observed single-GPU decode drop:

- Low, but not zero
- It is the only committed change that touches the 1-GPU decode hot path directly, so it remains the top code candidate by elimination
- It is not a strong candidate from first principles

Conclusion:

- This is the only committed diff item that deserves direct A/B measurement against the decode benchmark
- But analytically it does not look like an obvious regression

## Review of current uncommitted worktree

Current uncommitted files:

- `src/gpu_prefill.rs`
- `src/cuda/marlin/marlin_moe_vendor.cu`
- `src/cuda/marlin/moe/marlin_template.h`

These are fused-MoE prefill bugfixes:

- negative `-1` padded-row handling
- MoE block bounds guards
- pointer-table byte-address arithmetic cleanup
- per-block `B` progression reset
- corrected shared-memory metadata sizing / launch cap
- larger fused-MoE lock workspace and fp32 reduction scratch sizing

Performance relevance:

- Prefill path
- Not the Rust/GPU decode hot path that produced the `87-89 tok/s` measurements

Conclusion:

- These changes can affect prefill behavior and prefill performance
- They are not a credible explanation for the decode regression by themselves

## Ranked regression candidates

### For the observed 1-GPU decode tok/s drop

1. `src/gpu_decode.rs` safer expert padding
   - Only because it is the sole committed hot-path decode change in range
   - Code inspection does not make it look strongly regressive

2. Benchmark/data instability rather than code
   - Early EOS in 2 of 3 decode prompts in the rerun
   - Non-uniform decode results across prompt lengths
   - Current evidence is not yet a clean apples-to-apples decode timing set

3. Runtime/state/config drift outside this diff
   - GPU state, clocks, thermal behavior, other resident processes, allocator/cache state
   - Not proven yet, but more plausible than most of the committed code changes here

### For performance regressions anywhere, not just this 1-GPU decode number

1. Multi-GPU KV handoff in `copy_kv_to_aux()`
   - Most credible real regression in this range
   - Could hurt first-token latency and multi-GPU prompt handoff materially

2. Aux-store setup growth in `model.py`
   - Mostly startup/setup overhead
   - Not per-token decode throughput

3. Safer expert padding
   - Possible but weak candidate for decode throughput

## Bottom line

After reviewing every code change between `03f22e7` and the current tree:

- there is no strong committed code candidate that cleanly explains the observed 1-GPU decode drop
- the only committed single-GPU decode hot-path change is the safer expert padding logic in `src/gpu_decode.rs`
- that change does not look obviously slower from code inspection; if anything it removes a tiny host/device dependency in the common case
- the strongest real performance-regression risk in this range is actually multi-GPU KV handoff overhead, not single-GPU decode tok/s

So the correct next step is measurement, not theory:

1. A/B benchmark current `HEAD` vs `03f22e7`
2. If needed, A/B benchmark with only the safer expert padding hunk toggled
3. Use timing-enabled decode profiling only to localize slower components, not to claim speed numbers

Without that A/B data, the code review result is:

- multi-GPU startup regression risk: yes
- obvious 1-GPU decode regression source in this diff: no
- most likely code candidate to test anyway: safer expert padding in `src/gpu_decode.rs`

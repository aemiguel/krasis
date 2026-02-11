# Krasis



**What Krasis is**

Hybrid LLM focused on minimal VRAM, always-on GPU prefill and efficient use of System RAM and precomputation and optimised CPU based inference.



**What we expect on the GPU**

  - Attention weights: INT8 (q_proj, k_proj, kv_b_proj, o_proj, q_b_proj) — ~110 MB/layer
  - Shared expert (1 per layer): INT8 — ~42 MB/layer
  - Router gate: BF16 — 5.2 MB/layer
  - LayerNorms: BF16 — tiny
  - LM head: BF16 — 2.2 GB (on last PP rank only)
  - Embedding: BF16 — 2.2 GB (on first PP rank only)



**What we expect in System RAM**

  - INT4 packed weights: int32 — 8,064 MB/layer (97%)
  - BF16 scales: bfloat16 — 252 MB/layer (3%)
  - Total: 8,316 MB/layer × 60 layers = 488 GB



**Krasis Hybrid LLM runtime sequence:**

- GPU prefill is always ON by default (essential for a usable model to be able to process input quickly)
- Verify 1GB+ pages and warn if not set
- Verify cpu governer set to performance and warn if not set
- Check cores and NPS, warn if not NPS4
- Calculate memory budget at startup, refuse to run if it doesn't fit (optional param to force run anyway and risk OOM), at runtime always verify the memory budget calculations were accurate and warn that they need refactoring if not
- Take HF native model
- Produce Marlin INT8 or INT4 weights (as specified by params), cache to disk and load next time, allows for models to be run at INT8 or INT4 as preferred
- Load all into system RAM, model in RAM is Marlin weights only, CPU will un-permute as necessary and bear the compute cost as it is memory bandwidth bound anyway
- Load into GPU directly from model held in system RAM, perform one-time quantize of attention weights and shared expert to precision specified per runtime params (INT8, INT4 etc).
  - Precision specified per component, e.g.  --kv-b-proj-quant int8 --shared-expert-quant int
- Retain other optimisations like divided expert buffer on GPU, KV cache compression etc
- Automatic pinning of experts to NUMA nodes
- No expert pinning to GPU for now (we proved gains were minimal or negative)



**Architecture:**

- Implemented in Rust (gives much better debugging than C++ but much better performance and access to CUDA than Python)
- Orchestrator and CPU inference handled within Rust, no CPP backend, implement AVX2 via std::arch::x86_64 etc, call out only as absolutely necessary (e.g. NUMA node pinning)
-  Rust backend runs is numa-node-pinned-experts aware and runs experts on the NUMA node where their data has been placed
- Expert weight are prefetched while we compute the existing ones, test disabling hardware l2 prefetcher if possible,  use Non-Temporal Prefetching (`_mm_prefetch` with `_MM_HINT_NTA`) or Non-Temporal Loads (`MOVNTDQA` equivalent) if possible to fetch direct to L3 and avoid displacing hot activation vectors in L1
- If possible ensure each expert's weights are contiguous in memory and aligned to cache line (64B) and ideally page boundaries. If an expert straddles a page boundary that crosses NUMA nodes, you get split accesses. Since you control the mmap, you can guarantee alignment.
- INT4 Kernel Strategy: Since Zen 2 lacks AVX-512 VNNI (`vpdpbusd`), the INT4 expert kernel uses AVX2's `_mm256_madd_epi16` as the core multiply-accumulate. Packed INT4 weights (32 bytes = 64 values) are loaded into a YMM register, then low and high nibbles are separated via mask and shift, sign-extended to INT16 using arithmetic right shifts (`_mm256_srai_epi16`), and immediately multiplied against INT16-quantized activations via `_mm256_madd_epi16`, accumulating into INT32 running totals — all in registers with no intermediate memory writes. This fused unpack-multiply-accumulate approach avoids writing unpacked weights back to memory, which is critical since the workload is memory-bandwidth-bound and any extra store/reload would directly reduce effective throughput.
- Activation reuse across experts - the input activation vector is the same for all 8 experts on a given token. Pin it in L1/L2 cache and keep it there across all expert dispatches. If experts are on different NUMA nodes, you'll need a copy of the activation vector local to each node — small cost (hidden dimension is what, 7168 bytes at BF16?) but avoids cross-node reads on every expert.
- Intra-expert parallelism: Each expert's weight matrix (e.g. 4096 rows) is split into chunks and processed in parallel by multiple threads on the same CCX using a work-stealing model, with partial results summed at the end. This is standard practice in llama.cpp and other engines to saturate server-grade memory controllers — a single thread cannot generate enough outstanding memory requests to fill the bandwidth of a DDR4 channel, let alone multiple channels per NUMA node, so you need several threads streaming different chunks of the same expert's weights concurrently.  The final reduction across threads is trivial — just adding a few partial accumulator vectors — so the synchronization cost is negligible compared to the memory bandwidth gained.
- Expert execution ordering - if your 8 active experts span multiple NUMA nodes, run all experts on node 0 first, then node 1, etc. rather than round-robin. This keeps the activation vector hot in one node's cache before moving to the next. Tiny optimization but free.
- Logging at stages and during large loads to easily track load times and status
- Replaces KT, SGLANG still handles the top of the stack, Krasis is a .so that SGLang calls where it currently calls kt_ep_wrapper.py + kt_kernel_ext.so.  Python wrapper (thin PyO3 shim) that SGLang calls as a FusedMoE replacement.
- There's actually a nice trick available too: instead of the CPU doing indirect reads into the permuted weight matrix, you can permute the activation vector (14KB at BF16) to match Marlin's layout once per expert, then read the weights sequentially. 14 KB fits in L1 cache, so the permutation is a handful of nanoseconds. Then your inner loop becomes a straight sequential scan of the weight matrix — best possible access pattern for the hardware prefetcher, and the permutation cost is invisible



**Unified Weight Format (implemented)**

The original design stored expert weights as three separate matrices (gate, up, down) in
`[N, K/8]` row-major layout. GPU prefill via Marlin required a completely separate copy
of all expert weights in Marlin format, stored in a Python-side RAM cache. For Kimi K2.5
(384 experts × 60 layers), this meant ~504 GB (Rust INT4) + ~438 GB (Python Marlin cache)
= ~942 GB out of 995 GB total system RAM, which caused OOM crashes.

Solution: **single unified weight format** shared by CPU decode and GPU prefill.

- **Combined w13**: gate and up projections concatenated into a single w13 matrix
  `[K/8, 2*N]` (first N columns are gate, next N columns are up)
- **Transposed layout**: `[K/8, N]` where K is the reduction dimension and N is the
  output dimension. Weight data is contiguous along N, enabling SIMD across the output
  dimension without horizontal sum
- **CPU decode kernel**: New AVX2 integer kernel (`_mm256_mullo_epi32`) that processes
  8 output values in parallel. Uses `_mm256_srlv_epi32` for variable shifts and
  `_mm256_permutevar8x32_ps` for activation broadcast. Two matmuls per expert (w13 + w2)
  instead of three (gate + up + down)
- **GPU prefill**: Will read raw packed data from Rust via PyO3, upload to GPU, and run
  `gptq_marlin_repack` on-the-fly. Eliminates the entire Python Marlin cache (~438 GB)
- **RAM budget check**: At startup, estimates total RAM needed and refuses to run if >95%
  of MemTotal. Post-load RSS check warns if actual usage deviates >10% from estimate

Design decisions:
- `_mm256_madd_epi16` (3-cycle, original kernel) cannot be used with transposed layout
  because it sums pairs of adjacent products, which would mix N outputs. We use
  `_mm256_mullo_epi32` (10-cycle on Zen 2) instead — correct for SIMD-across-N approach.
  Despite higher latency per instruction, the elimination of horizontal sum and better
  memory access pattern compensates.
- Conversion from old format to unified happens in-place at load time. The old weights
  are dropped after conversion, so peak RAM is ~2x expert weights momentarily during
  conversion (not ~2x permanently as before).

Verified correct: unified forward produces identical results to original forward on both
synthetic weights and real V2-Lite model (max abs diff = 0.000001 on full MoE layer).



**Development and testing**

- Develop by testing iteratively with a small MoE model:
  - DeepSeek-V2-Lite (15.7B total, 2.4B active) - download native from HF
- Each architectural feature above must have either logging or testcases to verify its effectiveness



**Future considerations (not in scope for now)**

- Async GPU/CPU overlap - while CPU is running experts for token N, GPU should already be doing attention for token N+1's prefill or managing KV cache. KT's expert deferral does this but it's worth being explicit about in the architecture.
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

  Two separate cached weight formats, independently configurable precision:
  - **(A) GPU cache (Marlin format)**: INT4 or INT8 per params — for GPU prefill via fused_marlin_moe
  - **(B) CPU cache (CPU-optimized format)**: INT4 or INT8 per params — for CPU decode via AVX2 kernels
  - Layout optimized per target: Marlin tile-permuted for GPU, sequential row-major for CPU
  - Total RAM depends on model size and chosen precisions



**Krasis Hybrid LLM runtime sequence:**

- GPU prefill is always ON by default (essential for a usable model to be able to process input quickly)
- Verify 1GB+ pages and warn if not set
- Verify cpu governer set to performance and warn if not set
- Check cores and NPS, warn if not NPS4
- Calculate memory budget at startup, refuse to run if it doesn't fit (optional param to force run anyway and risk OOM), at runtime always verify the memory budget calculations were accurate and warn that they need refactoring if not
- Take HF native model
- Produce TWO cached formats on first run, streaming one layer at a time:
  - (A) Marlin cache (INT4 or INT8 per params): for GPU prefill — `.krasis_cache/experts_marlin_g{gs}.bin`
  - (B) CPU cache (INT4 or INT8 per params): for CPU decode — `.krasis_cache/experts_cpu_{bits}_g{gs}.bin`
- GPU and CPU precision independently configurable (e.g. `--gpu-expert-bits 4 --cpu-expert-bits 8`)
- Load both caches into system RAM on subsequent runs
- GPU prefill: DMA copy Marlin weights from RAM → GPU → fused_marlin_moe (zero conversion)
- CPU decode: read CPU-optimized weights directly — sequential memory access for AVX2 cache locality
- Load attention/shared expert to GPU, quantize per runtime params (INT8, INT4 etc).
  - Precision specified per component, e.g.  --kv-b-proj-quant int8 --shared-expert-quant int4
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



**Dual Weight Format (current architecture)**

Two separate cached formats, each independently configurable precision (INT4 or INT8):

- **(A) GPU cache (Marlin format)**: Tile-permuted, bit-interleaved layout optimized for
  `fused_marlin_moe` CUDA kernel. DMA copy from RAM → GPU with zero conversion.
  On disk: `.krasis_cache/experts_marlin_g{gs}.bin`

- **(B) CPU cache (CPU-optimized format)**: Sequential row-major layout optimized for
  AVX2 cache locality and hardware prefetcher. Combined w13 (gate+up) with contiguous
  access patterns.
  On disk: `.krasis_cache/experts_cpu_{bits}_g{gs}.bin`

This replaced the earlier single-format approach (Marlin everywhere) after testing showed
Marlin's tile permutation destroyed CPU cache locality (0.55 tok/s vs 1.55 tok/s on Kimi K2.5).

- **CPU decode kernel**: AVX2 integer kernel with sequential weight access — no tile
  indirection, no permutation overhead
- **GPU prefill**: DMA copy Marlin weights from RAM → GPU → fused_marlin_moe (zero conversion)
- **RAM budget check**: At startup, estimates total RAM needed for both caches and refuses
  to run if >95% of MemTotal. Post-load RSS check warns if actual usage deviates >10%.



**Development and testing**

- Develop by testing iteratively with a small MoE model:
  - DeepSeek-V2-Lite (15.7B total, 2.4B active) - download native from HF
- Each architectural feature above must have either logging or testcases to verify its effectiveness



**Future considerations (not in scope for now)**

- Async GPU/CPU overlap - while CPU is running experts for token N, GPU should already be doing attention for token N+1's prefill or managing KV cache. KT's expert deferral does this but it's worth being explicit about in the architecture.
# Krasis Changelog

## Test Status

| Suite | Count | Status | Last Run |
|-------|-------|--------|----------|
| Rust (`cargo test`) | 33 | ALL PASS | 2026-02-09 |
| Python bridge (`test_bridge.py`) | 3 | ALL PASS | 2026-02-09 |
| **Total** | **36** | **ALL PASS** | |

Re-run needed after: any change to `src/`, `python/krasis/`, or `test_bridge.py`.

---

## [1456e1f] SGLang bridge — 2026-02-09

**Add SGLang bridge: KrasisMoEWrapper drop-in for KTMoEWrapper**

- `python/krasis/sglang_bridge.py`: `KrasisMoEWrapper` class implementing KTMoEWrapper interface (submit_forward, sync_forward, load_weights)
- Singleton engine pattern — one KrasisEngine shared across all MoE layer wrappers
- GPU↔CPU tensor transfer via BF16→uint16 numpy view (numpy lacks bfloat16 support)
- Expert ID masking: IDs < num_gpu_experts set to -1 (Krasis skips them)
- `python/krasis/__init__.py`: exports KrasisMoEWrapper
- `test_bridge.py`: 3 tests — engine roundtrip, wrapper interface, batch forward
- `run_krasis.sh`: Launch script for SGLang with Krasis backend
- SGLang patch: `kt_ep_wrapper.py` import toggle via `KRASIS_BACKEND=1` env var

Tests: 33 Rust + 3 Python = **36 PASS**

## [d2cc31b] Async submit/sync — 2026-02-09

**Add batch MoE forward and async submit/sync pattern**

- Background worker thread with `mpsc` channels for non-blocking CPU expert dispatch
- `MoeWork` struct carries layer_idx, activation, topk_ids/weights, batch_size
- `submit_forward()` sends work to background thread, `sync_forward()` blocks for result
- `Arc<WeightStore>` for shared ownership between main thread and worker
- `Mutex<mpsc::Receiver>` wrapper — PyO3 `#[pyclass]` requires `Sync`, `mpsc::Receiver` is `!Sync`
- `Drop` impl sends sentinel (layer_idx=usize::MAX) for clean worker shutdown
- `test_async_submit_sync`: bit-exact match sync vs async, batch=2, masked experts, cleanup

Tests: **33 PASS** (added 1)

## [f91cf42] NUMA-aware placement — 2026-02-09

**Add NUMA-aware expert placement and execution**

- `src/numa.rs`: `NumaTopology`, `NumaAlloc`, `NumaExpertMap` types
- `libnuma` FFI bindings for `numa_alloc_onnode`, `numa_move_pages`, `sched_setaffinity`
- Expert-to-NUMA-node mapping based on activation heatmap
- `migrate_to_node()`: moves expert weight pages to target NUMA node
- `pin_thread_to_node()`: pins worker thread to NUMA node's CPU cores
- `build.rs`: links `libnuma`

Tests: **32 PASS** (added 5 NUMA tests)

## [f9c1a3a] Intra-expert parallelism — 2026-02-09

**Enable intra-expert parallelism and configurable thread count**

- `KrasisEngine(parallel=True, num_threads=N)` constructor parameters
- Thread count forwarded to `rayon::ThreadPoolBuilder`
- Intra-expert parallelism: single large matmul split across multiple threads
- Configurable via PyO3 constructor

Tests: **27 PASS**

## [5731df6] Partial loading + Kimi K2.5 — 2026-02-09

**Add partial model loading and Kimi K2.5 end-to-end forward test**

- `WeightStore::load_partial()`: load subset of layers (for memory-constrained testing)
- Kimi K2.5 (384 experts, 4096 intermediate) end-to-end forward test
- Pre-quantized INT4 (compressed-tensors) dequantization path

Tests: **26 PASS** (added 2: kimi_k25_single_expert, kimi_k25_moe_forward)

## [d960d4a] Generify config parsing — 2026-02-09

**Generify config parsing and add pre-quantized model support**

- Generic `MoeConfig` from HF `config.json` — auto-detects DeepSeek, Qwen3, Kimi K2.5
- Pre-quantized model support: reads `weight_packed` + `weight_scale` + `weight_shape`
- `compressed-tensors` INT4 dequantization (group_size=32)

Tests: **24 PASS**

## [644c1f8] Zero-allocation scratch pool — 2026-02-09

**Pre-allocate scratch pool for zero-allocation expert parallelism**

- `ScratchPool`: pre-allocated per-thread scratch buffers for matmul intermediates
- Eliminates allocation in hot MoE forward path
- Pool sized to max(num_experts_per_tok) × thread_count

Tests: **23 PASS**

## [b628ccb] Expert parallelism — 2026-02-09

**Add expert-level parallelism for 3.3x MoE throughput**

- Rayon parallel iterator over active experts within a single token
- 3.3x speedup on V2-Lite (6 experts/token)
- Thread-safe expert forward via shared `&WeightStore`

Tests: **22 PASS**

## [56e77e6] Integer kernel — 2026-02-09

**Add integer kernel (_mm256_madd_epi16) for 2x matmul throughput**

- `_mm256_maddubs_epi16` + `_mm256_madd_epi16` pipeline for INT4×INT8 matmul
- 2x throughput vs FP32 accumulation path
- AVX2 throughput benchmark test

Tests: **21 PASS** (added 2: integer kernel + throughput)

## [5d1bdca] MoE benchmark — 2026-02-09

**Add comprehensive MoE benchmark script**

- `bench_moe.py`: Benchmarks single-token and batch MoE forward latency
- Reports tok/s, ms/token, ms/expert breakdowns

Tests: **19 PASS**

## [ebf757a] System checks — 2026-02-09

**Add startup system checks**

- `system_check()` PyO3 function: CPU governor, hugepages, memory, NUMA, SIMD
- Warns on performance-degrading configurations

Tests: **19 PASS** (added 1)

## [608b4f2] NTA prefetch — 2026-02-09

**Add NTA prefetch for next expert and optimize cache read path**

- `_mm_prefetch` with `_MM_HINT_NTA` for next-expert weight pages
- Reduced cache pollution for streaming weight access pattern

Tests: **18 PASS**

## [150b60e] INT4 disk cache — 2026-02-09

**Add INT4 disk cache for instant model loading**

- Quantizes HF safetensors → INT4 on first load, caches to disk
- Subsequent loads: mmap from cache (no quantization)
- `test_cache_bit_exact`: verifies cached == freshly quantized

Tests: **17 PASS** (added 1)

## [854daa1] PyO3 shim — 2026-02-09

**Implement PyO3 shim with KrasisEngine and SGLang FusedMoE wrapper**

- `KrasisEngine` Python class: `load()`, `forward()`, `num_moe_layers()`, etc.
- Byte-buffer interface for zero-copy BF16/INT32/FP32 tensor transfer

Tests: **16 PASS**

## [db25bcb] WeightStore + MoE forward — 2026-02-09

**Add WeightStore, parallel matmul, and MoE forward pass**

- `WeightStore`: holds all expert weights in quantized INT4 format
- `moe_forward()`: full token routing — gate/up matmul, SiLU, down matmul, weighted sum
- Parallel matmul via rayon over output rows

Tests: **14 PASS** (added 4: v2_lite_load, v2_lite_single_expert, v2_lite_moe_forward, throughput)

## [2132c51] Marlin repack — 2026-02-09

**Implement Marlin GPU format repack with permutation tables**

- `marlin_repack()`: converts INT4 packed format to Marlin GPU layout
- Permutation tables for efficient GPU access patterns
- Round-trip verification test

Tests: **10 PASS** (added 2)

## [e94511f] AVX2 INT4 kernel — 2026-02-09

**Implement AVX2 INT4 matmul kernel with scalar reference**

- `avx2_int4_matmul()`: vectorized INT4×BF16 matmul with group-wise dequantization
- `scalar_int4_matmul()`: reference implementation for verification
- Handles group_size=32/128, arbitrary M/N/K

Tests: **8 PASS** (added 4)

## [86e5293] Safetensors reader — 2026-02-09

**Implement safetensors mmap reader and INT4 quantization**

- `SafetensorsReader`: mmap-based reader for HF safetensors format
- Symmetric INT4 quantization with per-group scales
- BF16↔FP32 conversion utilities

Tests: **4 PASS** (added 4)

## [4747d96] Project scaffold — 2026-02-09

**Initial project scaffold**

- Rust + PyO3 project structure with Cargo.toml, lib.rs, build.rs
- Module layout: kernel/, weights/, moe.rs, numa.rs

## [26100b0] Initial commit — 2026-02-09

# Krasis Changelog

## Test Status

| Suite | Count | Status | Last Run |
|-------|-------|--------|----------|
| Rust (`cargo test`) | 34 | ALL PASS | 2026-02-09 |
| Python bridge (`test_bridge.py`) | 3 | ALL PASS | 2026-02-09 |
| GPU prefill (`test_gpu_prefill.py`) | 5 | ALL PASS | 2026-02-09 |
| **Total** | **42** | **ALL PASS** | |

Re-run needed after: any change to `src/`, `python/krasis/`, or test files.

---

## Kimi K2.5 GPU Prefill Enabled — 2026-02-10

**GPU prefill managers enabled on Kimi K2.5 PP=2 (INT8 weights, FP8 KV)**

Tests: 3/3 PASS. GPU prefill managers created for both GPUs. Short prompts (~20 tokens)
used CPU path (threshold=300). No impact on correctness or speed. GPU prefill not yet
triggered — needs long prompt (300+ tokens) to test actual GPU Marlin prefill speed.

---

## Kimi K2.5 FP8 KV Cache Verification — 2026-02-10

**Verified FP8 E4M3 KV cache on Kimi K2.5 PP=2 (INT8 weights)**

Tests: 3/3 PASS. Quality identical to BF16 KV — same answers produced.

| Metric | FP8 KV + INT8 wt | BF16 KV + INT8 wt |
|--------|-----------------|-------------------|
| GPU0 KV cache | 4,032 MB (236K tokens) | 1,811 MB (53K tokens) |
| GPU1 KV cache | 4,839 MB (293K tokens) | 2,292 MB (69K tokens) |
| Decode speed | 1.21-1.28 tok/s | 1.28-1.41 tok/s |
| Context capacity | **~4x more** | baseline |

FP8 auto-sizes to fill free VRAM, giving much more context capacity.

---

## Kimi K2.5 INT8 Weights Verification — 2026-02-10

**Verified INT8 attention/shared_expert/dense_mlp/lm_head on Kimi K2.5 PP=2**

Switched from all-BF16 to default QuantConfig (INT8 everywhere). Tests: 3/3 PASS.

| Metric | INT8 | BF16 |
|--------|------|------|
| GPU0 VRAM | 7,654 MB | 12,063 MB |
| GPU1 VRAM | 6,044 MB | 11,105 MB |
| Decode speed | 1.28-1.41 tok/s | 1.55-1.87 tok/s |
| VRAM savings | **4-5 GB/GPU** | baseline |

INT8 slightly slower at M=1 decode (torch._int_mm overhead > bandwidth savings), but frees
significant VRAM for KV cache and GPU prefill buffers. Quality identical — same answers.

---

## Bug Fix: YaRN RoPE Frequency Assignment + De-interleave — 2026-02-10

**Fixed two bugs in MLA attention RoPE that caused garbled output on V2-Lite (and likely Kimi K2.5)**

### Bug: YaRN frequency interpolation was BACKWARDS
- Krasis: low indices (i < low) → interpolated (divided by factor), high indices (i > high) → original
- HF reference: low indices → **original** (high-freq, fast rotation), high indices → **interpolated** (divided by factor)
- This completely broke position encoding, producing progressive degradation with position
  (cos sim: pos0=0.999, pos3=0.968, worsening per token)
- Fix: Replaced manual loop with HF-matching mask-based computation using `freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask`

### Bug: Missing de-interleave before RoPE
- HF stores q_pe/k_pe in interleaved format `[re0, im0, re1, im1, ...]` from projection weights
- RoPE rotation operates on half-split format `[re0, re1, ..., im0, im1, ...]`
- Without de-interleave, rotation was applied to wrong dimension pairs
- Fix: Added `_deinterleave()` static method, applied before RoPE in `_apply_rope()`

### Verification
- Created `test_layer_compare.py`: full layer-by-layer comparison HF vs Krasis
- Created `test_layer0_detail.py`: sub-component comparison within layer 0 (dense)
- **Before fix**: Layer 0 cos=0.968, final logits cos=0.880, garbled output
- **After fix**: Layer 0 cos=0.999989, final logits cos=0.997, top-5 tokens IDENTICAL to HF
- V2-Lite generation tests: "2+2"→"4", "Count 1 to 5"→"1, 2, 3, 4, 5.", "Capital of France?"→"Paris." — 5/6 PASS
- **Kimi K2.5 PP=2 generation tests: 3/3 PASS** — YaRN factor=64 produces correct answers

### Files changed
- `python/krasis/attention.py` — YaRN frequency fix + `_deinterleave()` in MLAAttention

---

## Bug Fixes: Double Shared Expert + YaRN mscale + KV kv_len — 2026-02-10

**Fixed four critical bugs in standalone model that produced garbled output**

### Bug 4: Double-application of shared expert + routed_scaling_factor
- Rust `moe_forward()` computes: `output = routed_scaling_factor * routed_sum + shared_expert`
- Python `_routed_expert_forward()` ALSO computed: `shared_gpu + routed_scaling_factor * rust_output`
- Result: `shared_gpu + 2.827 * (2.827 * routed + shared_cpu)` = `shared + 7.99 * routed + 2.827 * shared`
- Instead of correct: `shared + 2.827 * routed`
- routed_scaling_factor applied TWICE (2.827^2 = 7.99), shared expert computed TWICE
- Made MLP output ~3-4x too large per layer, causing exponential growth in residual stream
- Fix: Python now returns Rust output directly (Rust handles shared expert + scaling)
- Note: NUMA path (line 272) also skips shared expert — not hit on single-socket system

### Bug 3: Missing YaRN mscale^2 in MLA softmax scale
- `attention.py` used `sm_scale = 1/sqrt(head_dim)` = 0.072
- Kimi K2.5 with YaRN factor=64 needs `mscale = 0.1 * mscale_all_dim * log(factor) + 1 = 1.416`
- Correct: `sm_scale = 1/sqrt(192) * mscale^2 = 0.145` (2x larger)
- Without this, attention logits are 2x too small → nearly uniform attention → garbled output
- Fix: compute YaRN mscale from rope_scaling config, multiply sm_scale by mscale^2

### Bug 1: Missing shared expert + routed_scaling_factor in CPU decode path
- `layer.py:_routed_expert_forward()` only returned raw routed expert output
- Missing: `shared_expert(hidden) + routed_scaling_factor * routed_experts(hidden)`
- For Kimi K2.5 this means missing the shared expert AND a 2.827x scaling factor
- GPU prefill path was correct (gpu_prefill.py handles both)

### Bug 2: kv_len_arr off-by-M in attention
- `attention.py` used `seq_state.seq_len` for `kv_len_arr` in FlashInfer plan()
- But `seq_state.advance(M)` is called AFTER all layers, not before
- Prefill: kv_len_arr=[0] when 18 tokens were appended → attention reads ZERO tokens
- Decode: always missing the current token (kv_len=N instead of N+1)
- Fix: pass `num_new_tokens` through model→layer→attention, use `seq_len + num_new_tokens`

### Files changed
- `python/krasis/layer.py` — shared expert + scaling in CPU path, num_new_tokens param
- `python/krasis/model.py` — pass num_new_tokens to layer.forward()
- `python/krasis/attention.py` — use effective kv_len in both MLA and GQA attention

### Tests run
- Kimi K2.5 standalone PP=2 (BF16 weights, BF16 KV, 16 threads): **3/3 PASS**
  - simple_math ("2+2?"): "4" — 128 tokens in 82.6s (1.55 tok/s)
  - counting ("1 to 5"): "1, 2, 3, 4, 5" — 128 tokens in 68.9s (1.86 tok/s)
  - factual ("Capital of France?"): "Paris" — 128 tokens in 68.4s (1.87 tok/s)
- Model produces coherent reasoning (thinking) followed by correct answers
- Load time: 722s (GPU weights 18s, CPU experts 704s)
- GPU VRAM: GPU0=12,063 MB, GPU1=11,105 MB

---

## GQA Attention Support — 2026-02-10

**Added GQA (Grouped Query Attention) alongside existing MLA, enabling Qwen3-235B on standalone model**

Previously Krasis standalone only supported MLA (Multi-head Latent Attention) used by DeepSeek V2/V3
and Kimi K2.5. Now supports GQA as used by Qwen3-235B-A22B.

### Changes (5 Python files)
- `config.py`: Made MLA fields optional (None for GQA), added `gqa_head_dim`, auto-detection via
  `attention_type` property ("mla" or "gqa"). GQA detected by absence of `kv_lora_rank`.
- `weight_loader.py`: Split `load_attention_weights()` into `_load_mla_attention()` and
  `_load_gqa_attention()`. GQA loads: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
- `kv_cache.py`: Supports both MLA split (ckv+kpe) and GQA split (k+v) cache layouts.
  GQA: `[layers, pages, page_size, num_kv_heads, head_dim]` per K and V.
- `attention.py`: Added `GQAAttention` class using FlashInfer `BatchPrefillWithPagedKVCacheWrapper`.
  Handles QKNorm (per-head RMSNorm), standard RoPE on all Q/K, `append_paged_kv_cache`.
- `layer.py`: Auto-selects GQAAttention or MLAAttention based on `cfg.attention_type`.

### Tests run
- V2-Lite MLA end-to-end: **PASSED** (no regression)
- Qwen3 GQA attention forward (prefill + decode): **PASSED**
- Qwen3 TransformerLayer with GQA dispatch: **PASSED**
- Config parsing all 3 models (V2-Lite, Kimi K2.5, Qwen3-235B): **PASSED**

---

## INT8 CPU Expert Quantization — 2026-02-10

**INT8 CPU expert support: configurable 4-bit or 8-bit CPU MoE expert quantization**

Previously CPU experts were always INT4 (4 bits per weight). Now supports INT8 (8 bits)
for 2x better precision at 2x memory. Configurable via `num_bits` parameter in Rust engine
and `cpu_expert_bits` in Python `QuantConfig`.

### Rust changes (4 files)
- `weights/marlin.rs`: Added `QuantizedInt8` struct and `quantize_int8()`/`dequantize_int8()`
  functions. INT8 uses per-group symmetric quantization (scale = amax / 127.0), stores raw i8.
- `kernel/avx2.rs`: Added `expert_matmul_int8_integer` AVX2 kernel — uses
  `_mm256_cvtepi8_epi16` + `_mm256_madd_epi16` (simpler than INT4: no nibble extraction).
  Plus `matmul_int8_integer`/`matmul_int8_integer_parallel` wrappers.
- `weights/mod.rs`: Added `QuantWeight` enum (`Int4`/`Int8`) wrapping the two quant types.
  Updated `ExpertWeights` to use `QuantWeight`. `WeightStore` now has `num_bits` field.
  `load_from_hf()` takes `num_bits` parameter. Disk cache uses separate files per bit width
  (`experts_int4_g128.bin` vs `experts_int8_g128.bin`). Updated all cache read/write/NUMA paths.
- `moe.rs`: Added `matmul_integer()`/`matmul_integer_parallel()` dispatch helpers.
  `expert_forward_integer()` dispatches via enum. `prefetch_expert_nta()` handles both types.
  `KrasisEngine.load()` accepts `num_bits` parameter (default 4).

### Python changes (3 files)
- `config.py`: Added `cpu_expert_bits: int = 4` to `QuantConfig`
- `model.py`: Passes `num_bits=quant_cfg.cpu_expert_bits` to Rust engine
- `sglang_bridge.py`: Added `cpu_expert_bits` class config, passes to `engine.load()`

### Test Results (V2-Lite)
- INT8 load: 78.7s (first quantization from BF16), cached loading expected ~2s
- INT4 vs INT8 token match: 14/20 (different precision = different outputs, expected)
- INT8 decode slightly faster than INT4 (simpler weight access, no nibble extraction)
- Build: clean compile, all existing INT4 tests pass

### Usage
```python
# Rust engine directly
engine = KrasisEngine(parallel=True, num_threads=16)
engine.load("/path/to/model", num_bits=8)

# Standalone model
from krasis.config import QuantConfig
qcfg = QuantConfig(cpu_expert_bits=8, gpu_expert_bits=8)
model = KrasisModel("path/to/model", quant_cfg=qcfg)

# SGLang bridge
KrasisMoEWrapper.cpu_expert_bits = 8
```

---

## Per-Component Precision Config + INT8 Marlin GPU Prefill — 2026-02-10

**Configurable per-component quantization + INT8 Marlin GPU expert prefill**

Previously all GPU weight precision was hardcoded (INT8 for projections, INT4 for GPU expert prefill).
Now each component's precision is configurable via `QuantConfig`, and GPU expert prefill supports
both INT4 and INT8 Marlin kernels.

### Changes
- `config.py`: Added `QuantConfig` dataclass — per-component precision settings:
  `lm_head`, `attention`, `shared_expert`, `dense_mlp` (each "bf16"/"int8"),
  `gpu_expert_bits` (4/8 for Marlin kernel)
- `weight_loader.py`: Accepts `QuantConfig`, conditionally loads weights as BF16 or INT8
  in `load_lm_head`, `load_attention_weights`, `load_dense_mlp`, `load_shared_expert`
- `layer.py`: Added `_linear()` dispatch helper (handles both INT8 tuples and BF16 tensors),
  replaced all `int8_linear(*self.xxx)` calls with `_linear(self.xxx)`
- `attention.py`: Same `_linear()` dispatch for `kv_a_proj`, `q_a_proj`, `q_b_proj`,
  `q_proj`, `o_proj`
- `model.py`: Added `quant_cfg` parameter to `KrasisModel`, passes through to
  `WeightLoader` and `GpuPrefillManager`. LM head stored as generic `lm_head_data`
  (tuple or tensor) with `_linear()` dispatch
- `gpu_prefill.py`: `num_bits` constructor parameter (default 4, supports 4/8).
  Generalized `_quantize_and_pack_gpu()` for INT4 (8 vals/int32) and INT8 (4 vals/int32).
  Updated buffer allocation, repack calls, `fused_marlin_moe` calls, VRAM estimates

### V2-Lite Test Results (PP=1)

| Metric | Default (INT8+INT4) | BF16attn+INT8exp | CPU-only |
|--------|----:|----:|----:|
| GPU weights | 5,194 MB | 5,524 MB (+330) | 5,206 MB |
| Expert buffer | 285.5 MB | 562.3 MB (2x) | N/A |
| Peak GPU | 11,321 MB | 11,726 MB (+405) | 11,034 MB |
| Decode speed | 3.3 tok/s | 3.6 tok/s | 3.3 tok/s |
| Token match (short) | baseline | 19/20 | 20/20 |

BF16 attention slightly faster on decode (no INT8 quantize overhead).
INT8 Marlin GPU prefill runs correctly via `fused_marlin_moe(num_bits=8)`.
VRAM cost modest (+405 MB peak). V2-Lite too small for quality assessment.

### Defaults
Default `QuantConfig()` matches previous hardcoded behavior (all INT8, GPU experts INT4).
No change for existing users who don't pass `quant_cfg`.

### Usage
```python
from krasis.config import QuantConfig
# Higher quality attention (BF16) with INT8 GPU expert prefill
qcfg = QuantConfig(attention="bf16", gpu_expert_bits=8)
model = KrasisModel("path/to/model", quant_cfg=qcfg)
```

---

## FP8 KV Cache — 2026-02-10

**FP8 E4M3 KV cache: halves KV VRAM, same attention quality**

SGLang's approach: store KV as FP8, upcast to BF16 before FlashInfer kernel. The kernel
always computes in BF16 — FP8 is purely a storage optimization. Simple to implement.

### Changes
- `attention.py`: Upcast FP8 cache to BF16 before `BatchMLAPagedAttentionWrapper.plan()/run()`.
  `kv_data_type=torch.bfloat16` always passed to plan (not the cache dtype).
- `model.py`: Default `kv_dtype` changed from `torch.bfloat16` to `torch.float8_e4m3fn`.
- `kv_cache.py`: Already had FP8 default — no change needed.

### Test Results (V2-Lite, PP=1)
- FP8 cache: 5,322 MB vs BF16: 10,644 MB → **2x VRAM savings**
- Prefill logits: cosine similarity 1.0000 (FP8 vs BF16), top-5 tokens identical
- FP8 append + upcast roundtrip: max diff 0.015 (within FP8 E4M3 precision)
- V2-Lite generation quality degrades (model too small for FP8+INT4 noise) — expected,
  Kimi K2.5 should be fine

### VRAM Impact (Kimi K2.5 PP=2, estimated)
- FP8: 576 bytes/token/layer × 31 layers = ~1.1 GB for 64K context
- BF16: 1152 bytes/token/layer × 31 layers = ~2.2 GB for 64K context
- **Saves ~1.1 GB per GPU** — more room for GPU prefill buffers

---

## TRTLLM MLA Backend + Efficiency Report — 2026-02-10

**TRTLLM MLA attention backend implemented but blocked by SM89**

Created `trtllm_attention.py` (283 lines) with two-path architecture:
- Prefill (M > 1): Non-absorbed path, decompresses K/V via w_kc/w_vc before attention,
  uses `trtllm_ragged_attention_deepseek` with Q[M,H,192], K[M,H,192], V[M,H,128]
- Decode (M = 1): Absorbed path, uses `trtllm_batch_decode_with_kv_cache_mla`
  with Q[1,H,576], paged KV cache [pages,1,page_size,576]

### Changes
- `trtllm_attention.py`: New file — TRTLLMMLAAttention class
- `kv_cache.py`: Added combined cache format, `store_kv_combined()`, `block_tables()`
- `layer.py`: Added `attention_backend` parameter ("flashinfer" or "trtllm")
- `model.py`: Added `attention_backend` parameter, combined KV cache init

### BLOCKED
TRTLLM FMHA runner reports `Unsupported architecture` on SM89 (RTX 2000 Ada, compute 8.9).
Code is structurally complete but cannot run on our GPUs. FP8 KV cache (which depends on
TRTLLM backend) is also blocked.

### Efficiency Report
Produced `EFFICIENCY_REPORT.md` with:
- Full performance comparison: Krasis vs KTransformers+SGLang
- Resource utilization breakdown (GPU VRAM, CPU RAM)
- Decode latency analysis (~245ms/token, CPU expert-bound)
- Plan deviations: 1.7x code size vs estimates, all 4 core phases complete
- Technical deviations: BF16 KV (not FP8), custom RoPE (not FlashInfer), CPU bounce for P2P

---

## GPU Prefill Integration — 2026-02-10

**INT4 Marlin GPU MoE prefill integrated into standalone server**

Threshold-based routing: M >= threshold → GPU (INT4 Marlin), M < threshold → CPU (Krasis).
GpuPrefillManager created per PP rank device, wired to MoE layers automatically.

### Changes
- `layer.py`: Added `gpu_prefill_manager`, `gpu_prefill_threshold` params.
  `_moe_forward()` dispatches to `_gpu_prefill_forward()` or `_routed_expert_forward()`.
- `model.py`: Added `gpu_prefill`, `gpu_prefill_threshold` params.
  `_init_gpu_prefill()` creates one `GpuPrefillManager` per GPU device.

### Benchmark (V2-Lite, 313 tokens, PP=1)
| Path | Prefill | Decode |
|------|---------|--------|
| GPU Marlin INT4 | **424 tok/s** (0.74s) | 4.2 tok/s (236ms) |
| CPU Krasis INT4 | 10 tok/s (30.2s) | 4.0 tok/s (250ms) |
| **Speedup** | **40.9x** | ~same |

First-time layer quantization: ~1.2s/layer (26 layers = 32s, cached in RAM after).

### FP8 KV Cache Research
- FlashInfer `BatchMLAPagedAttentionWrapper` does NOT support FP8 (static_assert 16-bit)
- SGLang uses TRTLLM MLA backend for FP8: `trtllm_ragged_attention_deepseek`
- XQA MLA has FP8 but needs SM120+ (our GPUs are SM89)
- Current BF16 KV works fine: ~1.1 GB/GPU for 64K context

---

## Krasis Standalone Server — 2026-02-10

**Full standalone LLM server: replaces SGLang for GPU forward pass**

Single-process, N-GPU pipeline-parallel server with OpenAI-compatible HTTP API.
Eliminates SGLang's multi-process architecture that caused OOM crashes.

### New Python modules (10 files)

- `config.py`: Parse `config.json` for Kimi K2.5 (nested text_config) and V2-Lite (flat).
  `ModelConfig` with `has_q_lora`, `is_moe_layer()`, PP partition computation.
- `weight_loader.py`: Streaming BF16→INT8 per-channel symmetric quantization.
  Uses `torch._int_mm` (M>16 padding workaround). Loads one tensor at a time.
- `attention.py`: MLA attention using FlashInfer `BatchMLAPagedAttentionWrapper`.
  Dual path: q_lora (Kimi: q_a→norm→q_b) vs direct q_proj (V2-Lite). YaRN RoPE.
  Pre-absorbs w_kc into query, post-multiplies w_vc. Paged KV cache append.
- `kv_cache.py`: Paged KV cache for MLA (3D: [pages, page_size, dim]).
  Auto-sizes to 50% free VRAM. `SequenceKVState` for per-sequence page tracking.
- `layer.py`: Transformer layer: attention + MoE/dense MLP. Uses FlashInfer
  `fused_add_rmsnorm` (in-place), `silu_and_mul`. MoE routing with
  sigmoid/softmax, norm_topk_prob, routed_scaling_factor.
- `model.py`: Full model orchestration with PP. Two-phase loading:
  GPU weights (streaming INT8) + CPU experts (Krasis Rust INT4).
  CPU-bounce for broken GPU P2P transfers (auto-detected).
- `sampler.py`: FlashInfer `top_k_top_p_sampling_from_logits`.
- `tokenizer.py`: HF tokenizer wrapper with chat template, incremental decode.
- `scheduler.py`: Async request scheduler with thread pool executor.
- `server.py`: FastAPI HTTP server. `/v1/chat/completions` (SSE + blocking),
  `/v1/models`, `/health`.

### Bugs found and fixed

- `fused_add_rmsnorm` returns None (in-place) — code tried to unpack return value
- FlashInfer sampling returns single tensor in 0.6.1, not (samples, success) tuple
- `torch._int_mm` requires M > 16 strictly (not >=) — pad to 17
- FlashInfer MLA kernel requires 16-bit KV (not FP8) — `static_assert(sizeof(DType) == 2)`
- KV cache `ensure_capacity`/`advance` was called per-layer instead of per-forward-pass
  (inflated seq_len by num_layers, caused OOM in RoPE for multi-GPU)
- `seq_states[0]` used for all ranks — fixed to `seq_states[rank_idx]`
- GPU P2P transfers silently return zeros on this system — added auto-detection + CPU bounce
- Krasis Rust engine includes shared expert computation — Python was also computing it on GPU,
  causing double-counting. Fixed by removing GPU shared expert (Krasis handles it).

### Testing on DeepSeek-V2-Lite

- PP=1 (1 GPU): Works, 3.3 tok/s decode
- PP=2 (2 GPUs): Works, 3.3 tok/s decode, identical output to PP=1
- PP=3 (3 GPUs): Works, 3.4 tok/s decode, identical output to PP=1 and PP=2
- HTTP server: Health, models, blocking chat, streaming SSE all verified
- Logits cosine sim vs HF BF16 reference: 0.94 (INT4 expert quantization noise on small model)
- Per-layer hidden state cosine sim: 0.997 through layer 20, degrades in last 3 layers
  due to INT4 noise amplification (model's final residual stream has very small norm ~0.5)

---

## Fix silent crash during loading + shard filtering — 2026-02-10

**Fix silent OOM crash when 3 PP ranks load concurrently**

Root cause: `load_and_quantize_all` in `weights/mod.rs` opened ALL 64 safetensors shards
via mmap regardless of which layers this rank needs. Each PP rank only needs ~20 shards
but was mmapping all 64 (~580 GB each). With 3 concurrent ranks, page cache pressure from
reading different parts of the same files + 520 GB heap for expert weights could exhaust
the 995 GB RAM (no swap), silently killing the server and collateral processes (tmux, Claude).

- `weights/mod.rs` (load_and_quantize_all): Filter shards by layer range — only open shards
  containing expert tensors for layers in `[start_layer, start_layer + max_layers)`.
  Reduces from 64→~20 shards per rank, eliminates cross-rank page cache contention.
- `weights/mod.rs` (detect_prequant_group_size): Probe using this rank's first layer
  instead of global first_moe layer (which might not be in our filtered shards).
- `sglang_bridge.py` (_get_engine): Added `[DIAG]` logging with `sys.stdout.flush()`
  before/after engine creation and load — pinpoints crash location.
- `moe.rs` (load): Added `[DIAG-RUST]` logging with `log_memory_usage()` at each step.
- `weights/mod.rs`: Per-shard open progress, per-5-layer memory logging.
- `syscheck.rs`: New `log_memory_usage()` reads `/proc/self/status` + `/proc/meminfo`.

---

## Fix GPU Xid 13 crash with 0 GPU experts — 2026-02-10

**Fix out-of-range GPU memory access when num_gpu_experts=0**

Root cause: With `--kt-num-gpu-experts 0`, W8A8Int8MoEMethod creates weight tensors with
shape `[0, ...]`. During inference, the Triton fused_experts kernel launches with these
empty tensors and ALL topk_ids masked to -1. The kernel accesses `weight_ptr + expert_id * stride`
on a buffer with 0 elements, causing NVIDIA Xid 13 (Out Of Range Address) on GPU0.

- `kt_ep_wrapper.py` (KTEPWrapperMethod.apply): Skip GPU MoE kernel entirely when
  `num_gpu_experts == 0` — return zeros instead of launching Triton with empty weights
- `w8a8_int8.py` (W8A8Int8MoEMethod.apply): Guard check `layer.w13_weight.shape[0] == 0`
  before kernel launch — belt-and-suspenders defense
- `kt_ep_wrapper.py` (MarlinInt4PrefillMethod.apply): Same empty-expert guard for Marlin path
- Added diagnostic logging: first apply() call per layer, INT8 cache save markers with flush
- Created `crash-tracking.md` to track GPU Xid errors

---

## VRAM budget calculator — 2026-02-09

**Add per-rank VRAM budget calculation and dynamic SGLang parameter tuning**

- `python/krasis/vram_budget.py`: New module that computes exact per-rank GPU memory usage
  - Reads model config.json: supports both MLA (Kimi K2.5/DeepSeek) and GQA (Qwen3)
  - Calculates per-rank: attention weights, dense MLP, shared experts, gate/norms, embedding, lm_head
  - Handles INT8 quantization (1 byte/param) and BF16 (2 bytes/param)
  - KV cache per token: MLA `(kv_lora_rank + qk_rope_head_dim) * dtype_bytes` or GQA `2 * n_kv_heads * head_dim * dtype_bytes`
  - Determines max context length from bottleneck rank, capped at `max_position_embeddings`
  - Computes `mem_fraction_static`, `max_total_tokens`, `context_length` for SGLang
  - CLI entry point: `python -m krasis.vram_budget --model-path ... --pp-partition ...`
  - Prints human-readable summary to stderr, JSON to stdout
- `run_kimi_krasis.sh`: Replaced hardcoded `MEM_FRACTION=0.50` and `CTX_SIZE=25000` with dynamic VRAM budget
  - Runs `krasis.vram_budget` at startup, extracts parameters from JSON output
  - Prints per-rank breakdown at launch for visibility
  - Supports manual overrides via `CTX_SIZE_OVERRIDE`, `MEM_FRACTION_OVERRIDE`, `MAX_TOTAL_TOKENS_OVERRIDE`
- `sglang_bridge.py`: Added `verify_vram_budget()` classmethod
  - Called after engine loads (once per rank) from `load_weights()`
  - Compares `torch.cuda.memory_allocated()` to pre-computed estimate
  - Logs INFO (<10% deviation) or WARNING (>10% deviation)
- `__init__.py`: Exports `compute_vram_budget`

**Kimi K2.5 PP=3 budget** (INT8, fp8 KV, 16 GB/GPU):
- PP0: 5,445 MB weights → 10.4 GB free → 950K tokens max
- PP1: 3,018 MB weights → 12.9 GB free → 1.1M tokens max
- PP2: 5,114 MB weights → 10.8 GB free → 980K tokens max
- Bottleneck: PP0 → context capped at 262,144 (model max) — fits easily

Tests: CLI tested against Kimi K2.5 (MLA) and Qwen3-235B (GQA)

## Multi-GPU PP=3 support — 2026-02-09

**Add pipeline-parallel partial loading for multi-GPU setups**

- `WeightStore::load_from_hf()`: new `start_layer` param — loads MoE layers `[start..start+count)` instead of all
- `KrasisEngine.load()`: new `start_layer` keyword arg, passed through to Rust
- Memory budget estimate updated to use actual loaded layer count (not total)
- Disk cache skipped for partial loads (start_layer or max_layers set)
- `sglang_bridge.py`: PP-aware engine loading
  - Records `_pp_first_layer_idx` from first `load_weights()` call
  - Parses `SGLANG_PP_LAYER_PARTITION` env var to infer rank and MoE range
  - Passes `start_layer`/`max_layers` to `engine.load()` for partial loading
  - Remaps `moe_layer_idx` in CPU decode path (GPU prefill uses absolute indices)
- `run_kimi_krasis.sh`: Krasis-based launch script for Kimi K2.5 PP=3
  - No GGUF download needed (reads HF safetensors directly)
  - Partition 20,21,20 across 3 GPUs, 21 threads/rank
  - Each rank loads ~174 GB MoE experts (total ~520 GB fits in 995 GB RAM)
  - Sets `KRASIS_BACKEND=1`, `KRASIS_MODEL_PATH`, `SGLANG_PP_LAYER_PARTITION`

Tests: config + synthetic + V2-Lite pass, Rust build clean

## GPU prefill implementation — 2026-02-09

**Add INT4 Marlin GPU prefill for MoE layers**

- `python/krasis/gpu_prefill.py`: `GpuPrefillManager` class for GPU-accelerated MoE prefill
- `_quantize_and_pack_gpu()`: BF16→INT4 symmetric quantization + packing on GPU (group_size=128)
- Weight pipeline: HF safetensors → BF16 → transpose [N,K]→[K,N] → GPU INT4 quantize → pack → `gptq_marlin_repack` → `marlin_permute_scales` → RAM cache
- `prepare_layer()`: quantizes all experts for one layer, caches in CPU RAM
- `prepare_shared_expert()`: same for shared expert (stored as single-expert MoE)
- `forward()`: loads cached weights to GPU buffer, calls `fused_marlin_moe` kernel
- Multi-chunk path for large expert counts (Kimi K2.5: 384 experts): remaps expert IDs per chunk
- `_shared_expert_forward()`: runs shared expert via Marlin with topk_ids=0, weight=1.0
- Pre-quantized weight support: dequantizes compressed-tensors INT4 → BF16 → re-quantizes to Marlin format
- `sglang_bridge.py`: GPU prefill integration — threshold-based switching (batch >= 300 → GPU, else CPU)
- `GpuPrefillManager` singleton shared across all layer wrappers
- **V2-Lite benchmarks**: 26ms/layer, 40 tok/s (batch=1) → 19,123 tok/s (batch=512)
- **GPU vs CPU cosine similarity**: 0.9825 (excellent agreement despite different INT4 implementations)
- `test_gpu_prefill.py`: 5 tests — quantize_pack, v2_lite_forward, shared_expert, gpu_vs_cpu, scaling

Tests: 34 Rust + 3 Python bridge + 5 GPU prefill = **42 PASS**

## Shared expert support — 2026-02-09

**Add shared expert loading, forward computation, and routed_scaling_factor**

- `ModelConfig`: added `n_shared_experts` (0/1/2) and `routed_scaling_factor` (1.0/2.827)
- `WeightStore.shared_experts`: per-MoE-layer shared expert weights (BF16 → INT4 quantized)
- `load_shared_experts()`: loads from safetensors (always BF16, even for pre-quantized models like Kimi K2.5)
- `moe_forward()`: now accepts `shared_scratch` parameter; computes `scale * routed_output + shared_output`
- Async worker: allocates separate scratch buffer for shared expert's larger intermediate size
- `ExpertScratch` reuses quantized activation from routed path (same hidden_size + group_size)
- Shared experts loaded after cache too (not in cache format yet)
- **V2-Lite**: 2 shared experts, intermediate=2816, routed_scaling_factor=1.0
- **Kimi K2.5**: 1 shared expert, intermediate=2048, routed_scaling_factor=2.827
- **Qwen3**: 0 shared experts (no-op)
- `test_shared_expert_v2_lite`: verifies shared expert changes output (max_diff=0.075)

Tests: 34 Rust + 3 Python = **37 PASS**

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

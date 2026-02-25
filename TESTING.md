# Krasis Testing & Running Guide

## Prerequisites

- Python 3.10+ virtualenv at `krasis/.venv`
- Rust toolchain (maturin builds the PyO3 extension)
- Models stored in `~/.krasis/models/<ModelName>/` (safetensors + config.json)
- 3x NVIDIA RTX 2000 Ada (16 GB each), AMD EPYC 7742 (AVX2, 64 cores, 995 GB RAM)

Build the Rust extension into the venv:

```
cd krasis
maturin develop --release
```

## Running Krasis

### Interactive Launcher (TUI)

```
python -m krasis.launcher
```

Arrow-key driven menu to select model, configure quantization, GPU count, and launch
the server. Saves config to `testconfigs/<model>.conf` for reuse.

### Non-Interactive (from saved config)

```
python -m krasis.launcher --non-interactive
```

Reads the last saved config and launches without the TUI.

### Direct Server Launch

```
python -m krasis.server \
  --model-path ~/.krasis/models/Qwen3-Coder-Next \
  --pp-partition 48 \
  --gpu-expert-bits 4 \
  --cpu-expert-bits 4 \
  --kv-dtype fp8_e4m3 \
  --host 0.0.0.0 --port 8012
```

Serves an OpenAI-compatible API at `/v1/chat/completions` (streaming and blocking).

### Running a Benchmark (no server)

```
./run_benchmark.sh --model-path ~/.krasis/models/Qwen3-Coder-Next --pp-partition 48
```

Or directly:

```
python -m krasis.server --benchmark --model-path ~/.krasis/models/Qwen3-Coder-Next
```

Runs standardized prefill (20K/35K/50K tokens) and decode (64 tokens x 3 runs)
benchmarks, then exits. Output is logged and archived to `benchmarks/`.

### Benchmark Suite (multi-model, multi-config)

```
python -m krasis.suite                               # default config
python -m krasis.suite --config benchmarks/qcn_1gpu_int4.toml
```

Reads a TOML file defining model x config combinations, runs each as a subprocess,
and writes a summary markdown table to `benchmarks/suite_logs/`.

Suite config format (see `benchmarks/benchmark_suite.toml`):

```toml
[[config]]
num_gpus = 1
gpu_expert_bits = 4
cpu_expert_bits = 4

[[model]]
name = "Qwen3-Coder-Next"
```

## Test Configs

Saved configs live in `testconfigs/`. **Do not modify these files** -- they are
reference configs for reproducible runs.

- `qcn-4-4.conf` -- Qwen3-Coder-Next, 1 GPU, INT4/INT4
- `v2lite-4-4.conf` -- DeepSeek-V2-Lite, 1 GPU, INT4/INT4

## Test Files

All tests are in `tests/`. Run individual tests with:

```
python tests/<test_file>.py
```

### Model Correctness Tests

These load a real model and verify output is coherent:

| File | What it tests |
|------|--------------|
| `test_v2lite_sanity.py` | V2-Lite basic generation (MLA model) |
| `test_v2lite_thorough.py` | V2-Lite extended generation checks |
| `test_v2lite_dual_format.py` | Dual GPU/CPU format correctness |
| `test_v2lite_gpu_prefill.py` | V2-Lite GPU prefill at various lengths |
| `test_v2lite_10k.py` | V2-Lite with 10K token prompt |
| `test_qwen3_next_generate.py` | QCN hybrid model generation (linear attn + GQA) |
| `test_kimi_k25.py` | Kimi K2.5 generation tests |
| `test_pp2_v2lite.py` | V2-Lite with pipeline parallelism (2 GPUs) |
| `test_pp2_qcn.py` | QCN with pipeline parallelism |
| `test_pp2_qwen235b.py` | Qwen3-235B with pipeline parallelism |

### Component Tests

| File | What it tests |
|------|--------------|
| `test_pyo3.py` | Rust-Python binding works |
| `test_bridge.py` | Rust MoE engine bridge |
| `test_rust_decode.py` | Full Rust decode loop |
| `test_rust_vs_python.py` | Rust vs Python decode output match |
| `test_moe_sanity.py` | MoE forward pass correctness |
| `test_gpu_prefill.py` | GPU prefill kernel correctness |
| `test_fp8_kv.py` | FP8 KV cache precision |
| `test_quant_config.py` | Quantization config parsing |
| `test_la_graph.py` | Linear attention CUDA graph |
| `test_la_inplace.py` | Linear attention in-place ops |
| `test_parallel_prefill.py` | Multi-GPU parallel prefill |
| `test_gpu_decode.py` | GPU decode path (M=1 Marlin) |
| `test_attn_verify.py` | Attention output verification |
| `test_linear_attn_compare.py` | Linear attention vs reference |
| `test_gqa_compare.py` | GQA attention vs reference |

### Benchmarks (in tests/)

| File | What it measures |
|------|-----------------|
| `bench_engine_isolated.py` | Raw Rust MoE throughput (no model load) |
| `bench_decode_quick.py` | Quick decode speed check |
| `bench_8k_decode.py` | Decode with 8K context |
| `bench_combined.py` | Combined prefill + decode timing |
| `bench_model.py` | Full model benchmark |
| `bench_prefill_only.py` | Prefill-only throughput |
| `bench_prefill_timed.py` | Prefill with wall-clock timing |
| `bench_prefill_verify.py` | Prefill correctness + speed |
| `bench_prefill_10k.py` | Prefill at 10K tokens |
| `bench_prefill_order.py` | Prefill ordering effects |
| `bench_hot_cached_static.py` | HCS expert cache hit rates |
| `token_scaling_bench.py` | Decode scaling vs token count |

### Network Tests

| File | What it tests |
|------|--------------|
| `test_network.py` | HTTP API validation against running server |

Requires a running server. Usage:

```
python tests/test_network.py --port 8012
python tests/test_network.py --port 8012 --large    # include large-prompt tests
python tests/test_network.py --port 8012 --quick    # known-answer only
```

### Profiling

| File | Purpose |
|------|---------|
| `profile_decode.py` | Full decode profiling with instrumentation |
| `profile_decode_nopin.py` | Decode profiling without expert pinning |
| `test_decode_timing.py` | Decode per-component timing breakdown |
| `test_pure_cpu_timing.py` | Pure CPU decode timing |
| `test_qwen3_next_decode_timing.py` | QCN decode timing |

## Benchmarks Directory

`benchmarks/` contains:

- `bench_decode_harness.py` -- Synthetic decode benchmark (fake weights, real memory
  access patterns). Measures raw engine throughput without model loading.
- `bench_decode.py` -- Server decode benchmark
- `bench.py` -- Old MoE throughput benchmark
- `kt_benchmark.py` -- KTransformers comparison benchmark
- `BENCHMARKS.md` -- Summary table of all benchmark runs with links to full logs
- `*.log` -- Full benchmark output logs
- `suite_logs/` -- Benchmark suite output
- `*.toml` -- Suite config files
- Prompt files bundled in package (`python/krasis/prompts/`): `prefill_prompt_{1-6}`, `decode_prompt_{1-6}`
  - Files 1-3 used for warmup, files 4-6 used for timed runs

## Utility Shell Scripts

| File | Purpose | Requires sudo |
|------|---------|:---:|
| `run_benchmark.sh` | Run benchmark and exit cleanly | No |
| `gpu_cleanup.sh` | Kill zombie GPU processes, reclaim VRAM | Yes |
| `gpu_reset.sh` | Full NVIDIA driver reload (stops/restarts Xorg) | Yes |
| `setup_pcie.sh` | GPU persistence mode, disable ASPM, max power | Yes |
| `fix-oomd.sh` | Raise systemd-oomd kill threshold to 95% | Yes |
| `dump_trace.sh` | Dump Python + native stack trace of running server | No |

## Analysis Scripts

In the root `scripts/` directory:

| File | Purpose |
|------|---------|
| `analyze_expert_rank.py` | Analyze expert weight distributions |
| `analyze_heatmap.py` | Analyze expert activation heatmaps |
| `generate_heatmap.py` | Generate heatmap JSON from model runs |
| `run_heatmap_prompts.py` | Run prompts to collect heatmap data |
| `clear_page_cache.py` | Evict OS page cache (for cold benchmarks) |
| `krasis_monitor.py` | Live monitoring of running Krasis instance |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KRASIS_DECODE_TIMING` | `0` | Enable per-component decode timing |
| `KRASIS_PREFILL_TIMING` | `0` | Enable prefill timing |
| `KRASIS_CPU_DECODE_TIMING` | `0` | Enable CPU decode step timing |
| `KRASIS_TIMING_INTERVAL` | `20` | Steps between timing reports |
| `KRASIS_BENCH_RUNS` | `3` | Number of decode runs in benchmark |
| `KRASIS_DEBUG_DECODE` | `0` | Debug logging for decode path |
| `KRASIS_DEBUG_SYNC` | `0` | Synchronous CUDA error checking |
| `KRASIS_DIAG` | `0` | Diagnostic logging (per-MoE-layer) |
| `KRASIS_NO_PIPELINE` | `0` | Disable pipeline parallelism |
| `KRASIS_FUSED_LINEAR_ATTN` | `1` | Use fused linear attention kernel |
| `KRASIS_HOME` | `~/.krasis` | Model and cache storage root |
| `KRASIS_LAYER_TIMING` | `0` | Per-layer timing in prefill |
| `KRASIS_PREFILL_THRESHOLD` | `500` | Token count above which GPU prefill is used |

## Typical Workflows

### Quick correctness check (V2-Lite, fastest model)

```
python tests/test_v2lite_sanity.py
```

### Measure decode engine speed (no model load)

```
python benchmarks/bench_decode_harness.py --steps 100 --timing
```

### Full benchmark run (QCN, 1 GPU, INT4)

```
python -m krasis.suite --config benchmarks/qcn_1gpu_int4.toml
```

### Start server and run network tests

```
# Terminal 1:
python -m krasis.launcher

# Terminal 2 (after server is up):
python tests/test_network.py --port 8012
```

### Recover from GPU crash

```
sudo ./gpu_cleanup.sh       # try soft cleanup first
sudo ./gpu_reset.sh          # if cleanup doesn't work, full driver reload
```

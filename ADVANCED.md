# Advanced Configuration

## Install Pre-release

```bash
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- prerelease
```

This installs the latest pre-release build. Normal `install.sh` (without the `prerelease` flag) always installs the latest stable release.

## Running Krasis

Krasis has two entry points:

- `krasis` — the installed command (use for production, release testing)
- `./dev` — the development entry point (handles conda, auto-rebuild, GPU cleanup)

Never run Python scripts directly. Always use one of these commands.

BF16 validation policy:

- BF16-heavy configs are validation-only. Use them to prove correctness or isolate quantization from logic bugs.
- Production runs must use the normal Rust serving path with quantized configs.
- `gpu_expert_bits = 16` is not a production mode.

### Dev Commands

| Command | Description |
|---------|-------------|
| `./dev build` | Rebuild Rust extension (maturin develop --release) |
| `./dev run <config> [flags]` | Launch server from a test config |
| `./dev benchmark <config>` | Run standard benchmark and exit |
| `./dev release-test <model>` | Run full release test (4 configs, produces markdown report) |
| `./dev test <config>` | Short model test (benchmark + network tests) |
| `./dev test <config> --thorough` | Thorough test (+ stress + large prompts) |
| `./dev network <port> [--large] [--quick]` | Run network tests against a running server |
| `./dev perplexity <config>` | Run perplexity eval (WikiText-2) and exit |
| `./dev awq-calibrate <config>` | Run AWQ attention calibration (produces template) |
| `./dev kill` | Kill all krasis/GPU processes and reset |

Add `--timing` to `run` or `benchmark` for a per-layer decode timing breakdown. This adds ~30-50% overhead so do not use it for speed benchmarks — only for profiling.

## Config Files

The preferred way to run Krasis is with a config file:

```bash
krasis --config path/to/config.conf
./dev run qcn            # resolves to testconfigs/qcn.conf
./dev benchmark qcn      # same
```

Config files use `KEY=VALUE` format. CLI flags override config file values.

## Server Flags

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | — | Config file (KEY=VALUE format), CLI flags override |
| `--model-path PATH` | — | HuggingFace model directory (safetensors + config.json) |
| `--num-gpus N` | all | Number of GPUs to use |
| `--selected-gpus IDX` | all | Comma-separated GPU indices (e.g. `0,2`) |
| `--pp-partition STR` | auto | Layer partition across GPUs (e.g. `24,24`) |
| `--host ADDR` | 0.0.0.0 | Server bind address |
| `--port PORT` | 8012 | Server port |

### Quantization

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu-expert-bits` | 4 | GPU Marlin expert bits: `4` or `8` |
| `--cpu-expert-bits` | 4 | CPU decode expert bits: `4` or `8` |
| `--attention-quant` | bf16 direct, hqq8 launcher | Attention weight precision: `hqq8` quality-first high-fidelity option, or explicit `bf16`, `awq`, `hqq4` |
| `--shared-expert-quant` | int8 | Shared expert quant: `int8` or `bf16` |
| `--dense-mlp-quant` | int8 | Dense MLP quant: `int8` or `bf16` |
| `--lm-head-quant` | int8 | LM head quant: `int8` or `bf16` |
| `--kv-dtype` | k6v6 | KV cache format: `k6v6` Quality, `k4v4` Ultra Compact, or `bf16` Full Precision |

Legacy `int4`/`int8` values for `--attention-quant` are auto-migrated to `awq`.

When BF16 is selected for experts or major components, treat that run as validation-only rather than production.

### Memory & Caching

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-cache-mb N` | 1000 | KV cache size in MB |
| `--hcs` / `--no-hcs` | on | Hot Cache Strategy for expert pinning |
| `--multi-gpu-hcs` | off | Pin HCS experts across all GPUs |
| `--vram-safety-margin N` | 600 | Reserved VRAM in MB below which warnings fire |
| `--stream-attention` | off | Stream attention weights from CPU (for very large models) |
| `--force-load` | — | Override RAM safety checks and load anyway |
| `--force-rebuild-cache` | — | Delete existing expert caches and rebuild from safetensors |
| `--build-cache` | — | Build expert caches (if missing) and exit without starting server |
| `--heatmap-path PATH` | — | Path to expert_heatmap.json for HCS init |

### Prefill & Decode

| Flag | Default | Description |
|------|---------|-------------|
| `--layer-group-size N` | 2 | MoE layers to load per group during prefill |
| `--gpu-prefill-threshold N` | 300 | Minimum tokens to use GPU prefill |
| `--krasis-threads N` | 40 | CPU threads for expert computation |
| `--gguf-path PATH` | — | GGUF file for CPU experts (instead of native cache) |

### Speculative Decoding

| Flag | Default | Description |
|------|---------|-------------|
| `--draft-model PATH` | — | Draft model for speculative decoding (e.g. `~/.krasis/models/Qwen3-0.6B`) |
| `--draft-k N` | 3 | Tokens to draft per speculative round |
| `--draft-context N` | 512 | Context window for draft model warmup |

### Inference Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature F` | 0.6 | Sampling temperature |
| `--enable-thinking` / `--no-enable-thinking` | on | Enable thinking/reasoning mode |
| `--session-enabled` / `--no-session-enabled` | off | Enable Session messenger bridge |

### Benchmarking & Testing

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | — | Run benchmark before launching server |
| `--benchmark-only` | — | Run benchmark and exit (no server) |
| `--timing` | — | Enable per-layer decode timing instrumentation |
| `--stress-test` | — | Run stress test (diverse prompts) and exit |
| `--perplexity` | — | Run perplexity evaluation and exit |
| `--note TEXT` | — | Description note written to log file header |

## Per-Component Quantization Summary

Krasis lets you quantize each component independently. The defaults are a good starting point — increase precision if you need better quality, decrease if you need to fit in less VRAM/RAM.

| Component | Options | Default |
|-----------|---------|---------|
| GPU experts | INT4, INT8 | INT4 |
| CPU experts | INT4, INT8 | INT4 |
| Attention | AWQ, HQQ4, BF16 | BF16 |
| Shared expert | INT8, BF16 | INT8 |
| Dense MLP | INT8, BF16 | INT8 |
| LM head | INT8, BF16 | INT8 |
| KV cache | FP8, BF16 | FP8 |

Embeddings, norms, and routing gates are always kept at BF16.

AWQ attention uses calibrated per-tensor quantization (run `./dev awq-calibrate <config>` to generate the template). `hqq4` is the native HQQ INT4 backend: artifacts live under the normal model cache tree and the runtime restores staged prefill/decode descriptors from that cache. Remaining HQQ work is parity closure against AWQ operational details such as cached-stage generation strategy, slot-reuse policy, and wider product validation. BF16 is full precision with no calibration needed.

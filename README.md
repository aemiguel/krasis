# Krasis

Python-orchestrated Rust LLM runtime. Runs 200B+ parameter models on commodity hardware with full GPU prefill and decode.

You can [contact me here](https://forms.gle/ue4nvyvNNHtUZ7MQ7) but for bugs, difficulties, suggestions or requests for support for other models please report an issue instead.

If you want to easily monitor Krasis during runs, [check out ktop](https://github.com/brontoguana/ktop).

![Krasis Server](krasis_server_3.png)

## Krasis runs LLMs fast on consumer grade hardware (single GPU)

Krasis can run large language models that are much too large to fit in a consumer GPU (multi-hundred gigabyte model with 80- 500+ billion parameters) on consumer or accessible server hardware that doesn't require the huge cost to host the entire model in VRAM. 

## Latest News

* [Krasis subreddit is now available](https://www.reddit.com/r/krasis/) for anyone that wants to follow Krasis news and updates more closely.
* Multi-GPU pre-release now available via the latest release.  This won't improve prefill speed but could improve decode speed on multi GPU systems with similar GPUs.
* Fix for higher system RAM usage than expected.

## Benchmarks

### 1x RTX 5090 32GB (PCIE 4.0)

- 1x Epyc 7742, 8-channel DDR4
- 1x RTX 5090 on PCIE 4.0 (bottlenecked to 32GB/sec since the 5090 supports PCIE 5.0 at 64GB/sec)
- **Most models (Qwen-3.5-35B at Q4 being the exception) are unable to entirely fit in GPU VRAM.** Krasis **streams through VRAM** as necessary using algorithms optimised for the prefill stage and then the decode stage.
- Q4 model (BF16 attention, AWQ also supported for more limited VRAM cards)

| Model                 | Params | BF16 Size / INT4 Size | Prefill (pp) | Decode (tg)   |
| :-------------------- | :----: | :-------------------: | ------------ | ------------- |
| **Qwen3.5-35B-A3B**   |  35B   |     67 GB / 16GB      | 4475 tok/sec | 109.1 tok/sec |
| **Qwen3-Coder-Next**  |  80B   |     159 GB / 38GB     | 3560 tok/sec | 70.3 tok/sec  |
| **Qwen3.5-122B-A10B** |  122B  |     234 GB / 56GB     | 2897 tok/sec | 27.7 tok/sec  |
| **Qwen3-235B-A22B**   |  235B  |    438 GB / 110GB     | 2124 tok/sec | 9.3 tok/sec   |

### 1x RTX 5080 16GB (PCIE 4.0)

| Model                | Params | BF16 Size / INT4 Size | Prefill (pp) | Decode (tg)  |
| -------------------- | :----: | :-------------------: | ------------ | ------------ |
| **Qwen3-Coder-Next** |  80B   |     159 GB / 38GB     | 1801 tok/sec | 26.8 tok/sec |

## Krasis tradeoffs

In order to achieve these speeds, Krasis has a few requirements.

- Krasis currently only works with **NVidia GPUs**
- Krasis must be given the **BF16 safetensors model** downloaded from [HuggingFace](https://huggingface.co/)
- **Krasis uses comparable system RAM to other runtimes**.  Krasis will auto-quantize to INT4 or INT8 from the safetensors model provided and build a cache on disk, then load that into system RAM.
- Krasis **will take some time to load on the first run** as it is doing a lot of pre-run work to optimise everything for runtime, much of this is cached for later runs though so subsequent runs will be quicker.
- Krasis optimises models and caches them in <home folder>/.krasis, these can be large so you may need disk space for the original model BF16 model plus the quantised model size.
- **Krasis is optimised to run models at Q4 and Q8**, which are generally very good trade-offs vs either running the full precision weights or heavily quantised models much lowered quality.

## Perplexity (Quantization Quality)

Measured with INT4 GPU + INT4 CPU experts (Q4), BF16 attention, INT8 shared/MLP/lm_head, FP8 KV cache. Sliding window (2048 tokens, stride 1024).

| Model | Dataset | Tokens | PPL | BPC |
|-------|---------|:------:|:---:|:---:|
| **Qwen3-Coder-Next** | WikiText-2 | 299K | 7.23 | 2.85 |
| **Qwen3-Coder-Next** | C4 validation | 1M | 12.52 | 3.65 |
| **DeepSeek V2-Lite** | WikiText-2 | 307K | 6.03 | 2.59 |
| **DeepSeek V2-Lite** | C4 validation | 500K | 9.22 | 3.20 |
| **Qwen3.5-35B-A3B** | WikiText-2 | 297K | 6.41 | 2.68 |

## Running Krasis - Quick Start

### Requirements

- **Linux** (Ubuntu 24.04+, or WSL2 on Windows)
- **Python 3.10+**
- **NVIDIA GPU** with CUDA drivers installed
- **System RAM**: roughly 2x the quantised model size
    - e.g. if BF16 is 100GB, Q8 is 50GB, Q4 is 25GB
        - to run at Q8 required 2x50GB = 100GB system RAM
        - to run at Q4 requires 2x25GB = 50GB system RAM
- **Disk space**: roughly the BF16 model size plus 2x the quantised model size (see system ram)

### 1. Install Krasis

```bash
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash
```

This creates a Python environment at `~/.krasis/venv`, installs Krasis, symlinks the commands into `~/.local/bin`, and adds that directory to your PATH. No sudo required. Works immediately in the current terminal session.

### 2. Install CUDA dependencies

```bash
krasis-setup
```

This installs the CUDA toolkit (if needed), PyTorch, and FlashInfer. May need sudo for the CUDA toolkit. Only required once.

### 3. Download a model

```bash
pip install huggingface-hub   # if you don't have it

huggingface-cli download deepseek-ai/DeepSeek-V2-Lite \
    --local-dir ~/.krasis/models/DeepSeek-V2-Lite
```

Krasis needs the BF16 safetensors model from HuggingFace. Put it under `~/.krasis/models/`. Some other models you can download:

```bash
# Qwen3-Coder-Next (80B params, 148 GB, fastest on consumer hardware)
huggingface-cli download Qwen/Qwen3-Coder-Next \
    --local-dir ~/.krasis/models/Qwen3-Coder-Next

# Qwen3-235B (235B params, 438 GB, needs ~500 GB RAM)
huggingface-cli download Qwen/Qwen3-235B-A22B \
    --local-dir ~/.krasis/models/Qwen3-235B-A22B
```

### 4. Run

```bash
krasis
```

The launcher walks you through model selection and configuration via a TUI. First run takes longer as Krasis builds optimised weight caches (these are saved to disk for subsequent runs).

### Upgrade / Uninstall

```bash
# Upgrade
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash

# Uninstall (keeps model files)
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- --uninstall
```

### WSL2 (Windows)

Krasis works on WSL2. By default WSL only uses 50% of your system RAM, which is usually not enough for large models. Create or edit `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
memory=120GB
```

Adjust the value to leave ~8 GB for Windows. Restart WSL from PowerShell with `wsl --shutdown`, then follow the install steps above inside WSL.

### Install from source

Requires a Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`):

```bash
git clone https://github.com/brontoguana/krasis.git
cd krasis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
krasis-setup
krasis
```

## Usage

### Interactive Launcher

```bash
krasis
```

The launcher walks you through a TUI with four screens:

1. **Model selection** — scans `~/.krasis/models/` for safetensors models, shows architecture, layer count, expert count, and estimated RAM
2. **CPU expert source** — build INT4 or INT8 from the native model, or select an existing GGUF file
3. **GPU selection** — multi-select your GPUs (Space to toggle, Enter to confirm)
4. **Configuration editor** — tune all quantization and runtime options with a live VRAM budget display showing per-GPU memory usage and estimated context length

All settings are saved to `~/.krasis/config` and reloaded on subsequent launches.

On the final screen you can choose to launch immediately or run a benchmark first.

### Non-Interactive Launch

```bash
# Use saved config from last TUI session
krasis --non-interactive

# Override specific settings
krasis --non-interactive --model-path /path/to/model --num-gpus 2 --benchmark
```

### Benchmark Suite

Run all model × config combinations automatically from a single config file. Edit `benchmarks/benchmark_suite.toml` to define which models and hardware configurations to test:

```toml
[[config]]
num_gpus = 1
gpu_expert_bits = 4
cpu_expert_bits = 4

[[config]]
num_gpus = 2
gpu_expert_bits = 4
cpu_expert_bits = 4

[[model]]
name = "DeepSeek-V2-Lite"

[[model]]
name = "Qwen3-235B-A22B"
gguf_name = "Qwen3-235B-A22B-GGUF"   # searched in ~/.krasis/models/ subdirs
```

Model `name` is the directory name under `~/.krasis/models/`. Use `gguf_name` to pair a native model with a GGUF for CPU experts (filename searched in models dir), or `gguf_path` for an absolute path. Config fields include `num_gpus`, `gpu_expert_bits`, `cpu_expert_bits`, `attention_quant`, `kv_dtype`, and more — see the config file comments for the full list.

Run the suite:

```bash
krasis --benchmark-suite                           # uses benchmarks/benchmark_suite.toml
krasis --benchmark-suite /path/to/custom.toml      # custom config
```

Each combination runs as an isolated subprocess. Per-combo logs are saved to `benchmarks/suite_logs/` and a markdown summary table is generated at the end.

For launcher flags, per-component quantization options, and direct server usage, see [ADVANCED.md](ADVANCED.md).

### Chat Client

```bash
krasis-chat                          # auto-discovers running servers
krasis-chat --port 8012              # connect to specific port
krasis-chat --url http://host:8012   # connect to remote server
krasis-chat --temperature 0.3        # override sampling temperature
```

The chat client auto-discovers running Krasis servers via `~/.krasis/servers/`. Commands: `/new` (clear history), `/system PROMPT` (change system prompt), `/exit`.

### API

The server exposes an OpenAI-compatible API at `http://localhost:8012/v1/chat/completions` with SSE streaming, compatible with Cursor, OpenCode, and any OpenAI SDK client.

Additional endpoints:
- `GET /health` — server status
- `GET /v1/models` — list loaded models
- `POST /v1/timing` — toggle instrumentation at runtime

## License

SSPL-1.0

Krasis is free to use, modify and distribute.  

If you want to support the project or offer Krasis as part of a commercial product or a hosted/managed service, please [get in touch](https://forms.gle/ue4nvyvNNHtUZ7MQ7).


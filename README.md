# Krasis

Rust + PyO3 MoE runtime for large mixture-of-experts LLMs. Runs 350B+ parameter models on commodity hardware with full GPU prefill and efficient CPU decode.

You can [contact me here](https://forms.gle/ue4nvyvNNHtUZ7MQ7) but for bugs, difficulties, suggestions or requests for support for other models please report an issue instead.


## Krasis runs MoE LLMs fast on consumer level hardware

Krasis can run MoE language models that are much too large to fit in a consumer GPU (multi-hundred gigabyte modesl with 100 - 500+ billion parameters) on consumer or accessible server hardware you can actually buy without a second mortgage and your own personal power station. 

**Most importantly, it runs these models at usable speeds.**

## Qwen3-Coder-Next Results

Qwen3-Coder-Next (80B params, 148 GB BF16) is a model which will clearly not fit inside a single consumer GPU unless very heavily quantised to Q2 or less.

At Q4 Qwen3-Coder-Next is around 37GB, too much to fit on even the largest consumer GPUs. 

Krasis is able to run Qwen3-Coder-Next (Q4 quantised) with **one 16GB GPU** at the following speeds:

- 5900X, 3200 DDR4, 1x 5080 16GB (PCIE4.0x16) : **3595 tok/sec prefill, 14.7 tok/sec decode**
- Epyc 7742, 2666 DDR4, 1x RTX Ada 2000 16GB (PCIE4.0x8) : **1060 tok/sec prefill, 18.9 tok/sec decode**

Krasis can likely run QCN at speed with even more VRAM limited GPUs than these (further testing is planned).

## Why LLM's run slow, and how Krasis runs them fast

LLM model operation consist of two key steps:

1) Prefill (handling potentially large amounts of input coming into the model)
2) Decode (handling the generation of text after processing the input data)

These are essentially the **LLM reading (prefill) and writing (decode)**.

Prefill is best handled by the GPUs (large amounts of very parallel matrix multiplication), but on typical LLM runtimes its not possible to do more than offload a little of the large model onto the GPU.

The GPU runs part of the model fast, then the CPU runs most of the model at a snail's pace.

The result is if you enter a simple chat prompt it may respond in a reasonable time, but **if you hand it a file to read or try to work with it in an IDE, you wait minutes for it to even start generating text.**

Krasis employs a different approach that utilises the GPU and system RAM more heavily which results in much faster prefill times.  In practice this means the model will generate text at a similar speed (faster in some cases due to other optimisations) **but you wait much less time for an answer, and the model can read files much more quickly.**

## Krasis tradeoffs
In order to achieve these speeds, Krasis has a few requirements.

- **Krasis uses more system RAM than other runtimes**, you may need 2x the model weights worth of system ram (so to run a model which is 30GB at Q4 you will need 60GB of system ram), but this is generally **far more obtainable and cost effective than the equivalent VRAM**.
- Krasis must be given the **BF16 safetensors model** downloaded from [HuggingFace](https://huggingface.co/)
- Krasis can build everything it needs from this model or if you prefer you can give it **both** the BF16 safetensors and a second GGUF model with an optimised quantisation you prefer (e.g. unsloth Q4_K models)
- Krasis currently only works with **NVidia GPUs**
- Krasis **will take some time to load on the first run** as it is doing a lot of pre-run work to optimise everything for runtime, much of this is cached for later runs though so subsequent runs will be quicker.
- Krasis optimises models and caches them in <home folder>/.krasis, these can be large so you may need disk space for the original model BF16 model plus **2x the quantized model** (for QCN at Q4 this would be 149GB + 38GB + 38GB = ~225GB).
- **Krasis is optimised to run models at Q4 and Q8**, which are generally very good tradeoffs vs either running the full precision weights or heavily quantised models much lowered quality.

## Supported Models

| Model | Params | BF16 Size | Experts | Attention |
|-------|:------:|:---------:|---------|-----------|
| **Qwen3-Coder-Next** | 80B | 148 GB | 512 routed, top-10 | Hybrid (36 linear + 12 GQA) |
| **DeepSeek V2-Lite** | 16B | 29 GB | 64 + 2 shared, top-6 | MLA |

## Models in progress

| Model | Params | BF16 Size | Experts | Attention |
|-------|:------:|:---------:|---------|-----------|
| **Qwen3-235B-A22B** | 235B | 438 GB | 128 routed, top-8 | GQA |
| **GLM-4.7** | 358B | 667 GB | 160 + 1 shared, top-8 | GQA (partial RoPE, bias) |

## Perplexity (Quantization Quality)

Measured with INT4 GPU + INT4 CPU experts (Q4), BF16 attention, INT8 shared/MLP/lm_head, FP8 KV cache. Sliding window (2048 tokens, stride 1024).

| Model | Dataset | Tokens | PPL | BPC | Throughput |
|-------|---------|:------:|:---:|:---:|:----------:|
| **Qwen3-Coder-Next** | WikiText-2 | 299K | 7.23 | 2.85 | 128 tok/s |
| **Qwen3-Coder-Next** | C4 validation | 1M | - | - | - tok/s |
| **DeepSeek V2-Lite** | WikiText-2 | 307K | 6.03 | 2.59 | 593 tok/s |
| **DeepSeek V2-Lite** | C4 validation | 500K | 9.22 | 3.20 | 573 tok/s |

## Benchmark: AMD 5900X + 1x RTX 5080 16GB

- **Hardware:** AMD 5900X (16 cores), DDR4-3200 2-channel, 1x NVIDIA RTX 5080 16 GB, PCIe 4.0 x16.
- **Config:** BF16 attention, FP8 KV cache, INT8 shared/MLP/lm_head, LGS=2, 16 CPU threads.

Benchmark uses 10K–50K token prompts (prefill) and 64-token generation runs (decode). Prefill speed is best of 20K/35K/50K. Decode is average of 3 runs with different prompts.

| Model | Expert Quant | Prefill (tok/s) | TTFT | Decode (tok/s) | ms/tok |
|-------|:------------:|:---------------:|:----------:|:--------------:|:------:|
| **Qwen3-Coder-Next** | INT4 GPU + INT4 CPU | 3,595 | 9.7s | 14.7 | 68 |
| **DeepSeek V2-Lite** | INT4 GPU + INT4 CPU | - | - | - | - |

## Benchmark: EPYC 7742 + 1x RTX 2000 Ada 16 GB

- **Hardware:** AMD EPYC 7742 (64 cores, 4 NUMA nodes), DDR4-2666 8-channel, 1x NVIDIA RTX 2000 Ada 16 GB, PCIe 4.0 x8.
- **Config:** BF16 attention, FP8 KV cache, INT8 shared/MLP/lm_head, LGS=2, 40 CPU threads, NUMA-aware thread pinning (required NPS4) + interleaved allocation.

Benchmark uses 10K–50K token prompts (prefill) and 64-token generation runs (decode). Prefill speed is best of 20K/35K/50K. Decode is average of 3 runs with different prompts.

| Model | Expert Quant | Prefill (tok/s) | TTFT | Decode (tok/s) | ms/tok |
|-------|:------------:|:---------------:|:----------:|:--------------:|:------:|
| **Qwen3-Coder-Next** | INT4 GPU + INT4 CPU | 1,060 | 18.9s | 15.81 | 63.6 |
| **Qwen3-Coder-Next** | INT8 GPU + INT8 CPU | 873 | 40.1s | 12.41 | 80.6 |
| **DeepSeek V2-Lite** | INT4 GPU + INT4 CPU | 1,477 | 13.6s | 20.18 | 49.7 |
| **DeepSeek V2-Lite** | INT8 GPU + INT8 CPU | 1,317 | 15.2s | 17.84 | 56.2 |

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

Krasis is free to use, modify and distribute.  If you want to offer Krasis as part of a commercial product or a hosted/managed service, please [get in touch](https://forms.gle/ue4nvyvNNHtUZ7MQ7).


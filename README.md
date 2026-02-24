# Krasis

Rust + PyO3 MoE runtime for large mixture-of-experts LLMs. Runs 350B+ parameter models on commodity hardware with full GPU prefill and efficient CPU decode.

You can [contact me here](https://forms.gle/ue4nvyvNNHtUZ7MQ7) but please don't ask for help getting Krasis working.  If a model doesn't work or a particular hardware config then you can try to narrow it down and then report an issue.


## Krasis runs MoE LLMs fast on consumer level hardware

Krasis can run MoE language models that are much too large to fit in a consumer GPU (multi-hundred gigabyte modesl with 100 - 500+ billion parameters) on consumer or accessible server hardware you can actually buy without a second mortgage and your own personal power station. 

**Crucially, it runs these models at a speed that is usable.**

## Qwen3-Coder-Next / 856 tok/s prefill / 10.5 tok/s decode##

For example, running Qwen3-Coder-Next (80B params, 146GB BF16) on a single-cpu Epyc server (7742) with 2x Ada 2000 16GB, Krasis achieves **856 tokens/sec prefill** and **10.5 tokens/sec decode**

## How LLMs work

LLM model operation consist of two key steps:

1) Prefill (handling potentially large amounts of input coming into the model)
2) Decode (handling the generation of text after processing the input data)

These are essentially the **LLM reading (prefill) and writing (decode)**.

Prefill is best handled by the GPUs (large amounts of very parallel matrix multiplication, but on typical LLM runtimes its not possible to do more than offload a little of the large model onto the GPU.

The result is that you enter a simple chat prompt and it responds in a reasonable time, but **if you hand it a file to read or try to work with it in an IDE, you wait minutes for it to even start generating text.**

Krasis employs a different approach that utilises the GPU and system RAM more heavily which results in much faster prefill times.  In practice this means the model will generate text at a similar speed (faster in some cases due to other optimisations) **but you wait much less time for an answer, and the model can read files much more quickly.**

## Krasis tradeoffs
In order to achieve these speeds, Krasis has a few requirements.

- **Krasis uses more system RAM than other runtimes**, you may need 2x the model weights worth of system ram (so to run a 100GB model you may need 200GB of system ram), but this is almost always **far more achievable than the equivalent VRAM**.
- Krasis must be given the *BF16 safetensors model** downloaded from (HuggingFace)[https://huggingface.co/]
- Krasis can build everything it needs from this model or if you prefer you can give it a second GGUF model (in addition to the BF16 safetensors model) which takes advantage of more advanced quantisation (e.g. unsloth Q4_K models)
- Krasis currently only works with **NVidia GPUs**
- Krasis **may take some time on the first run** as it is doing a lot of pre-run work to optimise everything, major parts of this are cached for later runs though so they are generally much shorter startup times.
- Krasis optimises models and caches them in .krasis, these can be large so you may need the original model **x3 space** or if you provide a GGUF in addition to the BF16 you may need **4x the space**.

## Known Supported Models and Benchmark Speeds

Speeds reported in the following models are benchmarked on the following hardware:

- Epyc 7742
- DDR4 2666 RAM (8x channels)
- 2x RTX Ada 2000

| Model | Params | BF16 Size | Experts | Attention | Prefill | Decode |
|-------|:------:|:---------:|---------|-----------|:-------:|:------:|
| **Qwen3-Coder-Next** | 80B | 148 GB | 512 routed, top-10 | Hybrid (36 linear + 12 GQA) | 812 tok/s | 10.5 tok/s |
| **Qwen3-235B-A22B** | 235B | 438 GB | 128 routed, top-8 | GQA | 198 tok/s | 1.65 tok/s |
| **DeepSeek V2-Lite** | 16B | 29 GB | 64 + 2 shared, top-6 | MLA | 2,400 tok/s | 5.8 tok/s |
| **GLM-4.7** | 358B | 667 GB | 160 + 1 shared, top-8 | GQA (partial RoPE, bias) | untested | untested |


## Perplexity (Quantization Quality)

Measured with INT4 GPU + INT4 CPU experts, BF16 attention, INT8 shared/MLP/lm_head, FP8 KV cache. Sliding window (2048 tokens, stride 1024), GPU Marlin prefill.

| Model | Dataset | Tokens | PPL | BPC |
|-------|---------|:------:|:---:|:---:|
| **Qwen3-Coder-Next** | WikiText-2 | 299K | 10.64 | 3.41 |
| **Qwen3-Coder-Next** | C4 validation | 500K | 12.44 | 3.64 |

## Quick Start

### Install

```bash
# Update APT
sudo apt update   # Ubuntu/Debian

# Install pipx if you don't have it
sudo apt install pipx   # Ubuntu/Debian
# or: pip install --user pipx

# Install Krasis
pipx install krasis
pipx ensurepath        # adds ~/.local/bin to PATH (restart terminal or source ~/.bashrc)

# Run setup — installs CUDA toolkit, PyTorch, FlashInfer, ninja
# (will prompt for your password when installing system packages)
krasis-setup
```

### Download a model

```bash
# Install huggingface-cli if you don't have it
pip install huggingface-hub

# Download a model into ~/.krasis/models/
huggingface-cli download Qwen/Qwen3-Coder-Next \
    --local-dir ~/.krasis/models/Qwen3-Coder-Next
```

### Run

```bash
krasis
```

That's it. The launcher walks you through model selection and configuration. First run takes longer as Krasis builds optimised weight caches.

### WSL (Windows Subsystem for Linux)

Krasis works on WSL2. By default WSL only uses 50% of your system RAM, which is usually not enough for large models. Create or edit `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
memory=120GB
```

Adjust the value to leave ~8 GB for Windows. Then restart WSL from PowerShell:

```powershell
wsl --shutdown
```

Then follow the install steps above inside WSL.

### Alternative: pip in a venv

```bash
python3 -m venv ~/.krasis-env && source ~/.krasis-env/bin/activate
pip install krasis
krasis-setup
```

### Alternative: from source

```bash
git clone https://github.com/brontoguana/krasis.git
cd krasis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
krasis-setup
./krasis
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

AGPL-3.0

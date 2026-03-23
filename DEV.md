# Krasis Dev Environment

Dev machine setup for building and testing Krasis from source.

## Hardware

- CPU: AMD EPYC 7742 (64 cores, AVX2, NO AVX-512/AMX)
- RAM: 995 GB DDR4
- GPUs: 1x NVIDIA RTX 5090 (32 GB, PCIe 4.0 x16) + 1x NVIDIA RTX 2000 Ada (16 GB, PCIe 4.0 x8)
- OS: Ubuntu, Linux 6.17

## Dev Environment

The dev environment is a conda env originally named `ktransformers` (historical),
symlinked to `krasis` for clarity:

```
/home/main/miniconda3/envs/ktransformers/   (real)
/home/main/miniconda3/envs/krasis/          (symlink)
```

Python: `/home/main/miniconda3/envs/krasis/bin/python` (3.11.14)

### Key packages and pinned versions

These versions are known to work together. Don't upgrade without testing.

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.9.1+cu128 | CUDA 12.8 |
| sgl-kernel | 0.0.1+ | Provides Marlin GEMM + fused MoE kernels |
| transformers | 4.57.1 | |
| safetensors | 0.7.0 | |
| maturin | 1.12.4 | At ~/.local/bin/maturin |

### Installing from scratch

If you need to recreate the env:

```bash
conda create -n krasis python=3.11 -y
conda activate krasis

# PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# GPU kernels (these are the [gpu] optional deps from pyproject.toml)
pip install sgl-kernel

# Krasis dev install
pip install maturin
cd ~/Documents/Claude/krasis
maturin develop --release
```

## The `dev` Script

Single entry point for all dev workflows. Handles Python path, auto-rebuild,
config parsing. Run `./dev help` for full usage.

### Common commands

```bash
# Rebuild Rust extension after changing src/*.rs
./dev build

# Launch QCN server
./dev run qcn

# Launch QCN with benchmark (runs engine + network benchmark, then serves)
./dev run qcn --benchmark

# Short model test: benchmark + network multi-prompt validation
./dev test qcn

# Thorough model test: adds stress test + large prompt tests
./dev test qcn --thorough

# Run network tests against an already-running server
./dev network 8012
./dev network 8012 --large

# Run any Python command with the dev env
./dev python -m krasis.launcher
./dev python tests/test_network.py --port 8012
```

### Auto-rebuild

The `run`, `test`, and `network` commands check if any file in `src/` or
`Cargo.toml` is newer than the compiled `.so`. If so, `maturin develop --release`
runs automatically before launching. No more stale Rust code.

### Config shortcuts

| Shortcut | Config file |
|----------|------------|
| qcn | testconfigs/qcn-4-4.conf |
| v2lite | testconfigs/v2lite-4-4.conf |
| deepseek-vl | testconfigs/deepseek-vl2-4-4.conf |

Or pass a path to any .conf file directly.

## File Locations

| What | Where |
|------|-------|
| Krasis repo | ~/Documents/Claude/krasis |
| Krasis internal docs | ~/Documents/Claude/krasis-internal |
| Models | ~/.krasis/models/ |
| Conda env | /home/main/miniconda3/envs/krasis/ |
| maturin | ~/.local/bin/maturin |
| Compiled .so | python/krasis/krasis.cpython-311-x86_64-linux-gnu.so |
| Test configs | testconfigs/*.conf |
| Benchmark logs | benchmarks/*.log |

## Troubleshooting

### "No module named sgl_kernel"

The [gpu] optional deps aren't installed. Fix:
```bash
./dev python -m pip install sgl-kernel
```

### Stale Rust code

If you see unexpected behavior after Rust changes, force a rebuild:
```bash
./dev build
```

### GPU in error state

```bash
sudo ./gpu_cleanup.sh       # soft cleanup
sudo ./gpu_reset.sh          # full driver reload (stops Xorg)
```

### Wrong Python being used

The `dev` script hardcodes the path. If the conda env moves, update the
`PYTHON` variable at the top of `./dev`.

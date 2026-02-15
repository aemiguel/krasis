# CI Plan: Publish Krasis to PyPI

## Goal

Let users install Krasis with `pip install krasis` instead of cloning the repo and installing Rust.

## What We Need

### 1. PyPI Account
- Create a free account at https://pypi.org
- Register the package name "krasis" (first-come-first-served, currently available)

### 2. GitHub Actions Workflow
- A config file at `.github/workflows/release.yml`
- When we create a GitHub release (e.g. tag `v0.1.0`), GitHub automatically:
  1. Spins up a Linux VM
  2. Compiles the Rust code inside a manylinux Docker container (ensures compatibility across distros)
  3. Packages the compiled binary + Python code into a wheel (.whl file)
  4. Uploads the wheel to PyPI
- Uses `maturin-action` — a pre-made GitHub Action that handles the Rust→wheel build

### 3. PyPI API Token
- Generate an API token on pypi.org
- Store it as a GitHub repository secret (`PYPI_API_TOKEN`)
- The workflow uses this to authenticate when uploading

## What Changes for Users

**Before (current)**:
```bash
sudo apt install python3.12-venv
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/brontoguana/krasis.git
cd krasis
./krasis  # builds from source on first run
```

**After**:
```bash
pip install krasis torch
# Download a model
huggingface-cli download Qwen/Qwen3-Coder-Next --local-dir models/Qwen3-Coder-Next
# Run
krasis
```

## Workflow File (approximate)

```yaml
# .github/workflows/release.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
          manylinux: auto
      - uses: pypa/gh-action-pypi-publish@v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## Things to Figure Out

- **Platform support**: Do we build for x86_64 Linux only, or also aarch64 (ARM)?
  Current users are all x86_64 Linux so start there.
- **CUDA dependency**: The wheel doesn't include CUDA libraries — users still need
  PyTorch with CUDA installed separately. Document this clearly.
- **Version numbering**: Decide on a versioning scheme (e.g. 0.1.0, 0.2.0, etc.)
  Every time Rust code changes, we need to publish a new version.
- **flashinfer**: Still needs separate install (`pip install flashinfer`) — it's not
  a hard dependency, just recommended for GPU attention performance.
- **Entry point**: Register `krasis` as a console script in pyproject.toml so users
  can type `krasis` directly instead of `./krasis` or `python -m krasis.launcher`.

## When to Implement

Once the codebase is stable enough that we're not changing the Rust code every day.
Publishing a new wheel for every small change is annoying — better to batch changes
into releases.

#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KRASIS_DEV_SCRIPT:-}" ]]; then
    echo "ERROR: This script must be run via ./dev capture-preflight, not directly." >&2
    echo "  Usage: ./dev capture-preflight [--bootstrap]" >&2
    exit 1
fi

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
BOLD="\033[1m"
NC="\033[0m"

info()  { echo -e "${CYAN}${BOLD}=>${NC} $*"; }
ok()    { echo -e "${GREEN}${BOLD}OK${NC} $*"; }
warn()  { echo -e "${YELLOW}${BOLD}!!${NC} $*"; }
err()   { echo -e "${RED}${BOLD}ERROR${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KRASIS_DEV_PYTHON:-python3}"
PIP_BIN="${KRASIS_DEV_PIP:-}"
MATURIN_BIN="${KRASIS_DEV_MATURIN:-maturin}"
CAPTURE_ENV_DIR="${KRASIS_REFERENCE_CAPTURE_VENV:-${HOME}/.krasis/reference-capture-venv}"
CAPTURE_PYTHON="${CAPTURE_ENV_DIR}/bin/python"
READY_JSON="${KRASIS_REFERENCE_CAPTURE_READY_JSON:-${HOME}/.krasis/capture-host-ready.json}"
READY_STAMP="${KRASIS_REFERENCE_CAPTURE_READY_STAMP:-${HOME}/.krasis/capture-host-ready.stamp}"
BOOTSTRAP=0

usage() {
    cat <<'EOF'
Usage: ./dev capture-preflight [options]

Verify that a host is genuinely ready for HF reference capture before any
download queue or detached capture starts.

Options:
  --bootstrap     Create/repair the isolated capture venv first via reference-prep --deps-only
  -h, --help      Show this help

Success writes:
  ~/.krasis/capture-host-ready.json
  ~/.krasis/capture-host-ready.stamp

Failure removes those readiness artifacts so queue logic can fail closed.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap)
            BOOTSTRAP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            ;;
    esac
done

mkdir -p "$(dirname "$READY_JSON")"

if [[ "$BOOTSTRAP" -eq 1 ]]; then
    info "Bootstrapping isolated capture environment"
    KRASIS_DEV_SCRIPT=1 \
    KRASIS_DEV_PYTHON="$PYTHON_BIN" \
    KRASIS_DEV_PIP="$PIP_BIN" \
    KRASIS_REFERENCE_CAPTURE_VENV="$CAPTURE_ENV_DIR" \
    "$SCRIPT_DIR/scripts/reference_capture_prep.sh" --deps-only
fi

[[ -x "$PYTHON_BIN" ]] || err "Repo Python not found at $PYTHON_BIN"
[[ -x "$MATURIN_BIN" ]] || err "maturin not found at $MATURIN_BIN"
[[ -x "$CAPTURE_PYTHON" ]] || err "Capture Python not found at $CAPTURE_PYTHON. Run ./dev capture-preflight --bootstrap"

tmp_json="$(mktemp)"
set +e
PYTHONNOUSERSITE=1 PYTHONPATH="$SCRIPT_DIR/python:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" SCRIPT_DIR="$SCRIPT_DIR" PYTHON_BIN="$PYTHON_BIN" MATURIN_BIN="$MATURIN_BIN" CAPTURE_PYTHON="$CAPTURE_PYTHON" READY_JSON="$READY_JSON" "$CAPTURE_PYTHON" - <<'PY' >"$tmp_json"
import importlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

repo = pathlib.Path(os.environ["SCRIPT_DIR"]).resolve()
python_bin = pathlib.Path(os.environ["PYTHON_BIN"])
maturin_bin = pathlib.Path(os.environ["MATURIN_BIN"])
capture_python = pathlib.Path(os.environ["CAPTURE_PYTHON"])
ready_json = pathlib.Path(os.environ["READY_JSON"])

result = {
    "kind": "capture_host_ready",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "repo_root": str(repo),
    "capture_env_dir": str(capture_python.parent.parent),
    "capture_python": str(capture_python),
    "repo_python": str(python_bin),
    "maturin": str(maturin_bin),
    "checks": [],
}

def add_check(name, ok, **details):
    result["checks"].append({"name": name, "ok": ok, **details})
    return ok

def inside_repo(path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(repo)
        return True
    except ValueError:
        return False

def command_output(cmd, cwd=None):
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return completed.returncode, stdout, stderr

repo_bindings_ok = True
try:
    pkg = importlib.import_module("krasis")
    contract = importlib.import_module("tests.reference_contract")
    pkg_path = pathlib.Path(pkg.__file__).resolve()
    contract_path = pathlib.Path(contract.__file__).resolve()
    capture_path = (repo / "tests" / "generate_reference.py").resolve()
    repo_bindings_ok = (
        capture_path.is_file()
        and inside_repo(pkg_path)
        and inside_repo(contract_path)
        and inside_repo(capture_path)
    )
    add_check(
        "repo_bindings",
        repo_bindings_ok,
        krasis=str(pkg_path),
        reference_contract=str(contract_path),
        generate_reference=str(capture_path),
    )
except Exception as exc:
    add_check("repo_bindings", False, error=repr(exc))

capture_packages = {}
for package in (
    "torch",
    "transformers",
    "huggingface_hub",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "triton",
):
    try:
        capture_packages[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        capture_packages[package] = None
result["capture_packages"] = capture_packages

required_versions_ok = True
for package, expected in {
    "transformers": "4.57.1",
    "huggingface_hub": "0.36.2",
    "accelerate": "1.13.0",
    "safetensors": "0.7.0",
}.items():
    installed = capture_packages.get(package)
    if installed != expected:
        required_versions_ok = False
protobuf_version = capture_packages.get("protobuf")
if protobuf_version is None:
    required_versions_ok = False
else:
    try:
        if int(protobuf_version.split(".", 1)[0]) >= 7:
            required_versions_ok = False
    except ValueError:
        required_versions_ok = False
if capture_packages.get("sentencepiece") is None:
    required_versions_ok = False
add_check("capture_python_deps", required_versions_ok, packages=capture_packages)

hf_cli_command = None
for candidate in (
    [str(capture_python.parent / "hf"), "--help"],
    [str(capture_python.parent / "huggingface-cli"), "--help"],
    [str(capture_python), "-m", "huggingface_hub.commands.huggingface_cli", "--help"],
    [str(capture_python), "-m", "hf", "--help"],
):
    if not pathlib.Path(candidate[0]).exists():
        continue
    rc, stdout, stderr = command_output(candidate)
    if rc == 0:
        hf_cli_command = candidate[:-1]
        add_check(
            "hf_cli",
            True,
            resolved_command=candidate[:-1],
            probe=" ".join(candidate),
        )
        break
if hf_cli_command is None:
    add_check("hf_cli", False, error="No working Hugging Face CLI found in capture env")
result["hf_cli"] = hf_cli_command

torch_ok = False
try:
    import torch

    device_count = torch.cuda.device_count()
    is_available = torch.cuda.is_available()
    devices = []
    for index in range(device_count):
        capability = torch.cuda.get_device_capability(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": f"sm_{capability[0]}{capability[1]}",
                "free_mb": free_bytes // (1024 * 1024),
                "total_mb": total_bytes // (1024 * 1024),
            }
        )
    torch_ok = device_count > 0 and is_available
    add_check(
        "torch_cuda",
        torch_ok,
        device_count=device_count,
        is_available=is_available,
        devices=devices,
        ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
    )
    result["cuda_devices"] = devices
except Exception as exc:
    add_check("torch_cuda", False, error=repr(exc), ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""))

triton_ok = False
try:
    import triton
    from triton.compiler import ASTSource

    version_text = getattr(triton, "__version__", "unknown")
    arch_params = list(inspect.signature(ASTSource).parameters.keys())
    numeric_version = tuple(int(part) for part in version_text.split(".")[:2])
    max_supported_arch = 120 if numeric_version >= (3, 5) else 90
    visible_archs = []
    for device in result.get("cuda_devices", []):
        try:
            visible_archs.append(int(device["capability"].split("_", 1)[1]))
        except Exception:
            pass
    compile_targets = [arch for arch in (80, 89, 90, 120) if arch <= max_supported_arch]
    triton_ok = (
        ("constexprs" in arch_params or "constants" in arch_params)
        and len(compile_targets) > 0
    )
    add_check(
        "triton",
        triton_ok,
        version=version_text,
        astsource_params=arch_params,
        visible_archs=[f"sm_{arch}" for arch in visible_archs],
        compile_targets=[f"sm_{arch}" for arch in compile_targets],
    )
    result["triton"] = {
        "version": version_text,
        "astsource_params": arch_params,
        "visible_archs": [f"sm_{arch}" for arch in visible_archs],
        "compile_targets": [f"sm_{arch}" for arch in compile_targets],
    }
except Exception as exc:
    add_check("triton", False, error=repr(exc))

rc, cargo_stdout, cargo_stderr = command_output(["cargo", "--version"])
cargo_ok = rc == 0
add_check("cargo", cargo_ok, version=cargo_stdout or cargo_stderr)

rc, rustc_stdout, rustc_stderr = command_output(["rustc", "--version"])
rustc_ok = rc == 0
add_check("rustc", rustc_ok, version=rustc_stdout or rustc_stderr)

rc, metadata_stdout, metadata_stderr = command_output(
    ["cargo", "metadata", "--format-version", "1", "--locked", "--no-deps"],
    cwd=str(repo),
)
metadata_ok = rc == 0
add_check(
    "cargo_metadata_locked",
    metadata_ok,
    stdout_preview=metadata_stdout[:400],
    stderr_preview=metadata_stderr[:400],
)

rc, maturin_stdout, maturin_stderr = command_output([str(maturin_bin), "--version"])
add_check("maturin", rc == 0, version=maturin_stdout or maturin_stderr)

result["ready"] = all(check["ok"] for check in result["checks"])
result["readiness_artifact"] = str(ready_json)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ready"] else 1)
PY
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    mv "$tmp_json" "$READY_JSON"
    touch "$READY_STAMP"
    ok "Capture host is ready"
    info "Readiness JSON: $READY_JSON"
    info "Readiness stamp: $READY_STAMP"
    cat "$READY_JSON"
    exit 0
fi

rm -f "$READY_JSON" "$READY_STAMP"
warn "Capture host is not ready; readiness artifacts were cleared"
cat "$tmp_json"
rm -f "$tmp_json"
exit 1

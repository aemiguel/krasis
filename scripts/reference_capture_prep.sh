#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KRASIS_DEV_SCRIPT:-}" ]]; then
    echo "ERROR: This script must be run via ./dev reference-prep, not directly." >&2
    echo "  Usage: ./dev reference-prep <model>" >&2
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

PYTHON_BIN="${KRASIS_DEV_PYTHON:-python3}"
PIP_BIN="${KRASIS_DEV_PIP:-}"
PYTHON_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
CAPTURE_ENV_DIR="${KRASIS_REFERENCE_CAPTURE_VENV:-${HOME}/.krasis/reference-capture-venv}"
CAPTURE_PYTHON="${CAPTURE_ENV_DIR}/bin/python"
CAPTURE_PIP="${CAPTURE_ENV_DIR}/bin/pip"
DEST_ROOT="${HOME}/.krasis/models"
LOG_ROOT=""
MODEL_ARG=""
LIST_MODELS=0
PRINT_URLS=0
DETACH=0
INSTALL_DEPS=1
START_DOWNLOAD=1
URLS_ONLY=0
REQUIRED_CAPTURE_DEPS=(
    "huggingface_hub==0.36.2"
    "accelerate==1.13.0"
    "transformers==4.57.1"
    "safetensors==0.7.0"
    "sentencepiece"
    "protobuf<7"
)

usage() {
    cat <<'EOF'
Usage: ./dev reference-prep <model> [options]
       ./dev reference-prep --deps-only

Prepare the HF reference-capture environment and download a public model to
~/.krasis/models without using an HF token.

Dependencies are installed into a dedicated capture venv at:
  ~/.krasis/reference-capture-venv
This keeps HF capture deps isolated from the main Krasis serving env.

Options:
  --list-models           Show supported aliases and exit
  --print-urls            Print the exact public HF repo URLs before download
  --urls-only             Print the resolved repo and destination, then exit
  --detach                Leave the download running in the background and log to krasis/logs
  --deps-only             Install/repair capture dependencies only; no model required
  --download-only         Start the download only; skip dependency installs
  --dest-root PATH        Override the model root (default: ~/.krasis/models)
  -h, --help              Show this help

Supported aliases:
  qcn       -> Qwen/Qwen3-Coder-Next
  qwen35    -> Qwen/Qwen3.5-35B-A3B
  q122b     -> Qwen/Qwen3.5-122B-A10B
  gemma     -> google/gemma-4-26B-A4B-it
  minimax   -> MiniMaxAI/MiniMax-M2.5
  q235      -> Qwen/Qwen3-235B-A22B
  q397      -> Qwen/Qwen3.5-397B-A17B

You can also pass a full Hugging Face repo id directly, for example:
  ./dev reference-prep Qwen/Qwen3-Coder-Next
EOF
}

list_models() {
    cat <<'EOF'
qcn      Qwen/Qwen3-Coder-Next
qwen35   Qwen/Qwen3.5-35B-A3B
q122b    Qwen/Qwen3.5-122B-A10B
gemma    google/gemma-4-26B-A4B-it
minimax  MiniMaxAI/MiniMax-M2.5
q235     Qwen/Qwen3-235B-A22B
q397     Qwen/Qwen3.5-397B-A17B
EOF
}

normalize_model_arg() {
    local raw="$1"
    local lower
    lower=$(echo "$raw" | tr '[:upper:]' '[:lower:]')
    case "$lower" in
        qcn|qwen3-coder-next) echo "Qwen/Qwen3-Coder-Next" ;;
        qwen35|q35b|qwen3.5-35b-a3b) echo "Qwen/Qwen3.5-35B-A3B" ;;
        q122b|qwen122|qwen3.5-122b-a10b) echo "Qwen/Qwen3.5-122B-A10B" ;;
        gemma|gemma26|gemma-4-26b-a4b-it) echo "google/gemma-4-26B-A4B-it" ;;
        minimax|minimax25|minimax-m2.5) echo "MiniMaxAI/MiniMax-M2.5" ;;
        q235|qwen235|qwen3-235b-a22b) echo "Qwen/Qwen3-235B-A22B" ;;
        q397|qwen397|qwen3.5-397b-a17b) echo "Qwen/Qwen3.5-397B-A17B" ;;
        */*) echo "$raw" ;;
        *)
            err "Unknown model alias: $raw
  Run ./dev reference-prep --list-models for supported aliases, or pass a full repo id."
            ;;
    esac
}

detect_torch_index_url() {
    local ver=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)
    fi
    case "$ver" in
        12.8*|12.9*|13.*) echo "https://download.pytorch.org/whl/cu128" ;;
        12.6*|12.7*) echo "https://download.pytorch.org/whl/cu126" ;;
        12.4*|12.5*) echo "https://download.pytorch.org/whl/cu124" ;;
        12.1*|12.2*|12.3*) echo "https://download.pytorch.org/whl/cu121" ;;
        *) echo "https://download.pytorch.org/whl/cu118" ;;
    esac
}

ensure_torch() {
    info "Checking PyTorch/CUDA in the reference-capture env"
    local need_torch=0
    local torch_status
    torch_status=$("$CAPTURE_PYTHON" - <<'PY' 2>/dev/null || true
try:
    import torch
except Exception:
    print("missing")
    raise SystemExit(0)
print("ok" if torch.cuda.is_available() else "cpu_only")
PY
)
    case "$torch_status" in
        ok)
            ok "CUDA torch already available"
            return 0
            ;;
        cpu_only)
            warn "torch is installed but CUDA is not available in this env"
            need_torch=1
            ;;
        *)
            warn "torch is not installed in this env"
            need_torch=1
            ;;
    esac

    local index_url
    index_url=$(detect_torch_index_url)
    info "Installing CUDA torch from $index_url"
    "$CAPTURE_PIP" install torch torchvision torchaudio --index-url "$index_url"
}

ensure_capture_env() {
    if [[ -x "$CAPTURE_PYTHON" && -x "$CAPTURE_PIP" ]]; then
        ok "Reference-capture venv already exists: $CAPTURE_ENV_DIR"
        return 0
    fi

    info "Creating reference-capture venv: $CAPTURE_ENV_DIR"
    mkdir -p "$(dirname "$CAPTURE_ENV_DIR")"
    "$PYTHON_BIN" -m venv "$CAPTURE_ENV_DIR"
    "$CAPTURE_PIP" install pip setuptools wheel
}

capture_deps_status() {
    ensure_capture_env >/dev/null
    "$CAPTURE_PYTHON" - <<'PY'
from importlib import metadata
import json
import sys

requirements = {
    "huggingface_hub": "0.36.2",
    "accelerate": "1.13.0",
    "transformers": "4.57.1",
    "safetensors": "0.7.0",
}
status = {"ok": True, "packages": {}, "issues": []}
for package, expected in requirements.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    status["packages"][package] = installed
    if installed != expected:
        status["ok"] = False
        status["issues"].append(
            {
                "package": package,
                "installed": installed,
                "expected": expected,
            }
        )

for package in ("sentencepiece", "protobuf"):
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    status["packages"][package] = installed
    if package == "sentencepiece" and installed is None:
        status["ok"] = False
        status["issues"].append(
            {
                "package": package,
                "installed": None,
                "expected": "installed",
            }
        )
    if package == "protobuf":
        major = None
        if installed:
            try:
                major = int(installed.split(".", 1)[0])
            except ValueError:
                major = None
        if installed is None or major is None or major >= 7:
            status["ok"] = False
            status["issues"].append(
                {
                    "package": package,
                    "installed": installed,
                    "expected": "<7",
                }
            )

print(json.dumps(status))
PY
}

print_capture_dep_versions() {
    ensure_capture_env
    "$CAPTURE_PYTHON" - <<'PY'
from importlib import metadata

for package in (
    "huggingface_hub",
    "accelerate",
    "transformers",
    "safetensors",
    "sentencepiece",
    "protobuf",
):
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "missing"
    print(f"{package}={version}")
PY
}

install_capture_deps() {
    ensure_capture_env
    ensure_torch
    local dep_status
    dep_status=$(capture_deps_status)
    if "$CAPTURE_PYTHON" - "$dep_status" <<'PY'
import json
import sys

status = json.loads(sys.argv[1])
raise SystemExit(0 if status["ok"] else 1)
PY
    then
        ok "Reference-capture Python deps already satisfy pinned versions"
        print_capture_dep_versions
        return 0
    fi

    warn "Reference-capture Python deps need repair"
    "$CAPTURE_PYTHON" - "$dep_status" <<'PY'
import json
import sys

status = json.loads(sys.argv[1])
for issue in status["issues"]:
    print(
        f"  {issue['package']}: installed={issue['installed'] or 'missing'} "
        f"expected={issue['expected']}"
    )
PY
    info "Installing pinned reference-capture and Hugging Face download dependencies"
    "$CAPTURE_PIP" install "${REQUIRED_CAPTURE_DEPS[@]}"
    print_capture_dep_versions
}

resolve_hf_cli() {
    ensure_capture_env
    if [[ -x "$CAPTURE_ENV_DIR/bin/hf" ]]; then
        echo "$CAPTURE_ENV_DIR/bin/hf"
        return 0
    fi
    if [[ -x "$CAPTURE_ENV_DIR/bin/huggingface-cli" ]]; then
        echo "$CAPTURE_ENV_DIR/bin/huggingface-cli"
        return 0
    fi
    if "$CAPTURE_PYTHON" -m huggingface_hub.commands.huggingface_cli --help >/dev/null 2>&1; then
        echo "$CAPTURE_PYTHON -m huggingface_hub.commands.huggingface_cli"
        return 0
    fi
    if "$CAPTURE_PYTHON" -m hf --help >/dev/null 2>&1; then
        echo "$CAPTURE_PYTHON -m hf"
        return 0
    fi
    err "No Hugging Face CLI found after dependency install."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-models)
            LIST_MODELS=1
            shift
            ;;
        --print-urls)
            PRINT_URLS=1
            shift
            ;;
        --urls-only)
            PRINT_URLS=1
            INSTALL_DEPS=0
            START_DOWNLOAD=0
            URLS_ONLY=1
            shift
            ;;
        --detach)
            DETACH=1
            shift
            ;;
        --deps-only)
            START_DOWNLOAD=0
            shift
            ;;
        --download-only)
            INSTALL_DEPS=0
            shift
            ;;
        --dest-root)
            [[ $# -ge 2 ]] || err "--dest-root requires a path"
            DEST_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            err "Unknown option: $1"
            ;;
        *)
            [[ -z "$MODEL_ARG" ]] || err "Only one model/repo may be specified"
            MODEL_ARG="$1"
            shift
            ;;
    esac
done

if [[ "$LIST_MODELS" -eq 1 ]]; then
    list_models
    exit 0
fi

if [[ -z "$MODEL_ARG" && "$START_DOWNLOAD" -eq 1 ]]; then
    err "Missing model argument
  Run ./dev reference-prep --help for usage."
fi

if [[ -n "$MODEL_ARG" ]]; then
    REPO_ID=$(normalize_model_arg "$MODEL_ARG")
    LOCAL_NAME="${REPO_ID##*/}"
    DEST_DIR="${DEST_ROOT%/}/${LOCAL_NAME}"
    REPO_URL="https://huggingface.co/${REPO_ID}"
    TREE_URL="${REPO_URL}/tree/main"

    info "Model repo: $REPO_ID"
    info "Local dir: $DEST_DIR"
    if [[ "$PRINT_URLS" -eq 1 ]]; then
        info "Public repo URL: $REPO_URL"
        info "Public file tree: $TREE_URL"
    fi

    if [[ "$URLS_ONLY" -eq 1 ]]; then
        ok "Resolved model download target"
        exit 0
    fi
fi

mkdir -p "$DEST_ROOT"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    install_capture_deps
fi

if [[ "$START_DOWNLOAD" -eq 0 ]]; then
    ok "Dependency preparation complete"
    exit 0
fi

HF_CLI=$(resolve_hf_cli)
info "Hugging Face CLI: $HF_CLI"
DOWNLOAD_CMD=(
    $HF_CLI
    download
    "$REPO_ID"
    --local-dir "$DEST_DIR"
)

if [[ "$DETACH" -eq 1 ]]; then
    LOG_ROOT="${LOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs}"
    mkdir -p "$LOG_ROOT"
    ts=$(date +%Y%m%d_%H%M%S)
    safe_name=$(echo "$LOCAL_NAME" | tr '/ ' '__')
    log_file="$LOG_ROOT/reference-prep_${safe_name}_${ts}.log"
    info "Starting detached download"
    info "Log file: $log_file"
    printf '%s\n' "Repo: $REPO_ID" "Dest: $DEST_DIR" "URL: $REPO_URL" > "$log_file"
    printf '$ %q ' "${DOWNLOAD_CMD[@]}" >> "$log_file"
    printf '\n\n' >> "$log_file"
    nohup "${DOWNLOAD_CMD[@]}" >> "$log_file" 2>&1 < /dev/null &
    pid=$!
    ok "Download started in background (pid $pid)"
    echo "Tail with: tail -f $log_file"
    exit 0
fi

info "Starting download"
printf '$ %q ' "${DOWNLOAD_CMD[@]}"
printf '\n'
"${DOWNLOAD_CMD[@]}"
ok "Download complete: $DEST_DIR"

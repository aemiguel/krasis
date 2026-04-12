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
CAPTURE_ROOT="${KRASIS_REFERENCE_CAPTURE_ROOT:-${HOME}/.krasis}"
CAPTURE_ROOT_SOURCE="${KRASIS_REFERENCE_CAPTURE_ROOT_SOURCE:-home}"
CAPTURE_ENV_DIR="${KRASIS_REFERENCE_CAPTURE_VENV:-${CAPTURE_ROOT}/reference-capture-venv}"
CAPTURE_PYTHON="${CAPTURE_ENV_DIR}/bin/python"
CAPTURE_PIP="${CAPTURE_ENV_DIR}/bin/pip"
DEST_ROOT="${KRASIS_REFERENCE_CAPTURE_MODELS_DIR:-${CAPTURE_ROOT}/models}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOG_ROOT=""
MODEL_ARG=""
REPO_ID=""
LOCAL_NAME=""
LIST_MODELS=0
PRINT_URLS=0
DETACH=0
INSTALL_DEPS=1
START_DOWNLOAD=1
URLS_ONLY=0
PRINT_DEPS_JSON=0
PRINT_DEP_GROUPS=0
PRINT_REQUIRED_DEPS_JSON=0
BASE_ONLY=0
NO_OVERLAP=0
INTERNAL_DOWNLOAD_LOOP=0
HF_HUB_VERSION="${KRASIS_REFERENCE_CAPTURE_HF_HUB_VERSION:-1.10.1}"
LEGACY_HF_HUB_VERSION="${KRASIS_REFERENCE_CAPTURE_LEGACY_HF_HUB_VERSION:-0.36.2}"
ACCELERATE_VERSION="${KRASIS_REFERENCE_CAPTURE_ACCELERATE_VERSION:-1.13.0}"
TRANSFORMERS_VERSION="${KRASIS_REFERENCE_CAPTURE_TRANSFORMERS_VERSION:-5.5.3}"
SAFETENSORS_VERSION="${KRASIS_REFERENCE_CAPTURE_SAFETENSORS_VERSION:-0.7.0}"
MAMBA_SSM_VERSION="${KRASIS_REFERENCE_CAPTURE_MAMBA_SSM_VERSION:-2.3.1}"
CAUSAL_CONV1D_VERSION="${KRASIS_REFERENCE_CAPTURE_CAUSAL_CONV1D_VERSION:-1.6.1}"
EINOPS_VERSION="${KRASIS_REFERENCE_CAPTURE_EINOPS_VERSION:-0.8.2}"
NINJA_VERSION="${KRASIS_REFERENCE_CAPTURE_NINJA_VERSION:-1.13.0}"
MAMBA_CAPTURE_DEPS=(
    "mamba-ssm==${MAMBA_SSM_VERSION}"
    "causal-conv1d==${CAUSAL_CONV1D_VERSION}"
    "einops==${EINOPS_VERSION}"
    "ninja==${NINJA_VERSION}"
)
DOWNLOAD_MAX_WORKERS="${KRASIS_REFERENCE_CAPTURE_MAX_WORKERS:-8}"
DOWNLOAD_RETRY_COUNT="${KRASIS_REFERENCE_CAPTURE_DOWNLOAD_RETRY_COUNT:-6}"
DOWNLOAD_RETRY_BACKOFF_SEC="${KRASIS_REFERENCE_CAPTURE_DOWNLOAD_RETRY_BACKOFF_SEC:-15}"
DISABLE_XET="${KRASIS_REFERENCE_CAPTURE_DISABLE_XET:-1}"

usage() {
    cat <<'EOF'
Usage: ./dev reference-prep <model> [options]
       ./dev reference-prep --deps-only

Prepare the HF reference-capture environment and download a public model to
the capture-root models dir without using an HF token.

Dependencies are installed into a dedicated capture venv at:
  <capture-root>/reference-capture-venv
This keeps HF capture deps isolated from the main Krasis serving env.

Options:
  --list-models           Show supported aliases and exit
  --print-urls            Print the exact public HF repo URLs before download
  --urls-only             Print the resolved repo and destination, then exit
  --detach                Leave the download running in the background and log to krasis/logs
  --deps-only             Install/repair capture dependencies only; no model required
  --download-only         Start the download only; skip dependency installs
  --base-only             Only install the base HF capture stack; skip model-family extras
  --no-overlap            Do not overlap model-family extra builds with downloads
  --max-workers N         Set hf download worker concurrency (default: 8)
  --retry-count N         Retry failed downloads this many times (default: 6)
  --retry-backoff-sec N   Sleep between failed attempts (default: 15)
  --enable-xet            Leave HF Xet enabled for this run
  --disable-xet           Disable HF Xet for this run (default)
  --print-deps-json       Print the authoritative pinned capture dependency set as JSON and exit
  --print-dep-groups      Print the dependency groups and supported model families as JSON and exit
  --print-required-deps-json  Print the flattened required dependency set for the target model/mode
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
  nemotron-nano   -> nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  nemotron-super  -> nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

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
nemotron-nano   nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
nemotron-super  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
EOF
}

print_required_capture_deps_json() {
    local resolved_hf_hub_version
    local resolved_transformers_version
    resolved_hf_hub_version="$(resolve_capture_hf_hub_version)"
    resolved_transformers_version="$(resolve_capture_transformers_version)"
    "$PYTHON_BIN" - "$resolved_hf_hub_version" "$ACCELERATE_VERSION" "$resolved_transformers_version" "$SAFETENSORS_VERSION" "$MAMBA_SSM_VERSION" "$CAUSAL_CONV1D_VERSION" "$EINOPS_VERSION" "$NINJA_VERSION" <<'PY'
import json
import sys

payload = {
    "base": {
        "huggingface_hub": sys.argv[1],
        "accelerate": sys.argv[2],
        "transformers": sys.argv[3],
        "safetensors": sys.argv[4],
        "sentencepiece": "installed",
        "protobuf": "<7",
    },
    "groups": {
        "mamba_runtime": {
            "mamba-ssm": sys.argv[5],
            "causal-conv1d": sys.argv[6],
            "einops": sys.argv[7],
            "ninja": sys.argv[8],
        }
    },
    "model_groups": {
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": ["mamba_runtime"],
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ["mamba_runtime"],
    },
}
print(json.dumps(payload, sort_keys=True))
PY
}

model_dep_groups_json() {
    print_required_capture_deps_json
}

print_flat_required_capture_deps_json() {
    local mode="${1:-full}"
    KRASIS_CAPTURE_EXPECTED_DEPS="$(collect_required_capture_deps "$mode")" "$PYTHON_BIN" - <<'PY'
import json
import os

deps = os.environ["KRASIS_CAPTURE_EXPECTED_DEPS"].splitlines()
payload = {}
for dep in deps:
    if not dep:
        continue
    if dep == "sentencepiece":
        payload["sentencepiece"] = "installed"
    elif dep.startswith("protobuf<"):
        payload["protobuf"] = dep.split("<", 1)[1]
    else:
        name, version = dep.split("==", 1)
        payload[name] = version
print(json.dumps(payload, sort_keys=True))
PY
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
        nemotron-nano|nemotronnano|nano|nvidia-nemotron-3-nano-30b-a3b-bf16) echo "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" ;;
        nemotron-super|nemotronsuper|super|nvidia-nemotron-3-super-120b-a12b-bf16) echo "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" ;;
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

required_dep_group_names_for_model() {
    local repo_id="$1"
    case "$repo_id" in
        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
            echo "mamba_runtime"
            ;;
        *)
            ;;
    esac
}

append_dep_group_packages() {
    local group_name="$1"
    case "$group_name" in
        mamba_runtime)
            printf '%s\n' "${MAMBA_CAPTURE_DEPS[@]}"
            ;;
        *)
            err "Unknown dependency group: $group_name"
            ;;
    esac
}

resolve_capture_transformers_version() {
    local resolved_version="$TRANSFORMERS_VERSION"
    local config_path=""

    if [[ -n "$LOCAL_NAME" ]]; then
        config_path="${DEST_ROOT%/}/${LOCAL_NAME}/config.json"
    fi

    if [[ -n "$config_path" && -f "$config_path" ]]; then
        local model_specific_version
        model_specific_version=$("$PYTHON_BIN" - "$config_path" "$resolved_version" <<'PY'
import json
import sys

config_path = sys.argv[1]
default_version = sys.argv[2]

try:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    print(default_version)
    raise SystemExit(0)

model_type = payload.get("model_type")
transformers_version = payload.get("transformers_version")

if model_type == "nemotron_h" and isinstance(transformers_version, str) and transformers_version.strip():
    print(transformers_version.strip())
else:
    print(default_version)
PY
)
        if [[ -n "$model_specific_version" ]]; then
            resolved_version="$model_specific_version"
        fi
    fi

    echo "$resolved_version"
}

resolve_capture_hf_hub_version() {
    local resolved_version="$HF_HUB_VERSION"
    local transformers_version
    transformers_version="$(resolve_capture_transformers_version)"

    if [[ "$transformers_version" == 4.* ]]; then
        resolved_version="$LEGACY_HF_HUB_VERSION"
    fi

    echo "$resolved_version"
}

emit_base_capture_deps() {
    local resolved_hf_hub_version
    local resolved_transformers_version
    resolved_hf_hub_version="$(resolve_capture_hf_hub_version)"
    resolved_transformers_version="$(resolve_capture_transformers_version)"
    printf '%s\n' \
        "huggingface_hub==${resolved_hf_hub_version}" \
        "accelerate==${ACCELERATE_VERSION}" \
        "transformers==${resolved_transformers_version}" \
        "safetensors==${SAFETENSORS_VERSION}" \
        "sentencepiece" \
        "protobuf<7"
}

collect_required_capture_deps() {
    local mode="${1:-full}"
    local -a deps=()
    while IFS= read -r package; do
        [[ -n "$package" ]] || continue
        deps+=("$package")
    done < <(emit_base_capture_deps)

    if [[ "$mode" != "base" && -n "$REPO_ID" ]]; then
        while IFS= read -r group_name; do
            [[ -n "$group_name" ]] || continue
            while IFS= read -r package; do
                [[ -n "$package" ]] || continue
                deps+=("$package")
            done < <(append_dep_group_packages "$group_name")
        done < <(required_dep_group_names_for_model "$REPO_ID")
    fi

    printf '%s\n' "${deps[@]}"
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
    local mode="${1:-full}"
    local deps_json
    deps_json=$(KRASIS_CAPTURE_EXPECTED_DEPS="$(collect_required_capture_deps "$mode")" "$PYTHON_BIN" - <<'PY'
import json
import os

deps = os.environ["KRASIS_CAPTURE_EXPECTED_DEPS"].splitlines()
payload = {}
for dep in deps:
    if not dep:
        continue
    if dep == "sentencepiece":
        payload["sentencepiece"] = "installed"
    elif dep.startswith("protobuf<"):
        payload["protobuf"] = dep.split("<", 1)[1]
    else:
        name, version = dep.split("==", 1)
        payload[name] = version
print(json.dumps(payload))
PY
)
    "$CAPTURE_PYTHON" - "$deps_json" <<'PY'
from importlib import metadata
import json
import sys

requirements = json.loads(sys.argv[1])
status = {"ok": True, "packages": {}, "issues": []}
for package, expected in requirements.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    status["packages"][package] = installed
    if package not in ("sentencepiece", "protobuf") and installed != expected:
        status["ok"] = False
        status["issues"].append(
            {
                "package": package,
                "installed": installed,
                "expected": expected,
            }
        )

if "sentencepiece" in requirements:
    installed = status["packages"].get("sentencepiece")
    if installed is None:
        status["ok"] = False
        status["issues"].append(
            {
                "package": "sentencepiece",
                "installed": None,
                "expected": "installed",
            }
        )

if "protobuf" in requirements:
    installed = status["packages"].get("protobuf")
    major = None
    if installed:
        try:
            major = int(installed.split(".", 1)[0])
        except ValueError:
            major = None
    max_major_exclusive = None
    try:
        max_major_exclusive = int(requirements["protobuf"])
    except ValueError:
        max_major_exclusive = None
    if installed is None or major is None or max_major_exclusive is None or major >= max_major_exclusive:
        status["ok"] = False
        status["issues"].append(
            {
                "package": "protobuf",
                "installed": installed,
                "expected": f"<{requirements['protobuf']}",
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
    "mamba-ssm",
    "causal-conv1d",
    "einops",
    "ninja",
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
    local mode="${1:-full}"
    ensure_capture_env
    ensure_torch
    local dep_status
    dep_status=$(capture_deps_status "$mode")
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
    info "Installing pinned reference-capture dependencies (mode: $mode)"
    mapfile -t deps_to_install < <(collect_required_capture_deps "$mode")
    "$CAPTURE_PIP" install "${deps_to_install[@]}"
    print_capture_dep_versions
}

install_capture_extra_deps_background() {
    local mode="${1:-full}"
    local log_file="$2"
    [[ "$mode" == "full" ]] || return 0
    mapfile -t extra_groups < <(required_dep_group_names_for_model "$REPO_ID")
    [[ "${#extra_groups[@]}" -gt 0 ]] || return 0

    mapfile -t extra_deps < <(
        for group_name in "${extra_groups[@]}"; do
            append_dep_group_packages "$group_name"
        done
    )
    [[ "${#extra_deps[@]}" -gt 0 ]] || return 0

    info "Starting model-family extra dependency build in background"
    printf '%s\n' "Model: ${LOCAL_NAME}" "Repo: ${REPO_ID}" "Mode: ${mode}" > "$log_file"
    printf '$ %q ' "$CAPTURE_PIP" install "${extra_deps[@]}" >> "$log_file"
    printf '\n\n' >> "$log_file"
    "$CAPTURE_PIP" install "${extra_deps[@]}" >> "$log_file" 2>&1 &
    echo $!
}

resolve_hf_cli() {
    ensure_capture_env >/dev/null
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

validate_positive_int() {
    local value="$1"
    local label="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || err "$label must be a non-negative integer, got: $value"
}

download_lock_dir() {
    echo "${DEST_DIR}/.cache/huggingface/download"
}

count_matching_files() {
    local root="$1"
    local pattern="$2"
    if [[ -d "$root" ]]; then
        find "$root" -type f -name "$pattern" | wc -l | tr -d ' '
    else
        echo 0
    fi
}

live_download_pids_for_target() {
    local pattern
    pattern="download[[:space:]]+${REPO_ID}.*--local-dir[[:space:]]+${DEST_DIR}"
    pgrep -af "$pattern" | awk -v self="$$" '$1 != self {print $1}'
}

clear_stale_local_dir_locks() {
    local lock_dir
    lock_dir="$(download_lock_dir)"
    [[ -d "$lock_dir" ]] || return 0

    local -a active_pids=()
    mapfile -t active_pids < <(live_download_pids_for_target || true)
    if [[ "${#active_pids[@]}" -gt 0 ]]; then
        warn "Leaving local-dir shard locks in place; active downloader pid(s): ${active_pids[*]}"
        return 0
    fi

    local lock_count
    lock_count=$(count_matching_files "$lock_dir" "*.lock")
    if [[ "$lock_count" -eq 0 ]]; then
        return 0
    fi

    warn "Clearing ${lock_count} stale local-dir shard lock(s) under $lock_dir"
    find "$lock_dir" -type f -name '*.lock' -delete
}

build_download_command() {
    DOWNLOAD_CMD=(
        $HF_CLI
        download
        "$REPO_ID"
        --local-dir "$DEST_DIR"
        --max-workers "$DOWNLOAD_MAX_WORKERS"
    )
}

print_download_state() {
    local lock_dir incomplete_count lock_count
    lock_dir="$(download_lock_dir)"
    incomplete_count=$(count_matching_files "$lock_dir" "*.incomplete")
    lock_count=$(count_matching_files "$lock_dir" "*.lock")
    info "Download state: workers=${DOWNLOAD_MAX_WORKERS} retries=${DOWNLOAD_RETRY_COUNT} backoff=${DOWNLOAD_RETRY_BACKOFF_SEC}s xet_disabled=${DISABLE_XET} incomplete=${incomplete_count} locks=${lock_count}"
}

run_download_with_retries() {
    local max_attempts attempt status current_workers
    validate_positive_int "$DOWNLOAD_MAX_WORKERS" "--max-workers"
    validate_positive_int "$DOWNLOAD_RETRY_COUNT" "--retry-count"
    validate_positive_int "$DOWNLOAD_RETRY_BACKOFF_SEC" "--retry-backoff-sec"

    max_attempts=$((DOWNLOAD_RETRY_COUNT + 1))
    current_workers="$DOWNLOAD_MAX_WORKERS"
    attempt=1

    while [[ "$attempt" -le "$max_attempts" ]]; do
        DOWNLOAD_MAX_WORKERS="$current_workers"
        clear_stale_local_dir_locks
        build_download_command
        print_download_state
        info "Starting download attempt ${attempt}/${max_attempts}"
        printf '$ '
        if [[ "$DISABLE_XET" -eq 1 ]]; then
            printf 'HF_HUB_DISABLE_XET=1 '
        fi
        printf '%q ' "${DOWNLOAD_CMD[@]}"
        printf '\n'

        status=0
        if [[ "$DISABLE_XET" -eq 1 ]]; then
            HF_HUB_DISABLE_XET=1 "${DOWNLOAD_CMD[@]}" || status=$?
        else
            "${DOWNLOAD_CMD[@]}" || status=$?
        fi
        if [[ "$status" -eq 0 ]]; then
            ok "Download complete: $DEST_DIR"
            return 0
        fi

        warn "Download attempt ${attempt}/${max_attempts} failed with exit code ${status}"
        print_download_state
        clear_stale_local_dir_locks
        if [[ "$attempt" -ge "$max_attempts" ]]; then
            break
        fi

        if [[ "$current_workers" -gt 1 ]]; then
            current_workers=$(((current_workers + 1) / 2))
            warn "Reducing hf download workers for retry: ${current_workers}"
        fi

        if [[ "$DOWNLOAD_RETRY_BACKOFF_SEC" -gt 0 ]]; then
            info "Sleeping ${DOWNLOAD_RETRY_BACKOFF_SEC}s before retry"
            sleep "$DOWNLOAD_RETRY_BACKOFF_SEC"
        fi
        attempt=$((attempt + 1))
    done

    err "Download failed after ${max_attempts} attempt(s): $DEST_DIR"
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
        --base-only)
            BASE_ONLY=1
            shift
            ;;
        --no-overlap)
            NO_OVERLAP=1
            shift
            ;;
        --max-workers)
            [[ $# -ge 2 ]] || err "--max-workers requires an integer"
            DOWNLOAD_MAX_WORKERS="$2"
            shift 2
            ;;
        --retry-count)
            [[ $# -ge 2 ]] || err "--retry-count requires an integer"
            DOWNLOAD_RETRY_COUNT="$2"
            shift 2
            ;;
        --retry-backoff-sec)
            [[ $# -ge 2 ]] || err "--retry-backoff-sec requires an integer"
            DOWNLOAD_RETRY_BACKOFF_SEC="$2"
            shift 2
            ;;
        --enable-xet)
            DISABLE_XET=0
            shift
            ;;
        --disable-xet)
            DISABLE_XET=1
            shift
            ;;
        --print-deps-json)
            PRINT_DEPS_JSON=1
            shift
            ;;
        --print-dep-groups)
            PRINT_DEP_GROUPS=1
            shift
            ;;
        --print-required-deps-json)
            PRINT_REQUIRED_DEPS_JSON=1
            shift
            ;;
        --dest-root)
            [[ $# -ge 2 ]] || err "--dest-root requires a path"
            DEST_ROOT="$2"
            shift 2
            ;;
        --internal-download-loop)
            INTERNAL_DOWNLOAD_LOOP=1
            shift
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

if [[ "$PRINT_DEPS_JSON" -eq 1 ]]; then
    print_required_capture_deps_json
    exit 0
fi

if [[ "$PRINT_DEP_GROUPS" -eq 1 ]]; then
    model_dep_groups_json
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

    if [[ "$PRINT_REQUIRED_DEPS_JSON" -eq 1 ]]; then
        dep_mode="full"
        if [[ "$BASE_ONLY" -eq 1 ]]; then
            dep_mode="base"
        fi
        print_flat_required_capture_deps_json "$dep_mode"
        exit 0
    fi

    info "Model repo: $REPO_ID"
    info "Local dir: $DEST_DIR"
    info "Capture root: $CAPTURE_ROOT (source: $CAPTURE_ROOT_SOURCE)"
    if [[ "$PRINT_URLS" -eq 1 ]]; then
        info "Public repo URL: $REPO_URL"
        info "Public file tree: $TREE_URL"
    fi

    if [[ "$URLS_ONLY" -eq 1 ]]; then
        ok "Resolved model download target"
        exit 0
    fi
fi

if [[ "$PRINT_REQUIRED_DEPS_JSON" -eq 1 ]]; then
    dep_mode="full"
    if [[ "$BASE_ONLY" -eq 1 || -z "$REPO_ID" ]]; then
        dep_mode="base"
    fi
    print_flat_required_capture_deps_json "$dep_mode"
    exit 0
fi

mkdir -p "$DEST_ROOT"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    dep_mode="full"
    if [[ "$BASE_ONLY" -eq 1 || -z "$REPO_ID" ]]; then
        dep_mode="base"
    elif [[ "$NO_OVERLAP" -eq 0 ]]; then
        dep_mode="base"
    fi
    install_capture_deps "$dep_mode"
fi

if [[ "$START_DOWNLOAD" -eq 0 ]]; then
    ok "Dependency preparation complete"
    exit 0
fi

HF_CLI=$(resolve_hf_cli)
info "Hugging Face CLI: $HF_CLI"
build_download_command

if [[ "$DETACH" -eq 1 ]]; then
    LOG_ROOT="${LOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs}"
    mkdir -p "$LOG_ROOT"
    ts=$(date +%Y%m%d_%H%M%S)
    safe_name=$(echo "$LOCAL_NAME" | tr '/ ' '__')
    log_file="$LOG_ROOT/reference-prep_${safe_name}_${ts}.log"
    local_xet_flag="--disable-xet"
    detached_cmd=""
    extra_deps_pid=""
    extra_deps_log=""
    if [[ "$DISABLE_XET" -eq 0 ]]; then
        local_xet_flag="--enable-xet"
    fi
    if [[ "$INSTALL_DEPS" -eq 1 && "$BASE_ONLY" -eq 0 && "$NO_OVERLAP" -eq 0 ]]; then
        extra_deps_log="$LOG_ROOT/reference-prep-extra-deps_${safe_name}_${ts}.log"
        extra_deps_pid=$(install_capture_extra_deps_background "full" "$extra_deps_log" || true)
    fi
    info "Starting detached download"
    info "Log file: $log_file"
    printf '%s\n' \
        "Repo: $REPO_ID" \
        "Dest: $DEST_DIR" \
        "URL: $REPO_URL" \
        "Max workers: $DOWNLOAD_MAX_WORKERS" \
        "Retry count: $DOWNLOAD_RETRY_COUNT" \
        "Retry backoff sec: $DOWNLOAD_RETRY_BACKOFF_SEC" \
        "Disable Xet: $DISABLE_XET" > "$log_file"
    printf -v detached_cmd '%q ' \
        env \
        KRASIS_DEV_SCRIPT=1 \
        KRASIS_DEV_PYTHON="$PYTHON_BIN" \
        KRASIS_DEV_PIP="$PIP_BIN" \
        KRASIS_REFERENCE_CAPTURE_ROOT="$CAPTURE_ROOT" \
        KRASIS_REFERENCE_CAPTURE_ROOT_SOURCE="$CAPTURE_ROOT_SOURCE" \
        KRASIS_REFERENCE_CAPTURE_MODELS_DIR="$DEST_ROOT" \
        KRASIS_REFERENCE_CAPTURE_VENV="$CAPTURE_ENV_DIR" \
        "$SCRIPT_PATH" "$MODEL_ARG" --download-only --internal-download-loop \
        --max-workers "$DOWNLOAD_MAX_WORKERS" \
        --retry-count "$DOWNLOAD_RETRY_COUNT" \
        --retry-backoff-sec "$DOWNLOAD_RETRY_BACKOFF_SEC" \
        --dest-root "$DEST_ROOT" \
        "$local_xet_flag"
    detached_cmd="${detached_cmd% }"
    printf '$ %s\n\n' "$detached_cmd" >> "$log_file"
    nohup bash -lc "$detached_cmd; status=\$?; printf '\\nDETACHED_EXIT:%s\\n' \"\$status\"; exit \"\$status\"" \
        >> "$log_file" 2>&1 < /dev/null &
    pid=$!
    ok "Download started in background (pid $pid)"
    if [[ -n "$extra_deps_pid" ]]; then
        ok "Model-family extra dependency build started in background (pid $extra_deps_pid)"
        echo "Extra deps log: $extra_deps_log"
    fi
    echo "Tail with: tail -f $log_file"
    exit 0
fi

if [[ "$INTERNAL_DOWNLOAD_LOOP" -eq 1 ]]; then
    run_download_with_retries
    exit 0
fi

extra_deps_pid=""
extra_deps_log=""
if [[ "$INSTALL_DEPS" -eq 1 && "$BASE_ONLY" -eq 0 && "$NO_OVERLAP" -eq 0 ]]; then
    LOG_ROOT="${LOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs}"
    mkdir -p "$LOG_ROOT"
    ts=$(date +%Y%m%d_%H%M%S)
    safe_name=$(echo "$LOCAL_NAME" | tr '/ ' '__')
    extra_deps_log="$LOG_ROOT/reference-prep-extra-deps_${safe_name}_${ts}.log"
    extra_deps_pid=$(install_capture_extra_deps_background "full" "$extra_deps_log" || true)
fi

info "Starting download"
run_download_with_retries
if [[ -n "$extra_deps_pid" ]]; then
    info "Waiting for model-family extra dependency build to finish (pid $extra_deps_pid)"
    if ! wait "$extra_deps_pid"; then
        err "Model-family extra dependency build failed. See log: $extra_deps_log"
    fi
fi

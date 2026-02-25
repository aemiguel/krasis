#!/bin/bash
# Krasis installer — downloads pre-built wheels from GitHub releases.
# No PyPI, no pipx, no sudo. Works out of the box.
#
# Install / upgrade:
#   curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash
#
# Uninstall:
#   curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- --uninstall
set -euo pipefail

REPO="brontoguana/krasis"
VENV_DIR="$HOME/.krasis/venv"
BIN_DIR="$HOME/.local/bin"
COMMANDS="krasis krasis-chat krasis-setup"

BOLD="\033[1m"
DIM="\033[2m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

info()  { echo -e "${CYAN}${BOLD}=>${NC} $1"; }
ok()    { echo -e "${GREEN}${BOLD}OK${NC} $1"; }
warn()  { echo -e "${YELLOW}${BOLD}!!${NC} $1"; }
err()   { echo -e "${RED}${BOLD}ERROR${NC} $1"; exit 1; }

# ── Uninstall ──────────────────────────────────────────────────────────
if [[ "${1:-}" == "--uninstall" ]]; then
    info "Uninstalling Krasis..."
    for cmd in $COMMANDS; do
        rm -f "$BIN_DIR/$cmd"
    done
    rm -rf "$VENV_DIR"
    ok "Krasis removed. Model files in ~/.krasis/models/ were kept."
    exit 0
fi

# ── Platform check ────────────────────────────────────────────────────
[[ "$(uname -s)" == "Linux" ]] || err "Krasis only supports Linux. On Windows, use WSL2."

# ── Check curl ────────────────────────────────────────────────────────
command -v curl &>/dev/null || err "curl is required but not found.\n  sudo apt install curl"

# ── Find all available Python 3.10+ ──────────────────────────────────
# Order matches our build matrix. Newest first so users get the best
# available wheel.  python3 is last as a catch-all — its version is
# checked, so it won't match 3.9 or older.
N_PY=0
PY_BINS=()
PY_VERS=()
PY_DOTS=()
SEEN_VERS=""

for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$candidate" &>/dev/null || continue
    ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
    major="${ver%%.*}"; minor="${ver##*.}"
    [[ "$major" -eq 3 && "$minor" -ge 10 ]] || continue
    pyver="${major}${minor}"
    # Deduplicate (python3 may resolve to same as python3.12)
    [[ "$SEEN_VERS" == *" $pyver "* ]] && continue
    SEEN_VERS="$SEEN_VERS $pyver "
    PY_BINS[$N_PY]="$candidate"
    PY_VERS[$N_PY]="$pyver"
    PY_DOTS[$N_PY]="$ver"
    ((N_PY++)) || true
done

[[ $N_PY -eq 0 ]] && err "Python 3.10+ not found. Install it:\n  sudo apt install python3"

# Use the first Python we found for JSON parsing
PARSE_PY="${PY_BINS[0]}"

# ── Fetch latest release from GitHub ─────────────────────────────────
ARCH=$(uname -m)
info "Checking latest release..."
RELEASE_JSON=$(curl -sSf "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null) \
    || err "Cannot reach GitHub API. Check your internet connection."

# ── Match best Python + wheel from release assets ────────────────────
# Tries each installed Python version (newest first) against the
# available wheels. Falls back to sdist (build from source) if no
# pre-built wheel matches any installed Python.
export KRASIS_ARCH="$ARCH"
export KRASIS_PY_VERS="${PY_VERS[*]}"
export KRASIS_PY_BINS="${PY_BINS[*]}"
export KRASIS_PY_DOTS="${PY_DOTS[*]}"

RESULT=$("$PARSE_PY" -c '
import json, sys, os

data = json.loads(sys.stdin.read())

# Check for API errors (rate limit, not found, etc.)
if "message" in data and "assets" not in data:
    print("ERROR")
    print(data["message"])
    sys.exit(0)

tag = data.get("tag_name", "unknown")
arch = os.environ["KRASIS_ARCH"]
py_vers = os.environ["KRASIS_PY_VERS"].split()
py_bins = os.environ["KRASIS_PY_BINS"].split()
py_dots = os.environ["KRASIS_PY_DOTS"].split()

# Try each installed Python, newest first — pick the first with a wheel
for i, pyver in enumerate(py_vers):
    cpver = f"cp{pyver}"
    for asset in data.get("assets", []):
        name = asset["name"]
        if name.endswith(".whl") and cpver in name and arch in name:
            print("WHEEL")
            print(tag)
            print(py_bins[i])
            print(pyver)
            print(py_dots[i])
            print(asset["browser_download_url"])
            sys.exit(0)

# No wheel for any installed Python — try sdist (build from source)
for asset in data.get("assets", []):
    if asset["name"].endswith(".tar.gz"):
        print("SDIST")
        print(tag)
        print(py_bins[0])
        print(py_vers[0])
        print(py_dots[0])
        print(asset["browser_download_url"])
        sys.exit(0)

print("NONE")
print(tag)
' <<< "$RELEASE_JSON")

unset KRASIS_ARCH KRASIS_PY_VERS KRASIS_PY_BINS KRASIS_PY_DOTS

# ── Parse result ─────────────────────────────────────────────────────
mapfile -t LINES <<< "$RESULT"
RTYPE="${LINES[0]}"

case "$RTYPE" in
    ERROR)
        err "GitHub API: ${LINES[1]}\n  If rate-limited, wait a minute and try again."
        ;;
    WHEEL)
        VERSION="${LINES[1]}"
        PYTHON="${LINES[2]}"
        PY_VER="${LINES[3]}"
        PY_VER_DOT="${LINES[4]}"
        DOWNLOAD_URL="${LINES[5]}"
        info "Latest release: $VERSION (Python $PY_VER_DOT, $ARCH)"
        ;;
    SDIST)
        VERSION="${LINES[1]}"
        PYTHON="${LINES[2]}"
        PY_VER="${LINES[3]}"
        PY_VER_DOT="${LINES[4]}"
        DOWNLOAD_URL="${LINES[5]}"
        warn "No pre-built wheel for your Python version(s) on $ARCH."
        info "Building from source (requires Rust toolchain + C compiler)..."
        command -v cargo &>/dev/null \
            || err "Rust toolchain not found. Install it first:\n  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        ;;
    NONE)
        err "No packages found in release ${LINES[1]} for $ARCH.\n  Visit https://github.com/$REPO/releases"
        ;;
    *)
        err "Unexpected response from GitHub API."
        ;;
esac

info "Using $PYTHON (Python $PY_VER_DOT)"

# ── Ensure venv module ───────────────────────────────────────────────
if ! "$PYTHON" -c "import venv" &>/dev/null; then
    warn "Python venv module not found. Installing python${PY_VER_DOT}-venv..."
    if command -v apt-get &>/dev/null; then
        SUDO=()
        [[ "$(id -u)" -ne 0 ]] && SUDO=(sudo)
        "${SUDO[@]}" apt-get update -qq 2>/dev/null
        "${SUDO[@]}" apt-get install -y "python${PY_VER_DOT}-venv" \
            || err "Failed to install python${PY_VER_DOT}-venv.\n  Try manually: sudo apt install python${PY_VER_DOT}-venv"
    elif command -v dnf &>/dev/null; then
        SUDO=()
        [[ "$(id -u)" -ne 0 ]] && SUDO=(sudo)
        "${SUDO[@]}" dnf install -y "python3-libs" 2>/dev/null \
            || err "Failed to install python venv. Try: sudo dnf install python3-libs"
    else
        err "Python venv module not found.\n  Install it for your distro (e.g. sudo apt install python${PY_VER_DOT}-venv)"
    fi
    # Verify it worked
    "$PYTHON" -c "import venv" &>/dev/null \
        || err "Python venv module still not available after install attempt."
fi

# ── Setup venv ───────────────────────────────────────────────────────
# If the venv exists, check it's healthy and uses the right Python.
# If Python was upgraded or removed, the venv will be broken — recreate.
NEED_VENV=false
if [[ -d "$VENV_DIR" ]]; then
    VENV_PYVER=$("$VENV_DIR/bin/python" -c \
        "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null || echo "0")
    if [[ "$VENV_PYVER" != "$PY_VER" ]]; then
        warn "Venv was Python ${VENV_PYVER:0:1}.${VENV_PYVER:1}, now using $PY_VER_DOT. Recreating..."
        rm -rf "$VENV_DIR"
        NEED_VENV=true
    elif ! "$VENV_DIR/bin/python" -c "import pip" &>/dev/null; then
        warn "Existing venv is broken. Recreating..."
        rm -rf "$VENV_DIR"
        NEED_VENV=true
    fi
else
    NEED_VENV=true
fi

if [[ "$NEED_VENV" == true ]]; then
    info "Creating environment at $VENV_DIR..."
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip -q 2>&1 | tail -1
fi

# ── Create models directory ──────────────────────────────────────────
mkdir -p "$HOME/.krasis/models"

# ── Download and install ─────────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

FILENAME="${DOWNLOAD_URL##*/}"
info "Downloading $FILENAME..."
curl -sSfL -o "$TMPDIR/$FILENAME" "$DOWNLOAD_URL" \
    || err "Download failed. Check your internet connection."

info "Installing Krasis ${VERSION#v}..."
"$VENV_DIR/bin/pip" install "$TMPDIR/$FILENAME" 2>&1 | tail -3

# ── Symlink commands ─────────────────────────────────────────────────
mkdir -p "$BIN_DIR"
for cmd in $COMMANDS; do
    if [[ -f "$VENV_DIR/bin/$cmd" ]]; then
        ln -sf "$VENV_DIR/bin/$cmd" "$BIN_DIR/$cmd"
    fi
done
ok "Commands installed: $COMMANDS"

# ── Ensure PATH ──────────────────────────────────────────────────────
ensure_path() {
    local rc="$1"
    local line='export PATH="$HOME/.local/bin:$PATH"'
    # Skip if .local/bin is already referenced
    if [[ -f "$rc" ]] && grep -q '\.local/bin' "$rc" 2>/dev/null; then
        return
    fi
    if [[ -f "$rc" ]]; then
        echo "" >> "$rc"
        echo "# Added by Krasis installer" >> "$rc"
        echo "$line" >> "$rc"
    fi
}

if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    current_shell="$(basename "${SHELL:-bash}")"
    case "$current_shell" in
        zsh)
            ensure_path "$HOME/.zshrc"
            ;;
        bash)
            ensure_path "$HOME/.bashrc"
            [[ -f "$HOME/.profile" ]] && ensure_path "$HOME/.profile"
            ;;
        *)
            ensure_path "$HOME/.bashrc"
            ensure_path "$HOME/.profile"
            ;;
    esac
    # Make available in THIS session immediately
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Verify ───────────────────────────────────────────────────────────
if command -v krasis &>/dev/null; then
    ok "Krasis $(krasis --version 2>/dev/null || echo "${VERSION#v}") is ready!"
    echo ""
    info "Next steps:"
    echo -e "  1. Run ${BOLD}krasis-setup${NC}   — installs CUDA toolkit, PyTorch, FlashInfer"
    echo -e "  2. Run ${BOLD}krasis${NC}         — launch the interactive TUI"
    echo ""
else
    ok "Installed Krasis ${VERSION#v}."
    echo ""
    info "To start using it in this terminal, run:"
    echo ""
    echo -e "  ${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo ""
    info "Then:"
    echo -e "  1. Run ${BOLD}krasis-setup${NC}   — installs CUDA toolkit, PyTorch, FlashInfer"
    echo -e "  2. Run ${BOLD}krasis${NC}         — launch the interactive TUI"
    echo ""
    info "(New terminals will pick up the PATH automatically.)"
    echo ""
fi
echo -e "${DIM}Upgrade:    curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash${NC}"
echo -e "${DIM}Uninstall:  curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- --uninstall${NC}"

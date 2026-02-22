"""Krasis Setup — one-command environment setup for CUDA + GPU dependencies.

Handles:
  1. CUDA toolkit installation (nvcc needed for FlashInfer JIT)
  2. CUDA-enabled PyTorch
  3. FlashInfer + sgl-kernel

Usage:
    krasis-setup          # auto-detect and install everything
    sudo krasis-setup     # if CUDA toolkit needs installing

The launcher calls lightweight checks on every start, but this script
handles the heavy one-time setup including system packages that need sudo.
"""

import os
import platform
import shutil
import subprocess
import sys

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def _run(cmd, check=True, **kwargs):
    """Run a command, printing it first."""
    print(f"  {DIM}$ {' '.join(cmd)}{NC}")
    return subprocess.run(cmd, check=check, **kwargs)


def _has_nvidia_gpu():
    """Check if an NVIDIA GPU is present."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.check_output(["nvidia-smi"], timeout=10, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _get_cuda_version_from_driver():
    """Detect CUDA toolkit version from nvidia-smi driver version."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().split("\n")[0]
        major = int(out.split(".")[0])
        if major >= 560:
            return "cu126", "12.6"
        elif major >= 545:
            return "cu124", "12.4"
        elif major >= 535:
            return "cu121", "12.1"
        elif major >= 525:
            return "cu118", "11.8"
    except Exception:
        pass
    return "cu126", "12.6"  # default


def _has_nvcc():
    """Check if nvcc is available (needed for FlashInfer JIT)."""
    if shutil.which("nvcc"):
        return True
    # Check common CUDA install locations
    for path in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]:
        if os.path.isfile(path):
            return True
    return False


def _detect_distro():
    """Detect Linux distro for package manager selection."""
    try:
        with open("/etc/os-release") as f:
            lines = f.read()
        if "Ubuntu" in lines or "Debian" in lines:
            return "debian"
        elif "Fedora" in lines or "Red Hat" in lines or "CentOS" in lines:
            return "rhel"
        elif "Arch" in lines:
            return "arch"
    except FileNotFoundError:
        pass
    return "unknown"


def _is_wsl():
    """Check if running in WSL."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def _install_cuda_toolkit():
    """Install CUDA toolkit (nvcc) for FlashInfer JIT compilation."""
    print(f"\n{BOLD}Step 1: CUDA Toolkit (nvcc){NC}")

    if _has_nvcc():
        print(f"  {GREEN}nvcc already installed.{NC}")
        return True

    print(f"  {YELLOW}nvcc not found — needed for FlashInfer JIT compilation.{NC}")
    print(f"  Installing CUDA toolkit (will ask for your password)...\n")

    distro = _detect_distro()
    # Use sudo in the commands — it will prompt for password if needed
    sudo = [] if os.geteuid() == 0 else ["sudo"]

    if distro == "debian":
        cmds = [
            sudo + ["apt-get", "update", "-qq"],
            sudo + ["apt-get", "install", "-y", "nvidia-cuda-toolkit"],
        ]
        for cmd in cmds:
            ret = _run(cmd, check=False)
            if ret.returncode != 0:
                print(f"  {RED}Failed. Try manually: sudo apt install nvidia-cuda-toolkit{NC}")
                return False
    elif distro == "rhel":
        ret = _run(sudo + ["dnf", "install", "-y", "cuda-toolkit"], check=False)
        if ret.returncode != 0:
            print(f"  {RED}Failed. Try manually: sudo dnf install cuda-toolkit{NC}")
            return False
    else:
        print(f"  {RED}Unknown distro. Install CUDA toolkit manually:{NC}")
        print(f"    sudo apt install nvidia-cuda-toolkit  (Debian/Ubuntu)")
        print(f"    sudo dnf install cuda-toolkit          (Fedora/RHEL)")
        return False

    if _has_nvcc():
        print(f"  {GREEN}CUDA toolkit installed successfully.{NC}")
        return True
    else:
        print(f"  {YELLOW}Package installed but nvcc not found on PATH.{NC}")
        print(f"  Try: export PATH=/usr/local/cuda/bin:$PATH")
        return False


def _install_cuda_torch():
    """Install CUDA-enabled PyTorch."""
    print(f"\n{BOLD}Step 2: CUDA PyTorch{NC}")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  {GREEN}CUDA torch already installed (v{torch.__version__}).{NC}")
            return True
        else:
            print(f"  torch {torch.__version__} installed but CUDA not available.")
    except ImportError:
        print(f"  torch not installed.")

    cu_tag, cu_ver = _get_cuda_version_from_driver()
    index_url = f"https://download.pytorch.org/whl/{cu_tag}"
    print(f"  Installing CUDA torch ({cu_tag})...")

    ret = _run([
        sys.executable, "-m", "pip", "install",
        "torch", "--index-url", index_url,
        "--quiet", "--no-warn-conflicts",
    ], check=False)

    if ret.returncode != 0:
        print(f"  {RED}Failed. Install manually:{NC}")
        print(f"    pip install torch --index-url {index_url}")
        return False

    print(f"  {GREEN}CUDA torch installed.{NC}")
    return True


def _install_gpu_packages():
    """Install FlashInfer and sgl-kernel."""
    print(f"\n{BOLD}Step 3: GPU Packages (FlashInfer, sgl-kernel){NC}")

    packages = [
        ("flashinfer-python", "flashinfer"),
        ("sgl-kernel", "sgl_kernel"),
    ]

    missing = []
    for pip_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"  {GREEN}{pip_name} already installed.{NC}")
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return True

    print(f"  Installing: {', '.join(missing)}...")
    ret = _run(
        [sys.executable, "-m", "pip", "install"] + missing + ["--quiet"],
        check=False,
    )

    if ret.returncode != 0:
        print(f"  {RED}Failed. Install manually:{NC}")
        print(f"    pip install {' '.join(missing)}")
        return False

    print(f"  {GREEN}GPU packages installed.{NC}")
    return True


def main():
    print(f"\n{BOLD}{CYAN}Krasis Setup{NC}")
    print(f"{DIM}{'─' * 50}{NC}\n")

    # Check platform
    if platform.system() != "Linux":
        print(f"{YELLOW}Krasis GPU support requires Linux (or WSL).{NC}")
        return

    # Check for NVIDIA GPU
    if not _has_nvidia_gpu():
        print(f"{YELLOW}No NVIDIA GPU detected. Krasis will run in CPU-only mode.{NC}")
        return

    print(f"NVIDIA GPU detected.")

    results = {}

    # Step 1: CUDA toolkit
    results["nvcc"] = _install_cuda_toolkit()

    # Step 2: CUDA torch
    results["torch"] = _install_cuda_torch()

    # Step 3: GPU packages
    results["packages"] = _install_gpu_packages()

    # Summary
    print(f"\n{DIM}{'─' * 50}{NC}")
    print(f"{BOLD}Summary:{NC}")
    for name, ok in results.items():
        if ok is True:
            print(f"  {GREEN}✓{NC} {name}")
        elif ok is False:
            print(f"  {RED}✗{NC} {name}")
        else:
            print(f"  {DIM}–{NC} {name} (skipped)")

    if all(v is True for v in results.values()):
        print(f"\n{GREEN}{BOLD}Setup complete! Run 'krasis' to start.{NC}\n")
    else:
        print(f"\n{YELLOW}Some steps failed. See above for details.{NC}\n")


if __name__ == "__main__":
    main()

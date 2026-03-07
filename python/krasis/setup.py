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


def _get_nvcc_version():
    """Get nvcc major.minor version, or None if not found."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        for path in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.8/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12.1/bin/nvcc",
            "/usr/local/cuda-11.8/bin/nvcc",
            "/usr/bin/nvcc",
        ]:
            if os.path.isfile(path):
                nvcc = path
                break
    if not nvcc:
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True, timeout=5)
        # Parse "release 12.8, V12.8.89" or similar
        for line in out.split("\n"):
            if "release" in line:
                # e.g. "Cuda compilation tools, release 12.8, V12.8.89"
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                major, minor = parts.split(".")[:2]
                return (int(major), int(minor))
    except Exception:
        pass
    return (0, 0)  # found but can't parse version


def _has_nvcc():
    """Check if nvcc is available (needed for FlashInfer JIT)."""
    return _get_nvcc_version() is not None


def _detect_distro():
    """Detect Linux distro for package manager selection.

    Returns (family, version_id) e.g. ("debian", "24.04") or ("rhel", "39").
    """
    try:
        with open("/etc/os-release") as f:
            lines = f.read()
        version_id = ""
        for line in lines.split("\n"):
            if line.startswith("VERSION_ID="):
                version_id = line.split("=", 1)[1].strip('"')
        if "Ubuntu" in lines or "Debian" in lines:
            return "debian", version_id
        elif "Fedora" in lines or "Red Hat" in lines or "CentOS" in lines:
            return "rhel", version_id
        elif "Arch" in lines:
            return "arch", version_id
    except FileNotFoundError:
        pass
    return "unknown", ""


def _is_wsl():
    """Check if running in WSL."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def _get_driver_cuda_version():
    """Get the CUDA version supported by the installed driver.

    Returns (major, minor) tuple from nvidia-smi, e.g. (12, 8).
    Returns None if it can't be determined.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        # Parse "CUDA Version: 12.8" from nvidia-smi header
        for line in out.split("\n"):
            if "CUDA Version:" in line:
                ver_str = line.split("CUDA Version:")[1].strip().split()[0]
                major, minor = ver_str.split(".")[:2]
                return (int(major), int(minor))
    except Exception:
        pass
    return None


def _get_required_cuda_version():
    """Determine which CUDA toolkit version to install.

    Prefers the driver's supported version (most likely to be in repos),
    falling back to the minimum required for this GPU's compute capability.
    Returns (major, minor) tuple, e.g. (12, 8).
    """
    # Prefer the driver's CUDA version — it's the most compatible and
    # most likely to exist in the NVIDIA package repos
    driver_ver = _get_driver_cuda_version()
    if driver_ver is not None:
        return driver_ver

    # Fallback: minimum version based on compute capability
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().split("\n")[0]
        major, minor = out.split(".")
        sm = (int(major), int(minor))
        if sm >= (12, 0):
            return (12, 8)
        if sm >= (10, 0):
            return (12, 6)
        return (12, 6)  # modern default
    except Exception:
        return (12, 6)  # safe default


def _need_python_dev():
    """Check if Python development headers are installed (needed for FlashInfer JIT)."""
    import sysconfig
    inc = sysconfig.get_path("include")
    if inc and os.path.isfile(os.path.join(inc, "Python.h")):
        return False
    return True


def _install_system_deps():
    """Install system packages: CUDA toolkit (nvcc), ninja, and python-dev headers."""
    print(f"\n{BOLD}Step 1: System Packages (nvcc, ninja, python-dev){NC}")

    need_ninja = not shutil.which("ninja") and not shutil.which("ninja-build")
    need_pydev = _need_python_dev()

    # Check nvcc: either missing or too old for this GPU
    nvcc_ver = _get_nvcc_version()
    required_ver = _get_required_cuda_version()
    need_nvcc = nvcc_ver is None
    nvcc_too_old = False
    if nvcc_ver is not None and nvcc_ver < required_ver:
        nvcc_too_old = True
        need_nvcc = True

    if not need_nvcc and not need_ninja and not need_pydev:
        print(f"  {GREEN}nvcc {nvcc_ver[0]}.{nvcc_ver[1]}, ninja, and python-dev already installed.{NC}")
        return True

    missing = []
    if need_nvcc:
        if nvcc_too_old:
            missing.append(f"nvcc (have {nvcc_ver[0]}.{nvcc_ver[1]}, need {required_ver[0]}.{required_ver[1]}+)")
        else:
            missing.append("nvcc")
    if need_ninja:
        missing.append("ninja")
    if need_pydev:
        missing.append("python-dev headers (Python.h)")
    print(f"  {YELLOW}{'Need upgrade' if nvcc_too_old else 'Missing'}: {', '.join(missing)}{NC}")
    print(f"  Installing (will ask for your password)...\n")

    distro, distro_ver = _detect_distro()
    is_wsl = _is_wsl()
    sudo = [] if os.geteuid() == 0 else ["sudo"]

    # Install ninja and python-dev headers (simple apt/dnf packages)
    apt_pkgs = []
    dnf_pkgs = []
    if need_ninja:
        apt_pkgs.append("ninja-build")
        dnf_pkgs.append("ninja-build")
    if need_pydev:
        py_ver_dot = f"{sys.version_info.major}.{sys.version_info.minor}"
        apt_pkgs.append(f"python{py_ver_dot}-dev")
        dnf_pkgs.append(f"python{sys.version_info.major}-devel")

    if apt_pkgs or dnf_pkgs:
        if distro == "debian":
            _run(sudo + ["apt-get", "update", "-qq"], check=False)
            ret = _run(sudo + ["apt-get", "install", "-y"] + apt_pkgs, check=False)
        elif distro == "rhel":
            ret = _run(sudo + ["dnf", "install", "-y"] + dnf_pkgs, check=False)
        else:
            ret = type("R", (), {"returncode": 1})()
        if ret.returncode != 0:
            print(f"  {RED}Failed to install {', '.join(apt_pkgs or dnf_pkgs)}.{NC}")

    # Install CUDA toolkit from NVIDIA repo (not the ancient distro package)
    if need_nvcc and distro == "debian":
        cuda_pkg = f"cuda-toolkit-{required_ver[0]}-{required_ver[1]}"
        if is_wsl:
            repo = "wsl-ubuntu"
        elif distro_ver.startswith("24."):
            repo = "ubuntu2404"
        else:
            repo = "ubuntu2204"  # works for 22.04 and most Debian
        keyring_url = (
            f"https://developer.download.nvidia.com/compute/cuda/repos/"
            f"{repo}/x86_64/cuda-keyring_1.1-1_all.deb"
        )
        print(f"  Adding NVIDIA CUDA {required_ver[0]}.{required_ver[1]} repository...")

        # Remove old distro nvcc if present
        if nvcc_too_old:
            _run(sudo + ["apt-get", "remove", "-y", "nvidia-cuda-toolkit"], check=False)

        # Download and install keyring
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".deb", delete=False) as tmp:
            keyring_path = tmp.name
        ret = _run(["wget", "-q", "-O", keyring_path, keyring_url], check=False)
        if ret.returncode != 0:
            # Try curl as fallback
            ret = _run(["curl", "-sL", "-o", keyring_path, keyring_url], check=False)
        cuda_installed = False
        if ret.returncode != 0:
            print(f"  {YELLOW}Could not download CUDA keyring, checking if nvcc is already available...{NC}")
        else:
            cmds = [
                sudo + ["dpkg", "-i", keyring_path],
                sudo + ["apt-get", "update", "-qq"],
            ]
            repo_ok = True
            for cmd in cmds:
                ret = _run(cmd, check=False)
                if ret.returncode != 0:
                    repo_ok = False
                    break

            if repo_ok:
                # Try exact version first, fall back to major meta-package
                ret = _run(sudo + ["apt-get", "install", "-y", cuda_pkg], check=False)
                if ret.returncode != 0:
                    fallback_pkg = f"cuda-toolkit-{required_ver[0]}"
                    print(f"  {YELLOW}{cuda_pkg} not found, trying {fallback_pkg}...{NC}")
                    ret = _run(sudo + ["apt-get", "install", "-y", fallback_pkg], check=False)
                if ret.returncode == 0:
                    cuda_installed = True
            else:
                print(f"  {YELLOW}Could not set up CUDA repo, checking if nvcc is already available...{NC}")

            # Clean up keyring
            try:
                os.unlink(keyring_path)
            except OSError:
                pass

        # Add to PATH if needed
        cuda_bin = f"/usr/local/cuda-{required_ver[0]}.{required_ver[1]}/bin"
        if not os.path.isdir(cuda_bin):
            cuda_bin = "/usr/local/cuda/bin"
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
            print(f"  Added {cuda_bin} to PATH for this session.")
        # Set CUDA_HOME so FlashInfer/PyTorch JIT find the right toolkit
        cuda_home = os.path.dirname(cuda_bin)
        if os.path.isdir(cuda_home):
            os.environ["CUDA_HOME"] = cuda_home
            # Add to bashrc so it persists
            bashrc = os.path.expanduser("~/.bashrc")
            export_line = f'export PATH={cuda_bin}:$PATH'
            export_cuda_home = f'export CUDA_HOME={cuda_home}'
            try:
                existing = ""
                if os.path.isfile(bashrc):
                    with open(bashrc) as f:
                        existing = f.read()
                lines_to_add = []
                if export_line not in existing:
                    lines_to_add.append(export_line)
                if export_cuda_home not in existing:
                    lines_to_add.append(export_cuda_home)
                if lines_to_add:
                    with open(bashrc, "a") as f:
                        f.write(f"\n# Added by krasis-setup\n")
                        for line in lines_to_add:
                            f.write(f"{line}\n")
                    print(f"  Added to ~/.bashrc (will persist across sessions).")
            except OSError:
                print(f"  {DIM}Add to ~/.bashrc manually: {export_line}{NC}")

    elif need_nvcc and distro == "rhel":
        cuda_pkg = f"cuda-toolkit-{required_ver[0]}-{required_ver[1]}"
        ret = _run(sudo + ["dnf", "install", "-y", cuda_pkg], check=False)
        if ret.returncode != 0:
            fallback_pkg = f"cuda-toolkit-{required_ver[0]}"
            print(f"  {YELLOW}{cuda_pkg} not found, trying {fallback_pkg}...{NC}")
            ret = _run(sudo + ["dnf", "install", "-y", fallback_pkg], check=False)
            if ret.returncode != 0:
                print(f"  {YELLOW}Could not install {cuda_pkg} via dnf, checking if nvcc is already available...{NC}")

        # Add to PATH and set CUDA_HOME (same as debian path)
        cuda_bin = f"/usr/local/cuda-{required_ver[0]}.{required_ver[1]}/bin"
        if not os.path.isdir(cuda_bin):
            cuda_bin = "/usr/local/cuda/bin"
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
            print(f"  Added {cuda_bin} to PATH for this session.")
        cuda_home = os.path.dirname(cuda_bin)
        if os.path.isdir(cuda_home):
            os.environ["CUDA_HOME"] = cuda_home
            bashrc = os.path.expanduser("~/.bashrc")
            export_line = f'export PATH={cuda_bin}:$PATH'
            export_cuda_home = f'export CUDA_HOME={cuda_home}'
            try:
                existing = ""
                if os.path.isfile(bashrc):
                    with open(bashrc) as f:
                        existing = f.read()
                lines_to_add = []
                if export_line not in existing:
                    lines_to_add.append(export_line)
                if export_cuda_home not in existing:
                    lines_to_add.append(export_cuda_home)
                if lines_to_add:
                    with open(bashrc, "a") as f:
                        f.write(f"\n# Added by krasis-setup\n")
                        for line in lines_to_add:
                            f.write(f"{line}\n")
                    print(f"  Added to ~/.bashrc (will persist across sessions).")
            except OSError:
                print(f"  {DIM}Add to ~/.bashrc manually: {export_line}{NC}")
    elif need_nvcc:
        print(f"  {YELLOW}Unknown distro, checking if nvcc is already available...{NC}")

    # Verify
    ok = True
    new_nvcc = _get_nvcc_version()
    if need_nvcc and (new_nvcc is None or new_nvcc < required_ver):
        if new_nvcc:
            print(f"  {YELLOW}nvcc {new_nvcc[0]}.{new_nvcc[1]} still too old (need {required_ver[0]}.{required_ver[1]}+).{NC}")
        else:
            print(f"  {YELLOW}nvcc still not found on PATH.{NC}")
        print(f"  Try: export PATH=/usr/local/cuda/bin:$PATH")
        ok = False
    elif need_nvcc and new_nvcc:
        print(f"  {GREEN}nvcc {new_nvcc[0]}.{new_nvcc[1]} installed.{NC}")
    if need_ninja and not shutil.which("ninja") and not shutil.which("ninja-build"):
        print(f"  {YELLOW}ninja still not found on PATH.{NC}")
        ok = False
    if need_pydev and _need_python_dev():
        print(f"  {YELLOW}Python.h still not found. Install python3-dev manually.{NC}")
        ok = False
    if ok:
        print(f"  {GREEN}System packages installed successfully.{NC}")
    return ok


def _install_cuda_torch():
    """Install CUDA-enabled PyTorch."""
    print(f"\n{BOLD}Step 2: CUDA PyTorch{NC}")

    need_reinstall = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  {GREEN}CUDA torch already installed (v{torch.__version__}).{NC}")
            return True
        else:
            print(f"  torch {torch.__version__} installed but CUDA not available.")
            need_reinstall = True
    except ImportError:
        print(f"  torch not installed.")

    cu_tag, cu_ver = _get_cuda_version_from_driver()
    index_url = f"https://download.pytorch.org/whl/{cu_tag}"
    print(f"  Installing CUDA torch ({cu_tag})...")

    pip_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "--index-url", index_url,
        "--quiet", "--no-warn-conflicts",
    ]
    if need_reinstall:
        pip_cmd.append("--force-reinstall")
    ret = _run(pip_cmd, check=False)

    if ret.returncode != 0:
        print(f"  {RED}Failed. Install manually:{NC}")
        print(f"    pip install torch --index-url {index_url}")
        return False

    print(f"  {GREEN}CUDA torch installed.{NC}")
    return True


def _install_gpu_packages():
    """Install FlashInfer, sgl-kernel, and sglang."""
    print(f"\n{BOLD}Step 3: GPU Packages (FlashInfer, sgl-kernel, sglang){NC}")

    packages = [
        ("flashinfer-python", "flashinfer"),
        ("sgl-kernel", "sgl_kernel"),
        ("sglang", "sglang"),
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


def _install_session_deps():
    """Optionally install Node.js and Bun for Session messenger integration."""
    print(f"\n{BOLD}Step 4: Session Messenger (optional){NC}")

    # Ask user
    print(f"  Session messenger lets you chat with your Krasis server")
    print(f"  from the Session app on any device.")
    print(f"  Requires Node.js and Bun (will be installed if missing).\n")
    try:
        answer = input(f"  Set up Session messenger support? [y/N] ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if not answer.lower().startswith("y"):
        print(f"  {DIM}Skipped. You can re-run krasis-setup later to add it.{NC}")
        return None  # skipped

    sudo = [] if os.geteuid() == 0 else ["sudo"]
    distro, _ = _detect_distro()
    ok = True

    # Install Node.js if missing
    if not shutil.which("node"):
        print(f"  {YELLOW}Node.js not found. Installing...{NC}")
        if distro == "debian":
            _run(sudo + ["apt-get", "update", "-qq"], check=False)
            ret = _run(sudo + ["apt-get", "install", "-y", "nodejs", "npm"], check=False)
        elif distro == "rhel":
            ret = _run(sudo + ["dnf", "install", "-y", "nodejs", "npm"], check=False)
        else:
            ret = type("R", (), {"returncode": 1})()
        if ret.returncode != 0:
            print(f"  {RED}Failed to install Node.js. Install manually: https://nodejs.org{NC}")
            ok = False
        elif shutil.which("node"):
            print(f"  {GREEN}Node.js installed.{NC}")
        else:
            print(f"  {YELLOW}Node.js still not found on PATH.{NC}")
            ok = False
    else:
        print(f"  {GREEN}Node.js already installed.{NC}")

    # Install Bun if missing
    if not shutil.which("bun"):
        print(f"  Installing Bun...")
        # Bun's official installer doesn't need sudo — installs to ~/.bun
        ret = _run(["bash", "-c", "curl -fsSL https://bun.sh/install | bash"], check=False)
        if ret.returncode != 0:
            print(f"  {RED}Failed to install Bun. Install manually: https://bun.sh{NC}")
            ok = False
        else:
            # Add bun to PATH for this session
            bun_bin = os.path.expanduser("~/.bun/bin")
            if os.path.isdir(bun_bin) and bun_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bun_bin + ":" + os.environ.get("PATH", "")
            if shutil.which("bun"):
                print(f"  {GREEN}Bun installed.{NC}")
            else:
                print(f"  {YELLOW}Bun installed but not on PATH. Restart your terminal.{NC}")
                ok = False
    else:
        print(f"  {GREEN}Bun already installed.{NC}")

    if ok:
        print(f"  {GREEN}Session messenger dependencies ready.{NC}")
        print(f"  Enable it in the Krasis TUI (Options > Session messenger)")
        print(f"  or with --session-enabled on the command line.")
    return ok


def main():
    print(f"\n{BOLD}{CYAN}Krasis Setup{NC}")
    print(f"{DIM}{'─' * 50}{NC}\n")

    # Check platform
    if platform.system() != "Linux":
        print(f"{YELLOW}Krasis GPU support requires Linux (or WSL).{NC}")
        return

    # Check for NVIDIA GPU — Krasis requires GPU for prefill
    if not _has_nvidia_gpu():
        print(f"{RED}No NVIDIA GPU detected. Krasis requires at least one NVIDIA GPU.{NC}")
        print(f"  If you have an NVIDIA GPU, install the driver first:")
        print(f"    sudo apt install nvidia-driver-560  # or newer")
        sys.exit(1)

    print(f"NVIDIA GPU detected.")

    results = {}

    # Step 1: CUDA toolkit
    results["system"] = _install_system_deps()

    # Step 2: CUDA torch
    results["torch"] = _install_cuda_torch()

    # Step 3: GPU packages
    results["packages"] = _install_gpu_packages()

    # Step 4: Session messenger (optional)
    results["session"] = _install_session_deps()

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

    required = {k: v for k, v in results.items() if k != "session"}
    if all(v is True for v in required.values()):
        print(f"\n{GREEN}{BOLD}Setup complete! Run 'krasis' to start.{NC}\n")
    else:
        print(f"\n{YELLOW}Some steps failed. See above for details.{NC}\n")


if __name__ == "__main__":
    main()

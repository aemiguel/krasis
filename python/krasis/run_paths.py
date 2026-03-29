import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logs_root() -> Path:
    root = repo_root() / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "run"


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = Path(f"{path}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def create_run_dir(run_type: str, parent: Optional[os.PathLike] = None) -> Path:
    parent_path = Path(parent) if parent is not None else logs_root()
    parent_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _unique_dir(parent_path / f"{slugify(run_type)}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_run_dir(default_type: str = "run") -> Path:
    existing = os.environ.get("KRASIS_RUN_DIR", "").strip()
    if existing:
        run_dir = Path(existing)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    run_type = os.environ.get("KRASIS_RUN_TYPE", "").strip() or default_type
    run_dir = create_run_dir(run_type)
    os.environ["KRASIS_RUN_DIR"] = str(run_dir)
    os.environ.setdefault("KRASIS_RUN_TYPE", run_type)
    return run_dir


def get_run_file(filename: str, default_type: str = "run") -> Path:
    path = get_run_dir(default_type) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

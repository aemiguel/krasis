#!/usr/bin/env python3

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import zipfile


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def find_record_name(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if name.endswith(".dist-info/RECORD"):
            return name
    raise RuntimeError("wheel has no RECORD")


def inject_sidecars(wheel_path: Path, sidecars: list[tuple[Path, str]]) -> None:
    with tempfile.TemporaryDirectory(dir=wheel_path.parent) as tmpdir:
        tmp_path = Path(tmpdir) / wheel_path.name
        with zipfile.ZipFile(wheel_path, "r") as src:
            record_name = find_record_name(src)
            record_rows = []
            record_seen = False
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    data = src.read(info.filename)
                    if info.filename == record_name:
                        reader = csv.reader(io.StringIO(data.decode("utf-8")))
                        for row in reader:
                            if row and not any(row[0] == arcname for _, arcname in sidecars):
                                record_rows.append(row)
                        record_seen = True
                        continue
                    dst.writestr(info, data)

                if not record_seen:
                    raise RuntimeError(f"{wheel_path} missing RECORD entry")

                for sidecar_path, arcname in sidecars:
                    data = sidecar_path.read_bytes()
                    dst.writestr(arcname, data)
                    record_rows.append([arcname, record_hash(data), str(len(data))])

                output = io.StringIO()
                writer = csv.writer(output, lineterminator="\n")
                for row in record_rows:
                    writer.writerow(row)
                writer.writerow([record_name, "", ""])
                dst.writestr(record_name, output.getvalue().encode("utf-8"))

        tmp_path.replace(wheel_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", required=True)
    parser.add_argument("--marlin")
    parser.add_argument("--flash-attn")
    parser.add_argument("--fla-dir",
                        help="Directory containing libkrasis_fla_sm*.so files")
    args = parser.parse_args()

    sidecars: list[tuple[Path, str]] = []
    if args.marlin:
        sidecars.append((Path(args.marlin), "krasis/libkrasis_marlin.so"))
    if args.flash_attn:
        sidecars.append((Path(args.flash_attn), "krasis/libkrasis_flash_attn.so"))
    if args.fla_dir:
        fla_dir = Path(args.fla_dir)
        for fla_so in sorted(fla_dir.glob("libkrasis_fla_sm*.so")):
            sidecars.append((fla_so, f"krasis/{fla_so.name}"))
    if not sidecars:
        raise SystemExit("no sidecars provided")

    for sidecar_path, _ in sidecars:
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)

    wheel_dir = Path(args.wheel_dir)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found in {wheel_dir}")

    for wheel_path in wheels:
        inject_sidecars(wheel_path, sidecars)
        print(f"Injected sidecars into {wheel_path}")


if __name__ == "__main__":
    main()

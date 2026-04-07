#!/usr/bin/env python3
"""Build-time FLA kernel compiler.

Compiles the vendored FLA (Flash Linear Attention) Triton kernels for the Gated
DeltaNet algorithm used by QCN's 36 linear attention layers.

This script:
1. Imports the vendored FLA code
2. Runs chunk_gated_delta_rule_fwd with representative inputs to trigger Triton JIT
3. Extracts compiled cubins from the Triton cache
4. Extracts kernel metadata (arg types, shared mem, num_warps, function name)
5. Generates a C wrapper source file (fla_vendor.cu) with:
   - Embedded cubin data as byte arrays
   - cuModuleLoadData + cuModuleGetFunction init
   - Extern "C" wrapper functions for each sub-kernel
6. Compiles fla_vendor.cu into libkrasis_fla.so

Usage:
    python compile_kernels.py --output-dir <path> [--arch sm_120]

The resulting libkrasis_fla.so is loaded by Rust via dlopen at runtime.
No Python, Triton, or torch dependencies at runtime.

Must be run with: KRASIS_DEV_SCRIPT=1
"""

import argparse
import json
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

# Guard: only run via ./dev build
if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print("ERROR: This script must be run via ./dev build, not directly.", file=sys.stderr)
    sys.exit(1)


def find_all_triton_cache_entries(kernel_name: str, arch: int) -> list[dict]:
    """Find ALL compiled Triton variants for a kernel name and architecture.

    Returns a list of cache entries, one per unique constexpr configuration.
    Each entry has: meta, cubin_path, source_path, name, n_args, entry_dir.
    """
    cache_dir = Path.home() / ".triton" / "cache"
    if not cache_dir.exists():
        return []

    candidates = []
    for entry_dir in cache_dir.iterdir():
        if not entry_dir.is_dir():
            continue
        json_path = entry_dir / f"{kernel_name}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("target", {}).get("arch") == arch:
            cubin_path = entry_dir / f"{kernel_name}.cubin"
            source_path = entry_dir / f"{kernel_name}.source"
            if cubin_path.exists():
                n_args = 0
                if source_path.exists():
                    args = extract_kernel_args(str(source_path))
                    n_args = len(args)
                candidates.append({
                    "meta": meta,
                    "cubin_path": str(cubin_path),
                    "source_path": str(source_path) if source_path.exists() else None,
                    "name": kernel_name,
                    "n_args": n_args,
                    "entry_dir": str(entry_dir),
                })
    return candidates


def find_triton_cache_entry(kernel_name: str, arch: int) -> dict | None:
    """Find a compiled Triton kernel in the cache by name and architecture.

    When multiple compiled variants exist (different constexpr configurations),
    selects the one with the MOST runtime parameters. This ensures we get
    the variant with all optional features enabled (e.g., USE_INITIAL_STATE=True,
    STORE_FINAL_STATE=True) rather than a stripped-down version.
    """
    cache_dir = Path.home() / ".triton" / "cache"
    if not cache_dir.exists():
        return None

    candidates = []
    for entry_dir in cache_dir.iterdir():
        if not entry_dir.is_dir():
            continue
        json_path = entry_dir / f"{kernel_name}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("target", {}).get("arch") == arch:
            cubin_path = entry_dir / f"{kernel_name}.cubin"
            source_path = entry_dir / f"{kernel_name}.source"
            if cubin_path.exists():
                # Count runtime args from TTIR to rank variants
                n_args = 0
                if source_path.exists():
                    args = extract_kernel_args(str(source_path))
                    n_args = len(args)
                candidates.append({
                    "meta": meta,
                    "cubin_path": str(cubin_path),
                    "source_path": str(source_path) if source_path.exists() else None,
                    "name": kernel_name,
                    "n_args": n_args,
                })

    if not candidates:
        return None

    # Pick the variant with the most runtime arguments (most features enabled)
    best = max(candidates, key=lambda c: c["n_args"])
    if len(candidates) > 1:
        print(f"  {kernel_name}: {len(candidates)} variants, selected {best['n_args']}-arg variant")
    return best


def extract_kernel_args(source_path: str) -> list[tuple[str, str]]:
    """Extract kernel argument names and types from Triton TTIR source.

    Returns list of (name, c_type) tuples.
    """
    if not source_path or not os.path.exists(source_path):
        return []

    with open(source_path) as f:
        source = f.read()

    # Parse the TTIR function signature
    # Format: @kernel_name(%name: !tt.ptr<bf16> ..., %name2: i32 ...)
    # Find the function definition line
    args = []
    func_match = re.search(r'tt\.func public @\w+\((.+?)\)\s+attributes', source, re.DOTALL)
    if not func_match:
        return []

    params_str = func_match.group(1)
    # Each param: %name: type {attrs} loc(...)
    for param in re.finditer(r'%(\w+):\s+(!tt\.ptr<[^>]+>|i32|i64|f32|f64)', params_str):
        name = param.group(1)
        tt_type = param.group(2)

        if tt_type.startswith('!tt.ptr'):
            c_type = "CUdeviceptr"
        elif tt_type == 'i32':
            c_type = "int32_t"
        elif tt_type == 'i64':
            c_type = "int64_t"
        elif tt_type == 'f32':
            c_type = "float"
        elif tt_type == 'f64':
            c_type = "double"
        else:
            c_type = "void*"  # fallback

        args.append((name, c_type))

    return args


def generate_c_wrapper(kernels: list[dict], output_path: str):
    """Generate a C wrapper source file with embedded cubins and launch functions.

    Uses assembler .incbin to embed cubin files directly from disk, avoiding
    nvcc issues with huge C byte arrays (38K+ lines). The incbin approach produces
    identical binary output but compiles in seconds instead of minutes.
    """

    lines = []
    lines.append("// Auto-generated by compile_kernels.py — DO NOT EDIT")
    lines.append("// FLA (Flash Linear Attention) vendored kernel wrappers")
    lines.append("// Gated DeltaNet forward pass for Krasis prefill")
    lines.append("")
    lines.append('#include <cuda.h>')
    lines.append('#include <stdint.h>')
    lines.append('#include <stdio.h>')
    lines.append("")

    # Embed cubin data via assembler .incbin (avoids nvcc large-array issues)
    for k in kernels:
        sn = k['safe_name']
        cubin_path = k["cubin_path"]
        import os
        cubin_size = os.path.getsize(cubin_path)
        lines.append(f"// {k['name']} ({cubin_size} bytes)")
        lines.append(f'__asm__(".section .rodata\\n"')
        lines.append(f'        ".global cubin_{sn}\\n"')
        lines.append(f'        ".global cubin_{sn}_end\\n"')
        lines.append(f'        "cubin_{sn}:\\n"')
        lines.append(f'        ".incbin \\"{cubin_path}\\"\\n"')
        lines.append(f'        "cubin_{sn}_end:\\n"')
        lines.append(f'        ".previous\\n");')
        lines.append(f"extern const unsigned char cubin_{sn}[];")
        lines.append(f"extern const unsigned char cubin_{sn}_end[];")
        lines.append("")

    # Module and function handles
    for k in kernels:
        sn = k['safe_name']
        lines.append(f"static CUmodule module_{sn} = NULL;")
        lines.append(f"static CUfunction func_{sn} = NULL;")
    lines.append("")

    # Init function
    lines.append('extern "C" int krasis_fla_init(void) {')
    lines.append("    CUresult res;")
    for k in kernels:
        sn = k['safe_name']
        shared_mem = k['meta'].get('shared', 0)
        lines.append(f'    res = cuModuleLoadData(&module_{sn}, cubin_{sn});')
        lines.append(f'    if (res != CUDA_SUCCESS) {{ fprintf(stderr, "FLA: failed to load module {sn}: %d\\n", res); return -1; }}')
        lines.append(f'    res = cuModuleGetFunction(&func_{sn}, module_{sn}, "{k["name"]}");')
        lines.append(f'    if (res != CUDA_SUCCESS) {{ fprintf(stderr, "FLA: failed to get function {sn}: %d\\n", res); return -1; }}')
        # Triton kernels that use >48KB shared memory need this attribute set
        if shared_mem > 49152:
            lines.append(f'    res = cuFuncSetAttribute(func_{sn}, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {shared_mem});')
            lines.append(f'    if (res != CUDA_SUCCESS) {{ fprintf(stderr, "FLA: failed to set shared mem for {sn}: %d\\n", res); return -1; }}')
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    # Cleanup function
    lines.append('extern "C" void krasis_fla_cleanup(void) {')
    for k in kernels:
        sn = k['safe_name']
        lines.append(f'    if (module_{sn}) {{ cuModuleUnload(module_{sn}); module_{sn} = NULL; }}')
    lines.append("}")
    lines.append("")

    # Per-kernel launch wrappers
    # NOTE: Triton 3.5+ compiled cubins expect two hidden trailing arguments:
    #   global_scratch (CUdeviceptr) - used for global scratch memory (0 = none)
    #   profile_scratch (CUdeviceptr) - used for profiling (0 = none)
    # These must be included in kernel_args or cuLaunchKernel returns INVALID_VALUE.
    for k in kernels:
        sn = k['safe_name']
        args = k['args']
        meta = k['meta']
        num_warps = meta.get('num_warps', 4)
        shared_mem = meta.get('shared', 0)

        # Build C function signature
        c_args = []
        for name, ctype in args:
            c_args.append(f"    {ctype} {name}")
        c_args.append("    unsigned int grid_x")
        c_args.append("    unsigned int grid_y")
        c_args.append("    unsigned int grid_z")
        c_args.append("    CUstream stream")

        sig = ",\n".join(c_args)
        lines.append(f'extern "C" CUresult krasis_fla_{sn}(')
        lines.append(sig)
        lines.append(") {")

        # Build args array — include Triton hidden args (global_scratch, profile_scratch)
        lines.append(f"    CUdeviceptr global_scratch = 0;")
        lines.append(f"    CUdeviceptr profile_scratch = 0;")
        arg_names = [name for name, _ in args]
        lines.append(f"    void* kernel_args[] = {{")
        for name in arg_names:
            lines.append(f"        &{name},")
        lines.append(f"        &global_scratch,")
        lines.append(f"        &profile_scratch,")
        lines.append("    };")
        lines.append(f"    return cuLaunchKernel(func_{sn},")
        lines.append(f"        grid_x, grid_y, grid_z,")
        lines.append(f"        {num_warps * 32}, 1, 1,")
        lines.append(f"        {shared_mem},")
        lines.append(f"        stream,")
        lines.append(f"        kernel_args, NULL);")
        lines.append("}")
        lines.append("")

    # Write output
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated {output_path} ({len(lines)} lines)")


def main():
    parser = argparse.ArgumentParser(description="Compile FLA Triton kernels to C wrapper")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated files")
    parser.add_argument("--arch", type=int, default=None, help="GPU architecture (e.g. 120 for sm_120)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect GPU arch if not specified
    if args.arch is None:
        import torch
        cc = torch.cuda.get_device_capability(0)
        arch = cc[0] * 10 + cc[1]  # e.g. (12, 0) -> 120
        print(f"Detected GPU arch: sm_{arch}")
    else:
        arch = args.arch
        print(f"Using specified GPU arch: sm_{arch}")

    # Step 1: Import vendored FLA and trigger JIT compilation
    fla_dir = Path(__file__).parent.parent.parent / "cuda"
    sys.path.insert(0, str(fla_dir))

    print("Importing vendored FLA...")
    import torch
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    # Compile FLA kernels for all supported head counts.
    # H is a tl.constexpr in every FLA kernel (used for stride computation),
    # so we need separate cubins for each distinct num_v_heads value.
    # Known models: QCN/Q35B use H=32, Qwen3.5-122B uses H=64.
    h_values = [32, 64]
    device = "cuda:0"
    dtype = torch.bfloat16
    B, T, K, V = 1, 1024, 128, 128

    for H in h_values:
        print(f"Running FLA forward with H={H} to trigger Triton JIT compilation...")
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(B, T, H, dtype=dtype, device=device))
        beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
        initial_state = torch.zeros(B, H, K, V, dtype=dtype, device=device)
        scale = K ** -0.5

        chunk_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, initial_state=initial_state,
            output_final_state=True,
        )
        torch.cuda.synchronize()
        print(f"  H={H} JIT compilation complete.")
    print("All FLA kernel variants compiled.")

    # Step 2: Find compiled kernels in Triton cache, grouped by H value.
    # Strategy: find ALL FLA kernel names, then for each kernel, find all
    # variants and group by their cache directory (each H produces a different cubin).
    # We match variants to H values by tracking which directories appeared
    # after each H compilation.
    cache_dir = Path.home() / ".triton" / "cache"
    all_fla_kernels = set()
    if cache_dir.exists():
        for entry_dir in cache_dir.iterdir():
            if not entry_dir.is_dir():
                continue
            for f in entry_dir.iterdir():
                if f.suffix == '.json' and not f.name.startswith('__grp__'):
                    name = f.stem
                    with open(f) as fp:
                        meta = json.load(fp)
                    if meta.get("target", {}).get("arch") == arch:
                        source_file = entry_dir / f"{name}.source"
                        if source_file.exists():
                            src_text = source_file.read_text()[:200]
                            if 'fla/ops' in src_text:
                                all_fla_kernels.add(name)

    print(f"Found {len(all_fla_kernels)} FLA kernel names for arch {arch}:")
    for name in sorted(all_fla_kernels):
        print(f"  {name}")

    # Step 3: For each kernel, find all variants and group by H.
    # We distinguish H by cubin content: different H values produce different cubins.
    # Since we compiled exactly h_values=[32,64], we pick the best variant for each H
    # by matching cubin size patterns (H=32 and H=64 produce cubins of different sizes).
    kernels = []  # list of dicts with "h_tag" field
    for name in sorted(all_fla_kernels):
        variants = find_all_triton_cache_entries(name, arch)
        if not variants:
            print(f"  WARNING: no cache entries for {name}")
            continue

        # Group by cubin content (hash). Different H → different cubin.
        by_cubin = {}
        for v in variants:
            cubin_size = os.path.getsize(v["cubin_path"])
            key = (cubin_size, v["n_args"])
            if key not in by_cubin:
                by_cubin[key] = []
            by_cubin[key].append(v)

        # For each unique cubin, pick the best variant (most args) and tag with H.
        # We expect exactly len(h_values) distinct cubins per kernel.
        unique_variants = []
        for key, group in by_cubin.items():
            best = max(group, key=lambda c: c["n_args"])
            unique_variants.append(best)

        # Filter to only keep variants with the most runtime args.
        # The Triton cache may contain stale entries from previous compilations
        # with different constexpr configurations (e.g., USE_INITIAL_STATE=False,
        # STORE_FINAL_STATE=False). Those produce cubins with fewer runtime args
        # (missing h0/ht pointers). We always want the fully-featured variant
        # that matches our compilation settings (initial_state=non-None,
        # output_final_state=True).
        max_n_args = max(v["n_args"] for v in unique_variants)
        unique_variants = [v for v in unique_variants if v["n_args"] == max_n_args]

        if len(unique_variants) == 1:
            # Only one variant — shared across all H values (e.g. l2norm_fwd_kernel
            # which doesn't use H). Generate one wrapper without H suffix.
            v = unique_variants[0]
            args = extract_kernel_args(v.get("source_path"))
            safe_name = name.replace("-", "_")
            kernels.append({
                "name": name,
                "safe_name": safe_name,
                "meta": v["meta"],
                "cubin_path": v["cubin_path"],
                "args": args,
                "h_tag": None,  # no H suffix needed
            })
            cubin_size = os.path.getsize(v["cubin_path"])
            print(f"  {name}: {len(args)} args, {cubin_size}B (H-independent)")
        else:
            if len(unique_variants) != len(h_values):
                print(f"  WARNING: {name}: expected {len(h_values)} variants after "
                      f"filtering, got {len(unique_variants)}")
            # Multiple variants — one per H. Sort by cubin size (smaller = fewer heads).
            unique_variants.sort(key=lambda v: os.path.getsize(v["cubin_path"]))
            for idx, (h_val, v) in enumerate(zip(sorted(h_values), unique_variants)):
                args = extract_kernel_args(v.get("source_path"))
                safe_name = f"{name.replace('-', '_')}_h{h_val}"
                kernels.append({
                    "name": name,
                    "safe_name": safe_name,
                    "meta": v["meta"],
                    "cubin_path": v["cubin_path"],
                    "args": args,
                    "h_tag": h_val,
                })
                cubin_size = os.path.getsize(v["cubin_path"])
                print(f"  {name} (H={h_val}): {len(args)} args, {cubin_size}B, "
                      f"warps={v['meta'].get('num_warps')}")

    if not kernels:
        print("ERROR: No FLA kernels found in Triton cache!", file=sys.stderr)
        sys.exit(1)

    # Step 4: Generate C wrapper
    c_path = output_dir / "fla_vendor.cu"
    generate_c_wrapper(kernels, str(c_path))

    # Step 5: Save individual cubins
    cubin_dir = output_dir / "cubins"
    cubin_dir.mkdir(exist_ok=True)
    for k in kernels:
        import shutil
        dst = cubin_dir / f"{k['safe_name']}_sm{arch}.cubin"
        shutil.copy2(k["cubin_path"], str(dst))
        print(f"  Saved {dst.name}")

    # Step 6: Compile fla_vendor.cu to libkrasis_fla.so
    import subprocess
    nvcc = "nvcc"
    for p in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]:
        if os.path.exists(p):
            nvcc = p
            break

    so_path = output_dir / "libkrasis_fla.so"
    compile_cmd = [
        nvcc, "-shared",
        "-o", str(so_path),
        "-Xcompiler", "-fPIC",
        "-Wno-deprecated-gpu-targets",
        str(c_path),
        "-lcuda",
    ]
    print(f"\nCompiling: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: nvcc failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"  Compiled {so_path.name} ({so_path.stat().st_size} bytes)")

    print(f"\nDone. Generated files in {output_dir}:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

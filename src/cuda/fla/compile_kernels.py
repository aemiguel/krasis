#!/usr/bin/env python3
"""Build-time FLA kernel compiler with cross-compilation.

Compiles the vendored FLA (Flash Linear Attention) Triton kernels for the Gated
DeltaNet algorithm used by linear attention layers during prefill.

This script uses triton.compile() directly with GPUTarget to cross-compile
cubins for ANY GPU architecture from ANY machine. No GPU of the target
architecture needs to be present.

It produces one libkrasis_fla_sm{arch}.so per target architecture, each
containing all model-dimension variants (H=32, H=64, etc.) as separate
symbols. At runtime, Rust detects the GPU architecture and dlopen's the
matching .so.

Usage:
    python compile_kernels.py --output-dir <path> [--arch sm_120 ...]

Must be run with: KRASIS_DEV_SCRIPT=1
"""

import argparse
import contextlib
import inspect
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

from packaging import version

# Guard: only run via ./dev build
if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print("ERROR: This script must be run via ./dev build, not directly.", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
#  Target GPU architectures
# ═══════════════════════════════════════════════════════════════════
# These cover all NVIDIA GPUs with >= 16GB VRAM capable of running
# large language models.
TARGET_ARCHS = [80, 89, 90, 120]
#  80 = Ampere (A100, A6000, RTX 3090) — also covers sm_86/87 via PTX fallback
#  89 = Ada Lovelace (RTX 4090, L40, RTX 2000 Ada)
#  90 = Hopper (H100, H200)
# 120 = Blackwell (RTX 5090, B200)

# ═══════════════════════════════════════════════════════════════════
#  Model dimension variants
# ═══════════════════════════════════════════════════════════════════
# H (num_value_heads) is the only dimension that varies across models.
# K=128 and V=128 are universal across all known Qwen3 LA models.
H_VALUES = [32, 64]
# QCN/Q35B: H=32 (32 LA value heads)
# Qwen3.5-122B: H=64 (64 LA value heads)

# ═══════════════════════════════════════════════════════════════════
#  Kernel specifications
# ═══════════════════════════════════════════════════════════════════
# Each kernel has:
#   - import_path: where to find the decorated Triton kernel
#   - symbol_name: the name it gets in the .so
#   - signature: runtime parameter types (string keys = param names)
#   - constexprs: compile-time constants (H is filled per-variant)
#   - options: num_warps, num_stages for the compiler
#   - h_dependent: whether we compile separate H variants

KERNEL_SPECS = [
    {
        "name": "chunk_fwd_kernel_o",
        "import": ("fla.ops.common.chunk_o", "chunk_fwd_kernel_o"),
        "symbol_base": "chunk_fwd_kernel_o",
        "signature": {
            "q": "*bf16", "k": "*bf16", "v": "*bf16", "h": "*bf16",
            "g": "*fp32", "g_gamma": "*bf16", "o": "*bf16",
            "cu_seqlens": "*i32", "chunk_indices": "*i32",
            "scale": "fp32", "T": "i32",
        },
        "constexprs": {
            "K": 128, "V": 128, "BT": 64, "BK": 128, "BV": 128,
            "USE_G": True, "USE_G_GAMMA": False,
            "TRANSPOSE_STATE": False, "IS_VARLEN": False,
        },
        # Autotuner config 1: BK=128, BV=128, num_warps=8
        # Rust grid uses o_bv=128, so BV must match.
        "options": {"num_warps": 8, "num_stages": 3},
        "h_dependent": True,
    },
    {
        "name": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "import": ("fla.ops.common.chunk_delta_h",
                    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
        "symbol_base": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "signature": {
            "k": "*bf16", "v": "*bf16", "w": "*bf16", "v_new": "*bf16",
            "g": "*fp32", "gk": "*bf16", "h": "*bf16", "h0": "*bf16",
            "ht": "*fp32", "cu_seqlens": "*i32", "chunk_offsets": "*i32",
            "T": "i32",
        },
        "constexprs": {
            "K": 128, "V": 128, "BT": 64, "BV": 32,
            "USE_G": True, "USE_GK": False,
            "USE_INITIAL_STATE": True, "STORE_FINAL_STATE": True,
            # Keep recurrence on the same natural-log gate contract as the
            # rest of the vendored FLA forward path (cumsum/kkt/wy/output),
            # which all use exp rather than exp2 on g.
            "SAVE_NEW_VALUE": True, "USE_EXP2": False,
            "TRANSPOSE_STATE": False, "IS_VARLEN": False,
        },
        # Autotuner config 1: BV=32, num_warps=2
        # Rust grid uses sr_grid_x = ceil(dv/32), so BV must be 32.
        "options": {"num_warps": 2, "num_stages": 2},
        "h_dependent": True,
    },
    {
        "name": "recompute_w_u_fwd_kernel",
        "import": ("fla.ops.gated_delta_rule.wy_fast",
                    "recompute_w_u_fwd_kernel"),
        "symbol_base": "recompute_w_u_fwd_kernel",
        "signature": {
            "k": "*bf16", "v": "*bf16", "beta": "*bf16",
            "w": "*bf16", "u": "*bf16", "A": "*bf16",
            "g": "*fp32", "cu_seqlens": "*i32", "chunk_indices": "*i32",
            "T": "i32",
        },
        "constexprs": {
            "K": 128, "V": 128, "BT": 64, "BK": 64, "BV": 64,
            "USE_G": True, "IS_VARLEN": False,
        },
        "options": {"num_warps": 4, "num_stages": 2},
        "h_dependent": True,
    },
    {
        "name": "chunk_scaled_dot_kkt_fwd_kernel",
        "import": ("fla.ops.common.chunk_scaled_dot_kkt",
                    "chunk_scaled_dot_kkt_fwd_kernel"),
        "symbol_base": "chunk_scaled_dot_kkt_fwd_kernel",
        "signature": {
            "k": "*bf16", "g": "*fp32", "beta": "*bf16", "A": "*fp32",
            "cu_seqlens": "*i32", "chunk_indices": "*i32", "T": "i32",
        },
        "constexprs": {
            "K": 128, "BT": 64, "BK": 64,
            "IS_VARLEN": False, "USE_G": True,
        },
        "options": {"num_warps": 4, "num_stages": 3},
        "h_dependent": True,
    },
    {
        "name": "chunk_local_cumsum_scalar_kernel",
        "import": ("fla.ops.utils.cumsum",
                    "chunk_local_cumsum_scalar_kernel"),
        "symbol_base": "chunk_local_cumsum_scalar_kernel",
        "signature": {
            "s": "*bf16", "o": "*fp32", "scale": "fp32",
            "cu_seqlens": "*i32", "chunk_indices": "*i32", "T": "i32",
        },
        "constexprs": {
            "B": 1, "BT": 64, "REVERSE": False,
            "HAS_SCALE": True, "IS_VARLEN": False, "HEAD_FIRST": False,
        },
        "options": {"num_warps": 4, "num_stages": 3},
        "h_dependent": True,
    },
    {
        "name": "merge_16x16_to_64x64_inverse_kernel",
        "import": ("fla.ops.utils.solve_tril",
                    "merge_16x16_to_64x64_inverse_kernel"),
        "symbol_base": "merge_16x16_to_64x64_inverse_kernel",
        "signature": {
            "A": "*fp32", "Ai": "*bf16",
            "cu_seqlens": "*i32", "chunk_indices": "*i32", "T": "i32",
        },
        "constexprs": {
            "BT": 64, "USE_TMA": False,
            "IS_VARLEN": False, "DOT_PRECISION": "ieee",
        },
        "options": {"num_warps": 4, "num_stages": 3},
        "h_dependent": True,
    },
    {
        "name": "l2norm_fwd_kernel",
        "import": ("fla.modules.l2norm", "l2norm_fwd_kernel"),
        "symbol_base": "l2norm_fwd_kernel",
        "signature": {
            "x": "*bf16", "y": "*bf16", "rstd": "*fp32",
            "eps": "fp32", "T": "i32",
        },
        "constexprs": {
            "D": 128, "BD": 128, "NB": 1, "BT": 128,
        },
        "options": {"num_warps": 4, "num_stages": 3},
        "h_dependent": False,
    },
]


def import_kernel(module_path: str, kernel_name: str):
    """Import a Triton kernel and unwrap decorators to get the JITFunction."""
    import importlib
    mod = importlib.import_module(module_path)
    kernel = getattr(mod, kernel_name)

    # Unwrap Heuristics -> Autotuner -> JITFunction
    fn = kernel
    while hasattr(fn, 'fn') and type(fn).__name__ != 'JITFunction':
        fn = fn.fn
    if type(fn).__name__ != 'JITFunction':
        raise RuntimeError(f"Could not unwrap {kernel_name} to JITFunction "
                          f"(got {type(fn).__name__})")
    return fn


def extract_kernel_args_from_compiled(compiled) -> list[tuple[str, str]]:
    """Extract kernel argument names and types from a compiled Triton kernel's TTIR.

    Returns list of (name, c_type) tuples.
    """
    ttir = compiled.asm.get('ttir', '')
    if not ttir:
        return []

    func_match = re.search(r'tt\.func public @\w+\((.+?)\)\s+attributes',
                          ttir, re.DOTALL)
    if not func_match:
        return []

    args = []
    params_str = func_match.group(1)
    for param in re.finditer(r'%(\w+):\s+(!tt\.ptr<[^>]+>|i32|i64|f32|f64)',
                            params_str):
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
            c_type = "void*"
        args.append((name, c_type))

    return args


def compile_kernel(spec: dict, h_value: int | None, arch: int):
    """Cross-compile a single kernel for a specific (arch, H) combination.

    Returns dict with cubin, metadata, args, safe_name.
    """
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    target = GPUTarget(backend='cuda', arch=arch, warp_size=32)

    # Triton 3.2 still consults driver.active for some target-dependent
    # constexprs during compilation even when an explicit GPUTarget is passed.
    # Some vendored FLA modules also hit Triton's autotuner at import time,
    # which consults driver.active.get_benchmarker() before we ever reach
    # triton.compile(). Install the temporary shim before importing kernels so
    # those lookups resolve to the requested target instead of requiring a live
    # GPU driver inside manylinux CI.
    @contextlib.contextmanager
    def offline_compile_driver():
        if os.environ.get("KRASIS_FLA_CROSS_COMPILE") != "1":
            yield
            return

        from triton.runtime.driver import driver as triton_driver
        import importlib

        triton_driver_module = importlib.import_module("triton.runtime.driver")

        class _OfflineCompileDriver:
            def __init__(self, compile_target):
                self._target = compile_target

            def get_current_target(self):
                return self._target

            def get_benchmarker(self):
                def _offline_benchmarker(*args, **kwargs):
                    raise RuntimeError(
                        "Triton benchmark requested during offline cross-compilation"
                    )

                return _offline_benchmarker

        offline_driver = _OfflineCompileDriver(target)
        prev_active = triton_driver.active
        prev_default = triton_driver.default
        prev_create_driver = getattr(triton_driver_module, "_create_driver")
        triton_driver_module._create_driver = lambda: offline_driver
        triton_driver.default = offline_driver
        triton_driver.active = offline_driver
        try:
            yield
        finally:
            triton_driver_module._create_driver = prev_create_driver
            triton_driver.default = prev_default
            triton_driver.active = prev_active

    with offline_compile_driver():
        mod_path, kernel_name = spec["import"]
        jit_fn = import_kernel(mod_path, kernel_name)

        # Build constexprs with H filled in
        constexprs = dict(spec["constexprs"])
        if spec["h_dependent"] and h_value is not None:
            constexprs["H"] = h_value

        ast_params = inspect.signature(ASTSource).parameters
        ast_kwargs = {
            "fn": jit_fn,
            "signature": spec["signature"],
        }
        if "constexprs" in ast_params:
            ast_kwargs["constexprs"] = constexprs
        elif "constants" in ast_params:
            ast_kwargs["constants"] = constexprs
        else:
            raise RuntimeError(
                f"Unsupported Triton ASTSource signature: {inspect.signature(ASTSource)}"
            )

        src = ASTSource(**ast_kwargs)
        compiled = triton.compile(src, target=target, options=spec["options"])

    # Extract runtime args from TTIR
    args = extract_kernel_args_from_compiled(compiled)

    # Build safe name for C symbol
    base = spec["symbol_base"].replace("-", "_")
    if spec["h_dependent"] and h_value is not None:
        safe_name = f"{base}_h{h_value}"
    else:
        safe_name = base

    return {
        "name": spec["name"],
        "safe_name": safe_name,
        "cubin": compiled.asm['cubin'],
        "meta": {
            "num_warps": compiled.metadata.num_warps,
            "shared": compiled.metadata.shared,
            "name": compiled.metadata.name,
        },
        "args": args,
        "h_tag": h_value if spec["h_dependent"] else None,
    }


def generate_c_wrapper(kernels: list[dict], output_path: str, cubin_dir: str):
    """Generate a C wrapper source file with embedded cubins and launch functions."""

    lines = []
    lines.append("// Auto-generated by compile_kernels.py - DO NOT EDIT")
    lines.append("// FLA (Flash Linear Attention) vendored kernel wrappers")
    lines.append("// Gated DeltaNet forward pass for Krasis prefill")
    lines.append("")
    lines.append('#include <cuda.h>')
    lines.append('#include <stdint.h>')
    lines.append('#include <stdio.h>')
    lines.append("")

    # Embed cubin data via assembler .incbin
    for k in kernels:
        sn = k['safe_name']
        cubin_path = os.path.join(cubin_dir, f"{sn}.cubin")
        cubin_size = len(k['cubin'])
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
        lines.append(f'    if (res != CUDA_SUCCESS) {{ '
                    f'fprintf(stderr, "FLA: failed to load module {sn}: %d\\n", res); '
                    f'return -1; }}')
        lines.append(f'    res = cuModuleGetFunction(&func_{sn}, module_{sn}, '
                    f'"{k["meta"]["name"]}");')
        lines.append(f'    if (res != CUDA_SUCCESS) {{ '
                    f'fprintf(stderr, "FLA: failed to get function {sn}: %d\\n", res); '
                    f'return -1; }}')
        if shared_mem > 49152:
            lines.append(f'    res = cuFuncSetAttribute(func_{sn}, '
                        f'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, '
                        f'{shared_mem});')
            lines.append(f'    if (res != CUDA_SUCCESS) {{ '
                        f'fprintf(stderr, "FLA: failed to set shared mem for '
                        f'{sn}: %d\\n", res); return -1; }}')
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    # Cleanup function
    lines.append('extern "C" void krasis_fla_cleanup(void) {')
    for k in kernels:
        sn = k['safe_name']
        lines.append(f'    if (module_{sn}) {{ cuModuleUnload(module_{sn}); '
                    f'module_{sn} = NULL; }}')
    lines.append("}")
    lines.append("")

    # Per-kernel launch wrappers
    # Triton 3.5+ cubins expect two hidden trailing arguments:
    #   global_scratch (CUdeviceptr) and profile_scratch (CUdeviceptr)
    for k in kernels:
        sn = k['safe_name']
        args = k['args']
        meta = k['meta']
        num_warps = meta.get('num_warps', 4)
        shared_mem = meta.get('shared', 0)

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

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Generated {output_path} ({len(lines)} lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-compile FLA Triton kernels for multiple GPU architectures")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for generated files")
    parser.add_argument("--arch", type=int, nargs="*", default=None,
                       help="Target GPU architectures (e.g. 80 89 90 120). "
                            "Default: all supported architectures.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archs = args.arch if args.arch else TARGET_ARCHS

    import triton
    triton_version = version.parse(triton.__version__)
    max_supported_arch = 120 if triton_version >= version.parse("3.5.0") else 90
    filtered_archs = [arch for arch in archs if arch <= max_supported_arch]
    skipped_archs = [arch for arch in archs if arch > max_supported_arch]
    if skipped_archs:
        print(
            f"WARNING: Triton {triton.__version__} cannot cross-compile "
            f"{', '.join(f'sm_{arch}' for arch in skipped_archs)}; "
            f"building up to sm_{max_supported_arch} only."
        )
    if not filtered_archs:
        print(
            f"ERROR: Triton {triton.__version__} does not support any requested "
            f"architectures: {', '.join(f'sm_{arch}' for arch in archs)}",
            file=sys.stderr,
        )
        sys.exit(1)
    archs = filtered_archs

    print(f"Target architectures: {['sm_' + str(a) for a in archs]}")
    print(f"H values: {H_VALUES}")
    print(f"Kernels: {len(KERNEL_SPECS)}")
    total = sum(
        len(H_VALUES) if s["h_dependent"] else 1 for s in KERNEL_SPECS
    ) * len(archs)
    print(f"Total compilations: {total}")

    # Add FLA source to path
    fla_dir = Path(__file__).parent.parent.parent / "cuda"
    sys.path.insert(0, str(fla_dir))

    # Find nvcc
    nvcc = None
    for p in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.6/bin/nvcc", "/usr/bin/nvcc"]:
        if os.path.exists(p):
            nvcc = p
            break
    if nvcc is None:
        print("ERROR: nvcc not found", file=sys.stderr)
        sys.exit(1)

    cuda_root = Path(nvcc).resolve().parent.parent
    cuda_stub_dir = None
    cuda_stub_lib = None
    for candidate in [
        cuda_root / "targets" / "x86_64-linux" / "lib" / "stubs",
        cuda_root / "lib64" / "stubs",
        cuda_root / "lib" / "stubs",
    ]:
        stub_lib = candidate / "libcuda.so"
        if stub_lib.exists():
            cuda_stub_dir = candidate
            cuda_stub_lib = stub_lib
            break

    # Compile for each architecture
    for arch in archs:
        print(f"\n{'='*60}")
        print(f"  Compiling for sm_{arch}")
        print(f"{'='*60}")

        arch_dir = output_dir / f"sm_{arch}"
        arch_dir.mkdir(exist_ok=True)
        cubin_dir = arch_dir / "cubins"
        cubin_dir.mkdir(exist_ok=True)

        kernels = []
        for spec in KERNEL_SPECS:
            if spec["h_dependent"]:
                h_variants = H_VALUES
            else:
                h_variants = [None]

            for h_val in h_variants:
                h_label = f" H={h_val}" if h_val is not None else ""
                print(f"  Compiling {spec['name']}{h_label} for sm_{arch}...",
                      end="", flush=True)
                try:
                    k = compile_kernel(spec, h_val, arch)
                    cubin_size = len(k['cubin'])
                    n_args = len(k['args'])
                    print(f" {cubin_size}B, {n_args} args, "
                          f"warps={k['meta']['num_warps']}, "
                          f"shared={k['meta']['shared']}")

                    # Save cubin to disk (needed for .incbin)
                    cubin_path = cubin_dir / f"{k['safe_name']}.cubin"
                    with open(cubin_path, 'wb') as f:
                        f.write(k['cubin'])

                    kernels.append(k)
                except Exception as e:
                    print(f" FAILED: {e}")
                    sys.exit(1)

        # Generate C wrapper for this architecture
        c_path = arch_dir / "fla_vendor.cu"
        generate_c_wrapper(kernels, str(c_path), str(cubin_dir))

        # Compile to .so
        so_name = f"libkrasis_fla_sm{arch}.so"
        so_path = arch_dir / so_name
        compile_cmd = [
            nvcc, "-shared",
            "-o", str(so_path),
            "-Xcompiler", "-fPIC",
            "-Wno-deprecated-gpu-targets",
            str(c_path),
        ]
        compile_env = os.environ.copy()
        if cuda_stub_dir is not None:
            stub_path = str(cuda_stub_dir)
            compile_cmd.extend(["-L", str(cuda_stub_dir)])
            library_path = compile_env.get("LIBRARY_PATH", "")
            compile_env["LIBRARY_PATH"] = (
                f"{stub_path}:{library_path}" if library_path else stub_path
            )
        if cuda_stub_lib is not None:
            compile_cmd.append(str(cuda_stub_lib))
        else:
            compile_cmd.append("-lcuda")
        print(f"  Compiling {so_name}...")
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            env=compile_env,
        )
        if result.returncode != 0:
            print(f"ERROR: nvcc failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"  {so_name}: {so_path.stat().st_size} bytes")

        # Also copy to the top-level output dir for easy access
        import shutil
        top_so = output_dir / so_name
        shutil.copy2(str(so_path), str(top_so))

    # Summary
    print(f"\n{'='*60}")
    print(f"  Build complete")
    print(f"{'='*60}")
    for arch in archs:
        so_path = output_dir / f"libkrasis_fla_sm{arch}.so"
        if so_path.exists():
            print(f"  libkrasis_fla_sm{arch}.so: {so_path.stat().st_size:,} bytes")

    n_kernels = sum(
        len(H_VALUES) if s["h_dependent"] else 1 for s in KERNEL_SPECS
    )
    print(f"\n  {n_kernels} kernel variants x {len(archs)} architectures = "
          f"{n_kernels * len(archs)} total cubins")
    print(f"  H values: {H_VALUES}")
    print(f"  K=128, V=128 (all known models)")


if __name__ == "__main__":
    main()

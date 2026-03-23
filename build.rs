fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_numa)");
    println!("cargo::rustc-check-cfg=cfg(has_decode_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_prefill_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");

    // Probe for libnuma — link only if the library is found.
    // The runtime code (numa.rs) checks numa_available() and falls back
    // gracefully, but the linker needs -lnuma at build time if we use
    // extern "C" FFI declarations.
    //
    // When libnuma is NOT found (e.g. CI manylinux containers), we set
    // cfg(no_numa) so numa.rs can stub out the FFI calls.
    let has_numa = probe_lib("numa");
    if has_numa {
        println!("cargo:rustc-link-lib=numa");
    } else {
        println!("cargo:rustc-cfg=no_numa");
        println!("cargo:warning=libnuma not found — NUMA support disabled (will use fallback)");
    }

    // Compile CUDA decode kernels to PTX if nvcc is available.
    // The PTX is embedded as a string constant via include_str!.
    compile_cuda_kernels();

    // Compile CUDA prefill kernels to PTX (Rust prefill path).
    compile_prefill_kernels();

    // Compile vendored Marlin GEMM kernels into libkrasis_marlin.so
    compile_marlin_kernels();
}

fn compile_cuda_kernels() {
    let cu_src = "src/cuda/decode_kernels.cu";
    println!("cargo:rerun-if-changed={cu_src}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=decode_kernels.cu not found — GPU decode kernels disabled");
        return;
    }

    // Find nvcc
    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — GPU decode kernels disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/decode_kernels.ptx");

    // Compile .cu to .ptx targeting sm_80 (works on Ampere, Ada, Hopper)
    let status = std::process::Command::new(&nvcc)
        .args([
            "-ptx",
            "-arch=sm_80",
            "-O3",
            "--use_fast_math",
            "-o", &ptx_path,
            cu_src,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_decode_kernels");
            println!("cargo:warning=Compiled GPU decode kernels to PTX ({ptx_path})");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed with status {s} — GPU decode kernels disabled");
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution error: {e} — GPU decode kernels disabled");
        }
    }

    println!("cargo:rerun-if-changed={cu_src}");
}

fn compile_prefill_kernels() {
    let cu_src = "src/cuda/prefill_kernels.cu";
    println!("cargo:rerun-if-changed={cu_src}");
    if !std::path::Path::new(cu_src).exists() {
        println!("cargo:warning=prefill_kernels.cu not found — GPU prefill kernels disabled");
        return;
    }

    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — GPU prefill kernels disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/prefill_kernels.ptx");

    let status = std::process::Command::new(&nvcc)
        .args([
            "-ptx",
            "-arch=sm_80",
            "-O3",
            "--use_fast_math",
            "-o", &ptx_path,
            cu_src,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_prefill_kernels");
            println!("cargo:warning=Compiled GPU prefill kernels to PTX ({ptx_path})");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed with status {s} — GPU prefill kernels disabled");
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution error: {e} — GPU prefill kernels disabled");
        }
    }

    println!("cargo:rerun-if-changed={cu_src}");
    println!("cargo:rerun-if-changed=src/cuda/prefill_shim.h");
}

fn compile_marlin_kernels() {
    let marlin_dir = "src/cuda/marlin";
    let src_regular = format!("{marlin_dir}/marlin_vendor.cu");
    let src_moe = format!("{marlin_dir}/marlin_moe_vendor.cu");

    // Track all Marlin source files for incremental rebuilds
    for f in [
        &src_regular, &src_moe,
        &format!("{marlin_dir}/marlin_vendor_common.h"),
        &format!("{marlin_dir}/kernel.h"),
        &format!("{marlin_dir}/marlin_template.h"),
        &format!("{marlin_dir}/marlin_dtypes.cuh"),
        &format!("{marlin_dir}/dequant.h"),
        &format!("{marlin_dir}/scalar_type.hpp"),
        &format!("{marlin_dir}/moe/kernel.h"),
        &format!("{marlin_dir}/moe/marlin_template.h"),
    ] {
        println!("cargo:rerun-if-changed={f}");
    }

    if !std::path::Path::new(&src_regular).exists() {
        println!("cargo:warning=Marlin vendor sources not found — Marlin kernels disabled");
        return;
    }

    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — Marlin kernels disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj_regular = format!("{out_dir}/marlin_vendor.o");
    let obj_moe = format!("{out_dir}/marlin_moe_vendor.o");
    let so_path = format!("{out_dir}/libkrasis_marlin.so");

    let common_args = [
        "--expt-relaxed-constexpr",
        "-Xcompiler", "-fPIC",
        "-arch=sm_80",
        "-O3",
        "--use_fast_math",
        "-I", marlin_dir,
    ];

    // Compile regular Marlin
    let status = std::process::Command::new(&nvcc)
        .arg("-c")
        .arg("-o").arg(&obj_regular)
        .args(&common_args)
        .arg(&src_regular)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=nvcc failed compiling regular Marlin with status {s}");
            return;
        }
        Err(e) => {
            println!("cargo:warning=nvcc error compiling regular Marlin: {e}");
            return;
        }
    }

    // Compile MoE Marlin
    let status = std::process::Command::new(&nvcc)
        .arg("-c")
        .arg("-o").arg(&obj_moe)
        .args(&common_args)
        .arg(&src_moe)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=nvcc failed compiling MoE Marlin with status {s}");
            return;
        }
        Err(e) => {
            println!("cargo:warning=nvcc error compiling MoE Marlin: {e}");
            return;
        }
    }

    // Link into shared library
    let status = std::process::Command::new(&nvcc)
        .arg("-shared")
        .arg("-o").arg(&so_path)
        .arg(&obj_regular)
        .arg(&obj_moe)
        .arg("-Wno-deprecated-gpu-targets")
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:warning=Compiled vendored Marlin kernels to {so_path}");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed linking Marlin .so with status {s}");
        }
        Err(e) => {
            println!("cargo:warning=nvcc link error for Marlin .so: {e}");
        }
    }
}

fn find_nvcc() -> Option<String> {
    // Check CUDA_HOME / CUDA_PATH
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(cuda_dir) = std::env::var(var) {
            let nvcc = format!("{cuda_dir}/bin/nvcc");
            if std::path::Path::new(&nvcc).exists() {
                return Some(nvcc);
            }
        }
    }
    // Check common paths
    for path in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ] {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    // Try PATH
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        return Some("nvcc".to_string());
    }
    None
}

/// Try to find a shared library by compiling a minimal C program that links it.
fn probe_lib(name: &str) -> bool {
    // Quick check: see if the lib exists in common paths
    for dir in &["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"] {
        let so = format!("{dir}/lib{name}.so");
        if std::path::Path::new(&so).exists() {
            return true;
        }
    }
    // Try pkg-config as fallback
    std::process::Command::new("pkg-config")
        .args(["--exists", name])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

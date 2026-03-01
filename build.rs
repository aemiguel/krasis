fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_numa)");
    println!("cargo::rustc-check-cfg=cfg(has_decode_kernels)");

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
}

fn compile_cuda_kernels() {
    let cu_src = "src/cuda/decode_kernels.cu";
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

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_numa)");
    println!("cargo::rustc-check-cfg=cfg(has_decode_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_prefill_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_flash_attn_kernels)");

    // Force rerun when env changes (e.g. CUDA_HOME)
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

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

    // Compile vendored FlashAttention-2 kernels into libkrasis_flash_attn.so
    compile_flash_attn_kernels();
}

fn is_output_fresh(inputs: &[&str], outputs: &[&str]) -> bool {
    if outputs.is_empty() || outputs.iter().any(|path| !std::path::Path::new(path).exists()) {
        return false;
    }

    let newest_input = inputs
        .iter()
        .filter_map(|path| file_mtime(path))
        .max()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let oldest_output = outputs
        .iter()
        .filter_map(|path| file_mtime(path))
        .min()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    oldest_output >= newest_input
}

fn file_mtime(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

fn vendor_python_sidecar(so_path: &str) {
    let src = std::path::Path::new(so_path);
    if !src.exists() {
        return;
    }

    let Some(name) = src.file_name() else {
        println!("cargo:warning=Skipping vendored sidecar copy for invalid path {so_path}");
        return;
    };

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let dst = std::path::Path::new(&manifest_dir).join("python").join("krasis").join(name);
    if let Some(parent) = dst.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            println!(
                "cargo:warning=Failed to create vendored sidecar directory {}: {}",
                parent.display(),
                e
            );
            return;
        }
    }

    let should_copy = match (std::fs::metadata(src), std::fs::metadata(&dst)) {
        (Ok(src_meta), Ok(dst_meta)) => {
            let src_len = src_meta.len();
            let dst_len = dst_meta.len();
            let src_mtime = src_meta.modified().ok();
            let dst_mtime = dst_meta.modified().ok();
            src_len != dst_len
                || match (src_mtime, dst_mtime) {
                    (Some(src_time), Some(dst_time)) => dst_time < src_time,
                    _ => true,
                }
        }
        (Ok(_), Err(_)) => true,
        _ => false,
    };

    if !should_copy {
        println!("cargo:warning=Vendored sidecar already current at {}", dst.display());
        return;
    }

    match std::fs::copy(src, &dst) {
        Ok(_) => {
            println!("cargo:warning=Vendored sidecar copied to {}", dst.display());
        }
        Err(e) => {
            println!(
                "cargo:warning=Failed to copy vendored sidecar {} -> {}: {}",
                src.display(),
                dst.display(),
                e
            );
        }
    }
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

    // Compile sm_80 PTX (works on Ampere SM86/3090, Ada SM89, Hopper, Blackwell via JIT)
    let ptx_sm80 = format!("{out_dir}/decode_kernels_sm80.ptx");

    if is_output_fresh(&[cu_src], &[&ptx_sm80]) {
        println!("cargo:rustc-cfg=has_decode_kernels");
        println!("cargo:warning=Reusing cached GPU decode kernels at {ptx_sm80}");
    } else {
        let status = std::process::Command::new(&nvcc)
            .args([
                "-ptx",
                "-arch=sm_80",
                "-O3",
                "--use_fast_math",
                "-o", &ptx_sm80,
                cu_src,
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-cfg=has_decode_kernels");
                println!("cargo:warning=Compiled GPU decode kernels to sm_80 PTX ({ptx_sm80})");
            }
            Ok(s) => {
                println!("cargo:warning=nvcc failed with status {s} — GPU decode kernels disabled");
                return;
            }
            Err(e) => {
                println!("cargo:warning=nvcc execution error: {e} — GPU decode kernels disabled");
                return;
            }
        }
    }

    // Compile sm_120 PTX (native Blackwell — better warp scheduling on RTX 50x0)
    let ptx_sm120 = format!("{out_dir}/decode_kernels_sm120.ptx");
    if is_output_fresh(&[cu_src], &[&ptx_sm120]) {
        println!("cargo:rustc-cfg=has_decode_kernels_sm120");
        println!("cargo:warning=Reusing cached sm_120 GPU decode kernels at {ptx_sm120}");
    } else {
        let status_120 = std::process::Command::new(&nvcc)
            .args([
                "-ptx",
                "-arch=sm_120",
                "-O3",
                "--use_fast_math",
                "-o", &ptx_sm120,
                cu_src,
            ])
            .status();

        match status_120 {
            Ok(s) if s.success() => {
                println!("cargo:rustc-cfg=has_decode_kernels_sm120");
                println!("cargo:warning=Compiled GPU decode kernels to sm_120 PTX ({ptx_sm120})");
            }
            Ok(s) => {
                println!("cargo:warning=nvcc sm_120 failed with status {s} — sm_80 PTX will be used via JIT");
            }
            Err(e) => {
                println!("cargo:warning=nvcc sm_120 execution error: {e} — sm_80 PTX will be used via JIT");
            }
        }
    }
}

fn compile_prefill_kernels() {
    let cu_src = "src/cuda/prefill_kernels.cu";
    let shim_header = "src/cuda/prefill_shim.h";
    println!("cargo:rerun-if-changed={cu_src}");
    println!("cargo:rerun-if-changed={shim_header}");
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

    if is_output_fresh(&[cu_src, shim_header], &[&ptx_path]) {
        println!("cargo:rustc-cfg=has_prefill_kernels");
        println!("cargo:warning=Reusing cached GPU prefill kernels at {ptx_path}");
        return;
    }

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
}

fn compile_marlin_kernels() {
    let marlin_dir = "src/cuda/marlin";
    let src_regular = format!("{marlin_dir}/marlin_vendor.cu");
    let src_moe = format!("{marlin_dir}/marlin_moe_vendor.cu");
    let tracked_inputs = [
        src_regular.as_str(),
        src_moe.as_str(),
        "src/cuda/marlin/marlin_vendor_common.h",
        "src/cuda/marlin/kernel.h",
        "src/cuda/marlin/marlin_template.h",
        "src/cuda/marlin/marlin_dtypes.cuh",
        "src/cuda/marlin/dequant.h",
        "src/cuda/marlin/scalar_type.hpp",
        "src/cuda/marlin/moe/kernel.h",
        "src/cuda/marlin/moe/marlin_template.h",
    ];

    // Track all Marlin source files for incremental rebuilds
    for f in tracked_inputs {
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

    if is_output_fresh(tracked_inputs.as_slice(), &[&obj_regular, &obj_moe, &so_path]) {
        println!("cargo:rustc-cfg=has_marlin_kernels");
        println!("cargo:warning=Reusing cached vendored Marlin kernels at {so_path}");
        vendor_python_sidecar(&so_path);
        return;
    }

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
            vendor_python_sidecar(&so_path);
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed linking Marlin .so with status {s}");
        }
        Err(e) => {
            println!("cargo:warning=nvcc link error for Marlin .so: {e}");
        }
    }
}

fn compile_flash_attn_kernels() {
    let fa_dir = "src/cuda/flash_attn/fa2";
    let cutlass_dir = "src/cuda/flash_attn/cutlass";
    let vendor_src = format!("{fa_dir}/flash_attn_vendor.cu");
    let tracked_inputs = [
        vendor_src.as_str(),
        "src/cuda/flash_attn/fa2/flash_attn_vendor.h",
        "src/cuda/flash_attn/fa2/flash.h",
        "src/cuda/flash_attn/fa2/flash_fwd_kernel.h",
        "src/cuda/flash_attn/fa2/flash_fwd_launch_template.h",
        "src/cuda/flash_attn/fa2/kernel_traits.h",
        "src/cuda/flash_attn/fa2/softmax.h",
        "src/cuda/flash_attn/fa2/mask.h",
        "src/cuda/flash_attn/fa2/utils.h",
    ];

    // Track key source files for incremental rebuilds
    for f in tracked_inputs {
        println!("cargo:rerun-if-changed={f}");
    }

    if !std::path::Path::new(&vendor_src).exists() {
        println!("cargo:warning=FlashAttention vendor sources not found — FA kernels disabled");
        return;
    }

    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        println!("cargo:warning=nvcc not found — FlashAttention kernels disabled");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let so_path = format!("{out_dir}/libkrasis_flash_attn.so");

    // Skip if .so is newer than all FA2 header sources.
    // The .cu template instantiation files are generated and never edited,
    // so we only need to check the headers that contain the actual kernel logic.
    let fa_sources_owned: Vec<String> = vec![
        vendor_src.clone(),
        format!("{fa_dir}/flash_attn_vendor.h"),
        format!("{fa_dir}/flash.h"),
        format!("{fa_dir}/flash_fwd_kernel.h"),
        format!("{fa_dir}/flash_fwd_launch_template.h"),
        format!("{fa_dir}/kernel_traits.h"),
        format!("{fa_dir}/softmax.h"),
        format!("{fa_dir}/mask.h"),
        format!("{fa_dir}/utils.h"),
    ];
    let fa_sources: Vec<&str> = fa_sources_owned.iter().map(|s| s.as_str()).collect();
    if is_output_fresh(&fa_sources, &[&so_path]) {
        println!("cargo:rustc-cfg=has_flash_attn_kernels");
        println!("cargo:warning=Reusing cached FlashAttention kernels at {so_path}");
        return;
    }

    // Include paths: FA2 source dir + vendored CUTLASS headers
    let common_args = vec![
        "--expt-relaxed-constexpr".to_string(),
        "--expt-extended-lambda".to_string(),
        "-Xcompiler".to_string(), "-fPIC".to_string(),
        "-arch=sm_80".to_string(),
        "-O3".to_string(),
        "--use_fast_math".to_string(),
        "-DKRASIS_FA_VENDOR".to_string(),
        "-DFLASHATTENTION_DISABLE_DROPOUT".to_string(),
        "-DFLASHATTENTION_DISABLE_ALIBI".to_string(),
        "-DFLASHATTENTION_DISABLE_SOFTCAP".to_string(),
        "-DFLASHATTENTION_DISABLE_LOCAL".to_string(),
        format!("-I{fa_dir}"),
        format!("-I{cutlass_dir}"),
    ];

    // Template instantiations to compile (BF16 forward only + FP8 KV variants)
    let cu_files = [
        "flash_attn_vendor.cu",
        "flash_fwd_hdim64_bf16_causal_sm80.cu",
        "flash_fwd_hdim64_bf16_sm80.cu",
        "flash_fwd_hdim96_bf16_causal_sm80.cu",
        "flash_fwd_hdim96_bf16_sm80.cu",
        "flash_fwd_hdim128_bf16_causal_sm80.cu",
        "flash_fwd_hdim128_bf16_sm80.cu",
        "flash_fwd_hdim192_bf16_causal_sm80.cu",
        "flash_fwd_hdim192_bf16_sm80.cu",
        "flash_fwd_hdim256_bf16_causal_sm80.cu",
        "flash_fwd_hdim256_bf16_sm80.cu",
        // FP8 E4M3 K/V variants (BF16 Q, all head dims)
        "flash_fwd_hdim64_bf16q_fp8kv_causal_sm80.cu",
        "flash_fwd_hdim64_bf16q_fp8kv_sm80.cu",
        "flash_fwd_hdim96_bf16q_fp8kv_causal_sm80.cu",
        "flash_fwd_hdim96_bf16q_fp8kv_sm80.cu",
        "flash_fwd_hdim128_bf16q_fp8kv_causal_sm80.cu",
        "flash_fwd_hdim128_bf16q_fp8kv_sm80.cu",
        "flash_fwd_hdim192_bf16q_fp8kv_causal_sm80.cu",
        "flash_fwd_hdim192_bf16q_fp8kv_sm80.cu",
        "flash_fwd_hdim256_bf16q_fp8kv_causal_sm80.cu",
        "flash_fwd_hdim256_bf16q_fp8kv_sm80.cu",
    ];

    let source_paths: Vec<String> = cu_files
        .iter()
        .map(|cu_file| format!("{fa_dir}/{cu_file}"))
        .collect();
    let mut freshness_inputs: Vec<&str> = tracked_inputs.to_vec();
    freshness_inputs.extend(source_paths.iter().map(|s| s.as_str()));
    let obj_paths: Vec<String> = cu_files
        .iter()
        .map(|cu_file| format!("{out_dir}/fa2_{}", cu_file.replace(".cu", ".o")))
        .collect();
    let mut freshness_outputs: Vec<&str> = obj_paths.iter().map(|s| s.as_str()).collect();
    freshness_outputs.push(so_path.as_str());

    if is_output_fresh(freshness_inputs.as_slice(), freshness_outputs.as_slice()) {
        println!("cargo:rustc-cfg=has_flash_attn_kernels");
        println!("cargo:warning=Reusing cached vendored FlashAttention-2 kernels at {so_path}");
        vendor_python_sidecar(&so_path);
        return;
    }

    // Compile each .cu to .o
    let mut obj_files = Vec::new();
    for cu_file in &cu_files {
        let src_path = format!("{fa_dir}/{cu_file}");
        let obj_name = cu_file.replace(".cu", ".o");
        let obj_path = format!("{out_dir}/fa2_{obj_name}");

        println!("cargo:rerun-if-changed={src_path}");

        let status = std::process::Command::new(&nvcc)
            .arg("-c")
            .arg("-o").arg(&obj_path)
            .args(&common_args)
            .arg(&src_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                obj_files.push(obj_path);
            }
            Ok(s) => {
                println!("cargo:warning=nvcc failed compiling {cu_file} with status {s}");
                return;
            }
            Err(e) => {
                println!("cargo:warning=nvcc error compiling {cu_file}: {e}");
                return;
            }
        }
    }

    // Link all .o files into shared library
    let mut link_cmd = std::process::Command::new(&nvcc);
    link_cmd.arg("-shared")
        .arg("-o").arg(&so_path);
    for obj in &obj_files {
        link_cmd.arg(obj);
    }
    link_cmd.arg("-Wno-deprecated-gpu-targets");

    let status = link_cmd.status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-cfg=has_flash_attn_kernels");
            println!("cargo:warning=Compiled vendored FlashAttention-2 kernels to {so_path}");
            vendor_python_sidecar(&so_path);
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed linking FlashAttention .so with status {s}");
        }
        Err(e) => {
            println!("cargo:warning=nvcc link error for FlashAttention .so: {e}");
        }
    }
}

/// Returns true if `output` exists and is newer than every file in `sources`.
/// Used to skip expensive CUDA recompilation when only unrelated sources changed.
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

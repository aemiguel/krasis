//! Startup system checks — validates hardware/OS config for optimal performance.
//!
//! Checks:
//!   1. CPU governor set to "performance" (not "powersave" or "ondemand")
//!   2. Transparent hugepages enabled (reduces TLB misses for large weight arrays)
//!   3. Available memory vs. estimated model footprint
//!   4. NUMA topology (number of nodes, for future NUMA-aware pinning)
//!
//! All checks emit warnings via `log::warn!` — none are fatal.

use pyo3::prelude::*;
use std::path::Path;

/// Run all startup checks and log results.
///
/// Called from KrasisEngine.load() or can be called manually from Python.
pub fn run_startup_checks(model_ram_gb: f64) {
    check_cpu_governor();
    check_hugepages();
    check_memory(model_ram_gb);
    check_numa();
    check_cpu_features();
}

/// Check that CPU frequency governor is set to "performance".
fn check_cpu_governor() {
    let gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";
    match std::fs::read_to_string(gov_path) {
        Ok(gov) => {
            let gov = gov.trim();
            if gov == "performance" {
                log::info!("CPU governor: performance ✓");
            } else {
                log::warn!(
                    "CPU governor is '{}' — set to 'performance' for best throughput: \
                     sudo cpupower frequency-set -g performance",
                    gov,
                );
            }
        }
        Err(_) => {
            log::info!("CPU governor: could not read (may not be available in this environment)");
        }
    }
}

/// Check transparent hugepages and explicit hugepage allocation.
fn check_hugepages() {
    // Check THP (transparent hugepages)
    let thp_path = "/sys/kernel/mm/transparent_hugepage/enabled";
    match std::fs::read_to_string(thp_path) {
        Ok(content) => {
            // Format: "always [madvise] never" — bracketed is active
            if content.contains("[always]") || content.contains("[madvise]") {
                log::info!("Transparent hugepages: enabled ✓");
            } else {
                log::warn!(
                    "Transparent hugepages are disabled — enable for better TLB performance: \
                     echo madvise | sudo tee {}",
                    thp_path,
                );
            }
        }
        Err(_) => {
            log::info!("Transparent hugepages: could not read");
        }
    }

    // Check for explicit 1GB hugepages
    let hp_1g_path = "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages";
    match std::fs::read_to_string(hp_1g_path) {
        Ok(content) => {
            let n: usize = content.trim().parse().unwrap_or(0);
            if n > 0 {
                log::info!("1GB hugepages: {} allocated ({} GB) ✓", n, n);
            } else {
                log::info!(
                    "1GB hugepages: none allocated (optional — can improve TLB for large models)"
                );
            }
        }
        Err(_) => {}
    }

    // Check for 2MB hugepages
    let hp_2m_path = "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages";
    match std::fs::read_to_string(hp_2m_path) {
        Ok(content) => {
            let n: usize = content.trim().parse().unwrap_or(0);
            if n > 0 {
                log::info!("2MB hugepages: {} allocated ({:.1} GB)", n, n as f64 * 2.0 / 1024.0);
            }
        }
        Err(_) => {}
    }
}

/// Check available system memory against estimated model footprint.
fn check_memory(model_ram_gb: f64) {
    let meminfo_path = "/proc/meminfo";
    match std::fs::read_to_string(meminfo_path) {
        Ok(content) => {
            let mut total_kb: u64 = 0;
            let mut avail_kb: u64 = 0;

            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total_kb = parse_meminfo_value(line);
                } else if line.starts_with("MemAvailable:") {
                    avail_kb = parse_meminfo_value(line);
                }
            }

            let total_gb = total_kb as f64 / 1024.0 / 1024.0;
            let avail_gb = avail_kb as f64 / 1024.0 / 1024.0;

            log::info!(
                "System memory: {:.1} GB total, {:.1} GB available",
                total_gb, avail_gb,
            );

            if model_ram_gb > 0.0 {
                let headroom = avail_gb - model_ram_gb;
                if headroom < 0.0 {
                    log::warn!(
                        "INSUFFICIENT MEMORY: model needs {:.1} GB but only {:.1} GB available. \
                         System may OOM or swap heavily.",
                        model_ram_gb, avail_gb,
                    );
                } else if headroom < 32.0 {
                    log::warn!(
                        "Low memory headroom: model needs {:.1} GB, {:.1} GB available ({:.1} GB free). \
                         Consider closing other applications.",
                        model_ram_gb, avail_gb, headroom,
                    );
                } else {
                    log::info!(
                        "Memory budget: {:.1} GB for model, {:.1} GB headroom ✓",
                        model_ram_gb, headroom,
                    );
                }
            }
        }
        Err(_) => {
            log::info!("Memory info: could not read /proc/meminfo");
        }
    }
}

/// Parse a value from /proc/meminfo (format: "FieldName: 12345 kB").
fn parse_meminfo_value(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Check NUMA topology.
fn check_numa() {
    let numa_dir = Path::new("/sys/devices/system/node");
    if !numa_dir.exists() {
        log::info!("NUMA info: not available");
        return;
    }

    // Count NUMA nodes
    let mut num_nodes = 0;
    if let Ok(entries) = std::fs::read_dir(numa_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("node") && name_str[4..].parse::<u32>().is_ok() {
                num_nodes += 1;
            }
        }
    }

    if num_nodes <= 1 {
        log::info!("NUMA: single node (no NUMA-aware pinning needed)");
    } else {
        log::info!(
            "NUMA: {} nodes detected — future: NUMA-aware expert pinning",
            num_nodes,
        );
    }
}

/// Check CPU SIMD features.
fn check_cpu_features() {
    let has_avx2 = is_x86_feature_detected!("avx2");
    let has_fma = is_x86_feature_detected!("fma");
    let has_avx512f = is_x86_feature_detected!("avx512f");

    if has_avx2 && has_fma {
        if has_avx512f {
            log::info!("CPU features: AVX2 + FMA + AVX-512 ✓");
        } else {
            log::info!("CPU features: AVX2 + FMA ✓ (no AVX-512, using AVX2 kernel)");
        }
    } else {
        log::warn!(
            "CPU features: AVX2={}, FMA={} — both required for INT4 matmul!",
            has_avx2, has_fma,
        );
    }
}

/// Log current memory usage from /proc/self/status and /proc/meminfo.
/// Called from diagnostic logging points to track memory consumption.
pub fn log_memory_usage(label: &str) {
    // Process-level memory from /proc/self/status
    let (vm_rss_kb, vm_size_kb) = match std::fs::read_to_string("/proc/self/status") {
        Ok(content) => {
            let mut rss: u64 = 0;
            let mut vsz: u64 = 0;
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    rss = parse_meminfo_value(line);
                } else if line.starts_with("VmSize:") {
                    vsz = parse_meminfo_value(line);
                }
            }
            (rss, vsz)
        }
        Err(_) => (0, 0),
    };

    // System-level memory from /proc/meminfo
    let (total_kb, avail_kb, cached_kb, buffers_kb) = match std::fs::read_to_string("/proc/meminfo") {
        Ok(content) => {
            let mut total: u64 = 0;
            let mut avail: u64 = 0;
            let mut cached: u64 = 0;
            let mut buffers: u64 = 0;
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total = parse_meminfo_value(line);
                } else if line.starts_with("MemAvailable:") {
                    avail = parse_meminfo_value(line);
                } else if line.starts_with("Cached:") && !line.starts_with("CachedSwap") {
                    cached = parse_meminfo_value(line);
                } else if line.starts_with("Buffers:") {
                    buffers = parse_meminfo_value(line);
                }
            }
            (total, avail, cached, buffers)
        }
        Err(_) => (0, 0, 0, 0),
    };

    log::info!(
        "{}: process RSS={:.1} GiB, VSZ={:.1} GiB | system avail={:.1}/{:.1} GiB, page_cache={:.1} GiB",
        label,
        vm_rss_kb as f64 / 1024.0 / 1024.0,
        vm_size_kb as f64 / 1024.0 / 1024.0,
        avail_kb as f64 / 1024.0 / 1024.0,
        total_kb as f64 / 1024.0 / 1024.0,
        (cached_kb + buffers_kb) as f64 / 1024.0 / 1024.0,
    );
}

/// Python-callable system check function.
#[pyfunction]
pub fn system_check() -> PyResult<()> {
    run_startup_checks(0.0);
    Ok(())
}

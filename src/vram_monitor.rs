//! VRAM monitor — background thread polling cudaMemGetInfo every ~50ms.
//!
//! Tracks min-free-VRAM (peak usage) per GPU device. Used to:
//! 1. Measure actual VRAM headroom after warmup (for HCS budget)
//! 2. Warn at runtime when free VRAM hits new lows below safety margin
//! 3. Record VRAM report (periodic samples + named events) when enabled
//!
//! CUDA functions are loaded via dlsym at runtime — no link-time dependency
//! on libcudart. PyTorch always loads it, so it's available.

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// CUDA runtime function signatures (resolved via dlsym)
type CudaSetDeviceFn = unsafe extern "C" fn(i32) -> i32;
type CudaMemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;

/// Load cudaSetDevice + cudaMemGetInfo from the already-loaded libcudart.
/// Returns None if the library isn't loaded or symbols aren't found.
fn load_cuda_fns() -> Option<(CudaSetDeviceFn, CudaMemGetInfoFn)> {
    unsafe {
        // Try common names — RTLD_NOLOAD means "only find already-loaded lib"
        let lib_names: &[&[u8]] = &[
            b"libcudart.so\0",
            b"libcudart.so.12\0",
            b"libcudart.so.11\0",
        ];
        let mut lib = std::ptr::null_mut();
        for name in lib_names {
            lib = libc::dlopen(
                name.as_ptr() as *const libc::c_char,
                libc::RTLD_NOW | libc::RTLD_NOLOAD,
            );
            if !lib.is_null() {
                break;
            }
        }
        if lib.is_null() {
            return None;
        }
        let set_device = libc::dlsym(lib, b"cudaSetDevice\0".as_ptr() as *const libc::c_char);
        let mem_get_info = libc::dlsym(lib, b"cudaMemGetInfo\0".as_ptr() as *const libc::c_char);
        if set_device.is_null() || mem_get_info.is_null() {
            return None;
        }
        Some((
            std::mem::transmute(set_device),
            std::mem::transmute(mem_get_info),
        ))
    }
}

/// Query free VRAM in bytes for a specific device.
fn query_free_bytes(
    set_device: CudaSetDeviceFn,
    mem_get_info: CudaMemGetInfoFn,
    device_id: i32,
) -> Option<usize> {
    unsafe {
        if (set_device)(device_id) != 0 {
            return None;
        }
        let mut free: usize = 0;
        let mut total: usize = 0;
        if (mem_get_info)(&mut free, &mut total) != 0 {
            return None;
        }
        Some(free)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VRAM Report — global state for recording time-series + events
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct ReportEntry {
    timestamp_ms: u64,
    event: String, // empty for periodic samples
    gpu_free_mb: Vec<u64>,
}

struct VramReportState {
    start_time: Instant,
    entries: Vec<ReportEntry>,
    device_ids: Vec<i32>,
}

static REPORT: Mutex<Option<VramReportState>> = Mutex::new(None);

/// Enable VRAM reporting. Called once at startup.
pub fn report_enable(device_ids: Vec<i32>) {
    let mut report = REPORT.lock().unwrap();
    *report = Some(VramReportState {
        start_time: Instant::now(),
        entries: Vec::with_capacity(10000),
        device_ids,
    });
}

/// Check if reporting is enabled.
pub fn report_is_enabled() -> bool {
    REPORT.lock().unwrap().is_some()
}

/// Record a periodic VRAM sample (called from poll thread, no CUDA query needed).
fn report_sample(gpu_free_mb: Vec<u64>) {
    let mut report = REPORT.lock().unwrap();
    if let Some(ref mut state) = *report {
        let ts = state.start_time.elapsed().as_millis() as u64;
        state.entries.push(ReportEntry {
            timestamp_ms: ts,
            event: String::new(),
            gpu_free_mb,
        });
    }
}

/// Query free VRAM (in MB) for all report devices.
fn query_all_free_mb(device_ids: &[i32]) -> Vec<u64> {
    let Some((set_device, mem_get_info)) = load_cuda_fns() else {
        return vec![0; device_ids.len()];
    };
    device_ids
        .iter()
        .map(|&id| {
            query_free_bytes(set_device, mem_get_info, id)
                .map(|f| (f as u64) / (1024 * 1024))
                .unwrap_or(0)
        })
        .collect()
}

/// Record a named event with current VRAM snapshot.
/// No-op if reporting is not enabled.
pub fn report_event(event: &str) {
    // Peek at device_ids without holding lock during CUDA query
    let device_ids = {
        let report = REPORT.lock().unwrap();
        match *report {
            Some(ref state) => state.device_ids.clone(),
            None => return,
        }
    };

    let gpu_free_mb = query_all_free_mb(&device_ids);

    let mut report = REPORT.lock().unwrap();
    if let Some(ref mut state) = *report {
        let ts = state.start_time.elapsed().as_millis() as u64;
        state.entries.push(ReportEntry {
            timestamp_ms: ts,
            event: event.to_string(),
            gpu_free_mb,
        });
    }
}

/// Write VRAM report CSV to file.
pub fn report_write(path: &str) -> std::io::Result<()> {
    let report = REPORT.lock().unwrap();
    let Some(ref state) = *report else {
        return Ok(());
    };

    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    // Header
    write!(f, "timestamp_ms,event")?;
    for &id in &state.device_ids {
        write!(f, ",gpu{}_free_mb", id)?;
    }
    writeln!(f)?;

    // Data
    for entry in &state.entries {
        write!(f, "{},{}", entry.timestamp_ms, entry.event)?;
        for &mb in &entry.gpu_free_mb {
            write!(f, ",{}", mb)?;
        }
        writeln!(f)?;
    }

    Ok(())
}

/// Get summary of key events as list of (event, timestamp_ms, [gpu_free_mb, ...]).
pub fn report_summary() -> Vec<(String, u64, Vec<u64>)> {
    let report = REPORT.lock().unwrap();
    let Some(ref state) = *report else {
        return vec![];
    };

    state
        .entries
        .iter()
        .filter(|e| !e.event.is_empty())
        .map(|e| (e.event.clone(), e.timestamp_ms, e.gpu_free_mb.clone()))
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VramMonitor — background polling + min-free tracking
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct DeviceState {
    device_id: i32,
    total_bytes: AtomicU64,
    min_free_bytes: AtomicU64,
}

/// Background VRAM monitor that polls cudaMemGetInfo and tracks peak usage.
///
/// Usage:
///   monitor = VramMonitor([0, 1], poll_interval_ms=50, safety_margin_mb=3000)
///   monitor.start()
///   # ... run prefill + decode warmup ...
///   min_free = monitor.min_free_mb(0)  # measured min free during warmup
///   hcs_budget = min_free - 3000
///   # ... load HCS experts ...
///   monitor.enable_warnings()          # warns on each new low below margin
///   # ... run server (monitor stays on, no reset needed) ...
///   monitor.stop()
#[pyclass]
pub struct VramMonitor {
    devices: Arc<Vec<DeviceState>>,
    running: Arc<AtomicBool>,
    warn_enabled: Arc<AtomicBool>,
    safety_margin_bytes: Arc<AtomicU64>,
    poll_interval_ms: u64,
    thread_handle: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl VramMonitor {
    #[new]
    #[pyo3(signature = (device_indices, poll_interval_ms=50, safety_margin_mb=3000))]
    fn new(
        device_indices: Vec<i32>,
        poll_interval_ms: u64,
        safety_margin_mb: u64,
    ) -> PyResult<Self> {
        let devices: Vec<DeviceState> = device_indices
            .into_iter()
            .map(|id| DeviceState {
                device_id: id,
                total_bytes: AtomicU64::new(0),
                min_free_bytes: AtomicU64::new(u64::MAX),
            })
            .collect();

        Ok(Self {
            devices: Arc::new(devices),
            running: Arc::new(AtomicBool::new(false)),
            warn_enabled: Arc::new(AtomicBool::new(false)),
            safety_margin_bytes: Arc::new(AtomicU64::new(safety_margin_mb * 1024 * 1024)),
            poll_interval_ms,
            thread_handle: None,
        })
    }

    /// Start the background monitoring thread.
    fn start(&mut self) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Verify CUDA is available before spawning thread
        let _ = load_cuda_fns().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "VRAM monitor: failed to load CUDA runtime (libcudart.so not loaded)",
            )
        })?;

        self.running.store(true, Ordering::Release);

        let devices = self.devices.clone();
        let running = self.running.clone();
        let warn_enabled = self.warn_enabled.clone();
        let safety_margin = self.safety_margin_bytes.clone();
        let poll_ms = self.poll_interval_ms;

        let handle = thread::Builder::new()
            .name("vram-monitor".into())
            .spawn(move || {
                // Load CUDA fns on this thread (avoids Send issues with fn pointers)
                let Some((set_device, mem_get_info)) = load_cuda_fns() else {
                    log::error!("VRAM monitor thread: failed to load CUDA runtime");
                    return;
                };

                // Initial reading to populate total_bytes
                for dev in devices.iter() {
                    unsafe {
                        if (set_device)(dev.device_id) == 0 {
                            let mut free: usize = 0;
                            let mut total: usize = 0;
                            if (mem_get_info)(&mut free, &mut total) == 0 {
                                dev.total_bytes.store(total as u64, Ordering::Relaxed);
                                dev.min_free_bytes.store(free as u64, Ordering::Relaxed);
                            }
                        }
                    }
                }

                log::info!(
                    "VRAM monitor started: {} device(s), poll interval {}ms",
                    devices.len(),
                    poll_ms,
                );

                let interval = Duration::from_millis(poll_ms);
                let report_every = std::cmp::max(1, 200 / poll_ms); // ~200ms between samples
                let mut poll_count = 0u64;

                while running.load(Ordering::Acquire) {
                    let mut readings = Vec::with_capacity(devices.len());

                    for dev in devices.iter() {
                        if let Some(free) = query_free_bytes(set_device, mem_get_info, dev.device_id)
                        {
                            let free_u64 = free as u64;
                            readings.push(free_u64 / (1024 * 1024));

                            let prev_min = dev.min_free_bytes.load(Ordering::Relaxed);

                            if free_u64 < prev_min {
                                dev.min_free_bytes.store(free_u64, Ordering::Relaxed);

                                // Warn on new lows below safety margin (when enabled)
                                if warn_enabled.load(Ordering::Relaxed) {
                                    let margin = safety_margin.load(Ordering::Relaxed);
                                    if free_u64 < margin {
                                        let free_mb = free_u64 / (1024 * 1024);
                                        let margin_mb = margin / (1024 * 1024);
                                        let deficit_mb = margin_mb.saturating_sub(free_mb);
                                        log::warn!(
                                            "VRAM MONITOR: new low on cuda:{} — {} MB free \
                                             (safety margin: {} MB, deficit: {} MB)",
                                            dev.device_id,
                                            free_mb,
                                            margin_mb,
                                            deficit_mb,
                                        );
                                        eprintln!(
                                            "\x1b[1;33m⚠ VRAM MONITOR: new low on cuda:{} — \
                                             {} MB free (safety margin: {} MB, deficit: {} MB)\x1b[0m",
                                            dev.device_id,
                                            free_mb,
                                            margin_mb,
                                            deficit_mb,
                                        );
                                    }
                                }
                            }
                        } else {
                            readings.push(0);
                        }
                    }

                    // Record periodic sample for VRAM report (every ~200ms)
                    poll_count += 1;
                    if poll_count % report_every == 0 {
                        report_sample(readings);
                    }

                    thread::sleep(interval);
                }

                log::info!("VRAM monitor stopped");
            })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to spawn VRAM monitor thread: {}",
                    e
                ))
            })?;

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Stop the background monitoring thread.
    fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Get the minimum free VRAM observed on a device (in MB).
    fn min_free_mb(&self, device_index: i32) -> u64 {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                let bytes = dev.min_free_bytes.load(Ordering::Relaxed);
                if bytes == u64::MAX {
                    return 0;
                }
                return bytes / (1024 * 1024);
            }
        }
        0
    }

    /// Get the total VRAM on a device (in MB).
    fn total_mb(&self, device_index: i32) -> u64 {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                return dev.total_bytes.load(Ordering::Relaxed) / (1024 * 1024);
            }
        }
        0
    }

    /// Get the peak VRAM used on a device (in MB) = total - min_free.
    fn peak_used_mb(&self, device_index: i32) -> u64 {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                let total = dev.total_bytes.load(Ordering::Relaxed);
                let min_free = dev.min_free_bytes.load(Ordering::Relaxed);
                if min_free == u64::MAX || min_free > total {
                    return 0;
                }
                return (total - min_free) / (1024 * 1024);
            }
        }
        0
    }

    /// Reset min-free tracking for a single device (used by 4-point VRAM calibration).
    fn reset(&self, device_index: i32) {
        for dev in self.devices.iter() {
            if dev.device_id == device_index {
                dev.min_free_bytes.store(u64::MAX, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Reset min-free tracking on all devices (e.g. after warmup, before runtime).
    fn reset_min_free(&self) {
        for dev in self.devices.iter() {
            dev.min_free_bytes.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Get current free VRAM on a device (in MB) — live reading, not tracked min.
    fn current_free_mb(&self, device_index: i32) -> u64 {
        let Some((set_device, mem_get_info)) = load_cuda_fns() else {
            return 0;
        };
        match query_free_bytes(set_device, mem_get_info, device_index) {
            Some(free) => (free as u64) / (1024 * 1024),
            None => 0,
        }
    }

    /// Enable runtime warnings when free VRAM drops below safety margin.
    /// Resets min-free tracking so the next poll immediately captures current state
    /// and warns if already below the margin (e.g. right after HCS allocation).
    fn enable_warnings(&self) {
        // Reset min_free so the very next poll sets a fresh baseline.
        // If post-HCS free is already below safety margin, the first poll
        // will detect it as a "new low" and fire the warning immediately.
        for dev in self.devices.iter() {
            dev.min_free_bytes.store(u64::MAX, Ordering::Relaxed);
        }
        self.warn_enabled.store(true, Ordering::Release);
    }

    /// Disable runtime warnings.
    fn disable_warnings(&self) {
        self.warn_enabled.store(false, Ordering::Release);
    }

    /// Update safety margin (in MB).
    fn set_safety_margin_mb(&self, margin_mb: u64) {
        self.safety_margin_bytes
            .store(margin_mb * 1024 * 1024, Ordering::Relaxed);
    }

    // ── VRAM Report methods ──

    /// Enable VRAM reporting. Periodic samples (~200ms) and named events
    /// are recorded to an in-memory buffer. Call write_report() to flush to CSV.
    fn enable_report(&self) {
        let device_ids: Vec<i32> = self.devices.iter().map(|d| d.device_id).collect();
        report_enable(device_ids);
    }

    /// Log a named event with current VRAM snapshot. No-op if report not enabled.
    fn report_event(&self, event: &str) {
        crate::vram_monitor::report_event(event);
    }

    /// Write VRAM report CSV to file. Contains periodic samples and events.
    fn write_report(&self, path: &str) -> PyResult<()> {
        report_write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write VRAM report: {}", e))
        })
    }

    /// Get summary: list of (event_name, timestamp_ms, [gpu_free_mb, ...]) for all events.
    fn report_summary(&self) -> Vec<(String, u64, Vec<u64>)> {
        crate::vram_monitor::report_summary()
    }
}

impl Drop for VramMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

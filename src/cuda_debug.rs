use cudarc::driver::sys as cuda_sys;
use std::collections::{HashMap, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const CUDA_DEBUG_RING_CAPACITY: usize = 256;
const CUDA_DEBUG_FAILURE_TAIL: usize = 8;
const CUDA_DEBUG_SMALL_GUARD_BYTES: usize = 256;
const CUDA_DEBUG_LARGE_GUARD_BYTES: usize = 4096;
const CUDA_DEBUG_LARGE_GUARD_THRESHOLD: usize = 64 * 1024;
const CUDA_DEBUG_QUARANTINE_LIMIT: usize = 8;
const CUDA_DEBUG_ALLOC_PATTERN: u8 = 0xA7;
const CUDA_DEBUG_PREFIX_PATTERN: u8 = 0xC3;
const CUDA_DEBUG_SUFFIX_PATTERN: u8 = 0x3C;
const CUDA_DEBUG_FREED_PATTERN: u8 = 0xDD;
const CUDA_DEBUG_GUARD_RESULT_CLEAR: u32 = 0;
const CUDA_DEBUG_GUARD_RESULT_PREFIX: u32 = 1;
const CUDA_DEBUG_GUARD_RESULT_SUFFIX: u32 = 2;
const CUDA_DEBUG_GUARD_NO_MISMATCH: u32 = u32::MAX;
const CUDA_DEBUG_KERNEL_STATUS_CLEAR: u32 = 0;

const CUDA_DEBUG_SITE_SIGMOID_TOPK_INVALID_EXPERTS: u32 = 1001;
const CUDA_DEBUG_SITE_SIGMOID_TOPK_INVALID_TOPK: u32 = 1002;
const CUDA_DEBUG_SITE_SIGMOID_TOPK_SMEM_TOO_SMALL: u32 = 1003;

const CUDA_DEBUG_SITE_MOE_SCATTER_INVALID_EXPERT_ID: u32 = 2001;
const CUDA_DEBUG_SITE_MOE_SCATTER_BASE_OUT_OF_RANGE: u32 = 2002;
const CUDA_DEBUG_SITE_MOE_SCATTER_NEGATIVE_SLOT: u32 = 2003;
const CUDA_DEBUG_SITE_MOE_SCATTER_DST_OUT_OF_RANGE: u32 = 2004;

#[cfg(has_cuda_debug_kernels)]
const CUDA_DEBUG_KERNELS_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/cuda_debug_kernels.ptx"));

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaDebugMode {
    Off,
    Stage,
    Alloc,
    Kernel,
}

impl CudaDebugMode {
    pub fn from_env() -> Self {
        match std::env::var("KRASIS_CUDA_DEBUG") {
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "stage" | "1" => Self::Stage,
                "alloc" => Self::Alloc,
                "kernel" => Self::Kernel,
                _ => Self::Off,
            },
            Err(_) => Self::Off,
        }
    }

    pub fn sync_each_stage(self) -> bool {
        matches!(self, Self::Stage | Self::Alloc | Self::Kernel)
    }
}

#[derive(Clone, Debug)]
struct RequestContext {
    request_id: u64,
    kind: String,
    tokens: usize,
}

#[derive(Clone, Debug)]
struct StageContext {
    launch_id: u64,
    request_id: Option<u64>,
    request_kind: Option<String>,
    layer_idx: Option<usize>,
    stage_name: String,
    stream_name: String,
    device_ordinal: i32,
    dims: String,
}

#[derive(Clone, Debug)]
struct DebugEvent {
    launch_id: u64,
    request_id: Option<u64>,
    kind: &'static str,
    stage_name: String,
    layer_idx: Option<usize>,
    stream_name: String,
    device_ordinal: i32,
    dims: String,
    detail: String,
}

#[derive(Clone, Debug)]
struct AllocationMeta {
    alloc_id: u64,
    generation: u64,
    raw_ptr: u64,
    payload_ptr: u64,
    total_bytes: usize,
    payload_bytes: usize,
    guard_before: usize,
    guard_after: usize,
    device_ordinal: i32,
    file: String,
    line: u32,
    tag: String,
    tag_hash: u64,
    created_launch_id: Option<u64>,
    last_use_launch_id: Option<u64>,
    last_use_stage: Option<String>,
    last_use_request_id: Option<u64>,
    live: bool,
}

#[derive(Clone, Debug)]
struct BufferTouchContext {
    launch_id: Option<u64>,
    request_id: Option<u64>,
    request_kind: Option<String>,
    layer_idx: Option<usize>,
    stage_name: Option<String>,
    stream_name: Option<String>,
    device_ordinal: Option<i32>,
    dims: Option<String>,
    detail: String,
}

#[derive(Clone, Debug, Default)]
struct TransientBufferMeta {
    last_writer: Option<BufferTouchContext>,
    last_user: Option<BufferTouchContext>,
    active: bool,
}

#[derive(Clone, Debug)]
struct QuarantinedAlloc {
    payload_ptr: u64,
    raw_ptr: u64,
    device_ordinal: i32,
}

#[derive(Clone, Copy)]
struct GuardCheckKernel {
    module: cuda_sys::CUmodule,
    func: cuda_sys::CUfunction,
}

unsafe impl Send for GuardCheckKernel {}
unsafe impl Sync for GuardCheckKernel {}

#[derive(Clone, Copy)]
struct KernelDebugModule {
    module: cuda_sys::CUmodule,
}

unsafe impl Send for KernelDebugModule {}
unsafe impl Sync for KernelDebugModule {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct KernelDebugErrorRecord {
    status: u32,
    site: u32,
    block_x: u32,
    thread_x: u32,
    index: i32,
    bound: i32,
    aux0: i32,
    aux1: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct OwnedCudaAllocation {
    pub raw_ptr: u64,
    pub payload_ptr: u64,
    pub payload_bytes: usize,
    pub total_bytes: usize,
    pub guard_before: usize,
    pub guard_after: usize,
}

#[derive(Default)]
struct DebugState {
    active_request: Option<RequestContext>,
    active_stage: Option<StageContext>,
    ring: VecDeque<DebugEvent>,
    allocs: HashMap<u64, AllocationMeta>,
    transient_buffers: HashMap<String, TransientBufferMeta>,
    quarantine: VecDeque<QuarantinedAlloc>,
    alloc_generation_seq: u64,
}

pub struct CudaDebugController {
    mode: CudaDebugMode,
    request_seq: AtomicU64,
    launch_seq: AtomicU64,
    state: Mutex<DebugState>,
    guard_check_kernels: Mutex<HashMap<i32, GuardCheckKernel>>,
    kernel_debug_modules: Mutex<HashMap<i32, KernelDebugModule>>,
    kernel_debug_error_buffers: Mutex<HashMap<i32, u64>>,
}

impl CudaDebugController {
    pub fn from_env() -> Self {
        Self {
            mode: CudaDebugMode::from_env(),
            request_seq: AtomicU64::new(1),
            launch_seq: AtomicU64::new(1),
            state: Mutex::new(DebugState::default()),
            guard_check_kernels: Mutex::new(HashMap::new()),
            kernel_debug_modules: Mutex::new(HashMap::new()),
            kernel_debug_error_buffers: Mutex::new(HashMap::new()),
        }
    }

    pub fn mode(&self) -> CudaDebugMode {
        self.mode
    }

    pub fn enabled(&self) -> bool {
        self.mode != CudaDebugMode::Off
    }

    pub fn alloc_debug_enabled(&self) -> bool {
        matches!(self.mode, CudaDebugMode::Alloc | CudaDebugMode::Kernel)
    }

    pub fn kernel_debug_enabled(&self) -> bool {
        matches!(self.mode, CudaDebugMode::Kernel)
    }

    pub fn begin_request(&self, kind: &str, tokens: usize) -> Option<u64> {
        if !self.enabled() {
            return None;
        }
        let request_id = self.request_seq.fetch_add(1, Ordering::Relaxed);
        let ctx = RequestContext {
            request_id,
            kind: kind.to_string(),
            tokens,
        };
        let mut state = self.state.lock().unwrap();
        state.active_request = Some(ctx.clone());
        self.push_event_locked(
            &mut state,
            DebugEvent {
                launch_id: 0,
                request_id: Some(request_id),
                kind: "request_begin",
                stage_name: ctx.kind.clone(),
                layer_idx: None,
                stream_name: String::new(),
                device_ordinal: -1,
                dims: format!("tokens={}", tokens),
                detail: format!("request_id={} kind={} tokens={}", request_id, ctx.kind, ctx.tokens),
            },
        );
        Some(request_id)
    }

    pub fn end_request(&self, detail: &str) {
        if !self.enabled() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        if let Some(req) = state.active_request.take() {
            self.push_event_locked(
                &mut state,
                DebugEvent {
                    launch_id: 0,
                    request_id: Some(req.request_id),
                    kind: "request_end",
                    stage_name: req.kind.clone(),
                    layer_idx: None,
                    stream_name: String::new(),
                    device_ordinal: -1,
                    dims: format!("tokens={}", req.tokens),
                    detail: detail.to_string(),
                },
            );
        }
        state.active_stage = None;
    }

    pub fn note_marker(
        &self,
        kind: &'static str,
        stage_name: &str,
        layer_idx: Option<usize>,
        stream_name: &str,
        device_ordinal: i32,
        dims: String,
        detail: String,
    ) {
        if !self.enabled() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let request_id = state.active_request.as_ref().map(|r| r.request_id);
        let launch_id = state.active_stage.as_ref().map(|s| s.launch_id).unwrap_or(0);
        self.push_event_locked(
            &mut state,
            DebugEvent {
                launch_id,
                request_id,
                kind,
                stage_name: stage_name.to_string(),
                layer_idx,
                stream_name: stream_name.to_string(),
                device_ordinal,
                dims,
                detail,
            },
        );
    }

    pub fn note_transient_buffer_write(&self, name: &str, detail: impl Into<String>) {
        self.note_transient_buffer_access(name, "transient_write", true, true, detail.into());
    }

    pub fn note_transient_buffer_use(&self, name: &str, detail: impl Into<String>) {
        self.note_transient_buffer_access(name, "transient_use", false, true, detail.into());
    }

    pub fn note_transient_buffer_clear(&self, name: &str, detail: impl Into<String>) {
        self.note_transient_buffer_access(name, "transient_clear", false, false, detail.into());
    }

    pub fn transient_buffer_summary(&self, names: &[&str]) -> String {
        if !self.enabled() {
            return String::new();
        }
        let state = self.state.lock().unwrap();
        let mut parts = Vec::new();
        for name in names {
            if let Some(meta) = state.transient_buffers.get(*name) {
                parts.push(format!(
                    "{}{{active={} writer={} user={}}}",
                    name,
                    meta.active,
                    format_touch_context(meta.last_writer.as_ref()),
                    format_touch_context(meta.last_user.as_ref()),
                ));
            }
        }
        parts.join(" ")
    }

    pub fn begin_stage(
        &self,
        layer_idx: Option<usize>,
        stage_name: &str,
        stream_name: &str,
        device_ordinal: i32,
        dims: String,
    ) -> Option<u64> {
        if !self.enabled() {
            return None;
        }
        let launch_id = self.launch_seq.fetch_add(1, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap();
        let (request_id, request_kind) = state
            .active_request
            .as_ref()
            .map(|r| (Some(r.request_id), Some(r.kind.clone())))
            .unwrap_or((None, None));
        let ctx = StageContext {
            launch_id,
            request_id,
            request_kind,
            layer_idx,
            stage_name: stage_name.to_string(),
            stream_name: stream_name.to_string(),
            device_ordinal,
            dims: dims.clone(),
        };
        state.active_stage = Some(ctx.clone());
        self.push_event_locked(
            &mut state,
            DebugEvent {
                launch_id,
                request_id,
                kind: "stage_begin",
                stage_name: ctx.stage_name.clone(),
                layer_idx,
                stream_name: ctx.stream_name.clone(),
                device_ordinal,
                dims,
                detail: String::new(),
            },
        );
        Some(launch_id)
    }

    pub fn finish_stage_ok(
        &self,
        launch_id: Option<u64>,
        stream: cuda_sys::CUstream,
    ) -> Result<(), String> {
        let Some(launch_id) = launch_id else {
            return Ok(());
        };
        if self.mode.sync_each_stage() {
            unsafe {
                let err = cuda_sys::lib().cuStreamSynchronize(stream);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(self.finish_stage_err(Some(launch_id), &format!("stage sync: {:?}", err)));
                }
            }
        }
        let mut state = self.state.lock().unwrap();
        let ctx = match state.active_stage.take() {
            Some(ctx) if ctx.launch_id == launch_id => ctx,
            Some(ctx) => {
                state.active_stage = Some(ctx);
                return Ok(());
            }
            None => return Ok(()),
        };
        self.push_event_locked(
            &mut state,
            DebugEvent {
                launch_id,
                request_id: ctx.request_id,
                kind: "stage_ok",
                stage_name: ctx.stage_name,
                layer_idx: ctx.layer_idx,
                stream_name: ctx.stream_name,
                device_ordinal: ctx.device_ordinal,
                dims: ctx.dims,
                detail: String::new(),
            },
        );
        Ok(())
    }

    pub fn finish_stage_err(&self, launch_id: Option<u64>, detail: &str) -> String {
        if !self.enabled() {
            return detail.to_string();
        }
        let mut state = self.state.lock().unwrap();
        let ctx = match launch_id {
            Some(id) => match state.active_stage.take() {
                Some(ctx) if ctx.launch_id == id => ctx,
                Some(ctx) => {
                    let fallback = self.format_failure_message_locked(Some(&ctx), detail, &state);
                    state.active_stage = Some(ctx);
                    return fallback;
                }
                None => return detail.to_string(),
            },
            None => match state.active_stage.take() {
                Some(ctx) => ctx,
                None => return detail.to_string(),
            },
        };
        self.push_event_locked(
            &mut state,
            DebugEvent {
                launch_id: ctx.launch_id,
                request_id: ctx.request_id,
                kind: "stage_err",
                stage_name: ctx.stage_name.clone(),
                layer_idx: ctx.layer_idx,
                stream_name: ctx.stream_name.clone(),
                device_ordinal: ctx.device_ordinal,
                dims: ctx.dims.clone(),
                detail: detail.to_string(),
            },
        );
        self.format_failure_message_locked(Some(&ctx), detail, &state)
    }

    pub fn alloc_zeroed_owned(
        &self,
        payload_bytes: usize,
        device_ordinal: i32,
        tag: &str,
        file: &str,
        line: u32,
    ) -> Result<OwnedCudaAllocation, String> {
        if payload_bytes == 0 {
            return Ok(OwnedCudaAllocation {
                raw_ptr: 0,
                payload_ptr: 0,
                payload_bytes: 0,
                total_bytes: 0,
                guard_before: 0,
                guard_after: 0,
            });
        }

        let (guard_before, guard_after) = if self.alloc_debug_enabled() {
            let guard = if payload_bytes >= CUDA_DEBUG_LARGE_GUARD_THRESHOLD {
                CUDA_DEBUG_LARGE_GUARD_BYTES
            } else {
                CUDA_DEBUG_SMALL_GUARD_BYTES
            };
            (guard, guard)
        } else {
            (0usize, 0usize)
        };
        let total_bytes = payload_bytes + guard_before + guard_after;
        let mut raw_ptr: u64 = 0;
        let alloc_err = unsafe { cuda_sys::lib().cuMemAlloc_v2(&mut raw_ptr, total_bytes) };
        if alloc_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemAlloc_v2({} bytes, tag={}) failed: {:?}",
                total_bytes, tag, alloc_err
            ));
        }
        let payload_ptr = raw_ptr + guard_before as u64;
        let memset_err = unsafe { cuda_sys::lib().cuMemsetD8_v2(payload_ptr, 0, payload_bytes) };
        if memset_err != cuda_sys::CUresult::CUDA_SUCCESS {
            unsafe {
                cuda_sys::lib().cuMemFree_v2(raw_ptr);
            }
            return Err(format!(
                "cuMemsetD8(payload={} bytes, tag={}) failed: {:?}",
                payload_bytes, tag, memset_err
            ));
        }

        if self.alloc_debug_enabled() {
            let meta = {
                let mut state = self.state.lock().unwrap();
                let alloc_id = self.launch_seq.fetch_add(1, Ordering::Relaxed);
                state.alloc_generation_seq += 1;
                let generation = state.alloc_generation_seq;
                let tag_hash = hash_debug_tag(tag);
                let meta = AllocationMeta {
                    alloc_id,
                    generation,
                    raw_ptr,
                    payload_ptr,
                    total_bytes,
                    payload_bytes,
                    guard_before,
                    guard_after,
                    device_ordinal,
                    file: file.to_string(),
                    line,
                    tag: tag.to_string(),
                    tag_hash,
                    created_launch_id: state.active_stage.as_ref().map(|s| s.launch_id),
                    last_use_launch_id: state.active_stage.as_ref().map(|s| s.launch_id),
                    last_use_stage: state.active_stage.as_ref().map(|s| s.stage_name.clone()),
                    last_use_request_id: state.active_request.as_ref().map(|r| r.request_id),
                    live: true,
                };
                state.allocs.insert(payload_ptr, meta.clone());
                let active_launch_id = state.active_stage.as_ref().map(|s| s.launch_id).unwrap_or(0);
                let active_request_id = state.active_request.as_ref().map(|r| r.request_id);
                let active_layer_idx = state.active_stage.as_ref().and_then(|s| s.layer_idx);
                let active_stream_name = state
                    .active_stage
                    .as_ref()
                    .map(|s| s.stream_name.clone())
                    .unwrap_or_default();
                self.push_event_locked(
                    &mut state,
                    DebugEvent {
                        launch_id: active_launch_id,
                        request_id: active_request_id,
                        kind: "alloc",
                        stage_name: meta.tag.clone(),
                        layer_idx: active_layer_idx,
                        stream_name: active_stream_name,
                        device_ordinal,
                        dims: format!("payload_bytes={} total_bytes={}", payload_bytes, total_bytes),
                        detail: format!(
                            "alloc_id={} generation={} ptr=0x{:x} raw=0x{:x} file={}:{}",
                            meta.alloc_id, meta.generation, payload_ptr, raw_ptr, meta.file, meta.line
                        ),
                    },
                );
                meta
            };

            if let Err(err) = self.write_guards(&meta, false) {
                unsafe {
                    cuda_sys::lib().cuMemFree_v2(raw_ptr);
                }
                let mut state = self.state.lock().unwrap();
                state.allocs.remove(&payload_ptr);
                return Err(err);
            }
        }

        Ok(OwnedCudaAllocation {
            raw_ptr,
            payload_ptr,
            payload_bytes,
            total_bytes,
            guard_before,
            guard_after,
        })
    }

    pub fn note_allocation_use(&self, payload_ptr: u64, detail: &str) {
        if !self.alloc_debug_enabled() || payload_ptr == 0 {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let stage_launch = state.active_stage.as_ref().map(|s| s.launch_id);
        let stage_name = state.active_stage.as_ref().map(|s| s.stage_name.clone());
        let request_id = state.active_request.as_ref().map(|r| r.request_id);
        let layer_idx = state.active_stage.as_ref().and_then(|s| s.layer_idx);
        let stream_name = state
            .active_stage
            .as_ref()
            .map(|s| s.stream_name.clone())
            .unwrap_or_default();
        let device_ordinal = state
            .active_stage
            .as_ref()
            .map(|s| s.device_ordinal)
            .unwrap_or(-1);
        let meta_snapshot = if let Some(meta) = state.allocs.get_mut(&payload_ptr) {
            meta.last_use_launch_id = stage_launch;
            meta.last_use_stage = stage_name.clone();
            meta.last_use_request_id = request_id;
            Some(meta.clone())
        } else {
            None
        };
        if let Some(meta) = meta_snapshot {
            self.push_event_locked(
                &mut state,
                DebugEvent {
                    launch_id: stage_launch.unwrap_or(0),
                    request_id,
                    kind: if meta.live { "alloc_use" } else { "alloc_use_after_free" },
                    stage_name: meta.tag.clone(),
                    layer_idx,
                    stream_name,
                    device_ordinal,
                    dims: format!("payload_bytes={}", meta.payload_bytes),
                    detail: format!(
                        "alloc_id={} generation={} ptr=0x{:x} {}",
                        meta.alloc_id, meta.generation, payload_ptr, detail
                    ),
                },
            );
        }
    }

    pub fn validate_device_buffer(
        &self,
        ptr: u64,
        expected_bytes: usize,
        device_ordinal: i32,
        label: &str,
    ) -> Result<(), String> {
        if !self.enabled() || expected_bytes == 0 {
            return Ok(());
        }
        if ptr == 0 {
            return Err(format!("{}: null device pointer", label));
        }

        if let Some(meta) = self.find_tracked_allocation_for_ptr(ptr) {
            if !meta.live {
                return Err(self.format_alloc_failure(
                    &meta,
                    &format!("{}: tracked allocation used after free", label),
                ));
            }
            if meta.device_ordinal != device_ordinal {
                return Err(self.format_alloc_failure(
                    &meta,
                    &format!(
                        "{}: tracked allocation on device {} used on device {}",
                        label, meta.device_ordinal, device_ordinal
                    ),
                ));
            }
            let offset = ptr.saturating_sub(meta.payload_ptr) as usize;
            let end = offset.checked_add(expected_bytes).ok_or_else(|| {
                self.format_alloc_failure(&meta, &format!("{}: range overflow", label))
            })?;
            if end > meta.payload_bytes {
                return Err(self.format_alloc_failure(
                    &meta,
                    &format!(
                        "{}: range {} bytes at offset {} exceeds tracked payload {}",
                        label, expected_bytes, offset, meta.payload_bytes
                    ),
                ));
            }
            return Ok(());
        }

        with_primary_context(device_ordinal, || unsafe {
            let mut memory_type: u32 = 0;
            let mem_err = cuda_sys::lib().cuPointerGetAttribute(
                (&mut memory_type as *mut u32).cast(),
                cuda_sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                ptr,
            );
            if mem_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "{}: cuPointerGetAttribute(memory_type) failed: {:?}",
                    label, mem_err
                ));
            }
            if memory_type != cuda_sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE as u32 {
                return Err(format!(
                    "{}: pointer 0x{:x} has non-device memory type {}",
                    label, ptr, memory_type
                ));
            }

            let mut ptr_device: i32 = -1;
            let dev_err = cuda_sys::lib().cuPointerGetAttribute(
                (&mut ptr_device as *mut i32).cast(),
                cuda_sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                ptr,
            );
            if dev_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "{}: cuPointerGetAttribute(device_ordinal) failed: {:?}",
                    label, dev_err
                ));
            }
            if ptr_device != device_ordinal {
                return Err(format!(
                    "{}: pointer 0x{:x} is on device {} but expected {}",
                    label, ptr, ptr_device, device_ordinal
                ));
            }

            let mut base_ptr: u64 = 0;
            let mut alloc_bytes: usize = 0;
            let range_err = cuda_sys::lib().cuMemGetAddressRange_v2(
                &mut base_ptr,
                &mut alloc_bytes,
                ptr,
            );
            if range_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "{}: cuMemGetAddressRange_v2 failed: {:?}",
                    label, range_err
                ));
            }
            let offset = ptr.checked_sub(base_ptr).ok_or_else(|| {
                format!(
                    "{}: pointer 0x{:x} fell below allocation base 0x{:x}",
                    label, ptr, base_ptr
                )
            })?;
            let end = offset.checked_add(expected_bytes as u64).ok_or_else(|| {
                format!("{}: range overflow for pointer 0x{:x}", label, ptr)
            })?;
            if end > alloc_bytes as u64 {
                return Err(format!(
                    "{}: range {} bytes at offset {} exceeds allocation {}",
                    label, expected_bytes, offset, alloc_bytes
                ));
            }
            Ok(())
        })
    }

    pub fn free_owned_allocation(&self, payload_ptr: u64) -> Result<(), String> {
        if payload_ptr == 0 {
            return Ok(());
        }
        if !self.alloc_debug_enabled() {
            return Err(format!("allocation 0x{:x} not registered in alloc-debug mode", payload_ptr));
        }

        let meta = {
            let state = self.state.lock().unwrap();
            state.allocs.get(&payload_ptr).cloned()
        }
        .ok_or_else(|| format!("unknown tracked allocation 0x{:x}", payload_ptr))?;

        if !meta.live {
            return Err(self.format_alloc_failure(&meta, "double free on tracked allocation"));
        }

        self.validate_allocation_guards(payload_ptr)?;

        let poison_err = unsafe { cuda_sys::lib().cuMemsetD8_v2(meta.payload_ptr, CUDA_DEBUG_FREED_PATTERN, meta.payload_bytes) };
        if poison_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(self.format_alloc_failure(
                &meta,
                &format!("poison payload failed: {:?}", poison_err),
            ));
        }
        self.write_guards(&meta, true)?;

        let to_release = {
            let mut state = self.state.lock().unwrap();
            let stage_launch = state.active_stage.as_ref().map(|s| s.launch_id);
            let stage_name = state.active_stage.as_ref().map(|s| s.stage_name.clone());
            let request_id = state.active_request.as_ref().map(|r| r.request_id);
            let layer_idx = state.active_stage.as_ref().and_then(|s| s.layer_idx);
            let stream_name = state
                .active_stage
                .as_ref()
                .map(|s| s.stream_name.clone())
                .unwrap_or_default();
            let device_ordinal = state
                .active_stage
                .as_ref()
                .map(|s| s.device_ordinal)
                .unwrap_or(meta.device_ordinal);
            if let Some(entry) = state.allocs.get_mut(&payload_ptr) {
                entry.live = false;
                entry.last_use_launch_id = stage_launch;
                entry.last_use_stage = stage_name.clone();
                entry.last_use_request_id = request_id;
            }
            self.push_event_locked(
                &mut state,
                DebugEvent {
                    launch_id: stage_launch.unwrap_or(0),
                    request_id,
                    kind: "alloc_free",
                    stage_name: meta.tag.clone(),
                    layer_idx,
                    stream_name,
                    device_ordinal,
                    dims: format!("payload_bytes={}", meta.payload_bytes),
                    detail: format!(
                        "alloc_id={} generation={} ptr=0x{:x}",
                        meta.alloc_id, meta.generation, payload_ptr
                    ),
                },
            );
            state.quarantine.push_back(QuarantinedAlloc {
                payload_ptr,
                raw_ptr: meta.raw_ptr,
                device_ordinal: meta.device_ordinal,
            });

            let mut to_release = Vec::new();
            while state.quarantine.len() > CUDA_DEBUG_QUARANTINE_LIMIT {
                if let Some(old) = state.quarantine.pop_front() {
                    to_release.push(old);
                }
            }
            to_release
        };

        for old in to_release {
            free_raw_allocation(old.raw_ptr, old.device_ordinal)?;
            let mut state = self.state.lock().unwrap();
            state.allocs.remove(&old.payload_ptr);
        }

        Ok(())
    }

    pub fn validate_allocation_guards(&self, payload_ptr: u64) -> Result<(), String> {
        if !self.alloc_debug_enabled() || payload_ptr == 0 {
            return Ok(());
        }
        let meta = {
            let state = self.state.lock().unwrap();
            state.allocs.get(&payload_ptr).cloned()
        }
        .ok_or_else(|| format!("unknown tracked allocation 0x{:x}", payload_ptr))?;

        if meta.guard_before == 0 && meta.guard_after == 0 {
            return Ok(());
        }

        if matches!(self.mode, CudaDebugMode::Kernel) {
            match self.validate_allocation_guards_on_device(&meta) {
                Ok(()) => return Ok(()),
                Err(err) if err.contains("kernel unavailable") => {}
                Err(err) => return Err(err),
            }
        }

        let expected_prefix = build_guard_bytes(&meta, true, meta.live);
        let expected_suffix = build_guard_bytes(&meta, false, meta.live);

        let mut prefix = vec![0u8; meta.guard_before];
        if meta.guard_before > 0 {
            let err = unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    prefix.as_mut_ptr() as *mut std::ffi::c_void,
                    meta.raw_ptr,
                    meta.guard_before,
                )
            };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(self.format_alloc_failure(
                    &meta,
                    &format!("guard prefix copy failed: {:?}", err),
                ));
            }
        }

        let mut suffix = vec![0u8; meta.guard_after];
        if meta.guard_after > 0 {
            let err = unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    suffix.as_mut_ptr() as *mut std::ffi::c_void,
                    meta.payload_ptr + meta.payload_bytes as u64,
                    meta.guard_after,
                )
            };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(self.format_alloc_failure(
                    &meta,
                    &format!("guard suffix copy failed: {:?}", err),
                ));
            }
        }

        if prefix != expected_prefix {
            return Err(self.format_alloc_failure(
                &meta,
                "prefix guard corrupted",
            ));
        }
        if suffix != expected_suffix {
            return Err(self.format_alloc_failure(
                &meta,
                "suffix guard corrupted",
            ));
        }
        Ok(())
    }

    pub unsafe fn launch_kernel_debug_variant(
        &self,
        device_ordinal: i32,
        func_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        stream: cuda_sys::CUstream,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        if !self.kernel_debug_enabled() {
            return Err("kernel debug launch requested while KRASIS_CUDA_DEBUG!=kernel".to_string());
        }

        let func = self.kernel_debug_function(device_ordinal, func_name)?;
        let error_buf_ptr = self.reset_kernel_debug_error_buffer(device_ordinal, stream)?;
        let mut launch_params = Vec::with_capacity(params.len() + 1);
        launch_params.extend_from_slice(params);
        launch_params.push((&error_buf_ptr as *const u64).cast_mut().cast());

        let launch_err = cuda_sys::lib().cuLaunchKernel(
            func,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            smem,
            stream,
            launch_params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if launch_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "kernel debug launch {} failed: {:?} (grid={:?}, block={:?}, smem={}, nparams={})",
                func_name,
                launch_err,
                grid,
                block,
                smem,
                launch_params.len(),
            ));
        }

        let sync_err = cuda_sys::lib().cuStreamSynchronize(stream);
        if sync_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "kernel debug stream sync {} failed: {:?}",
                func_name, sync_err
            ));
        }

        let record = self.read_kernel_debug_error_buffer(device_ordinal)?;
        if record.status != CUDA_DEBUG_KERNEL_STATUS_CLEAR {
            return Err(self.format_kernel_debug_failure(func_name, &record));
        }
        Ok(())
    }

    fn validate_allocation_guards_on_device(&self, meta: &AllocationMeta) -> Result<(), String> {
        #[cfg(not(has_cuda_debug_kernels))]
        {
            let _ = meta;
            return Err("kernel unavailable: cuda debug kernels were not built".to_string());
        }

        #[cfg(has_cuda_debug_kernels)]
        {
            let kernel = self.guard_check_kernel(meta.device_ordinal)?;
            let prefix_ptr = meta.raw_ptr;
            let suffix_ptr = meta.payload_ptr + meta.payload_bytes as u64;
            let mut result_flags = CUDA_DEBUG_GUARD_RESULT_CLEAR;
            let mut prefix_idx = CUDA_DEBUG_GUARD_NO_MISMATCH;
            let mut suffix_idx = CUDA_DEBUG_GUARD_NO_MISMATCH;
            let result_flags_ptr = alloc_host_u32(CUDA_DEBUG_GUARD_RESULT_CLEAR)?;
            let prefix_idx_ptr = alloc_host_u32(CUDA_DEBUG_GUARD_NO_MISMATCH)?;
            let suffix_idx_ptr = alloc_host_u32(CUDA_DEBUG_GUARD_NO_MISMATCH)?;

            let launch_result = with_primary_context(meta.device_ordinal, || unsafe {
                let mut params: [*mut std::ffi::c_void; 12] = [
                    (&prefix_ptr as *const u64).cast_mut().cast(),
                    (&suffix_ptr as *const u64).cast_mut().cast(),
                    (&(meta.guard_before as i32) as *const i32).cast_mut().cast(),
                    (&(meta.guard_after as i32) as *const i32).cast_mut().cast(),
                    (&meta.alloc_id as *const u64).cast_mut().cast(),
                    (&meta.generation as *const u64).cast_mut().cast(),
                    (&(meta.payload_bytes as u64) as *const u64).cast_mut().cast(),
                    (&meta.tag_hash as *const u64).cast_mut().cast(),
                    (&(if meta.live { 1i32 } else { 0i32 }) as *const i32).cast_mut().cast(),
                    (&result_flags_ptr as *const u64).cast_mut().cast(),
                    (&prefix_idx_ptr as *const u64).cast_mut().cast(),
                    (&suffix_idx_ptr as *const u64).cast_mut().cast(),
                ];
                let threads = 256u32;
                let work_items = (meta.guard_before + meta.guard_after).max(1);
                let grid = ((work_items.div_ceil(threads as usize)) as u32).min(256);
                let launch_err = cuda_sys::lib().cuLaunchKernel(
                    kernel.func,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if launch_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("guard-check launch failed: {:?}", launch_err));
                }
                let sync_err = cuda_sys::lib().cuCtxSynchronize();
                if sync_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("guard-check sync failed: {:?}", sync_err));
                }
                let copy_flags_err = cuda_sys::lib().cuMemcpyDtoH_v2(
                    (&mut result_flags as *mut u32).cast(),
                    result_flags_ptr,
                    std::mem::size_of::<u32>(),
                );
                if copy_flags_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("guard-check flags copy failed: {:?}", copy_flags_err));
                }
                let copy_prefix_err = cuda_sys::lib().cuMemcpyDtoH_v2(
                    (&mut prefix_idx as *mut u32).cast(),
                    prefix_idx_ptr,
                    std::mem::size_of::<u32>(),
                );
                if copy_prefix_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("guard-check prefix copy failed: {:?}", copy_prefix_err));
                }
                let copy_suffix_err = cuda_sys::lib().cuMemcpyDtoH_v2(
                    (&mut suffix_idx as *mut u32).cast(),
                    suffix_idx_ptr,
                    std::mem::size_of::<u32>(),
                );
                if copy_suffix_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("guard-check suffix copy failed: {:?}", copy_suffix_err));
                }
                Ok(())
            });

            free_raw_allocation(result_flags_ptr, meta.device_ordinal)?;
            free_raw_allocation(prefix_idx_ptr, meta.device_ordinal)?;
            free_raw_allocation(suffix_idx_ptr, meta.device_ordinal)?;
            launch_result?;

            if (result_flags & CUDA_DEBUG_GUARD_RESULT_PREFIX) != 0 {
                return Err(self.format_alloc_failure(
                    meta,
                    &format!("prefix guard corrupted (device check idx={})", prefix_idx),
                ));
            }
            if (result_flags & CUDA_DEBUG_GUARD_RESULT_SUFFIX) != 0 {
                return Err(self.format_alloc_failure(
                    meta,
                    &format!("suffix guard corrupted (device check idx={})", suffix_idx),
                ));
            }
            Ok(())
        }
    }

    fn push_event_locked(&self, state: &mut DebugState, event: DebugEvent) {
        if state.ring.len() >= CUDA_DEBUG_RING_CAPACITY {
            state.ring.pop_front();
        }
        state.ring.push_back(event);
    }

    fn format_failure_message_locked(
        &self,
        ctx: Option<&StageContext>,
        detail: &str,
        state: &DebugState,
    ) -> String {
        let mut msg = String::new();
        msg.push_str("[CUDA-DEBUG] ");
        if let Some(ctx) = ctx {
            msg.push_str(&format!(
                "launch_id={} request_id={} request_kind={} layer={} stage={} stream={} device={} dims={} error={}",
                ctx.launch_id,
                ctx.request_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
                ctx.request_kind.clone().unwrap_or_else(|| "none".to_string()),
                ctx.layer_idx.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
                ctx.stage_name,
                ctx.stream_name,
                ctx.device_ordinal,
                ctx.dims,
                detail,
            ));
        } else {
            msg.push_str(detail);
        }

        let mut tail = Vec::new();
        for event in state.ring.iter().rev().take(CUDA_DEBUG_FAILURE_TAIL).rev() {
            tail.push(format!(
                "kind={} launch_id={} request_id={} layer={} stage={} stream={} device={} dims={} detail={}",
                event.kind,
                event.launch_id,
                event.request_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
                event.layer_idx.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
                event.stage_name,
                event.stream_name,
                event.device_ordinal,
                event.dims,
                event.detail,
            ));
        }
        if !tail.is_empty() {
            msg.push_str(" recent_events=[");
            msg.push_str(&tail.join(" | "));
            msg.push(']');
        }
        msg
    }

    fn write_guards(&self, meta: &AllocationMeta, freed: bool) -> Result<(), String> {
        if meta.guard_before > 0 {
            let prefix = build_guard_bytes(meta, true, !freed);
            let err = unsafe {
                cuda_sys::lib().cuMemcpyHtoD_v2(
                    meta.raw_ptr,
                    prefix.as_ptr() as *const std::ffi::c_void,
                    meta.guard_before,
                )
            };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(self.format_alloc_failure(
                    meta,
                    &format!("write prefix guard failed: {:?}", err),
                ));
            }
        }
        if meta.guard_after > 0 {
            let suffix = build_guard_bytes(meta, false, !freed);
            let err = unsafe {
                cuda_sys::lib().cuMemcpyHtoD_v2(
                    meta.payload_ptr + meta.payload_bytes as u64,
                    suffix.as_ptr() as *const std::ffi::c_void,
                    meta.guard_after,
                )
            };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(self.format_alloc_failure(
                    meta,
                    &format!("write suffix guard failed: {:?}", err),
                ));
            }
        }
        Ok(())
    }

    fn format_alloc_failure(&self, meta: &AllocationMeta, detail: &str) -> String {
        let state = self.state.lock().unwrap();
        let mut msg = self.format_failure_message_locked(state.active_stage.as_ref(), detail, &state);
        msg.push_str(&format!(
            " alloc_id={} generation={} tag={} ptr=0x{:x} raw=0x{:x} payload_bytes={} total_bytes={} guard_before={} guard_after={} device={} created_at={}:{} created_launch_id={} last_use_launch_id={} last_use_stage={} last_use_request_id={} live={}",
            meta.alloc_id,
            meta.generation,
            meta.tag,
            meta.payload_ptr,
            meta.raw_ptr,
            meta.payload_bytes,
            meta.total_bytes,
            meta.guard_before,
            meta.guard_after,
            meta.device_ordinal,
            meta.file,
            meta.line,
            meta.created_launch_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            meta.last_use_launch_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            meta.last_use_stage.clone().unwrap_or_else(|| "none".to_string()),
            meta.last_use_request_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            meta.live,
        ));
        msg
    }

    #[cfg(has_cuda_debug_kernels)]
    fn guard_check_kernel(&self, device_ordinal: i32) -> Result<GuardCheckKernel, String> {
        if let Some(existing) = self.guard_check_kernels.lock().unwrap().get(&device_ordinal).copied() {
            return Ok(existing);
        }

        let kernel = with_primary_context(device_ordinal, || unsafe {
            let mut module: cuda_sys::CUmodule = std::ptr::null_mut();
            let ptx = std::ffi::CString::new(CUDA_DEBUG_KERNELS_PTX)
                .map_err(|_| "cuda debug PTX contained interior NUL".to_string())?;
            let load_err = cuda_sys::lib().cuModuleLoadData(
                &mut module,
                ptx.as_ptr().cast(),
            );
            if load_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData(cuda_debug_kernels) failed: {:?}", load_err));
            }

            let func_name = CString::new("krasis_guard_check_kernel").unwrap();
            let mut func: cuda_sys::CUfunction = std::ptr::null_mut();
            let func_err = cuda_sys::lib().cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if func_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleGetFunction(krasis_guard_check_kernel) failed: {:?}", func_err));
            }
            Ok(GuardCheckKernel { module, func })
        })?;

        let mut cache = self.guard_check_kernels.lock().unwrap();
        let entry = cache.entry(device_ordinal).or_insert(kernel);
        Ok(*entry)
    }

    #[cfg(has_cuda_debug_kernels)]
    fn kernel_debug_module(&self, device_ordinal: i32) -> Result<KernelDebugModule, String> {
        if let Some(existing) = self
            .kernel_debug_modules
            .lock()
            .unwrap()
            .get(&device_ordinal)
            .copied()
        {
            return Ok(existing);
        }

        let module = with_primary_context(device_ordinal, || unsafe {
            let mut module: cuda_sys::CUmodule = std::ptr::null_mut();
            let ptx = CString::new(CUDA_DEBUG_KERNELS_PTX)
                .map_err(|_| "cuda debug PTX contained interior NUL".to_string())?;
            let load_err = cuda_sys::lib().cuModuleLoadData(&mut module, ptx.as_ptr().cast());
            if load_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "cuModuleLoadData(cuda_debug_kernels) failed: {:?}",
                    load_err
                ));
            }
            Ok(KernelDebugModule { module })
        })?;

        let mut cache = self.kernel_debug_modules.lock().unwrap();
        let entry = cache.entry(device_ordinal).or_insert(module);
        Ok(*entry)
    }

    #[cfg(has_cuda_debug_kernels)]
    fn kernel_debug_function(
        &self,
        device_ordinal: i32,
        func_name: &str,
    ) -> Result<cuda_sys::CUfunction, String> {
        let module = self.kernel_debug_module(device_ordinal)?;
        with_primary_context(device_ordinal, || unsafe {
            let mut func: cuda_sys::CUfunction = std::ptr::null_mut();
            let func_name = CString::new(func_name)
                .map_err(|_| format!("kernel debug function name had interior NUL: {func_name}"))?;
            let func_err = cuda_sys::lib().cuModuleGetFunction(&mut func, module.module, func_name.as_ptr());
            if func_err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "cuModuleGetFunction({}) failed: {:?}",
                    func_name.to_string_lossy(),
                    func_err
                ));
            }
            Ok(func)
        })
    }

    #[cfg(has_cuda_debug_kernels)]
    fn reset_kernel_debug_error_buffer(
        &self,
        device_ordinal: i32,
        stream: cuda_sys::CUstream,
    ) -> Result<u64, String> {
        let ptr = if let Some(existing) = self
            .kernel_debug_error_buffers
            .lock()
            .unwrap()
            .get(&device_ordinal)
            .copied()
        {
            existing
        } else {
            let ptr = with_primary_context(device_ordinal, || unsafe {
                let mut ptr: u64 = 0;
                let alloc_err = cuda_sys::lib().cuMemAlloc_v2(
                    &mut ptr,
                    std::mem::size_of::<KernelDebugErrorRecord>(),
                );
                if alloc_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!(
                        "cuMemAlloc_v2(kernel_debug_error_buffer, device={}) failed: {:?}",
                        device_ordinal, alloc_err
                    ));
                }
                Ok(ptr)
            })?;
            self.kernel_debug_error_buffers
                .lock()
                .unwrap()
                .insert(device_ordinal, ptr);
            ptr
        };

        let memset_err = unsafe {
            cuda_sys::lib().cuMemsetD8Async(
                ptr,
                0,
                std::mem::size_of::<KernelDebugErrorRecord>(),
                stream,
            )
        };
        if memset_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemsetD8Async(kernel_debug_error_buffer, device={}) failed: {:?}",
                device_ordinal, memset_err
            ));
        }
        Ok(ptr)
    }

    #[cfg(has_cuda_debug_kernels)]
    fn read_kernel_debug_error_buffer(
        &self,
        device_ordinal: i32,
    ) -> Result<KernelDebugErrorRecord, String> {
        let ptr = self
            .kernel_debug_error_buffers
            .lock()
            .unwrap()
            .get(&device_ordinal)
            .copied()
            .ok_or_else(|| format!("kernel debug error buffer missing for device {}", device_ordinal))?;
        let mut record = KernelDebugErrorRecord::default();
        let copy_err = unsafe {
            cuda_sys::lib().cuMemcpyDtoH_v2(
                (&mut record as *mut KernelDebugErrorRecord).cast(),
                ptr,
                std::mem::size_of::<KernelDebugErrorRecord>(),
            )
        };
        if copy_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemcpyDtoH_v2(kernel_debug_error_buffer, device={}) failed: {:?}",
                device_ordinal, copy_err
            ));
        }
        Ok(record)
    }

    #[cfg(has_cuda_debug_kernels)]
    fn format_kernel_debug_failure(
        &self,
        func_name: &str,
        record: &KernelDebugErrorRecord,
    ) -> String {
        format!(
            "device kernel debug failure: kernel={} site={} ({}) block.x={} thread.x={} index={} bound={} aux0={} aux1={}",
            func_name,
            record.site,
            kernel_debug_site_label(record.site),
            record.block_x,
            record.thread_x,
            record.index,
            record.bound,
            record.aux0,
            record.aux1,
        )
    }

    #[cfg(not(has_cuda_debug_kernels))]
    fn kernel_debug_function(
        &self,
        _device_ordinal: i32,
        _func_name: &str,
    ) -> Result<cuda_sys::CUfunction, String> {
        Err("kernel unavailable: cuda debug kernels were not built".to_string())
    }

    #[cfg(not(has_cuda_debug_kernels))]
    fn reset_kernel_debug_error_buffer(
        &self,
        _device_ordinal: i32,
        _stream: cuda_sys::CUstream,
    ) -> Result<u64, String> {
        Err("kernel unavailable: cuda debug kernels were not built".to_string())
    }

    #[cfg(not(has_cuda_debug_kernels))]
    fn read_kernel_debug_error_buffer(
        &self,
        _device_ordinal: i32,
    ) -> Result<KernelDebugErrorRecord, String> {
        Err("kernel unavailable: cuda debug kernels were not built".to_string())
    }

    fn find_tracked_allocation_for_ptr(&self, ptr: u64) -> Option<AllocationMeta> {
        let state = self.state.lock().unwrap();
        state
            .allocs
            .values()
            .find(|meta| {
                let start = meta.payload_ptr;
                let end = meta.payload_ptr + meta.payload_bytes as u64;
                ptr >= start && ptr < end
            })
            .cloned()
    }

    fn note_transient_buffer_access(
        &self,
        name: &str,
        event_kind: &'static str,
        is_write: bool,
        active: bool,
        detail: String,
    ) {
        if !self.enabled() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let ctx = touch_context_from_state(&state, detail.clone());
        let meta = state
            .transient_buffers
            .entry(name.to_string())
            .or_default();
        meta.active = active;
        if is_write {
            meta.last_writer = Some(ctx.clone());
        } else {
            meta.last_user = Some(ctx.clone());
        }
        self.push_event_locked(
            &mut state,
            DebugEvent {
                launch_id: ctx.launch_id.unwrap_or(0),
                request_id: ctx.request_id,
                kind: event_kind,
                stage_name: name.to_string(),
                layer_idx: ctx.layer_idx,
                stream_name: ctx.stream_name.clone().unwrap_or_default(),
                device_ordinal: ctx.device_ordinal.unwrap_or(-1),
                dims: ctx.dims.clone().unwrap_or_default(),
                detail,
            },
        );
    }
}

fn hash_debug_tag(tag: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    hasher.finish()
}

fn touch_context_from_state(state: &DebugState, detail: String) -> BufferTouchContext {
    BufferTouchContext {
        launch_id: state.active_stage.as_ref().map(|s| s.launch_id),
        request_id: state.active_request.as_ref().map(|r| r.request_id),
        request_kind: state.active_request.as_ref().map(|r| r.kind.clone()),
        layer_idx: state.active_stage.as_ref().and_then(|s| s.layer_idx),
        stage_name: state.active_stage.as_ref().map(|s| s.stage_name.clone()),
        stream_name: state.active_stage.as_ref().map(|s| s.stream_name.clone()),
        device_ordinal: state.active_stage.as_ref().map(|s| s.device_ordinal),
        dims: state.active_stage.as_ref().map(|s| s.dims.clone()),
        detail,
    }
}

fn format_touch_context(ctx: Option<&BufferTouchContext>) -> String {
    match ctx {
        Some(ctx) => format!(
            "launch={} request={} kind={} layer={} stage={} stream={} device={} dims={} detail={}",
            ctx.launch_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            ctx.request_id.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            ctx.request_kind.clone().unwrap_or_else(|| "none".to_string()),
            ctx.layer_idx.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            ctx.stage_name.clone().unwrap_or_else(|| "none".to_string()),
            ctx.stream_name.clone().unwrap_or_else(|| "none".to_string()),
            ctx.device_ordinal.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
            ctx.dims.clone().unwrap_or_else(|| "none".to_string()),
            ctx.detail,
        ),
        None => "none".to_string(),
    }
}

fn kernel_debug_site_label(site: u32) -> &'static str {
    match site {
        CUDA_DEBUG_SITE_SIGMOID_TOPK_INVALID_EXPERTS => "sigmoid_topk.invalid_num_experts",
        CUDA_DEBUG_SITE_SIGMOID_TOPK_INVALID_TOPK => "sigmoid_topk.invalid_topk",
        CUDA_DEBUG_SITE_SIGMOID_TOPK_SMEM_TOO_SMALL => "sigmoid_topk.shared_memory_too_small",
        CUDA_DEBUG_SITE_MOE_SCATTER_INVALID_EXPERT_ID => "moe_scatter_sorted.invalid_expert_id",
        CUDA_DEBUG_SITE_MOE_SCATTER_BASE_OUT_OF_RANGE => "moe_scatter_sorted.base_out_of_range",
        CUDA_DEBUG_SITE_MOE_SCATTER_NEGATIVE_SLOT => "moe_scatter_sorted.negative_slot",
        CUDA_DEBUG_SITE_MOE_SCATTER_DST_OUT_OF_RANGE => "moe_scatter_sorted.dst_out_of_range",
        _ => "unknown",
    }
}

fn alloc_host_u32(value: u32) -> Result<u64, String> {
    let mut ptr: u64 = 0;
    let alloc_err = unsafe { cuda_sys::lib().cuMemAlloc_v2(&mut ptr, std::mem::size_of::<u32>()) };
    if alloc_err != cuda_sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAlloc_v2(guard-check u32) failed: {:?}", alloc_err));
    }
    let copy_err = unsafe {
        cuda_sys::lib().cuMemcpyHtoD_v2(
            ptr,
            (&value as *const u32).cast(),
            std::mem::size_of::<u32>(),
        )
    };
    if copy_err != cuda_sys::CUresult::CUDA_SUCCESS {
        unsafe {
            cuda_sys::lib().cuMemFree_v2(ptr);
        }
        return Err(format!("cuMemcpyHtoD_v2(guard-check init) failed: {:?}", copy_err));
    }
    Ok(ptr)
}

fn with_primary_context<T, F>(device_ordinal: i32, f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    unsafe {
        let mut ctx: cuda_sys::CUcontext = std::ptr::null_mut();
        let retain_err = cuda_sys::lib().cuDevicePrimaryCtxRetain(&mut ctx, device_ordinal);
        if retain_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuDevicePrimaryCtxRetain(device={}) failed: {:?}",
                device_ordinal, retain_err
            ));
        }
        let set_err = cuda_sys::lib().cuCtxSetCurrent(ctx);
        if set_err != cuda_sys::CUresult::CUDA_SUCCESS {
            cuda_sys::lib().cuDevicePrimaryCtxRelease_v2(device_ordinal);
            return Err(format!(
                "cuCtxSetCurrent(device={}) failed: {:?}",
                device_ordinal, set_err
            ));
        }
        let result = f();
        let release_err = cuda_sys::lib().cuDevicePrimaryCtxRelease_v2(device_ordinal);
        if release_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuDevicePrimaryCtxRelease_v2(device={}) failed: {:?}",
                device_ordinal, release_err
            ));
        }
        result
    }
}

fn build_guard_bytes(meta: &AllocationMeta, is_prefix: bool, live: bool) -> Vec<u8> {
    let len = if is_prefix { meta.guard_before } else { meta.guard_after };
    let mut guard = vec![if live { CUDA_DEBUG_ALLOC_PATTERN } else { CUDA_DEBUG_FREED_PATTERN }; len];
    if len == 0 {
        return guard;
    }

    let pattern = if !live {
        CUDA_DEBUG_FREED_PATTERN
    } else if is_prefix {
        CUDA_DEBUG_PREFIX_PATTERN
    } else {
        CUDA_DEBUG_SUFFIX_PATTERN
    };
    guard.fill(pattern);

    write_u64(&mut guard, 0, meta.alloc_id);
    write_u64(&mut guard, 8, meta.generation);
    write_u64(&mut guard, 16, meta.payload_bytes as u64);
    write_u64(&mut guard, 24, meta.tag_hash);
    if len >= 40 {
        write_u64(&mut guard, len.saturating_sub(16), meta.alloc_id ^ meta.tag_hash);
        write_u64(&mut guard, len.saturating_sub(8), meta.generation ^ meta.payload_bytes as u64);
    }
    guard
}

fn write_u64(buf: &mut [u8], offset: usize, value: u64) {
    if offset + 8 <= buf.len() {
        buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}

fn free_raw_allocation(raw_ptr: u64, device_ordinal: i32) -> Result<(), String> {
    unsafe {
        let mut ctx: cuda_sys::CUcontext = std::ptr::null_mut();
        let retain_err = cuda_sys::lib().cuDevicePrimaryCtxRetain(&mut ctx, device_ordinal);
        if retain_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuDevicePrimaryCtxRetain(device={}) failed during debug free: {:?}",
                device_ordinal, retain_err
            ));
        }
        let set_err = cuda_sys::lib().cuCtxSetCurrent(ctx);
        if set_err != cuda_sys::CUresult::CUDA_SUCCESS {
            cuda_sys::lib().cuDevicePrimaryCtxRelease_v2(device_ordinal);
            return Err(format!(
                "cuCtxSetCurrent(device={}) failed during debug free: {:?}",
                device_ordinal, set_err
            ));
        }
        let free_err = cuda_sys::lib().cuMemFree_v2(raw_ptr);
        cuda_sys::lib().cuDevicePrimaryCtxRelease_v2(device_ordinal);
        if free_err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuMemFree_v2(ptr=0x{:x}, device={}) failed during debug free: {:?}",
                raw_ptr, device_ordinal, free_err
            ));
        }
    }
    Ok(())
}

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
#[cfg(has_hqq_search_kernels)]
use std::sync::Arc;

#[cfg(has_hqq_search_kernels)]
use cudarc::driver::sys as cuda_sys;
#[cfg(has_hqq_search_kernels)]
use cudarc::driver::{CudaDevice, CudaFunction};

const HQQ4_QMAX: f32 = 15.0;
const HQQ4_SCALE_EPS: f32 = 1e-8;
const HQQ4_DENOM_EPS: f32 = 1e-12;
const HQQ4_GLOBAL_STEPS: usize = 65;
const HQQ4_LOCAL_STEPS: usize = 33;
const HQQ4_ITERS: usize = 6;

#[cfg(has_hqq_search_kernels)]
const HQQ_SEARCH_KERNELS_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/hqq_search_kernels.ptx"));
#[cfg(has_hqq_search_kernels)]
const HQQ_SEARCH_MODULE_NAME: &str = "hqq_search_kernels";
#[cfg(has_hqq_search_kernels)]
const HQQ_SEARCH_KERNEL_NAMES: &[&str] = &[
    "hqq_search_global_kernel",
    "hqq_search_zero_bounds_kernel",
    "hqq_search_local_kernel",
];

#[derive(Clone)]
struct RowEval {
    q: Vec<u8>,
    scale: f32,
    zero: f32,
    rmse: f32,
}

#[derive(Clone)]
struct RowInit {
    q: Vec<u8>,
    scale: f32,
    zero: f32,
}

fn hqq_num_groups(cols: usize, group_size: usize) -> usize {
    (cols + group_size - 1) / group_size
}

fn hqq_padded_cols(cols: usize, group_size: usize) -> usize {
    hqq_num_groups(cols, group_size) * group_size
}

fn hqq_packed_cols(cols: usize, group_size: usize) -> usize {
    (hqq_padded_cols(cols, group_size) + 1) / 2
}

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn make_linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start];
    }
    let denom = (steps - 1) as f64;
    let start64 = start as f64;
    let delta = (end as f64) - start64;
    (0..steps)
        .map(|idx| (start64 + delta * ((idx as f64) / denom)) as f32)
        .collect()
}

fn quantize_row_with_scale_zero_solve(chunk: &[f32], scale: f32, zero: f32) -> Vec<u8> {
    chunk
        .iter()
        .map(|&value| {
            let q = ((value / scale) + zero).round_ties_even();
            clamp(q, 0.0, HQQ4_QMAX) as u8
        })
        .collect()
}

fn quantize_row_with_scale_zero_init(chunk: &[f32], scale: f32, zero: f32) -> Vec<u8> {
    chunk
        .iter()
        .map(|&value| {
            let q = ((value / scale) + zero).round_ties_even();
            clamp(q, 0.0, HQQ4_QMAX) as u8
        })
        .collect()
}

fn quantize_row_with_scale_zero_into(chunk: &[f32], scale: f32, zero: f32, q: &mut [u8]) {
    for (dst, &value) in q.iter_mut().zip(chunk.iter()) {
        let quant = ((value / scale) + zero).round_ties_even();
        *dst = clamp(quant, 0.0, HQQ4_QMAX) as u8;
    }
}

fn quantize_group_current_row(chunk: &[f32]) -> (Vec<u8>, f32, f32) {
    let mut minv = f32::INFINITY;
    let mut maxv = f32::NEG_INFINITY;
    for &value in chunk {
        minv = minv.min(value);
        maxv = maxv.max(value);
    }
    let scale = ((maxv - minv) / HQQ4_QMAX).max(HQQ4_SCALE_EPS);
    let zero = clamp(-minv / scale, 0.0, HQQ4_QMAX);
    let q = quantize_row_with_scale_zero_init(chunk, scale, zero);
    (q, scale, zero)
}

fn compute_rmse(chunk: &[f32], q: &[u8], scale: f32, zero: f32) -> f32 {
    let sum_sq = reduce_like_torch_f32(chunk.iter().zip(q.iter()).map(|(&value, &qv)| {
        let deq = ((qv as f32) - zero) * scale;
        let diff = deq - value;
        diff * diff
    }));
    sum_sq / (chunk.len() as f32)
}

fn reduce_like_torch_f32(values: impl Iterator<Item = f32>) -> f32 {
    // Match torch CPU's vectorized float32 reduction shape on this path:
    // four 8-wide accumulators, lane-wise merge, then horizontal sum.
    let mut acc = [[0.0f32; 8]; 4];
    let mut idx = 0usize;
    for value in values {
        let block = idx & 31;
        let vec_idx = block >> 3;
        let lane_idx = block & 7;
        acc[vec_idx][lane_idx] += value;
        idx += 1;
    }
    let mut merged = [0.0f32; 8];
    for lane_idx in 0..8 {
        let mut lane_sum = 0.0f32;
        for vec_idx in 0..4 {
            lane_sum += acc[vec_idx][lane_idx];
        }
        merged[lane_idx] = lane_sum;
    }
    let mut total = 0.0f32;
    for lane_sum in merged {
        total += lane_sum;
    }
    total
}

fn compute_rmse_from_quantized_row(chunk: &[f32], q: &[u8], scale: f32, zero: f32) -> f32 {
    let mut acc = [[0.0f32; 8]; 4];
    for (idx, (&value, &qv)) in chunk.iter().zip(q.iter()).enumerate() {
        let deq = ((qv as f32) - zero) * scale;
        let diff = deq - value;
        let block = idx & 31;
        let vec_idx = block >> 3;
        let lane_idx = block & 7;
        acc[vec_idx][lane_idx] += diff * diff;
    }
    let mut total = 0.0f32;
    for lane_idx in 0..8 {
        let mut lane_sum = 0.0f32;
        for vec_idx in 0..4 {
            lane_sum += acc[vec_idx][lane_idx];
        }
        total += lane_sum;
    }
    total / (chunk.len() as f32)
}

fn fixed_zero_denom_numer(chunk: &[f32], q: &[u8], zero: f32) -> (f32, f32) {
    let mut denom_acc = [[0.0f32; 8]; 4];
    let mut numer_acc = [[0.0f32; 8]; 4];
    for (idx, (&value, &qv)) in chunk.iter().zip(q.iter()).enumerate() {
        let centered = (qv as f32) - zero;
        let block = idx & 31;
        let vec_idx = block >> 3;
        let lane_idx = block & 7;
        denom_acc[vec_idx][lane_idx] += centered * centered;
        numer_acc[vec_idx][lane_idx] += value * centered;
    }

    let mut denom = 0.0f32;
    let mut numer = 0.0f32;
    for lane_idx in 0..8 {
        let mut denom_lane = 0.0f32;
        let mut numer_lane = 0.0f32;
        for vec_idx in 0..4 {
            denom_lane += denom_acc[vec_idx][lane_idx];
            numer_lane += numer_acc[vec_idx][lane_idx];
        }
        denom += denom_lane;
        numer += numer_lane;
    }
    (denom, numer)
}

fn solve_fixed_zero_row(chunk: &[f32], zero: f32, scale_seed: f32) -> (Vec<u8>, f32) {
    let mut scale = scale_seed;
    for _ in 0..HQQ4_ITERS {
        let q = quantize_row_with_scale_zero_solve(chunk, scale, zero);
        let centered: Vec<f32> = q.iter().map(|&qv| (qv as f32) - zero).collect();
        let denom = reduce_like_torch_f32(centered.iter().map(|&v| v * v));
        let numer = reduce_like_torch_f32(
            chunk
                .iter()
                .zip(centered.iter())
                .map(|(&value, &centered)| value * centered),
        );
        if denom > HQQ4_DENOM_EPS {
            scale = (numer / denom).max(HQQ4_SCALE_EPS);
        } else {
            scale = scale.max(HQQ4_SCALE_EPS);
        }
    }
    (quantize_row_with_scale_zero_solve(chunk, scale, zero), scale)
}

fn solve_fixed_zero_row_into(chunk: &[f32], zero: f32, scale_seed: f32, q: &mut [u8]) -> f32 {
    let mut scale = scale_seed;
    for _ in 0..HQQ4_ITERS {
        quantize_row_with_scale_zero_into(chunk, scale, zero, q);
        let (denom, numer) = fixed_zero_denom_numer(chunk, q, zero);
        if denom > HQQ4_DENOM_EPS {
            scale = (numer / denom).max(HQQ4_SCALE_EPS);
        } else {
            scale = scale.max(HQQ4_SCALE_EPS);
        }
    }
    quantize_row_with_scale_zero_into(chunk, scale, zero, q);
    scale
}

fn update_best_group_fit(
    chunk: &[f32],
    rows: usize,
    chunk_cols: usize,
    phases: &[(&[f32], &[f32])],
    best_q: &mut [u8],
    best_scale: &mut [f32],
    best_zero: &mut [f32],
    best_rmse: &mut [f32],
) {
    let evals: Vec<Option<RowEval>> = (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &chunk[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
            let mut local_best_rmse = best_rmse[row_idx];
            let mut local_best_scale = best_scale[row_idx];
            let mut local_best_zero = best_zero[row_idx];
            let mut local_best_q: Option<Vec<u8>> = None;
            let mut candidate_q = vec![0u8; chunk_cols];
            for &(zero_grid, scale_seed) in phases {
                for &zero in zero_grid {
                    let scale = solve_fixed_zero_row_into(row, zero, scale_seed[row_idx], &mut candidate_q);
                    let rmse = compute_rmse_from_quantized_row(row, &candidate_q, scale, zero);
                    if rmse < local_best_rmse {
                        local_best_rmse = rmse;
                        local_best_scale = scale;
                        local_best_zero = zero;
                        let best_candidate = local_best_q.get_or_insert_with(|| vec![0u8; chunk_cols]);
                        best_candidate.copy_from_slice(&candidate_q);
                    }
                }
            }
            local_best_q.map(|q| RowEval {
                q,
                scale: local_best_scale,
                zero: local_best_zero,
                rmse: local_best_rmse,
            })
        })
        .collect();

    for (row_idx, eval) in evals.into_iter().enumerate() {
        if let Some(eval) = eval {
            best_rmse[row_idx] = eval.rmse;
            best_scale[row_idx] = eval.scale;
            best_zero[row_idx] = eval.zero;
            let dst = &mut best_q[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
            dst.copy_from_slice(&eval.q);
        }
    }
}

fn refine_group_fit(chunk: &[f32], rows: usize, chunk_cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let init_rows: Vec<(Vec<u8>, f32, f32, f32)> = (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &chunk[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
            let (q, scale, zero) = quantize_group_current_row(row);
            let rmse = compute_rmse(row, &q, scale, zero);
            (q, scale, zero, rmse)
        })
        .collect();

    let mut best_q = vec![0u8; rows * chunk_cols];
    let mut best_scale = vec![0.0f32; rows];
    let mut best_zero = vec![0.0f32; rows];
    let mut best_rmse = vec![0.0f32; rows];
    let mut range_scale = vec![0.0f32; rows];
    let mut abs_scale = vec![0.0f32; rows];

    for (row_idx, (q, scale, zero, rmse)) in init_rows.into_iter().enumerate() {
        best_q[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)].copy_from_slice(&q);
        best_scale[row_idx] = scale;
        best_zero[row_idx] = zero;
        best_rmse[row_idx] = rmse;

        let row = &chunk[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
        let mut minv = f32::INFINITY;
        let mut maxv = f32::NEG_INFINITY;
        let mut amax = 0.0f32;
        for &value in row {
            minv = minv.min(value);
            maxv = maxv.max(value);
            amax = amax.max(value.abs());
        }
        range_scale[row_idx] = ((maxv - minv) / HQQ4_QMAX).max(HQQ4_SCALE_EPS);
        abs_scale[row_idx] = ((2.0 * amax) / HQQ4_QMAX).max(HQQ4_SCALE_EPS);
    }

    let global_zero_grid = make_linspace(0.0, HQQ4_QMAX, HQQ4_GLOBAL_STEPS);
    update_best_group_fit(
        chunk,
        rows,
        chunk_cols,
        &[(&global_zero_grid, &range_scale), (&global_zero_grid, &abs_scale)],
        &mut best_q,
        &mut best_scale,
        &mut best_zero,
        &mut best_rmse,
    );

    let mut local_zero_min = f32::INFINITY;
    let mut local_zero_max = f32::NEG_INFINITY;
    for &zero in &best_zero {
        local_zero_min = local_zero_min.min(zero);
        local_zero_max = local_zero_max.max(zero);
    }
    local_zero_min = (local_zero_min - 0.5).max(0.0);
    local_zero_max = (local_zero_max + 0.5).min(HQQ4_QMAX);
    let local_zero_grid = make_linspace(local_zero_min, local_zero_max, HQQ4_LOCAL_STEPS);
    let local_scale_seed = best_scale.clone();
    update_best_group_fit(
        chunk,
        rows,
        chunk_cols,
        &[(&local_zero_grid, &local_scale_seed)],
        &mut best_q,
        &mut best_scale,
        &mut best_zero,
        &mut best_rmse,
    );

    (best_q, best_scale, best_zero)
}

fn quantize_group_current_rows(chunk: &[f32], rows: usize, chunk_cols: usize) -> Vec<RowInit> {
    (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &chunk[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
            let (q, scale, zero) = quantize_group_current_row(row);
            RowInit { q, scale, zero }
        })
        .collect()
}

#[cfg(has_hqq_search_kernels)]
#[derive(Clone, Copy)]
struct RawCuFunc(cuda_sys::CUfunction);

#[cfg(has_hqq_search_kernels)]
fn extract_cu_func(func: &CudaFunction) -> RawCuFunc {
    unsafe {
        let struct_ptr = func as *const _ as *const u8;
        let word0: cuda_sys::CUfunction = std::ptr::read(struct_ptr as *const _);
        let mut dummy = 0i32;
        let w0_valid = cuda_sys::lib().cuFuncGetAttribute(
            &mut dummy,
            cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
            word0,
        ) == cuda_sys::CUresult::CUDA_SUCCESS;
        RawCuFunc(if w0_valid {
            word0
        } else {
            std::ptr::read(struct_ptr.add(8) as *const _)
        })
    }
}

#[cfg(has_hqq_search_kernels)]
unsafe fn launch_kernel(
    func: RawCuFunc,
    grid_x: u32,
    block_x: u32,
    params: &mut [*mut std::ffi::c_void],
) -> Result<(), String> {
    let err = cuda_sys::lib().cuLaunchKernel(
        func.0,
        grid_x,
        1,
        1,
        block_x,
        1,
        1,
        0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if err == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "HQQ CUDA search kernel launch failed: {:?} (grid_x={grid_x}, block_x={block_x})",
            err
        ))
    }
}

#[cfg(has_hqq_search_kernels)]
fn load_hqq_search_func(device: &Arc<CudaDevice>, name: &str) -> PyResult<RawCuFunc> {
    let func = device
        .get_func(HQQ_SEARCH_MODULE_NAME, name)
        .ok_or_else(|| PyRuntimeError::new_err(format!("HQQ CUDA search kernel not found: {name}")))?;
    Ok(extract_cu_func(&func))
}

#[cfg(has_hqq_search_kernels)]
fn cuda_check_sync(label: &str) -> PyResult<()> {
    let err = unsafe { cuda_sys::lib().cuCtxSynchronize() };
    if err == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(format!("{label}: {:?}", err)))
    }
}

#[cfg(has_hqq_search_kernels)]
fn cuda_alloc_f32(count: usize) -> PyResult<cuda_sys::CUdeviceptr> {
    let mut ptr: cuda_sys::CUdeviceptr = 0;
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| PyValueError::new_err("HQQ CUDA scratch allocation overflow"))?;
    let err = unsafe { cuda_sys::lib().cuMemAlloc_v2(&mut ptr, bytes) };
    if err == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(ptr)
    } else {
        Err(PyRuntimeError::new_err(format!(
            "HQQ CUDA scratch allocation failed for {bytes} bytes: {:?}",
            err
        )))
    }
}

#[cfg(has_hqq_search_kernels)]
fn cuda_free(ptr: cuda_sys::CUdeviceptr) {
    if ptr != 0 {
        let _ = unsafe { cuda_sys::lib().cuMemFree_v2(ptr) };
    }
}

#[pyfunction]
#[pyo3(signature = (
    input_ptr,
    rows,
    cols,
    group_size,
    nbits,
    global_steps,
    local_steps,
    iters,
    quant_ptr,
    scales_ptr,
    zeros_ptr,
    device_idx = 0
))]
#[allow(clippy::too_many_arguments)]
pub fn hqq_search_cuda_tensor_ptr(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    nbits: usize,
    global_steps: usize,
    local_steps: usize,
    iters: usize,
    quant_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
    device_idx: usize,
) -> PyResult<()> {
    hqq_search_cuda_tensor_ptr_impl(
        input_ptr,
        rows,
        cols,
        group_size,
        nbits,
        global_steps,
        local_steps,
        iters,
        quant_ptr,
        scales_ptr,
        zeros_ptr,
        device_idx,
    )
}

#[cfg(not(has_hqq_search_kernels))]
#[allow(clippy::too_many_arguments)]
fn hqq_search_cuda_tensor_ptr_impl(
    _input_ptr: usize,
    _rows: usize,
    _cols: usize,
    _group_size: usize,
    _nbits: usize,
    _global_steps: usize,
    _local_steps: usize,
    _iters: usize,
    _quant_ptr: usize,
    _scales_ptr: usize,
    _zeros_ptr: usize,
    _device_idx: usize,
) -> PyResult<()> {
    Err(PyRuntimeError::new_err(
        "HQQ CUDA search kernels were not compiled in this build",
    ))
}

#[cfg(has_hqq_search_kernels)]
#[allow(clippy::too_many_arguments)]
fn hqq_search_cuda_tensor_ptr_impl(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    nbits: usize,
    global_steps: usize,
    local_steps: usize,
    iters: usize,
    quant_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
    device_idx: usize,
) -> PyResult<()> {
    if input_ptr == 0 || quant_ptr == 0 || scales_ptr == 0 || zeros_ptr == 0 {
        return Err(PyValueError::new_err("HQQ CUDA search received a null pointer"));
    }
    if rows == 0 || cols == 0 || group_size == 0 {
        return Err(PyValueError::new_err(
            "HQQ CUDA search expects positive rows, cols, and group_size",
        ));
    }
    if !matches!(nbits, 4 | 6 | 8) {
        return Err(PyValueError::new_err("HQQ CUDA search supports nbits 4, 6, or 8"));
    }
    if group_size > 256 {
        return Err(PyValueError::new_err(
            "HQQ CUDA search prototype currently supports group_size <= 256",
        ));
    }
    if global_steps == 0 || local_steps == 0 || iters == 0 {
        return Err(PyValueError::new_err(
            "HQQ CUDA search expects positive global_steps, local_steps, and iters",
        ));
    }

    let groups = hqq_num_groups(cols, group_size);
    let padded_cols = hqq_padded_cols(cols, group_size);
    let row_groups = rows
        .checked_mul(groups)
        .ok_or_else(|| PyValueError::new_err("rows * groups overflowed"))?;
    if row_groups > u32::MAX as usize {
        return Err(PyValueError::new_err("HQQ CUDA search grid exceeds u32::MAX"));
    }
    if groups > u32::MAX as usize {
        return Err(PyValueError::new_err("HQQ CUDA search group grid exceeds u32::MAX"));
    }

    let device = CudaDevice::new(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("HQQ CUDA device init failed: {e}")))?;
    {
        use cudarc::nvrtc::Ptx;
        device
            .load_ptx(
                Ptx::from_src(HQQ_SEARCH_KERNELS_PTX),
                HQQ_SEARCH_MODULE_NAME,
                HQQ_SEARCH_KERNEL_NAMES,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Load HQQ CUDA search PTX: {e}")))?;
    }

    let global_func = load_hqq_search_func(&device, "hqq_search_global_kernel")?;
    let bounds_func = load_hqq_search_func(&device, "hqq_search_zero_bounds_kernel")?;
    let local_func = load_hqq_search_func(&device, "hqq_search_local_kernel")?;

    let best_rmse = cuda_alloc_f32(row_groups)?;
    let local_min = cuda_alloc_f32(groups)?;
    let local_max = cuda_alloc_f32(groups)?;
    let result = (|| -> PyResult<()> {
        let block_x = group_size.next_power_of_two().clamp(32, 256) as u32;
        let bounds_block_x = 256u32;
        let qmax = ((1u32 << nbits) - 1) as f32;

        let mut p_input = input_ptr as u64;
        let mut p_rows = rows as i32;
        let mut p_cols = cols as i32;
        let mut p_group_size = group_size as i32;
        let mut p_groups = groups as i32;
        let mut p_padded_cols = padded_cols as i32;
        let mut p_nbits = nbits as i32;
        let mut p_global_steps = global_steps as i32;
        let mut p_iters = iters as i32;
        let mut p_quant = quant_ptr as u64;
        let mut p_scales = scales_ptr as u64;
        let mut p_zeros = zeros_ptr as u64;
        let mut p_best_rmse = best_rmse;
        let mut global_params: [*mut std::ffi::c_void; 13] = [
            &mut p_input as *mut _ as *mut std::ffi::c_void,
            &mut p_rows as *mut _ as *mut std::ffi::c_void,
            &mut p_cols as *mut _ as *mut std::ffi::c_void,
            &mut p_group_size as *mut _ as *mut std::ffi::c_void,
            &mut p_groups as *mut _ as *mut std::ffi::c_void,
            &mut p_padded_cols as *mut _ as *mut std::ffi::c_void,
            &mut p_nbits as *mut _ as *mut std::ffi::c_void,
            &mut p_global_steps as *mut _ as *mut std::ffi::c_void,
            &mut p_iters as *mut _ as *mut std::ffi::c_void,
            &mut p_quant as *mut _ as *mut std::ffi::c_void,
            &mut p_scales as *mut _ as *mut std::ffi::c_void,
            &mut p_zeros as *mut _ as *mut std::ffi::c_void,
            &mut p_best_rmse as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel(global_func, row_groups as u32, block_x, &mut global_params)
                .map_err(PyRuntimeError::new_err)?;
        }

        let mut b_zeros = zeros_ptr as u64;
        let mut b_rows = rows as i32;
        let mut b_groups = groups as i32;
        let mut b_qmax = qmax;
        let mut b_local_min = local_min;
        let mut b_local_max = local_max;
        let mut bounds_params: [*mut std::ffi::c_void; 6] = [
            &mut b_zeros as *mut _ as *mut std::ffi::c_void,
            &mut b_rows as *mut _ as *mut std::ffi::c_void,
            &mut b_groups as *mut _ as *mut std::ffi::c_void,
            &mut b_qmax as *mut _ as *mut std::ffi::c_void,
            &mut b_local_min as *mut _ as *mut std::ffi::c_void,
            &mut b_local_max as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel(bounds_func, groups as u32, bounds_block_x, &mut bounds_params)
                .map_err(PyRuntimeError::new_err)?;
        }

        let mut l_input = input_ptr as u64;
        let mut l_rows = rows as i32;
        let mut l_cols = cols as i32;
        let mut l_group_size = group_size as i32;
        let mut l_groups = groups as i32;
        let mut l_padded_cols = padded_cols as i32;
        let mut l_nbits = nbits as i32;
        let mut l_local_steps = local_steps as i32;
        let mut l_iters = iters as i32;
        let mut l_local_min = local_min;
        let mut l_local_max = local_max;
        let mut l_quant = quant_ptr as u64;
        let mut l_scales = scales_ptr as u64;
        let mut l_zeros = zeros_ptr as u64;
        let mut l_best_rmse = best_rmse;
        let mut local_params: [*mut std::ffi::c_void; 15] = [
            &mut l_input as *mut _ as *mut std::ffi::c_void,
            &mut l_rows as *mut _ as *mut std::ffi::c_void,
            &mut l_cols as *mut _ as *mut std::ffi::c_void,
            &mut l_group_size as *mut _ as *mut std::ffi::c_void,
            &mut l_groups as *mut _ as *mut std::ffi::c_void,
            &mut l_padded_cols as *mut _ as *mut std::ffi::c_void,
            &mut l_nbits as *mut _ as *mut std::ffi::c_void,
            &mut l_local_steps as *mut _ as *mut std::ffi::c_void,
            &mut l_iters as *mut _ as *mut std::ffi::c_void,
            &mut l_local_min as *mut _ as *mut std::ffi::c_void,
            &mut l_local_max as *mut _ as *mut std::ffi::c_void,
            &mut l_quant as *mut _ as *mut std::ffi::c_void,
            &mut l_scales as *mut _ as *mut std::ffi::c_void,
            &mut l_zeros as *mut _ as *mut std::ffi::c_void,
            &mut l_best_rmse as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel(local_func, row_groups as u32, block_x, &mut local_params)
                .map_err(PyRuntimeError::new_err)?;
        }

        cuda_check_sync("HQQ CUDA search synchronize")
    })();
    cuda_free(best_rmse);
    cuda_free(local_min);
    cuda_free(local_max);
    result
}

#[pyfunction]
pub fn hqq4_init_group_ptr(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    q_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
) -> PyResult<()> {
    if input_ptr == 0 || q_ptr == 0 || scales_ptr == 0 || zeros_ptr == 0 {
        return Err(PyValueError::new_err("HQQ Rust init received a null pointer"));
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("HQQ Rust init expects positive rows and cols"));
    }

    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed"))?;
    let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
    let q = unsafe { std::slice::from_raw_parts_mut(q_ptr as *mut u8, input_len) };
    let scales = unsafe { std::slice::from_raw_parts_mut(scales_ptr as *mut f32, rows) };
    let zeros = unsafe { std::slice::from_raw_parts_mut(zeros_ptr as *mut f32, rows) };

    let init_rows = quantize_group_current_rows(input, rows, cols);
    for (row_idx, init) in init_rows.into_iter().enumerate() {
        let dst = &mut q[(row_idx * cols)..((row_idx + 1) * cols)];
        dst.copy_from_slice(&init.q);
        scales[row_idx] = init.scale;
        zeros[row_idx] = init.zero;
    }

    Ok(())
}

#[pyfunction]
pub fn hqq4_solve_group_ptr(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    zero_ptr: usize,
    scale_seed_ptr: usize,
    q_ptr: usize,
    scales_ptr: usize,
) -> PyResult<()> {
    if input_ptr == 0
        || zero_ptr == 0
        || scale_seed_ptr == 0
        || q_ptr == 0
        || scales_ptr == 0
    {
        return Err(PyValueError::new_err("HQQ Rust solve received a null pointer"));
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("HQQ Rust solve expects positive rows and cols"));
    }

    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed"))?;
    let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
    let zero = unsafe { std::slice::from_raw_parts(zero_ptr as *const f32, rows) };
    let scale_seed = unsafe { std::slice::from_raw_parts(scale_seed_ptr as *const f32, rows) };
    let q = unsafe { std::slice::from_raw_parts_mut(q_ptr as *mut u8, input_len) };
    let scales = unsafe { std::slice::from_raw_parts_mut(scales_ptr as *mut f32, rows) };

    let solved_rows: Vec<(Vec<u8>, f32)> = (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &input[(row_idx * cols)..((row_idx + 1) * cols)];
            solve_fixed_zero_row(row, zero[row_idx], scale_seed[row_idx])
        })
        .collect();

    for (row_idx, (row_q, scale)) in solved_rows.into_iter().enumerate() {
        let dst = &mut q[(row_idx * cols)..((row_idx + 1) * cols)];
        dst.copy_from_slice(&row_q);
        scales[row_idx] = scale;
    }

    Ok(())
}

#[pyfunction]
pub fn hqq4_rmse_group_ptr(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    q_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
    rmse_ptr: usize,
) -> PyResult<()> {
    if input_ptr == 0
        || q_ptr == 0
        || scales_ptr == 0
        || zeros_ptr == 0
        || rmse_ptr == 0
    {
        return Err(PyValueError::new_err("HQQ Rust RMSE received a null pointer"));
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("HQQ Rust RMSE expects positive rows and cols"));
    }

    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed"))?;
    let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
    let q = unsafe { std::slice::from_raw_parts(q_ptr as *const u8, input_len) };
    let scales = unsafe { std::slice::from_raw_parts(scales_ptr as *const f32, rows) };
    let zeros = unsafe { std::slice::from_raw_parts(zeros_ptr as *const f32, rows) };
    let rmse = unsafe { std::slice::from_raw_parts_mut(rmse_ptr as *mut f32, rows) };

    let rmse_rows: Vec<f32> = (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &input[(row_idx * cols)..((row_idx + 1) * cols)];
            let row_q = &q[(row_idx * cols)..((row_idx + 1) * cols)];
            compute_rmse(row, row_q, scales[row_idx], zeros[row_idx])
        })
        .collect();

    rmse.copy_from_slice(&rmse_rows);
    Ok(())
}

fn hqq4_quantize_tensor_impl(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
    inner_threads: Option<usize>,
) -> PyResult<()> {
    if input_ptr == 0 || packed_ptr == 0 || scales_ptr == 0 || zeros_ptr == 0 {
        return Err(PyValueError::new_err("HQQ Rust quantizer received a null pointer"));
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("HQQ Rust quantizer expects positive rows and cols"));
    }
    if group_size == 0 {
        return Err(PyValueError::new_err("HQQ Rust quantizer expects positive group_size"));
    }

    let groups = hqq_num_groups(cols, group_size);
    let padded_cols = hqq_padded_cols(cols, group_size);
    let packed_cols = hqq_packed_cols(cols, group_size);
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflowed"))?;
    let packed_len = rows
        .checked_mul(packed_cols)
        .ok_or_else(|| PyValueError::new_err("rows * packed_cols overflowed"))?;
    let group_meta_len = rows
        .checked_mul(groups)
        .ok_or_else(|| PyValueError::new_err("rows * groups overflowed"))?;

    let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
    let scales = unsafe { std::slice::from_raw_parts_mut(scales_ptr as *mut f32, group_meta_len) };
    let zeros = unsafe { std::slice::from_raw_parts_mut(zeros_ptr as *mut f32, group_meta_len) };
    let packed = unsafe { std::slice::from_raw_parts_mut(packed_ptr as *mut u8, packed_len) };

    let mut quant = vec![0u8; rows * padded_cols];
    let mut run_quantize = || {
        for group_idx in 0..groups {
            let start = group_idx * group_size;
            let end = (start + group_size).min(cols);
            let chunk_cols = end - start;
            let mut chunk = vec![0.0f32; rows * chunk_cols];
            for row_idx in 0..rows {
                let src_start = row_idx * cols + start;
                let src_end = src_start + chunk_cols;
                let dst_start = row_idx * chunk_cols;
                let dst_end = dst_start + chunk_cols;
                chunk[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
            }

            let (best_q, best_scale, best_zero) = refine_group_fit(&chunk, rows, chunk_cols);
            for row_idx in 0..rows {
                let quant_row_offset = row_idx * padded_cols + start;
                let q_src_offset = row_idx * chunk_cols;
                let q_src_end = q_src_offset + chunk_cols;
                quant[quant_row_offset..quant_row_offset + chunk_cols]
                    .copy_from_slice(&best_q[q_src_offset..q_src_end]);
                scales[row_idx * groups + group_idx] = best_scale[row_idx];
                zeros[row_idx * groups + group_idx] = best_zero[row_idx];
            }
        }
    };

    if let Some(thread_count) = inner_threads.filter(|&n| n > 0) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|exc| PyRuntimeError::new_err(format!("failed to build HQQ4 local rayon pool: {exc}")))?
            .install(run_quantize);
    } else {
        run_quantize();
    }

    for row_idx in 0..rows {
        let quant_row = &quant[(row_idx * padded_cols)..((row_idx + 1) * padded_cols)];
        let packed_row = &mut packed[(row_idx * packed_cols)..((row_idx + 1) * packed_cols)];
        for out_idx in 0..packed_cols {
            let lo = quant_row[out_idx * 2] & 0x0f;
            let hi = if out_idx * 2 + 1 < padded_cols {
                (quant_row[out_idx * 2 + 1] & 0x0f) << 4
            } else {
                0
            };
            packed_row[out_idx] = lo | hi;
        }
    }

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    input_ptr,
    rows,
    cols,
    group_size,
    packed_ptr,
    scales_ptr,
    zeros_ptr,
    inner_threads = None
))]
pub fn hqq4_quantize_tensor_ptr(
    py: Python<'_>,
    input_ptr: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
    inner_threads: Option<usize>,
) -> PyResult<()> {
    py.allow_threads(move || {
        hqq4_quantize_tensor_impl(
            input_ptr,
            rows,
            cols,
            group_size,
            packed_ptr,
            scales_ptr,
            zeros_ptr,
            inner_threads,
        )
    })
}

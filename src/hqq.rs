use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const HQQ4_QMAX: f32 = 15.0;
const HQQ4_SCALE_EPS: f32 = 1e-8;
const HQQ4_DENOM_EPS: f32 = 1e-12;
const HQQ4_GLOBAL_STEPS: usize = 129;
const HQQ4_LOCAL_STEPS: usize = 65;
const HQQ4_ITERS: usize = 6;

#[derive(Clone)]
struct RowEval {
    q: Vec<u8>,
    scale: f32,
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

fn update_best_group_fit(
    chunk: &[f32],
    rows: usize,
    chunk_cols: usize,
    zero_grid: &[f32],
    scale_seed: &[f32],
    best_q: &mut [u8],
    best_scale: &mut [f32],
    best_zero: &mut [f32],
    best_rmse: &mut [f32],
) {
    for &zero in zero_grid {
        let evals: Vec<RowEval> = (0..rows)
            .into_par_iter()
            .map(|row_idx| {
                let row = &chunk[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
                let (q, scale) = solve_fixed_zero_row(row, zero, scale_seed[row_idx]);
                let rmse = compute_rmse(row, &q, scale, zero);
                RowEval { q, scale, rmse }
            })
            .collect();

        for (row_idx, eval) in evals.into_iter().enumerate() {
            if eval.rmse < best_rmse[row_idx] {
                best_rmse[row_idx] = eval.rmse;
                best_scale[row_idx] = eval.scale;
                best_zero[row_idx] = zero;
                let dst = &mut best_q[(row_idx * chunk_cols)..((row_idx + 1) * chunk_cols)];
                dst.copy_from_slice(&eval.q);
            }
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
        &global_zero_grid,
        &range_scale,
        &mut best_q,
        &mut best_scale,
        &mut best_zero,
        &mut best_rmse,
    );
    update_best_group_fit(
        chunk,
        rows,
        chunk_cols,
        &global_zero_grid,
        &abs_scale,
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
        &local_zero_grid,
        &local_scale_seed,
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

#[pyfunction]
pub fn hqq4_quantize_tensor_ptr(
    input_ptr: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_ptr: usize,
    scales_ptr: usize,
    zeros_ptr: usize,
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

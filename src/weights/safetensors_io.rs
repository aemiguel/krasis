//! Safetensors file reading via mmap.
//!
//! Provides zero-copy access to weight tensors stored in safetensors format.
//! Files are mmap'd so the OS manages page-in/eviction automatically.
//!
//! Safetensors format:
//!   [8 bytes LE u64: header_size]
//!   [header_size bytes: JSON header]
//!   [tensor data...]
//!
//! JSON header maps tensor names to {dtype, shape, data_offsets: [start, end]}.
//! Offsets are relative to the end of the header.

use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SafetensorsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("File too small to contain header")]
    FileTooSmall,
    #[error("Header parse error: {0}")]
    HeaderParse(#[from] serde_json::Error),
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    #[error("Data offset out of bounds for tensor {0}")]
    OffsetOutOfBounds(String),
}

/// Tensor dtype as stored in safetensors header.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Dtype {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    #[serde(alias = "BF16")]
    Bf16,
    F32,
    F64,
}

impl Dtype {
    /// Bytes per element.
    pub fn element_size(&self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::I16 | Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::I32 | Dtype::F32 => 4,
            Dtype::I64 | Dtype::F64 => 8,
        }
    }
}

/// Metadata for a single tensor in the file.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

impl TensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

/// A memory-mapped safetensors file with parsed tensor index.
pub struct MmapSafetensors {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl MmapSafetensors {
    /// Open and mmap a safetensors file, parse the header.
    pub fn open(path: &Path) -> Result<Self, SafetensorsError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(SafetensorsError::FileTooSmall);
        }

        // Read header size (first 8 bytes, little-endian u64)
        let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_size;

        if mmap.len() < data_start {
            return Err(SafetensorsError::FileTooSmall);
        }

        // Parse JSON header
        let header_bytes = &mmap[8..data_start];
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(header_bytes)?;

        // Build tensor index, skipping __metadata__
        let mut tensors = HashMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value)?;
            tensors.insert(name, info);
        }

        // Advise sequential access for optimal kernel readahead
        let _ = mmap.advise(memmap2::Advice::Sequential);

        log::info!(
            "Opened {} — {} tensors, {:.1} MB data",
            path.display(),
            tensors.len(),
            (mmap.len() - data_start) as f64 / 1024.0 / 1024.0,
        );

        Ok(MmapSafetensors {
            mmap,
            data_start,
            tensors,
        })
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get metadata for a tensor.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get a raw byte slice for a named tensor (zero-copy from mmap).
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let start = self.data_start + info.data_offsets[0];
        let end = self.data_start + info.data_offsets[1];

        if end > self.mmap.len() {
            return Err(SafetensorsError::OffsetOutOfBounds(name.to_string()));
        }

        Ok(&self.mmap[start..end])
    }

    /// Get tensor data as a typed slice (zero-copy).
    /// Panics if the tensor dtype doesn't match T's size.
    pub fn tensor_as_slice<T: Copy>(&self, name: &str) -> Result<&[T], SafetensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        let data = self.tensor_data(name)?;

        assert_eq!(
            std::mem::size_of::<T>(),
            info.dtype.element_size(),
            "Type size mismatch: requested {} bytes, tensor dtype is {:?} ({} bytes)",
            std::mem::size_of::<T>(),
            info.dtype,
            info.dtype.element_size(),
        );

        let ptr = data.as_ptr() as *const T;
        let len = info.numel();
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Advise the kernel to prefetch tensor data (MADV_WILLNEED).
    /// Returns silently if the tensor doesn't exist.
    pub fn prefetch_tensor(&self, name: &str) {
        if let Some(info) = self.tensors.get(name) {
            let start = self.data_start + info.data_offsets[0];
            let len = info.data_offsets[1] - info.data_offsets[0];
            let _ = self.mmap.advise_range(memmap2::Advice::WillNeed, start, len);
        }
    }

    /// Evict all pages from page cache. Call after all data has been copied out.
    pub fn evict_page_cache(&self) {
        let _ = unsafe { self.mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_v2_lite() {
        let path = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite/model-00001-of-000004.safetensors");
        if !path.exists() {
            eprintln!("Skipping test — V2-Lite not downloaded");
            return;
        }

        let st = MmapSafetensors::open(path).expect("Failed to open safetensors");

        // Check we can find expert weights
        let gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight";
        let info = st.tensor_info(gate_name).expect("Expert weight not found");

        // V2-Lite: gate_proj is [moe_intermediate_size, hidden_size] = [1408, 2048]
        assert_eq!(info.shape, vec![1408, 2048], "Unexpected shape: {:?}", info.shape);
        assert_eq!(info.dtype, Dtype::Bf16);
        assert_eq!(info.byte_size(), 1408 * 2048 * 2); // BF16 = 2 bytes

        // Read actual data
        let data = st.tensor_data(gate_name).expect("Failed to read tensor data");
        assert_eq!(data.len(), 1408 * 2048 * 2);

        // Read as u16 slice (BF16 raw values)
        let bf16_data: &[u16] = st.tensor_as_slice(gate_name).expect("Failed to cast");
        assert_eq!(bf16_data.len(), 1408 * 2048);

        // Verify it's not all zeros (sanity check)
        let nonzero = bf16_data.iter().filter(|&&v| v != 0).count();
        assert!(nonzero > bf16_data.len() / 2, "Too many zeros — data looks wrong");

        // Print stats
        let as_f32: Vec<f32> = bf16_data.iter().map(|&v| bf16_to_f32(v)).collect();
        let min = as_f32.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = as_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = as_f32.iter().sum::<f32>() / as_f32.len() as f32;
        eprintln!(
            "gate_proj.weight [{}, {}] BF16: min={:.4}, max={:.4}, mean={:.6}, nonzero={}/{}",
            info.shape[0], info.shape[1], min, max, mean, nonzero, bf16_data.len()
        );
    }

    /// Convert a raw BF16 u16 to f32.
    fn bf16_to_f32(v: u16) -> f32 {
        f32::from_bits((v as u32) << 16)
    }
}

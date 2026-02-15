//! GGUF file parser and block dequantizer.
//!
//! Reads GGUF v3 files: header, metadata (skipped), tensor info table.
//! Supports dequantization of Q4_K, Q5_K, Q6_K, F16, BF16, F32 blocks.
//! Used to load pre-quantized expert weights for CPU decode.

use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

/// GGML tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2_K),
            11 => Ok(GgmlType::Q3_K),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            15 => Ok(GgmlType::Q8_K),
            30 => Ok(GgmlType::BF16),
            _ => Err(format!("Unknown GGML type: {v}")),
        }
    }

    /// Block size (number of elements per quantization block).
    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 => 32,
            GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
            GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
            GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K |
            GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_K => 256,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 | GgmlType::BF16 => 2,
            GgmlType::Q4_0 => 2 + 16,       // 18: fp16 scale + 16 bytes (32 nibbles)
            GgmlType::Q4_1 => 2 + 2 + 16,   // 20: fp16 d + fp16 m + 16 bytes
            GgmlType::Q5_0 => 2 + 4 + 16,   // 22: fp16 d + 4 bytes qh + 16 bytes qs
            GgmlType::Q5_1 => 2 + 2 + 4 + 16, // 24
            GgmlType::Q8_0 => 2 + 32,       // 34: fp16 d + 32 bytes
            GgmlType::Q8_1 => 4 + 4 + 32,   // 40: fp32 d + fp32 s + 32 bytes
            GgmlType::Q2_K => 2 + 2 + 16 + 64, // 84
            GgmlType::Q3_K => 2 + 32 + 12 + 64, // 110
            GgmlType::Q4_K => 2 + 2 + 12 + 128, // 144
            GgmlType::Q5_K => 2 + 2 + 12 + 32 + 128, // 176
            GgmlType::Q6_K => 128 + 64 + 16 + 2, // 210
            GgmlType::Q8_K => 4 + 256 + 16,  // 276: fp32 d + 256 bytes qs + 16 fp16 bsums
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q8_1 => "Q8_1",
            GgmlType::Q2_K => "Q2_K",
            GgmlType::Q3_K => "Q3_K",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::Q8_K => "Q8_K",
        }
    }
}

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[allow(dead_code, non_camel_case_types)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// Info about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset from start of data section.
    pub offset: u64,
    /// Total number of elements.
    pub n_elements: u64,
}

impl GgufTensorInfo {
    /// Total bytes of this tensor's data.
    pub fn data_bytes(&self) -> usize {
        let bs = self.dtype.block_size() as u64;
        let n_blocks = (self.n_elements + bs - 1) / bs;
        n_blocks as usize * self.dtype.block_bytes()
    }
}

/// Parsed GGUF file (mmap-backed).
pub struct GgufFile {
    mmap: Mmap,
    /// Start of tensor data section within the file.
    pub data_offset: usize,
    /// All tensors in the file, indexed by name.
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Ordered tensor names (insertion order from file).
    pub tensor_names: Vec<String>,
    /// Selected metadata values we care about.
    pub metadata: HashMap<String, GgufMetaValue>,
}

/// Metadata values we extract.
#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    Uint32(u32),
    Uint64(u64),
    Int32(i32),
    Float32(f32),
    String(String),
    Bool(bool),
}

/// Helper to read little-endian values from a byte slice.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.pos >= self.data.len() {
            return Err("Unexpected EOF reading u8".into());
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        if self.pos + 2 > self.data.len() {
            return Err("Unexpected EOF reading u16".into());
        }
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("Unexpected EOF reading u32".into());
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("Unexpected EOF reading i32".into());
        }
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        if self.pos + 8 > self.data.len() {
            return Err("Unexpected EOF reading u64".into());
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        if self.pos + 8 > self.data.len() {
            return Err("Unexpected EOF reading i64".into());
        }
        let v = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("Unexpected EOF reading f32".into());
        }
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        if self.pos + 8 > self.data.len() {
            return Err("Unexpected EOF reading f64".into());
        }
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() {
            return Err(format!("Unexpected EOF reading string of length {len}"));
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| format!("Invalid UTF-8 in string: {e}"))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!("Unexpected EOF reading {n} bytes"));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Skip a metadata value based on its type.
    fn skip_value(&mut self, vtype: u32) -> Result<(), String> {
        match vtype {
            0 => { self.pos += 1; } // u8
            1 => { self.pos += 1; } // i8
            2 => { self.pos += 2; } // u16
            3 => { self.pos += 2; } // i16
            4 => { self.pos += 4; } // u32
            5 => { self.pos += 4; } // i32
            6 => { self.pos += 4; } // f32
            7 => { self.pos += 1; } // bool
            8 => { let _ = self.read_string()?; } // string
            9 => {
                // Array: element_type (u32) + count (u64) + elements
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            10 => { self.pos += 8; } // u64
            11 => { self.pos += 8; } // i64
            12 => { self.pos += 8; } // f64
            _ => return Err(format!("Unknown metadata value type: {vtype}")),
        }
        if self.pos > self.data.len() {
            return Err("Unexpected EOF while skipping metadata value".into());
        }
        Ok(())
    }
}

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in LE
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open GGUF file: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap GGUF file: {e}"))?;

        if mmap.len() < 24 {
            return Err("GGUF file too small for header".into());
        }

        let mut r = Reader::new(&mmap);

        // Header
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "Bad GGUF magic: 0x{magic:08x} (expected 0x{GGUF_MAGIC:08x})"
            ));
        }

        let version = r.read_u32()?;
        if version < 2 || version > 3 {
            return Err(format!("Unsupported GGUF version: {version} (supported: 2-3)"));
        }

        let tensor_count = r.read_u64()? as usize;
        let metadata_kv_count = r.read_u64()? as usize;

        log::info!(
            "GGUF v{version}: {tensor_count} tensors, {metadata_kv_count} metadata KV pairs, {:.1} GB",
            mmap.len() as f64 / 1e9,
        );

        // Read metadata — extract keys we care about, skip the rest
        let mut metadata = HashMap::new();
        let interesting_keys = [
            "general.architecture",
            "general.name",
            "general.file_type",
            "general.quantization_version",
        ];

        for i in 0..metadata_kv_count {
            let key = r.read_string()?;
            let vtype = r.read_u32()?;

            if interesting_keys.contains(&key.as_str()) {
                // Read and store the value
                let value = match vtype {
                    4 => GgufMetaValue::Uint32(r.read_u32()?),
                    5 => GgufMetaValue::Int32(r.read_i32()?),
                    6 => GgufMetaValue::Float32(r.read_f32()?),
                    7 => GgufMetaValue::Bool(r.read_u8()? != 0),
                    8 => GgufMetaValue::String(r.read_string()?),
                    10 => GgufMetaValue::Uint64(r.read_u64()?),
                    _ => {
                        r.skip_value(vtype)?;
                        continue;
                    }
                };
                metadata.insert(key, value);
            } else {
                r.skip_value(vtype)?;
            }

            if (i + 1) % 100 == 0 {
                log::debug!("Parsed {}/{} metadata entries", i + 1, metadata_kv_count);
            }
        }

        // Read tensor info table
        let mut tensors = HashMap::with_capacity(tensor_count);
        let mut tensor_names = Vec::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32()? as usize;

            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.read_u64()?);
            }

            let dtype_raw = r.read_u32()?;
            let dtype = GgmlType::from_u32(dtype_raw)?;
            let offset = r.read_u64()?;

            let n_elements: u64 = dims.iter().product();

            tensor_names.push(name.clone());
            tensors.insert(name, GgufTensorInfo {
                name: tensor_names.last().unwrap().clone(),
                dims,
                dtype,
                offset,
                n_elements,
            });
        }

        // Compute data section offset (aligned)
        let alignment = GGUF_DEFAULT_ALIGNMENT;
        let header_end = r.pos;
        let data_offset = (header_end + alignment - 1) / alignment * alignment;

        log::info!(
            "GGUF parsed: {} tensors, data starts at offset {data_offset} ({:.1} MB header+metadata)",
            tensors.len(),
            data_offset as f64 / 1e6,
        );

        // Log some tensor info for debugging
        let mut type_counts: HashMap<&str, usize> = HashMap::new();
        for info in tensors.values() {
            *type_counts.entry(info.dtype.name()).or_insert(0) += 1;
        }
        let mut type_summary: Vec<_> = type_counts.iter().collect();
        type_summary.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        log::info!(
            "GGUF tensor types: {}",
            type_summary.iter().map(|(t, c)| format!("{t}={c}")).collect::<Vec<_>>().join(", "),
        );

        Ok(GgufFile {
            mmap,
            data_offset,
            tensors,
            tensor_names,
            metadata,
        })
    }

    /// Get raw bytes for a tensor.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> Result<&[u8], String> {
        let start = self.data_offset + info.offset as usize;
        let len = info.data_bytes();
        let end = start + len;
        if end > self.mmap.len() {
            return Err(format!(
                "Tensor '{}' data [{start}..{end}) exceeds file size {}",
                info.name, self.mmap.len(),
            ));
        }
        Ok(&self.mmap[start..end])
    }

    /// Dequantize a tensor to FP32.
    pub fn dequantize_tensor(&self, info: &GgufTensorInfo) -> Result<Vec<f32>, String> {
        let data = self.tensor_data(info)?;
        let n_elements = info.n_elements as usize;

        match info.dtype {
            GgmlType::F32 => dequant_f32(data, n_elements),
            GgmlType::F16 => dequant_f16(data, n_elements),
            GgmlType::BF16 => dequant_bf16(data, n_elements),
            GgmlType::Q4_0 => dequant_q4_0(data, n_elements),
            GgmlType::Q5_0 => dequant_q5_0(data, n_elements),
            GgmlType::Q4_K => dequant_q4_k(data, n_elements),
            GgmlType::Q5_K => dequant_q5_k(data, n_elements),
            GgmlType::Q6_K => dequant_q6_k(data, n_elements),
            GgmlType::Q8_0 => dequant_q8_0(data, n_elements),
            other => Err(format!("Dequantization not implemented for {}", other.name())),
        }
    }

    /// Find expert tensor names for a given layer.
    ///
    /// Returns (gate_name, up_name, down_name) or None if not found.
    /// Supports both merged (`ffn_gate_exps`) and per-expert (`ffn_gate.{E}`) naming.
    pub fn find_expert_tensors(&self, layer: usize, expert: usize) -> Option<(String, String, String)> {
        // Try per-expert naming first: blk.{L}.ffn_gate.{E}.weight
        let gate_per = format!("blk.{layer}.ffn_gate.{expert}.weight");
        let up_per = format!("blk.{layer}.ffn_up.{expert}.weight");
        let down_per = format!("blk.{layer}.ffn_down.{expert}.weight");

        if self.tensors.contains_key(&gate_per) {
            return Some((gate_per, up_per, down_per));
        }

        // Try merged naming: blk.{L}.ffn_gate_exps.weight (all experts in one tensor)
        // Caller will need to slice the expert dimension
        let gate_merged = format!("blk.{layer}.ffn_gate_exps.weight");
        let up_merged = format!("blk.{layer}.ffn_up_exps.weight");
        let down_merged = format!("blk.{layer}.ffn_down_exps.weight");

        if self.tensors.contains_key(&gate_merged) {
            return Some((gate_merged, up_merged, down_merged));
        }

        None
    }

    /// Check if this GGUF uses merged expert tensors (ffn_gate_exps).
    pub fn has_merged_experts(&self) -> bool {
        self.tensor_names.iter().any(|n| n.contains("ffn_gate_exps"))
    }

    /// Find shared expert tensor names for a given layer.
    pub fn find_shared_expert_tensors(&self, layer: usize) -> Option<(String, String, String)> {
        let gate = format!("blk.{layer}.ffn_gate_shexp.weight");
        let up = format!("blk.{layer}.ffn_up_shexp.weight");
        let down = format!("blk.{layer}.ffn_down_shexp.weight");

        if self.tensors.contains_key(&gate) {
            return Some((gate, up, down));
        }
        None
    }

    /// Evict all pages from page cache. Call after all data has been copied out.
    pub fn evict_page_cache(&self) {
        let _ = unsafe { self.mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
    }
}

// ── FP16/BF16/F32 dequantization ──────────────────────────────────────

fn dequant_f32(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if data.len() < n * 4 {
        return Err(format!("F32 data too short: {} < {}", data.len(), n * 4));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = f32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
    }
    Ok(out)
}

fn dequant_f16(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if data.len() < n * 2 {
        return Err(format!("F16 data too short: {} < {}", data.len(), n * 2));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let bits = u16::from_le_bytes(data[i * 2..(i + 1) * 2].try_into().unwrap());
        out[i] = half::f16::from_bits(bits).to_f32();
    }
    Ok(out)
}

fn dequant_bf16(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if data.len() < n * 2 {
        return Err(format!("BF16 data too short: {} < {}", data.len(), n * 2));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let bits = u16::from_le_bytes(data[i * 2..(i + 1) * 2].try_into().unwrap());
        out[i] = f32::from_bits((bits as u32) << 16);
    }
    Ok(out)
}

// ── Q8_0 dequantization ───────────────────────────────────────────────

/// Q8_0 block: fp16 scale + 32 x int8 weights. Block size = 32.
fn dequant_q8_0(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // 34 bytes per block
    let nb = n / QK;
    if data.len() < nb * BLOCK_BYTES {
        return Err(format!("Q8_0 data too short: {} < {}", data.len(), nb * BLOCK_BYTES));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = half::f16::from_bits(
            u16::from_le_bytes(block[0..2].try_into().unwrap())
        ).to_f32();
        let qs = &block[2..2 + QK];
        for j in 0..QK {
            out[i * QK + j] = d * (qs[j] as i8) as f32;
        }
    }
    Ok(out)
}

// ── Q5_0 dequantization ───────────────────────────────────────────────

/// Q5_0: 32 elements per block, 22 bytes = fp16 d + 4 bytes qh + 16 bytes qs (nibbles).
/// Each element = d * ((qs_nibble | (qh_bit << 4)) - 16).
fn dequant_q5_0(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 22; // fp16 d (2) + qh (4) + qs (16)
    let nb = n / QK;
    if data.len() < nb * BLOCK_BYTES {
        return Err(format!("Q5_0 data too short: {} < {}", data.len(), nb * BLOCK_BYTES));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = half::f16::from_bits(
            u16::from_le_bytes(block[0..2].try_into().unwrap())
        ).to_f32();
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..6 + 16];

        for j in 0..QK {
            // Lower 4 bits from nibble
            let q4 = if j < 16 {
                qs[j] & 0x0F
            } else {
                (qs[j - 16] >> 4) & 0x0F
            };
            // 5th bit from qh
            let q5bit = ((qh >> j) & 1) as u8;
            let q = (q4 | (q5bit << 4)) as i32 - 16;
            out[i * QK + j] = d * q as f32;
        }
    }
    Ok(out)
}

// ── Q4_0 dequantization ───────────────────────────────────────────────

/// Q4_0: 32 elements per block, 18 bytes = fp16 d + 16 bytes qs (nibbles).
/// Each element = d * (nibble - 8).
fn dequant_q4_0(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18; // fp16 d (2) + qs (16)
    let nb = n / QK;
    if data.len() < nb * BLOCK_BYTES {
        return Err(format!("Q4_0 data too short: {} < {}", data.len(), nb * BLOCK_BYTES));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = half::f16::from_bits(
            u16::from_le_bytes(block[0..2].try_into().unwrap())
        ).to_f32();
        let qs = &block[2..2 + 16];

        for j in 0..QK {
            let nibble = if j < 16 {
                qs[j] & 0x0F
            } else {
                (qs[j - 16] >> 4) & 0x0F
            };
            out[i * QK + j] = d * (nibble as i32 - 8) as f32;
        }
    }
    Ok(out)
}

// ── Q4_K dequantization ───────────────────────────────────────────────

/// Extract 6-bit scale and min from packed scales array (Q4_K/Q5_K format).
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q4_K blocks to FP32.
///
/// Q4_K block (144 bytes per 256 elements):
///   fp16 d (super-block scale), fp16 dmin (super-block min),
///   12 bytes packed 6-bit scales/mins, 128 bytes 4-bit quants.
fn dequant_q4_k(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let nb = n / QK;
    if data.len() < nb * BLOCK_BYTES {
        return Err(format!("Q4_K data too short: {} < {}", data.len(), nb * BLOCK_BYTES));
    }

    let mut out = vec![0.0f32; n];

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];

        let d = half::f16::from_bits(
            u16::from_le_bytes(block[0..2].try_into().unwrap())
        ).to_f32();
        let dmin = half::f16::from_bits(
            u16::from_le_bytes(block[2..4].try_into().unwrap())
        ).to_f32();

        let scales = &block[4..16]; // 12 bytes
        let qs = &block[16..144];   // 128 bytes

        let base = i * QK;
        let mut is = 0usize;
        let mut q_offset = 0usize;

        for j in (0..QK).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            // First 32: low nibbles
            for l in 0..32 {
                out[base + j + l] = d1 * (qs[q_offset + l] & 0xF) as f32 - m1;
            }
            // Next 32: high nibbles
            for l in 0..32 {
                out[base + j + 32 + l] = d2 * (qs[q_offset + l] >> 4) as f32 - m2;
            }

            q_offset += 32;
            is += 2;
        }
    }

    Ok(out)
}

// ── Q5_K dequantization ───────────────────────────────────────────────

/// Dequantize Q5_K blocks to FP32.
///
/// Q5_K block (176 bytes per 256 elements):
///   fp16 d, fp16 dmin, 12 bytes scales, 32 bytes qh (5th bits), 128 bytes qs (4-bit).
fn dequant_q5_k(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let nb = n / QK;
    if data.len() < nb * BLOCK_BYTES {
        return Err(format!("Q5_K data too short: {} < {}", data.len(), nb * BLOCK_BYTES));
    }

    let mut out = vec![0.0f32; n];

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];

        let d = half::f16::from_bits(
            u16::from_le_bytes(block[0..2].try_into().unwrap())
        ).to_f32();
        let dmin = half::f16::from_bits(
            u16::from_le_bytes(block[2..4].try_into().unwrap())
        ).to_f32();

        let scales = &block[4..16];   // 12 bytes
        let qh = &block[16..48];      // 32 bytes (256 bits)
        let qs = &block[48..176];     // 128 bytes

        let base = i * QK;
        let mut is = 0usize;
        let mut q_offset = 0usize;
        // Mask bits: u1 tests even bits of qh, u2 tests odd bits
        // Each 64-element group shifts the mask left by 2
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _j in (0..QK).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            let out_base = base + is * 32; // is increments by 2 per 64 elements

            // First 32: low nibbles + 5th bit from u1 mask
            for l in 0..32 {
                let q4 = (qs[q_offset + l] & 0xF) as u32;
                let q5 = if qh[l] & u1 != 0 { 16u32 } else { 0 };
                out[out_base + l] = d1 * (q4 + q5) as f32 - m1;
            }
            // Next 32: high nibbles + 5th bit from u2 mask
            for l in 0..32 {
                let q4 = (qs[q_offset + l] >> 4) as u32;
                let q5 = if qh[l] & u2 != 0 { 16u32 } else { 0 };
                out[out_base + 32 + l] = d2 * (q4 + q5) as f32 - m2;
            }

            q_offset += 32;
            u1 <<= 2;
            u2 <<= 2;
            is += 2;
        }
    }

    Ok(out)
}

// ── Q6_K dequantization ───────────────────────────────────────────────

/// Dequantize Q6_K blocks to FP32.
///
/// Q6_K block (210 bytes per 256 elements):
///   128 bytes ql (lower 4 bits), 64 bytes qh (upper 2 bits),
///   16 bytes scales (int8), fp16 d (super-block scale).
fn dequant_q6_k(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let nb = n / QK;
    if data.len() < nb * BLOCK_BYTES {
        return Err(format!("Q6_K data too short: {} < {}", data.len(), nb * BLOCK_BYTES));
    }

    let mut out = vec![0.0f32; n];

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];

        let ql_all = &block[0..128];     // lower 4 bits
        let qh_all = &block[128..192];   // upper 2 bits
        let sc_all = &block[192..208];   // int8 scales
        let d = half::f16::from_bits(
            u16::from_le_bytes(block[208..210].try_into().unwrap())
        ).to_f32();

        let base = i * QK;
        let mut out_offset = 0usize;
        let mut ql_offset = 0usize;
        let mut qh_offset = 0usize;
        let mut sc_offset = 0usize;

        // Process 128 elements at a time (two groups per block of 256)
        for _half in 0..2 {
            let ql = &ql_all[ql_offset..];
            let qh = &qh_all[qh_offset..];
            let sc = &sc_all[sc_offset..];

            for l in 0..32 {
                // 4 interleaved sub-groups of 32 elements
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

                out[base + out_offset + l] = d * sc[0] as i8 as f32 * q1 as f32;
                out[base + out_offset + l + 32] = d * sc[2] as i8 as f32 * q2 as f32;
                out[base + out_offset + l + 64] = d * sc[4] as i8 as f32 * q3 as f32;
                out[base + out_offset + l + 96] = d * sc[6] as i8 as f32 * q4 as f32;
            }

            out_offset += 128;
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
    }

    Ok(out)
}

/// Dequantize raw GGUF-format bytes to FP32.
///
/// Accepts raw byte data + GGML type + element count. Used by the GGUF→AVX2
/// cache builder to dequantize individual expert tensors from already-loaded data.
pub fn dequantize_raw_data(dtype: GgmlType, data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    match dtype {
        GgmlType::F32 => dequant_f32(data, n_elements),
        GgmlType::F16 => dequant_f16(data, n_elements),
        GgmlType::BF16 => dequant_bf16(data, n_elements),
        GgmlType::Q4_0 => dequant_q4_0(data, n_elements),
        GgmlType::Q5_0 => dequant_q5_0(data, n_elements),
        GgmlType::Q4_K => dequant_q4_k(data, n_elements),
        GgmlType::Q5_K => dequant_q5_k(data, n_elements),
        GgmlType::Q6_K => dequant_q6_k(data, n_elements),
        GgmlType::Q8_0 => dequant_q8_0(data, n_elements),
        other => Err(format!("Dequantization not implemented for {}", other.name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
        assert_eq!(GgmlType::Q4_K.block_bytes(), 144);
        assert_eq!(GgmlType::Q5_K.block_bytes(), 176);
        assert_eq!(GgmlType::Q6_K.block_bytes(), 210);
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F32.block_bytes(), 4);
    }

    #[test]
    fn test_get_scale_min_k4() {
        // Test with known values
        let scales = [0x3F, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80, 0, 0, 0, 0];
        let (d, m) = get_scale_min_k4(0, &scales);
        assert_eq!(d, 0x3F); // scales[0] & 63
        assert_eq!(m, 0x04); // scales[4] & 63
    }

    #[test]
    fn test_dequant_f32_roundtrip() {
        let values = vec![1.0f32, -2.5, 3.14, 0.0];
        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let result = dequant_f32(&bytes, values.len()).unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_dequant_bf16_roundtrip() {
        let values = vec![1.0f32, -2.0, 0.0, 0.5];
        let mut bytes = Vec::new();
        for v in &values {
            let bits = v.to_bits();
            let bf16 = (bits >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        let result = dequant_bf16(&bytes, values.len()).unwrap();
        // BF16 has limited precision but these values are exact
        assert_eq!(result, values);
    }
}

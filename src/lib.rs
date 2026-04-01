pub mod chat_template;
pub mod decode;
pub mod draft_model;
pub mod gguf;
pub mod gguf_kernels;
pub mod gpu_decode;
pub mod gpu_prefill;
pub mod kernel;
pub mod moe;
pub mod numa;
pub mod server;
pub mod syscheck;
pub mod vram_monitor;
pub mod weights;

use pyo3::prelude::*;

/// Krasis — hybrid LLM MoE runtime
#[pymodule]
fn krasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<moe::KrasisEngine>()?;
    m.add_class::<weights::WeightStore>()?;
    m.add_class::<decode::CpuDecodeStore>()?;
    m.add_class::<server::RustServer>()?;
    m.add_class::<gpu_decode::GpuDecodeStore>()?;
    m.add_class::<vram_monitor::VramMonitor>()?;
    m.add_function(wrap_pyfunction!(syscheck::system_check, m)?)?;
    Ok(())
}

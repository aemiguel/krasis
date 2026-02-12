pub mod gguf;
pub mod gguf_kernels;
pub mod kernel;
pub mod moe;
pub mod numa;
pub mod syscheck;
pub mod weights;

use pyo3::prelude::*;

/// Krasis — hybrid LLM MoE runtime
#[pymodule]
fn krasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<moe::KrasisEngine>()?;
    m.add_class::<weights::WeightStore>()?;
    m.add_function(wrap_pyfunction!(syscheck::system_check, m)?)?;
    Ok(())
}

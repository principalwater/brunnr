// SPDX-License-Identifier: Apache-2.0
//! Native Python bindings for Artesian (scaffold).
//!
//! This is the thin PyO3 layer that exposes the Rust core to Python. It is intentionally minimal
//! today: as the core stabilizes (transactional substrate + ACC), the memory and control-plane
//! surfaces are bound here so Python users run the same audited Rust core in-process.

use pyo3::prelude::*;

/// Return the Artesian bindings version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A handle to the in-process memory control plane.
///
/// TODO: back this with `headgate` (ACC control plane) + `aquifer` (retrieval) once the core API is
/// stable. The intent is `recall` / `commit` mirroring the MCP tools, running in-process.
#[pyclass]
struct Memory {}

#[pymethods]
impl Memory {
    #[new]
    fn new() -> Self {
        Memory {}
    }

    /// Placeholder. Will recall a high-signal slice for the current query (small-to-big).
    fn recall(&self, _query: &str) -> PyResult<Vec<String>> {
        Ok(Vec::new())
    }

    /// Placeholder. Will run the qualify-gate and commit a durable learning into the CCS.
    fn commit(&self, _text: &str) -> PyResult<bool> {
        Ok(false)
    }
}

#[pymodule]
fn _artesian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Memory>()?;
    Ok(())
}

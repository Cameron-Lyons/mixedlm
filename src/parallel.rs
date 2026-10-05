//! Process-wide choice between rayon's global pool and sequential kernels.
//!
//! A child created by `fork` inherits rayon's global registry but none of its
//! worker threads, so the first job it hands to the pool would wait forever.
//! The module registers `after_fork_in_child` with `os.register_at_fork`, and
//! a forked child then runs every kernel sequentially. faer reads its global
//! parallelism for implicit products and decompositions as well as for our
//! explicit `get_global_parallelism` choices; rayon loops ask `rayon_threads`.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::types::PyDict;

static SEQUENTIAL: AtomicBool = AtomicBool::new(false);

/// Workers a rayon loop may use: one in a forked child.
pub fn rayon_threads() -> usize {
    if SEQUENTIAL.load(Ordering::Relaxed) {
        1
    } else {
        rayon::current_num_threads()
    }
}

fn run_sequentially() {
    SEQUENTIAL.store(true, Ordering::Relaxed);
    faer::set_global_parallelism(faer::Par::Seq);
}

#[pyfunction]
fn after_fork_in_child() {
    run_sequentially();
}

/// Make children forked from this interpreter run sequentially, where Python can fork.
pub fn register_at_fork(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let os = py.import("os")?;
    if !os.hasattr("register_at_fork")? {
        return Ok(());
    }
    let hooks = PyDict::new(py);
    hooks.set_item(
        "after_in_child",
        wrap_pyfunction!(after_fork_in_child, module)?,
    )?;
    os.call_method("register_at_fork", (), Some(&hooks))?;
    Ok(())
}

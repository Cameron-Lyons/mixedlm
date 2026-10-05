use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::sparse_chol::FillOrdering;

mod blocked_chol;
mod covariance;
mod csc;
mod glmm;
mod glmm_sparse;
mod linalg;
mod lmm;
mod nlmm;
mod quadrature;
mod reml_algorithms;
mod simulation;
mod sparse_chol;

fn owned_array2<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    shape: (usize, usize),
) -> PyResult<Py<PyArray2<f64>>> {
    let array = Array2::from_shape_vec(shape, values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array).into())
}

fn checked_i64_vec_to_usize(values: &[i64], field_name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            usize::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "{field_name}[{idx}] must be non-negative, got {value}"
                ))
            })
        })
        .collect()
}

/// Every sparse Cholesky entry point accepts the same ordering names.
fn fill_ordering(name: &str) -> PyResult<FillOrdering> {
    match name {
        "amd" => Ok(FillOrdering::Amd),
        "natural" => Ok(FillOrdering::Natural),
        _ => Err(PyValueError::new_err("ordering must be 'amd' or 'natural'")),
    }
}

#[pyclass(frozen)]
pub struct SparseCholeskySymbolic {
    inner: sparse_chol::SymbolicCholeskyCache,
}

#[pymethods]
impl SparseCholeskySymbolic {
    #[new]
    #[pyo3(signature = (indices, indptr, n, *, ordering = "amd"))]
    fn new(
        py: Python<'_>,
        indices: PyArrayLike1<'_, i64>,
        indptr: PyArrayLike1<'_, i64>,
        n: usize,
        ordering: &str,
    ) -> PyResult<Self> {
        let ordering = fill_ordering(ordering)?;
        let indices = checked_i64_vec_to_usize(indices.as_slice()?, "indices")?;
        let indptr = checked_i64_vec_to_usize(indptr.as_slice()?, "indptr")?;
        let inner =
            py.detach(|| sparse_chol::SymbolicCholeskyCache::new(&indices, &indptr, n, ordering))?;
        Ok(Self { inner })
    }

    fn factor(
        &self,
        py: Python<'_>,
        data: PyArrayLike1<'_, f64>,
    ) -> PyResult<SparseCholeskyNumeric> {
        // Detached numerical work must own every Python-backed input buffer.
        let data = data.as_slice()?.to_vec();
        let numeric = py.detach(|| self.inner.factor(&data))?;
        Ok(SparseCholeskyNumeric { inner: numeric })
    }

    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Number of entries in the factor, including its diagonal and fill.
    fn factor_nonzeros(&self) -> usize {
        self.inner.factor_nonzeros()
    }
}

#[pyclass(frozen)]
pub struct SparseCholeskyNumeric {
    inner: sparse_chol::NumericFactorization,
}

#[pymethods]
impl SparseCholeskyNumeric {
    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyArrayLike2<'py, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let shape = (b.as_array().nrows(), b.as_array().ncols());
        // Logical iteration handles strided views and snapshots both values and
        // shape before another Python thread can alter the original array.
        let rhs = b.as_array().iter().copied().collect();
        let result = py.detach(|| self.inner.solve_owned(rhs, shape))?;
        owned_array2(py, result, shape)
    }

    fn logdet(&self, py: Python<'_>) -> f64 {
        py.detach(|| self.inner.logdet())
    }
}

#[pyfunction]
#[pyo3(signature = (a_data, a_indices, a_indptr, a_shape, b, *, ordering = "amd"))]
fn sparse_cholesky_solve<'py>(
    py: Python<'py>,
    a_data: PyArrayLike1<'py, f64>,
    a_indices: PyArrayLike1<'py, i64>,
    a_indptr: PyArrayLike1<'py, i64>,
    a_shape: (usize, usize),
    b: PyArrayLike2<'py, f64>,
    ordering: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let ordering = fill_ordering(ordering)?;
    let matrix = linalg::square_csc_from_scipy(
        a_data.as_slice()?,
        a_indices.as_slice()?,
        a_indptr.as_slice()?,
        a_shape,
    )?;
    let shape = (b.as_array().nrows(), b.as_array().ncols());
    let rhs = b.as_array().iter().copied().collect();
    let result = py.detach(|| linalg::sparse_cholesky_solve(&matrix, rhs, shape, ordering))?;
    owned_array2(py, result, shape)
}

#[pyfunction]
#[pyo3(signature = (a_data, a_indices, a_indptr, a_shape, *, ordering = "amd"))]
fn sparse_cholesky_logdet<'py>(
    py: Python<'py>,
    a_data: PyArrayLike1<'py, f64>,
    a_indices: PyArrayLike1<'py, i64>,
    a_indptr: PyArrayLike1<'py, i64>,
    a_shape: (usize, usize),
    ordering: &str,
) -> PyResult<f64> {
    let ordering = fill_ordering(ordering)?;
    let matrix = linalg::square_csc_from_scipy(
        a_data.as_slice()?,
        a_indices.as_slice()?,
        a_indptr.as_slice()?,
        a_shape,
    )?;
    py.detach(|| linalg::sparse_cholesky_logdet(&matrix, ordering))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn update_cholesky_factor<'py>(
    py: Python<'py>,
    l_data: PyArrayLike1<'py, f64>,
    l_indices: PyArrayLike1<'py, i64>,
    l_indptr: PyArrayLike1<'py, i64>,
    l_shape: (usize, usize),
    theta: PyArrayLike1<'py, f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let (data, indices, indptr) = linalg::update_cholesky_factor(
        l_data.as_slice()?,
        l_indices.as_slice()?,
        l_indptr.as_slice()?,
        l_shape,
        theta.as_slice()?,
    )?;
    Ok((
        PyArray1::from_vec(py, data).into(),
        PyArray1::from_vec(py, indices).into(),
        PyArray1::from_vec(py, indptr).into(),
    ))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("_source_fingerprint", env!("MIXEDLM_SOURCE_FINGERPRINT"))?;
    m.add_class::<lmm::LmmDesign>()?;
    m.add_class::<lmm::LmmResponse>()?;
    m.add_class::<glmm::GlmmProblem>()?;
    m.add_class::<SparseCholeskySymbolic>()?;
    m.add_class::<SparseCholeskyNumeric>()?;
    m.add_function(wrap_pyfunction!(sparse_cholesky_solve, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_cholesky_logdet, m)?)?;
    m.add_function(wrap_pyfunction!(update_cholesky_factor, m)?)?;
    m.add_function(wrap_pyfunction!(quadrature::gauss_hermite, m)?)?;
    m.add_function(wrap_pyfunction!(quadrature::adaptive_gauss_hermite_1d, m)?)?;
    m.add_function(wrap_pyfunction!(lmm::profiled_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(lmm::profiled_deviance_cached, m)?)?;
    m.add_function(wrap_pyfunction!(lmm::compute_ztwz, m)?)?;
    m.add_function(wrap_pyfunction!(lmm::profiled_deviance_with_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(glmm::pirls, m)?)?;
    m.add_function(wrap_pyfunction!(glmm::laplace_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(glmm::glmm_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(glmm::adaptive_gh_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(nlmm::pnls_step, m)?)?;
    m.add_function(wrap_pyfunction!(nlmm::nlmm_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(nlmm::nlmm_deviance_with_status, m)?)?;
    m.add_function(wrap_pyfunction!(simulation::simulate_re_batch, m)?)?;
    m.add_function(wrap_pyfunction!(simulation::compute_zu, m)?)?;
    m.add_function(wrap_pyfunction!(reml_algorithms::mm_reml, m)?)?;
    m.add_function(wrap_pyfunction!(reml_algorithms::augmented_ai_reml, m)?)?;
    m.add_function(wrap_pyfunction!(reml_algorithms::riemannian_reml, m)?)?;
    Ok(())
}

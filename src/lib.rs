use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::prelude::*;

mod blocked_chol;
mod glmm;
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
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array).into())
}

fn checked_i64_vec_to_usize(values: &[i64], field_name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            usize::try_from(value).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{field_name}[{idx}] must be non-negative, got {value}"
                ))
            })
        })
        .collect()
}

#[pyclass]
pub struct SparseCholeskySymbolic {
    inner: sparse_chol::SymbolicCholeskyCache,
    indices: Vec<usize>,
    indptr: Vec<usize>,
}

#[pymethods]
impl SparseCholeskySymbolic {
    #[new]
    fn new(
        indices: PyArrayLike1<'_, i64>,
        indptr: PyArrayLike1<'_, i64>,
        n: usize,
    ) -> PyResult<Self> {
        let indices_slice = indices.as_slice()?;
        let indptr_slice = indptr.as_slice()?;

        let indices_usize = checked_i64_vec_to_usize(indices_slice, "indices")?;
        let indptr_usize = checked_i64_vec_to_usize(indptr_slice, "indptr")?;

        let cache = sparse_chol::SymbolicCholeskyCache::new(&indices_usize, &indptr_usize, n)?;
        Ok(Self {
            inner: cache,
            indices: indices_usize,
            indptr: indptr_usize,
        })
    }

    fn factor(&self, data: PyArrayLike1<'_, f64>) -> PyResult<SparseCholeskyNumeric> {
        let data_slice = data.as_slice()?;
        let numeric = self.inner.factor(data_slice, &self.indices, &self.indptr)?;
        Ok(SparseCholeskyNumeric { inner: numeric })
    }

    fn n(&self) -> usize {
        self.inner.n()
    }
}

#[pyclass]
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
        let b_array = b.as_array();
        let (n, m) = (b_array.nrows(), b_array.ncols());
        if n != self.inner.n() {
            return Err(linalg::LinalgError::DimensionMismatch(format!(
                "right-hand side has {n} rows, expected {}",
                self.inner.n()
            ))
            .into());
        }

        let mut result = vec![0.0; n * m];

        for j in 0..m {
            let col: Vec<f64> = b_array.column(j).to_vec();
            let x = self.inner.solve(&col)?;
            for (i, value) in x.into_iter().enumerate() {
                result[i * m + j] = value;
            }
        }

        owned_array2(py, result, (n, m))
    }

    fn logdet(&self) -> f64 {
        self.inner.logdet()
    }
}

#[pyfunction]
fn sparse_cholesky_solve<'py>(
    py: Python<'py>,
    a_data: PyArrayLike1<'py, f64>,
    a_indices: PyArrayLike1<'py, i64>,
    a_indptr: PyArrayLike1<'py, i64>,
    a_shape: (usize, usize),
    b: PyArrayLike2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let (result, n, m) = linalg::sparse_cholesky_solve(
        a_data.as_slice()?,
        a_indices.as_slice()?,
        a_indptr.as_slice()?,
        a_shape,
        b.as_array(),
    )?;
    owned_array2(py, result, (n, m))
}

#[pyfunction]
fn sparse_cholesky_logdet<'py>(
    a_data: PyArrayLike1<'py, f64>,
    a_indices: PyArrayLike1<'py, i64>,
    a_indptr: PyArrayLike1<'py, i64>,
    a_shape: (usize, usize),
) -> PyResult<f64> {
    linalg::sparse_cholesky_logdet(
        a_data.as_slice()?,
        a_indices.as_slice()?,
        a_indptr.as_slice()?,
        a_shape,
    )
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
    m.add_function(wrap_pyfunction!(glmm::adaptive_gh_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(nlmm::pnls_step, m)?)?;
    m.add_function(wrap_pyfunction!(nlmm::nlmm_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(simulation::simulate_re_batch, m)?)?;
    m.add_function(wrap_pyfunction!(simulation::compute_zu, m)?)?;
    m.add_function(wrap_pyfunction!(reml_algorithms::mm_reml, m)?)?;
    m.add_function(wrap_pyfunction!(reml_algorithms::augmented_ai_reml, m)?)?;
    m.add_function(wrap_pyfunction!(reml_algorithms::riemannian_reml, m)?)?;
    Ok(())
}

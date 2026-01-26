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

        let indices_usize: Vec<usize> = indices_slice.iter().map(|&i| i as usize).collect();
        let indptr_usize: Vec<usize> = indptr_slice.iter().map(|&i| i as usize).collect();

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

        let mut result = vec![vec![0.0; m]; n];

        for j in 0..m {
            let col: Vec<f64> = b_array.column(j).to_vec();
            let x = self.inner.solve(&col);
            for (i, row) in result.iter_mut().enumerate() {
                row[j] = x[i];
            }
        }

        Ok(PyArray2::from_vec2(py, &result)?.into())
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
    let result = linalg::sparse_cholesky_solve(
        a_data.as_slice()?,
        a_indices.as_slice()?,
        a_indptr.as_slice()?,
        a_shape,
        b.as_array(),
    )?;
    Ok(PyArray2::from_vec2(py, &result)?.into())
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

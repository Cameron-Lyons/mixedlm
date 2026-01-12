use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod linalg;

#[pyfunction]
fn sparse_cholesky_solve(
    py: Python<'_>,
    a_data: PyReadonlyArray1<'_, f64>,
    a_indices: PyReadonlyArray1<'_, i64>,
    a_indptr: PyReadonlyArray1<'_, i64>,
    a_shape: (usize, usize),
    b: PyReadonlyArray2<'_, f64>,
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
fn sparse_cholesky_logdet(
    a_data: PyReadonlyArray1<'_, f64>,
    a_indices: PyReadonlyArray1<'_, i64>,
    a_indptr: PyReadonlyArray1<'_, i64>,
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
fn update_cholesky_factor(
    py: Python<'_>,
    l_data: PyReadonlyArray1<'_, f64>,
    l_indices: PyReadonlyArray1<'_, i64>,
    l_indptr: PyReadonlyArray1<'_, i64>,
    l_shape: (usize, usize),
    theta: PyReadonlyArray1<'_, f64>,
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
    m.add_function(wrap_pyfunction!(sparse_cholesky_solve, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_cholesky_logdet, m)?)?;
    m.add_function(wrap_pyfunction!(update_cholesky_factor, m)?)?;
    Ok(())
}

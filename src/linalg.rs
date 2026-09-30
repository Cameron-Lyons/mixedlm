use numpy::ndarray::ArrayView2;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use crate::csc::CscMatrix;
use crate::sparse_chol::SymbolicCholeskyCache;

#[derive(Debug, Clone)]
pub enum LinalgError {
    NotPositiveDefinite,
    InvalidSparseFormat(String),
    DimensionMismatch(String),
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPositiveDefinite => formatter.write_str("Matrix is not positive definite"),
            Self::InvalidSparseFormat(message) => {
                write!(formatter, "Invalid sparse matrix format: {message}")
            }
            Self::DimensionMismatch(message) => {
                write!(formatter, "Dimension mismatch: {message}")
            }
        }
    }
}

impl std::error::Error for LinalgError {}

impl From<LinalgError> for pyo3::PyErr {
    fn from(err: LinalgError) -> pyo3::PyErr {
        PyValueError::new_err(err.to_string())
    }
}

fn validate_square(shape: (usize, usize)) -> Result<(), LinalgError> {
    if shape.0 != shape.1 {
        return Err(LinalgError::DimensionMismatch(format!(
            "matrix must be square, got {}x{}",
            shape.0, shape.1
        )));
    }
    Ok(())
}

fn csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix, LinalgError> {
    CscMatrix::try_from_i64(data, indices, indptr, shape)
}

pub fn sparse_cholesky_solve(
    a_data: &[f64],
    a_indices: &[i64],
    a_indptr: &[i64],
    a_shape: (usize, usize),
    b: ArrayView2<'_, f64>,
) -> PyResult<(Vec<f64>, usize, usize)> {
    validate_square(a_shape)?;
    if b.nrows() != a_shape.0 {
        return Err(LinalgError::DimensionMismatch(format!(
            "right-hand side has {} rows, expected {}",
            b.nrows(),
            a_shape.0
        ))
        .into());
    }

    let a = csc_from_scipy(a_data, a_indices, a_indptr, a_shape)?;

    let (n, m) = (b.nrows(), b.ncols());
    let cache = SymbolicCholeskyCache::new(a.row_indices(), a.col_offsets(), n)?;
    let factor = cache.factor(a.values(), a.row_indices(), a.col_offsets())?;
    let result = factor.solve(b)?;

    Ok((result, n, m))
}

pub fn sparse_cholesky_logdet(
    a_data: &[f64],
    a_indices: &[i64],
    a_indptr: &[i64],
    a_shape: (usize, usize),
) -> PyResult<f64> {
    validate_square(a_shape)?;
    let a = csc_from_scipy(a_data, a_indices, a_indptr, a_shape)?;

    let cache = SymbolicCholeskyCache::new(a.row_indices(), a.col_offsets(), a.nrows())?;
    let factor = cache.factor(a.values(), a.row_indices(), a.col_offsets())?;
    Ok(factor.logdet())
}

pub fn update_cholesky_factor(
    l_data: &[f64],
    l_indices: &[i64],
    l_indptr: &[i64],
    l_shape: (usize, usize),
    theta: &[f64],
) -> PyResult<(Vec<f64>, Vec<i64>, Vec<i64>)> {
    validate_square(l_shape)?;
    let l = csc_from_scipy(l_data, l_indices, l_indptr, l_shape)?;

    let n = l.nrows();
    let ntheta = theta.len();

    if ntheta == 0 {
        return Ok((l_data.to_vec(), l_indices.to_vec(), l_indptr.to_vec()));
    }

    let q = ((1.0 + 8.0 * ntheta as f64).sqrt() - 1.0) / 2.0;
    let q = q.round() as usize;

    if q * (q + 1) / 2 != ntheta {
        return Err(LinalgError::DimensionMismatch(format!(
            "theta length {} does not correspond to lower triangular matrix",
            ntheta
        ))
        .into());
    }

    let mut new_data = l.values().to_vec();
    let row_indices = l.row_indices();
    let col_offsets = l.col_offsets();

    for col in 0..q.min(n) {
        let col_start = col_offsets[col];
        let col_end = col_offsets[col + 1];

        for idx in col_start..col_end {
            let row = row_indices[idx];
            if row < q {
                let theta_idx = row * (row + 1) / 2 + col;
                if theta_idx < ntheta {
                    new_data[idx] = theta[theta_idx];
                }
            }
        }
    }

    let indices: Vec<i64> = row_indices.iter().map(|&i| i as i64).collect();
    let indptr: Vec<i64> = col_offsets.iter().map(|&i| i as i64).collect();

    Ok((new_data, indices, indptr))
}

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use crate::csc::CscMatrix;
use crate::sparse_chol::{FillOrdering, NumericFactorization, SymbolicCholeskyCache};

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

pub(crate) fn square_csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix, LinalgError> {
    validate_square(shape)?;
    csc_from_scipy(data, indices, indptr, shape)
}

fn factor_csc(a: &CscMatrix, ordering: FillOrdering) -> Result<NumericFactorization, LinalgError> {
    SymbolicCholeskyCache::new(a.row_indices(), a.col_offsets(), a.nrows(), ordering)?
        .factor(a.values())
}

pub fn sparse_cholesky_solve(
    a: &CscMatrix,
    rhs: Vec<f64>,
    shape: (usize, usize),
    ordering: FillOrdering,
) -> PyResult<Vec<f64>> {
    if shape.0 != a.nrows() {
        return Err(LinalgError::DimensionMismatch(format!(
            "right-hand side has {} rows, expected {}",
            shape.0,
            a.nrows()
        ))
        .into());
    }
    Ok(factor_csc(a, ordering)?.solve_owned(rhs, shape)?)
}

pub fn sparse_cholesky_logdet(a: &CscMatrix, ordering: FillOrdering) -> PyResult<f64> {
    Ok(factor_csc(a, ordering)?.logdet())
}

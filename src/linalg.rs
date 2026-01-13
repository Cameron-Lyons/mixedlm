use nalgebra::DMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::factorization::CscCholesky;
use ndarray::ArrayView2;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum LinalgError {
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
    #[error("Invalid sparse matrix format: {0}")]
    InvalidSparseFormat(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

impl From<LinalgError> for pyo3::PyErr {
    fn from(err: LinalgError) -> pyo3::PyErr {
        PyValueError::new_err(err.to_string())
    }
}

fn csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix<f64>, LinalgError> {
    let (nrows, ncols) = shape;

    let indices_usize: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
    let indptr_usize: Vec<usize> = indptr.iter().map(|&i| i as usize).collect();

    CscMatrix::try_from_csc_data(nrows, ncols, indptr_usize, indices_usize, data.to_vec())
        .map_err(|e| LinalgError::InvalidSparseFormat(format!("{:?}", e)))
}

pub fn sparse_cholesky_solve(
    a_data: &[f64],
    a_indices: &[i64],
    a_indptr: &[i64],
    a_shape: (usize, usize),
    b: ArrayView2<'_, f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let a = csc_from_scipy(a_data, a_indices, a_indptr, a_shape)?;

    let cholesky = CscCholesky::factor(&a).map_err(|_| LinalgError::NotPositiveDefinite)?;

    let (n, m) = (b.nrows(), b.ncols());
    let mut result = vec![vec![0.0; m]; n];

    for j in 0..m {
        let col: Vec<f64> = b.column(j).to_vec();
        let b_matrix = DMatrix::from_vec(n, 1, col);
        let x = cholesky.solve(&b_matrix);
        for i in 0..n {
            result[i][j] = x[(i, 0)];
        }
    }

    Ok(result)
}

pub fn sparse_cholesky_logdet(
    a_data: &[f64],
    a_indices: &[i64],
    a_indptr: &[i64],
    a_shape: (usize, usize),
) -> PyResult<f64> {
    let a = csc_from_scipy(a_data, a_indices, a_indptr, a_shape)?;

    let cholesky = CscCholesky::factor(&a).map_err(|_| LinalgError::NotPositiveDefinite)?;

    let l = cholesky.l();
    let mut logdet = 0.0;

    for i in 0..l.nrows() {
        let diag = l.get_entry(i, i).map(|e| e.into_value()).unwrap_or(0.0);
        if diag <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite.into());
        }
        logdet += diag.ln();
    }

    Ok(2.0 * logdet)
}

pub fn update_cholesky_factor(
    l_data: &[f64],
    l_indices: &[i64],
    l_indptr: &[i64],
    l_shape: (usize, usize),
    theta: &[f64],
) -> PyResult<(Vec<f64>, Vec<i64>, Vec<i64>)> {
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

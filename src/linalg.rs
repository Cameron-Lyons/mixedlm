use nalgebra::DMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::factorization::CscCholesky;
use numpy::ndarray::ArrayView2;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

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

fn checked_i64_to_usize(value: i64, field_name: &str, index: usize) -> Result<usize, LinalgError> {
    usize::try_from(value).map_err(|_| {
        LinalgError::InvalidSparseFormat(format!(
            "{field_name}[{index}] must be non-negative, got {value}"
        ))
    })
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
) -> Result<CscMatrix<f64>, LinalgError> {
    let (nrows, ncols) = shape;

    let indices_usize: Vec<usize> = indices
        .iter()
        .enumerate()
        .map(|(idx, &value)| checked_i64_to_usize(value, "indices", idx))
        .collect::<Result<Vec<_>, _>>()?;
    let indptr_usize: Vec<usize> = indptr
        .iter()
        .enumerate()
        .map(|(idx, &value)| checked_i64_to_usize(value, "indptr", idx))
        .collect::<Result<Vec<_>, _>>()?;

    CscMatrix::try_from_csc_data(nrows, ncols, indptr_usize, indices_usize, data.to_vec())
        .map_err(|e| LinalgError::InvalidSparseFormat(format!("{:?}", e)))
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

    let cholesky = CscCholesky::factor(&a).map_err(|_| LinalgError::NotPositiveDefinite)?;

    let (n, m) = (b.nrows(), b.ncols());
    let b_matrix = DMatrix::from_fn(n, m, |row, col| b[(row, col)]);
    let solution = cholesky.solve(&b_matrix);
    let result = (0..n)
        .flat_map(|row| {
            let solution = &solution;
            (0..m).map(move |col| solution[(row, col)])
        })
        .collect();

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

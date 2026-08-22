use crate::linalg::LinalgError;

/// Minimal owned compressed-sparse-column matrix used at the Python boundary.
///
/// Keeping this representation in-tree avoids coupling model code to a sparse
/// solver's container type. Numerical factorization is delegated to `faer`.
#[derive(Debug, Clone)]
pub struct CscMatrix {
    nrows: usize,
    ncols: usize,
    col_offsets: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CscMatrix {
    pub fn try_from_i64(
        data: &[f64],
        indices: &[i64],
        indptr: &[i64],
        shape: (usize, usize),
    ) -> Result<Self, LinalgError> {
        let row_indices = indices
            .iter()
            .enumerate()
            .map(|(index, &value)| checked_i64_to_usize(value, "indices", index))
            .collect::<Result<Vec<_>, _>>()?;
        let col_offsets = indptr
            .iter()
            .enumerate()
            .map(|(index, &value)| checked_i64_to_usize(value, "indptr", index))
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_from_usize(data, &row_indices, &col_offsets, shape)
    }

    pub fn try_from_usize(
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
        shape: (usize, usize),
    ) -> Result<Self, LinalgError> {
        let (nrows, ncols) = shape;
        if indptr.len() != ncols + 1 {
            return Err(LinalgError::InvalidSparseFormat(format!(
                "indptr has length {}, expected {}",
                indptr.len(),
                ncols + 1
            )));
        }
        if indptr.first().copied() != Some(0) {
            return Err(LinalgError::InvalidSparseFormat(
                "indptr must start at zero".to_string(),
            ));
        }
        if data.len() != indices.len() {
            return Err(LinalgError::InvalidSparseFormat(format!(
                "data has length {}, but indices has length {}",
                data.len(),
                indices.len()
            )));
        }
        if indptr.last().copied() != Some(indices.len()) {
            return Err(LinalgError::InvalidSparseFormat(format!(
                "indptr ends at {}, expected {}",
                indptr.last().copied().unwrap_or(0),
                indices.len()
            )));
        }

        let mut is_canonical = true;
        for column in 0..ncols {
            let start = indptr[column];
            let end = indptr[column + 1];
            if start > end || end > indices.len() {
                return Err(LinalgError::InvalidSparseFormat(format!(
                    "invalid range {start}..{end} for column {column}"
                )));
            }
            let mut previous = None;
            for &row in &indices[start..end] {
                if row >= nrows {
                    return Err(LinalgError::InvalidSparseFormat(format!(
                        "row index {row} in column {column} exceeds matrix row count {nrows}"
                    )));
                }
                if previous.is_some_and(|prior| prior >= row) {
                    is_canonical = false;
                }
                previous = Some(row);
            }
        }

        if is_canonical {
            return Ok(Self {
                nrows,
                ncols,
                col_offsets: indptr.to_vec(),
                row_indices: indices.to_vec(),
                values: data.to_vec(),
            });
        }

        let mut col_offsets = Vec::with_capacity(ncols + 1);
        let mut row_indices = Vec::with_capacity(indices.len());
        let mut values = Vec::with_capacity(data.len());
        col_offsets.push(0);
        for column in 0..ncols {
            let start = indptr[column];
            let end = indptr[column + 1];
            let mut entries: Vec<(usize, f64)> = indices[start..end]
                .iter()
                .copied()
                .zip(data[start..end].iter().copied())
                .collect();

            // SciPy permits unsorted indices and duplicate entries in a valid CSC
            // matrix. faer expects canonical columns, so sort and sum duplicates
            // at this boundary.
            entries.sort_unstable_by_key(|&(row, _)| row);
            for (row, value) in entries {
                if row_indices.last() == Some(&row) && row_indices.len() > col_offsets[column] {
                    let last = values
                        .last_mut()
                        .expect("row and value storage stay aligned");
                    *last += value;
                } else {
                    row_indices.push(row);
                    values.push(value);
                }
            }
            col_offsets.push(row_indices.len());
        }

        Ok(Self {
            nrows,
            ncols,
            col_offsets,
            row_indices,
            values,
        })
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn col_offsets(&self) -> &[usize] {
        &self.col_offsets
    }

    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Materialize the upper triangle of a self-adjoint matrix from its lower
    /// triangle. Columns and rows are canonical by construction.
    pub fn self_adjoint_upper_from_lower(&self) -> Self {
        debug_assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        let mut counts = vec![0usize; n];
        for column in 0..n {
            for &row in &self.row_indices[self.col_offsets[column]..self.col_offsets[column + 1]] {
                if row >= column {
                    counts[row] += 1;
                }
            }
        }

        let mut col_offsets = Vec::with_capacity(n + 1);
        col_offsets.push(0);
        for count in counts {
            col_offsets.push(col_offsets.last().copied().unwrap() + count);
        }
        let mut cursors = col_offsets[..n].to_vec();
        let mut row_indices = vec![0usize; *col_offsets.last().unwrap()];
        let mut values = vec![0.0; row_indices.len()];
        for column in 0..n {
            for position in self.col_offsets[column]..self.col_offsets[column + 1] {
                let row = self.row_indices[position];
                if row >= column {
                    let target = cursors[row];
                    row_indices[target] = column;
                    values[target] = self.values[position];
                    cursors[row] += 1;
                }
            }
        }

        Self {
            nrows: n,
            ncols: n,
            col_offsets,
            row_indices,
            values,
        }
    }
}

fn checked_i64_to_usize(value: i64, field_name: &str, index: usize) -> Result<usize, LinalgError> {
    usize::try_from(value).map_err(|_| {
        LinalgError::InvalidSparseFormat(format!(
            "{field_name}[{index}] must be non-negative, got {value}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_csc_invariants() {
        let error = CscMatrix::try_from_i64(&[1.0], &[-1], &[0, 1], (1, 1)).unwrap_err();
        assert!(error.to_string().contains("must be non-negative"));

        let matrix =
            CscMatrix::try_from_usize(&[1.0, 2.0, 3.0], &[1, 0, 1], &[0, 3], (2, 1)).unwrap();
        assert_eq!(matrix.row_indices(), &[0, 1]);
        assert_eq!(matrix.values(), &[2.0, 4.0]);

        let symmetric =
            CscMatrix::try_from_usize(&[4.0, 1.0, 1.0, 3.0], &[0, 1, 0, 1], &[0, 2, 4], (2, 2))
                .unwrap();
        let upper = symmetric.self_adjoint_upper_from_lower();
        assert_eq!(upper.col_offsets(), &[0, 1, 3]);
        assert_eq!(upper.row_indices(), &[0, 0, 1]);
        assert_eq!(upper.values(), &[4.0, 1.0, 3.0]);
    }
}

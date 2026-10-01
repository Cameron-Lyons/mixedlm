use std::sync::OnceLock;

use faer::Mat;

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
    rows: OnceLock<RowStorage>,
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
        let expected_offsets = ncols.checked_add(1).ok_or_else(|| {
            LinalgError::InvalidSparseFormat("matrix column count overflows indptr length".into())
        })?;
        if indptr.len() != expected_offsets {
            return Err(LinalgError::InvalidSparseFormat(format!(
                "indptr has length {}, expected {expected_offsets}",
                indptr.len(),
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
                rows: OnceLock::new(),
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
            rows: OnceLock::new(),
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

    /// Compute Z' diag(weights) Z, using row accumulation for sparse designs.
    /// The immutable row layout is built once and reused as PIRLS weights change.
    pub fn weighted_crossproduct(&self, weights: &[f64]) -> Mat<f64> {
        assert_eq!(weights.len(), self.nrows);
        let mut result = Mat::zeros(self.ncols, self.ncols);
        if self.values.is_empty() {
            return result;
        }
        // At high density, accumulating one column pair at a time avoids
        // repeatedly updating the output matrix and needs no row workspace.
        if self.values.len() / self.nrows > self.ncols / 4 {
            let fully_dense = self.values.len() / self.nrows == self.ncols;
            for left in 0..self.ncols {
                let left_end = self.col_offsets[left + 1];
                for right in 0..=left {
                    let mut i = self.col_offsets[left];
                    let mut j = self.col_offsets[right];
                    let right_end = self.col_offsets[right + 1];
                    let mut sum = 0.0;
                    if fully_dense {
                        // Canonical CSC with n*q entries has every row in every
                        // column, so no row-index intersections are needed.
                        sum = self.values[i..left_end]
                            .iter()
                            .zip(weights)
                            .zip(&self.values[j..right_end])
                            .map(|((&a, &w), &b)| a * w * b)
                            .sum();
                    } else {
                        while i < left_end && j < right_end {
                            let left_row = self.row_indices[i];
                            let right_row = self.row_indices[j];
                            if left_row == right_row {
                                sum += self.values[i] * weights[left_row] * self.values[j];
                                i += 1;
                                j += 1;
                            } else if left_row < right_row {
                                i += 1;
                            } else {
                                j += 1;
                            }
                        }
                    }
                    result[(left, right)] = sum;
                    result[(right, left)] = sum;
                }
            }
            return result;
        }
        let rows = self.rows.get_or_init(|| RowStorage::new(self));
        for (row, &weight) in weights.iter().enumerate() {
            let start = rows.offsets[row];
            let end = rows.offsets[row + 1];
            for left in start..end {
                let left_column = rows.columns[left];
                let weighted_left = weight * rows.values[left];
                for right in left..end {
                    let right_column = rows.columns[right];
                    let value = weighted_left * rows.values[right];
                    result[(left_column, right_column)] += value;
                    if left_column != right_column {
                        result[(right_column, left_column)] += value;
                    }
                }
            }
        }
        result
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
            rows: OnceLock::new(),
        }
    }
}

/// Row-oriented values for repeated weighted crossproducts. Columns within
/// each row are sorted because the source CSC is traversed in column order.
#[derive(Debug, Clone)]
struct RowStorage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}

impl RowStorage {
    fn new(matrix: &CscMatrix) -> Self {
        let mut offsets = vec![0; matrix.nrows + 1];
        for &row in &matrix.row_indices {
            offsets[row + 1] += 1;
        }
        for row in 0..matrix.nrows {
            offsets[row + 1] += offsets[row];
        }
        let mut positions = offsets[..matrix.nrows].to_vec();
        let mut columns = vec![0; matrix.values.len()];
        let mut values = vec![0.0; matrix.values.len()];
        for column in 0..matrix.ncols {
            for index in matrix.col_offsets[column]..matrix.col_offsets[column + 1] {
                let row = matrix.row_indices[index];
                let position = positions[row];
                columns[position] = column;
                values[position] = matrix.values[index];
                positions[row] += 1;
            }
        }
        Self {
            offsets,
            columns,
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
    fn crossproduct_reweights_sparse_and_dense_designs() {
        // Duplicate, unsorted entries, an explicit zero, and an empty column.
        for q in [3, 8] {
            let mut offsets = vec![0, 3, 3];
            offsets.resize(q + 1, 6);
            let matrix = CscMatrix::try_from_usize(
                &[2.0, 1.0, 3.0, 0.0, -2.0, 4.0],
                &[2, 0, 2, 1, 0, 2],
                &offsets,
                (4, q),
            )
            .unwrap();
            let dense = Mat::from_fn(4, q, |row, col| {
                if col < 3 {
                    [[1.0, 0.0, -2.0], [0.0; 3], [5.0, 0.0, 4.0], [0.0; 3]][row][col]
                } else {
                    0.0
                }
            });
            for weights in [[1.0; 4], [0.25, 2.0, 3.0, 0.5], [2.0, 0.0, 0.5, 4.0]] {
                let actual = matrix.weighted_crossproduct(&weights);
                let weighted = Mat::from_fn(4, q, |row, col| weights[row] * dense[(row, col)]);
                let expected = dense.transpose() * weighted;
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn crossproducts_handle_empty_dimensions() {
        for (n, q) in [(0, 0), (0, 3), (4, 0), (4, 3)] {
            let matrix = CscMatrix::try_from_usize(&[], &[], &vec![0; q + 1], (n, q)).unwrap();
            let weights = vec![1.0; n];
            assert_eq!(
                matrix.weighted_crossproduct(&weights),
                Mat::<f64>::zeros(q, q)
            );
        }
    }

    #[test]
    fn crossproduct_layout_can_initialize_from_multiple_threads() {
        let matrix =
            CscMatrix::try_from_usize(&[1.0, 2.0], &[0, 1], &[0, 2, 2, 2, 2, 2, 2, 2, 2], (2, 8))
                .unwrap();
        std::thread::scope(|scope| {
            let matrix = &matrix;
            let first = scope.spawn(move || matrix.weighted_crossproduct(&[1.0, 2.0]));
            let second = scope.spawn(move || matrix.weighted_crossproduct(&[3.0, 4.0]));
            assert_eq!(first.join().unwrap()[(0, 0)], 9.0);
            assert_eq!(second.join().unwrap()[(0, 0)], 19.0);
        });
    }

    #[test]
    fn sparse_column_count_overflow_is_an_error_in_all_build_modes() {
        let error = CscMatrix::try_from_usize(&[], &[], &[], (1, usize::MAX)).unwrap_err();
        assert!(matches!(error, LinalgError::InvalidSparseFormat(_)));
        assert!(error.to_string().contains("overflow"));
    }

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

use std::borrow::Cow;
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
        let (row_indices, col_offsets, is_canonical) =
            validate_i64_parts(data.len(), indices, indptr, shape)?;
        Ok(Self::from_validated_parts(
            data,
            row_indices.into(),
            col_offsets.into(),
            shape,
            is_canonical,
        ))
    }

    pub fn try_from_usize(
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
        shape: (usize, usize),
    ) -> Result<Self, LinalgError> {
        Self::try_from_parts(data, indices.into(), indptr.into(), shape)
    }

    fn try_from_parts(
        data: &[f64],
        indices: Cow<'_, [usize]>,
        indptr: Cow<'_, [usize]>,
        shape: (usize, usize),
    ) -> Result<Self, LinalgError> {
        let is_canonical = validate_parts(data.len(), &indices, &indptr, shape)?;
        Ok(Self::from_validated_parts(
            data,
            indices,
            indptr,
            shape,
            is_canonical,
        ))
    }

    fn from_validated_parts(
        data: &[f64],
        indices: Cow<'_, [usize]>,
        indptr: Cow<'_, [usize]>,
        shape: (usize, usize),
        is_canonical: bool,
    ) -> Self {
        let (nrows, ncols) = shape;
        if is_canonical {
            return Self {
                nrows,
                ncols,
                // Signed inputs already own their converted index buffers.
                col_offsets: indptr.into_owned(),
                row_indices: indices.into_owned(),
                values: data.to_vec(),
                rows: OnceLock::new(),
            };
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

        Self {
            nrows,
            ncols,
            col_offsets,
            row_indices,
            values,
            rows: OnceLock::new(),
        }
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

    /// Subtract a matrix-vector product without allocating a prediction vector.
    pub fn subtract_product(&self, vector: faer::ColRef<'_, f64>, output: &mut [f64]) {
        assert_eq!(vector.nrows(), self.ncols);
        assert_eq!(output.len(), self.nrows);
        match self.rows.get() {
            Some(rows) if self.values.len() > self.nrows => {
                // The cached rows reduce output updates when there are more
                // entries than observations. Sorted columns retain the CSC
                // subtraction order, including cancellation-sensitive values.
                for (row, value) in output.iter_mut().enumerate() {
                    let mut residual = *value;
                    for index in rows.offsets[row]..rows.offsets[row + 1] {
                        residual -= rows.values[index] * vector[rows.columns[index]];
                    }
                    *value = residual;
                }
            }
            _ => {
                // Very sparse and fully populated designs, and callers with
                // cached crossproducts, need no new traversal of empty rows.
                for column in 0..self.ncols {
                    for index in self.col_offsets[column]..self.col_offsets[column + 1] {
                        output[self.row_indices[index]] -= self.values[index] * vector[column];
                    }
                }
            }
        }
    }

    /// Compute Z' diag(weights) Z, using row accumulation for sparse designs.
    /// The immutable row layout is built once and reused as PIRLS weights change.
    pub fn weighted_crossproduct(&self, weights: &[f64]) -> Mat<f64> {
        assert_eq!(weights.len(), self.nrows);
        let mut result = Mat::zeros(self.ncols, self.ncols);
        if self.values.is_empty() {
            return result;
        }
        // Keep column accumulation for small and nearly full designs.
        // Larger incomplete designs amortize a row layout across column pairs.
        if self.values.len() / self.nrows > self.ncols / 4 {
            let fully_dense = self.values.len() / self.nrows == self.ncols;
            if self.ncols >= 32 && self.values.len() / self.nrows < self.ncols - self.ncols / 32 {
                self.accumulate_dense_rows(weights, &mut result);
                return result;
            }
            for left in 0..self.ncols {
                for right in 0..=left {
                    let sum = self.weighted_column_product(weights, left, right, fully_dense);
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

    fn accumulate_dense_rows(&self, weights: &[f64], result: &mut Mat<f64>) {
        let rows = self.rows.get_or_init(|| RowStorage::new(self));
        let mut weighted = vec![0.0; self.ncols];
        // Match the column path: preweight the higher-column operand,
        // and add each cell's contributions in ascending row order.
        for (row, &weight) in weights.iter().enumerate() {
            let start = rows.offsets[row];
            let end = rows.offsets[row + 1];
            for entry in start..end {
                weighted[entry - start] = rows.values[entry] * weight;
            }
            for low in start..end {
                let column = rows.columns[low];
                for high in low..end {
                    result[(rows.columns[high], column)] +=
                        weighted[high - start] * rows.values[low];
                }
            }
        }
        for column in 0..self.ncols {
            for row in column + 1..self.ncols {
                result[(column, row)] = result[(row, column)];
            }
        }
    }

    fn weighted_column_product(
        &self,
        weights: &[f64],
        left: usize,
        right: usize,
        fully_dense: bool,
    ) -> f64 {
        let mut i = self.col_offsets[left];
        let mut j = self.col_offsets[right];
        let left_end = self.col_offsets[left + 1];
        let right_end = self.col_offsets[right + 1];
        if fully_dense {
            // Canonical CSC with n*q entries contains every row in each column.
            return self.values[i..left_end]
                .iter()
                .zip(weights)
                .zip(&self.values[j..right_end])
                .map(|((&a, &w), &b)| a * w * b)
                .sum();
        }
        let mut sum = 0.0;
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
        sum
    }

    /// Compute independent repeated blocks without allocating a square matrix.
    /// Each pair gives (number of blocks, block width); results stack each set.
    /// A row spanning blocks, even through stored zeros, uses the dense fallback.
    pub fn weighted_repeated_block_crossproducts(
        &self,
        weights: &[f64],
        blocks: &[(usize, usize)],
    ) -> Option<Vec<Mat<f64>>> {
        assert_eq!(weights.len(), self.nrows);
        let mut column_sets = Vec::with_capacity(self.ncols);
        let mut column_levels = Vec::with_capacity(self.ncols);
        let mut offsets = Vec::with_capacity(blocks.len());
        for (set, &(count, width)) in blocks.iter().enumerate() {
            offsets.push(column_sets.len());
            for _ in 0..count {
                let start = column_sets.len();
                column_sets.extend(std::iter::repeat_n(set, width));
                column_levels.extend(std::iter::repeat_n(start, width));
            }
        }
        assert_eq!(column_sets.len(), self.ncols);
        let mut result: Vec<Mat<f64>> = blocks
            .iter()
            .map(|&(count, width)| Mat::zeros(count * width, width))
            .collect();
        if self.values.is_empty() {
            return Some(result);
        }
        // Match both arithmetic paths of weighted_crossproduct exactly.
        if self.values.len() / self.nrows > self.ncols / 4 {
            // Dense column accumulation needs no permanent row workspace.
            if column_levels.first() != column_levels.last() {
                let mut owners = vec![usize::MAX; self.nrows];
                for (column, &level) in column_levels.iter().enumerate() {
                    for &row in
                        &self.row_indices[self.col_offsets[column]..self.col_offsets[column + 1]]
                    {
                        if owners[row] == usize::MAX {
                            owners[row] = level;
                        } else if owners[row] != level {
                            return None;
                        }
                    }
                }
            }
            let fully_dense = self.values.len() / self.nrows == self.ncols;
            for (set, &(count, width)) in blocks.iter().enumerate() {
                for level in 0..count {
                    let local = level * width;
                    let start = offsets[set] + local;
                    for left in 0..width {
                        for right in 0..=left {
                            let value = self.weighted_column_product(
                                weights,
                                start + left,
                                start + right,
                                fully_dense,
                            );
                            result[set][(local + left, right)] = value;
                            result[set][(local + right, left)] = value;
                        }
                    }
                }
            }
        } else {
            let rows = self.rows.get_or_init(|| RowStorage::new(self));
            for (row, &weight) in weights.iter().enumerate() {
                let start = rows.offsets[row];
                let end = rows.offsets[row + 1];
                if start == end {
                    continue;
                }
                let level = column_levels[rows.columns[start]];
                if rows.columns[start..end]
                    .iter()
                    .any(|&column| column_levels[column] != level)
                {
                    return None;
                }
                let set = column_sets[rows.columns[start]];
                let output = &mut result[set];
                for left in start..end {
                    let left_column = rows.columns[left];
                    let weighted_left = weight * rows.values[left];
                    for right in left..end {
                        let right_column = rows.columns[right];
                        let value = weighted_left * rows.values[right];
                        output[(left_column - offsets[set], right_column - level)] += value;
                        if left_column != right_column {
                            output[(right_column - offsets[set], left_column - level)] += value;
                        }
                    }
                }
            }
        }
        Some(result)
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

/// Validate signed Python CSC buffers without reordering or merging their entries.
/// Returns owned index storage so callers can safely release the Python GIL.
pub(crate) fn validate_i64_parts(
    data_len: usize,
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<(Vec<usize>, Vec<usize>, bool), LinalgError> {
    let row_indices = checked_indices(indices, "indices")?;
    let col_offsets = checked_indices(indptr, "indptr")?;
    let is_canonical = validate_parts(data_len, &row_indices, &col_offsets, shape)?;
    Ok((row_indices, col_offsets, is_canonical))
}

/// Check CSC structure and report whether the entries are already canonical.
fn validate_parts(
    data_len: usize,
    indices: &[usize],
    indptr: &[usize],
    shape: (usize, usize),
) -> Result<bool, LinalgError> {
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
    if data_len != indices.len() {
        return Err(LinalgError::InvalidSparseFormat(format!(
            "data has length {}, but indices has length {}",
            data_len,
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
    Ok(is_canonical)
}

fn checked_indices(values: &[i64], field_name: &str) -> Result<Vec<usize>, LinalgError> {
    let mut converted = Vec::with_capacity(values.len());
    for (index, &value) in values.iter().enumerate() {
        converted.push(usize::try_from(value).map_err(|_| {
            LinalgError::InvalidSparseFormat(format!(
                "{field_name}[{index}] must be non-negative, got {value}"
            ))
        })?);
    }
    Ok(converted)
}

#[cfg(test)]
mod tests {
    #[test]
    fn repeated_block_crossproducts_match_both_dense_accumulation_paths() {
        for blocks in [
            vec![(1, 3)],
            vec![(2, 3)],
            vec![(4, 2)],
            vec![(3, 2), (2, 1)],
        ] {
            let levels: usize = blocks.iter().map(|&(count, _)| count).sum();
            let q: usize = blocks.iter().map(|&(count, width)| count * width).sum();
            let n = 7 * levels + 1;
            let mut column_levels = Vec::new();
            let mut level = 0;
            for &(count, width) in &blocks {
                for _ in 0..count {
                    column_levels.extend(std::iter::repeat_n(level, width));
                    level += 1;
                }
            }
            for empty in [false, true] {
                let (mut values, mut indices, mut offsets) = (Vec::new(), Vec::new(), vec![0]);
                for (column, &level) in column_levels.iter().enumerate() {
                    for row in 0..n {
                        if !empty && row % levels == level {
                            values.push(((3 * row + column + 1) % 7) as f64 / 9.0 - 0.25);
                            indices.push(row);
                        }
                    }
                    offsets.push(values.len());
                }
                let matrix =
                    super::CscMatrix::try_from_usize(&values, &indices, &offsets, (n, q)).unwrap();
                let weights: Vec<_> = (0..n).map(|row| 0.25 + row as f64 / 11.0).collect();
                let compact = matrix
                    .weighted_repeated_block_crossproducts(&weights, &blocks)
                    .unwrap();
                let mut expanded = faer::Mat::zeros(q, q);
                let mut offset = 0;
                for (&(count, width), block) in blocks.iter().zip(&compact) {
                    assert_eq!(block.shape(), (count * width, width));
                    for level in 0..count {
                        expanded
                            .submatrix_mut(
                                offset + level * width,
                                offset + level * width,
                                width,
                                width,
                            )
                            .copy_from(block.subrows(level * width, width));
                    }
                    offset += count * width;
                }
                assert_eq!(expanded, matrix.weighted_crossproduct(&weights));
            }
        }
    }

    #[test]
    fn repeated_blocks_reject_cross_level_entries_including_tiny_values_and_stored_zeros() {
        for value in [1.0, 1e-300, 0.0] {
            let matrix =
                super::CscMatrix::try_from_usize(&[1.0, value], &[0, 0], &[0, 1, 2], (2, 2))
                    .unwrap();
            assert!(
                matrix
                    .weighted_repeated_block_crossproducts(&[1.0; 2], &[(2, 1)])
                    .is_none()
            );
            assert!(
                matrix
                    .weighted_repeated_block_crossproducts(&[1.0; 2], &[(1, 1), (1, 1)])
                    .is_none()
            );
        }
        let cancelling = super::CscMatrix::try_from_usize(
            &[1.0, 1.0, 1.0, -1.0],
            &[0, 1, 0, 1],
            &[0, 2, 4],
            (2, 2),
        )
        .unwrap();
        assert_eq!(cancelling.weighted_crossproduct(&[1.0; 2])[(0, 1)], 0.0);
        assert!(
            cancelling
                .weighted_repeated_block_crossproducts(&[1.0; 2], &[(2, 1)])
                .is_none()
        );
    }

    use super::*;

    #[test]
    fn product_subtraction_handles_irregular_columns_and_both_layouts() {
        // Duplicate, unsorted entries, explicit zeros, and empty rows/columns.
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
            let vector = faer::Col::from_fn(q, |i| i as f64 - 1.0);
            let mut residual = [1.0, -2.0, 3.0, 4.0];
            matrix.subtract_product(vector.as_ref(), &mut residual);
            assert_eq!(residual, [4.0, -2.0, 4.0, 4.0]);
            assert!(matrix.rows.get().is_none());
            matrix.weighted_crossproduct(&[0.25, 2.0, 3.0, 0.5]);
            assert_eq!(matrix.rows.get().is_some(), q == 8);
            let mut repeated = [1.0, -2.0, 3.0, 4.0];
            matrix.subtract_product(vector.as_ref(), &mut repeated);
            assert_eq!(repeated, residual);
        }
    }

    #[test]
    fn cached_product_subtraction_preserves_cancellation_order() {
        let matrix = CscMatrix::try_from_usize(
            &[1e16, 1.0, -1e16],
            &[0, 0, 0],
            &[0, 1, 2, 3, 3, 3, 3, 3, 3],
            (2, 8),
        )
        .unwrap();
        let vector = faer::Col::from_fn(8, |_| 1.0);
        let mut before = [1.0, 2.0];
        matrix.subtract_product(vector.as_ref(), &mut before);
        // Forming Z * vector first and then subtracting would return 1 here.
        assert_eq!(before, [0.0, 2.0]);
        matrix.weighted_crossproduct(&[1.0; 2]);
        assert!(matrix.rows.get().is_some());
        let mut after = [1.0, 2.0];
        matrix.subtract_product(vector.as_ref(), &mut after);
        assert_eq!(after, before);
    }

    #[test]
    fn product_subtraction_preserves_empty_rows_and_dimensions() {
        for (n, q) in [(0, 0), (0, 3), (4, 0), (4, 3)] {
            let matrix = CscMatrix::try_from_usize(&[], &[], &vec![0; q + 1], (n, q)).unwrap();
            let vector = faer::Col::from_fn(q, |_| 1.0);
            let mut residual = vec![2.0; n];
            matrix.subtract_product(vector.as_ref(), &mut residual);
            assert_eq!(residual, vec![2.0; n]);
            matrix.weighted_crossproduct(&vec![1.0; n]);
            matrix.subtract_product(vector.as_ref(), &mut residual);
            assert_eq!(residual, vec![2.0; n]);
            assert!(matrix.rows.get().is_none());
        }
    }

    #[test]
    fn product_subtraction_can_share_a_cache_being_initialized() {
        let matrix = CscMatrix::try_from_usize(
            &[1.0, 3.0, 2.0],
            &[0, 0, 1],
            &[0, 1, 2, 3, 3, 3, 3, 3, 3],
            (2, 8),
        )
        .unwrap();
        std::thread::scope(|scope| {
            let matrix = &matrix;
            let handles: Vec<_> = (0..8)
                .map(|index| {
                    scope.spawn(move || {
                        if index % 2 == 0 {
                            matrix.weighted_crossproduct(&[1.0, 2.0]);
                        }
                        let scale = index as f64 + 1.0;
                        let vector = faer::Col::from_fn(8, |_| scale);
                        let mut residual = [10.0; 2];
                        matrix.subtract_product(vector.as_ref(), &mut residual);
                        assert_eq!(residual, [10.0 - 4.0 * scale, 10.0 - 2.0 * scale]);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert!(matrix.rows.get().is_some());
    }

    #[test]
    fn canonical_owned_indices_are_reused_and_values_are_snapshotted() {
        let mut data = vec![2.0, 0.0, -1.0];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 2, 3];
        let indices_ptr = indices.as_ptr();
        let indptr_ptr = indptr.as_ptr();
        let matrix =
            CscMatrix::try_from_parts(&data, Cow::Owned(indices), Cow::Owned(indptr), (3, 3))
                .unwrap();
        assert_eq!(matrix.row_indices().as_ptr(), indices_ptr);
        assert_eq!(matrix.col_offsets().as_ptr(), indptr_ptr);
        data.fill(100.0);
        assert_eq!(matrix.values(), &[2.0, 0.0, -1.0]);
    }

    #[test]
    fn raw_signed_validation_preserves_duplicates_and_original_entry_order() {
        let indices = [2, 0, 2, 1, 0, 2];
        let indptr = [0, 3, 3, 6];
        let (rows, offsets, is_canonical) =
            super::validate_i64_parts(6, &indices, &indptr, (4, 3)).unwrap();
        assert_eq!(rows, indices.map(|value| value as usize));
        assert_eq!(offsets, indptr.map(|value| value as usize));
        assert!(!is_canonical);
        assert!(
            super::validate_i64_parts(3, &[0, 2, 1], &[0, 2, 2, 3], (3, 3))
                .unwrap()
                .2
        );
    }

    #[test]
    fn signed_and_unsigned_constructors_preserve_irregular_columns() {
        for (data, indices, indptr, shape) in [
            (vec![], vec![], vec![0], (0, 0)),
            (vec![], vec![], vec![0, 0, 0, 0], (0, 3)),
            (vec![], vec![], vec![0, 0, 0], (5, 2)),
            (
                vec![2.0, 1.0, 3.0, 0.0, -2.0, 4.0],
                vec![2, 0, 2, 1, 0, 2],
                vec![0, 3, 3, 6],
                (4, 3),
            ),
        ] {
            let signed_indices: Vec<i64> = indices.iter().map(|&value| value as i64).collect();
            let signed_indptr: Vec<i64> = indptr.iter().map(|&value| value as i64).collect();
            let signed =
                CscMatrix::try_from_i64(&data, &signed_indices, &signed_indptr, shape).unwrap();
            let unsigned = CscMatrix::try_from_usize(&data, &indices, &indptr, shape).unwrap();
            assert_eq!(signed.col_offsets(), unsigned.col_offsets());
            assert_eq!(signed.row_indices(), unsigned.row_indices());
            assert_eq!(signed.values(), unsigned.values());
            if !data.is_empty() {
                assert_eq!(signed.col_offsets(), &[0, 2, 2, 5]);
                assert_eq!(signed.row_indices(), &[0, 2, 0, 1, 2]);
                assert_eq!(signed.values(), &[1.0, 5.0, -2.0, 0.0, 4.0]);
            }
        }
    }

    #[test]
    fn index_ownership_preserves_validation_errors() {
        for (data, indices, indptr, shape, expected) in [
            (vec![], vec![], vec![], (1, usize::MAX), "overflow"),
            (vec![1.0], vec![0], vec![0], (2, 1), "indptr has length"),
            (vec![1.0], vec![0], vec![1, 1], (2, 1), "start at zero"),
            (vec![], vec![0], vec![0, 1], (2, 1), "data has length"),
            (vec![1.0], vec![0], vec![0, 0], (2, 1), "indptr ends at"),
            (vec![1.0], vec![0], vec![0, 2, 1], (2, 2), "invalid range"),
            (
                vec![1.0],
                vec![2],
                vec![0, 1],
                (2, 1),
                "exceeds matrix row count",
            ),
        ] {
            let signed_indices: Vec<i64> = indices.iter().map(|&value| value as i64).collect();
            let signed_indptr: Vec<i64> = indptr.iter().map(|&value| value as i64).collect();
            let signed = CscMatrix::try_from_i64(&data, &signed_indices, &signed_indptr, shape)
                .unwrap_err()
                .to_string();
            let unsigned = CscMatrix::try_from_usize(&data, &indices, &indptr, shape)
                .unwrap_err()
                .to_string();
            assert_eq!(signed, unsigned);
            assert!(signed.contains(expected), "{signed}");
        }
        for (indices, indptr, expected) in [
            (
                vec![0, -1],
                vec![0, 2],
                "indices[1] must be non-negative, got -1",
            ),
            (
                vec![0, 1],
                vec![0, -2],
                "indptr[1] must be non-negative, got -2",
            ),
        ] {
            let error = CscMatrix::try_from_i64(&[1.0, 2.0], &indices, &indptr, (2, 1))
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn dense_row_products_preserve_column_order_and_missing_entries() {
        for q in [31, 32, 33] {
            for layout in ["partial", "nearly_full", "full"] {
                let mut values = Vec::new();
                let mut rows = Vec::new();
                let mut offsets = vec![0];
                for column in 0..q {
                    for row in 0..12 {
                        if (layout == "partial" && (row == 4 || (row + column) % 5 == 0))
                            || (layout == "nearly_full" && row == 4 && column == 0)
                        {
                            continue;
                        }
                        let value = match row {
                            0 => 1e16,
                            1 => 1.0,
                            2 => -1e16,
                            3 => -0.0,
                            5 => f64::MIN_POSITIVE,
                            6 => -f64::MIN_POSITIVE,
                            _ => (row + column) as f64 / 8.0,
                        };
                        values.push(value * if column % 2 == 0 { 1.0 } else { -0.5 });
                        rows.push(row);
                    }
                    offsets.push(rows.len());
                }
                let matrix = CscMatrix::try_from_usize(&values, &rows, &offsets, (12, q)).unwrap();
                for weights in [
                    [1.0; 12],
                    [
                        1.0, -2.0, -1.0, 0.0, 3.0, 1e-308, 2.0, 0.3, 1.1, 1.0, 2.0, 1.0,
                    ],
                    [
                        1.0,
                        2.0,
                        0.25,
                        f64::INFINITY,
                        f64::NAN,
                        0.0,
                        2.0,
                        0.5,
                        4.0,
                        1.0,
                        2.0,
                        1.0,
                    ],
                ] {
                    let actual = matrix.weighted_crossproduct(&weights);
                    for left in 0..q {
                        for right in 0..=left {
                            let right_start = offsets[right];
                            let right_rows = &rows[right_start..offsets[right + 1]];
                            let mut expected = 0.0;
                            for entry in offsets[left]..offsets[left + 1] {
                                if let Ok(position) = right_rows.binary_search(&rows[entry]) {
                                    expected += values[entry]
                                        * weights[rows[entry]]
                                        * values[right_start + position];
                                }
                            }
                            for value in [actual[(left, right)], actual[(right, left)]] {
                                if expected.is_nan() {
                                    assert!(value.is_nan());
                                } else {
                                    assert_eq!(value.to_bits(), expected.to_bits());
                                }
                            }
                        }
                    }
                    assert_eq!(matrix.rows.get().is_some(), q >= 32 && layout == "partial");
                }
            }
        }
    }

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

use std::collections::BTreeSet;
use std::fmt;

use faer::MatMut;
use faer::linalg::solvers::Llt;

use crate::covariance::CovarianceFactor;
use crate::csc::CscMatrix;
use crate::linalg::LinalgError;
use crate::sparse_chol::{NumericFactorization, SymbolicCholeskyCache};

/// The row layout of Z Lambda and the reusable pattern of its penalized crossproduct.
pub struct SparseWeightedDesign {
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    design_values: Vec<f64>,
    indices: Vec<usize>,
    offsets: Vec<usize>,
    symbolic: SymbolicCholeskyCache,
}

impl fmt::Debug for SparseWeightedDesign {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SparseWeightedDesign")
            .field("dimension", &self.symbolic.n())
            .field("design_nonzeros", &self.design_values.len())
            .field("precision_nonzeros", &self.indices.len())
            .finish_non_exhaustive()
    }
}

impl SparseWeightedDesign {
    pub fn new(design: &CscMatrix, covariance: &CovarianceFactor) -> Option<Self> {
        let q = design.ncols();
        // Small or densely populated systems benefit from the dense kernels.
        if q < 128 || design.values().len() / design.nrows().max(1) > q / 8 {
            return None;
        }
        let design = covariance.sparse_design(design)?;
        if design.values().len() / design.nrows().max(1) > q / 8 {
            return None;
        }
        let mut row_offsets = vec![0; design.nrows() + 1];
        for &row in design.row_indices() {
            row_offsets[row + 1] += 1;
        }
        for row in 0..design.nrows() {
            row_offsets[row + 1] += row_offsets[row];
        }
        let mut cursors = row_offsets[..design.nrows()].to_vec();
        let mut columns = vec![0; design.values().len()];
        let mut design_values = vec![0.0; design.values().len()];
        for column in 0..q {
            for entry in design.col_offsets()[column]..design.col_offsets()[column + 1] {
                let row = design.row_indices()[entry];
                let position = cursors[row];
                columns[position] = column;
                design_values[position] = design.values()[entry];
                cursors[row] += 1;
            }
        }

        let mut pattern: Vec<_> = (0..q).map(|column| BTreeSet::from([column])).collect();
        let mut nonzeros = q;
        let pattern_limit = q.saturating_mul(q) / 8;
        for row in 0..design.nrows() {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            for left in start..end {
                let column = columns[left];
                for &right in &columns[left..end] {
                    if pattern[column].insert(right) {
                        nonzeros += 1;
                        if nonzeros > pattern_limit {
                            return None;
                        }
                    }
                }
            }
        }
        let mut offsets = vec![0];
        let mut indices = Vec::with_capacity(nonzeros);
        for column in pattern {
            indices.extend(column);
            offsets.push(indices.len());
        }
        let symbolic = SymbolicCholeskyCache::new_amd(&indices, &offsets, q).ok()?;
        // Symbolic fill can make a sparse input expensive to factor. Preserve
        // the dense path when the factor itself would lose its sparsity.
        if symbolic.factor_nonzeros() > q.saturating_mul(q) / 4 {
            return None;
        }
        Some(Self {
            row_offsets,
            columns,
            design_values,
            indices,
            offsets,
            symbolic,
        })
    }

    pub fn factor(
        &self,
        weights: &[f64],
        regularization: f64,
    ) -> Result<NumericFactorization, LinalgError> {
        assert_eq!(weights.len() + 1, self.row_offsets.len());
        let mut values = vec![0.0; self.indices.len()];
        for &diagonal in &self.offsets[..self.symbolic.n()] {
            values[diagonal] = 1.0 + regularization;
        }
        for (row, &weight) in weights.iter().enumerate() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for left in start..end {
                let column = self.columns[left];
                let base = self.offsets[column];
                let pattern = &self.indices[base..self.offsets[column + 1]];
                let weighted = self.design_values[left] * weight;
                for right in left..end {
                    let position = pattern
                        .binary_search(&self.columns[right])
                        .expect("symbolic pattern includes each row crossproduct");
                    values[base + position] += weighted * self.design_values[right];
                }
            }
        }
        self.symbolic.factor(&values, &self.indices, &self.offsets)
    }
}

pub enum RandomFactor {
    Dense(Llt<f64>),
    Sparse(NumericFactorization),
}

impl RandomFactor {
    pub fn solve_lower_in_place(&self, rhs: MatMut<'_, f64>) {
        match self {
            Self::Dense(factor) => factor.L().solve_lower_triangular_in_place(rhs),
            Self::Sparse(factor) => factor.solve_lower_in_place(rhs),
        }
    }

    pub fn solve_upper_in_place(&self, rhs: MatMut<'_, f64>) {
        match self {
            Self::Dense(factor) => factor.L().transpose().solve_upper_triangular_in_place(rhs),
            Self::Sparse(factor) => factor.solve_upper_in_place(rhs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance::RandomEffectStructure;
    use faer::Mat;

    fn covariance(q: usize, scale: f64) -> CovarianceFactor {
        CovarianceFactor::new(
            &[scale],
            &[RandomEffectStructure {
                n_levels: q,
                n_terms: 1,
                correlated: true,
            }],
        )
    }

    #[test]
    fn independent_groups_use_linear_storage_and_reuse_weighted_pattern() {
        let q = 4096;
        let offsets: Vec<_> = (0..=q).map(|i| 2 * i).collect();
        let rows: Vec<_> = (0..2 * q).collect();
        let input =
            CscMatrix::try_from_usize(&vec![1.0; 2 * q], &rows, &offsets, (2 * q, q)).unwrap();
        let system = SparseWeightedDesign::new(&input, &covariance(q, 0.6)).unwrap();
        assert_eq!(system.indices.len(), q);
        assert_eq!(system.symbolic.factor_nonzeros(), q);
        for varying in [false, true] {
            let weights: Vec<_> = (0..2 * q)
                .map(|i| {
                    if varying {
                        0.5 + (i % 7) as f64 / 4.0
                    } else {
                        1.0
                    }
                })
                .collect();
            let diagonal: Vec<_> = (0..q)
                .map(|i| 1.0 + 0.36 * (weights[2 * i] + weights[2 * i + 1]))
                .collect();
            let factor = system.factor(&weights, 0.0).unwrap();
            let mut rhs = Mat::from_fn(q, 4, |i, j| (i % 11 + j) as f64 / 5.0);
            let expected = rhs.clone();
            factor.solve_lower_in_place(rhs.as_mut());
            for j in 0..4 {
                for i in 0..q {
                    assert!((rhs[(i, j)] * diagonal[i].sqrt() - expected[(i, j)]).abs() < 1e-12);
                }
            }
            factor.solve_upper_in_place(rhs.as_mut());
            for j in 0..4 {
                for i in 0..q {
                    assert!((rhs[(i, j)] * diagonal[i] - expected[(i, j)]).abs() < 1e-12);
                }
            }
            let logdet: f64 = diagonal.iter().map(|value| value.ln()).sum();
            assert!((factor.logdet() - logdet).abs() < 1e-9);
        }
    }

    #[test]
    fn dense_design_and_dense_precision_keep_the_dense_path() {
        let q = 128;
        let dense = CscMatrix::try_from_usize(
            &vec![1.0; q],
            &vec![0; q],
            &(0..=q).collect::<Vec<_>>(),
            (1, q),
        )
        .unwrap();
        assert!(SparseWeightedDesign::new(&dense, &covariance(q, 1.0)).is_none());
        // A sparse design can still produce a dense normal equation pattern.
        let mut column_rows = vec![Vec::new(); q];
        let mut row = 0;
        for left in 0..q {
            for right in left + 1..q {
                column_rows[left].push(row);
                column_rows[right].push(row);
                row += 1;
            }
        }
        let mut offsets = vec![0];
        let mut rows = Vec::new();
        for column in column_rows {
            rows.extend(column);
            offsets.push(rows.len());
        }
        let pairwise =
            CscMatrix::try_from_usize(&vec![1.0; rows.len()], &rows, &offsets, (row, q)).unwrap();
        assert!(SparseWeightedDesign::new(&pairwise, &covariance(q, 1.0)).is_none());
    }

    #[test]
    fn ordering_preserves_sparse_star_and_solves_in_original_model_order() {
        use faer::prelude::Solve;
        let q = 128;
        // Natural ordering eliminates the hub first and fills the entire factor.
        // AMD eliminates the leaves first, preserving linear storage.

        let mut offsets = vec![0, q - 1];
        let mut rows: Vec<_> = (0..q - 1).collect();
        for column in 1..q {
            rows.push(column - 1);
            offsets.push(rows.len());
        }
        let star =
            CscMatrix::try_from_usize(&vec![1.0; rows.len()], &rows, &offsets, (q - 1, q)).unwrap();
        let system = SparseWeightedDesign::new(&star, &covariance(q, 1.0)).unwrap();
        assert_eq!(system.indices.len(), 2 * q - 1);
        assert_eq!(system.symbolic.factor_nonzeros(), 2 * q - 1);
        let weights: Vec<_> = (0..q - 1).map(|i| 0.5 + (i % 7) as f64 / 4.0).collect();
        let factor = system.factor(&weights, 0.0).unwrap();
        let matrix = Mat::from_fn(q, q, |i, j| match (i, j) {
            (0, 0) => 1.0 + weights.iter().sum::<f64>(),
            (0, j) => weights[j - 1],
            (i, 0) => weights[i - 1],
            (i, j) if i == j => 1.0 + weights[i - 1],
            _ => 0.0,
        });
        let dense = Llt::new(matrix.as_ref(), faer::Side::Lower).unwrap();
        let logdet: f64 = (0..q).map(|i| 2.0 * dense.L()[(i, i)].ln()).sum();
        assert!((factor.logdet() - logdet).abs() < 1e-11);
        for columns in [0, 1, 7] {
            let rhs = Mat::from_fn(q, columns, |i, j| ((i * 3 + j) % 17) as f64 / 5.0);
            let expected = dense.solve(&rhs);
            let expected_gram = rhs.transpose() * &expected;
            let mut actual = rhs.clone();
            factor.solve_lower_in_place(actual.as_mut());
            let gram = actual.transpose() * &actual;
            for j in 0..columns {
                for i in 0..columns {
                    assert!((gram[(i, j)] - expected_gram[(i, j)]).abs() < 1e-10);
                }
            }
            factor.solve_upper_in_place(actual.as_mut());
            for j in 0..columns {
                for i in 0..q {
                    assert!((actual[(i, j)] - expected[(i, j)]).abs() < 1e-11);
                }
            }
        }
    }
}

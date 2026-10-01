use std::sync::Arc;

#[cfg(test)]
use faer::Mat;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::simplicial::SimplicialLdltRef;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
    factorize_symbolic_cholesky,
};
use faer::sparse::linalg::{SupernodalThreshold, amd};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};
use numpy::ndarray::ArrayView2;

use crate::csc::CscMatrix;
use crate::linalg::LinalgError;

pub struct SymbolicCholeskyCache {
    symbolic: Arc<SymbolicCholesky<usize>>,
    input_indices: Vec<usize>,
    input_indptr: Vec<usize>,
    upper_indices: Vec<usize>,
    upper_indptr: Vec<usize>,
    upper_value_sources: Vec<(usize, usize)>,
    n: usize,
}

impl SymbolicCholeskyCache {
    pub fn new(indices: &[usize], indptr: &[usize], n: usize) -> Result<Self, LinalgError> {
        Self::with_ordering(indices, indptr, n, SymmetricOrdering::Identity)
    }

    pub fn new_amd(indices: &[usize], indptr: &[usize], n: usize) -> Result<Self, LinalgError> {
        Self::with_ordering(indices, indptr, n, SymmetricOrdering::Amd)
    }

    fn with_ordering(
        indices: &[usize],
        indptr: &[usize],
        n: usize,
        ordering: SymmetricOrdering<'_, usize>,
    ) -> Result<Self, LinalgError> {
        let pattern_values = vec![1.0; indices.len()];
        let mat = build_csc_matrix(&pattern_values, indices, indptr, n)?;
        let upper = mat.self_adjoint_upper_from_lower();
        // A simplicial factor exposes a stable CSC diagonal layout for logdet.
        // This is also the appropriate representation for the model matrices,
        // whose factors remain highly sparse after ordering.
        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        };
        // faer's AMD preprocessing reads the previous Cell value when filling
        // column pointers. Initialize its scratch before invoking it, then pass
        // the computed ordering to symbolic factorization.
        let mut permutation = Vec::new();
        let mut inverse = Vec::new();
        let ordering = if matches!(ordering, SymmetricOrdering::Amd) {
            permutation.resize(n, 0);
            inverse.resize(n, 0);
            let mut memory = MemBuffer::new(amd::order_maybe_unsorted_scratch::<usize>(
                n,
                upper.values().len(),
            ));
            for byte in memory.iter_mut() {
                byte.write(0);
            }
            amd::order_maybe_unsorted(
                &mut permutation,
                &mut inverse,
                matrix_ref(&upper).symbolic(),
                params.amd_params,
                MemStack::new(&mut memory),
            )
            .map_err(|error| LinalgError::InvalidSparseFormat(format!("{error:?}")))?;
            SymmetricOrdering::Custom(PermRef::new_checked(&permutation, &inverse, n))
        } else {
            ordering
        };
        let symbolic = factorize_symbolic_cholesky(
            matrix_ref(&upper).symbolic(),
            Side::Upper,
            ordering,
            params,
        )
        .map_err(|error| LinalgError::InvalidSparseFormat(format!("{error:?}")))?;
        let mut upper_value_sources = Vec::with_capacity(upper.values().len());
        for column in 0..n {
            for (position, &row) in indices
                .iter()
                .enumerate()
                .take(indptr[column + 1])
                .skip(indptr[column])
            {
                if row < column {
                    continue;
                }
                let upper_start = upper.col_offsets()[row];
                let upper_end = upper.col_offsets()[row + 1];
                let offset = upper.row_indices()[upper_start..upper_end]
                    .binary_search(&column)
                    .expect("canonical upper pattern contains every lower entry");
                upper_value_sources.push((position, upper_start + offset));
            }
        }
        Ok(Self {
            symbolic: Arc::new(symbolic),
            input_indices: indices.to_vec(),
            input_indptr: indptr.to_vec(),
            upper_indices: upper.row_indices().to_vec(),
            upper_indptr: upper.col_offsets().to_vec(),
            upper_value_sources,
            n,
        })
    }

    pub fn factor(
        &self,
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
    ) -> Result<NumericFactorization, LinalgError> {
        if indices != self.input_indices || indptr != self.input_indptr {
            return Err(LinalgError::InvalidSparseFormat(
                "numeric matrix pattern differs from symbolic factorization".to_string(),
            ));
        }
        if data.len() != self.input_indices.len() {
            return Err(LinalgError::InvalidSparseFormat(format!(
                "data has length {}, but indices has length {}",
                data.len(),
                self.input_indices.len()
            )));
        }
        let mut upper_values = vec![0.0; self.upper_indices.len()];
        for &(source, target) in &self.upper_value_sources {
            upper_values[target] += data[source];
        }

        // Sparse model factors in mixed models are typically too fine-grained
        // for per-factor Rayon scheduling to pay for itself. Outer model and
        // simulation loops retain their existing parallelism.
        let par = Par::Seq;
        let mut values = vec![0.0; self.symbolic.len_val()];
        let mut memory = MemBuffer::new(
            self.symbolic
                .factorize_numeric_ldlt_scratch::<f64>(par, Default::default()),
        );
        let stack = MemStack::new(&mut memory);
        self.symbolic
            .factorize_numeric_ldlt(
                &mut values,
                matrix_ref_from_parts(
                    self.n,
                    &self.upper_indptr,
                    &self.upper_indices,
                    &upper_values,
                ),
                Side::Upper,
                LdltRegularization::default(),
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| LinalgError::NotPositiveDefinite)?;
        let SymbolicCholeskyRaw::Simplicial(symbolic) = self.symbolic.raw() else {
            unreachable!("symbolic factorization is forced to be simplicial")
        };
        if symbolic
            .col_ptr()
            .iter()
            .take(self.n)
            .any(|&diagonal| values[diagonal] <= 0.0 || !values[diagonal].is_finite())
        {
            return Err(LinalgError::NotPositiveDefinite);
        }
        Ok(NumericFactorization {
            symbolic: Arc::clone(&self.symbolic),
            values,
            n: self.n,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn factor_nonzeros(&self) -> usize {
        self.symbolic.len_val()
    }
}

pub struct NumericFactorization {
    symbolic: Arc<SymbolicCholesky<usize>>,
    values: Vec<f64>,
    n: usize,
}

impl NumericFactorization {
    /// Whiten by D^(-1/2) L^(-1) P for P A P^T = L D L^T.
    pub fn solve_lower_in_place(&self, mut rhs: MatMut<'_, f64>) {
        let SymbolicCholeskyRaw::Simplicial(symbolic) = self.symbolic.raw() else {
            unreachable!("symbolic factorization is forced to be simplicial")
        };
        if let Some(permutation) = self.symbolic.perm() {
            let input = rhs.as_ref().to_owned();
            faer::perm::permute_rows(rhs.as_mut(), input.as_ref(), permutation);
        }
        let lower =
            matrix_ref_from_parts(self.n, symbolic.col_ptr(), symbolic.row_idx(), &self.values);
        faer::sparse::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            lower,
            Conj::No,
            rhs.as_mut(),
            Par::Seq,
        );
        for column in 0..rhs.ncols() {
            for row in 0..self.n {
                rhs[(row, column)] /= self.values[symbolic.col_ptr()[row]].sqrt();
            }
        }
    }

    /// Apply P^T L^(-T) D^(-1/2), returning to the original model order.
    pub fn solve_upper_in_place(&self, mut rhs: MatMut<'_, f64>) {
        let SymbolicCholeskyRaw::Simplicial(symbolic) = self.symbolic.raw() else {
            unreachable!("symbolic factorization is forced to be simplicial")
        };
        for column in 0..rhs.ncols() {
            for row in 0..self.n {
                rhs[(row, column)] /= self.values[symbolic.col_ptr()[row]].sqrt();
            }
        }
        let lower =
            matrix_ref_from_parts(self.n, symbolic.col_ptr(), symbolic.row_idx(), &self.values);
        faer::sparse::linalg::triangular_solve::solve_unit_lower_triangular_transpose_in_place(
            lower,
            Conj::No,
            rhs.as_mut(),
            Par::Seq,
        );
        if let Some(permutation) = self.symbolic.perm() {
            let input = rhs.as_ref().to_owned();
            faer::perm::permute_rows(rhs, input.as_ref(), permutation.inverse());
        }
    }

    pub fn solve(&self, b: ArrayView2<'_, f64>) -> Result<Vec<f64>, LinalgError> {
        if b.nrows() != self.n {
            return Err(LinalgError::DimensionMismatch(format!(
                "right-hand side has {} rows, expected {}",
                b.nrows(),
                self.n
            )));
        }
        // ndarray iteration follows logical row-major order even for strided inputs.
        // Solve all columns directly in the owned Python result buffer.
        let mut result: Vec<f64> = b.iter().copied().collect();
        let rhs = MatMut::from_row_major_slice_mut(&mut result, self.n, b.ncols());
        let SymbolicCholeskyRaw::Simplicial(symbolic) = self.symbolic.raw() else {
            unreachable!("symbolic factorization is forced to be simplicial")
        };
        debug_assert!(self.symbolic.perm().is_none());
        // Identity ordering needs no permutation buffer. The simplicial solve's
        // own scratch requirement is empty, independent of the number of columns.
        let par = Par::Seq;
        let factor = SimplicialLdltRef::new(symbolic, &self.values);
        let mut memory = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(b.ncols()));
        let stack = MemStack::new(&mut memory);
        factor.solve_in_place_with_conj(Conj::No, rhs, par, stack);
        Ok(result)
    }

    pub fn logdet(&self) -> f64 {
        let SymbolicCholeskyRaw::Simplicial(symbolic) = self.symbolic.raw() else {
            unreachable!("symbolic factorization is forced to be simplicial")
        };
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        (0..self.n)
            .map(|column| {
                let diagonal = col_ptr[column];
                debug_assert_eq!(row_idx[diagonal], column);
                self.values[diagonal].ln()
            })
            .sum::<f64>()
    }
}

fn build_csc_matrix(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    n: usize,
) -> Result<CscMatrix, LinalgError> {
    CscMatrix::try_from_usize(data, indices, indptr, (n, n))
}

fn matrix_ref(matrix: &CscMatrix) -> SparseColMatRef<'_, usize, f64> {
    matrix_ref_from_parts(
        matrix.nrows(),
        matrix.col_offsets(),
        matrix.row_indices(),
        matrix.values(),
    )
}

fn matrix_ref_from_parts<'a>(
    n: usize,
    col_offsets: &'a [usize],
    row_indices: &'a [usize],
    values: &'a [f64],
) -> SparseColMatRef<'a, usize, f64> {
    let symbolic = SymbolicSparseColMatRef::new_checked(n, n, col_offsets, None, row_indices);
    SparseColMatRef::new(symbolic, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array2, array};

    #[test]
    fn sparse_whitening_and_backsolve_match_dense_cholesky() {
        let data = vec![4.0, 1.0, -0.5, 3.0, 0.25, 2.0];
        let indices = vec![0, 1, 2, 1, 2, 2];
        let offsets = vec![0, 3, 5, 6];
        let symbolic = SymbolicCholeskyCache::new(&indices, &offsets, 3).unwrap();
        let factor = symbolic.factor(&data, &indices, &offsets).unwrap();
        let matrix = Mat::from_fn(3, 3, |i, j| {
            [[4.0, 1.0, -0.5], [1.0, 3.0, 0.25], [-0.5, 0.25, 2.0]][i][j]
        });
        let dense = faer::linalg::solvers::Llt::new(matrix.as_ref(), Side::Lower).unwrap();
        for columns in [0, 1, 7] {
            let rhs = Mat::from_fn(3, columns, |i, j| (i + 2 * j) as f64 - 4.0);
            let mut actual = rhs.clone();
            factor.solve_lower_in_place(actual.as_mut());
            let whitened = dense.L() * &actual;
            for j in 0..columns {
                for i in 0..3 {
                    assert!((whitened[(i, j)] - rhs[(i, j)]).abs() < 1e-12);
                }
            }
            factor.solve_upper_in_place(actual.as_mut());
            let reconstructed = &matrix * &actual;
            for j in 0..columns {
                for i in 0..3 {
                    assert!((reconstructed[(i, j)] - rhs[(i, j)]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_symbolic_cache_basic() {
        let n = 3;
        let data = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];

        let cache = SymbolicCholeskyCache::new(&indices, &indptr, n).unwrap();
        let numeric = cache.factor(&data, &indices, &indptr).unwrap();

        let b = array![[1.0], [2.0], [3.0]];
        let x = numeric.solve(b.view()).unwrap();

        let reconstructed = [
            4.0 * x[0] + x[1],
            x[0] + 4.0 * x[1] + x[2],
            x[1] + 4.0 * x[2],
        ];
        for (actual, expected) in reconstructed.into_iter().zip(b.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_multiple_strided_rhs_and_empty_columns() {
        let indices = [0, 1, 0, 1, 2, 1, 2];
        let indptr = [0, 2, 5, 7];
        let data = [4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let cache = SymbolicCholeskyCache::new(&indices, &indptr, 3).unwrap();
        let numeric = cache.factor(&data, &indices, &indptr).unwrap();
        let transposed_rhs =
            Array2::from_shape_fn((5, 3), |(column, row)| (1 + column + 2 * row) as f64);
        let rhs = transposed_rhs.t();
        let result = numeric.solve(rhs).unwrap();
        for column in 0..5 {
            let x = [result[column], result[5 + column], result[10 + column]];
            let reconstructed = [
                4.0 * x[0] + x[1],
                x[0] + 4.0 * x[1] + x[2],
                x[1] + 4.0 * x[2],
            ];
            for row in 0..3 {
                assert!((reconstructed[row] - rhs[(row, column)]).abs() < 1e-12);
            }
        }
        assert!(
            numeric
                .solve(Array2::zeros((3, 0)).view())
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            numeric.solve(Array2::zeros((4, 2)).view()),
            Err(LinalgError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn test_symbolic_cache_reuse() {
        let n = 3;
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];

        let cache = SymbolicCholeskyCache::new(&indices, &indptr, n).unwrap();

        let data1 = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let numeric1 = cache.factor(&data1, &indices, &indptr).unwrap();
        let logdet1 = numeric1.logdet();

        let data2 = vec![5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0];
        let numeric2 = cache.factor(&data2, &indices, &indptr).unwrap();
        let logdet2 = numeric2.logdet();

        assert!((logdet1 - logdet2).abs() > 0.01);
    }
}

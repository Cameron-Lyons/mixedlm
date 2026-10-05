use std::sync::Arc;

#[cfg(test)]
use faer::Mat;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::simplicial::{SimplicialLdltRef, SymbolicSimplicialCholesky};
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
    factorize_symbolic_cholesky,
};
use faer::sparse::linalg::{SupernodalThreshold, amd};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, MatRef, Par, Side};

use crate::csc::CscMatrix;
use crate::linalg::LinalgError;

/// Fill-reducing ordering applied before the symbolic factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillOrdering {
    /// Factor in the given order, which suits banded and block systems.
    Natural,
    /// Approximate minimum degree, which keeps hubs and crossed factors sparse.
    Amd,
}

struct Permutation {
    forward: Vec<usize>,
    inverse: Vec<usize>,
}

impl Permutation {
    fn perm(&self) -> PermRef<'_, usize> {
        PermRef::new_checked(&self.forward, &self.inverse, self.forward.len())
    }
}

pub struct SymbolicCholeskyCache {
    symbolic: Arc<SymbolicCholesky<usize>>,
    permutation: Option<Arc<Permutation>>,
    input_nonzeros: usize,
    upper_indices: Vec<usize>,
    upper_indptr: Vec<usize>,
    upper_value_sources: Vec<(usize, usize)>,
    n: usize,
}

impl SymbolicCholeskyCache {
    pub fn new(
        indices: &[usize],
        indptr: &[usize],
        n: usize,
        ordering: FillOrdering,
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
        // column pointers. Initialize its scratch before invoking it and retain
        // the computed ordering with the cache.
        let permutation = if ordering == FillOrdering::Amd {
            let mut forward = vec![0; n];
            let mut inverse = vec![0; n];
            let mut memory = MemBuffer::new(amd::order_maybe_unsorted_scratch::<usize>(
                n,
                upper.values().len(),
            ));
            for byte in memory.iter_mut() {
                byte.write(0);
            }
            amd::order_maybe_unsorted(
                &mut forward,
                &mut inverse,
                matrix_ref(&upper).symbolic(),
                params.amd_params,
                MemStack::new(&mut memory),
            )
            .map_err(|error| LinalgError::InvalidSparseFormat(format!("{error:?}")))?;
            Some(Arc::new(Permutation { forward, inverse }))
        } else {
            None
        };
        // Permute initialized owned storage ourselves: faer's internal custom
        // ordering path also fills uninitialized column pointers via Cell::set.
        let upper = if let Some(permutation) = &permutation {
            let mut columns = vec![Vec::new(); n];
            for column in 0..n {
                for &row in &upper.row_indices()
                    [upper.col_offsets()[column]..upper.col_offsets()[column + 1]]
                {
                    let i = permutation.inverse[row];
                    let j = permutation.inverse[column];
                    columns[i.max(j)].push(i.min(j));
                }
            }
            let mut pointers = vec![0];
            let mut rows = Vec::with_capacity(upper.values().len());
            for column in &mut columns {
                column.sort_unstable();
                rows.extend_from_slice(column);
                pointers.push(rows.len());
            }
            build_csc_matrix(&vec![1.0; rows.len()], &rows, &pointers, n)?
        } else {
            upper
        };
        let symbolic = factorize_symbolic_cholesky(
            matrix_ref(&upper).symbolic(),
            Side::Upper,
            SymmetricOrdering::Identity,
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
                let (i, j) = match &permutation {
                    Some(permutation) => (permutation.inverse[row], permutation.inverse[column]),
                    None => (row, column),
                };
                let upper_column = i.max(j);
                let upper_start = upper.col_offsets()[upper_column];
                let upper_end = upper.col_offsets()[upper_column + 1];
                let offset = upper.row_indices()[upper_start..upper_end]
                    .binary_search(&i.min(j))
                    .expect("canonical upper pattern contains every lower entry");
                upper_value_sources.push((position, upper_start + offset));
            }
        }
        Ok(Self {
            symbolic: Arc::new(symbolic),
            permutation,
            input_nonzeros: indices.len(),
            upper_indices: upper.row_indices().to_vec(),
            upper_indptr: upper.col_offsets().to_vec(),
            upper_value_sources,
            n,
        })
    }

    /// Factor values stored in the analyzed input pattern.
    pub fn factor(&self, data: &[f64]) -> Result<NumericFactorization, LinalgError> {
        if data.len() != self.input_nonzeros {
            return Err(LinalgError::InvalidSparseFormat(format!(
                "data has length {}, but indices has length {}",
                data.len(),
                self.input_nonzeros
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
        let mut sqrt_diagonal = Vec::with_capacity(self.n);
        for &diagonal in &symbolic.col_ptr()[..self.n] {
            let value = values[diagonal];
            if value <= 0.0 || !value.is_finite() {
                return Err(LinalgError::NotPositiveDefinite);
            }
            sqrt_diagonal.push(value.sqrt());
        }
        Ok(NumericFactorization {
            symbolic: Arc::clone(&self.symbolic),
            permutation: self.permutation.clone(),
            values,
            sqrt_diagonal,
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
    permutation: Option<Arc<Permutation>>,
    values: Vec<f64>,
    // sqrt(D), computed once per factorization for the whitening solves.
    sqrt_diagonal: Vec<f64>,
    n: usize,
}

impl NumericFactorization {
    fn simplicial(&self) -> &SymbolicSimplicialCholesky<usize> {
        let SymbolicCholeskyRaw::Simplicial(symbolic) = self.symbolic.raw() else {
            unreachable!("symbolic factorization is forced to be simplicial")
        };
        symbolic
    }

    fn unit_lower(&self) -> SparseColMatRef<'_, usize, f64> {
        let symbolic = self.simplicial();
        matrix_ref_from_parts(self.n, symbolic.col_ptr(), symbolic.row_idx(), &self.values)
    }

    /// Divide rows by sqrt(D), contiguously for the column-major model solves.
    fn divide_by_sqrt_diagonal(&self, mut rhs: MatMut<'_, f64>) {
        assert_eq!(rhs.nrows(), self.n);
        for mut column in rhs.as_mut().col_iter_mut() {
            for (value, &scale) in column.as_mut().iter_mut().zip(&self.sqrt_diagonal) {
                *value /= scale;
            }
        }
    }

    /// Whiten by D^(-1/2) L^(-1) P for P A P^T = L D L^T.
    pub fn solve_lower_in_place(&self, mut rhs: MatMut<'_, f64>) {
        if let Some(permutation) = &self.permutation {
            let input = rhs.as_ref().to_owned();
            faer::perm::permute_rows(rhs.as_mut(), input.as_ref(), permutation.perm());
        }
        faer::sparse::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            self.unit_lower(),
            Conj::No,
            rhs.as_mut(),
            Par::Seq,
        );
        self.divide_by_sqrt_diagonal(rhs);
    }

    /// Apply P^T L^(-T) D^(-1/2), returning to the original model order.
    pub fn solve_upper_in_place(&self, mut rhs: MatMut<'_, f64>) {
        self.divide_by_sqrt_diagonal(rhs.as_mut());
        faer::sparse::linalg::triangular_solve::solve_unit_lower_triangular_transpose_in_place(
            self.unit_lower(),
            Conj::No,
            rhs.as_mut(),
            Par::Seq,
        );
        if let Some(permutation) = &self.permutation {
            let input = rhs.as_ref().to_owned();
            faer::perm::permute_rows(rhs, input.as_ref(), permutation.perm().inverse());
        }
    }

    /// Solve in a caller-owned row-major buffer, which is returned as the result.
    pub fn solve_owned(
        &self,
        mut result: Vec<f64>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, LinalgError> {
        if shape.0 != self.n {
            return Err(LinalgError::DimensionMismatch(format!(
                "right-hand side has {} rows, expected {}",
                shape.0, self.n
            )));
        }
        if shape.0.checked_mul(shape.1) != Some(result.len()) {
            return Err(LinalgError::DimensionMismatch(
                "right-hand side data does not match its shape".to_string(),
            ));
        }
        let symbolic = self.simplicial();
        let factor = SimplicialLdltRef::new(symbolic, &self.values);
        // The simplicial solve's own scratch requirement is empty, independent
        // of the number of columns.
        let mut memory = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(shape.1));
        let stack = MemStack::new(&mut memory);
        let Some(permutation) = &self.permutation else {
            let rhs = MatMut::from_row_major_slice_mut(&mut result, self.n, shape.1);
            factor.solve_in_place_with_conj(Conj::No, rhs, Par::Seq, stack);
            return Ok(result);
        };
        // A fill-reducing order costs one permuted copy in each direction;
        // the solve itself is the same fused LDL^T solve as natural order.
        let mut permuted = vec![0.0; result.len()];
        let mut work = MatMut::from_row_major_slice_mut(&mut permuted, self.n, shape.1);
        faer::perm::permute_rows(
            work.as_mut(),
            MatRef::from_row_major_slice(&result, self.n, shape.1),
            permutation.perm(),
        );
        factor.solve_in_place_with_conj(Conj::No, work.as_mut(), Par::Seq, stack);
        faer::perm::permute_rows(
            MatMut::from_row_major_slice_mut(&mut result, self.n, shape.1),
            work.as_ref(),
            permutation.perm().inverse(),
        );
        Ok(result)
    }

    pub fn logdet(&self) -> f64 {
        let symbolic = self.simplicial();
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
    fn owned_solves_reuse_the_result_buffer_and_validate_shapes() {
        let indices = [0, 1, 2, 1, 2, 2];
        let offsets = [0, 3, 5, 6];
        let data = [4.0, 1.0, -0.5, 3.0, 0.25, 2.0];
        for cache in [
            SymbolicCholeskyCache::new(&indices, &offsets, 3, FillOrdering::Natural).unwrap(),
            SymbolicCholeskyCache::new(&indices, &offsets, 3, FillOrdering::Amd).unwrap(),
        ] {
            let factor = cache.factor(&data).unwrap();
            let rhs = vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
            let pointer = rhs.as_ptr();
            let result = factor.solve_owned(rhs, (3, 2)).unwrap();
            assert_eq!(result.as_ptr(), pointer);
            assert!(matches!(
                factor.solve_owned(vec![0.0; 5], (3, 2)),
                Err(LinalgError::DimensionMismatch(_))
            ));
            assert!(matches!(
                factor.solve_owned(vec![], (3, usize::MAX)),
                Err(LinalgError::DimensionMismatch(_))
            ));
        }
        let empty = SymbolicCholeskyCache::new(&[], &[0], 0, FillOrdering::Amd).unwrap();
        let factor = empty.factor(&[]).unwrap();
        assert!(factor.solve_owned(vec![], (0, 5)).unwrap().is_empty());
    }

    #[test]
    fn sparse_whitening_and_backsolve_match_dense_cholesky() {
        let data = vec![4.0, 1.0, -0.5, 3.0, 0.25, 2.0];
        let indices = vec![0, 1, 2, 1, 2, 2];
        let offsets = vec![0, 3, 5, 6];
        let symbolic =
            SymbolicCholeskyCache::new(&indices, &offsets, 3, FillOrdering::Natural).unwrap();
        let factor = symbolic.factor(&data).unwrap();
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
    fn amd_factorization_preserves_rhs_order() {
        let indices = [0, 1, 2, 3, 1, 2, 3];
        let offsets = [0, 4, 5, 6, 7];
        let symbolic =
            SymbolicCholeskyCache::new(&indices, &offsets, 4, FillOrdering::Amd).unwrap();
        let rhs = array![[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0], [7.0, 8.0]];
        for diagonal in [5.0, 7.0] {
            let data = [diagonal, 1.0, 1.0, 1.0, diagonal, diagonal, diagonal];
            let factor = symbolic.factor(&data).unwrap();
            let solution = factor
                .solve_owned(rhs.iter().copied().collect(), rhs.dim())
                .unwrap();
            for column in 0..2 {
                for row in 0..4 {
                    let cross = if row == 0 {
                        (1..4).map(|i| solution[i * 2 + column]).sum::<f64>()
                    } else {
                        solution[column]
                    };
                    assert!(
                        (diagonal * solution[row * 2 + column] + cross - rhs[(row, column)]).abs()
                            < 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn fill_reducing_order_preserves_solutions_and_whitened_products() {
        // A hub-first arrowhead fills completely in natural order; AMD
        // eliminates the leaves first and keeps the factor linear.
        let n = 40;
        let coupling = |row: usize| 0.1 * (row as f64).cos();
        let diagonal = |row: usize| 2.0 + (row % 3) as f64;
        let mut indices: Vec<_> = (0..n).collect();
        let mut data: Vec<_> = (0..n)
            .map(|row| if row == 0 { n as f64 } else { coupling(row) })
            .collect();
        let mut offsets = vec![0, n];
        for column in 1..n {
            indices.push(column);
            data.push(diagonal(column));
            offsets.push(indices.len());
        }
        let natural =
            SymbolicCholeskyCache::new(&indices, &offsets, n, FillOrdering::Natural).unwrap();
        let amd = SymbolicCholeskyCache::new(&indices, &offsets, n, FillOrdering::Amd).unwrap();
        assert_eq!(natural.factor_nonzeros(), n * (n + 1) / 2);
        assert_eq!(amd.factor_nonzeros(), 2 * n - 1);
        let natural = natural.factor(&data).unwrap();
        let amd = amd.factor(&data).unwrap();
        assert!((natural.logdet() - amd.logdet()).abs() < 1e-12);
        for columns in [1, 3, 128] {
            let rhs = Mat::from_fn(n, columns, |i, j| ((7 * i + 3 * j) % 23) as f64 / 5.0 - 2.0);
            let row_major: Vec<_> = (0..n)
                .flat_map(|i| (0..columns).map(move |j| (i, j)))
                .map(|(i, j)| rhs[(i, j)])
                .collect();
            let expected = natural
                .solve_owned(row_major.clone(), (n, columns))
                .unwrap();
            let actual = amd.solve_owned(row_major, (n, columns)).unwrap();
            for j in 0..columns {
                let x = |i: usize| actual[i * columns + j];
                let hub = n as f64 * x(0) + (1..n).map(|i| coupling(i) * x(i)).sum::<f64>();
                assert!((hub - rhs[(0, j)]).abs() < 1e-12);
                for i in 1..n {
                    let product = coupling(i) * x(0) + diagonal(i) * x(i);
                    assert!((product - rhs[(i, j)]).abs() < 1e-12);
                    assert!((x(i) - expected[i * columns + j]).abs() < 1e-12);
                }
            }
            // Whitened vectors from different orders differ by an orthogonal
            // factor, so compare their products and the completed solve.
            let mut whitened_natural = rhs.clone();
            natural.solve_lower_in_place(whitened_natural.as_mut());
            let mut whitened_amd = rhs.clone();
            amd.solve_lower_in_place(whitened_amd.as_mut());
            let gram = whitened_amd.transpose() * &whitened_amd;
            let expected_gram = whitened_natural.transpose() * &whitened_natural;
            assert!((&gram - &expected_gram).norm_max() < 1e-12);
            amd.solve_upper_in_place(whitened_amd.as_mut());
            for j in 0..columns {
                for i in 0..n {
                    assert!((whitened_amd[(i, j)] - expected[i * columns + j]).abs() < 1e-12);
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

        let cache =
            SymbolicCholeskyCache::new(&indices, &indptr, n, FillOrdering::Natural).unwrap();
        let numeric = cache.factor(&data).unwrap();

        let b = array![[1.0], [2.0], [3.0]];
        let x = numeric
            .solve_owned(b.iter().copied().collect(), b.dim())
            .unwrap();

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
        let cache =
            SymbolicCholeskyCache::new(&indices, &indptr, 3, FillOrdering::Natural).unwrap();
        let numeric = cache.factor(&data).unwrap();
        let transposed_rhs =
            Array2::from_shape_fn((5, 3), |(column, row)| (1 + column + 2 * row) as f64);
        let rhs = transposed_rhs.t();
        let result = numeric
            .solve_owned(rhs.iter().copied().collect(), rhs.dim())
            .unwrap();
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
        assert!(numeric.solve_owned(vec![], (3, 0)).unwrap().is_empty());
        assert!(matches!(
            numeric.solve_owned(vec![0.0; 8], (4, 2)),
            Err(LinalgError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn test_symbolic_cache_reuse() {
        let n = 3;
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];

        let cache =
            SymbolicCholeskyCache::new(&indices, &indptr, n, FillOrdering::Natural).unwrap();

        let data1 = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let numeric1 = cache.factor(&data1).unwrap();
        let logdet1 = numeric1.logdet();

        let data2 = vec![5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0];
        let numeric2 = cache.factor(&data2).unwrap();
        let logdet2 = numeric2.logdet();

        assert!((logdet1 - logdet2).abs() > 0.01);
    }
}

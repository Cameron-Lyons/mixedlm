use std::sync::Arc;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
    factorize_symbolic_cholesky,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, Par, Side};

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
        let symbolic = factorize_symbolic_cholesky(
            matrix_ref(&upper).symbolic(),
            Side::Upper,
            // Match the previous sparse backends' natural ordering. Model
            // matrices are assembled in a structure-aware order already, and
            // avoiding a fresh AMD permutation makes repeated refactors cheap.
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
}

pub struct NumericFactorization {
    symbolic: Arc<SymbolicCholesky<usize>>,
    values: Vec<f64>,
    n: usize,
}

impl NumericFactorization {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, LinalgError> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionMismatch(format!(
                "right-hand side has {} rows, expected {}",
                b.len(),
                self.n
            )));
        }
        let mut rhs = Mat::from_fn(self.n, 1, |row, _| b[row]);
        let par = Par::Seq;
        let factor = LdltRef::new(&self.symbolic, &self.values);
        let mut memory = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, par));
        let stack = MemStack::new(&mut memory);
        factor.solve_in_place_with_conj(Conj::No, rhs.as_mut(), par, stack);
        Ok((0..self.n).map(|row| rhs[(row, 0)]).collect())
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

    #[test]
    fn test_symbolic_cache_basic() {
        let n = 3;
        let data = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];

        let cache = SymbolicCholeskyCache::new(&indices, &indptr, n).unwrap();
        let numeric = cache.factor(&data, &indices, &indptr).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let x = numeric.solve(&b).unwrap();

        let reconstructed = [
            4.0 * x[0] + x[1],
            x[0] + 4.0 * x[1] + x[2],
            x[1] + 4.0 * x[2],
        ];
        for (actual, expected) in reconstructed.into_iter().zip(b) {
            assert!((actual - expected).abs() < 1e-12);
        }
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

use sprs::{CsMat, TriMat};
use sprs_ldl::{LdlNumeric, LdlSymbolic};

use crate::linalg::LinalgError;

pub struct SymbolicCholeskyCache {
    symbolic: LdlSymbolic<usize>,
    n: usize,
}

impl SymbolicCholeskyCache {
    pub fn new(indices: &[usize], indptr: &[usize], n: usize) -> Result<Self, LinalgError> {
        let mat = build_sprs_pattern(indices, indptr, n)?;
        let symbolic = LdlSymbolic::new(mat.view());
        Ok(Self { symbolic, n })
    }

    pub fn factor(
        &self,
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
    ) -> Result<NumericFactorization, LinalgError> {
        let mat = build_sprs_matrix(data, indices, indptr, self.n)?;
        let numeric = self
            .symbolic
            .clone()
            .factor(mat.view())
            .map_err(|_| LinalgError::NotPositiveDefinite)?;
        Ok(NumericFactorization { numeric })
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

pub struct NumericFactorization {
    numeric: LdlNumeric<f64, usize>,
}

impl NumericFactorization {
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        self.numeric.solve(b).to_vec()
    }

    pub fn logdet(&self) -> f64 {
        let diag = self.numeric.d();
        diag.iter().map(|&d| d.ln()).sum::<f64>()
    }
}

fn build_sprs_pattern(
    indices: &[usize],
    indptr: &[usize],
    n: usize,
) -> Result<CsMat<f64>, LinalgError> {
    let mut triplets = TriMat::new((n, n));
    for col in 0..(indptr.len() - 1) {
        let start = indptr[col];
        let end = indptr[col + 1];
        for &row in &indices[start..end] {
            triplets.add_triplet(row, col, 1.0);
        }
    }
    Ok(triplets.to_csc())
}

fn build_sprs_matrix(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    n: usize,
) -> Result<CsMat<f64>, LinalgError> {
    CsMat::try_new_csc((n, n), indptr.to_vec(), indices.to_vec(), data.to_vec())
        .map_err(|e| LinalgError::InvalidSparseFormat(format!("{:?}", e)))
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
        let x = numeric.solve(&b);

        assert!(x.len() == 3);
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

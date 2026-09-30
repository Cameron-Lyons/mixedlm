use faer::{Col, Mat, MatRef};

#[derive(Debug, Clone, Copy)]
pub struct RandomEffectStructure {
    pub n_levels: usize,
    pub n_terms: usize,
    pub correlated: bool,
}

pub fn build_lambda_blocks(theta: &[f64], structures: &[RandomEffectStructure]) -> Vec<Mat<f64>> {
    let mut blocks = Vec::new();
    let mut theta_idx = 0;

    for structure in structures {
        let q = structure.n_terms;

        let l_block = if structure.correlated {
            let n_theta = q * (q + 1) / 2;
            let theta_block = &theta[theta_idx..theta_idx + n_theta];
            theta_idx += n_theta;

            let mut l = Mat::zeros(q, q);
            let mut idx = 0;
            for i in 0..q {
                for j in 0..=i {
                    l[(i, j)] = theta_block[idx];
                    idx += 1;
                }
            }
            l
        } else {
            let theta_block = &theta[theta_idx..theta_idx + q];
            theta_idx += q;

            let mut l = Mat::zeros(q, q);
            for i in 0..q {
                l[(i, i)] = theta_block[i];
            }
            l
        };

        blocks.push(l_block);
    }

    blocks
}

struct FactorBlock {
    offset: usize,
    n_levels: usize,
    diagonal: bool,
    lower: Mat<f64>,
}

/// One covariance factor per structure, repeated over its grouping levels.
/// Multiplication preserves every cross-structure and cross-level contribution.
pub struct CovarianceFactor {
    blocks: Vec<FactorBlock>,
    dimension: usize,
}

impl CovarianceFactor {
    pub fn new(theta: &[f64], structures: &[RandomEffectStructure]) -> Self {
        let dimension = structures.iter().map(|s| s.n_levels * s.n_terms).sum();
        if dimension == 0 {
            return Self {
                blocks: Vec::new(),
                dimension,
            };
        }
        let mut offset = 0;
        let blocks = build_lambda_blocks(theta, structures)
            .into_iter()
            .zip(structures)
            .map(|(lower, structure)| {
                let block = FactorBlock {
                    offset,
                    n_levels: structure.n_levels,
                    diagonal: !structure.correlated || structure.n_terms == 1,
                    lower,
                };
                offset += structure.n_levels * structure.n_terms;
                block
            })
            .collect();
        Self { blocks, dimension }
    }

    pub fn apply(&self, vector: &Col<f64>) -> Col<f64> {
        self.apply_vector::<false>(vector)
    }

    pub fn transpose_apply_vector(&self, vector: &Col<f64>) -> Col<f64> {
        self.apply_vector::<true>(vector)
    }

    fn apply_vector<const TRANSPOSE: bool>(&self, vector: &Col<f64>) -> Col<f64> {
        assert_eq!(vector.nrows(), self.dimension);
        let mut result = Col::zeros(self.dimension);
        for block in &self.blocks {
            let width = block.lower.nrows();
            for level in 0..block.n_levels {
                let offset = block.offset + level * width;
                for i in 0..width {
                    let indices = if block.diagonal {
                        i..i + 1
                    } else if TRANSPOSE {
                        i..width
                    } else {
                        0..i + 1
                    };
                    let mut value = 0.0;
                    for k in indices {
                        let coefficient = if TRANSPOSE {
                            block.lower[(k, i)]
                        } else {
                            block.lower[(i, k)]
                        };
                        value += coefficient * vector[offset + k];
                    }
                    result[offset + i] = value;
                }
            }
        }
        result
    }

    pub fn transpose_apply(&self, matrix: MatRef<'_, f64>) -> Mat<f64> {
        assert_eq!(matrix.nrows(), self.dimension);
        let mut result = matrix.to_owned();
        self.transpose_apply_in_place(&mut result);
        result
    }

    fn transpose_apply_in_place(&self, matrix: &mut Mat<f64>) {
        for block in &self.blocks {
            let width = block.lower.nrows();
            if !block.diagonal && width >= 16 {
                // Wide correlated blocks benefit from the optimized matrix
                // kernel. Temporary storage is limited to one block's rows.
                for level in 0..block.n_levels {
                    let offset = block.offset + level * width;
                    let transformed = block.lower.transpose() * matrix.subrows(offset, width);
                    matrix
                        .subrows_mut(offset, width)
                        .copy_from(transformed.as_ref());
                }
                continue;
            }
            for column in 0..matrix.ncols() {
                for level in 0..block.n_levels {
                    let offset = block.offset + level * width;
                    // Lambda' is upper triangular: ascending rows only read
                    // entries that have not yet been overwritten.
                    for i in 0..width {
                        let end = if block.diagonal { i + 1 } else { width };
                        let mut value = 0.0;
                        for k in i..end {
                            value += block.lower[(k, i)] * matrix[(offset + k, column)];
                        }
                        matrix[(offset + i, column)] = value;
                    }
                }
            }
        }
    }

    /// Transform the owned crossproduct in place to Lambda' * S * Lambda + I.
    pub fn penalized_crossproduct(&self, mut matrix: Mat<f64>) -> Mat<f64> {
        assert_eq!(matrix.nrows(), self.dimension);
        assert_eq!(matrix.ncols(), self.dimension);
        self.transpose_apply_in_place(&mut matrix);
        for block in &self.blocks {
            let width = block.lower.nrows();
            for level in 0..block.n_levels {
                let offset = block.offset + level * width;
                if !block.diagonal && width >= 16 {
                    let transformed = matrix.subcols(offset, width) * &block.lower;
                    matrix
                        .subcols_mut(offset, width)
                        .copy_from(transformed.as_ref());
                    continue;
                }
                // Right multiplication by a lower triangular block allows
                // ascending columns for the same reason as the left transform.
                for j in 0..width {
                    let end = if block.diagonal { j + 1 } else { width };
                    for row in 0..self.dimension {
                        let mut value = 0.0;
                        for k in j..end {
                            value += matrix[(row, offset + k)] * block.lower[(k, j)];
                        }
                        matrix[(row, offset + j)] = value;
                    }
                }
            }
        }
        for i in 0..self.dimension {
            matrix[(i, i)] += 1.0;
        }
        matrix
    }

    /// Retain the existing rank-revealing solve for explicit random-effect starts.
    pub fn to_dense(&self) -> Mat<f64> {
        let mut result = Mat::zeros(self.dimension, self.dimension);
        for block in &self.blocks {
            let width = block.lower.nrows();
            for level in 0..block.n_levels {
                let offset = block.offset + level * width;
                for i in 0..width {
                    for j in 0..=i {
                        result[(offset + i, offset + j)] = block.lower[(i, j)];
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<RandomEffectStructure>, Vec<f64>, Mat<f64>) {
        let structures = vec![
            RandomEffectStructure {
                n_levels: 0,
                n_terms: 1,
                correlated: true,
            },
            RandomEffectStructure {
                n_levels: 2,
                n_terms: 3,
                correlated: true,
            },
            RandomEffectStructure {
                n_levels: 3,
                n_terms: 2,
                correlated: false,
            },
            RandomEffectStructure {
                n_levels: 2,
                n_terms: 1,
                correlated: true,
            },
        ];
        let theta = vec![9.0, 0.8, 0.2, 0.6, -0.3, 0.1, 0.4, 0.5, 0.0, 1.2];
        let dense = Mat::from_fn(14, 14, |i, j| {
            if i < 6 && j < 6 && i / 3 == j / 3 {
                [[0.8, 0.0, 0.0], [0.2, 0.6, 0.0], [-0.3, 0.1, 0.4]][i % 3][j % 3]
            } else if i == j && (6..12).contains(&i) {
                [0.5, 0.0][(i - 6) % 2]
            } else if i == j && i >= 12 {
                1.2
            } else {
                0.0
            }
        });
        (structures, theta, dense)
    }

    fn assert_close(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>) {
        assert_eq!(actual.shape(), expected.shape());
        for j in 0..actual.ncols() {
            for i in 0..actual.nrows() {
                assert!((actual[(i, j)] - expected[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn mixed_blocks_match_dense_vector_and_matrix_products() {
        let (structures, theta, dense) = fixture();
        let factor = CovarianceFactor::new(&theta, &structures);
        assert_eq!(factor.to_dense(), dense);
        let vector = Col::from_fn(14, |i| i as f64 - 5.0);
        assert_close(factor.apply(&vector).as_mat(), (&dense * &vector).as_mat());
        assert_close(
            factor.transpose_apply_vector(&vector).as_mat(),
            (dense.transpose() * &vector).as_mat(),
        );
        for ncols in [0, 1, 4] {
            let matrix = Mat::from_fn(14, ncols, |i, j| (i + 2 * j) as f64 - 8.0);
            assert_close(
                factor.transpose_apply(matrix.as_ref()).as_ref(),
                (dense.transpose() * &matrix).as_ref(),
            );
        }
    }

    #[test]
    fn in_place_transforms_preserve_cross_level_and_cross_structure_entries() {
        let (structures, theta, dense) = fixture();
        // Deliberately nonsymmetric to catch transposition and overwrite errors.
        let matrix = Mat::from_fn(14, 14, |i, j| ((i + 2 * j) as f64).sin());
        let expected = dense.transpose() * &matrix * &dense + Mat::<f64>::identity(14, 14);
        let actual = CovarianceFactor::new(&theta, &structures).penalized_crossproduct(matrix);
        assert_close(actual.as_ref(), expected.as_ref());
    }

    #[test]
    fn zero_variance_produces_an_identity_penalty() {
        let (structures, mut theta, _) = fixture();
        theta.fill(0.0);
        let factor = CovarianceFactor::new(&theta, &structures);
        let matrix = Mat::from_fn(14, 14, |i, j| (i + j + 1) as f64);
        assert_eq!(
            factor.penalized_crossproduct(matrix),
            Mat::<f64>::identity(14, 14)
        );
        assert_eq!(factor.apply(&Col::full(14, 3.0)), Col::<f64>::zeros(14));
    }

    #[test]
    fn wide_correlated_blocks_match_dense_products() {
        let structures = [
            RandomEffectStructure {
                n_levels: 2,
                n_terms: 20,
                correlated: true,
            },
            RandomEffectStructure {
                n_levels: 2,
                n_terms: 3,
                correlated: false,
            },
        ];
        let coefficient = |i, j| {
            if i == j {
                0.3
            } else if i > j {
                0.01 * (i - j) as f64
            } else {
                0.0
            }
        };
        let mut theta = Vec::new();
        for i in 0..20 {
            for j in 0..=i {
                theta.push(coefficient(i, j));
            }
        }
        theta.extend([0.2, 0.4, 0.6]);
        let factor = CovarianceFactor::new(&theta, &structures);
        let dense = Mat::from_fn(46, 46, |i, j| {
            if i < 40 && j < 40 && i / 20 == j / 20 {
                coefficient(i % 20, j % 20)
            } else if i == j && i >= 40 {
                [0.2, 0.4, 0.6][(i - 40) % 3]
            } else {
                0.0
            }
        });
        assert_eq!(factor.to_dense(), dense);
        let matrix = Mat::from_fn(46, 7, |i, j| ((i + 3 * j) as f64).cos());
        assert_close(
            factor.transpose_apply(matrix.as_ref()).as_ref(),
            (dense.transpose() * &matrix).as_ref(),
        );
        let crossproduct = &matrix * matrix.transpose();
        let expected = dense.transpose() * &crossproduct * &dense + Mat::<f64>::identity(46, 46);
        assert_close(
            factor.penalized_crossproduct(crossproduct).as_ref(),
            expected.as_ref(),
        );
    }

    #[test]
    fn empty_factors_preserve_empty_shapes() {
        for structures in [
            vec![],
            vec![RandomEffectStructure {
                n_levels: 0,
                n_terms: 3,
                correlated: true,
            }],
        ] {
            let factor = CovarianceFactor::new(&[], &structures);
            assert_eq!(factor.to_dense(), Mat::<f64>::zeros(0, 0));
            assert_eq!(factor.apply(&Col::zeros(0)), Col::<f64>::zeros(0));
            assert_eq!(
                factor.transpose_apply(Mat::zeros(0, 4).as_ref()),
                Mat::<f64>::zeros(0, 4)
            );
            assert_eq!(
                factor.penalized_crossproduct(Mat::zeros(0, 0)),
                Mat::<f64>::zeros(0, 0)
            );
        }
    }
}

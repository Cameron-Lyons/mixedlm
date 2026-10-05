use faer::{Col, Mat, MatMut, MatRef};

use crate::csc::CscMatrix;

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

/// Apply a repeated transposed factor to a view's rows without replacing its buffer.
pub(crate) fn transpose_apply_repeated_in_place(
    lower: &Mat<f64>,
    n_levels: usize,
    diagonal: bool,
    mut matrix: MatMut<'_, f64>,
) {
    let width = lower.nrows();
    assert_eq!(lower.ncols(), width);
    assert_eq!(matrix.nrows(), n_levels * width);
    if !diagonal && width >= 16 {
        // Temporary storage is limited to one level's rows.
        for level in 0..n_levels {
            let offset = level * width;
            let transformed = lower.transpose() * matrix.as_ref().subrows(offset, width);
            matrix
                .as_mut()
                .subrows_mut(offset, width)
                .copy_from(transformed.as_ref());
        }
        return;
    }
    for column in 0..matrix.ncols() {
        for level in 0..n_levels {
            let offset = level * width;
            // Ascending rows only read entries not yet overwritten.
            for i in 0..width {
                let end = if diagonal { i + 1 } else { width };
                let mut value = 0.0;
                for k in i..end {
                    value += lower[(k, i)] * matrix[(offset + k, column)];
                }
                matrix[(offset + i, column)] = value;
            }
        }
    }
}

/// Apply a repeated factor to a view's columns without replacing its buffer.
pub(crate) fn right_apply_repeated_in_place(
    lower: &Mat<f64>,
    n_levels: usize,
    diagonal: bool,
    mut matrix: MatMut<'_, f64>,
) {
    let width = lower.nrows();
    assert_eq!(lower.ncols(), width);
    assert_eq!(matrix.ncols(), n_levels * width);
    for level in 0..n_levels {
        let offset = level * width;
        if !diagonal && width >= 16 {
            let transformed = matrix.as_ref().subcols(offset, width) * lower;
            matrix
                .as_mut()
                .subcols_mut(offset, width)
                .copy_from(transformed.as_ref());
            continue;
        }
        // Ascending columns only read entries not yet overwritten.
        for j in 0..width {
            let end = if diagonal { j + 1 } else { width };
            for row in 0..matrix.nrows() {
                let mut value = 0.0;
                for k in j..end {
                    value += matrix[(row, offset + k)] * lower[(k, j)];
                }
                matrix[(row, offset + j)] = value;
            }
        }
    }
}

/// Multiply each level's rows by a diagonal factor.
fn scale_levels(lower: &Mat<f64>, values: &mut [f64]) {
    let width = lower.nrows();
    if width == 1 {
        let scale = lower[(0, 0)];
        for value in values {
            *value *= scale;
        }
        return;
    }
    for level in values.chunks_exact_mut(width) {
        for (i, value) in level.iter_mut().enumerate() {
            *value *= lower[(i, i)];
        }
    }
}

#[derive(Debug)]
struct FactorBlock {
    offset: usize,
    n_levels: usize,
    diagonal: bool,
    lower: Mat<f64>,
}

/// One covariance factor per structure, repeated over its grouping levels.
/// Multiplication preserves every cross-structure and cross-level contribution.
#[derive(Debug)]
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
        Self::from_blocks(build_lambda_blocks(theta, structures), structures)
    }

    /// Reuse factors already built for a blocked normal-matrix factorization.
    pub(crate) fn from_blocks(blocks: Vec<Mat<f64>>, structures: &[RandomEffectStructure]) -> Self {
        debug_assert_eq!(blocks.len(), structures.len());
        let mut offset = 0;
        let blocks = blocks
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
        Self {
            blocks,
            dimension: offset,
        }
    }

    pub fn apply(&self, vector: &Col<f64>) -> Col<f64> {
        self.apply_vector::<false>(vector)
    }

    /// Return the repeated diagonal when the covariance factor has no coupling.
    pub fn diagonal_values(&self) -> Option<Vec<f64>> {
        let mut values = Vec::with_capacity(self.dimension);
        for block in &self.blocks {
            let width = block.lower.nrows();
            for row in 0..width {
                if !block.lower[(row, row)].is_finite()
                    || (0..row).any(|column| block.lower[(row, column)] != 0.0)
                {
                    return None;
                }
            }
            for _ in 0..block.n_levels {
                values.extend((0..width).map(|row| block.lower[(row, row)]));
            }
        }
        Some(values)
    }

    /// Form Z * Lambda without allocating a dense covariance or design matrix.
    /// A row accumulator merges contributions from correlated slope columns.
    pub fn sparse_design(&self, design: &CscMatrix) -> Option<CscMatrix> {
        assert_eq!(design.ncols(), self.dimension);
        if design.values().iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut offsets = vec![0];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut accumulator = vec![0.0; design.nrows()];
        let mut present = vec![false; design.nrows()];
        let mut rows = Vec::new();
        for block in &self.blocks {
            let width = block.lower.nrows();
            for level in 0..block.n_levels {
                let offset = block.offset + level * width;
                for column in 0..width {
                    let end = if block.diagonal { column + 1 } else { width };
                    for source in column..end {
                        let coefficient = block.lower[(source, column)];
                        if !coefficient.is_finite() {
                            return None;
                        }
                        if coefficient == 0.0 {
                            continue;
                        }
                        let source = offset + source;
                        for position in
                            design.col_offsets()[source]..design.col_offsets()[source + 1]
                        {
                            let row = design.row_indices()[position];
                            if !present[row] {
                                present[row] = true;
                                rows.push(row);
                            }
                            accumulator[row] += coefficient * design.values()[position];
                        }
                    }
                    rows.sort_unstable();
                    for row in rows.drain(..) {
                        let value = accumulator[row];
                        if !value.is_finite() {
                            return None;
                        }
                        if value != 0.0 {
                            indices.push(row);
                            values.push(value);
                        }
                        accumulator[row] = 0.0;
                        present[row] = false;
                    }
                    offsets.push(values.len());
                }
            }
        }
        Some(
            CscMatrix::try_from_usize(
                &values,
                &indices,
                &offsets,
                (design.nrows(), self.dimension),
            )
            .expect("transformed design has canonical columns"),
        )
    }

    pub fn transpose_apply_vector(&self, vector: &Col<f64>) -> Col<f64> {
        self.apply_vector::<true>(vector)
    }

    fn apply_vector<const TRANSPOSE: bool>(&self, vector: &Col<f64>) -> Col<f64> {
        self.product::<TRANSPOSE>(vector.as_mat()).col(0).to_owned()
    }

    /// Multiply a matrix by Lambda from the left.
    pub fn apply_matrix(&self, matrix: MatRef<'_, f64>) -> Mat<f64> {
        self.product::<false>(matrix)
    }

    fn product<const TRANSPOSE: bool>(&self, matrix: MatRef<'_, f64>) -> Mat<f64> {
        assert_eq!(matrix.nrows(), self.dimension);
        let mut result = matrix.to_owned();
        for column in 0..result.ncols() {
            let values = result.col_as_slice_mut(column);
            for block in &self.blocks {
                let width = block.lower.nrows();
                let values = &mut values[block.offset..block.offset + block.n_levels * width];
                if block.diagonal {
                    scale_levels(&block.lower, values);
                    continue;
                }
                let lower = &block.lower;
                for level in values.chunks_exact_mut(width) {
                    if TRANSPOSE {
                        // Ascending rows only read entries not yet overwritten.
                        for i in 0..width {
                            level[i] = (i..width).map(|k| lower[(k, i)] * level[k]).sum();
                        }
                    } else {
                        // Descending rows only read entries not yet overwritten.
                        for i in (0..width).rev() {
                            level[i] = (0..=i).map(|k| lower[(i, k)] * level[k]).sum();
                        }
                    }
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
            if block.diagonal {
                let rows = block.offset..block.offset + block.n_levels * block.lower.nrows();
                for column in 0..matrix.ncols() {
                    scale_levels(
                        &block.lower,
                        &mut matrix.col_as_slice_mut(column)[rows.clone()],
                    );
                }
                continue;
            }
            transpose_apply_repeated_in_place(
                &block.lower,
                block.n_levels,
                block.diagonal,
                matrix.subrows_mut(block.offset, block.n_levels * block.lower.nrows()),
            );
        }
    }

    /// Multiply by Lambda directly, without transposing a full intermediate.
    #[cfg(test)]
    pub fn right_apply(&self, matrix: MatRef<'_, f64>) -> Mat<f64> {
        assert_eq!(matrix.ncols(), self.dimension);
        let mut result = matrix.to_owned();
        self.right_apply_in_place(&mut result);
        result
    }

    fn right_apply_in_place(&self, matrix: &mut Mat<f64>) {
        for block in &self.blocks {
            right_apply_repeated_in_place(
                &block.lower,
                block.n_levels,
                block.diagonal,
                matrix.subcols_mut(block.offset, block.n_levels * block.lower.nrows()),
            );
        }
    }

    /// Transform the owned crossproduct in place to Lambda' * S * Lambda + I.
    pub fn penalized_crossproduct(&self, mut matrix: Mat<f64>) -> Mat<f64> {
        assert_eq!(matrix.nrows(), self.dimension);
        assert_eq!(matrix.ncols(), self.dimension);
        self.transpose_apply_in_place(&mut matrix);
        self.right_apply_in_place(&mut matrix);
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
    fn repeated_transforms_only_modify_the_requested_view() {
        let width = if cfg!(miri) { 3 } else { 17 };
        let q = 2 * width;
        let lower = Mat::from_fn(width, width, |i, j| {
            if i >= j {
                (i + j + 1) as f64 / 32.0
            } else {
                0.0
            }
        });
        let dense = Mat::from_fn(q, q, |i, j| {
            if i / width == j / width {
                lower[(i % width, j % width)]
            } else {
                0.0
            }
        });
        for left in [false, true] {
            let (rows, columns) = if left { (q, 5) } else { (5, q) };
            let mut buffer = Mat::from_fn(rows + 2, columns + 2, |i, j| ((i + 3 * j) as f64).cos());
            let original = buffer.clone();
            let view = original.submatrix(1, 1, rows, columns);
            let expected = if left {
                dense.transpose() * view
            } else {
                view * &dense
            };
            let view = buffer.submatrix_mut(1, 1, rows, columns);
            if left {
                transpose_apply_repeated_in_place(&lower, 2, false, view);
            } else {
                right_apply_repeated_in_place(&lower, 2, false, view);
            }
            assert_close(buffer.submatrix(1, 1, rows, columns), expected.as_ref());
            for j in 0..columns + 2 {
                for i in 0..rows + 2 {
                    if i == 0 || i == rows + 1 || j == 0 || j == columns + 1 {
                        assert_eq!(buffer[(i, j)], original[(i, j)]);
                    }
                }
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
            assert_close(
                factor.right_apply(matrix.transpose()).as_ref(),
                (matrix.transpose() * &dense).as_ref(),
            );
        }
    }

    #[test]
    fn right_transforms_preserve_rectangular_views_at_kernel_boundary() {
        let widths: &[usize] = if cfg!(miri) { &[3] } else { &[15, 16, 17, 32] };
        let rows: &[usize] = if cfg!(miri) {
            &[0, 1, 4]
        } else {
            &[0, 1, 7, 65]
        };
        for &width in widths {
            for correlated in [false, true] {
                let structures = [RandomEffectStructure {
                    n_levels: 2,
                    n_terms: width,
                    correlated,
                }];
                for zero in [false, true] {
                    let coefficient = |i, j| {
                        if zero || i < j || j == width - 1 || (!correlated && i != j) {
                            0.0
                        } else if i == j {
                            0.3 + i as f64 / 32.0
                        } else {
                            (i + j + 1) as f64 / 64.0
                        }
                    };
                    let mut theta = Vec::new();
                    for i in 0..width {
                        for j in if correlated { 0..i + 1 } else { i..i + 1 } {
                            theta.push(coefficient(i, j));
                        }
                    }
                    let q = 2 * width;
                    let dense = Mat::from_fn(q, q, |i, j| {
                        if i / width == j / width {
                            coefficient(i % width, j % width)
                        } else {
                            0.0
                        }
                    });
                    let factor = CovarianceFactor::new(&theta, &structures);
                    for &nrows in rows {
                        // Padding and transposed views exercise both storage strides.
                        let buffer =
                            Mat::from_fn(nrows + 2, q + 2, |i, j| ((3 * i + 7 * j) as f64).sin());
                        let original = buffer.clone();
                        let matrix = buffer.submatrix(1, 1, nrows, q);
                        let transposed = matrix.transpose().to_owned();
                        let expected = matrix * &dense;
                        for view in [matrix, transposed.transpose()] {
                            assert_close(factor.right_apply(view).as_ref(), expected.as_ref());
                        }
                        assert_eq!(buffer, original);
                    }
                }
            }
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
    fn sparse_design_matches_dense_covariance_product() {
        let (structures, mut theta, dense) = fixture();
        let design = Mat::from_fn(31, 14, |i, j| {
            if (i + 2 * j) % 5 == 0 {
                (i + j + 1) as f64 / 17.0
            } else {
                0.0
            }
        });
        let mut offsets = vec![0];
        let mut rows = Vec::new();
        let mut values = Vec::new();
        for column in 0..14 {
            for row in 0..31 {
                if design[(row, column)] != 0.0 {
                    rows.push(row);
                    values.push(design[(row, column)]);
                }
            }
            offsets.push(values.len());
        }
        let input = CscMatrix::try_from_usize(&values, &rows, &offsets, (31, 14)).unwrap();
        let transformed = CovarianceFactor::new(&theta, &structures)
            .sparse_design(&input)
            .unwrap();
        let mut actual = Mat::zeros(31, 14);
        for column in 0..14 {
            for entry in transformed.col_offsets()[column]..transformed.col_offsets()[column + 1] {
                actual[(transformed.row_indices()[entry], column)] = transformed.values()[entry];
            }
        }
        assert_close(actual.as_ref(), (&design * &dense).as_ref());
        theta.fill(0.0);
        let zero = CovarianceFactor::new(&theta, &structures)
            .sparse_design(&input)
            .unwrap();
        assert_eq!(zero.ncols(), 14);
        assert!(zero.values().is_empty());
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
                factor.right_apply(Mat::zeros(4, 0).as_ref()),
                Mat::<f64>::zeros(4, 0)
            );
            assert_eq!(
                factor.penalized_crossproduct(Mat::zeros(0, 0)),
                Mat::<f64>::zeros(0, 0)
            );
        }
    }
}

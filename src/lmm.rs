use std::sync::Arc;

use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Llt, Solve};
use faer::{Accum, ColRef, Mat, MatMut, MatRef, Par, Side};
use numpy::PyArray1;
use numpy::ndarray::ArrayView1;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::blocked_chol::{BlockedCholesky, LevelCholesky, SparseCholesky, SparseLdl};
pub use crate::covariance::RandomEffectStructure;
use crate::covariance::{CovarianceFactor, build_lambda_blocks};
use crate::csc::{CscMatrix, LevelTiles, SCALAR_WIDTH};
use crate::linalg::LinalgError;

fn random_effect_structures(
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
) -> PyResult<Vec<RandomEffectStructure>> {
    if n_levels.len() != n_terms.len() || n_levels.len() != correlated.len() {
        return Err(PyValueError::new_err(
            "random-effect structure arrays must have equal lengths",
        ));
    }
    Ok(n_levels
        .into_iter()
        .zip(n_terms)
        .zip(correlated)
        .map(|((n_levels, n_terms), correlated)| RandomEffectStructure {
            n_levels,
            n_terms,
            correlated,
        })
        .collect())
}

fn csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix, LinalgError> {
    CscMatrix::try_from_i64(data, indices, indptr, shape)
}

/// A covariance parameter selects one entry of a repeated factor block.
struct LambdaDerivative {
    /// Index of the structure and of its first random effect.
    structure: usize,
    offset: usize,
    n_levels: usize,
    n_terms: usize,
    row: usize,
    column: usize,
}

impl LambdaDerivative {
    fn bilinear(&self, left: ColRef<'_, f64>, right: ColRef<'_, f64>) -> f64 {
        (0..self.n_levels)
            .map(|level| {
                let offset = self.offset + level * self.n_terms;
                left[offset + self.row] * right[offset + self.column]
            })
            .sum()
    }

    fn apply<const TRANSPOSE: bool>(&self, matrix: &Mat<f64>) -> Mat<f64> {
        let mut result = Mat::zeros(matrix.nrows(), matrix.ncols());
        let (source, target) = if TRANSPOSE {
            (self.row, self.column)
        } else {
            (self.column, self.row)
        };
        for column in 0..matrix.ncols() {
            for level in 0..self.n_levels {
                let offset = self.offset + level * self.n_terms;
                result[(offset + target, column)] = matrix[(offset + source, column)];
            }
        }
        result
    }

    #[cfg(test)]
    fn crossproduct_derivative(&self, ztwz_lambda: &Mat<f64>) -> Mat<f64> {
        // d(Lambda' Z'WZ Lambda) = dLambda' (Z'WZ Lambda) + its transpose.
        // Retain products across every level and structure, including overlap.
        let dimension = ztwz_lambda.nrows();
        let mut derivative = Mat::zeros(dimension, dimension);
        for level in 0..self.n_levels {
            let offset = self.offset + level * self.n_terms;
            for column in 0..dimension {
                let value = ztwz_lambda[(offset + self.row, column)];
                if value != 0.0 {
                    derivative[(offset + self.column, column)] += value;
                    derivative[(column, offset + self.column)] += value;
                }
            }
        }
        derivative
    }
}

fn lambda_derivatives(structures: &[RandomEffectStructure]) -> Vec<LambdaDerivative> {
    let mut derivatives = Vec::new();
    let mut offset = 0;
    for (index, structure) in structures.iter().enumerate() {
        for row in 0..structure.n_terms {
            let columns = if structure.correlated {
                0..row + 1
            } else {
                row..row + 1
            };
            for column in columns {
                derivatives.push(LambdaDerivative {
                    structure: index,
                    offset,
                    n_levels: structure.n_levels,
                    n_terms: structure.n_terms,
                    row,
                    column,
                });
            }
        }
        offset += structure.n_levels * structure.n_terms;
    }
    derivatives
}

/// Shared adjoint for derivatives of the conditional norm and spherical penalty.
struct ModeGradient<'a> {
    mode: ColRef<'a, f64>,
    conditional: ColRef<'a, f64>,
    adjoint: Mat<f64>,
}

impl<'a> ModeGradient<'a> {
    fn new(
        mode: &'a Mat<f64>,
        conditional: &'a Mat<f64>,
        factor: &CovarianceFactor,
        chol: &PrecisionFactor,
    ) -> Self {
        // The coefficient of du is u - Lambda' Z'W residual. Solve its adjoint
        // once so every parameter can contract with the mode equation's RHS.
        // Keep this residual even though it vanishes in exact arithmetic:
        // large covariance factors can amplify the mode solve's rounding error.
        let residual = mode - factor.transpose_apply(conditional.as_ref());
        Self {
            mode: mode.col(0),
            conditional: conditional.col(0),
            adjoint: chol.solve(&residual),
        }
    }

    fn derivative(&self, derivative: &LambdaDerivative, rhs: &Mat<f64>) -> f64 {
        2.0 * ((0..self.mode.nrows())
            .map(|i| self.adjoint[(i, 0)] * rhs[(i, 0)])
            .sum::<f64>()
            - derivative.bilinear(self.conditional, self.mode))
    }
    fn project(
        &self,
        marginal: &Mat<f64>,
        products: &GradientCrossproducts,
    ) -> ProjectedModeGradient<'_> {
        ProjectedModeGradient {
            mode: self.mode,
            adjoint: self.adjoint.col(0),
            marginal_residual: marginal - products.apply(self.mode.as_mat()),
            adjusted_conditional: self.conditional.as_mat() + products.apply(self.adjoint.as_ref()),
        }
    }
}

/// Reuse two projections when wide factors have many covariance derivatives.
struct ProjectedModeGradient<'a> {
    mode: ColRef<'a, f64>,
    adjoint: ColRef<'a, f64>,
    marginal_residual: Mat<f64>,
    adjusted_conditional: Mat<f64>,
}

impl ProjectedModeGradient<'_> {
    fn derivative(&self, derivative: &LambdaDerivative) -> f64 {
        // A = Z'WZ Lambda, a = V^-1(u - Lambda' c), E = dLambda.
        // z and c are the marginal and conditional residual crossproducts.
        // a'(E'z - (E'A + A'E)u) - c'Eu
        //   = (E a)'(z - A u) - (E u)'(c + A a).
        2.0 * (derivative.bilinear(self.marginal_residual.col(0), self.adjoint)
            - derivative.bilinear(self.adjusted_conditional.col(0), self.mode))
    }
}

/// Shared products for the REML fixed-information log determinant.
struct FixedEffectGradient {
    weighted: Mat<f64>,
    residual: Mat<f64>,
}

impl FixedEffectGradient {
    fn new(
        products: &GradientCrossproducts,
        projection: &Mat<f64>,
        weighted: Mat<f64>,
        ztwx: &Mat<f64>,
    ) -> Self {
        let mut residual = products.apply(projection.as_ref());
        residual -= ztwx;
        Self { weighted, residual }
    }

    fn derivative(&self, derivative: &LambdaDerivative) -> f64 {
        // P = V^-1 B, W = P C^-1, A = Z'W_obs Z Lambda, T = Z'W_obs X.
        // Symmetry of C^-1 combines the two dV terms:
        // tr(C^-1 dC) = 2 <dLambda W, A P - T>.
        let mut trace = 0.0;
        for column in 0..self.weighted.ncols() {
            for level in 0..derivative.n_levels {
                let offset = derivative.offset + level * derivative.n_terms;
                trace += self.weighted[(offset + derivative.column, column)]
                    * self.residual[(offset + derivative.row, column)];
            }
        }
        2.0 * trace
    }
}

/// A = Z'WZ Lambda as an operator, without forming the square product.
struct GradientCrossproducts<'a> {
    crossproducts: &'a LevelTiles,
    factor: &'a CovarianceFactor,
}

impl GradientCrossproducts<'_> {
    fn apply(&self, rhs: MatRef<'_, f64>) -> Mat<f64> {
        self.crossproducts
            .symmetric_product(&self.factor.apply_matrix(rhs))
    }

    /// dV rhs = E'(A rhs) + A'(E rhs) for the selection E = dLambda.
    fn derivative_product(&self, derivative: &LambdaDerivative, rhs: &Mat<f64>) -> Mat<f64> {
        let mut product = derivative.apply::<true>(&self.apply(rhs.as_ref()));
        let selected = self
            .crossproducts
            .symmetric_product(&derivative.apply::<false>(rhs));
        product += self.factor.transpose_apply(selected.as_ref());
        product
    }
}

/// tr(V^-1 dV) for each covariance parameter, from V^-1 on the tile pattern.
/// With S = Z'WZ, A = S Lambda and E = dLambda, tr(V^-1 (E'A + A'E)) =
/// 2 tr(A V^-1 E') sums one entry of each of the parameter's level blocks on
/// the diagonal of A V^-1. Those blocks only need V^-1 where S couples levels.
fn covariance_traces(
    tiles: &LevelTiles,
    lambda: &[Mat<f64>],
    inverse: &[f64],
    derivatives: &[LambdaDerivative],
) -> Vec<f64> {
    // Diagonal level blocks of A V^-1, summed over each structure's levels.
    let mut sums: Vec<Vec<f64>> = lambda
        .iter()
        .map(|factor| vec![0.0; factor.nrows() * factor.nrows()])
        .collect();
    let mut product = Vec::new();
    if tiles.block_diagonal() {
        for ((width, span, _), (factor, sum)) in tiles.runs().zip(lambda.iter().zip(&mut sums)) {
            let (crossproducts, selected) = (&tiles.values()[span.clone()], &inverse[span]);
            if width == 1 {
                let trace: f64 = crossproducts.iter().zip(selected).map(|(s, v)| s * v).sum();
                sum[0] = factor[(0, 0)] * trace;
                continue;
            }
            let size = width * width;
            for (crossproduct, selected) in crossproducts
                .chunks_exact(size)
                .zip(selected.chunks_exact(size))
            {
                add_tile_trace(sum, crossproduct, factor, selected, &mut product);
            }
        }
    } else {
        for column in 0..tiles.n_levels() {
            let width = tiles.width(column);
            let lambda_column = &lambda[tiles.block(column)];
            for tile in tiles.column(column) {
                let row = tiles.row(tile);
                let lambda_row = &lambda[tiles.block(row)];
                let (crossproduct, selected) = (tiles.tile(tile), &inverse[tiles.span(tile)]);
                let sum = &mut sums[tiles.block(column)];
                add_tile_trace(sum, crossproduct, lambda_row, selected, &mut product);
                if row == column {
                    continue;
                }
                // The row's block gains G Lambda_column Z' through the mirrored tile.
                let height = tiles.width(row);
                product.clear();
                for j in 0..height {
                    product.extend((0..width).map(|a| {
                        (0..=a)
                            .map(|b| lambda_column[(a, b)] * selected[j + b * height])
                            .sum::<f64>()
                    }));
                }
                let sum = &mut sums[tiles.block(row)];
                for j in 0..height {
                    for i in 0..height {
                        sum[i + j * height] += (0..width)
                            .map(|a| crossproduct[i + a * height] * product[a + j * width])
                            .sum::<f64>();
                    }
                }
            }
        }
    }
    derivatives
        .iter()
        .map(|derivative| {
            2.0 * sums[derivative.structure]
                [derivative.row + derivative.column * derivative.n_terms]
        })
        .collect()
}

/// Add G' Lambda Z to a column level's block, for tiles G of S and Z of V^-1
/// and the row level's factor Lambda.
fn add_tile_trace(
    block: &mut [f64],
    crossproduct: &[f64],
    lambda: &Mat<f64>,
    selected: &[f64],
    product: &mut Vec<f64>,
) {
    let height = lambda.nrows();
    let width = crossproduct.len() / height;
    if height.max(width) >= SCALAR_WIDTH {
        let mut product = Mat::zeros(height, width);
        let selected = MatRef::from_column_major_slice(selected, height, width);
        matmul(
            product.as_mut(),
            Accum::Replace,
            lambda,
            selected,
            1.0,
            Par::Seq,
        );
        let crossproduct = MatRef::from_column_major_slice(crossproduct, height, width);
        let block = MatMut::from_column_major_slice_mut(block, width, width);
        matmul(
            block,
            Accum::Add,
            crossproduct.transpose(),
            &product,
            1.0,
            Par::Seq,
        );
        return;
    }
    product.clear();
    for j in 0..width {
        product.extend((0..height).map(|a| {
            (0..=a)
                .map(|b| lambda[(a, b)] * selected[b + j * height])
                .sum::<f64>()
        }));
    }
    for j in 0..width {
        for i in 0..width {
            block[i + j * width] += (0..height)
                .map(|a| crossproduct[a + i * height] * product[a + j * height])
                .sum::<f64>();
        }
    }
}

fn compute_ztwy_sparse(z: &CscMatrix, w: &[f64], y: &[f64], q: usize) -> Mat<f64> {
    let mut result = Mat::zeros(q, 1);

    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        let mut sum = 0.0;
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            sum += z.values()[idx] * w[i] * y[i];
        }
        result[(j, 0)] = sum;
    }

    result
}

fn compute_ztwx_sparse(z: &CscMatrix, w: &[f64], x: &Mat<f64>, q: usize, p: usize) -> Mat<f64> {
    let mut result = Mat::zeros(q, p);

    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];

        for pj in 0..p {
            let mut sum = 0.0;
            for idx in col_start..col_end {
                let i = z.row_indices()[idx];
                sum += z.values()[idx] * w[i] * x[(i, pj)];
            }
            result[(j, pj)] = sum;
        }
    }

    result
}

fn marginal_residual(x: &Mat<f64>, beta: &Mat<f64>, y: &[f64]) -> Vec<f64> {
    let mut residual = y.to_vec();
    if x.ncols() != 0 {
        // A matrix-vector product is memory bound; thread dispatch only adds latency.
        matmul(
            faer::ColMut::from_slice_mut(&mut residual).as_mat_mut(),
            Accum::Add,
            x,
            beta,
            -1.0,
            Par::Seq,
        );
    }
    residual
}

fn structure_blocks(structures: &[RandomEffectStructure]) -> Vec<(usize, usize)> {
    structures.iter().map(|s| (s.n_levels, s.n_terms)).collect()
}

fn structure_parameters(structure: &RandomEffectStructure) -> usize {
    if structure.correlated {
        structure.n_terms * (structure.n_terms + 1) / 2
    } else {
        structure.n_terms
    }
}

/// Coupled structures are eliminated with the most random-effect columns first,
/// ties going to more levels and then to caller order. Each structure's levels
/// are mutually independent, so a random crossing leaves only the smaller
/// structures for the blocked factorization's dense Schur complement, and a
/// nested factor with its parent's terms adds no fill. Inputs and outputs keep
/// caller order.
struct StructureOrder {
    /// Caller structure, parameter and random-effect column for each internal one.
    structures: Vec<usize>,
    parameters: Vec<usize>,
    columns: Vec<usize>,
}

impl StructureOrder {
    fn new(structures: &[RandomEffectStructure]) -> Option<Self> {
        let mut order: Vec<usize> = (0..structures.len()).collect();
        order.sort_by_key(|&index| {
            let structure = structures[index];
            std::cmp::Reverse((structure.n_levels * structure.n_terms, structure.n_levels))
        });
        if order
            .iter()
            .enumerate()
            .all(|(position, &index)| position == index)
        {
            return None;
        }
        let (mut parameter_starts, mut column_starts) = (vec![0], vec![0]);
        for structure in structures {
            parameter_starts
                .push(parameter_starts.last().unwrap() + structure_parameters(structure));
            column_starts
                .push(column_starts.last().unwrap() + structure.n_levels * structure.n_terms);
        }
        let ranges = |starts: &[usize]| {
            order
                .iter()
                .flat_map(|&index| starts[index]..starts[index + 1])
                .collect()
        };
        Some(Self {
            parameters: ranges(&parameter_starts),
            columns: ranges(&column_starts),
            structures: order,
        })
    }

    /// Gather caller values into elimination order.
    fn gather(caller: &[f64], positions: &[usize]) -> Vec<f64> {
        positions.iter().map(|&position| caller[position]).collect()
    }

    /// Return values in elimination order to their caller positions.
    fn scatter(internal: &[f64], positions: &[usize]) -> Vec<f64> {
        let mut caller = vec![0.0; internal.len()];
        for (&position, &value) in positions.iter().zip(internal) {
            caller[position] = value;
        }
        caller
    }
}

/// Dense Schur complements narrower than this are not worth a sparse analysis.
const SPARSE_MIN_DIMENSION: usize = 64;

/// Dense Schur kernels are taken to run this many times more operations per
/// second than simplicial elimination. With this ratio, measured random,
/// partial and regular crossings each picked their faster factorization.
const DENSE_SPEEDUP: usize = 4;

/// Z'WZ as level tiles in elimination order, with the factorization chosen for
/// its fill. Leading independent levels are eliminated one tile at a time and
/// the rest densely, unless a sparse factorization of the whole system needs
/// much less work. Random crossings fill the dense Schur complement anyway;
/// nested and regularly crossed factors leave it nearly empty.
struct DesignCrossproducts {
    tiles: LevelTiles,
    leading: usize,
    sparse: Option<SparseCholesky>,
}

impl DesignCrossproducts {
    fn new(tiles: LevelTiles) -> Self {
        let leading = tiles.independent_prefix();
        let dense = tiles.dimension() - tiles.start(leading);
        let sparse = if dense >= SPARSE_MIN_DIMENSION {
            // Compare with the m^3 / 3 operations of the dense Cholesky factor.
            let dense_flops = dense.saturating_pow(3) / 3;
            SparseCholesky::new(&tiles, dense_flops / DENSE_SPEEDUP)
        } else {
            None
        };
        Self {
            tiles,
            leading,
            sparse,
        }
    }

    fn factor(&self, lambda: &[Mat<f64>]) -> Result<PrecisionFactor<'_>, LinalgError> {
        let values = self.tiles.penalized(lambda);
        if self.tiles.block_diagonal() {
            return LevelCholesky::factor(&self.tiles, values).map(PrecisionFactor::Levels);
        }
        match &self.sparse {
            Some(analysis) => analysis.factor(&values).map(PrecisionFactor::Sparse),
            None => BlockedCholesky::factor(&self.tiles, self.leading, values)
                .map(PrecisionFactor::Blocked),
        }
    }

    /// Order of the dense Schur complement, if the blocked factorization is used.
    fn dense_dimension(&self) -> usize {
        if self.sparse.is_some() {
            0
        } else {
            self.tiles.dimension() - self.tiles.start(self.leading)
        }
    }
}

/// Factor of V = Lambda' Z'WZ Lambda + I from the design's chosen elimination.
enum PrecisionFactor<'a> {
    Levels(LevelCholesky<'a>),
    Blocked(BlockedCholesky<'a>),
    Sparse(SparseLdl<'a>),
}

impl PrecisionFactor<'_> {
    fn logdet(&self) -> f64 {
        match self {
            Self::Levels(chol) => chol.logdet(),
            Self::Blocked(chol) => chol.logdet(),
            Self::Sparse(factor) => factor.logdet(),
        }
    }

    /// Whiten a right-hand side: the result's crossproduct is b' V^-1 b.
    fn solve_lower(&self, b: &Mat<f64>) -> Mat<f64> {
        let mut result = b.clone();
        match self {
            Self::Levels(chol) => chol.solve_in_place::<false>(&mut result),
            Self::Blocked(chol) => chol.solve_lower_in_place(&mut result),
            Self::Sparse(factor) => factor.solve_lower_in_place(&mut result),
        }
        result
    }

    fn solve(&self, b: &Mat<f64>) -> Mat<f64> {
        let mut result = b.clone();
        match self {
            Self::Levels(chol) => {
                chol.solve_in_place::<false>(&mut result);
                chol.solve_in_place::<true>(&mut result);
            }
            Self::Blocked(chol) => {
                chol.solve_lower_in_place(&mut result);
                chol.solve_upper_in_place(&mut result);
            }
            Self::Sparse(factor) => factor.solve_in_place(&mut result),
        }
        result
    }

    fn selected_inverse(&self) -> Vec<f64> {
        match self {
            Self::Levels(chol) => chol.selected_inverse(),
            Self::Blocked(chol) => chol.selected_inverse(),
            Self::Sparse(factor) => factor.selected_inverse(),
        }
    }
}

/// Products that depend only on the design, shared by independent responses.
struct PreparedLmmDesign {
    x: Mat<f64>,
    wx: Mat<f64>,
    z: CscMatrix,
    weights: Vec<f64>,
    sqrt_weights: Vec<f64>,
    offset: Vec<f64>,
    logdet_weights: f64,
    xtwx: Mat<f64>,
    ztwx: Mat<f64>,
    crossproducts: DesignCrossproducts,
    /// Structures in elimination order, with the map back to caller order.
    structures: Vec<RandomEffectStructure>,
    order: Option<StructureOrder>,
    n_theta: usize,
}

impl PreparedLmmDesign {
    fn new(
        x: Mat<f64>,
        z: CscMatrix,
        weights: Vec<f64>,
        offset: Vec<f64>,
        structures: Vec<RandomEffectStructure>,
    ) -> Result<Self, &'static str> {
        let (n, p, q) = (x.nrows(), x.ncols(), z.ncols());
        if n == 0 || z.nrows() != n || weights.len() != n || offset.len() != n {
            return Err("design, weights and offset must have the same nonzero row count");
        }
        if weights.iter().any(|w| !w.is_finite()) {
            return Err("weights must contain only finite values");
        }
        if weights.iter().any(|w| *w <= 0.0) {
            return Err("weights must be strictly positive");
        }
        if offset.iter().any(|v| !v.is_finite())
            || z.values().iter().any(|v| !v.is_finite())
            || (0..n).any(|i| (0..p).any(|j| !x[(i, j)].is_finite()))
        {
            return Err("design and offset must contain only finite values");
        }
        let mut columns = 0usize;
        let mut n_theta = 0usize;
        for structure in &structures {
            let terms = structure.n_terms;
            if terms == 0 || structure.n_levels == 0 {
                return Err("random-effect structures must have positive dimensions");
            }
            columns = structure
                .n_levels
                .checked_mul(terms)
                .and_then(|size| columns.checked_add(size))
                .ok_or("random-effect dimensions overflow")?;
            let parameters = if structure.correlated {
                terms
                    .checked_add(1)
                    .and_then(|next| terms.checked_mul(next))
                    .map(|product| product / 2)
                    .ok_or("random-effect dimensions overflow")?
            } else {
                terms
            };
            n_theta = n_theta
                .checked_add(parameters)
                .ok_or("random-effect dimensions overflow")?;
        }
        if columns != q {
            return Err("random-effect structures do not match the design column count");
        }
        let blocks = structure_blocks(&structures);
        let independent = z.weighted_repeated_block_crossproducts(&weights, &blocks);
        let (z, structures, order, tiles) = match independent {
            Some(stacked) => {
                let tiles = LevelTiles::from_level_blocks(&blocks, &stacked);
                (z, structures, None, tiles)
            }
            None => {
                let order = StructureOrder::new(&structures);
                let (z, structures) = match &order {
                    Some(order) => (
                        z.select_columns(&order.columns),
                        order
                            .structures
                            .iter()
                            .map(|&index| structures[index])
                            .collect(),
                    ),
                    None => (z, structures),
                };
                let blocks = structure_blocks(&structures);
                // Rows spanning levels only through stored or cancelling zeros
                // leave no couplings; such levels are again eliminated blockwise.
                let tiles = z.weighted_level_crossproduct(&weights, &blocks);
                (z, structures, order, tiles)
            }
        };
        let crossproducts = DesignCrossproducts::new(tiles);
        let sqrt_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
        let logdet_weights = weights.iter().map(|w| w.ln()).sum();
        let wx = Mat::from_fn(n, p, |i, j| sqrt_weights[i] * x[(i, j)]);
        let xtwx = wx.transpose() * &wx;
        let ztwx = compute_ztwx_sparse(&z, &weights, &x, q, p);
        Ok(Self {
            x,
            wx,
            z,
            weights,
            sqrt_weights,
            offset,
            logdet_weights,
            xtwx,
            ztwx,
            crossproducts,
            structures,
            order,
            n_theta,
        })
    }

    /// Covariance parameters in elimination order.
    fn internal_theta<'t>(&self, theta: &'t [f64]) -> std::borrow::Cow<'t, [f64]> {
        match &self.order {
            Some(order) => StructureOrder::gather(theta, &order.parameters).into(),
            None => theta.into(),
        }
    }

    fn engine(&self) -> &'static str {
        let crossproducts = &self.crossproducts;
        if crossproducts.tiles.block_diagonal() {
            "levels"
        } else if crossproducts.sparse.is_some() {
            "sparse"
        } else {
            "blocked"
        }
    }

    fn with_response(
        self: &Arc<Self>,
        y: ArrayView1<'_, f64>,
    ) -> Result<PreparedLmmResponse, &'static str> {
        let (n, p, q) = (self.x.nrows(), self.x.ncols(), self.z.ncols());
        if y.len() != n || y.iter().any(|v| !v.is_finite()) {
            return Err("response must contain one finite value per observation");
        }
        let y_adj: Vec<f64> = y.iter().zip(&self.offset).map(|(y, o)| y - o).collect();
        // Retain the existing accumulation order for both likelihood paths.
        let xtwy = if q == 0 {
            let wy = Mat::from_fn(n, 1, |i, _| self.sqrt_weights[i] * y_adj[i]);
            self.wx.transpose() * &wy
        } else {
            Mat::from_fn(p, 1, |i, _| {
                (0..n)
                    .map(|row| self.wx[(row, i)] * self.sqrt_weights[row] * y_adj[row])
                    .sum()
            })
        };
        let ztwy = compute_ztwy_sparse(&self.z, &self.weights, &y_adj, q);
        Ok(PreparedLmmResponse {
            design: Arc::clone(self),
            y_adj,
            xtwy,
            ztwy,
        })
    }
}

struct PreparedLmmResponse {
    design: Arc<PreparedLmmDesign>,
    y_adj: Vec<f64>,
    xtwy: Mat<f64>,
    ztwy: Mat<f64>,
}

// deviance, beta, sigma, u, ldL2, ldRX2, wrss, ussq, pwrss, row-major fixed information.
type LmmEvaluation = (
    f64,
    Vec<f64>,
    f64,
    Vec<f64>,
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
);

impl PreparedLmmResponse {
    fn validate_parameters(&self, theta: &[f64], reml: bool) -> Result<(), &'static str> {
        if theta.len() != self.design.n_theta || theta.iter().any(|v| !v.is_finite()) {
            return Err("theta must contain one finite value per covariance parameter");
        }
        if reml && self.design.x.nrows() <= self.design.x.ncols() {
            return Err("REML requires more observations than fixed effects");
        }
        Ok(())
    }

    fn deviance(&self, theta: &[f64], reml: bool) -> f64 {
        self.evaluate::<false>(theta, reml)
            .map_or(1e10, |evaluation| evaluation.0)
    }

    fn deviance_with_gradient(&self, theta: &[f64], reml: bool) -> (f64, Vec<f64>) {
        let design = &self.design;
        let (x, z, w) = (&design.x, &design.z, &design.weights);
        let (n, p, q) = (x.nrows(), x.ncols(), z.ncols());
        let (xtwx, ztwx) = (&design.xtwx, &design.ztwx);
        let (y_adj, xtwy, ztwy) = (&self.y_adj, &self.xtwy, &self.ztwy);
        let structures = &design.structures;
        let logdet_w = design.logdet_weights;
        let n_theta = design.n_theta;
        if q == 0 {
            return (self.deviance(theta, reml), Vec::new());
        }

        let theta = design.internal_theta(theta);
        let lambda_blocks = build_lambda_blocks(&theta, structures);

        let chol_v = match design.crossproducts.factor(&lambda_blocks) {
            Ok(c) => c,
            Err(_) => return (1e10, vec![0.0; n_theta]),
        };

        let logdet_v = chol_v.logdet();

        let derivatives = lambda_derivatives(structures);
        let traces = covariance_traces(
            &design.crossproducts.tiles,
            &lambda_blocks,
            &chol_v.selected_inverse(),
            &derivatives,
        );
        let factor = CovarianceFactor::from_blocks(lambda_blocks, structures);
        let cu = factor.transpose_apply(ztwy.as_ref());
        let cu_star = chol_v.solve_lower(&cu);

        let lambdat_ztwx = factor.transpose_apply(ztwx.as_ref());
        let rzx = chol_v.solve_lower(&lambdat_ztwx);

        let rzx_t_rzx = rzx.transpose() * &rzx;
        let xtvinvx = xtwx - &rzx_t_rzx;

        let chol_xtvinvx = match Llt::new(xtvinvx.as_ref(), Side::Lower) {
            Ok(c) => c,
            Err(_) => return (1e10, vec![0.0; n_theta]),
        };

        let l_xtvinvx = chol_xtvinvx.L();
        let logdet_xtvinvx: f64 = 2.0 * (0..p).map(|i| l_xtvinvx[(i, i)].ln()).sum::<f64>();

        let cu_star_rzx_beta_term = rzx.transpose() * &cu_star;
        let xty_adj = xtwy - &cu_star_rzx_beta_term;
        let beta = chol_xtvinvx.solve(&xty_adj);

        let mut resid = marginal_residual(x, &beta, y_adj);

        let zt_w_resid = compute_ztwy_sparse(z, w, &resid, q);
        let lambda_t_zt_resid = factor.transpose_apply(zt_w_resid.as_ref());
        let u_star = chol_v.solve(&lambda_t_zt_resid);

        let random = factor.apply(&u_star.col(0).to_owned());
        z.subtract_product(random.as_ref(), &mut resid);
        let wrss: f64 = (0..n).map(|i| w[i] * resid[i] * resid[i]).sum();
        let ussq: f64 = (0..q).map(|i| u_star[(i, 0)] * u_star[(i, 0)]).sum();
        let pwrss = wrss + ussq;
        let zt_w_conditional = compute_ztwy_sparse(z, w, &resid, q);

        let denom = if reml { n - p } else { n } as f64;
        let sigma2 = pwrss / denom;

        let mut dev =
            denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v - logdet_w;
        if reml {
            dev += logdet_xtvinvx;
        }

        let gradient_products = GradientCrossproducts {
            crossproducts: &design.crossproducts.tiles,
            factor: &factor,
        };
        let fixed_gradient = if reml && p > 0 {
            let projection = chol_v.solve(&lambdat_ztwx);
            let weighted = chol_xtvinvx
                .solve(&projection.transpose())
                .transpose()
                .to_owned();
            Some(FixedEffectGradient::new(
                &gradient_products,
                &projection,
                weighted,
                ztwx,
            ))
        } else {
            None
        };
        let mode_gradient = ModeGradient::new(&u_star, &zt_w_conditional, &factor, &chol_v);
        // Intercepts and slopes offer little reuse for two extra projections.
        // Keep their direct arithmetic, including its optimizer stopping behavior.
        let projected_mode = structures
            .iter()
            .any(|structure| structure.n_terms > 2)
            .then(|| mode_gradient.project(&zt_w_resid, &gradient_products));
        let mut gradient = Vec::with_capacity(n_theta);

        for (derivative, &d_logdet_v) in derivatives.iter().zip(&traces) {
            // Holding beta fixed is valid at its optimum. Both forms remain
            // valid for singular factors and avoid marginal quadratic forms.
            let d_pwrss = if let Some(projected) = &projected_mode {
                projected.derivative(derivative)
            } else {
                let mut dc = derivative.apply::<true>(&zt_w_resid);
                dc -= gradient_products.derivative_product(derivative, &u_star);
                mode_gradient.derivative(derivative, &dc)
            };

            let mut grad_k = d_logdet_v + denom / pwrss * d_pwrss;

            if let Some(fixed_gradient) = &fixed_gradient {
                grad_k += fixed_gradient.derivative(derivative);
            }

            gradient.push(grad_k);
        }

        match &design.order {
            Some(order) => (dev, StructureOrder::scatter(&gradient, &order.parameters)),
            None => (dev, gradient),
        }
    }

    fn evaluate<const ESTIMATES: bool>(&self, theta: &[f64], reml: bool) -> Option<LmmEvaluation> {
        let design = &self.design;
        let (x, z, w) = (&design.x, &design.z, &design.weights);
        let (n, p, q) = (x.nrows(), x.ncols(), z.ncols());
        let (xtwx, ztwx) = (&design.xtwx, &design.ztwx);
        let (y_adj, xtwy, ztwy) = (&self.y_adj, &self.xtwy, &self.ztwy);
        let structures = &design.structures;
        let logdet_w = design.logdet_weights;
        if q == 0 {
            let chol = Llt::new(xtwx.as_ref(), Side::Lower).ok()?;

            let beta = chol.solve(xtwy);

            let mut wrss = 0.0;
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..p {
                    pred += x[(i, j)] * beta[(j, 0)];
                }
                let resid = y_adj[i] - pred;
                wrss += w[i] * resid * resid;
            }

            let denom = if reml { n - p } else { n } as f64;
            let sigma2 = wrss / denom;

            let logdet_xtwx: f64 = if reml {
                let l = chol.L();
                2.0 * (0..p).map(|i| l[(i, i)].ln()).sum::<f64>()
            } else {
                0.0
            };

            let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) - logdet_w;
            if reml {
                dev += logdet_xtwx;
            }

            return Some((
                dev,
                if ESTIMATES {
                    (0..p).map(|i| beta[(i, 0)]).collect()
                } else {
                    Vec::new()
                },
                if ESTIMATES { sigma2.sqrt() } else { 0.0 },
                Vec::new(),
                0.0,
                logdet_xtwx,
                wrss,
                0.0,
                wrss,
                if ESTIMATES {
                    (0..p)
                        .flat_map(|i| (0..p).map(move |j| xtwx[(i, j)]))
                        .collect()
                } else {
                    Vec::new()
                },
            ));
        }

        let theta = design.internal_theta(theta);
        let lambda_blocks = build_lambda_blocks(&theta, structures);

        let chol_v = design.crossproducts.factor(&lambda_blocks).ok()?;

        let logdet_v = chol_v.logdet();

        let factor = CovarianceFactor::from_blocks(lambda_blocks, structures);
        let cu = factor.transpose_apply(ztwy.as_ref());
        let cu_star = chol_v.solve_lower(&cu);

        let lambdat_ztwx = factor.transpose_apply(ztwx.as_ref());
        let rzx = chol_v.solve_lower(&lambdat_ztwx);

        let rzx_t_rzx = rzx.transpose() * &rzx;
        let xtvinvx = xtwx - &rzx_t_rzx;

        let chol_xtvinvx = Llt::new(xtvinvx.as_ref(), Side::Lower).ok()?;

        let l_xtvinvx = chol_xtvinvx.L();
        let logdet_xtvinvx: f64 = 2.0 * (0..p).map(|i| l_xtvinvx[(i, i)].ln()).sum::<f64>();

        let cu_star_rzx_beta_term = rzx.transpose() * &cu_star;
        let xty_adj = xtwy - &cu_star_rzx_beta_term;
        let beta = chol_xtvinvx.solve(&xty_adj);

        let mut resid = marginal_residual(x, &beta, y_adj);

        let zt_w_resid = compute_ztwy_sparse(z, w, &resid, q);
        let lambda_t_zt_resid = factor.transpose_apply(zt_w_resid.as_ref());
        let u_star = chol_v.solve(&lambda_t_zt_resid);

        let random = factor.apply(&u_star.col(0).to_owned());
        z.subtract_product(random.as_ref(), &mut resid);
        // Use conditional residuals and the spherical penalty for both the
        // objective and final scale. Subtracting marginal quadratic forms loses
        // precision when the random effects explain nearly all of the response.
        let wrss = (0..n).map(|i| w[i] * resid[i] * resid[i]).sum::<f64>();
        let ussq = (0..q).map(|i| u_star[(i, 0)] * u_star[(i, 0)]).sum::<f64>();
        let pwrss = wrss + ussq;

        let denom = if reml { n - p } else { n } as f64;
        let sigma2 = pwrss / denom;

        let mut dev =
            denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v - logdet_w;
        if reml {
            dev += logdet_xtvinvx;
        }

        Some((
            dev,
            if ESTIMATES {
                (0..p).map(|i| beta[(i, 0)]).collect()
            } else {
                Vec::new()
            },
            if ESTIMATES { sigma2.sqrt() } else { 0.0 },
            if ESTIMATES {
                let random: Vec<f64> = (0..q).map(|i| random[i]).collect();
                match &design.order {
                    Some(order) => StructureOrder::scatter(&random, &order.columns),
                    None => random,
                }
            } else {
                Vec::new()
            },
            logdet_v,
            if reml { logdet_xtvinvx } else { 0.0 },
            wrss,
            ussq,
            pwrss,
            if ESTIMATES {
                let information = &xtvinvx;
                (0..p)
                    .flat_map(|i| (0..p).map(move |j| information[(i, j)]))
                    .collect()
            } else {
                Vec::new()
            },
        ))
    }
}

/// Owned design snapshot. Responses share its immutable crossproducts.
#[pyclass(frozen)]
pub struct LmmDesign {
    inner: Arc<PreparedLmmDesign>,
}

#[pymethods]
impl LmmDesign {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        x: numpy::PyArrayLike2<'_, f64>,
        z_data: numpy::PyArrayLike1<'_, f64>,
        z_indices: numpy::PyArrayLike1<'_, i64>,
        z_indptr: numpy::PyArrayLike1<'_, i64>,
        z_shape: (usize, usize),
        weights: numpy::PyArrayLike1<'_, f64>,
        offset: numpy::PyArrayLike1<'_, f64>,
        n_levels: Vec<usize>,
        n_terms: Vec<usize>,
        correlated: Vec<bool>,
    ) -> PyResult<Self> {
        let structures = random_effect_structures(n_levels, n_terms, correlated)?;
        // Check before the CSC parser computes the expected indptr length.
        if z_shape.1 == usize::MAX {
            return Err(PyValueError::new_err("random-effect dimensions overflow"));
        }
        let x_owned = {
            let view = x.as_array();
            Mat::from_fn(view.nrows(), view.ncols(), |i, j| view[[i, j]])
        };
        let z = csc_from_scipy(
            z_data.as_slice()?,
            z_indices.as_slice()?,
            z_indptr.as_slice()?,
            z_shape,
        )?;
        let weights_owned = weights.as_array().to_vec();
        let offset_owned = offset.as_array().to_vec();
        // Drop every NumPy borrow before preprocessing the owned snapshot, so
        // the caller may change input values or layouts while Python is detached.
        drop((x, z_data, z_indices, z_indptr, weights, offset));
        let inner = py
            .detach(|| PreparedLmmDesign::new(x_owned, z, weights_owned, offset_owned, structures))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// The factorization used for this design: "levels" when random-effect
    /// levels are independent, otherwise "blocked" or "sparse" for coupled designs.
    #[getter]
    fn engine(&self) -> &'static str {
        self.inner.engine()
    }

    /// Order of the dense Schur complement the blocked factorization forms
    /// after eliminating independent levels; zero for the other engines.
    #[getter]
    fn dense_dimension(&self) -> usize {
        self.inner.crossproducts.dense_dimension()
    }

    /// Caller structure indices in the order their levels are eliminated.
    #[getter]
    fn elimination_order(&self) -> Vec<usize> {
        match &self.inner.order {
            Some(order) => order.structures.clone(),
            None => (0..self.inner.structures.len()).collect(),
        }
    }

    fn with_response(&self, y: numpy::PyArrayLike1<'_, f64>) -> PyResult<LmmResponse> {
        let inner = self
            .inner
            .with_response(y.as_array())
            .map_err(PyValueError::new_err)?;
        Ok(LmmResponse { inner })
    }
}

#[pyclass(frozen)]
pub struct LmmResponse {
    inner: PreparedLmmResponse,
}

#[pymethods]
impl LmmResponse {
    /// Return the profiled deviance and its covariance-parameter gradient.
    /// Reuse validated design products and keep detached solve state local.
    #[pyo3(signature = (theta, reml = true))]
    fn deviance_with_gradient(
        &self,
        py: Python<'_>,
        theta: numpy::PyArrayLike1<'_, f64>,
        reml: bool,
    ) -> PyResult<(f64, Py<PyArray1<f64>>)> {
        let values = theta.as_slice()?;
        self.inner
            .validate_parameters(values, reml)
            .map_err(PyValueError::new_err)?;
        let values = values.to_vec();
        // Drop the NumPy guard before detaching: the caller can then change
        // the parameter array's values or layout without affecting this solve.
        drop(theta);
        let (deviance, gradient) = py.detach(|| self.inner.deviance_with_gradient(&values, reml));
        Ok((deviance, PyArray1::from_vec(py, gradient).into()))
    }

    /// Return estimates and likelihood components, or None on factorization failure.
    /// Final residual scale uses conditional residuals plus the spherical penalty.
    #[pyo3(signature = (theta, reml = true))]
    fn evaluate(
        &self,
        py: Python<'_>,
        theta: numpy::PyArrayLike1<'_, f64>,
        reml: bool,
    ) -> PyResult<Option<LmmEvaluation>> {
        let values = theta.as_slice()?;
        self.inner
            .validate_parameters(values, reml)
            .map_err(PyValueError::new_err)?;
        let values = values.to_vec();
        drop(theta);
        Ok(py.detach(|| self.inner.evaluate::<true>(&values, reml)))
    }

    #[pyo3(signature = (theta, reml = true))]
    fn deviance(
        &self,
        py: Python<'_>,
        theta: numpy::PyArrayLike1<'_, f64>,
        reml: bool,
    ) -> PyResult<f64> {
        let theta_values = theta.as_slice()?;
        self.inner
            .validate_parameters(theta_values, reml)
            .map_err(PyValueError::new_err)?;
        // Release the NumPy borrow before the caller can change its array's
        // values or layout. The detached solve only reads owned native data.
        let theta_values = theta_values.to_vec();
        drop(theta);
        Ok(py.detach(|| self.inner.deviance(&theta_values, reml)))
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;
    use numpy::ndarray::ArrayView1;

    fn structure(n_levels: usize, n_terms: usize, correlated: bool) -> RandomEffectStructure {
        RandomEffectStructure {
            n_levels,
            n_terms,
            correlated,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    /// Tile a dense crossproduct, optionally forcing the sparse factorization.
    fn tiled(
        crossproduct: &Mat<f64>,
        structures: &[RandomEffectStructure],
        sparse: bool,
    ) -> DesignCrossproducts {
        let tiles = LevelTiles::lower_from_entries(&structure_blocks(structures), |i, j| {
            crossproduct[(i, j)]
        });
        let mut crossproducts = DesignCrossproducts::new(tiles);
        crossproducts.sparse = sparse
            .then(|| SparseCholesky::new(&crossproducts.tiles, usize::MAX))
            .flatten();
        crossproducts
    }

    #[test]
    fn independent_design_storage_scales_with_level_width() {
        let levels = if cfg!(miri) { 8 } else { 2048 };
        let z = CscMatrix::try_from_usize(&[], &[], &vec![0; levels + 1], (4, levels)).unwrap();
        let design = PreparedLmmDesign::new(
            Mat::full(4, 1, 1.0),
            z,
            vec![1.0; 4],
            vec![0.0; 4],
            vec![structure(levels, 1, true)],
        )
        .unwrap();
        assert_eq!(design.engine(), "levels");
        assert_eq!(design.crossproducts.tiles.n_values(), levels);
        assert_eq!(design.crossproducts.dense_dimension(), 0);
    }

    #[test]
    fn crossproducts_keep_level_blocks_and_tiny_couplings() {
        // The second structure has more columns, so coupled designs reorder.
        let structures = vec![structure(1, 2, true), structure(5, 1, false)];
        let q = 7;
        // Row j loads column j; each extra row loads a pair of columns.
        let design = |pairs: &[(usize, usize, f64)]| {
            let n = q + pairs.len();
            let mut columns: Vec<Vec<(usize, f64)>> =
                (0..q).map(|column| vec![(column, 1.0)]).collect();
            for (index, &(left, right, value)) in pairs.iter().enumerate() {
                columns[left].push((q + index, 1.0));
                columns[right].push((q + index, value));
            }
            let (mut values, mut rows, mut offsets) = (Vec::new(), Vec::new(), vec![0]);
            for (row, value) in columns.into_iter().flat_map(|column| {
                offsets.push(offsets.last().unwrap() + column.len());
                column
            }) {
                rows.push(row);
                values.push(value);
            }
            let z = CscMatrix::try_from_usize(&values, &rows, &offsets, (n, q)).unwrap();
            let x = Mat::full(n, 1, 1.0);
            PreparedLmmDesign::new(x, z, vec![1.0; n], vec![0.0; n], structures.clone()).unwrap()
        };
        let independent = design(&[(0, 1, 0.5)]);
        assert_eq!(independent.engine(), "levels");
        assert!(independent.order.is_none());
        let tiles = &independent.crossproducts.tiles;
        assert_eq!(tiles.tile(0), [2.0, 0.5, 0.5, 1.25]);
        assert_eq!(tiles.n_values(), 4 + 5);
        for pair in [(0, 2), (2, 0), (0, 4), (4, 0)] {
            // A tiny coupling across levels keeps its own tile after reordering.
            let design = design(&[(0, 1, 0.5), (pair.0, pair.1, 1e-300)]);
            assert_eq!(design.engine(), "blocked");
            assert_eq!(
                design.order.as_ref().map(|order| order.structures.clone()),
                Some(vec![1, 0])
            );
            let tiles = &design.crossproducts.tiles;
            let coupling = (0..tiles.n_levels())
                .flat_map(|level| tiles.column(level).skip(1))
                .map(|tile| tiles.tile(tile))
                .collect::<Vec<_>>();
            assert_eq!(coupling.len(), 1);
            assert!(coupling[0].contains(&1e-300));
        }
    }

    #[test]
    fn shared_mode_adjoint_retains_nonstationary_corrections() {
        let structures = [
            structure(0, 1, true),
            structure(2, 3, true),
            structure(3, 2, false),
        ];
        let q = 12;
        let design = Mat::from_fn(17, q, |i, j| ((i + 3 * j) % 11) as f64 / 8.0 - 0.5);
        let coupled = design.transpose() * &design;
        let level = |i| if i < 6 { i / 3 } else { 2 + (i - 6) / 2 };
        // Deliberately avoid stationarity: omitting the mode correction must fail.
        let mode = Mat::from_fn(q, 1, |i, _| (i % 5) as f64 / 4.0 - 0.5);
        let conditional = Mat::from_fn(q, 1, |i, _| (i % 7) as f64 / 8.0 + 0.25);
        let marginal = Mat::from_fn(q, 1, |i, _| (i % 3) as f64 / 2.0 - 0.25);
        for (independent, sparse) in [(false, false), (false, true), (true, false)] {
            let crossproduct = Mat::from_fn(q, q, |i, j| {
                if independent && level(i) != level(j) {
                    0.0
                } else {
                    coupled[(i, j)]
                }
            });
            let crossproducts = tiled(&crossproduct, &structures, sparse);
            for theta in [
                [0.4, 0.8, -0.1, 0.7, 0.05, -0.2, 0.6, 0.3, 0.9],
                [0.4, 0.0, -0.1, 0.7, 0.05, -0.2, 0.0, 0.3, 0.0],
                [0.0; 9],
            ] {
                let blocks = build_lambda_blocks(&theta, &structures);
                let chol = crossproducts.factor(&blocks).unwrap();
                let factor = CovarianceFactor::from_blocks(blocks, &structures);
                let lambda = factor.to_dense();
                let information =
                    Mat::<f64>::identity(q, q) + lambda.transpose() * &crossproduct * &lambda;
                let dense_chol = Llt::new(information.as_ref(), Side::Lower).unwrap();
                let shared = ModeGradient::new(&mode, &conditional, &factor, &chol);
                let products = GradientCrossproducts {
                    crossproducts: &crossproducts.tiles,
                    factor: &factor,
                };
                let projected = shared.project(&marginal, &products);
                let mut max_correction: f64 = 0.0;
                for derivative in lambda_derivatives(&structures) {
                    let entry = derivative.apply::<false>(&Mat::identity(q, q));
                    let dv = entry.transpose() * &crossproduct * &lambda
                        + lambda.transpose() * &crossproduct * &entry;
                    let rhs = entry.transpose() * &marginal - dv * &mode;
                    let du = dense_chol.solve(&rhs);
                    let random_derivative = &entry * &mode + &lambda * &du;
                    let expected = 2.0
                        * (0..q)
                            .map(|i| {
                                mode[(i, 0)] * du[(i, 0)]
                                    - conditional[(i, 0)] * random_derivative[(i, 0)]
                            })
                            .sum::<f64>();
                    let actual = shared.derivative(&derivative, &rhs);
                    assert!((actual - expected).abs() < 1e-11 * expected.abs().max(1.0));
                    let projected_actual = projected.derivative(&derivative);
                    assert!((projected_actual - expected).abs() < 1e-11 * expected.abs().max(1.0));
                    let stationary = -2.0 * derivative.bilinear(conditional.col(0), mode.col(0));
                    max_correction = max_correction.max((actual - stationary).abs());
                }
                assert!(max_correction > 0.1);
            }
        }
    }

    #[test]
    fn shared_fixed_effect_products_match_full_information_derivatives() {
        let fixed: &[usize] = if cfg!(miri) {
            &[0, 1, 3]
        } else {
            &[0, 1, 3, 8, 17]
        };
        let structures = [structure(2, 3, true), structure(3, 2, false)];
        let q = 12;
        let design = Mat::from_fn(15, q, |i, j| ((2 * i + 3 * j + 1) % 7) as f64 / 8.0 - 0.3);
        let crossproduct = design.transpose() * &design;
        let crossproducts = tiled(&crossproduct, &structures, false);
        let theta = [0.8, -0.2, 0.6, 0.1, 0.3, 0.9, 0.7, 0.4];
        let factor = CovarianceFactor::new(&theta, &structures);
        let product = &crossproduct * factor.to_dense();
        let products = GradientCrossproducts {
            crossproducts: &crossproducts.tiles,
            factor: &factor,
        };
        for &p in fixed {
            let design = Mat::from_fn(p + 2, p, |i, j| ((i + 2 * j) % 5) as f64 / 8.0);
            let information = Mat::<f64>::identity(p, p) + design.transpose() * &design;
            let chol = Llt::new(information.as_ref(), Side::Lower).unwrap();
            let projection = Mat::from_fn(q, p, |i, j| ((3 * i + 5 * j) % 11) as f64 / 8.0 - 0.5);
            let ztwx = Mat::from_fn(q, p, |i, j| ((i + 2 * j + 1) % 13) as f64 / 16.0 - 0.25);
            let applied = products.apply(projection.as_ref());
            let expected_applied = &product * &projection;
            for j in 0..p {
                for i in 0..q {
                    assert_close(applied[(i, j)], expected_applied[(i, j)]);
                }
            }
            let weighted = chol.solve(&projection.transpose()).transpose().to_owned();
            let shared = FixedEffectGradient::new(&products, &projection, weighted, &ztwx);
            for derivative in lambda_derivatives(&structures) {
                let derivative_b = derivative.apply::<true>(&ztwx);
                let derivative_v = derivative.crossproduct_derivative(&product);
                let derivative_information = projection.transpose() * derivative_v * &projection
                    - derivative_b.transpose() * &projection
                    - projection.transpose() * &derivative_b;
                let solved = chol.solve(&derivative_information);
                let expected: f64 = (0..p).map(|i| solved[(i, i)]).sum();
                assert_close(shared.derivative(&derivative), expected);
            }
        }
    }

    #[test]
    fn tile_traces_and_products_match_dense_derivatives() {
        // Narrow tiles use scalar kernels; a 17-term structure uses dense ones.
        for structures in [
            vec![
                structure(3, 2, true),
                structure(2, 3, false),
                structure(4, 1, true),
            ],
            vec![structure(2, 17, true), structure(3, 1, true)],
        ] {
            let level: Vec<usize> = structures
                .iter()
                .scan(0, |first, structure| {
                    let levels = *first..*first + structure.n_levels;
                    *first += structure.n_levels;
                    Some(levels.flat_map(|level| std::iter::repeat_n(level, structure.n_terms)))
                })
                .flatten()
                .collect();
            let q = level.len();
            let design = Mat::from_fn(23, q, |i, j| ((5 * i + 3 * j + 2) % 13) as f64 / 8.0 - 0.7);
            let coupled = design.transpose() * &design;
            let n_theta: usize = structures.iter().map(structure_parameters).sum();
            let theta: Vec<f64> = (0..n_theta)
                .map(|i| [0.9, -0.3, 0.5, 0.7, 0.2, 1.1, 0.6][i % 7] / (1 + i / 7) as f64)
                .collect();
            let blocks = build_lambda_blocks(&theta, &structures);
            let factor = CovarianceFactor::from_blocks(blocks.clone(), &structures);
            let lambda = factor.to_dense();
            let rhs = Mat::from_fn(q, 3, |i, j| ((i + 4 * j) % 9) as f64 / 4.0 - 1.0);
            for (independent, sparse) in [(true, false), (false, false), (false, true)] {
                let crossproduct = Mat::from_fn(q, q, |i, j| {
                    if independent && level[i] != level[j] {
                        0.0
                    } else {
                        coupled[(i, j)]
                    }
                });
                let crossproducts = tiled(&crossproduct, &structures, sparse);
                assert_eq!(crossproducts.tiles.block_diagonal(), independent);
                assert_eq!(crossproducts.sparse.is_some(), sparse);
                let chol = crossproducts.factor(&blocks).unwrap();
                let information =
                    Mat::<f64>::identity(q, q) + lambda.transpose() * &crossproduct * &lambda;
                let inverse = Llt::new(information.as_ref(), Side::Lower)
                    .unwrap()
                    .solve(Mat::<f64>::identity(q, q));
                let selected = chol.selected_inverse();
                let derivatives = lambda_derivatives(&structures);
                let traces =
                    covariance_traces(&crossproducts.tiles, &blocks, &selected, &derivatives);
                let products = GradientCrossproducts {
                    crossproducts: &crossproducts.tiles,
                    factor: &factor,
                };
                for (derivative, &trace) in derivatives.iter().zip(&traces) {
                    let entry = derivative.apply::<false>(&Mat::identity(q, q));
                    let dv = entry.transpose() * &crossproduct * &lambda
                        + lambda.transpose() * &crossproduct * &entry;
                    let expected = (0..q)
                        .map(|i| (0..q).map(|j| inverse[(i, j)] * dv[(j, i)]).sum::<f64>())
                        .sum::<f64>();
                    assert_close(trace, expected);
                    let product = products.derivative_product(derivative, &rhs);
                    let expected = &dv * &rhs;
                    for j in 0..rhs.ncols() {
                        for i in 0..q {
                            assert_close(product[(i, j)], expected[(i, j)]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn coordinate_derivatives_match_full_matrix_products() {
        let structures = [
            structure(0, 1, true),
            structure(2, 3, true),
            structure(2, 2, false),
            structure(1, 1, true),
        ];
        let coordinates = [
            (0, 0, 0, 1, 0, 0),
            (1, 0, 2, 3, 0, 0),
            (1, 0, 2, 3, 1, 0),
            (1, 0, 2, 3, 1, 1),
            (1, 0, 2, 3, 2, 0),
            (1, 0, 2, 3, 2, 1),
            (1, 0, 2, 3, 2, 2),
            (2, 6, 2, 2, 0, 0),
            (2, 6, 2, 2, 1, 1),
            (3, 10, 1, 1, 0, 0),
        ];
        let derivatives = lambda_derivatives(&structures);
        assert_eq!(derivatives.len(), coordinates.len());
        for (derivative, (structure, offset, levels, width, row, column)) in
            derivatives.iter().zip(coordinates)
        {
            assert_eq!(
                (
                    derivative.structure,
                    derivative.offset,
                    derivative.n_levels,
                    derivative.n_terms,
                    derivative.row,
                    derivative.column
                ),
                (structure, offset, levels, width, row, column),
            );
            let mut dense = Mat::<f64>::zeros(11, 11);
            for level in 0..levels {
                dense[(
                    offset + level * width + row,
                    offset + level * width + column,
                )] = 1.0;
            }
            for ncols in [0, 1, 4] {
                let matrix = Mat::from_fn(11, ncols, |i, j| (i + 3 * j) as f64 - 5.0);
                assert_eq!(derivative.apply::<false>(&matrix), &dense * &matrix);
                assert_eq!(
                    derivative.apply::<true>(&matrix),
                    dense.transpose() * &matrix
                );
            }
            let left = Mat::from_fn(11, 1, |i, _| (i % 3) as f64 - 1.0);
            let right = Mat::from_fn(11, 1, |i, _| (i % 5) as f64 - 2.0);
            let expected = left.transpose() * &dense * &right;
            assert_eq!(
                derivative.bilinear(left.col(0), right.col(0)),
                expected[(0, 0)]
            );
            let product = Mat::from_fn(11, 11, |i, j| (i + 3 * j) as f64 - 5.0);
            let term = dense.transpose() * &product;
            assert_eq!(
                derivative.crossproduct_derivative(&product),
                &term + term.transpose()
            );
        }
    }

    /// Two-level nesting: each of nine fine levels lies in one of three coarse
    /// levels. The fine structure has a correlated slope.
    fn nested_design(fine_first: bool) -> (Arc<PreparedLmmDesign>, Vec<f64>) {
        let n = 45;
        let fine = |row: usize| row % 9;
        let covariate = |row: usize| ((7 * row) % 11) as f64 / 5.0 - 1.0;
        let (mut values, mut rows, mut offsets) = (Vec::new(), Vec::new(), vec![0]);
        let mut fine_columns = Vec::new();
        for level in 0..9 {
            for term in 0..2 {
                let column: Vec<_> = (0..n)
                    .filter(|&row| fine(row) == level)
                    .map(|row| (row, if term == 0 { 1.0 } else { covariate(row) }))
                    .collect();
                fine_columns.push(column);
            }
        }
        let coarse_columns: Vec<Vec<_>> = (0..3)
            .map(|level| {
                (0..n)
                    .filter(|&row| fine(row) / 3 == level)
                    .map(|row| (row, 1.0))
                    .collect()
            })
            .collect();
        let columns = if fine_first {
            [fine_columns, coarse_columns].concat()
        } else {
            [coarse_columns, fine_columns].concat()
        };
        for column in columns {
            for (row, value) in column {
                rows.push(row);
                values.push(value);
            }
            offsets.push(rows.len());
        }
        let z = CscMatrix::try_from_usize(&values, &rows, &offsets, (n, 21)).unwrap();
        let x = Mat::from_fn(
            n,
            2,
            |row, column| if column == 0 { 1.0 } else { covariate(row) },
        );
        let (fine, coarse) = (structure(9, 2, true), structure(3, 1, true));
        let structures = if fine_first {
            vec![fine, coarse]
        } else {
            vec![coarse, fine]
        };
        let weights = (0..n).map(|row| 0.5 + (row % 4) as f64 / 4.0).collect();
        let design = PreparedLmmDesign::new(x, z, weights, vec![0.1; n], structures).unwrap();
        let y = (0..n)
            .map(|row| ((row * 13) % 17) as f64 / 3.0 + covariate(row))
            .collect();
        (Arc::new(design), y)
    }

    #[test]
    fn elimination_order_preserves_caller_parameters_and_effects() {
        let (formula_order, y) = nested_design(false);
        let (sorted, _) = nested_design(true);
        assert_eq!(
            formula_order
                .order
                .as_ref()
                .map(|order| order.structures.clone()),
            Some(vec![1, 0])
        );
        assert!(sorted.order.is_none());
        // The coarse factor couples each fine level to one parent: no fill.
        assert_eq!(formula_order.crossproducts.leading, 9);
        let first = formula_order.with_response(ArrayView1::from(&y)).unwrap();
        let second = sorted.with_response(ArrayView1::from(&y)).unwrap();
        // Caller theta lists the coarse intercept first, then the fine factor.
        let (coarse, fine) = ([0.7], [0.9, -0.3, 0.4]);
        let caller = [coarse.as_slice(), &fine].concat();
        let internal = [fine.as_slice(), &coarse].concat();
        for reml in [false, true] {
            let expected = second.evaluate::<true>(&internal, reml).unwrap();
            let actual = first.evaluate::<true>(&caller, reml).unwrap();
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1, expected.1);
            // Effects return in caller column order: coarse levels first.
            assert_eq!(actual.3[..3], expected.3[18..]);
            assert_eq!(actual.3[3..], expected.3[..18]);
            let (deviance, gradient) = first.deviance_with_gradient(&caller, reml);
            let (expected_deviance, expected_gradient) =
                second.deviance_with_gradient(&internal, reml);
            assert_eq!(deviance, expected_deviance);
            assert_eq!(deviance, actual.0);
            assert_eq!(gradient[0], expected_gradient[3]);
            assert_eq!(gradient[1..], expected_gradient[..3]);
        }
    }

    fn intercept_design() -> Arc<PreparedLmmDesign> {
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let z = CscMatrix::try_from_i64(&[], &[], &[0], (4, 0)).unwrap();
        Arc::new(PreparedLmmDesign::new(x, z, vec![1.0; 4], vec![0.0; 4], vec![]).unwrap())
    }

    #[test]
    fn prepared_fixed_likelihood_matches_closed_form() {
        let design = intercept_design();
        let response = design
            .with_response(ArrayView1::from(&[1.0, 2.0, 4.0, 7.0]))
            .unwrap();
        for reml in [false, true] {
            response.validate_parameters(&[], reml).unwrap();
            let df = if reml { 3.0 } else { 4.0 };
            let expected = df * (1.0 + (2.0 * std::f64::consts::PI * 21.0 / df).ln())
                + if reml { 4.0f64.ln() } else { 0.0 };
            assert!((response.deviance(&[], reml) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn responses_share_design_without_sharing_response_products() {
        let design = intercept_design();
        let first = design
            .with_response(ArrayView1::from(&[1.0, 2.0, 4.0, 7.0]))
            .unwrap();
        let second = design
            .with_response(ArrayView1::from(&[2.0, 4.0, 8.0, 14.0]))
            .unwrap();
        assert!(Arc::ptr_eq(&first.design, &second.design));
        let initial = first.deviance(&[], false);
        assert!((second.deviance(&[], false) - initial - 4.0 * 4.0f64.ln()).abs() < 1e-12);
        drop(design);
        assert!((first.deviance(&[], false) - initial).abs() <= 1e-12 * initial.abs().max(1.0));
        assert!(first.validate_parameters(&[1.0], false).is_err());
        assert!(
            second
                .design
                .with_response(ArrayView1::from(&[1.0]))
                .is_err()
        );
    }
}

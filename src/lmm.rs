use std::sync::Arc;

use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, Side};
use numpy::PyArray1;
use numpy::ndarray::{ArrayView1, ArrayView2};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::blocked_chol::{BlockedCholesky, BlockedMatrix};
pub use crate::covariance::RandomEffectStructure;
use crate::covariance::{CovarianceFactor, build_lambda_blocks};
use crate::csc::CscMatrix;
use crate::linalg::LinalgError;

fn validate_prior_weights(weights: ArrayView1<'_, f64>, n: usize) -> PyResult<(Vec<f64>, f64)> {
    if weights.len() != n {
        return Err(PyValueError::new_err(format!(
            "weights has length {}, expected {n}",
            weights.len()
        )));
    }

    let mut values = Vec::with_capacity(n);
    let mut logdet = 0.0;
    for &weight in weights {
        if !weight.is_finite() {
            return Err(PyValueError::new_err(
                "weights must contain only finite values",
            ));
        }
        if weight <= 0.0 {
            return Err(PyValueError::new_err("weights must be strictly positive"));
        }
        values.push(weight);
        logdet += weight.ln();
    }
    Ok((values, logdet))
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
    offset: usize,
    n_levels: usize,
    n_terms: usize,
    row: usize,
    column: usize,
}

impl LambdaDerivative {
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
    for structure in structures {
        for row in 0..structure.n_terms {
            let columns = if structure.correlated {
                0..row + 1
            } else {
                row..row + 1
            };
            for column in columns {
                derivatives.push(LambdaDerivative {
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

fn compute_ztwz_sparse(z: &CscMatrix, weights: &[f64]) -> Mat<f64> {
    z.weighted_crossproduct(weights)
}

fn mat_from_flat_array(data: &[f64], q: usize) -> Mat<f64> {
    Mat::from_fn(q, q, |i, j| data[i * q + j])
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
        faer::linalg::matmul::matmul(
            faer::ColMut::from_slice_mut(&mut residual).as_mat_mut(),
            faer::Accum::Add,
            x,
            beta,
            -1.0,
            faer::get_global_parallelism(),
        );
    }
    residual
}

fn subtract_random_prediction(residual: &mut [f64], z: &CscMatrix, random: &faer::Col<f64>) {
    for j in 0..z.ncols() {
        for index in z.col_offsets()[j]..z.col_offsets()[j + 1] {
            residual[z.row_indices()[index]] -= z.values()[index] * random[j];
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
    ztwz: Mat<f64>,
    independent_levels: Vec<bool>,
    structures: Vec<RandomEffectStructure>,
    n_theta: usize,
}

impl PreparedLmmDesign {
    fn new(
        x: Mat<f64>,
        z: CscMatrix,
        weights: Vec<f64>,
        offset: Vec<f64>,
        structures: Vec<RandomEffectStructure>,
        cached_ztwz: Option<&[f64]>,
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
        let sqrt_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
        let logdet_weights = weights.iter().map(|w| w.ln()).sum();
        let wx = Mat::from_fn(n, p, |i, j| sqrt_weights[i] * x[(i, j)]);
        let xtwx = wx.transpose() * &wx;
        let ztwx = compute_ztwx_sparse(&z, &weights, &x, q, p);
        let ztwz = if let Some(values) = cached_ztwz {
            if q.checked_mul(q) != Some(values.len()) || values.iter().any(|v| !v.is_finite()) {
                return Err("cached Z'WZ must contain q * q finite values");
            }
            mat_from_flat_array(values, q)
        } else {
            compute_ztwz_sparse(&z, &weights)
        };
        let independent_levels = BlockedMatrix::independent_levels(&ztwz, &structures);
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
            ztwz,
            independent_levels,
            structures,
            n_theta,
        })
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

    fn evaluate<const ESTIMATES: bool>(&self, theta: &[f64], reml: bool) -> Option<LmmEvaluation> {
        let design = &self.design;
        let (x, z, w) = (&design.x, &design.z, &design.weights);
        let (n, p, q) = (x.nrows(), x.ncols(), z.ncols());
        let (xtwx, ztwx, ztwz) = (&design.xtwx, &design.ztwx, &design.ztwz);
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

        let lambda_blocks = build_lambda_blocks(theta, structures);

        let blocked_v = BlockedMatrix::from_lambda_ztwz_with_pattern(
            ztwz,
            &lambda_blocks,
            structures,
            true,
            &design.independent_levels,
        );
        let chol_v = BlockedCholesky::factor(&blocked_v).ok()?;

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
        subtract_random_prediction(&mut resid, z, &random);
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
                (0..q).map(|i| random[i]).collect()
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
        if n_levels.len() != n_terms.len() || n_levels.len() != correlated.len() {
            return Err(PyValueError::new_err(
                "random-effect structure arrays must have equal lengths",
            ));
        }
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
        let structures = n_levels
            .into_iter()
            .zip(n_terms)
            .zip(correlated)
            .map(|((n_levels, n_terms), correlated)| RandomEffectStructure {
                n_levels,
                n_terms,
                correlated,
            })
            .collect();
        let weights_owned = weights.as_array().to_vec();
        let offset_owned = offset.as_array().to_vec();
        // Drop every NumPy borrow before preprocessing the owned snapshot, so
        // the caller may change input values or layouts while Python is detached.
        drop((x, z_data, z_indices, z_indptr, weights, offset));
        let inner = py
            .detach(|| {
                PreparedLmmDesign::new(x_owned, z, weights_owned, offset_owned, structures, None)
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
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

#[allow(clippy::too_many_arguments)]
pub fn profiled_deviance_impl(
    theta: &[f64],
    y: ArrayView1<'_, f64>,
    x_data: ArrayView2<'_, f64>,
    z_data: &[f64],
    z_indices: &[i64],
    z_indptr: &[i64],
    z_shape: (usize, usize),
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    structures: &[RandomEffectStructure],
    reml: bool,
    ztwz_cache: Option<&[f64]>,
) -> PyResult<f64> {
    let x = Mat::from_fn(x_data.nrows(), x_data.ncols(), |i, j| x_data[[i, j]]);
    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
    let design = Arc::new(
        PreparedLmmDesign::new(
            x,
            z,
            weights.to_vec(),
            offset.to_vec(),
            structures.to_vec(),
            ztwz_cache,
        )
        .map_err(PyValueError::new_err)?,
    );
    let response = design.with_response(y).map_err(PyValueError::new_err)?;
    response
        .validate_parameters(theta, reml)
        .map_err(PyValueError::new_err)?;
    Ok(response.deviance(theta, reml))
}

#[allow(clippy::too_many_arguments)]
pub fn profiled_deviance_with_gradient_impl(
    theta: &[f64],
    y: ArrayView1<'_, f64>,
    x_data: ArrayView2<'_, f64>,
    z_data: &[f64],
    z_indices: &[i64],
    z_indptr: &[i64],
    z_shape: (usize, usize),
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    structures: &[RandomEffectStructure],
    reml: bool,
    ztwz_cache: Option<&[f64]>,
) -> PyResult<(f64, Vec<f64>)> {
    let n = y.len();
    let p = x_data.ncols();
    let q = z_shape.1;
    let n_theta = theta.len();

    let y_adj: Vec<f64> = y
        .iter()
        .zip(offset.iter())
        .map(|(yi, oi)| yi - oi)
        .collect();
    let (w, logdet_w) = validate_prior_weights(weights, n)?;
    let sqrt_w: Vec<f64> = w.iter().map(|wi| wi.sqrt()).collect();

    let x = Mat::from_fn(n, p, |i, j| x_data[[i, j]]);

    if q == 0 {
        let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);
        let wy = Mat::from_fn(n, 1, |i, _| sqrt_w[i] * y_adj[i]);

        let xtwx = wx.transpose() * &wx;
        let xtwy = wx.transpose() * &wy;

        let chol = match Llt::new(xtwx.as_ref(), Side::Lower) {
            Ok(c) => c,
            Err(_) => return Ok((1e10, vec![0.0; n_theta])),
        };

        let beta = chol.solve(&xtwy);

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

        return Ok((dev, vec![0.0; n_theta]));
    }

    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
    let lambda_blocks = build_lambda_blocks(theta, structures);

    let ztwz = if let Some(cached_data) = ztwz_cache {
        mat_from_flat_array(cached_data, q)
    } else {
        compute_ztwz_sparse(&z, &w)
    };

    let blocked_v = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, structures, true);
    let chol_v = match BlockedCholesky::factor(&blocked_v) {
        Ok(c) => c,
        Err(_) => return Ok((1e10, vec![0.0; n_theta])),
    };

    let logdet_v = chol_v.logdet();

    let ztwy = compute_ztwy_sparse(&z, &w, &y_adj, q);
    let factor = CovarianceFactor::from_blocks(lambda_blocks, structures);
    let cu = factor.transpose_apply(ztwy.as_ref());
    let cu_star = chol_v.solve_lower(&cu);

    let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);

    let ztwx = compute_ztwx_sparse(&z, &w, &x, q, p);
    let lambdat_ztwx = factor.transpose_apply(ztwx.as_ref());
    let rzx = chol_v.solve_lower(&lambdat_ztwx);

    let xtwx = wx.transpose() * &wx;
    let mut xtwy = Mat::zeros(p, 1);
    for i in 0..p {
        let mut sum = 0.0;
        for row in 0..n {
            sum += wx[(row, i)] * sqrt_w[row] * y_adj[row];
        }
        xtwy[(i, 0)] = sum;
    }

    let rzx_t_rzx = rzx.transpose() * &rzx;
    let xtvinvx = &xtwx - &rzx_t_rzx;

    let chol_xtvinvx = match Llt::new(xtvinvx.as_ref(), Side::Lower) {
        Ok(c) => c,
        Err(_) => return Ok((1e10, vec![0.0; n_theta])),
    };

    let l_xtvinvx = chol_xtvinvx.L();
    let logdet_xtvinvx: f64 = 2.0 * (0..p).map(|i| l_xtvinvx[(i, i)].ln()).sum::<f64>();

    let cu_star_rzx_beta_term = rzx.transpose() * &cu_star;
    let xty_adj = &xtwy - &cu_star_rzx_beta_term;
    let beta = chol_xtvinvx.solve(&xty_adj);

    let mut resid = marginal_residual(&x, &beta, &y_adj);

    let zt_w_resid = compute_ztwy_sparse(&z, &w, &resid, q);
    let lambda_t_zt_resid = factor.transpose_apply(zt_w_resid.as_ref());
    let u_star = chol_v.solve(&lambda_t_zt_resid);

    let random = factor.apply(&u_star.col(0).to_owned());
    subtract_random_prediction(&mut resid, &z, &random);
    let wrss: f64 = (0..n).map(|i| w[i] * resid[i] * resid[i]).sum();
    let ussq: f64 = (0..q).map(|i| u_star[(i, 0)] * u_star[(i, 0)]).sum();
    let pwrss = wrss + ussq;
    let zt_w_conditional = compute_ztwy_sparse(&z, &w, &resid, q);

    let denom = if reml { n - p } else { n } as f64;
    let sigma2 = pwrss / denom;

    let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v - logdet_w;
    if reml {
        dev += logdet_xtvinvx;
    }

    let v_inv = chol_v.solve(&Mat::<f64>::identity(q, q));
    let v_inv_b = chol_v.solve(&lambdat_ztwx);
    let xtvinvx_inv = if reml {
        Some(chol_xtvinvx.solve(&Mat::<f64>::identity(p, p)))
    } else {
        None
    };

    let ztwz_lambda = factor.transpose_apply(ztwz.as_ref()).transpose().to_owned();
    let mut gradient = Vec::with_capacity(n_theta);

    for derivative in lambda_derivatives(structures) {
        let dv = derivative.crossproduct_derivative(&ztwz_lambda);

        let mut d_logdet_v = 0.0;
        for i in 0..q {
            for j in 0..q {
                d_logdet_v += v_inv[(i, j)] * dv[(j, i)];
            }
        }

        let dc = derivative.apply::<true>(&zt_w_resid);
        let dv_u = &dv * &u_star;
        let d_u = chol_v.solve(&(&dc - &dv_u));
        let lambda_d_u = factor.apply(&d_u.col(0).to_owned());
        let d_lambda_u = derivative.apply::<false>(&u_star);
        // Differentiate the conditional residual norm and spherical penalty
        // through the mode solve. Holding beta fixed is valid at its optimum.
        // This avoids cancellation between large marginal quadratic forms,
        // and remains valid when a covariance factor is singular.
        let d_pwrss = 2.0
            * (0..q)
                .map(|i| {
                    u_star[(i, 0)] * d_u[(i, 0)]
                        - zt_w_conditional[(i, 0)] * (d_lambda_u[(i, 0)] + lambda_d_u[i])
                })
                .sum::<f64>();

        let mut grad_k = d_logdet_v + denom / pwrss * d_pwrss;

        if let Some(xtvinvx_inv) = &xtvinvx_inv {
            let db = derivative.apply::<true>(&ztwx);
            let dv_v_inv_b = &dv * &v_inv_b;
            let mut d_logdet_xtvinvx = 0.0;

            for i in 0..p {
                for j in 0..p {
                    let mut dm_ji = 0.0;
                    for r in 0..q {
                        dm_ji -= db[(r, j)] * v_inv_b[(r, i)];
                        dm_ji -= v_inv_b[(r, j)] * db[(r, i)];
                        dm_ji += v_inv_b[(r, j)] * dv_v_inv_b[(r, i)];
                    }
                    d_logdet_xtvinvx += xtvinvx_inv[(i, j)] * dm_ji;
                }
            }

            grad_k += d_logdet_xtvinvx;
        }

        gradient.push(grad_k);
    }

    Ok((dev, gradient))
}

#[pyfunction]
pub fn compute_ztwz<'py>(
    py: Python<'py>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let z = csc_from_scipy(
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
    )?;
    let (w, _) = validate_prior_weights(weights.as_array(), z_shape.0)?;
    let q = z_shape.1;

    let ztwz = compute_ztwz_sparse(&z, &w);

    let mut flat_data = Vec::with_capacity(q * q);
    for i in 0..q {
        for j in 0..q {
            flat_data.push(ztwz[(i, j)]);
        }
    }

    Ok(PyArray1::from_vec(py, flat_data).into())
}

#[pyfunction]
#[pyo3(signature = (
    theta,
    y,
    x,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    weights,
    offset,
    n_levels,
    n_terms,
    correlated,
    reml = true,
    ztwz_cache = None
))]
#[allow(clippy::too_many_arguments)]
pub fn profiled_deviance_cached<'py>(
    theta: numpy::PyArrayLike1<'py, f64>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    reml: bool,
    ztwz_cache: Option<numpy::PyArrayLike1<'py, f64>>,
) -> PyResult<f64> {
    let structures: Vec<RandomEffectStructure> = n_levels
        .into_iter()
        .zip(n_terms)
        .zip(correlated)
        .map(|((nl, nt), c)| RandomEffectStructure {
            n_levels: nl,
            n_terms: nt,
            correlated: c,
        })
        .collect();

    let ztwz_data = ztwz_cache.as_ref().map(|arr| arr.as_slice()).transpose()?;

    profiled_deviance_impl(
        theta.as_slice()?,
        y.as_array(),
        x.as_array(),
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
        weights.as_array(),
        offset.as_array(),
        &structures,
        reml,
        ztwz_data,
    )
}

#[pyfunction]
#[pyo3(signature = (
    theta,
    y,
    x,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    weights,
    offset,
    n_levels,
    n_terms,
    correlated,
    reml = true
))]
#[allow(clippy::too_many_arguments)]
pub fn profiled_deviance<'py>(
    theta: numpy::PyArrayLike1<'py, f64>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    reml: bool,
) -> PyResult<f64> {
    let structures: Vec<RandomEffectStructure> = n_levels
        .into_iter()
        .zip(n_terms)
        .zip(correlated)
        .map(|((nl, nt), c)| RandomEffectStructure {
            n_levels: nl,
            n_terms: nt,
            correlated: c,
        })
        .collect();

    profiled_deviance_impl(
        theta.as_slice()?,
        y.as_array(),
        x.as_array(),
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
        weights.as_array(),
        offset.as_array(),
        &structures,
        reml,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (
    theta,
    y,
    x,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    weights,
    offset,
    n_levels,
    n_terms,
    correlated,
    reml = true
))]
#[allow(clippy::too_many_arguments)]
pub fn profiled_deviance_with_gradient<'py>(
    py: Python<'py>,
    theta: numpy::PyArrayLike1<'py, f64>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    reml: bool,
) -> PyResult<(f64, Py<PyArray1<f64>>)> {
    let structures: Vec<RandomEffectStructure> = n_levels
        .into_iter()
        .zip(n_terms)
        .zip(correlated)
        .map(|((nl, nt), c)| RandomEffectStructure {
            n_levels: nl,
            n_terms: nt,
            correlated: c,
        })
        .collect();

    let (dev, grad) = profiled_deviance_with_gradient_impl(
        theta.as_slice()?,
        y.as_array(),
        x.as_array(),
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
        weights.as_array(),
        offset.as_array(),
        &structures,
        reml,
        None,
    )?;

    Ok((dev, PyArray1::from_vec(py, grad).into()))
}

#[cfg(test)]
mod prepared_tests {
    use super::*;
    use numpy::ndarray::ArrayView1;

    #[test]
    fn coordinate_derivatives_match_full_matrix_products() {
        let structures = [
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
                n_levels: 2,
                n_terms: 2,
                correlated: false,
            },
            RandomEffectStructure {
                n_levels: 1,
                n_terms: 1,
                correlated: true,
            },
        ];
        let coordinates = [
            (0, 0, 1, 0, 0),
            (0, 2, 3, 0, 0),
            (0, 2, 3, 1, 0),
            (0, 2, 3, 1, 1),
            (0, 2, 3, 2, 0),
            (0, 2, 3, 2, 1),
            (0, 2, 3, 2, 2),
            (6, 2, 2, 0, 0),
            (6, 2, 2, 1, 1),
            (10, 1, 1, 0, 0),
        ];
        let derivatives = lambda_derivatives(&structures);
        assert_eq!(derivatives.len(), coordinates.len());
        for (derivative, (offset, levels, width, row, column)) in
            derivatives.iter().zip(coordinates)
        {
            assert_eq!(
                (
                    derivative.offset,
                    derivative.n_levels,
                    derivative.n_terms,
                    derivative.row,
                    derivative.column
                ),
                (offset, levels, width, row, column),
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
            for sparse in [false, true] {
                let product = Mat::from_fn(11, 11, |i, j| {
                    if sparse && i / 3 != j / 3 {
                        0.0
                    } else {
                        (i + 3 * j) as f64 - 5.0
                    }
                });
                let term = dense.transpose() * &product;
                assert_eq!(
                    derivative.crossproduct_derivative(&product),
                    &term + term.transpose()
                );
            }
        }
    }

    fn intercept_design() -> Arc<PreparedLmmDesign> {
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let z = CscMatrix::try_from_i64(&[], &[], &[0], (4, 0)).unwrap();
        Arc::new(PreparedLmmDesign::new(x, z, vec![1.0; 4], vec![0.0; 4], vec![], None).unwrap())
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
        assert_eq!(first.deviance(&[], false), initial);
        assert!(first.validate_parameters(&[1.0], false).is_err());
        assert!(
            second
                .design
                .with_response(ArrayView1::from(&[1.0]))
                .is_err()
        );
    }
}

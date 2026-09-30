use faer::linalg::solvers::{Llt, Solve, SolveLstsq};
use faer::{Col as DVector, Mat as DMatrix, Side};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::covariance::CovarianceFactor;
pub use crate::covariance::RandomEffectStructure;
use crate::csc::CscMatrix;
use crate::glmm_sparse::{RandomFactor, SparseWeightedDesign};
use crate::linalg::LinalgError;
use crate::quadrature::gauss_hermite_nodes_weights;

const PIRLS_MAX_ITER: usize = 100;
const PIRLS_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkFunction {
    Identity,
    Log,
    Logit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FamilyType {
    Gaussian,
    Binomial,
    Poisson,
}

impl LinkFunction {
    fn inverse(&self, eta: &DVector<f64>) -> DVector<f64> {
        match self {
            LinkFunction::Identity => eta.clone(),
            LinkFunction::Log => DVector::from_fn(eta.nrows(), |i| eta[i].exp()),
            LinkFunction::Logit => DVector::from_fn(eta.nrows(), |i| 1.0 / (1.0 + (-eta[i]).exp())),
        }
    }

    fn deriv(&self, mu: &DVector<f64>) -> DVector<f64> {
        match self {
            LinkFunction::Identity => DVector::full(mu.nrows(), 1.0),
            LinkFunction::Log => DVector::from_fn(mu.nrows(), |i| 1.0 / mu[i].max(1e-10)),
            LinkFunction::Logit => DVector::from_fn(mu.nrows(), |i| {
                let m = mu[i].clamp(1e-10, 1.0 - 1e-10);
                1.0 / (m * (1.0 - m))
            }),
        }
    }
}

impl FamilyType {
    fn clamp_mu(&self, mu: &mut DVector<f64>, eps: f64) {
        match self {
            FamilyType::Gaussian => {}
            FamilyType::Binomial => {
                for value in mu.iter_mut() {
                    *value = value.clamp(eps, 1.0 - eps);
                }
            }
            FamilyType::Poisson => {
                for value in mu.iter_mut() {
                    *value = value.max(eps);
                }
            }
        }
    }

    fn variance(&self, mu: &DVector<f64>) -> DVector<f64> {
        match self {
            FamilyType::Gaussian => DVector::full(mu.nrows(), 1.0),
            FamilyType::Binomial => DVector::from_fn(mu.nrows(), |i| {
                let m = mu[i].clamp(1e-10, 1.0 - 1e-10);
                m * (1.0 - m)
            }),
            FamilyType::Poisson => DVector::from_fn(mu.nrows(), |i| mu[i].max(1e-10)),
        }
    }

    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            FamilyType::Gaussian => (y - mu).powi(2),
            FamilyType::Binomial => {
                let bounded_mu = mu.clamp(1e-10, 1.0 - 1e-10);
                let success = if y > 1e-10 {
                    y * (y / bounded_mu).ln()
                } else {
                    0.0
                };
                let failure = if (1.0 - y) > 1e-10 {
                    (1.0 - y) * ((1.0 - y) / (1.0 - bounded_mu)).ln()
                } else {
                    0.0
                };
                2.0 * (success + failure)
            }
            FamilyType::Poisson => {
                let bounded_mu = mu.max(1e-10);
                let log_term = if y > 1e-10 {
                    y * (y / bounded_mu).ln()
                } else {
                    0.0
                };
                2.0 * (log_term - (y - bounded_mu))
            }
        }
    }

    fn deviance_resids(&self, y: &DVector<f64>, mu: &DVector<f64>, wt: &[f64]) -> f64 {
        (0..y.nrows())
            .map(|i| wt[i] * self.unit_deviance(y[i], mu[i]))
            .sum()
    }

    fn deviance_resids_rows(
        &self,
        y: &DVector<f64>,
        mu: &DVector<f64>,
        wt: &[f64],
        rows: &[usize],
    ) -> f64 {
        rows.iter()
            .map(|&i| wt[i] * self.unit_deviance(y[i], mu[i]))
            .sum()
    }

    fn weights(&self, mu: &DVector<f64>, link: LinkFunction) -> DVector<f64> {
        let link_deriv = link.deriv(mu);
        let variance = self.variance(mu);

        DVector::from_fn(mu.nrows(), |i| {
            let d = link_deriv[i];
            let v = variance[i].max(1e-10);
            1.0 / (d * d * v).max(1e-10)
        })
    }
}

fn parse_family_and_link(family: &str, link: &str) -> PyResult<(FamilyType, LinkFunction)> {
    let family_type = match family {
        "gaussian" => FamilyType::Gaussian,
        "binomial" => FamilyType::Binomial,
        "poisson" => FamilyType::Poisson,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported GLMM family: {family}"
            )));
        }
    };

    let link_function = match link {
        "identity" => LinkFunction::Identity,
        "log" => LinkFunction::Log,
        "logit" => LinkFunction::Logit,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported GLMM link: {link}"
            )));
        }
    };

    Ok((family_type, link_function))
}

fn csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix, LinalgError> {
    CscMatrix::try_from_i64(data, indices, indptr, shape)
}

fn dense_penalized_crossproduct(
    z: &CscMatrix,
    lambda: &CovarianceFactor,
    w_vec: &DVector<f64>,
) -> DMatrix<f64> {
    let q = z.ncols();
    let mut ztwz = DMatrix::zeros(q, q);
    for j1 in 0..q {
        let col1_start = z.col_offsets()[j1];
        let col1_end = z.col_offsets()[j1 + 1];

        for j2 in 0..=j1 {
            let col2_start = z.col_offsets()[j2];
            let col2_end = z.col_offsets()[j2 + 1];

            let mut sum = 0.0;
            let mut idx1 = col1_start;
            let mut idx2 = col2_start;

            while idx1 < col1_end && idx2 < col2_end {
                let row1 = z.row_indices()[idx1];
                let row2 = z.row_indices()[idx2];

                if row1 == row2 {
                    sum += z.values()[idx1] * w_vec[row1] * z.values()[idx2];
                    idx1 += 1;
                    idx2 += 1;
                } else if row1 < row2 {
                    idx1 += 1;
                } else {
                    idx2 += 1;
                }
            }

            ztwz[(j1, j2)] = sum;
            ztwz[(j2, j1)] = sum;
        }
    }
    lambda.penalized_crossproduct(ztwz)
}

fn factor_random_system(
    z: &CscMatrix,
    lambda: &CovarianceFactor,
    sparse: Option<&SparseWeightedDesign>,
    weights: &DVector<f64>,
) -> Result<RandomFactor, LinalgError> {
    if let Some(sparse) = sparse {
        let weights = weights
            .try_as_col_major()
            .expect("owned weights are contiguous");
        return sparse
            .factor(weights.as_slice(), 0.0)
            .or_else(|_| sparse.factor(weights.as_slice(), 1e-6))
            .map(RandomFactor::Sparse);
    }
    let mut matrix = dense_penalized_crossproduct(z, lambda, weights);
    match Llt::new(matrix.as_ref(), Side::Lower) {
        Ok(factor) => Ok(RandomFactor::Dense(factor)),
        Err(_) => {
            for i in 0..z.ncols() {
                matrix[(i, i)] += 1e-6;
            }
            Llt::new(matrix.as_ref(), Side::Lower)
                .map(RandomFactor::Dense)
                .map_err(|_| LinalgError::NotPositiveDefinite)
        }
    }
}

fn dense_logdet(z: &CscMatrix, lambda: &CovarianceFactor, weights: &DVector<f64>) -> f64 {
    let matrix = dense_penalized_crossproduct(z, lambda, weights);
    match Llt::new(matrix.as_ref(), Side::Lower) {
        Ok(factor) => 2.0 * (0..z.ncols()).map(|i| factor.L()[(i, i)].ln()).sum::<f64>(),
        Err(_) => matrix
            .self_adjoint_eigenvalues(Side::Lower)
            .unwrap_or_else(|_| vec![1e-10; z.ncols()])
            .iter()
            .map(|&value| value.max(1e-10).ln())
            .sum(),
    }
}

fn max_abs_diff(left: &DVector<f64>, right: &DVector<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

#[derive(Debug)]
pub struct PirlsResult {
    pub beta: DVector<f64>,
    pub spherical: DVector<f64>,
    pub u: DVector<f64>,
    pub deviance: f64,
    pub converged: bool,
    sparse_system: Option<SparseWeightedDesign>,
}

#[allow(clippy::too_many_arguments)]
pub fn pirls_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
    weights: &[f64],
    offset: &DVector<f64>,
    theta: &[f64],
    structures: &[RandomEffectStructure],
    family: FamilyType,
    link: LinkFunction,
    beta_start: Option<&DVector<f64>>,
    u_start: Option<&DVector<f64>>,
    maxiter: usize,
    tol: f64,
) -> PirlsResult {
    let n = y.nrows();
    let p = x.ncols();
    let q = z.ncols();

    let mut beta = if let Some(b) = beta_start {
        b.clone()
    } else {
        let eta = x * &DVector::<f64>::zeros(p) + offset;
        let mu = link.inverse(&eta);
        let link_deriv = link.deriv(&mu);
        let y_work: DVector<f64> = DVector::from_fn(n, |i| eta[i] + link_deriv[i] * (y[i] - mu[i]));

        let xtx = x.transpose() * x;
        let xty = x.transpose() * &y_work;

        match Llt::new(xtx.as_ref(), Side::Lower) {
            Ok(chol) => chol.solve(&xty),
            Err(_) => xtx.partial_piv_lu().solve(&xty),
        }
    };

    let lambda = CovarianceFactor::new(theta, structures);
    let sparse_system = SparseWeightedDesign::new(z, &lambda);
    let mut spherical = if let Some(u_init) = u_start {
        lambda.to_dense().col_piv_qr().solve_lstsq(u_init)
    } else {
        DVector::zeros(q)
    };

    let mut converged = false;

    for _iter in 0..maxiter {
        let random_effects = lambda.apply(&spherical);
        let mut eta = x * &beta + offset;
        for j in 0..q {
            let col_start = z.col_offsets()[j];
            let col_end = z.col_offsets()[j + 1];
            for idx in col_start..col_end {
                let i = z.row_indices()[idx];
                eta[i] += z.values()[idx] * random_effects[j];
            }
        }

        let mut mu = link.inverse(&eta);
        family.clamp_mu(&mut mu, 1e-10);

        let mut w_vec = family.weights(&mu, link);
        for i in 0..n {
            w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
        }

        let link_deriv = link.deriv(&mu);
        let z_vec: DVector<f64> =
            DVector::from_fn(n, |i| eta[i] - offset[i] + link_deriv[i] * (y[i] - mu[i]));

        let wx = DMatrix::from_fn(n, p, |i, j| w_vec[i].sqrt() * x[(i, j)]);
        let xtwx = wx.transpose() * &wx;

        let mut xtwz_mat = DMatrix::zeros(p, q);
        for j in 0..q {
            let col_start = z.col_offsets()[j];
            let col_end = z.col_offsets()[j + 1];
            for pj in 0..p {
                let mut sum = 0.0;
                for idx in col_start..col_end {
                    let i = z.row_indices()[idx];
                    sum += x[(i, pj)] * w_vec[i] * z.values()[idx];
                }
                xtwz_mat[(pj, j)] = sum;
            }
        }

        let ztwx = xtwz_mat.transpose();

        let xtwz_vec: DVector<f64> = DVector::from_fn(p, |i| {
            let mut sum = 0.0;
            for j in 0..n {
                sum += x[(j, i)] * w_vec[j] * z_vec[j];
            }
            sum
        });

        let mut ztwz_vec = DVector::zeros(q);
        for j in 0..q {
            let col_start = z.col_offsets()[j];
            let col_end = z.col_offsets()[j + 1];
            let mut sum = 0.0;
            for idx in col_start..col_end {
                let i = z.row_indices()[idx];
                sum += z.values()[idx] * w_vec[i] * z_vec[i];
            }
            ztwz_vec[j] = sum;
        }

        let chol_c = match factor_random_system(z, &lambda, sparse_system.as_ref(), &w_vec) {
            Ok(factor) => factor,
            Err(_) => {
                return PirlsResult {
                    beta,
                    spherical,
                    u: random_effects,
                    deviance: 1e10,
                    converged: false,
                    sparse_system,
                };
            }
        };

        let spherical_ztwx = lambda.transpose_apply(ztwx);
        let spherical_ztwz = lambda.transpose_apply_vector(&ztwz_vec);
        let mut rhs = DMatrix::from_fn(q, p + 1, |i, j| {
            if j == p {
                spherical_ztwz[i]
            } else {
                spherical_ztwx[(i, j)]
            }
        });
        chol_c.solve_lower_in_place(rhs.as_mut());
        let rzx = rhs.subcols(0, p);
        let cu = rhs.col(p);

        let xtvinvx = &xtwx - &(rzx.transpose() * rzx);
        let xtvinvz = &xtwz_vec - &(rzx.transpose() * cu);

        let beta_new = match Llt::new(xtvinvx.as_ref(), Side::Lower) {
            Ok(chol) => chol.solve(&xtvinvz),
            Err(_) => xtvinvx.partial_piv_lu().solve(&xtvinvz),
        };

        let mut spherical_new = cu - rzx * &beta_new;
        chol_c.solve_upper_in_place(spherical_new.as_mat_mut());

        let delta_beta = max_abs_diff(&beta_new, &beta);
        let delta_u = if q > 0 {
            max_abs_diff(&spherical_new, &spherical)
        } else {
            0.0
        };

        beta = beta_new;
        spherical = spherical_new;

        if delta_beta < tol && delta_u < tol {
            converged = true;
            break;
        }
    }

    let random_effects = lambda.apply(&spherical);
    let mut eta_final = x * &beta + offset;
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            eta_final[i] += z.values()[idx] * random_effects[j];
        }
    }

    let mut mu_final = link.inverse(&eta_final);
    family.clamp_mu(&mut mu_final, 1e-10);

    let dev_resids = family.deviance_resids(y, &mu_final, weights);

    let deviance = dev_resids + spherical.squared_norm_l2();

    PirlsResult {
        beta,
        spherical,
        u: random_effects,
        deviance,
        converged,
        sparse_system,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn laplace_deviance_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
    weights: &[f64],
    offset: &DVector<f64>,
    theta: &[f64],
    structures: &[RandomEffectStructure],
    family: FamilyType,
    link: LinkFunction,
    beta_start: Option<&DVector<f64>>,
    u_start: Option<&DVector<f64>>,
) -> (f64, DVector<f64>, DVector<f64>) {
    let n = y.nrows();
    let q = z.ncols();

    if q == 0 {
        let result = pirls_impl(
            y,
            x,
            z,
            weights,
            offset,
            theta,
            structures,
            family,
            link,
            beta_start,
            u_start,
            PIRLS_MAX_ITER,
            PIRLS_TOLERANCE,
        );
        return (result.deviance, result.beta, result.u);
    }

    let result = pirls_impl(
        y,
        x,
        z,
        weights,
        offset,
        theta,
        structures,
        family,
        link,
        beta_start,
        u_start,
        PIRLS_MAX_ITER,
        PIRLS_TOLERANCE,
    );

    let beta = result.beta;
    let u = result.u;
    let spherical = result.spherical;

    let lambda = CovarianceFactor::new(theta, structures);

    let mut eta = x * &beta + offset;
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            eta[i] += z.values()[idx] * u[j];
        }
    }

    let mut mu = link.inverse(&eta);
    family.clamp_mu(&mut mu, 1e-10);

    let dev_resids = family.deviance_resids(y, &mu, weights);

    let mut deviance = dev_resids + spherical.squared_norm_l2();

    let mut w_vec = family.weights(&mu, link);
    for i in 0..n {
        w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
    }

    let logdet_h = if let Some(sparse) = result.sparse_system.as_ref() {
        let weights = w_vec
            .try_as_col_major()
            .expect("owned weights are contiguous");
        match sparse.factor(weights.as_slice(), 0.0) {
            Ok(factor) => factor.logdet(),
            Err(_) => dense_logdet(z, &lambda, &w_vec),
        }
    } else {
        dense_logdet(z, &lambda, &w_vec)
    };

    deviance += logdet_h;

    (deviance, beta, u)
}

#[allow(clippy::too_many_arguments)]
fn compute_group_log_integral(
    g: usize,
    n_terms: usize,
    spherical: &DVector<f64>,
    h: &[f64],
    lambda: &CovarianceFactor,
    nodes: &[f64],
    weights: &[f64],
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
    beta: &DVector<f64>,
    offset: &DVector<f64>,
    prior_weights: &[f64],
    family: FamilyType,
    link: LinkFunction,
) -> f64 {
    let sqrt2 = std::f64::consts::SQRT_2;
    let q = z.ncols();

    let idx_start = g * n_terms;

    let spherical_mode = spherical.subrows(idx_start, n_terms).to_owned();

    debug_assert_eq!(n_terms, 1);
    let scale = 1.0 / (h[idx_start] + 1e-10).sqrt();

    let col_start = z.col_offsets()[idx_start];
    let col_end = z.col_offsets()[idx_start + 1];
    let group_rows = &z.row_indices()[col_start..col_end];
    if group_rows.is_empty() {
        return 0.0;
    }

    let mut log_terms = Vec::with_capacity(nodes.len());
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        let mut spherical_quad = spherical.clone();
        for i in 0..n_terms {
            spherical_quad[idx_start + i] = spherical_mode[i] + sqrt2 * scale * node;
        }
        let random_effects = lambda.apply(&spherical_quad);

        let mut eta_quad = x * beta + offset;
        for j in 0..q {
            let col_start = z.col_offsets()[j];
            let col_end = z.col_offsets()[j + 1];
            for idx in col_start..col_end {
                let i = z.row_indices()[idx];
                eta_quad[i] += z.values()[idx] * random_effects[j];
            }
        }

        let mut mu_quad = link.inverse(&eta_quad);
        family.clamp_mu(&mut mu_quad, 1e-10);

        let dev_resids = family.deviance_resids_rows(y, &mu_quad, prior_weights, group_rows);
        let log_lik_y = -0.5 * dev_resids;

        let spherical_block: DVector<f64> = spherical_quad.subrows(idx_start, n_terms).to_owned();
        let log_prior = -0.5 * spherical_block.squared_norm_l2();
        log_terms.push(weight.ln() + log_lik_y + log_prior + node * node);
    }

    let max_log = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = max_log
        + log_terms
            .iter()
            .map(|value| (value - max_log).exp())
            .sum::<f64>()
            .ln();
    scale.ln() - 0.5 * std::f64::consts::PI.ln() + log_sum
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_gh_deviance_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
    weights: &[f64],
    offset: &DVector<f64>,
    theta: &[f64],
    structures: &[RandomEffectStructure],
    family: FamilyType,
    link: LinkFunction,
    n_agq: usize,
    beta_start: Option<&DVector<f64>>,
    u_start: Option<&DVector<f64>>,
) -> (f64, DVector<f64>, DVector<f64>) {
    let q = z.ncols();

    if n_agq <= 1 || q == 0 {
        return laplace_deviance_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
        );
    }

    if structures.len() != 1 {
        return laplace_deviance_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
        );
    }

    let first_struct = &structures[0];
    let n_terms_first = first_struct.n_terms;
    let n_levels_first = first_struct.n_levels;

    if n_terms_first > 1 {
        return laplace_deviance_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
        );
    }

    let result = pirls_impl(
        y,
        x,
        z,
        weights,
        offset,
        theta,
        structures,
        family,
        link,
        beta_start,
        u_start,
        PIRLS_MAX_ITER,
        PIRLS_TOLERANCE,
    );

    let beta = result.beta;
    let u = result.u;
    let spherical = result.spherical;

    let n = y.nrows();
    let lambda = CovarianceFactor::new(theta, structures);

    let mut eta = x * &beta + offset;
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            eta[i] += z.values()[idx] * u[j];
        }
    }

    let mut mu = link.inverse(&eta);
    family.clamp_mu(&mut mu, 1e-10);

    let mut w_vec = family.weights(&mu, link);
    for i in 0..n {
        w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
    }

    // Scalar group quadrature only uses the diagonal curvature.
    let h: Vec<f64> = (0..q)
        .map(|column| {
            let curvature: f64 = (z.col_offsets()[column]..z.col_offsets()[column + 1])
                .map(|entry| z.values()[entry] * w_vec[z.row_indices()[entry]] * z.values()[entry])
                .sum();
            (theta[0] * curvature) * theta[0] + 1.0
        })
        .collect();

    let (nodes, gh_weights) = gauss_hermite_nodes_weights(n_agq);

    #[cfg(miri)]
    let iter = (0..n_levels_first).into_iter();
    #[cfg(not(miri))]
    let iter = (0..n_levels_first).into_par_iter();
    let log_integral: f64 = iter
        .map(|g| {
            compute_group_log_integral(
                g,
                n_terms_first,
                &spherical,
                &h,
                &lambda,
                &nodes,
                &gh_weights,
                y,
                x,
                z,
                &beta,
                offset,
                weights,
                family,
                link,
            )
        })
        .sum();

    let deviance = -2.0 * log_integral;

    (deviance, beta, u)
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    weights,
    offset,
    theta,
    n_levels,
    n_terms,
    correlated,
    family,
    link
))]
#[allow(clippy::too_many_arguments)]
pub fn pirls<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, f64, bool)> {
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

    let (family_type, link_fn) = parse_family_and_link(family, link)?;

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let n = y_arr.len();
    let p = x_arr.ncols();

    let y_vec: DVector<f64> = y_arr.iter().copied().collect();
    let x_mat = DMatrix::from_fn(n, p, |i, j| x_arr[[i, j]]);
    let z_mat = csc_from_scipy(
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
    )?;
    let offset_vec: DVector<f64> = offset.as_array().iter().copied().collect();

    let result = pirls_impl(
        &y_vec,
        &x_mat,
        &z_mat,
        weights.as_slice()?,
        &offset_vec,
        theta.as_slice()?,
        &structures,
        family_type,
        link_fn,
        None,
        None,
        PIRLS_MAX_ITER,
        PIRLS_TOLERANCE,
    );

    Ok((
        result.beta.iter().cloned().collect(),
        result.u.iter().cloned().collect(),
        result.deviance,
        result.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    weights,
    offset,
    theta,
    n_levels,
    n_terms,
    correlated,
    family,
    link
))]
#[allow(clippy::too_many_arguments)]
pub fn laplace_deviance<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
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

    let (family_type, link_fn) = parse_family_and_link(family, link)?;

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let n = y_arr.len();
    let p = x_arr.ncols();

    let y_vec: DVector<f64> = y_arr.iter().copied().collect();
    let x_mat = DMatrix::from_fn(n, p, |i, j| x_arr[[i, j]]);
    let z_mat = csc_from_scipy(
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
    )?;
    let offset_vec: DVector<f64> = offset.as_array().iter().copied().collect();

    let (deviance, beta, u) = laplace_deviance_impl(
        &y_vec,
        &x_mat,
        &z_mat,
        weights.as_slice()?,
        &offset_vec,
        theta.as_slice()?,
        &structures,
        family_type,
        link_fn,
        None,
        None,
    );

    Ok((
        deviance,
        beta.iter().cloned().collect(),
        u.iter().cloned().collect(),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    weights,
    offset,
    theta,
    n_levels,
    n_terms,
    correlated,
    family,
    link,
    n_agq
))]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_gh_deviance<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
    n_agq: usize,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
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

    let (family_type, link_fn) = parse_family_and_link(family, link)?;

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let n = y_arr.len();
    let p = x_arr.ncols();

    let y_vec: DVector<f64> = y_arr.iter().copied().collect();
    let x_mat = DMatrix::from_fn(n, p, |i, j| x_arr[[i, j]]);
    let z_mat = csc_from_scipy(
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
    )?;
    let offset_vec: DVector<f64> = offset.as_array().iter().copied().collect();

    let (deviance, beta, u) = adaptive_gh_deviance_impl(
        &y_vec,
        &x_mat,
        &z_mat,
        weights.as_slice()?,
        &offset_vec,
        theta.as_slice()?,
        &structures,
        family_type,
        link_fn,
        n_agq,
        None,
        None,
    );

    Ok((
        deviance,
        beta.iter().cloned().collect(),
        u.iter().cloned().collect(),
    ))
}

use faer::linalg::solvers::{DenseSolveCore, Llt, Solve};
use faer::{Col as DVector, Mat as DMatrix, Side};
use pyo3::PyResult;
use pyo3::prelude::*;
use std::collections::BTreeMap;

const PNLS_MAX_ITER: usize = 50;
const PNLS_TOLERANCE: f64 = 1e-6;

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum NlmeModel {
    SSasymp,
    SSlogis,
    SSmicmen,
    SSfpl,
    SSgompertz,
    SSbiexp,
}

#[cfg(test)]
mod logistic_tests {
    use super::*;

    #[test]
    fn logistic_is_finite_at_extreme_values() {
        assert_eq!(logistic(-1e6), 0.0);
        assert_eq!(logistic(0.0), 0.5);
        assert_eq!(logistic(1e6), 1.0);
    }

    #[test]
    fn sslogis_extreme_predictions_and_gradients_are_finite() {
        let model = NlmeModel::SSlogis;
        let params = [10.0, 0.0, 1.0];
        let x = [-1e6, 0.0, 1e6];

        assert_eq!(model.predict(&params, &x), vec![0.0, 5.0, 10.0]);
        assert!(
            model
                .gradient(&params, &x)
                .col_iter()
                .all(|column| column.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn ssfpl_extreme_predictions_and_gradients_are_finite() {
        let model = NlmeModel::SSfpl;
        let params = [2.0, 10.0, 0.0, 1.0];
        let x = [-1e6, 0.0, 1e6];

        assert_eq!(model.predict(&params, &x), vec![2.0, 6.0, 10.0]);
        assert!(
            model
                .gradient(&params, &x)
                .col_iter()
                .all(|column| column.iter().all(|value| value.is_finite()))
        );
    }
}

impl NlmeModel {
    fn n_params(&self) -> usize {
        match self {
            NlmeModel::SSasymp => 3,
            NlmeModel::SSlogis => 3,
            NlmeModel::SSmicmen => 2,
            NlmeModel::SSfpl => 4,
            NlmeModel::SSgompertz => 3,
            NlmeModel::SSbiexp => 4,
        }
    }

    fn predict(&self, params: &[f64], x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut result = vec![0.0; n];

        match self {
            NlmeModel::SSasymp => {
                let asym = params[0];
                let r0 = params[1];
                let lrc = params[2];
                let rc = lrc.exp();
                for i in 0..n {
                    result[i] = asym + (r0 - asym) * (-rc * x[i]).exp();
                }
            }
            NlmeModel::SSlogis => {
                let asym = params[0];
                let xmid = params[1];
                let scal = params[2];
                for i in 0..n {
                    result[i] = asym * logistic((x[i] - xmid) / scal);
                }
            }
            NlmeModel::SSmicmen => {
                let vm = params[0];
                let k = params[1];
                for i in 0..n {
                    result[i] = vm * x[i] / (k + x[i]);
                }
            }
            NlmeModel::SSfpl => {
                let a = params[0];
                let b = params[1];
                let xmid = params[2];
                let scal = params[3];
                for i in 0..n {
                    result[i] = a + (b - a) * logistic((x[i] - xmid) / scal);
                }
            }
            NlmeModel::SSgompertz => {
                let asym = params[0];
                let b2 = params[1];
                let b3 = params[2];
                for i in 0..n {
                    result[i] = asym * (-b2 * b3.powf(x[i])).exp();
                }
            }
            NlmeModel::SSbiexp => {
                let a1 = params[0];
                let lrc1 = params[1];
                let a2 = params[2];
                let lrc2 = params[3];
                let rc1 = lrc1.exp();
                let rc2 = lrc2.exp();
                for i in 0..n {
                    result[i] = a1 * (-rc1 * x[i]).exp() + a2 * (-rc2 * x[i]).exp();
                }
            }
        }

        result
    }

    fn gradient(&self, params: &[f64], x: &[f64]) -> DMatrix<f64> {
        let n = x.len();
        let p = self.n_params();
        let mut grad = DMatrix::zeros(n, p);

        match self {
            NlmeModel::SSasymp => {
                let asym = params[0];
                let r0 = params[1];
                let lrc = params[2];
                let rc = lrc.exp();
                for i in 0..n {
                    let exp_term = (-rc * x[i]).exp();
                    grad[(i, 0)] = 1.0 - exp_term;
                    grad[(i, 1)] = exp_term;
                    grad[(i, 2)] = -(r0 - asym) * rc * x[i] * exp_term;
                }
            }
            NlmeModel::SSlogis => {
                let asym = params[0];
                let xmid = params[1];
                let scal = params[2];
                for i in 0..n {
                    let fraction = logistic((x[i] - xmid) / scal);
                    let sensitivity = fraction * (1.0 - fraction);
                    grad[(i, 0)] = fraction;
                    grad[(i, 1)] = -asym * sensitivity / scal;
                    grad[(i, 2)] = asym * (xmid - x[i]) * sensitivity / (scal * scal);
                }
            }
            NlmeModel::SSmicmen => {
                let vm = params[0];
                let k = params[1];
                for i in 0..n {
                    let denom = k + x[i];
                    let denom_sq = denom * denom;
                    grad[(i, 0)] = x[i] / denom;
                    grad[(i, 1)] = -vm * x[i] / denom_sq;
                }
            }
            NlmeModel::SSfpl => {
                let a = params[0];
                let b = params[1];
                let xmid = params[2];
                let scal = params[3];
                for i in 0..n {
                    let fraction = logistic((x[i] - xmid) / scal);
                    let sensitivity = fraction * (1.0 - fraction);
                    grad[(i, 0)] = 1.0 - fraction;
                    grad[(i, 1)] = fraction;
                    grad[(i, 2)] = -(b - a) * sensitivity / scal;
                    grad[(i, 3)] = (b - a) * (xmid - x[i]) * sensitivity / (scal * scal);
                }
            }
            NlmeModel::SSgompertz => {
                let asym = params[0];
                let b2 = params[1];
                let b3 = params[2];
                for i in 0..n {
                    let b3_x = b3.powf(x[i]);
                    let exp_term = (-b2 * b3_x).exp();
                    grad[(i, 0)] = exp_term;
                    grad[(i, 1)] = -asym * b3_x * exp_term;
                    grad[(i, 2)] = -asym * b2 * x[i] * b3.powf(x[i] - 1.0) * exp_term;
                }
            }
            NlmeModel::SSbiexp => {
                let a1 = params[0];
                let lrc1 = params[1];
                let a2 = params[2];
                let lrc2 = params[3];
                let rc1 = lrc1.exp();
                let rc2 = lrc2.exp();
                for i in 0..n {
                    let exp1 = (-rc1 * x[i]).exp();
                    let exp2 = (-rc2 * x[i]).exp();
                    grad[(i, 0)] = exp1;
                    grad[(i, 1)] = -a1 * rc1 * x[i] * exp1;
                    grad[(i, 2)] = exp2;
                    grad[(i, 3)] = -a2 * rc2 * x[i] * exp2;
                }
            }
        }

        grad
    }
}

fn build_psi_factor(theta: &[f64], n_random: usize) -> DMatrix<f64> {
    if theta.is_empty() {
        return DMatrix::<f64>::identity(n_random, n_random);
    }

    let n_theta = theta.len();
    let q = ((-1.0 + (1.0 + 8.0 * n_theta as f64).sqrt()) / 2.0) as usize;

    if q * (q + 1) / 2 == n_theta {
        let mut l = DMatrix::zeros(q, q);
        let mut idx = 0;
        for i in 0..q {
            for j in 0..=i {
                l[(i, j)] = theta[idx];
                idx += 1;
            }
        }
        l
    } else {
        DMatrix::from_fn(n_random, n_random, |row, column| {
            if row == column {
                theta.get(row).copied().unwrap_or(0.0)
            } else {
                0.0
            }
        })
    }
}

fn build_psi_matrix(theta: &[f64], n_random: usize) -> DMatrix<f64> {
    let l = build_psi_factor(theta, n_random);
    &l * l.transpose()
}

fn invert_regularized(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    match Llt::new(matrix.as_ref(), Side::Lower) {
        Ok(cholesky) => cholesky.inverse(),
        Err(_) => matrix.partial_piv_lu().inverse(),
    }
}

fn quadratic_form(matrix: &DMatrix<f64>, vector: &DVector<f64>) -> f64 {
    let product = matrix * vector;
    vector
        .iter()
        .zip(product.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

fn validate_prior_weights(weights: &[f64], n: usize) -> PyResult<()> {
    if weights.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "weights has length {}, expected {}",
            weights.len(),
            n
        )));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must contain only finite values",
        ));
    }
    if weights.iter().any(|weight| *weight <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must be strictly positive",
        ));
    }
    Ok(())
}

fn grouped_observation_indices(groups: &[i64]) -> Vec<Vec<usize>> {
    let mut indices: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row, group) in groups.iter().enumerate() {
        indices.entry(*group).or_default().push(row);
    }
    indices.into_values().collect()
}

pub struct PnlsResult {
    pub phi: Vec<f64>,
    pub b: DMatrix<f64>,
    pub sigma_sq: f64,
    pub converged: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn pnls_step_impl(
    y: &[f64],
    x: &[f64],
    groups: &[i64],
    weights: &[f64],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    psi: &DMatrix<f64>,
    _sigma: f64,
    random_params: &[usize],
    maxiter: usize,
    tol: f64,
) -> PnlsResult {
    let group_indices = grouped_observation_indices(groups);
    pnls_step_with_groups(
        y,
        x,
        &group_indices,
        weights,
        model,
        phi,
        b,
        psi,
        random_params,
        maxiter,
        tol,
    )
}

struct PnlsGroup {
    x: Vec<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    sqrt_weights: Vec<f64>,
}

fn random_effect_penalty(b: &DMatrix<f64>, precision: &DMatrix<f64>) -> f64 {
    (0..b.nrows())
        .map(|group| {
            let effects = DVector::from_fn(b.ncols(), |j| b[(group, j)]);
            quadratic_form(precision, &effects)
        })
        .sum()
}

fn pnls_objective(
    groups: &[PnlsGroup],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    random_params: &[usize],
    precision: &DMatrix<f64>,
) -> f64 {
    let mut rss = 0.0;
    for (g, group) in groups.iter().enumerate() {
        let mut params = phi.to_vec();
        for (j, &parameter) in random_params.iter().enumerate() {
            params[parameter] += b[(g, j)];
        }
        let predicted = model.predict(&params, &group.x);
        for (i, predicted) in predicted.iter().enumerate() {
            let residual = group.y[i] - predicted;
            rss += group.weights[i] * residual * residual;
        }
    }
    rss + random_effect_penalty(b, precision)
}

#[allow(clippy::too_many_arguments)]
fn pnls_step_with_groups(
    y: &[f64],
    x: &[f64],
    group_indices: &[Vec<usize>],
    weights: &[f64],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    psi: &DMatrix<f64>,
    random_params: &[usize],
    maxiter: usize,
    tol: f64,
) -> PnlsResult {
    let n_phi = phi.len();
    let n_random = random_params.len();
    let groups: Vec<PnlsGroup> = group_indices
        .iter()
        .map(|rows| PnlsGroup {
            x: rows.iter().map(|&row| x[row]).collect(),
            y: rows.iter().map(|&row| y[row]).collect(),
            weights: rows.iter().map(|&row| weights[row]).collect(),
            sqrt_weights: rows.iter().map(|&row| weights[row].sqrt()).collect(),
        })
        .collect();
    let mut covariance = psi.clone();
    for i in 0..n_random {
        covariance[(i, i)] += 1e-8;
    }
    let precision = invert_regularized(&covariance);
    let mut phi_new = phi.to_vec();
    let mut b_new = b.clone();
    let mut converged = false;
    let mut pwrss = f64::INFINITY;

    for _iteration in 0..maxiter {
        let mut normal = DMatrix::<f64>::zeros(n_phi, n_phi);
        let mut rhs = DVector::<f64>::zeros(n_phi);
        let mut solutions = Vec::with_capacity(groups.len());
        let mut rss = 0.0;
        for (g, group) in groups.iter().enumerate() {
            let mut params = phi_new.clone();
            for (j, &parameter) in random_params.iter().enumerate() {
                params[parameter] += b_new[(g, j)];
            }
            let predicted = model.predict(&params, &group.x);
            let mut gradient = model.gradient(&params, &group.x);
            let residual = DVector::from_fn(group.x.len(), |i| {
                let value = group.y[i] - predicted[i];
                rss += group.weights[i] * value * value;
                value * group.sqrt_weights[i]
            });
            for i in 0..group.x.len() {
                for j in 0..n_phi {
                    gradient[(i, j)] *= group.sqrt_weights[i];
                }
            }
            let random_gradient = DMatrix::from_fn(group.x.len(), n_random, |i, j| {
                gradient[(i, random_params[j])]
            });
            let fixed_normal = gradient.transpose() * &gradient;
            let fixed_rhs = gradient.transpose() * &residual;
            let crossproduct = gradient.transpose() * &random_gradient;
            let random_normal = random_gradient.transpose() * &random_gradient + &precision;
            let effects = DVector::from_fn(n_random, |j| b_new[(g, j)]);
            let random_rhs = random_gradient.transpose() * &residual - &precision * effects;
            let joint_rhs = DMatrix::from_fn(n_random, n_phi + 1, |i, j| {
                if j == n_phi {
                    random_rhs[i]
                } else {
                    crossproduct[(j, i)]
                }
            });
            let solution = match Llt::new(random_normal.as_ref(), Side::Lower) {
                Ok(chol) => chol.solve(&joint_rhs),
                Err(_) => random_normal.partial_piv_lu().solve(&joint_rhs),
            };
            for i in 0..n_phi {
                rhs[i] += fixed_rhs[i]
                    - (0..n_random)
                        .map(|r| crossproduct[(i, r)] * solution[(r, n_phi)])
                        .sum::<f64>();
                for j in 0..n_phi {
                    normal[(i, j)] += fixed_normal[(i, j)]
                        - (0..n_random)
                            .map(|r| crossproduct[(i, r)] * solution[(r, j)])
                            .sum::<f64>();
                }
            }
            solutions.push(solution);
        }
        pwrss = rss + random_effect_penalty(&b_new, &precision);
        for i in 0..n_phi {
            normal[(i, i)] += 1e-6;
            for j in 0..i {
                let value = 0.5 * (normal[(i, j)] + normal[(j, i)]);
                normal[(i, j)] = value;
                normal[(j, i)] = value;
            }
        }
        let delta_phi = match Llt::new(normal.as_ref(), Side::Lower) {
            Ok(chol) => chol.solve(&rhs),
            Err(_) => normal.partial_piv_lu().solve(&rhs),
        };
        let delta_b = DMatrix::from_fn(groups.len(), n_random, |g, r| {
            solutions[g][(r, n_phi)]
                - (0..n_phi)
                    .map(|j| solutions[g][(r, j)] * delta_phi[j])
                    .sum::<f64>()
        });
        let mut max_delta: f64 = 0.0;
        for &value in delta_phi
            .iter()
            .chain(delta_b.col_iter().flat_map(|column| column.iter()))
        {
            max_delta = if value.is_finite() {
                max_delta.max(value.abs())
            } else {
                f64::INFINITY
            };
        }
        if !max_delta.is_finite() {
            break;
        }
        let slack = 16.0 * f64::EPSILON * pwrss.max(f64::MIN_POSITIVE);
        let mut step = 1.0;
        let mut accepted = false;
        for _backtrack in 0..21 {
            let trial_phi: Vec<f64> = phi_new
                .iter()
                .zip(delta_phi.iter())
                .map(|(&value, &delta)| value + step * delta)
                .collect();
            let trial_b = DMatrix::from_fn(groups.len(), n_random, |g, r| {
                b_new[(g, r)] + step * delta_b[(g, r)]
            });
            let candidate = pnls_objective(
                &groups,
                model,
                &trial_phi,
                &trial_b,
                random_params,
                &precision,
            );
            if candidate.is_finite() && candidate <= pwrss + slack {
                phi_new = trial_phi;
                b_new = trial_b;
                pwrss = candidate;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        // A shortened step alone cannot establish convergence.
        if max_delta < tol {
            converged = true;
            break;
        }
        if !accepted {
            break;
        }
    }
    let variance = pwrss / y.len() as f64;
    PnlsResult {
        phi: phi_new,
        b: b_new,
        sigma_sq: if variance.is_finite() {
            variance.max(f64::MIN_POSITIVE)
        } else {
            variance
        },
        converged,
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn nlmm_deviance_impl(
    theta: &[f64],
    y: &[f64],
    x: &[f64],
    groups: &[i64],
    weights: &[f64],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    random_params: &[usize],
    _sigma: f64,
) -> (f64, Vec<f64>, DMatrix<f64>, f64) {
    let (deviance, phi, b, sigma, _) = nlmm_deviance_with_status_impl(
        theta,
        y,
        x,
        groups,
        weights,
        model,
        phi,
        b,
        random_params,
        _sigma,
        PNLS_MAX_ITER,
        PNLS_TOLERANCE,
    );
    (deviance, phi, b, sigma)
}

#[allow(clippy::too_many_arguments)]
fn nlmm_deviance_with_status_impl(
    theta: &[f64],
    y: &[f64],
    x: &[f64],
    groups: &[i64],
    weights: &[f64],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    random_params: &[usize],
    _sigma: f64,
    maxiter: usize,
    tol: f64,
) -> (f64, Vec<f64>, DMatrix<f64>, f64, bool) {
    let n = y.len();
    let group_indices = grouped_observation_indices(groups);
    let n_random = random_params.len();
    let sqrt_weights: Vec<f64> = weights.iter().map(|weight| weight.sqrt()).collect();

    let psi_factor = build_psi_factor(theta, n_random);
    let psi = &psi_factor * psi_factor.transpose();

    let result = pnls_step_with_groups(
        y,
        x,
        &group_indices,
        weights,
        model,
        phi,
        b,
        &psi,
        random_params,
        maxiter,
        tol,
    );

    let phi_new = result.phi;
    let b_new = result.b;

    let sigma_sq = result.sigma_sq;
    let mut laplace_correction = 0.0;
    let identity = DMatrix::<f64>::identity(n_random, n_random);

    for (g_idx, mask) in group_indices.iter().enumerate() {
        let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
        let mut params_g = phi_new.clone();
        for (j, &p_idx) in random_params.iter().enumerate() {
            params_g[p_idx] += b_new[(g_idx, j)];
        }

        let grad_g = model.gradient(&params_g, &x_g);
        let mut z_g = DMatrix::zeros(mask.len(), n_random);
        for i in 0..mask.len() {
            for (j, &p_idx) in random_params.iter().enumerate() {
                z_g[(i, j)] = grad_g[(i, p_idx)];
            }
        }

        let mut weighted_z_g = z_g.clone();
        for i in 0..mask.len() {
            let sqrt_weight = sqrt_weights[mask[i]];
            for j in 0..n_random {
                weighted_z_g[(i, j)] *= sqrt_weight;
            }
        }
        let ztz = weighted_z_g.transpose() * &weighted_z_g;
        // Stable form of log|Psi| + log|Z'WZ + Psi^-1| for Psi = L L'.
        let system = &identity + psi_factor.transpose() * ztz * &psi_factor;
        let logdet = match Llt::new(system.as_ref(), Side::Lower) {
            Ok(chol) => {
                let l = chol.L();
                2.0 * (0..n_random).map(|i| l[(i, i)].ln()).sum::<f64>()
            }
            Err(_) => {
                let eigenvalues = system
                    .self_adjoint_eigenvalues(Side::Lower)
                    .unwrap_or_else(|_| vec![f64::NAN; n_random]);
                if eigenvalues
                    .iter()
                    .any(|value| *value <= 0.0 || !value.is_finite())
                {
                    return (1e100, phi_new, b_new, sigma_sq.sqrt(), false);
                }
                eigenvalues.iter().map(|value| value.ln()).sum()
            }
        };
        laplace_correction += logdet;
    }

    let deviance =
        n as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma_sq).ln()) + laplace_correction;

    (deviance, phi_new, b_new, sigma_sq.sqrt(), result.converged)
}

fn validate_pnls_controls(maxiter: usize, tol: f64) -> Result<(), &'static str> {
    if maxiter == 0 {
        return Err("pnls_maxiter must be a positive integer");
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("pnls_tol must be positive and finite");
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    groups,
    model_name,
    phi,
    b,
    theta,
    sigma,
    random_params,
    weights=None,
    maxiter=PNLS_MAX_ITER,
    tol=PNLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments)]
pub fn pnls_step<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike1<'py, f64>,
    groups: numpy::PyArrayLike1<'py, i64>,
    model_name: &str,
    phi: numpy::PyArrayLike1<'py, f64>,
    b: numpy::PyArrayLike2<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    sigma: f64,
    random_params: Vec<usize>,
    weights: Option<numpy::PyArrayLike1<'py, f64>>,
    maxiter: usize,
    tol: f64,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, f64)> {
    validate_pnls_controls(maxiter, tol).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let model = match model_name.to_lowercase().as_str() {
        "ssasymp" => NlmeModel::SSasymp,
        "sslogis" => NlmeModel::SSlogis,
        "ssmicmen" => NlmeModel::SSmicmen,
        "ssfpl" => NlmeModel::SSfpl,
        "ssgompertz" => NlmeModel::SSgompertz,
        "ssbiexp" => NlmeModel::SSbiexp,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model: {}. Supported models: SSasymp, SSlogis, SSmicmen, SSfpl, SSgompertz, SSbiexp",
                model_name
            )));
        }
    };

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let groups_arr = groups.as_array();
    let phi_arr = phi.as_array();
    let b_arr = b.as_array();
    let theta_arr = theta.as_array();

    let y_vec: Vec<f64> = y_arr.iter().cloned().collect();
    let x_vec: Vec<f64> = x_arr.iter().cloned().collect();
    let groups_vec: Vec<i64> = groups_arr.iter().cloned().collect();
    let phi_vec: Vec<f64> = phi_arr.iter().cloned().collect();

    let n_groups = b_arr.nrows();
    let n_random = b_arr.ncols();
    let b_mat = DMatrix::from_fn(n_groups, n_random, |i, j| b_arr[[i, j]]);

    let theta_vec: Vec<f64> = theta_arr.iter().cloned().collect();
    let weights_vec = match weights {
        Some(weights) => weights.as_array().iter().copied().collect(),
        None => vec![1.0; y_vec.len()],
    };
    validate_prior_weights(&weights_vec, y_vec.len())?;
    let psi = build_psi_matrix(&theta_vec, n_random);

    let result = pnls_step_impl(
        &y_vec,
        &x_vec,
        &groups_vec,
        &weights_vec,
        model,
        &phi_vec,
        &b_mat,
        &psi,
        sigma,
        &random_params,
        maxiter,
        tol,
    );

    let b_out: Vec<Vec<f64>> = (0..n_groups)
        .map(|i| (0..n_random).map(|j| result.b[(i, j)]).collect())
        .collect();

    Ok((result.phi, b_out, result.sigma_sq.sqrt()))
}

#[pyfunction]
#[pyo3(signature = (
    theta,
    y,
    x,
    groups,
    model_name,
    phi,
    b,
    random_params,
    sigma,
    weights=None,
    maxiter=PNLS_MAX_ITER,
    tol=PNLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn nlmm_deviance<'py>(
    theta: numpy::PyArrayLike1<'py, f64>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike1<'py, f64>,
    groups: numpy::PyArrayLike1<'py, i64>,
    model_name: &str,
    phi: numpy::PyArrayLike1<'py, f64>,
    b: numpy::PyArrayLike2<'py, f64>,
    random_params: Vec<usize>,
    sigma: f64,
    weights: Option<numpy::PyArrayLike1<'py, f64>>,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>, Vec<Vec<f64>>, f64)> {
    let (deviance, phi, b, sigma, _) = nlmm_deviance_with_status(
        theta,
        y,
        x,
        groups,
        model_name,
        phi,
        b,
        random_params,
        sigma,
        weights,
        maxiter,
        tol,
    )?;
    Ok((deviance, phi, b, sigma))
}

#[pyfunction]
#[pyo3(signature = (
    theta,
    y,
    x,
    groups,
    model_name,
    phi,
    b,
    random_params,
    sigma,
    weights=None,
    maxiter=PNLS_MAX_ITER,
    tol=PNLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn nlmm_deviance_with_status<'py>(
    theta: numpy::PyArrayLike1<'py, f64>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike1<'py, f64>,
    groups: numpy::PyArrayLike1<'py, i64>,
    model_name: &str,
    phi: numpy::PyArrayLike1<'py, f64>,
    b: numpy::PyArrayLike2<'py, f64>,
    random_params: Vec<usize>,
    sigma: f64,
    weights: Option<numpy::PyArrayLike1<'py, f64>>,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>, Vec<Vec<f64>>, f64, bool)> {
    validate_pnls_controls(maxiter, tol).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let model = match model_name.to_lowercase().as_str() {
        "ssasymp" => NlmeModel::SSasymp,
        "sslogis" => NlmeModel::SSlogis,
        "ssmicmen" => NlmeModel::SSmicmen,
        "ssfpl" => NlmeModel::SSfpl,
        "ssgompertz" => NlmeModel::SSgompertz,
        "ssbiexp" => NlmeModel::SSbiexp,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model: {}. Supported models: SSasymp, SSlogis, SSmicmen, SSfpl, SSgompertz, SSbiexp",
                model_name
            )));
        }
    };

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let groups_arr = groups.as_array();
    let phi_arr = phi.as_array();
    let b_arr = b.as_array();
    let theta_arr = theta.as_array();

    let y_vec: Vec<f64> = y_arr.iter().cloned().collect();
    let x_vec: Vec<f64> = x_arr.iter().cloned().collect();
    let groups_vec: Vec<i64> = groups_arr.iter().cloned().collect();
    let phi_vec: Vec<f64> = phi_arr.iter().cloned().collect();
    let theta_vec: Vec<f64> = theta_arr.iter().cloned().collect();
    let weights_vec = match weights {
        Some(weights) => weights.as_array().iter().copied().collect(),
        None => vec![1.0; y_vec.len()],
    };
    validate_prior_weights(&weights_vec, y_vec.len())?;

    let n_groups = b_arr.nrows();
    let n_random = b_arr.ncols();
    let b_mat = DMatrix::from_fn(n_groups, n_random, |i, j| b_arr[[i, j]]);

    let (deviance, phi_new, b_new, sigma_new, converged) = nlmm_deviance_with_status_impl(
        &theta_vec,
        &y_vec,
        &x_vec,
        &groups_vec,
        &weights_vec,
        model,
        &phi_vec,
        &b_mat,
        &random_params,
        sigma,
        maxiter,
        tol,
    );

    let b_out: Vec<Vec<f64>> = (0..n_groups)
        .map(|i| (0..n_random).map(|j| b_new[(i, j)]).collect())
        .collect();

    Ok((deviance, phi_new, b_out, sigma_new, converged))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn micmen_full_joint_step(
        x: &[f64],
        y: &[f64],
        groups: &[i64],
        weights: &[f64],
        phi: &[f64],
        b: &DMatrix<f64>,
    ) -> DVector<f64> {
        let precision = 1.0 / (0.25 + 1e-8);
        let design = DMatrix::from_fn(x.len(), 2 + b.nrows(), |i, j| {
            let group = groups[i] as usize;
            let derivative = match j {
                0 => x[i] / (phi[1] + x[i]),
                1 => -(phi[0] + b[(group, 0)]) * x[i] / (phi[1] + x[i]).powi(2),
                _ if j == 2 + group => x[i] / (phi[1] + x[i]),
                _ => 0.0,
            };
            derivative * weights[i].sqrt()
        });
        let residual = DVector::from_fn(x.len(), |i| {
            (y[i] - (phi[0] + b[(groups[i] as usize, 0)]) * x[i] / (phi[1] + x[i]))
                * weights[i].sqrt()
        });
        let mut normal = design.transpose() * &design;
        let mut rhs = design.transpose() * residual;
        for j in 0..2 {
            normal[(j, j)] += 1e-6;
        }
        for g in 0..b.nrows() {
            normal[(g + 2, g + 2)] += precision;
            rhs[g + 2] -= precision * b[(g, 0)];
        }
        Llt::new(normal.as_ref(), Side::Lower).unwrap().solve(&rhs)
    }

    fn micmen_score(x: &[f64], y: &[f64], groups: &[i64], phi: &[f64], b: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, x)| {
                let residual = y[i] - (phi[0] + b[groups[i] as usize]) * x / (phi[1] + x);
                residual * residual
            })
            .sum::<f64>()
            + b.iter().map(|b| b * b / (0.25 + 1e-8)).sum::<f64>()
    }

    #[test]
    fn joint_pnls_step_matches_an_independent_full_normal_system() {
        let x: Vec<f64> = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
            .into_iter()
            .cycle()
            .take(18)
            .collect();
        let groups: Vec<i64> = (0..18).map(|i| i / 6).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, x)| (2.8 + 0.3 * groups[i] as f64) * x / (0.9 + x))
            .collect();
        let weights: Vec<f64> = (0..18).map(|i| 0.5 + i as f64 / 17.0).collect();
        let phi = [2.0, 1.2];
        let b = DMatrix::from_fn(3, 1, |g, _| (g as f64 - 1.0) * 0.1);
        let delta = micmen_full_joint_step(&x, &y, &groups, &weights, &phi, &b);
        let result = pnls_step_impl(
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSmicmen,
            &phi,
            &b,
            &DMatrix::from_fn(1, 1, |_, _| 0.25),
            1.0,
            &[0],
            1,
            1e-12,
        );
        assert!(!result.converged);
        for j in 0..2 {
            assert!((result.phi[j] - phi[j] - delta[j]).abs() < 1e-12);
        }
        for g in 0..3 {
            assert!((result.b[(g, 0)] - b[(g, 0)] - delta[g + 2]).abs() < 1e-12);
        }
    }

    #[test]
    fn joint_pnls_backtracks_when_the_full_step_worsens_the_objective() {
        let x: Vec<f64> = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
            .into_iter()
            .cycle()
            .take(12)
            .collect();
        let groups: Vec<i64> = (0..12).map(|i| i / 6).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, x)| (2.8 + 0.4 * groups[i] as f64) * x / (1.0 + x))
            .collect();
        let weights = vec![1.0; 12];
        let phi = [1.0, 4.0];
        let b = DMatrix::zeros(2, 1);
        let delta = micmen_full_joint_step(&x, &y, &groups, &weights, &phi, &b);
        let initial = micmen_score(&x, &y, &groups, &phi, &[0.0, 0.0]);
        let full = micmen_score(
            &x,
            &y,
            &groups,
            &[phi[0] + delta[0], phi[1] + delta[1]],
            &[delta[2], delta[3]],
        );
        assert!(full > initial);
        let result = pnls_step_impl(
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSmicmen,
            &phi,
            &b,
            &DMatrix::from_fn(1, 1, |_, _| 0.25),
            1.0,
            &[0],
            1,
            1e-12,
        );
        for j in 0..2 {
            assert!((result.phi[j] - phi[j] - 0.125 * delta[j]).abs() < 1e-11);
        }
        assert!(result.sigma_sq * (y.len() as f64) < initial);
        assert!(!result.converged);
    }

    #[test]
    fn pnls_controls_separate_iteration_limit_from_convergence() {
        let grid = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0];
        let x: Vec<f64> = grid.into_iter().cycle().take(18).collect();
        let groups: Vec<i64> = (0..18).map(|row| row / 6).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(row, x)| (2.8 + 0.3 * groups[row] as f64) * x / (0.9 + x))
            .collect();
        let weights: Vec<f64> = (0..18).map(|row| 0.5 + row as f64 / 17.0).collect();
        let b = DMatrix::from_fn(3, 1, |row, _| (row as f64 - 1.0) * 0.1);
        let evaluate = |maxiter, tol| {
            nlmm_deviance_with_status_impl(
                &[0.4],
                &y,
                &x,
                &groups,
                &weights,
                NlmeModel::SSmicmen,
                &[2.0, 1.2],
                &b,
                &[0],
                0.3,
                maxiter,
                tol,
            )
        };
        let limited = evaluate(1, 1e-12);
        let loose = evaluate(1, 1e6);
        let complete = evaluate(1000, 1e-10);
        assert!(!limited.4);
        assert!(loose.4 && complete.4);
        assert_eq!(limited.0, loose.0);
        assert_eq!(limited.1, loose.1);
        assert_eq!(limited.3, loose.3);
        for row in 0..3 {
            assert_eq!(limited.2[(row, 0)], loose.2[(row, 0)]);
        }
        assert!((complete.1[0] - limited.1[0]).abs() > 1e-3);
    }

    #[test]
    fn pnls_controls_reject_invalid_limits_and_tolerances() {
        assert!(validate_pnls_controls(0, 1e-6).is_err());
        for tol in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(validate_pnls_controls(50, tol).is_err());
        }
        assert!(validate_pnls_controls(1, 1e-6).is_ok());
    }

    #[test]
    fn group_indices_preserve_label_and_observation_order() {
        let groups = [i64::MAX, -3, i64::MIN, -3, i64::MAX, i64::MIN];
        assert_eq!(
            grouped_observation_indices(&groups),
            vec![vec![2, 5], vec![1, 3], vec![0, 4]]
        );
        assert!(grouped_observation_indices(&[]).is_empty());
    }

    #[test]
    fn group_indices_handle_many_interleaved_labels() {
        let groups: Vec<i64> = (0..10_000).map(|row| ((row * 7) % 1000) - 400).collect();
        let indices = grouped_observation_indices(&groups);
        assert_eq!(indices.len(), 1000);
        for (group, rows) in indices.iter().enumerate() {
            assert_eq!(rows.len(), 10);
            assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
            for row in rows {
                assert_eq!(groups[*row], group as i64 - 400);
            }
        }
    }

    fn asymptotic_data() -> (Vec<f64>, Vec<f64>, Vec<i64>) {
        let base = [10.0, 0.5, -0.5];
        let group_effects = [-1.0, -0.3, 0.4, 1.0];
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();

        for (group, effect) in group_effects.iter().enumerate() {
            let params = [base[0] + effect, base[1], base[2]];
            for observation in 0..10 {
                let x_value = observation as f64 * 5.0 / 9.0;
                let noise = (observation as f64 % 3.0 - 1.0) * 0.05;
                x.push(x_value);
                y.push(NlmeModel::SSasymp.predict(&params, &[x_value])[0] + noise);
                groups.push(group as i64);
            }
        }

        (x, y, groups)
    }

    #[test]
    fn nlmm_deviance_is_repeatable_and_finite() {
        let (x, y, groups) = asymptotic_data();
        let weights = vec![1.0; y.len()];
        let phi = vec![10.0, 0.5, -0.5];
        let b = DMatrix::zeros(4, 1);
        let theta = vec![1.0];

        let first = nlmm_deviance_impl(
            &theta,
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );
        let repeated = nlmm_deviance_impl(
            &theta,
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );

        assert!(first.0.is_finite());
        assert!(first.3.is_finite() && first.3 > 0.0);
        assert!((first.0 - repeated.0).abs() < 1e-12);
    }

    #[test]
    fn laplace_correction_avoids_collapsed_variance_optimum() {
        let (x, y, groups) = asymptotic_data();
        let weights = vec![1.0; y.len()];
        let phi = vec![10.0, 0.5, -0.5];
        let b = DMatrix::zeros(4, 1);

        let collapsed = nlmm_deviance_impl(
            &[1e-6],
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );
        let nonzero = nlmm_deviance_impl(
            &[1.0],
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );

        assert!(collapsed.0.is_finite());
        assert!(nonzero.0 < collapsed.0);
    }

    #[test]
    fn prior_weights_downweight_an_outlier() {
        let (x, y, groups) = asymptotic_data();
        let weights = vec![1.0; y.len()];
        let phi = vec![10.0, 0.5, -0.5];
        let b = DMatrix::zeros(4, 1);
        let theta = vec![1.0];

        let clean = nlmm_deviance_impl(
            &theta,
            &y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );

        let mut contaminated_y = y.clone();
        contaminated_y[0] += 50.0;
        let unweighted = nlmm_deviance_impl(
            &theta,
            &contaminated_y,
            &x,
            &groups,
            &weights,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );

        let mut downweighted = weights.clone();
        downweighted[0] = 1e-5;
        let weighted = nlmm_deviance_impl(
            &theta,
            &contaminated_y,
            &x,
            &groups,
            &downweighted,
            NlmeModel::SSasymp,
            &phi,
            &b,
            &[0],
            0.3,
        );

        let distance = |estimate: &[f64], reference: &[f64]| {
            estimate
                .iter()
                .zip(reference.iter())
                .map(|(estimate, reference)| (estimate - reference).powi(2))
                .sum::<f64>()
                .sqrt()
        };

        assert!(distance(&weighted.1, &clean.1) < distance(&unweighted.1, &clean.1));
    }
}

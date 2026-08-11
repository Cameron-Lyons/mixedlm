use nalgebra::{Cholesky, DMatrix, DVector};
use pyo3::PyResult;
use pyo3::prelude::*;

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
                .iter()
                .all(|value| value.is_finite())
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
                .iter()
                .all(|value| value.is_finite())
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
        return DMatrix::identity(n_random, n_random);
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
        DMatrix::from_diagonal(&DVector::from_iterator(
            n_random,
            theta.iter().take(n_random).cloned(),
        ))
    }
}

fn build_psi_matrix(theta: &[f64], n_random: usize) -> DMatrix<f64> {
    let l = build_psi_factor(theta, n_random);
    &l * l.transpose()
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
    let mut unique_groups = groups.to_vec();
    unique_groups.sort_unstable();
    unique_groups.dedup();
    unique_groups
        .iter()
        .map(|group| {
            groups
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| (candidate == group).then_some(index))
                .collect()
        })
        .collect()
}

pub struct PnlsResult {
    pub phi: Vec<f64>,
    pub b: DMatrix<f64>,
    pub sigma: f64,
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
) -> PnlsResult {
    let n = y.len();
    let group_indices = grouped_observation_indices(groups);
    let n_phi = phi.len();
    let n_random = random_params.len();
    let sqrt_weights: Vec<f64> = weights.iter().map(|weight| weight.sqrt()).collect();

    let psi_reg = {
        let mut m = psi.clone();
        for i in 0..n_random {
            m[(i, i)] += 1e-8;
        }
        m
    };

    let psi_inv = match psi_reg.clone().try_inverse() {
        Some(inv) => inv,
        None => DMatrix::identity(n_random, n_random),
    };

    let mut phi_new: Vec<f64> = phi.to_vec();
    let mut b_new = b.clone();

    for _iteration in 0..PNLS_MAX_ITER {
        let phi_previous = phi_new.clone();
        let b_previous = b_new.clone();
        let mut resid_total = vec![0.0; n];
        let mut grad_total = DMatrix::zeros(n, n_phi);

        for (g_idx, mask) in group_indices.iter().enumerate() {
            let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
            let y_g: Vec<f64> = mask.iter().map(|&i| y[i]).collect();

            let mut params_g = phi_new.clone();
            for (j, &p_idx) in random_params.iter().enumerate() {
                params_g[p_idx] += b_new[(g_idx, j)];
            }

            let pred_g = model.predict(&params_g, &x_g);
            let grad_g = model.gradient(&params_g, &x_g);

            for (local_i, &global_i) in mask.iter().enumerate() {
                resid_total[global_i] = y_g[local_i] - pred_g[local_i];
                for p in 0..n_phi {
                    grad_total[(global_i, p)] = grad_g[(local_i, p)];
                }
            }
        }

        for i in 0..n {
            let sqrt_weight = sqrt_weights[i];
            resid_total[i] *= sqrt_weight;
            for p in 0..n_phi {
                grad_total[(i, p)] *= sqrt_weight;
            }
        }

        let gtg = grad_total.transpose() * &grad_total;
        let gtr: DVector<f64> =
            grad_total.transpose() * DVector::from_iterator(n, resid_total.iter().cloned());

        let gtg_reg = {
            let mut m = gtg.clone();
            for i in 0..n_phi {
                m[(i, i)] += 1e-6;
            }
            m
        };

        let delta_phi = match Cholesky::new(gtg_reg.clone()) {
            Some(chol) => chol.solve(&gtr),
            None => gtg_reg
                .try_inverse()
                .map_or(DVector::zeros(n_phi), |inv| inv * &gtr),
        };

        for i in 0..n_phi {
            phi_new[i] += 0.5 * delta_phi[i];
        }

        for (g_idx, mask) in group_indices.iter().enumerate() {
            let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
            let y_g: Vec<f64> = mask.iter().map(|&i| y[i]).collect();

            let mut params_g = phi_new.clone();
            for (j, &p_idx) in random_params.iter().enumerate() {
                params_g[p_idx] += b_new[(g_idx, j)];
            }

            let pred_g = model.predict(&params_g, &x_g);
            let grad_g = model.gradient(&params_g, &x_g);

            let n_g = mask.len();
            let mut z_g = DMatrix::zeros(n_g, n_random);
            for i in 0..n_g {
                for (j, &p_idx) in random_params.iter().enumerate() {
                    z_g[(i, j)] = grad_g[(i, p_idx)];
                }
            }

            let b_g: DVector<f64> = DVector::from_fn(n_random, |j, _| b_new[(g_idx, j)]);

            let mut resid_g = DVector::zeros(n_g);
            for i in 0..n_g {
                resid_g[i] = y_g[i] - pred_g[i];
                for j in 0..n_random {
                    resid_g[i] += z_g[(i, j)] * b_g[j];
                }
            }

            let mut weighted_z_g = z_g.clone();
            let mut weighted_resid_g = resid_g.clone();
            for i in 0..n_g {
                let sqrt_weight = sqrt_weights[mask[i]];
                weighted_resid_g[i] *= sqrt_weight;
                for j in 0..n_random {
                    weighted_z_g[(i, j)] *= sqrt_weight;
                }
            }

            let ztz = weighted_z_g.transpose() * &weighted_z_g;
            let ztr = weighted_z_g.transpose() * &weighted_resid_g;

            let c = &ztz + &psi_inv;

            let b_g_new = match Cholesky::new(c.clone()) {
                Some(chol) => chol.solve(&ztr),
                None => c
                    .try_inverse()
                    .map_or(DVector::zeros(n_random), |inv| inv * &ztr),
            };

            for j in 0..n_random {
                b_new[(g_idx, j)] = b_g_new[j];
            }
        }

        let max_delta: f64 = phi_new
            .iter()
            .zip(phi_previous.iter())
            .map(|(a, b)| (a - b).abs())
            .chain(
                b_new
                    .iter()
                    .zip(b_previous.iter())
                    .map(|(a, b)| (a - b).abs()),
            )
            .fold(0.0, f64::max);

        if max_delta < PNLS_TOLERANCE {
            break;
        }
    }

    let mut rss = 0.0;
    for (g_idx, mask) in group_indices.iter().enumerate() {
        let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
        let y_g: Vec<f64> = mask.iter().map(|&i| y[i]).collect();

        let mut params_g = phi_new.clone();
        for (j, &p_idx) in random_params.iter().enumerate() {
            params_g[p_idx] += b_new[(g_idx, j)];
        }

        let pred_g = model.predict(&params_g, &x_g);

        for i in 0..mask.len() {
            let r = y_g[i] - pred_g[i];
            rss += weights[mask[i]] * r * r;
        }
    }

    let mut penalty = 0.0;
    for g_idx in 0..group_indices.len() {
        let b_g: DVector<f64> = DVector::from_fn(n_random, |j, _| b_new[(g_idx, j)]);
        penalty += b_g.dot(&(&psi_inv * &b_g));
    }

    let sigma_new = ((rss + penalty) / n as f64).max(f64::MIN_POSITIVE).sqrt();

    PnlsResult {
        phi: phi_new,
        b: b_new,
        sigma: sigma_new,
    }
}

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
    sigma: f64,
) -> (f64, Vec<f64>, DMatrix<f64>, f64) {
    let n = y.len();
    let group_indices = grouped_observation_indices(groups);
    let n_groups = group_indices.len();
    let n_random = random_params.len();
    let sqrt_weights: Vec<f64> = weights.iter().map(|weight| weight.sqrt()).collect();

    let psi_factor = build_psi_factor(theta, n_random);
    let psi = &psi_factor * psi_factor.transpose();

    let result = pnls_step_impl(
        y,
        x,
        groups,
        weights,
        model,
        phi,
        b,
        &psi,
        sigma,
        random_params,
    );

    let phi_new = result.phi;
    let b_new = result.b;

    let mut rss = 0.0;
    for (g_idx, mask) in group_indices.iter().enumerate() {
        let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
        let y_g: Vec<f64> = mask.iter().map(|&i| y[i]).collect();

        let mut params_g = phi_new.clone();
        for (j, &p_idx) in random_params.iter().enumerate() {
            params_g[p_idx] += b_new[(g_idx, j)];
        }

        let pred_g = model.predict(&params_g, &x_g);

        for i in 0..mask.len() {
            let r = y_g[i] - pred_g[i];
            rss += weights[mask[i]] * r * r;
        }
    }

    let psi_reg = {
        let mut m = psi.clone();
        for i in 0..n_random {
            m[(i, i)] += 1e-8;
        }
        m
    };

    let psi_inv = match psi_reg.clone().try_inverse() {
        Some(inv) => inv,
        None => DMatrix::identity(n_random, n_random),
    };

    let mut penalty = 0.0;
    for g_idx in 0..n_groups {
        let b_g: DVector<f64> = DVector::from_fn(n_random, |j, _| b_new[(g_idx, j)]);
        penalty += b_g.dot(&(&psi_inv * &b_g));
    }

    let sigma_sq = ((rss + penalty) / n as f64).max(f64::MIN_POSITIVE);
    let mut laplace_correction = 0.0;
    let identity = DMatrix::identity(n_random, n_random);

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
        let logdet = match Cholesky::new(system.clone()) {
            Some(chol) => {
                let l = chol.l();
                2.0 * (0..n_random).map(|i| l[(i, i)].ln()).sum::<f64>()
            }
            None => {
                let eigenvalues = system.symmetric_eigenvalues();
                if eigenvalues
                    .iter()
                    .any(|value| *value <= 0.0 || !value.is_finite())
                {
                    return (1e100, phi_new, b_new, sigma_sq.sqrt());
                }
                eigenvalues.iter().map(|value| value.ln()).sum()
            }
        };
        laplace_correction += logdet;
    }

    let deviance =
        n as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma_sq).ln()) + laplace_correction;

    (deviance, phi_new, b_new, sigma_sq.sqrt())
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
    weights=None
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
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, f64)> {
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
    );

    let b_out: Vec<Vec<f64>> = (0..n_groups)
        .map(|i| (0..n_random).map(|j| result.b[(i, j)]).collect())
        .collect();

    Ok((result.phi, b_out, result.sigma))
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
    weights=None
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
) -> PyResult<(f64, Vec<f64>, Vec<Vec<f64>>, f64)> {
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

    let (deviance, phi_new, b_new, sigma_new) = nlmm_deviance_impl(
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
    );

    let b_out: Vec<Vec<f64>> = (0..n_groups)
        .map(|i| (0..n_random).map(|j| b_new[(i, j)]).collect())
        .collect();

    Ok((deviance, phi_new, b_out, sigma_new))
}

#[cfg(test)]
mod tests {
    use super::*;

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

use nalgebra::{Cholesky, DMatrix, DVector};
use pyo3::PyResult;
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NlmeModel {
    SSasymp,
    SSlogis,
    SSmicmen,
    SSfpl,
    SSgompertz,
    SSbiexp,
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
                    result[i] = asym / (1.0 + ((xmid - x[i]) / scal).exp());
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
                    result[i] = a + (b - a) / (1.0 + ((xmid - x[i]) / scal).exp());
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
                    let exp_term = ((xmid - x[i]) / scal).exp();
                    let denom = 1.0 + exp_term;
                    let denom_sq = denom * denom;
                    grad[(i, 0)] = 1.0 / denom;
                    grad[(i, 1)] = -asym * exp_term / (scal * denom_sq);
                    grad[(i, 2)] = asym * (xmid - x[i]) * exp_term / (scal * scal * denom_sq);
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
                    let exp_term = ((xmid - x[i]) / scal).exp();
                    let denom = 1.0 + exp_term;
                    let denom_sq = denom * denom;
                    grad[(i, 0)] = 1.0 - 1.0 / denom;
                    grad[(i, 1)] = 1.0 / denom;
                    grad[(i, 2)] = -(b - a) * exp_term / (scal * denom_sq);
                    grad[(i, 3)] = (b - a) * (xmid - x[i]) * exp_term / (scal * scal * denom_sq);
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

fn build_psi_matrix(theta: &[f64], n_random: usize) -> DMatrix<f64> {
    if theta.is_empty() {
        return DMatrix::identity(n_random, n_random);
    }

    let n_theta = theta.len();
    let q = ((-1.0 + (1.0 + 8.0 * n_theta as f64).sqrt()) / 2.0) as usize;

    let l = if q * (q + 1) / 2 == n_theta {
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
    };

    &l * l.transpose()
}

pub struct PnlsResult {
    pub phi: Vec<f64>,
    pub b: DMatrix<f64>,
    pub sigma: f64,
}

pub fn pnls_step_impl(
    y: &[f64],
    x: &[f64],
    groups: &[i64],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    psi: &DMatrix<f64>,
    sigma: f64,
    random_params: &[usize],
) -> PnlsResult {
    let n = y.len();
    let unique_groups: Vec<i64> = {
        let mut v: Vec<i64> = groups.to_vec();
        v.sort();
        v.dedup();
        v
    };
    let _n_groups = unique_groups.len();
    let n_phi = phi.len();
    let n_random = random_params.len();

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

    for _iteration in 0..10 {
        let mut resid_total = vec![0.0; n];
        let mut grad_total = DMatrix::zeros(n, n_phi);

        for (g_idx, &g) in unique_groups.iter().enumerate() {
            let mask: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|&(_, grp)| *grp == g)
                .map(|(i, _)| i)
                .collect();

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

        for (g_idx, &g) in unique_groups.iter().enumerate() {
            let mask: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|&(_, grp)| *grp == g)
                .map(|(i, _)| i)
                .collect();

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

            let ztz = z_g.transpose() * &z_g;
            let ztr = z_g.transpose() * &resid_g;

            let sigma_sq = sigma * sigma;
            let c = &ztz / sigma_sq + &psi_inv;

            let b_g_new = match Cholesky::new(c.clone()) {
                Some(chol) => chol.solve(&(&ztr / sigma_sq)),
                None => c
                    .try_inverse()
                    .map_or(DVector::zeros(n_random), |inv| inv * &ztr / sigma_sq),
            };

            for j in 0..n_random {
                b_new[(g_idx, j)] = b_g_new[j];
            }
        }

        let max_delta: f64 = phi_new
            .iter()
            .zip(phi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        if max_delta < 1e-6 {
            break;
        }
    }

    let mut rss = 0.0;
    for (g_idx, &g) in unique_groups.iter().enumerate() {
        let mask: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|&(_, grp)| *grp == g)
            .map(|(i, _)| i)
            .collect();

        let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
        let y_g: Vec<f64> = mask.iter().map(|&i| y[i]).collect();

        let mut params_g = phi_new.clone();
        for (j, &p_idx) in random_params.iter().enumerate() {
            params_g[p_idx] += b_new[(g_idx, j)];
        }

        let pred_g = model.predict(&params_g, &x_g);

        for i in 0..mask.len() {
            let r = y_g[i] - pred_g[i];
            rss += r * r;
        }
    }

    let sigma_new = (rss / n as f64).sqrt();

    PnlsResult {
        phi: phi_new,
        b: b_new,
        sigma: sigma_new,
    }
}

pub fn nlmm_deviance_impl(
    theta: &[f64],
    y: &[f64],
    x: &[f64],
    groups: &[i64],
    model: NlmeModel,
    phi: &[f64],
    b: &DMatrix<f64>,
    random_params: &[usize],
    sigma: f64,
) -> (f64, Vec<f64>, DMatrix<f64>, f64) {
    let n = y.len();
    let unique_groups: Vec<i64> = {
        let mut v: Vec<i64> = groups.to_vec();
        v.sort();
        v.dedup();
        v
    };
    let n_groups = unique_groups.len();
    let n_random = random_params.len();

    let psi = build_psi_matrix(theta, n_random);

    let result = pnls_step_impl(y, x, groups, model, phi, b, &psi, sigma, random_params);

    let phi_new = result.phi;
    let b_new = result.b;
    let sigma_new = result.sigma;

    let mut rss = 0.0;
    for (g_idx, &g) in unique_groups.iter().enumerate() {
        let mask: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|&(_, grp)| *grp == g)
            .map(|(i, _)| i)
            .collect();

        let x_g: Vec<f64> = mask.iter().map(|&i| x[i]).collect();
        let y_g: Vec<f64> = mask.iter().map(|&i| y[i]).collect();

        let mut params_g = phi_new.clone();
        for (j, &p_idx) in random_params.iter().enumerate() {
            params_g[p_idx] += b_new[(g_idx, j)];
        }

        let pred_g = model.predict(&params_g, &x_g);

        for i in 0..mask.len() {
            let r = y_g[i] - pred_g[i];
            rss += r * r;
        }
    }

    let sigma_sq = sigma_new * sigma_new;
    let mut deviance = n as f64 * (2.0 * std::f64::consts::PI * sigma_sq).ln() + rss / sigma_sq;

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

    for g_idx in 0..n_groups {
        let b_g: DVector<f64> = DVector::from_fn(n_random, |j, _| b_new[(g_idx, j)]);
        let penalty = b_g.dot(&(&psi_inv * &b_g));
        deviance += penalty;
    }

    let (sign, logdet) = {
        match Cholesky::new(psi.clone()) {
            Some(chol) => {
                let l = chol.l();
                let logdet = 2.0 * (0..n_random).map(|i| l[(i, i)].ln()).sum::<f64>();
                (1.0, logdet)
            }
            None => {
                let eigvals = psi.symmetric_eigenvalues();
                let logdet: f64 = eigvals.iter().map(|&e| e.max(1e-10).ln()).sum();
                (1.0, logdet)
            }
        }
    };

    if sign > 0.0 {
        deviance += n_groups as f64 * logdet;
    }

    (deviance, phi_new, b_new, sigma_new)
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
    random_params
))]
pub fn pnls_step(
    y: numpy::PyReadonlyArray1<'_, f64>,
    x: numpy::PyReadonlyArray1<'_, f64>,
    groups: numpy::PyReadonlyArray1<'_, i64>,
    model_name: &str,
    phi: numpy::PyReadonlyArray1<'_, f64>,
    b: numpy::PyReadonlyArray2<'_, f64>,
    theta: numpy::PyReadonlyArray1<'_, f64>,
    sigma: f64,
    random_params: Vec<usize>,
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
    let psi = build_psi_matrix(&theta_vec, n_random);

    let result = pnls_step_impl(
        &y_vec,
        &x_vec,
        &groups_vec,
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
    sigma
))]
pub fn nlmm_deviance(
    theta: numpy::PyReadonlyArray1<'_, f64>,
    y: numpy::PyReadonlyArray1<'_, f64>,
    x: numpy::PyReadonlyArray1<'_, f64>,
    groups: numpy::PyReadonlyArray1<'_, i64>,
    model_name: &str,
    phi: numpy::PyReadonlyArray1<'_, f64>,
    b: numpy::PyReadonlyArray2<'_, f64>,
    random_params: Vec<usize>,
    sigma: f64,
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

    let n_groups = b_arr.nrows();
    let n_random = b_arr.ncols();
    let b_mat = DMatrix::from_fn(n_groups, n_random, |i, j| b_arr[[i, j]]);

    let (deviance, phi_new, b_new, sigma_new) = nlmm_deviance_impl(
        &theta_vec,
        &y_vec,
        &x_vec,
        &groups_vec,
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

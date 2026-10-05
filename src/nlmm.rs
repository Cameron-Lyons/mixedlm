use faer::linalg::solvers::{DenseSolveCore, Llt, Solve};
use faer::{Mat as DMatrix, Side};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::ops::Range;

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
    fn from_name(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "ssasymp" => Ok(NlmeModel::SSasymp),
            "sslogis" => Ok(NlmeModel::SSlogis),
            "ssmicmen" => Ok(NlmeModel::SSmicmen),
            "ssfpl" => Ok(NlmeModel::SSfpl),
            "ssgompertz" => Ok(NlmeModel::SSgompertz),
            "ssbiexp" => Ok(NlmeModel::SSbiexp),
            _ => Err(PyValueError::new_err(format!(
                "Unknown model: {}. Supported models: SSasymp, SSlogis, SSmicmen, SSfpl, SSgompertz, SSbiexp",
                name
            ))),
        }
    }

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

    /// Writes the prediction at each `x` into `predicted`.
    fn predict_into(&self, params: &[f64], x: &[f64], predicted: &mut [f64]) {
        let points = predicted.iter_mut().zip(x);

        match self {
            NlmeModel::SSasymp => {
                let asym = params[0];
                let r0 = params[1];
                let lrc = params[2];
                let rc = lrc.exp();
                for (result, &x) in points {
                    *result = asym + (r0 - asym) * (-rc * x).exp();
                }
            }
            NlmeModel::SSlogis => {
                let asym = params[0];
                let xmid = params[1];
                let scal = params[2];
                for (result, &x) in points {
                    *result = asym * logistic((x - xmid) / scal);
                }
            }
            NlmeModel::SSmicmen => {
                let vm = params[0];
                let k = params[1];
                for (result, &x) in points {
                    *result = vm * x / (k + x);
                }
            }
            NlmeModel::SSfpl => {
                let a = params[0];
                let b = params[1];
                let xmid = params[2];
                let scal = params[3];
                for (result, &x) in points {
                    *result = a + (b - a) * logistic((x - xmid) / scal);
                }
            }
            NlmeModel::SSgompertz => {
                let asym = params[0];
                let b2 = params[1];
                let b3 = params[2];
                for (result, &x) in points {
                    *result = asym * (-b2 * b3.powf(x)).exp();
                }
            }
            NlmeModel::SSbiexp => {
                let a1 = params[0];
                let lrc1 = params[1];
                let a2 = params[2];
                let lrc2 = params[3];
                let rc1 = lrc1.exp();
                let rc2 = lrc2.exp();
                for (result, &x) in points {
                    *result = a1 * (-rc1 * x).exp() + a2 * (-rc2 * x).exp();
                }
            }
        }
    }

    /// Writes the prediction at each `x` and the row-major `x.len()` x
    /// `n_params` gradient, sharing the terms common to both.
    fn linearize_into(
        &self,
        params: &[f64],
        x: &[f64],
        predicted: &mut [f64],
        gradient: &mut [f64],
    ) {
        let points = predicted
            .iter_mut()
            .zip(gradient.chunks_exact_mut(self.n_params()))
            .zip(x);

        match self {
            NlmeModel::SSasymp => {
                let asym = params[0];
                let r0 = params[1];
                let lrc = params[2];
                let rc = lrc.exp();
                for ((result, grad), &x) in points {
                    let exp_term = (-rc * x).exp();
                    *result = asym + (r0 - asym) * exp_term;
                    grad[0] = 1.0 - exp_term;
                    grad[1] = exp_term;
                    grad[2] = -(r0 - asym) * rc * x * exp_term;
                }
            }
            NlmeModel::SSlogis => {
                let asym = params[0];
                let xmid = params[1];
                let scal = params[2];
                for ((result, grad), &x) in points {
                    let fraction = logistic((x - xmid) / scal);
                    let sensitivity = fraction * (1.0 - fraction);
                    *result = asym * fraction;
                    grad[0] = fraction;
                    grad[1] = -asym * sensitivity / scal;
                    grad[2] = asym * (xmid - x) * sensitivity / (scal * scal);
                }
            }
            NlmeModel::SSmicmen => {
                let vm = params[0];
                let k = params[1];
                for ((result, grad), &x) in points {
                    let denom = k + x;
                    let denom_sq = denom * denom;
                    *result = vm * x / denom;
                    grad[0] = x / denom;
                    grad[1] = -vm * x / denom_sq;
                }
            }
            NlmeModel::SSfpl => {
                let a = params[0];
                let b = params[1];
                let xmid = params[2];
                let scal = params[3];
                for ((result, grad), &x) in points {
                    let fraction = logistic((x - xmid) / scal);
                    let sensitivity = fraction * (1.0 - fraction);
                    *result = a + (b - a) * fraction;
                    grad[0] = 1.0 - fraction;
                    grad[1] = fraction;
                    grad[2] = -(b - a) * sensitivity / scal;
                    grad[3] = (b - a) * (xmid - x) * sensitivity / (scal * scal);
                }
            }
            NlmeModel::SSgompertz => {
                let asym = params[0];
                let b2 = params[1];
                let b3 = params[2];
                for ((result, grad), &x) in points {
                    let b3_x = b3.powf(x);
                    let exp_term = (-b2 * b3_x).exp();
                    *result = asym * exp_term;
                    grad[0] = exp_term;
                    grad[1] = -asym * b3_x * exp_term;
                    grad[2] = -asym * b2 * x * b3.powf(x - 1.0) * exp_term;
                }
            }
            NlmeModel::SSbiexp => {
                let a1 = params[0];
                let lrc1 = params[1];
                let a2 = params[2];
                let lrc2 = params[3];
                let rc1 = lrc1.exp();
                let rc2 = lrc2.exp();
                for ((result, grad), &x) in points {
                    let exp1 = (-rc1 * x).exp();
                    let exp2 = (-rc2 * x).exp();
                    *result = a1 * exp1 + a2 * exp2;
                    grad[0] = exp1;
                    grad[1] = -a1 * rc1 * x * exp1;
                    grad[2] = exp2;
                    grad[3] = -a2 * rc2 * x * exp2;
                }
            }
        }
    }

    #[cfg(test)]
    fn predict(&self, params: &[f64], x: &[f64]) -> Vec<f64> {
        let mut predicted = vec![0.0; x.len()];
        self.predict_into(params, x, &mut predicted);
        predicted
    }

    #[cfg(test)]
    fn gradient(&self, params: &[f64], x: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; x.len() * self.n_params()];
        self.linearize_into(params, x, &mut vec![0.0; x.len()], &mut gradient);
        gradient
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

fn invert_regularized(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    match Llt::new(matrix.as_ref(), Side::Lower) {
        Ok(cholesky) => cholesky.inverse(),
        Err(_) => matrix.partial_piv_lu().inverse(),
    }
}

fn row_major(matrix: &DMatrix<f64>) -> Vec<f64> {
    (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |column| matrix[(row, column)]))
        .collect()
}

/// Factors the row-major `n` x `n` SPD `matrix` as L L' into the lower triangle
/// of `factor`, rejecting the non-positive or non-finite pivots that faer does.
fn cholesky_into(matrix: &[f64], n: usize, factor: &mut [f64]) -> bool {
    factor[..n * n].copy_from_slice(&matrix[..n * n]);
    for j in 0..n {
        for i in j..n {
            let sum: f64 = (0..j).map(|k| factor[j * n + k] * factor[i * n + k]).sum();
            factor[i * n + j] -= sum;
        }
        let pivot = factor[j * n + j];
        if !(pivot.is_finite() && pivot > 0.0) {
            return false;
        }
        let root = pivot.sqrt();
        for i in j..n {
            factor[i * n + j] /= root;
        }
    }
    true
}

/// Overwrites the row-major `n` x `width` right-hand side with the solution of L L' X = B.
fn cholesky_solve(factor: &[f64], n: usize, rhs: &mut [f64], width: usize) {
    for column in 0..width {
        for i in 0..n {
            let sum: f64 = (0..i)
                .map(|k| factor[i * n + k] * rhs[k * width + column])
                .sum();
            rhs[i * width + column] = (rhs[i * width + column] - sum) / factor[i * n + i];
        }
        for i in (0..n).rev() {
            let sum: f64 = (i + 1..n)
                .map(|k| factor[k * n + i] * rhs[k * width + column])
                .sum();
            rhs[i * width + column] = (rhs[i * width + column] - sum) / factor[i * n + i];
        }
    }
}

/// Solves the small row-major SPD system `matrix` X = B in place of the
/// `width`-column `rhs`, using partial-pivoting LU if Cholesky breaks down.
fn solve_spd(matrix: &[f64], n: usize, factor: &mut [f64], rhs: &mut [f64], width: usize) {
    if cholesky_into(matrix, n, factor) {
        cholesky_solve(factor, n, rhs, width);
        return;
    }
    let lu = DMatrix::from_fn(n, n, |i, j| matrix[i * n + j]).partial_piv_lu();
    let solution = lu.solve(&DMatrix::from_fn(n, width, |i, j| rhs[i * width + j]));
    for i in 0..n {
        for j in 0..width {
            rhs[i * width + j] = solution[(i, j)];
        }
    }
}

fn quadratic_form(matrix: &[f64], vector: &[f64]) -> f64 {
    let n = vector.len();
    vector
        .iter()
        .enumerate()
        .map(|(i, &left)| left * (0..n).map(|j| matrix[i * n + j] * vector[j]).sum::<f64>())
        .sum()
}

fn validate_prior_weights(weights: &[f64], n: usize) -> PyResult<()> {
    if weights.len() != n {
        return Err(PyValueError::new_err(format!(
            "weights has length {}, expected {}",
            weights.len(),
            n
        )));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(PyValueError::new_err(
            "weights must contain only finite values",
        ));
    }
    if weights.iter().any(|weight| *weight <= 0.0) {
        return Err(PyValueError::new_err("weights must be strictly positive"));
    }
    Ok(())
}

/// Observations reordered so that each group's rows are contiguous.
struct GroupedData {
    x: Vec<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    sqrt_weights: Vec<f64>,
    /// Group `g`, in label order, owns rows `starts[g]..starts[g + 1]`.
    starts: Vec<usize>,
}

impl GroupedData {
    fn new(y: &[f64], x: &[f64], weights: &[f64], groups: &[i64]) -> Self {
        // The stable sort keeps observation order within each group.
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&row| groups[row]);
        let mut starts: Vec<usize> = (0..order.len())
            .filter(|&i| i == 0 || groups[order[i]] != groups[order[i - 1]])
            .collect();
        starts.push(order.len());
        let gather =
            |values: &[f64]| -> Vec<f64> { order.iter().map(|&row| values[row]).collect() };
        let weights = gather(weights);
        Self {
            x: gather(x),
            y: gather(y),
            sqrt_weights: weights.iter().map(|weight| weight.sqrt()).collect(),
            weights,
            starts,
        }
    }

    fn n_obs(&self) -> usize {
        self.y.len()
    }

    fn n_groups(&self) -> usize {
        self.starts.len() - 1
    }

    fn rows(&self, group: usize) -> Range<usize> {
        self.starts[group]..self.starts[group + 1]
    }

    fn largest_group(&self) -> usize {
        self.starts
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
            .max()
            .unwrap_or(0)
    }
}

/// Writes phi plus one group's random effects into `params`.
fn group_params(phi: &[f64], effects: &[f64], random_params: &[usize], params: &mut [f64]) {
    params.copy_from_slice(phi);
    for (&parameter, &effect) in random_params.iter().zip(effects) {
        params[parameter] += effect;
    }
}

/// Writes `base + step * delta` into `out`.
fn step_into(out: &mut [f64], base: &[f64], step: f64, delta: &[f64]) {
    for ((out, &value), &delta) in out.iter_mut().zip(base).zip(delta) {
        *out = value + step * delta;
    }
}

struct PnlsResult {
    phi: Vec<f64>,
    /// Row-major `n_groups` x `n_random` random effects.
    b: Vec<f64>,
    sigma_sq: f64,
    converged: bool,
}

fn random_effect_penalty(b: &[f64], precision: &[f64], n_random: usize) -> f64 {
    if n_random == 0 {
        return 0.0;
    }
    b.chunks_exact(n_random)
        .map(|effects| quadratic_form(precision, effects))
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn pnls_objective(
    data: &GroupedData,
    model: NlmeModel,
    phi: &[f64],
    b: &[f64],
    random_params: &[usize],
    precision: &[f64],
    params: &mut [f64],
    predicted: &mut [f64],
) -> f64 {
    let n_random = random_params.len();
    let mut rss = 0.0;
    for g in 0..data.n_groups() {
        let rows = data.rows(g);
        let effects = &b[g * n_random..(g + 1) * n_random];
        group_params(phi, effects, random_params, params);
        let predicted = &mut predicted[..rows.len()];
        model.predict_into(params, &data.x[rows.clone()], predicted);
        for (row, predicted) in rows.zip(predicted.iter()) {
            let residual = data.y[row] - predicted;
            rss += data.weights[row] * residual * residual;
        }
    }
    rss + random_effect_penalty(b, precision, n_random)
}

/// Joint penalized nonlinear least squares from `phi` and the row-major random
/// effects `b`. Each Gauss-Newton step eliminates every group's random effects
/// from the normal equations, and a halving line search accepts the step.
#[allow(clippy::too_many_arguments)]
fn pnls_step_impl(
    data: &GroupedData,
    model: NlmeModel,
    phi: &[f64],
    b: &[f64],
    psi: &DMatrix<f64>,
    random_params: &[usize],
    maxiter: usize,
    tol: f64,
) -> PnlsResult {
    // A compile-time parameter count lets the small per-group loops unroll.
    match phi.len() {
        2 => pnls::<2>(data, model, phi, b, psi, random_params, maxiter, tol),
        3 => pnls::<3>(data, model, phi, b, psi, random_params, maxiter, tol),
        4 => pnls::<4>(data, model, phi, b, psi, random_params, maxiter, tol),
        n => unreachable!("built-in models have two to four parameters, not {n}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn pnls<const P: usize>(
    data: &GroupedData,
    model: NlmeModel,
    phi: &[f64],
    b: &[f64],
    psi: &DMatrix<f64>,
    random_params: &[usize],
    maxiter: usize,
    tol: f64,
) -> PnlsResult {
    let n_random = random_params.len();
    let width = P + 1;
    let mut covariance = psi.clone();
    for i in 0..n_random {
        covariance[(i, i)] += 1e-8;
    }
    let precision = row_major(&invert_regularized(&covariance));
    // Scratch sized once per evaluation and reused by every group and iteration.
    let largest = data.largest_group();
    let mut params = [0.0; P];
    let mut predicted = vec![0.0; largest];
    let mut gradient = vec![[0.0; P]; largest];
    let mut random_normal = vec![0.0; n_random * n_random];
    let mut factor = vec![0.0; P.max(n_random).pow(2)];
    // Row (g, r) solves group g's random-effect system for [G'Z | Z'r - Psi^-1 b].
    let mut solutions = vec![0.0; b.len() * width];
    let mut delta_b = vec![0.0; b.len()];
    let mut phi_new: [f64; P] = phi.try_into().expect("one fixed effect per parameter");
    let mut b_new = b.to_vec();
    let mut trial_phi = phi_new;
    let mut trial_b = b.to_vec();
    let mut converged = false;
    let mut pwrss = f64::INFINITY;

    for _iteration in 0..maxiter {
        let mut normal = [[0.0; P]; P];
        let mut rhs = [0.0; P];
        let mut rss = 0.0;
        for g in 0..data.n_groups() {
            let rows = data.rows(g);
            let effects = &b_new[g * n_random..(g + 1) * n_random];
            group_params(&phi_new, effects, random_params, &mut params);
            let x = &data.x[rows.clone()];
            let predicted = &mut predicted[..x.len()];
            let gradient = &mut gradient[..x.len()];
            model.linearize_into(&params, x, predicted, gradient.as_flattened_mut());
            // Weighted Gauss-Newton crossproducts G'G and G'r. The random-effect
            // design Z is a column subset of G, so Z'Z, G'Z and Z'r are read off them.
            let mut lower = [[0.0; P]; P];
            let mut fixed_rhs = [0.0; P];
            for (row, (&predicted, grad)) in rows.zip(predicted.iter().zip(gradient.iter())) {
                let value = data.y[row] - predicted;
                rss += data.weights[row] * value * value;
                let sqrt_weight = data.sqrt_weights[row];
                let residual = value * sqrt_weight;
                let grad = grad.map(|entry| entry * sqrt_weight);
                for i in 0..P {
                    fixed_rhs[i] += grad[i] * residual;
                    for j in 0..=i {
                        lower[i][j] += grad[i] * grad[j];
                    }
                }
            }
            let fixed_normal: [[f64; P]; P] =
                std::array::from_fn(|i| std::array::from_fn(|j| lower[i.max(j)][i.min(j)]));
            let solution = &mut solutions[g * n_random * width..(g + 1) * n_random * width];
            for (r, &column) in random_params.iter().enumerate() {
                for (s, &other) in random_params.iter().enumerate() {
                    random_normal[r * n_random + s] =
                        fixed_normal[column][other] + precision[r * n_random + s];
                }
                for j in 0..P {
                    solution[r * width + j] = fixed_normal[j][column];
                }
                solution[r * width + P] = fixed_rhs[column]
                    - (0..n_random)
                        .map(|s| precision[r * n_random + s] * effects[s])
                        .sum::<f64>();
            }
            solve_spd(&random_normal, n_random, &mut factor, solution, width);
            for i in 0..P {
                let crossproduct = |r: usize| fixed_normal[i][random_params[r]];
                rhs[i] += fixed_rhs[i]
                    - (0..n_random)
                        .map(|r| crossproduct(r) * solution[r * width + P])
                        .sum::<f64>();
                for j in 0..P {
                    normal[i][j] += fixed_normal[i][j]
                        - (0..n_random)
                            .map(|r| crossproduct(r) * solution[r * width + j])
                            .sum::<f64>();
                }
            }
        }
        pwrss = rss + random_effect_penalty(&b_new, &precision, n_random);
        let normal: [[f64; P]; P] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == j {
                    normal[i][i] + 1e-6
                } else {
                    0.5 * (normal[i][j] + normal[j][i])
                }
            })
        });
        let mut delta_phi = rhs;
        solve_spd(normal.as_flattened(), P, &mut factor, &mut delta_phi, 1);
        for (delta, solution) in delta_b.iter_mut().zip(solutions.chunks_exact(width)) {
            *delta = solution[P] - (0..P).map(|j| solution[j] * delta_phi[j]).sum::<f64>();
        }
        let mut max_delta: f64 = 0.0;
        for &value in delta_phi.iter().chain(&delta_b) {
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
            step_into(&mut trial_phi, &phi_new, step, &delta_phi);
            step_into(&mut trial_b, &b_new, step, &delta_b);
            let candidate = pnls_objective(
                data,
                model,
                &trial_phi,
                &trial_b,
                random_params,
                &precision,
                &mut params,
                &mut predicted,
            );
            if candidate.is_finite() && candidate <= pwrss + slack {
                phi_new = trial_phi;
                std::mem::swap(&mut b_new, &mut trial_b);
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
    let variance = pwrss / data.n_obs() as f64;
    PnlsResult {
        phi: phi_new.to_vec(),
        b: b_new,
        sigma_sq: if variance.is_finite() {
            variance.max(f64::MIN_POSITIVE)
        } else {
            variance
        },
        converged,
    }
}

/// Sum over groups of log|I + L' Z'WZ L|, the stable form of
/// log|Psi| + log|Z'WZ + Psi^-1| for Psi = L L'. None marks an indefinite system.
fn laplace_correction(
    data: &GroupedData,
    model: NlmeModel,
    phi: &[f64],
    b: &[f64],
    psi_factor: &DMatrix<f64>,
    random_params: &[usize],
) -> Option<f64> {
    let n_phi = phi.len();
    let n_random = random_params.len();
    let psi_factor = row_major(psi_factor);
    let mut params = vec![0.0; n_phi];
    let mut predicted = vec![0.0; data.largest_group()];
    let mut gradient = vec![0.0; data.largest_group() * n_phi];
    let mut weighted_z = vec![0.0; n_random];
    let mut ztz = vec![0.0; n_random * n_random];
    let mut scaled = vec![0.0; n_random * n_random];
    let mut system = vec![0.0; n_random * n_random];
    let mut factor = vec![0.0; n_random * n_random];
    let mut correction = 0.0;

    for g in 0..data.n_groups() {
        let rows = data.rows(g);
        let effects = &b[g * n_random..(g + 1) * n_random];
        group_params(phi, effects, random_params, &mut params);
        let predicted = &mut predicted[..rows.len()];
        let gradient = &mut gradient[..rows.len() * n_phi];
        model.linearize_into(&params, &data.x[rows.clone()], predicted, gradient);
        ztz.fill(0.0);
        for (row, grad) in rows.zip(gradient.chunks_exact(n_phi)) {
            let sqrt_weight = data.sqrt_weights[row];
            for (value, &column) in weighted_z.iter_mut().zip(random_params) {
                *value = grad[column] * sqrt_weight;
            }
            for r in 0..n_random {
                for s in 0..=r {
                    ztz[r * n_random + s] += weighted_z[r] * weighted_z[s];
                }
            }
        }
        for r in 0..n_random {
            for s in 0..r {
                ztz[s * n_random + r] = ztz[r * n_random + s];
            }
        }
        for i in 0..n_random {
            for j in 0..n_random {
                scaled[i * n_random + j] = (0..n_random)
                    .map(|k| psi_factor[k * n_random + i] * ztz[k * n_random + j])
                    .sum();
            }
        }
        for i in 0..n_random {
            for j in 0..n_random {
                let identity = if i == j { 1.0 } else { 0.0 };
                system[i * n_random + j] = identity
                    + (0..n_random)
                        .map(|k| scaled[i * n_random + k] * psi_factor[k * n_random + j])
                        .sum::<f64>();
            }
        }
        let logdet = if cholesky_into(&system, n_random, &mut factor) {
            2.0 * (0..n_random)
                .map(|i| factor[i * n_random + i].ln())
                .sum::<f64>()
        } else {
            let eigenvalues = DMatrix::from_fn(n_random, n_random, |i, j| system[i * n_random + j])
                .self_adjoint_eigenvalues(Side::Lower)
                .unwrap_or_else(|_| vec![f64::NAN; n_random]);
            if eigenvalues
                .iter()
                .any(|value| *value <= 0.0 || !value.is_finite())
            {
                return None;
            }
            eigenvalues.iter().map(|value| value.ln()).sum()
        };
        correction += logdet;
    }
    Some(correction)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn nlmm_deviance_with_status_impl(
    data: &GroupedData,
    model: NlmeModel,
    phi: &[f64],
    b: &[f64],
    psi_factor: &DMatrix<f64>,
    random_params: &[usize],
    maxiter: usize,
    tol: f64,
) -> (f64, Vec<f64>, Vec<f64>, f64, bool) {
    let psi = psi_factor * psi_factor.transpose();
    let result = pnls_step_impl(data, model, phi, b, &psi, random_params, maxiter, tol);
    let sigma_sq = result.sigma_sq;
    let Some(correction) = laplace_correction(
        data,
        model,
        &result.phi,
        &result.b,
        psi_factor,
        random_params,
    ) else {
        return (1e100, result.phi, result.b, sigma_sq.sqrt(), false);
    };
    let deviance =
        data.n_obs() as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma_sq).ln()) + correction;

    (
        deviance,
        result.phi,
        result.b,
        sigma_sq.sqrt(),
        result.converged,
    )
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

/// Owned copies of a native NLMM call's arrays. Copying them releases the NumPy
/// borrows, so detached work never observes caller changes to the inputs.
struct NlmmInputs {
    y: Vec<f64>,
    x: Vec<f64>,
    groups: Vec<i64>,
    weights: Vec<f64>,
    phi: Vec<f64>,
    b: Vec<f64>,
    b_shape: (usize, usize),
    theta: Vec<f64>,
}

impl NlmmInputs {
    #[allow(clippy::too_many_arguments)]
    fn new(
        y: numpy::PyArrayLike1<'_, f64>,
        x: numpy::PyArrayLike1<'_, f64>,
        groups: numpy::PyArrayLike1<'_, i64>,
        phi: numpy::PyArrayLike1<'_, f64>,
        b: numpy::PyArrayLike2<'_, f64>,
        theta: numpy::PyArrayLike1<'_, f64>,
        weights: Option<numpy::PyArrayLike1<'_, f64>>,
    ) -> PyResult<Self> {
        let y = y.as_array().to_vec();
        let weights = match weights {
            Some(weights) => weights.as_array().to_vec(),
            None => vec![1.0; y.len()],
        };
        validate_prior_weights(&weights, y.len())?;
        let b_array = b.as_array();
        let inputs = Self {
            x: x.as_array().to_vec(),
            groups: groups.as_array().to_vec(),
            weights,
            phi: phi.as_array().to_vec(),
            b: b_array.iter().copied().collect(),
            b_shape: b_array.dim(),
            theta: theta.as_array().to_vec(),
            y,
        };
        for (name, length) in [("x", inputs.x.len()), ("groups", inputs.groups.len())] {
            if length != inputs.y.len() {
                return Err(PyValueError::new_err(format!(
                    "{name} has length {length}, expected {}",
                    inputs.y.len()
                )));
            }
        }
        Ok(inputs)
    }

    /// Groups the observations and checks every parameter shape against the
    /// model, returning the data and the random-effect covariance factor.
    fn prepare(
        &self,
        model: NlmeModel,
        random_params: &[usize],
    ) -> PyResult<(GroupedData, DMatrix<f64>)> {
        let data = GroupedData::new(&self.y, &self.x, &self.weights, &self.groups);
        let n_params = model.n_params();
        let n_random = random_params.len();
        if self.phi.len() != n_params {
            return Err(PyValueError::new_err(format!(
                "phi has length {}, expected {n_params}",
                self.phi.len()
            )));
        }
        if random_params.iter().any(|&parameter| parameter >= n_params) {
            return Err(PyValueError::new_err(format!(
                "random_params must index the model's {n_params} parameters"
            )));
        }
        let expected = (data.n_groups(), n_random);
        if self.b_shape != expected {
            return Err(PyValueError::new_err(format!(
                "b has shape {:?}, expected {expected:?}",
                self.b_shape
            )));
        }
        let psi_factor = build_psi_factor(&self.theta, n_random);
        if psi_factor.nrows() != n_random {
            return Err(PyValueError::new_err(format!(
                "theta of length {} does not define a {n_random} x {n_random} covariance factor",
                self.theta.len()
            )));
        }
        Ok((data, psi_factor))
    }
}

type Estimates<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// Returns phi and the row-major random effects, of shape `b_shape`, as NumPy arrays.
fn estimates_to_numpy(
    py: Python<'_>,
    phi: Vec<f64>,
    b: Vec<f64>,
    b_shape: (usize, usize),
) -> PyResult<Estimates<'_>> {
    let b = PyArray1::from_vec(py, b).reshape([b_shape.0, b_shape.1])?;
    Ok((PyArray1::from_vec(py, phi), b))
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
    py: Python<'py>,
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
) -> PyResult<(
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    bool,
)> {
    validate_pnls_controls(maxiter, tol).map_err(PyValueError::new_err)?;
    let model = NlmeModel::from_name(model_name)?;
    // PNLS estimates the residual scale itself; `sigma` is kept for compatibility.
    let _ = sigma;
    let inputs = NlmmInputs::new(y, x, groups, phi, b, theta, weights)?;
    let (deviance, phi, b, sigma, converged) = py.detach(|| {
        let (data, psi_factor) = inputs.prepare(model, &random_params)?;
        Ok::<_, PyErr>(nlmm_deviance_with_status_impl(
            &data,
            model,
            &inputs.phi,
            &inputs.b,
            &psi_factor,
            &random_params,
            maxiter,
            tol,
        ))
    })?;
    let (phi, b) = estimates_to_numpy(py, phi, b, inputs.b_shape)?;
    Ok((deviance, phi, b, sigma, converged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Col as DVector;

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn nlmm_deviance_impl(
        theta: &[f64],
        y: &[f64],
        x: &[f64],
        groups: &[i64],
        weights: &[f64],
        model: NlmeModel,
        phi: &[f64],
        b: &[f64],
        random_params: &[usize],
    ) -> (f64, Vec<f64>, Vec<f64>, f64) {
        let (deviance, phi, b, sigma, _) = nlmm_deviance_with_status_impl(
            &GroupedData::new(y, x, weights, groups),
            model,
            phi,
            b,
            &build_psi_factor(theta, random_params.len()),
            random_params,
            PNLS_MAX_ITER,
            PNLS_TOLERANCE,
        );
        (deviance, phi, b, sigma)
    }

    /// One Gauss-Newton step of the full joint system [fixed | block-diagonal
    /// random] built from analytic Michaelis-Menten derivatives.
    #[allow(clippy::too_many_arguments)]
    fn micmen_full_joint_step(
        x: &[f64],
        y: &[f64],
        groups: &[i64],
        weights: &[f64],
        phi: &[f64],
        b: &[f64],
        random_params: &[usize],
        psi: &DMatrix<f64>,
    ) -> DVector<f64> {
        let q = random_params.len();
        let n_groups = b.len() / q;
        let params = |group: usize| {
            let mut params = phi.to_vec();
            for (j, &parameter) in random_params.iter().enumerate() {
                params[parameter] += b[group * q + j];
            }
            params
        };
        let derivative = |params: &[f64], x: f64, parameter: usize| match parameter {
            0 => x / (params[1] + x),
            _ => -params[0] * x / (params[1] + x).powi(2),
        };
        let design = DMatrix::from_fn(x.len(), 2 + n_groups * q, |i, j| {
            let group = groups[i] as usize;
            let params = params(group);
            let value = if j < 2 {
                derivative(&params, x[i], j)
            } else if (j - 2) / q == group {
                derivative(&params, x[i], random_params[(j - 2) % q])
            } else {
                0.0
            };
            value * weights[i].sqrt()
        });
        let residual = DVector::from_fn(x.len(), |i| {
            let params = params(groups[i] as usize);
            (y[i] - params[0] * x[i] / (params[1] + x[i])) * weights[i].sqrt()
        });
        let mut normal = design.transpose() * &design;
        let mut rhs = design.transpose() * residual;
        for j in 0..2 {
            normal[(j, j)] += 1e-6;
        }
        let mut covariance = psi.clone();
        for r in 0..q {
            covariance[(r, r)] += 1e-8;
        }
        let precision = covariance.partial_piv_lu().inverse();
        for g in 0..n_groups {
            for r in 0..q {
                for s in 0..q {
                    normal[(2 + g * q + r, 2 + g * q + s)] += precision[(r, s)];
                    rhs[2 + g * q + r] -= precision[(r, s)] * b[g * q + s];
                }
            }
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
        let data = GroupedData::new(&y, &x, &weights, &groups);
        let phi = [2.0, 1.2];
        let correlated = DMatrix::from_fn(2, 2, |i, j| [[0.25, -0.05], [-0.05, 0.16]][i][j]);
        // Permuted and repeated parameters check the random-effect column mapping.
        for (random_params, psi) in [
            (vec![0], DMatrix::from_fn(1, 1, |_, _| 0.25)),
            (vec![1, 0], correlated.clone()),
            (vec![0, 0], correlated),
        ] {
            let q = random_params.len();
            let b: Vec<f64> = (0..3 * q).map(|index| (index as f64 - 1.0) * 0.1).collect();
            let delta =
                micmen_full_joint_step(&x, &y, &groups, &weights, &phi, &b, &random_params, &psi);
            let result = pnls_step_impl(
                &data,
                NlmeModel::SSmicmen,
                &phi,
                &b,
                &psi,
                &random_params,
                1,
                1e-12,
            );
            assert!(!result.converged);
            for j in 0..2 {
                assert!((result.phi[j] - phi[j] - delta[j]).abs() < 1e-12);
            }
            for (index, effect) in b.iter().enumerate() {
                assert!((result.b[index] - effect - delta[index + 2]).abs() < 1e-12);
            }
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
        let b = [0.0, 0.0];
        let psi = DMatrix::from_fn(1, 1, |_, _| 0.25);
        let delta = micmen_full_joint_step(&x, &y, &groups, &weights, &phi, &b, &[0], &psi);
        let initial = micmen_score(&x, &y, &groups, &phi, &b);
        let full = micmen_score(
            &x,
            &y,
            &groups,
            &[phi[0] + delta[0], phi[1] + delta[1]],
            &[delta[2], delta[3]],
        );
        assert!(full > initial);
        let result = pnls_step_impl(
            &GroupedData::new(&y, &x, &weights, &groups),
            NlmeModel::SSmicmen,
            &phi,
            &b,
            &psi,
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
        let b: Vec<f64> = (0..3).map(|row| (row as f64 - 1.0) * 0.1).collect();
        let data = GroupedData::new(&y, &x, &weights, &groups);
        let evaluate = |maxiter, tol| {
            nlmm_deviance_with_status_impl(
                &data,
                NlmeModel::SSmicmen,
                &[2.0, 1.2],
                &b,
                &build_psi_factor(&[0.4], 1),
                &[0],
                maxiter,
                tol,
            )
        };
        let limited = evaluate(1, 1e-12);
        let loose = evaluate(1, 1e6);
        let complete = evaluate(1000, 1e-10);
        assert!(!limited.4);
        assert!(loose.4 && complete.4);
        assert!((limited.0 - loose.0).abs() <= 1e-12 * loose.0.abs().max(1.0));
        assert_eq!(limited.1.len(), loose.1.len());
        for (actual, expected) in limited.1.iter().zip(&loose.1) {
            assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
        }
        assert!((limited.3 - loose.3).abs() <= 1e-12 * loose.3.abs().max(1.0));
        for row in 0..3 {
            assert!((limited.2[row] - loose.2[row]).abs() < 1e-12);
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

    /// Groups `labels` with each observation's row index stored as its `x`.
    fn group_rows(labels: &[i64]) -> GroupedData {
        let rows: Vec<f64> = (0..labels.len()).map(|row| row as f64).collect();
        let weights: Vec<f64> = rows.iter().map(|row| (row + 1.0).powi(2)).collect();
        GroupedData::new(&rows, &rows, &weights, labels)
    }

    #[test]
    fn grouped_data_preserves_label_and_observation_order() {
        let data = group_rows(&[i64::MAX, -3, i64::MIN, -3, i64::MAX, i64::MIN]);
        assert_eq!(data.n_groups(), 3);
        assert_eq!(data.largest_group(), 2);
        assert_eq!(
            (0..3).map(|group| data.rows(group)).collect::<Vec<_>>(),
            vec![0..2, 2..4, 4..6]
        );
        assert_eq!(data.x, [2.0, 5.0, 1.0, 3.0, 0.0, 4.0]);
        assert_eq!(data.y, data.x);
        assert_eq!(data.weights, [9.0, 36.0, 4.0, 16.0, 1.0, 25.0]);
        assert_eq!(data.sqrt_weights, [3.0, 6.0, 2.0, 4.0, 1.0, 5.0]);
        let empty = group_rows(&[]);
        assert_eq!((empty.n_groups(), empty.largest_group()), (0, 0));
    }

    #[test]
    fn grouped_data_handles_many_interleaved_labels() {
        let labels: Vec<i64> = (0..10_000).map(|row| ((row * 7) % 1000) - 400).collect();
        let data = group_rows(&labels);
        assert_eq!(data.n_groups(), 1000);
        for group in 0..1000 {
            let rows = &data.x[data.rows(group)];
            assert_eq!(rows.len(), 10);
            assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
            for &row in rows {
                assert_eq!(labels[row as usize], group as i64 - 400);
            }
        }
    }

    #[test]
    fn linearized_models_match_predictions_and_central_differences() {
        let x = [0.1, 0.7, 1.5, 3.0];
        for (model, params) in [
            (NlmeModel::SSasymp, vec![10.0, 0.5, -0.5]),
            (NlmeModel::SSlogis, vec![10.0, 1.0, 0.8]),
            (NlmeModel::SSmicmen, vec![3.0, 1.2]),
            (NlmeModel::SSfpl, vec![2.0, 10.0, 1.0, 0.8]),
            (NlmeModel::SSgompertz, vec![5.0, 2.0, 0.7]),
            (NlmeModel::SSbiexp, vec![3.0, 0.5, 1.0, -1.0]),
        ] {
            let p = params.len();
            assert_eq!(p, model.n_params());
            let mut predicted = vec![0.0; x.len()];
            let mut gradient = vec![0.0; x.len() * p];
            model.linearize_into(&params, &x, &mut predicted, &mut gradient);
            // The PNLS objective uses predict_into, so the fused pass must agree exactly.
            assert_eq!(predicted, model.predict(&params, &x));
            for j in 0..p {
                let shifted = |step: f64| {
                    let mut params = params.clone();
                    params[j] += step;
                    model.predict(&params, &x)
                };
                let (upper, lower) = (shifted(1e-6), shifted(-1e-6));
                for i in 0..x.len() {
                    let numeric = (upper[i] - lower[i]) / 2e-6;
                    let analytic = gradient[i * p + j];
                    assert!(
                        (numeric - analytic).abs() <= 1e-6 * analytic.abs().max(1.0),
                        "{model:?} parameter {j} at x={}: {numeric} vs {analytic}",
                        x[i]
                    );
                }
            }
        }
    }

    #[test]
    fn small_spd_solve_matches_faer_and_falls_back_to_lu() {
        let spd = [4.0, 1.2, -0.5, 1.2, 3.0, 0.4, -0.5, 0.4, 2.5];
        let indefinite = [1.0, 2.0, 0.0, 2.0, 1.0, 0.5, 0.0, 0.5, -3.0];
        let rhs = [1.0, -2.0, 0.5, 3.0, 0.25, -1.0];
        for matrix in [spd, indefinite] {
            let dense = DMatrix::from_fn(3, 3, |i, j| matrix[i * 3 + j]);
            let dense_rhs = DMatrix::from_fn(3, 2, |i, j| rhs[i * 2 + j]);
            let expected = match Llt::new(dense.as_ref(), Side::Lower) {
                Ok(cholesky) => cholesky.solve(&dense_rhs),
                Err(_) => dense.partial_piv_lu().solve(&dense_rhs),
            };
            let mut factor = [0.0; 9];
            let mut actual = rhs;
            solve_spd(&matrix, 3, &mut factor, &mut actual, 2);
            for i in 0..3 {
                for j in 0..2 {
                    assert!((actual[i * 2 + j] - expected[(i, j)]).abs() < 1e-13);
                }
            }
        }
        let mut factor = [0.0; 9];
        assert!(cholesky_into(&spd, 3, &mut factor));
        assert!(!cholesky_into(&indefinite, 3, &mut factor));
        assert!(!cholesky_into(&[f64::NAN], 1, &mut factor));
        assert!(!cholesky_into(&[f64::INFINITY], 1, &mut factor));
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
        let b = vec![0.0; 4];
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
        let b = vec![0.0; 4];

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
        );

        assert!(collapsed.0.is_finite());
        assert!(nonzero.0 < collapsed.0);
    }

    #[test]
    fn prior_weights_downweight_an_outlier() {
        let (x, y, groups) = asymptotic_data();
        let weights = vec![1.0; y.len()];
        let phi = vec![10.0, 0.5, -0.5];
        let b = vec![0.0; 4];
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

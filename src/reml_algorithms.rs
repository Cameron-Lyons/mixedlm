use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, MatRef, Side};
use numpy::PyArray1;
use pyo3::prelude::*;

pub struct RemlResult {
    pub variance_components: Vec<f64>,
    pub sigma2: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn matrix_is_finite(matrix: &Mat<f64>) -> bool {
    (0..matrix.nrows())
        .all(|row| (0..matrix.ncols()).all(|column| matrix[(row, column)].is_finite()))
}

fn validate_reml_inputs(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    variances: &[f64],
    sigma2: f64,
) -> Result<(), String> {
    let n = y.nrows();
    if n == 0 {
        return Err("y must contain at least one observation".to_string());
    }
    if y.ncols() != 1 {
        return Err("y must be a column vector".to_string());
    }
    if x.nrows() != n {
        return Err(format!(
            "x must have {n} rows to match y, got {}",
            x.nrows()
        ));
    }
    if x.ncols() >= n {
        return Err(format!(
            "x must have fewer columns than observations, got {} columns for {n} observations",
            x.ncols()
        ));
    }
    if z_blocks.len() != variances.len() {
        return Err(format!(
            "init_variances must contain one value per Z block, got {} values for {} blocks",
            variances.len(),
            z_blocks.len()
        ));
    }
    for (index, z) in z_blocks.iter().enumerate() {
        if z.nrows() != n {
            return Err(format!(
                "Z block {index} must have {n} rows to match y, got {}",
                z.nrows()
            ));
        }
        if z.ncols() == 0 {
            return Err(format!("Z block {index} must contain at least one column"));
        }
        if !matrix_is_finite(z) {
            return Err(format!("Z block {index} must contain only finite values"));
        }
    }
    if !matrix_is_finite(y) {
        return Err("y must contain only finite values".to_string());
    }
    if !matrix_is_finite(x) {
        return Err("x must contain only finite values".to_string());
    }
    if variances
        .iter()
        .any(|variance| !variance.is_finite() || *variance < 0.0)
    {
        return Err("init_variances must contain finite, nonnegative values".to_string());
    }
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err("init_sigma2 must be finite and greater than zero".to_string());
    }
    Ok(())
}

fn validate_iteration_parameters(tol: f64) -> Result<(), String> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and greater than zero".to_string());
    }
    Ok(())
}

fn compute_trace_product_ref(a: &Mat<f64>, b: MatRef<'_, f64>) -> f64 {
    let n = a.nrows();
    let m = a.ncols();
    let mut trace = 0.0;
    for i in 0..n {
        for j in 0..m {
            trace += a[(i, j)] * b[(j, i)];
        }
    }
    trace
}

fn squared_norm(x: &Mat<f64>) -> f64 {
    let mut result = 0.0;
    for row in 0..x.nrows() {
        for column in 0..x.ncols() {
            result += x[(row, column)] * x[(row, column)];
        }
    }
    result
}

/// The REML projection is applied through factorizations, without forming P.
/// Low-rank covariance derivatives can then be contracted with their Z blocks.
struct RemlProjection {
    covariance: Llt<f64>,
    weighted_x: Mat<f64>,
    fixed: Option<Llt<f64>>,
}

impl RemlProjection {
    fn new(v: &Mat<f64>, x: &Mat<f64>) -> Result<Self, String> {
        let covariance =
            Llt::new(v.as_ref(), Side::Lower).map_err(|_| "V not positive definite".to_string())?;
        let weighted_x = covariance.solve(x);
        let fixed = if x.ncols() == 0 {
            None
        } else {
            let information = x.transpose() * &weighted_x;
            Some(
                Llt::new(information.as_ref(), Side::Lower)
                    .map_err(|_| "X'V^-1 X not positive definite".to_string())?,
            )
        };
        Ok(Self {
            covariance,
            weighted_x,
            fixed,
        })
    }

    fn apply(&self, rhs: &Mat<f64>) -> Mat<f64> {
        let weighted_rhs = self.covariance.solve(rhs);
        if let Some(fixed) = &self.fixed {
            let coefficients = fixed.solve(&(self.weighted_x.transpose() * rhs));
            &weighted_rhs - &self.weighted_x * &coefficients
        } else {
            weighted_rhs
        }
    }

    fn trace(&self) -> f64 {
        let n = self.weighted_x.nrows();
        let inverse = self.covariance.solve(&Mat::<f64>::identity(n, n));
        let mut trace = (0..n).map(|i| inverse[(i, i)]).sum::<f64>();
        if let Some(fixed) = &self.fixed {
            let coefficients = fixed.solve(&self.weighted_x.transpose());
            trace -= compute_trace_product_ref(&self.weighted_x, coefficients.as_ref());
        }
        trace
    }

    fn residual_trace(&self, random_trace: f64, sigma2: f64) -> f64 {
        // tr(P V) = n-p. Reuse low-rank random-effect contractions instead
        // of solving against n identity columns. A saturated random design
        // can cause cancellation; use the direct trace in that rare case.
        let df = (self.weighted_x.nrows() - self.weighted_x.ncols()) as f64;
        let remainder = df - random_trace;
        if remainder > 1e-8 * df {
            remainder / sigma2
        } else {
            self.trace()
        }
    }

    fn objective(&self, y: &Mat<f64>) -> f64 {
        let mut logdet = cholesky_logdet(&self.covariance);
        if let Some(fixed) = &self.fixed {
            logdet += cholesky_logdet(fixed);
        }
        let projected_y = self.apply(y);
        logdet + (y.transpose() * &projected_y)[(0, 0)]
    }
}

fn cholesky_logdet(chol: &Llt<f64>) -> f64 {
    let l = chol.L();
    2.0 * (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>()
}

fn variance_covariance(
    z_blocks: &[Mat<f64>],
    variances: &[f64],
    sigma2: f64,
    n: usize,
) -> Mat<f64> {
    let mut covariance = Mat::from_fn(n, n, |i, j| if i == j { sigma2 } else { 0.0 });
    for (z, &variance) in z_blocks.iter().zip(variances) {
        covariance += variance * (z * z.transpose());
    }
    covariance
}

fn correlated_covariance(z_blocks: &[Mat<f64>], s: &Mat<f64>, sigma2: f64) -> Mat<f64> {
    let n = z_blocks[0].nrows();
    let mut covariance = Mat::from_fn(n, n, |i, j| if i == j { sigma2 } else { 0.0 });
    for (i, zi) in z_blocks.iter().enumerate() {
        for (j, zj) in z_blocks.iter().enumerate() {
            covariance += s[(i, j)] * (zi * zj.transpose());
        }
    }
    covariance
}

fn mm_variance_update(variance: f64, quadratic: f64, trace: f64) -> f64 {
    if trace > 0.0 {
        // MM minimizes a*v + b/v, so the update contains a square root.
        (variance * (quadratic.max(0.0) / trace).sqrt()).max(1e-10)
    } else {
        variance
    }
}

fn variance_score_norm(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    variances: &[f64],
    sigma2: f64,
) -> Result<f64, String> {
    let n = y.nrows() as f64;
    let v = variance_covariance(z_blocks, variances, sigma2, y.nrows());
    let projection = RemlProjection::new(&v, x)?;
    let py = projection.apply(y);
    let mut maximum = 0.0_f64;
    let mut random_trace = 0.0;
    for (z, &variance) in z_blocks.iter().zip(variances) {
        let pz = projection.apply(z);
        let trace = compute_trace_product_ref(&pz, z.transpose());
        random_trace += variance * trace;
        let score = 0.5 * (squared_norm(&(z.transpose() * &py)) - trace);
        let scale = variance.max(sigma2 * n / squared_norm(z).max(f64::MIN_POSITIVE));
        // At a zero variance, only a positive score violates the KKT condition.
        let violation = if variance <= 1e-10 {
            score.max(0.0)
        } else {
            score.abs()
        };
        maximum = maximum.max(violation * scale / n);
    }
    let residual_score =
        0.5 * (squared_norm(&py) - projection.residual_trace(random_trace, sigma2));
    let violation = if sigma2 <= 1e-10 {
        residual_score.max(0.0)
    } else {
        residual_score.abs()
    };
    Ok(maximum.max(violation * sigma2 / n))
}

pub fn mm_reml_step(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_variances: &[f64],
    sigma2: f64,
) -> Result<(Vec<f64>, f64), String> {
    validate_reml_inputs(y, x, z_blocks, current_variances, sigma2)?;
    let v = variance_covariance(z_blocks, current_variances, sigma2, y.nrows());
    let projection = RemlProjection::new(&v, x)?;
    let py = projection.apply(y);
    let mut random_trace = 0.0;
    let new_variances = z_blocks
        .iter()
        .zip(current_variances)
        .map(|(z, &variance)| {
            let pz = projection.apply(z);
            let trace = compute_trace_product_ref(&pz, z.transpose());
            random_trace += variance * trace;
            let quadratic = squared_norm(&(z.transpose() * &py));
            mm_variance_update(variance, quadratic, trace)
        })
        .collect();
    let new_sigma2 = mm_variance_update(
        sigma2,
        squared_norm(&py),
        projection.residual_trace(random_trace, sigma2),
    );
    Ok((new_variances, new_sigma2))
}

#[allow(clippy::too_many_arguments)]
pub fn mm_reml_iterate(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    init_variances: &[f64],
    init_sigma2: f64,
    max_iter: usize,
    tol: f64,
) -> Result<RemlResult, String> {
    validate_reml_inputs(y, x, z_blocks, init_variances, init_sigma2)?;
    validate_iteration_parameters(tol)?;

    let mut variances = init_variances.to_vec();
    let mut sigma2 = init_sigma2;

    for iter in 0..max_iter {
        let (new_variances, new_sigma2) = mm_reml_step(y, x, z_blocks, &variances, sigma2)?;

        let mut max_change = (new_sigma2 - sigma2).abs() / sigma2.max(1e-10);
        for (old, new) in variances.iter().zip(new_variances.iter()) {
            let change = (new - old).abs() / old.max(1e-10);
            if change > max_change {
                max_change = change;
            }
        }

        variances = new_variances;
        sigma2 = new_sigma2;

        if max_change < tol && variance_score_norm(y, x, z_blocks, &variances, sigma2)? < tol {
            return Ok(RemlResult {
                variance_components: variances,
                sigma2,
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    Ok(RemlResult {
        variance_components: variances,
        sigma2,
        iterations: max_iter,
        converged: false,
    })
}

pub fn augmented_ai_reml_step(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_variances: &[f64],
    sigma2: f64,
) -> Result<(Vec<f64>, f64, Mat<f64>), String> {
    validate_reml_inputs(y, x, z_blocks, current_variances, sigma2)?;
    let n = y.nrows();
    let k = z_blocks.len();
    let v = variance_covariance(z_blocks, current_variances, sigma2, n);
    let projection = RemlProjection::new(&v, x)?;
    let py = projection.apply(y);
    let mut score = vec![0.0; k + 1];
    let mut actions = Mat::zeros(n, k + 1);
    let mut random_trace = 0.0;
    for (i, z) in z_blocks.iter().enumerate() {
        let pz = projection.apply(z);
        let coefficients = z.transpose() * &py;
        let trace = compute_trace_product_ref(&pz, z.transpose());
        random_trace += current_variances[i] * trace;
        score[i] = 0.5 * (squared_norm(&coefficients) - trace);
        actions.col_mut(i).copy_from((z * &coefficients).col(0));
    }
    actions.col_mut(k).copy_from(py.col(0));
    score[k] = 0.5 * (squared_norm(&py) - projection.residual_trace(random_trace, sigma2));
    // I_A(i,j) = 0.5 (V_i P y)' P (V_j P y). Batch all projections
    // and retain only an n by (k+1) workspace, rather than n by n derivatives.
    let projected_actions = projection.apply(&actions);
    let information = 0.5 * (actions.transpose() * &projected_actions);
    let parameters: Vec<f64> = current_variances.iter().copied().chain([sigma2]).collect();
    let free: Vec<usize> = (0..=k)
        .filter(|&i| parameters[i] > 1e-10 || score[i] > 0.0)
        .collect();
    if free.is_empty() {
        return Ok((current_variances.to_vec(), sigma2, information));
    }
    let free_information = Mat::from_fn(free.len(), free.len(), |i, j| {
        information[(free[i], free[j])]
    });
    let chol = Llt::new(free_information.as_ref(), Side::Lower)
        .map_err(|_| "AI matrix not positive definite".to_string())?;
    let delta = chol.solve(&Mat::from_fn(free.len(), 1, |i, _| score[free[i]]));
    let mut full_parameters = parameters.clone();
    for (i, &index) in free.iter().enumerate() {
        full_parameters[index] = (parameters[index] + delta[(i, 0)]).max(1e-10);
    }
    let predicted_improvement = |candidate: &[f64]| {
        2.0 * (0..=k)
            .map(|i| score[i] * (candidate[i] - parameters[i]))
            .sum::<f64>()
    };
    let mut improvement = predicted_improvement(&full_parameters);
    if improvement < 0.0 {
        // Projecting a coupled AI direction onto the variance bounds may
        // remove ascent. Diagonal scaling retains the score's ascent signs.
        for &index in &free {
            full_parameters[index] =
                (parameters[index] + score[index] / information[(index, index)]).max(1e-10);
        }
        improvement = predicted_improvement(&full_parameters);
    }
    let full_covariance =
        variance_covariance(z_blocks, &full_parameters[..k], full_parameters[k], n);
    let covariance_direction = &full_covariance - &v;
    let objective = projection.objective(y);
    let mut step = 1.0;
    for _ in 0..40 {
        // All trial points lie on the same feasible parameter segment. V is
        // linear in the variance components, so reuse one covariance direction
        // rather than assembling every Z Z' again during step halving.
        let candidate: Vec<f64> = (0..=k)
            .map(|i| parameters[i] + step * (full_parameters[i] - parameters[i]))
            .collect();
        let candidate_covariance = &v + step * &covariance_direction;
        if let Ok(candidate_projection) = RemlProjection::new(&candidate_covariance, x) {
            let candidate_objective = candidate_projection.objective(y);
            if candidate_objective.is_finite()
                && candidate_objective
                    <= objective - 1e-4 * step * improvement.max(0.0)
                        + 1e-12 * objective.abs().max(1.0)
                && improvement >= 0.0
            {
                return Ok((candidate[..k].to_vec(), candidate[k], information));
            }
        }
        step *= 0.5;
    }
    Err("AI REML could not find a likelihood-improving step".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn augmented_ai_reml_iterate(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    init_variances: &[f64],
    init_sigma2: f64,
    max_iter: usize,
    tol: f64,
) -> Result<RemlResult, String> {
    validate_reml_inputs(y, x, z_blocks, init_variances, init_sigma2)?;
    validate_iteration_parameters(tol)?;

    let mut variances = init_variances.to_vec();
    let mut sigma2 = init_sigma2;

    for iter in 0..max_iter {
        let (new_variances, new_sigma2, _ai) =
            augmented_ai_reml_step(y, x, z_blocks, &variances, sigma2)?;

        let mut max_change = (new_sigma2 - sigma2).abs() / sigma2.max(1e-10);
        for (old, new) in variances.iter().zip(new_variances.iter()) {
            let change = (new - old).abs() / old.max(1e-10);
            if change > max_change {
                max_change = change;
            }
        }

        variances = new_variances;
        sigma2 = new_sigma2;

        if max_change < tol && variance_score_norm(y, x, z_blocks, &variances, sigma2)? < tol {
            return Ok(RemlResult {
                variance_components: variances,
                sigma2,
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    Ok(RemlResult {
        variance_components: variances,
        sigma2,
        iterations: max_iter,
        converged: false,
    })
}

fn symmetric_exponential(matrix: &Mat<f64>) -> Result<Mat<f64>, String> {
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| "Riemannian eigendecomposition did not converge".to_string())?;
    let eigenvectors = eigen.U();
    let eigenvalues = eigen.S();
    let scaled = Mat::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        eigenvectors[(i, j)] * eigenvalues[j].exp()
    });
    let result = &scaled * eigenvectors.transpose();
    if !matrix_is_finite(&result) {
        return Err("Riemannian covariance update overflowed".to_string());
    }
    Ok(result)
}

fn validate_riemannian_inputs(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_s: &Mat<f64>,
    sigma2: f64,
) -> Result<(), String> {
    let placeholder_variances = vec![0.0; z_blocks.len()];
    validate_reml_inputs(y, x, z_blocks, &placeholder_variances, sigma2)?;

    let k = z_blocks.len();
    if k == 0 {
        return Err("Riemannian REML requires at least one Z block".to_string());
    }
    if current_s.nrows() != k || current_s.ncols() != k {
        return Err(format!(
            "S must be a {k} by {k} matrix to match the Z blocks, got {} by {}",
            current_s.nrows(),
            current_s.ncols()
        ));
    }
    if !matrix_is_finite(current_s) {
        return Err("S must contain only finite values".to_string());
    }
    for row in 0..k {
        for column in 0..row {
            let tolerance = 1e-12
                * current_s[(row, column)]
                    .abs()
                    .max(current_s[(column, row)].abs())
                    .max(1.0);
            if (current_s[(row, column)] - current_s[(column, row)]).abs() > tolerance {
                return Err("S must be symmetric".to_string());
            }
        }
    }
    Llt::new(current_s.as_ref(), Side::Lower).map_err(|_| "S not positive definite")?;
    if let Some(first) = z_blocks.first() {
        let columns = first.ncols();
        for (index, z) in z_blocks.iter().enumerate().skip(1) {
            if z.ncols() != columns {
                return Err(format!(
                    "all Z blocks must have {columns} columns for a Riemannian covariance update; block {index} has {}",
                    z.ncols()
                ));
            }
        }
    }
    Ok(())
}

fn covariance_score(
    projection: &RemlProjection,
    y: &Mat<f64>,
    z_blocks: &[Mat<f64>],
) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let k = z_blocks.len();
    let py = projection.apply(y);
    let projected_z: Vec<Mat<f64>> = z_blocks.iter().map(|z| projection.apply(z)).collect();
    let coefficients: Vec<Mat<f64>> = z_blocks.iter().map(|z| z.transpose() * &py).collect();
    let mut score = Mat::zeros(k, k);
    let mut traces = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..=i {
            let trace = compute_trace_product_ref(&projected_z[i], z_blocks[j].transpose());
            // Covariance couples the same level in each random-effect block.
            let quadratic = (coefficients[i].transpose() * &coefficients[j])[(0, 0)];
            let value = 0.5 * (quadratic - trace);
            score[(i, j)] = value;
            score[(j, i)] = value;
            traces[(i, j)] = trace;
            traces[(j, i)] = trace;
        }
    }
    (score, py, traces)
}

fn covariance_score_norm(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    s: &Mat<f64>,
    sigma2: f64,
) -> Result<f64, String> {
    let n = y.nrows() as f64;
    let v = correlated_covariance(z_blocks, s, sigma2);
    let projection = RemlProjection::new(&v, x)?;
    let (score, py, traces) = covariance_score(&projection, y, z_blocks);
    let scales: Vec<f64> = z_blocks
        .iter()
        .enumerate()
        .map(|(i, z)| {
            s[(i, i)]
                .max(sigma2 * n / squared_norm(z).max(f64::MIN_POSITIVE))
                .sqrt()
        })
        .collect();
    let normalized = Mat::from_fn(s.nrows(), s.ncols(), |i, j| {
        score[(i, j)] * scales[i] * scales[j] / n
    });
    let eigenvalues = normalized
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|_| "Riemannian score eigendecomposition did not converge".to_string())?;
    // The PSD-constrained covariance KKT conditions are score <= 0 and
    // tr(S score) = 0. Positive scores must be detected even as S approaches 0.
    let positive_score = eigenvalues
        .into_iter()
        .fold(0.0_f64, |value, eigen| value.max(eigen));
    let complementarity = compute_trace_product_ref(s, score.as_ref()).abs() / n;
    let random_trace = compute_trace_product_ref(s, traces.as_ref());
    let residual_score =
        0.5 * (squared_norm(&py) - projection.residual_trace(random_trace, sigma2));
    let residual = if sigma2 <= 1e-10 {
        residual_score.max(0.0)
    } else {
        residual_score.abs()
    };
    Ok(positive_score
        .max(complementarity)
        .max(residual * sigma2 / n))
}

pub fn riemannian_reml_step(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_s: &Mat<f64>,
    sigma2: f64,
    step_size: f64,
) -> Result<(Mat<f64>, f64), String> {
    validate_riemannian_inputs(y, x, z_blocks, current_s, sigma2)?;
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err("step_size must be finite and greater than zero".to_string());
    }
    let v = correlated_covariance(z_blocks, current_s, sigma2);
    let projection = RemlProjection::new(&v, x)?;
    let (score, py, traces) = covariance_score(&projection, y, z_blocks);
    let chol = Llt::new(current_s.as_ref(), Side::Lower)
        .map_err(|_| "S not positive definite".to_string())?;
    let l = chol.L();
    let tangent = l.transpose() * &score * l;
    let random_trace = compute_trace_product_ref(current_s, traces.as_ref());
    let residual_update = mm_variance_update(
        sigma2,
        squared_norm(&py),
        projection.residual_trace(random_trace, sigma2),
    );
    let log_residual_ratio = (residual_update / sigma2).ln();
    let objective = projection.objective(y);
    let mut step = step_size;
    for _ in 0..40 {
        // Congruence with the Cholesky factor keeps the update symmetric and
        // positive definite, including when a Taylor expansion would fail.
        if let Ok(exponential) = symmetric_exponential(&(step * &tangent)) {
            let new_s = l * &exponential * l.transpose();
            let new_sigma2 = (sigma2 * (step / step_size * log_residual_ratio).exp()).max(1e-10);
            if Llt::new(new_s.as_ref(), Side::Lower).is_ok() {
                let v = correlated_covariance(z_blocks, &new_s, new_sigma2);
                if let Ok(candidate_projection) = RemlProjection::new(&v, x) {
                    let candidate_objective = candidate_projection.objective(y);
                    if candidate_objective.is_finite()
                        && candidate_objective <= objective + 1e-12 * objective.abs().max(1.0)
                    {
                        return Ok((new_s, new_sigma2));
                    }
                }
            }
        }
        step *= 0.5;
    }
    Err("Riemannian REML could not find a likelihood-improving step".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn riemannian_reml_iterate(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    init_s: &Mat<f64>,
    init_sigma2: f64,
    max_iter: usize,
    tol: f64,
    step_size: f64,
) -> Result<RemlResult, String> {
    validate_riemannian_inputs(y, x, z_blocks, init_s, init_sigma2)?;
    validate_iteration_parameters(tol)?;
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err("step_size must be finite and greater than zero".to_string());
    }

    let k = init_s.nrows();
    let mut s = init_s.clone();
    let mut sigma2 = init_sigma2;

    for iter in 0..max_iter {
        let (new_s, new_sigma2) = riemannian_reml_step(y, x, z_blocks, &s, sigma2, step_size)?;

        let mut max_change = (new_sigma2 - sigma2).abs() / sigma2.max(1e-10);
        for i in 0..k {
            for j in 0..k {
                let change = (new_s[(i, j)] - s[(i, j)]).abs() / s[(i, j)].abs().max(1e-10);
                if change > max_change {
                    max_change = change;
                }
            }
        }

        s = new_s;
        sigma2 = new_sigma2;

        if max_change < tol && covariance_score_norm(y, x, z_blocks, &s, sigma2)? < tol {
            let mut variances = vec![0.0; k];
            for i in 0..k {
                variances[i] = s[(i, i)];
            }

            return Ok(RemlResult {
                variance_components: variances,
                sigma2,
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    let mut variances = vec![0.0; k];
    for i in 0..k {
        variances[i] = s[(i, i)];
    }

    Ok(RemlResult {
        variance_components: variances,
        sigma2,
        iterations: max_iter,
        converged: false,
    })
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    z_data_list,
    init_variances,
    init_sigma2,
    max_iter = 100,
    tol = 1e-6
))]
#[allow(clippy::too_many_arguments)]
pub fn mm_reml<'py>(
    py: Python<'py>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data_list: Vec<numpy::PyArrayLike2<'py, f64>>,
    init_variances: numpy::PyArrayLike1<'py, f64>,
    init_sigma2: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(pyo3::Py<PyArray1<f64>>, f64, usize, bool)> {
    let y_array = y.as_array();
    let x_array = x.as_array();
    let n = y_array.len();
    let p = x_array.ncols();
    if x_array.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x must have {n} rows to match y, got {}",
            x_array.nrows()
        )));
    }

    let y_mat = Mat::from_fn(n, 1, |i, _| y_array[i]);
    let x_mat = Mat::from_fn(n, p, |i, j| x_array[[i, j]]);

    let z_blocks: Vec<Mat<f64>> = z_data_list
        .iter()
        .map(|z_arr| {
            let arr = z_arr.as_array();
            Mat::from_fn(arr.nrows(), arr.ncols(), |i, j| arr[[i, j]])
        })
        .collect();

    let init_vars: Vec<f64> = init_variances.as_slice()?.to_vec();

    let result = mm_reml_iterate(
        &y_mat,
        &x_mat,
        &z_blocks,
        &init_vars,
        init_sigma2,
        max_iter,
        tol,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        PyArray1::from_vec(py, result.variance_components).into(),
        result.sigma2,
        result.iterations,
        result.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    z_data_list,
    init_variances,
    init_sigma2,
    max_iter = 100,
    tol = 1e-6
))]
#[allow(clippy::too_many_arguments)]
pub fn augmented_ai_reml<'py>(
    py: Python<'py>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data_list: Vec<numpy::PyArrayLike2<'py, f64>>,
    init_variances: numpy::PyArrayLike1<'py, f64>,
    init_sigma2: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(pyo3::Py<PyArray1<f64>>, f64, usize, bool)> {
    let y_array = y.as_array();
    let x_array = x.as_array();
    let n = y_array.len();
    let p = x_array.ncols();
    if x_array.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x must have {n} rows to match y, got {}",
            x_array.nrows()
        )));
    }

    let y_mat = Mat::from_fn(n, 1, |i, _| y_array[i]);
    let x_mat = Mat::from_fn(n, p, |i, j| x_array[[i, j]]);

    let z_blocks: Vec<Mat<f64>> = z_data_list
        .iter()
        .map(|z_arr| {
            let arr = z_arr.as_array();
            Mat::from_fn(arr.nrows(), arr.ncols(), |i, j| arr[[i, j]])
        })
        .collect();

    let init_vars: Vec<f64> = init_variances.as_slice()?.to_vec();

    let result = augmented_ai_reml_iterate(
        &y_mat,
        &x_mat,
        &z_blocks,
        &init_vars,
        init_sigma2,
        max_iter,
        tol,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        PyArray1::from_vec(py, result.variance_components).into(),
        result.sigma2,
        result.iterations,
        result.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    x,
    z_data_list,
    init_variances,
    init_sigma2,
    max_iter = 100,
    tol = 1e-6,
    step_size = 0.1
))]
#[allow(clippy::too_many_arguments)]
pub fn riemannian_reml<'py>(
    py: Python<'py>,
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data_list: Vec<numpy::PyArrayLike2<'py, f64>>,
    init_variances: numpy::PyArrayLike1<'py, f64>,
    init_sigma2: f64,
    max_iter: usize,
    tol: f64,
    step_size: f64,
) -> PyResult<(pyo3::Py<PyArray1<f64>>, f64, usize, bool)> {
    let y_array = y.as_array();
    let x_array = x.as_array();
    let n = y_array.len();
    let p = x_array.ncols();
    if x_array.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x must have {n} rows to match y, got {}",
            x_array.nrows()
        )));
    }

    let y_mat = Mat::from_fn(n, 1, |i, _| y_array[i]);
    let x_mat = Mat::from_fn(n, p, |i, j| x_array[[i, j]]);

    let z_blocks: Vec<Mat<f64>> = z_data_list
        .iter()
        .map(|z_arr| {
            let arr = z_arr.as_array();
            Mat::from_fn(arr.nrows(), arr.ncols(), |i, j| arr[[i, j]])
        })
        .collect();

    let k = z_blocks.len();
    let init_vars: Vec<f64> = init_variances.as_slice()?.to_vec();
    if init_vars.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "init_variances must contain one value per Z block, got {} values for {k} blocks",
            init_vars.len()
        )));
    }
    if init_vars
        .iter()
        .any(|variance| !variance.is_finite() || *variance <= 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "init_variances must contain finite values greater than zero for Riemannian REML",
        ));
    }

    let mut init_s = Mat::zeros(k, k);
    for i in 0..k {
        init_s[(i, i)] = init_vars[i];
    }

    let result = riemannian_reml_iterate(
        &y_mat,
        &x_mat,
        &z_blocks,
        &init_s,
        init_sigma2,
        max_iter,
        tol,
        step_size,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        PyArray1::from_vec(py, result.variance_components).into(),
        result.sigma2,
        result.iterations,
        result.converged,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> (Mat<f64>, Mat<f64>, Vec<Mat<f64>>) {
        let n = 12;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { (i % 3) as f64 - 1.0 });
        let z = Mat::from_fn(n, 4, |i, j| if i / 3 == j { 1.0 } else { 0.0 });
        let slope = Mat::from_fn(n, 4, |i, j| z[(i, j)] * x[(i, 1)]);
        let y = Mat::from_fn(n, 1, |i, _| {
            1.0 + 0.3 * x[(i, 1)]
                + 0.7 * (i / 3) as f64
                + [
                    0.2, -0.5, 0.7, -0.4, 0.9, 0.1, 0.6, -0.2, 0.5, -0.6, 0.8, -0.3,
                ][i]
        });
        (y, x, vec![z, slope])
    }

    #[test]
    fn residual_trace_handles_saturated_covariance_without_cancellation() {
        let n = 5;
        let variance = 1e14;
        let sigma2 = 1.0;
        let z = Mat::<f64>::identity(n, n);
        let x = Mat::zeros(n, 0);
        let v = variance_covariance(std::slice::from_ref(&z), &[variance], sigma2, n);
        let projection = RemlProjection::new(&v, &x).unwrap();
        let random_trace =
            variance * compute_trace_product_ref(&projection.apply(&z), z.transpose());
        let actual = projection.residual_trace(random_trace, sigma2);
        let expected = n as f64 / (variance + sigma2);
        assert!((actual / expected - 1.0).abs() < 1e-14);
    }

    #[test]
    fn ai_information_matches_dense_derivative_bilinear_forms() {
        let (y, x, blocks) = problem();
        let variances = [0.8, 0.2];
        let sigma2 = 0.6;
        let (_, _, actual) = augmented_ai_reml_step(&y, &x, &blocks, &variances, sigma2).unwrap();
        let v = variance_covariance(&blocks, &variances, sigma2, y.nrows());
        let projection = RemlProjection::new(&v, &x).unwrap();
        let p = projection.apply(&Mat::<f64>::identity(y.nrows(), y.nrows()));
        let py = &p * &y;
        let mut derivatives: Vec<Mat<f64>> = blocks.iter().map(|z| z * z.transpose()).collect();
        derivatives.push(Mat::<f64>::identity(y.nrows(), y.nrows()));
        for i in 0..derivatives.len() {
            for j in 0..derivatives.len() {
                let expected =
                    0.5 * (py.transpose() * &derivatives[i] * &p * &derivatives[j] * &py)[(0, 0)];
                assert!((actual[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn covariance_score_matches_reml_likelihood_finite_differences() {
        let (y, x, blocks) = problem();
        let s = Mat::from_fn(2, 2, |i, j| if i == j { 0.7 + 0.3 * i as f64 } else { 0.2 });
        let sigma2 = 0.6;
        let projection =
            RemlProjection::new(&correlated_covariance(&blocks, &s, sigma2), &x).unwrap();
        let (score, _, _) = covariance_score(&projection, &y, &blocks);
        let epsilon = 1e-5;
        for i in 0..2 {
            for j in 0..=i {
                let mut plus = s.clone();
                let mut minus = s.clone();
                plus[(i, j)] += epsilon;
                minus[(i, j)] -= epsilon;
                if i != j {
                    plus[(j, i)] += epsilon;
                    minus[(j, i)] -= epsilon;
                }
                let objective = |s: &Mat<f64>| {
                    RemlProjection::new(&correlated_covariance(&blocks, s, sigma2), &x)
                        .unwrap()
                        .objective(&y)
                };
                let derivative = (objective(&plus) - objective(&minus)) / (2.0 * epsilon);
                let expected = -2.0 * score[(i, j)] * if i == j { 1.0 } else { 2.0 };
                assert!((derivative - expected).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn riemannian_large_step_preserves_positive_definiteness_and_likelihood() {
        let (y, x, blocks) = problem();
        let s = Mat::from_fn(2, 2, |i, j| if i == j { 0.7 + 0.3 * i as f64 } else { 0.2 });
        let sigma2 = 0.6;
        let objective = |s: &Mat<f64>, sigma2: f64| {
            RemlProjection::new(&correlated_covariance(&blocks, s, sigma2), &x)
                .unwrap()
                .objective(&y)
        };
        let (updated_s, updated_sigma2) =
            riemannian_reml_step(&y, &x, &blocks, &s, sigma2, 100.0).unwrap();
        assert!(Llt::new(updated_s.as_ref(), Side::Lower).is_ok());
        assert!((updated_s[(0, 1)] - updated_s[(1, 0)]).abs() < 1e-10);
        assert!(updated_sigma2 > 0.0);
        assert!(objective(&updated_s, updated_sigma2) < objective(&s, sigma2));
    }

    #[test]
    fn iterative_reml_algorithms_improve_a_noisy_mixed_model() {
        let (y, x, mut blocks) = problem();
        blocks.truncate(1);
        let initial =
            RemlProjection::new(&variance_covariance(&blocks, &[0.8], 0.6, y.nrows()), &x)
                .unwrap()
                .objective(&y);
        for result in [
            mm_reml_iterate(&y, &x, &blocks, &[0.8], 0.6, 30, 1e-5),
            augmented_ai_reml_iterate(&y, &x, &blocks, &[0.8], 0.6, 30, 1e-5),
            riemannian_reml_iterate(
                &y,
                &x,
                &blocks,
                &Mat::from_fn(1, 1, |_, _| 0.8),
                0.6,
                30,
                1e-5,
                0.1,
            ),
        ] {
            let result = result.unwrap();
            let v = variance_covariance(
                &blocks,
                &result.variance_components,
                result.sigma2,
                y.nrows(),
            );
            let objective = RemlProjection::new(&v, &x).unwrap().objective(&y);
            assert!(objective < initial);
        }
    }
}

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

fn compute_quadratic_form(x: &Mat<f64>, a: &Mat<f64>) -> f64 {
    let n = x.nrows();
    let mut result = 0.0;
    for i in 0..n {
        for j in 0..n {
            result += x[(i, 0)] * a[(i, j)] * x[(j, 0)];
        }
    }
    result
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

pub fn mm_reml_step(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_variances: &[f64],
    sigma2: f64,
) -> Result<(Vec<f64>, f64), String> {
    validate_reml_inputs(y, x, z_blocks, current_variances, sigma2)?;

    let n = y.nrows();
    let p = x.ncols();
    let k = z_blocks.len();

    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = sigma2;
    }
    for (idx, z) in z_blocks.iter().enumerate() {
        let zzt = z * z.transpose();
        for i in 0..n {
            for j in 0..n {
                v[(i, j)] += current_variances[idx] * zzt[(i, j)];
            }
        }
    }

    let chol_v = Llt::new(v.as_ref(), Side::Lower).map_err(|_| "V not positive definite")?;
    let v_inv_x = chol_v.solve(x);
    let xt_vinv_x = x.transpose() * &v_inv_x;
    let chol_xtvx = if p == 0 {
        None
    } else {
        Some(
            Llt::new(xt_vinv_x.as_ref(), Side::Lower)
                .map_err(|_| "X'V^-1 X not positive definite")?,
        )
    };

    let project = |rhs: &Mat<f64>| {
        let v_inv_rhs = chol_v.solve(rhs);
        if let Some(chol) = &chol_xtvx {
            let fixed_rhs = x.transpose() * &v_inv_rhs;
            let fixed_coefficients = chol.solve(&fixed_rhs);
            &v_inv_rhs - &v_inv_x * &fixed_coefficients
        } else {
            v_inv_rhs
        }
    };

    let py_vec = project(y);
    let df = (n - p) as f64;

    let mut new_variances = vec![0.0; k];
    for (idx, z) in z_blocks.iter().enumerate() {
        let zt = z.transpose();
        let pz = project(z);
        let trace_pzzt = compute_trace_product_ref(&pz, zt);

        let zt_py = zt.as_ref() * &py_vec;
        let quad_form = squared_norm(&zt_py);

        let c = current_variances[idx] * current_variances[idx];
        let numerator = quad_form;
        let denominator = trace_pzzt;

        if denominator.abs() > 1e-10 {
            new_variances[idx] = (c * numerator / denominator).max(1e-10);
        } else {
            new_variances[idx] = current_variances[idx];
        }
    }

    let py_quad = squared_norm(&py_vec);
    let new_sigma2 = (sigma2 * sigma2 * py_quad / df).max(1e-10);

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

        if max_change < tol {
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
    let p = x.ncols();
    let k = z_blocks.len();

    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = sigma2;
    }
    for (idx, z) in z_blocks.iter().enumerate() {
        let zzt = z * z.transpose();
        for i in 0..n {
            for j in 0..n {
                v[(i, j)] += current_variances[idx] * zzt[(i, j)];
            }
        }
    }

    let chol_v = Llt::new(v.as_ref(), Side::Lower).map_err(|_| "V not positive definite")?;
    let v_inv = chol_v.solve(&Mat::<f64>::identity(n, n));

    let xt_vinv = x.transpose() * &v_inv;
    let xt_vinv_x = &xt_vinv * x;
    let chol_xtvx =
        Llt::new(xt_vinv_x.as_ref(), Side::Lower).map_err(|_| "X'V^-1 X not positive definite")?;
    let xtvx_inv = chol_xtvx.solve(&Mat::<f64>::identity(p, p));

    let mut p_mat = v_inv.clone();
    let proj = &v_inv * x * &xtvx_inv * &xt_vinv;
    for i in 0..n {
        for j in 0..n {
            p_mat[(i, j)] -= proj[(i, j)];
        }
    }

    let py_vec = &p_mat * y;

    let mut score = vec![0.0; k + 1];
    let mut ai_matrix = Mat::zeros(k + 1, k + 1);

    for (idx, z) in z_blocks.iter().enumerate() {
        let zt = z.transpose();
        let pz = &p_mat * z;
        let zzt = z * zt.as_ref();

        let trace_pzzt = compute_trace_product_ref(&pz, zt);
        let zt_py = zt.as_ref() * &py_vec;
        let quad = squared_norm(&zt_py);

        score[idx] = -0.5 * trace_pzzt + 0.5 * quad;

        let pzzt = &p_mat * &zzt;
        let pzzt_py = &pzzt * &py_vec;
        ai_matrix[(idx, idx)] = 0.5 * compute_quadratic_form(&pzzt_py, &p_mat);

        for (jdx, z2) in z_blocks.iter().enumerate().skip(idx + 1) {
            let z2t = z2.transpose();
            let zz2t = z2 * z2t.as_ref();
            let cross = 0.5 * compute_quadratic_form(&pzzt_py, &(&p_mat * &zz2t));
            ai_matrix[(idx, jdx)] = cross;
            ai_matrix[(jdx, idx)] = cross;
        }

        let cross_sigma = 0.5 * compute_quadratic_form(&pzzt_py, &p_mat);
        ai_matrix[(idx, k)] = cross_sigma;
        ai_matrix[(k, idx)] = cross_sigma;
    }

    let trace_p = (0..n).map(|i| p_mat[(i, i)]).sum::<f64>();
    let py_quad = squared_norm(&py_vec);
    score[k] = -0.5 * trace_p + 0.5 * py_quad;

    let p_py = &p_mat * &py_vec;
    ai_matrix[(k, k)] = 0.5 * compute_quadratic_form(&p_py, &p_mat);

    let chol_ai =
        Llt::new(ai_matrix.as_ref(), Side::Lower).map_err(|_| "AI matrix not positive definite")?;
    let score_mat = Mat::from_fn(k + 1, 1, |i, _| score[i]);
    let delta = chol_ai.solve(&score_mat);

    let mut new_variances = vec![0.0; k];
    for i in 0..k {
        new_variances[i] = (current_variances[i] + delta[(i, 0)]).max(1e-10);
    }
    let new_sigma2 = (sigma2 + delta[(k, 0)]).max(1e-10);

    Ok((new_variances, new_sigma2, ai_matrix))
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

        if max_change < tol {
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

fn spd_exponential(x: &Mat<f64>) -> Mat<f64> {
    let n = x.nrows();
    let mut result = Mat::zeros(n, n);

    let mut term = Mat::<f64>::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] += term[(i, j)];
        }
    }

    for k in 1..20 {
        term = &term * x;
        let factor = 1.0 / (1..=k).product::<usize>() as f64;
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] += factor * term[(i, j)];
            }
        }
    }

    result
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

pub fn riemannian_gradient(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_s: &Mat<f64>,
    sigma2: f64,
) -> Result<Mat<f64>, String> {
    validate_riemannian_inputs(y, x, z_blocks, current_s, sigma2)?;

    let n = y.nrows();
    let p = x.ncols();
    let k = current_s.nrows();

    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = sigma2;
    }

    for (idx, z) in z_blocks.iter().enumerate() {
        for (jdx, z2) in z_blocks.iter().enumerate() {
            let factor = current_s[(idx, jdx)];
            let zzt = z * z2.transpose();
            for i in 0..n {
                for j in 0..n {
                    v[(i, j)] += factor * zzt[(i, j)];
                }
            }
        }
    }

    let chol_v = Llt::new(v.as_ref(), Side::Lower).map_err(|_| "V not positive definite")?;
    let v_inv = chol_v.solve(&Mat::<f64>::identity(n, n));

    let xt_vinv = x.transpose() * &v_inv;
    let xt_vinv_x = &xt_vinv * x;
    let chol_xtvx =
        Llt::new(xt_vinv_x.as_ref(), Side::Lower).map_err(|_| "X'V^-1 X not positive definite")?;
    let xtvx_inv = chol_xtvx.solve(&Mat::<f64>::identity(p, p));

    let mut p_mat = v_inv.clone();
    let proj = &v_inv * x * &xtvx_inv * &xt_vinv;
    for i in 0..n {
        for j in 0..n {
            p_mat[(i, j)] -= proj[(i, j)];
        }
    }

    let py_vec = &p_mat * y;

    let mut euclidean_grad = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..=i {
            let zi = &z_blocks[i];
            let zj = &z_blocks[j];
            let zi_t = zi.transpose();
            let zj_t = zj.transpose();

            let pzi = &p_mat * zi;

            let trace_term = compute_trace_product_ref(&pzi, zj_t);

            let zi_py = zi_t.as_ref() * &py_vec;
            let zj_py = zj_t.as_ref() * &py_vec;
            let quad_term = {
                let mut sum = 0.0;
                for ii in 0..zi.ncols() {
                    for jj in 0..zj.ncols() {
                        sum += zi_py[(ii, 0)] * zj_py[(jj, 0)];
                    }
                }
                sum
            };

            let grad_ij = -0.5 * trace_term + 0.5 * quad_term;
            euclidean_grad[(i, j)] = grad_ij;
            if i != j {
                euclidean_grad[(j, i)] = grad_ij;
            }
        }
    }

    let riemannian_grad = current_s * &euclidean_grad * current_s;

    Ok(riemannian_grad)
}

pub fn riemannian_reml_step(
    y: &Mat<f64>,
    x: &Mat<f64>,
    z_blocks: &[Mat<f64>],
    current_s: &Mat<f64>,
    sigma2: f64,
    step_size: f64,
) -> Result<(Mat<f64>, f64), String> {
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err("step_size must be finite and greater than zero".to_string());
    }
    let grad = riemannian_gradient(y, x, z_blocks, current_s, sigma2)?;

    let k = current_s.nrows();
    let chol_s =
        Llt::new(current_s.as_ref(), Side::Lower).map_err(|_| "S not positive definite")?;
    let s_inv = chol_s.solve(&Mat::<f64>::identity(k, k));

    let scaled_grad = Mat::from_fn(k, k, |i, j| step_size * grad[(i, j)]);
    let s_inv_grad = &s_inv * &scaled_grad;

    let exp_term = spd_exponential(&s_inv_grad);
    let new_s = current_s * &exp_term;

    let n = y.nrows();
    let p = x.ncols();

    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = sigma2;
    }
    for (idx, z) in z_blocks.iter().enumerate() {
        for (jdx, z2) in z_blocks.iter().enumerate() {
            let factor = current_s[(idx, jdx)];
            let zzt = z * z2.transpose();
            for i in 0..n {
                for j in 0..n {
                    v[(i, j)] += factor * zzt[(i, j)];
                }
            }
        }
    }

    let chol_v = Llt::new(v.as_ref(), Side::Lower).map_err(|_| "V not positive definite")?;
    let v_inv = chol_v.solve(&Mat::<f64>::identity(n, n));

    let xt_vinv = x.transpose() * &v_inv;
    let xt_vinv_x = &xt_vinv * x;
    let chol_xtvx =
        Llt::new(xt_vinv_x.as_ref(), Side::Lower).map_err(|_| "X'V^-1 X not positive definite")?;
    let xtvx_inv = chol_xtvx.solve(&Mat::<f64>::identity(p, p));

    let mut p_mat = v_inv.clone();
    let proj = &v_inv * x * &xtvx_inv * &xt_vinv;
    for i in 0..n {
        for j in 0..n {
            p_mat[(i, j)] -= proj[(i, j)];
        }
    }

    let py_vec = &p_mat * y;
    let df = (n - p) as f64;
    let py_quad = squared_norm(&py_vec);
    let new_sigma2 = (py_quad / df).max(1e-10);

    Ok((new_s, new_sigma2))
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

        if max_change < tol {
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

    #[test]
    fn test_mm_reml_basic() {
        let n = 20;
        let y = Mat::from_fn(n, 1, |i, _| (i as f64) + 0.5);
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let z = Mat::from_fn(n, 4, |i, j| if i / 5 == j { 1.0 } else { 0.0 });

        let result = mm_reml_iterate(&y, &x, &[z], &[1.0], 1.0, 50, 1e-4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_augmented_ai_reml_basic() {
        let n = 20;
        let y = Mat::from_fn(n, 1, |i, _| (i as f64) + 0.5);
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let z = Mat::from_fn(n, 4, |i, j| if i / 5 == j { 1.0 } else { 0.0 });

        let result = augmented_ai_reml_iterate(&y, &x, &[z], &[1.0], 1.0, 50, 1e-4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_riemannian_reml_basic() {
        let n = 20;
        let y = Mat::from_fn(n, 1, |i, _| (i as f64) + 0.5);
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let z = Mat::from_fn(n, 4, |i, j| if i / 5 == j { 1.0 } else { 0.0 });

        let init_s = Mat::from_fn(1, 1, |_, _| 1.0);
        let result = riemannian_reml_iterate(&y, &x, &[z], &init_s, 1.0, 50, 1e-4, 0.1);
        assert!(result.is_ok());
    }
}

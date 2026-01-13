use nalgebra::{Cholesky, DMatrix, DVector};
use nalgebra_sparse::csc::CscMatrix;
use pyo3::PyResult;
use pyo3::prelude::*;

use crate::linalg::LinalgError;

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
            LinkFunction::Log => DVector::from_fn(eta.len(), |i, _| eta[i].exp()),
            LinkFunction::Logit => {
                DVector::from_fn(eta.len(), |i, _| 1.0 / (1.0 + (-eta[i]).exp()))
            }
        }
    }

    fn deriv(&self, mu: &DVector<f64>) -> DVector<f64> {
        match self {
            LinkFunction::Identity => DVector::from_element(mu.len(), 1.0),
            LinkFunction::Log => DVector::from_fn(mu.len(), |i, _| 1.0 / mu[i].max(1e-10)),
            LinkFunction::Logit => DVector::from_fn(mu.len(), |i, _| {
                let m = mu[i].clamp(1e-10, 1.0 - 1e-10);
                1.0 / (m * (1.0 - m))
            }),
        }
    }
}

impl FamilyType {
    fn variance(&self, mu: &DVector<f64>) -> DVector<f64> {
        match self {
            FamilyType::Gaussian => DVector::from_element(mu.len(), 1.0),
            FamilyType::Binomial => DVector::from_fn(mu.len(), |i, _| {
                let m = mu[i].clamp(1e-10, 1.0 - 1e-10);
                m * (1.0 - m)
            }),
            FamilyType::Poisson => DVector::from_fn(mu.len(), |i, _| mu[i].max(1e-10)),
        }
    }

    fn deviance_resids(&self, y: &DVector<f64>, mu: &DVector<f64>, wt: &[f64]) -> f64 {
        let n = y.len();
        let mut sum = 0.0;

        match self {
            FamilyType::Gaussian => {
                for i in 0..n {
                    let r = y[i] - mu[i];
                    sum += wt[i] * r * r;
                }
            }
            FamilyType::Binomial => {
                for i in 0..n {
                    let yi = y[i];
                    let mui = mu[i].clamp(1e-10, 1.0 - 1e-10);

                    let term1 = if yi > 1e-10 {
                        yi * (yi / mui).ln()
                    } else {
                        0.0
                    };

                    let term2 = if (1.0 - yi) > 1e-10 {
                        (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln()
                    } else {
                        0.0
                    };

                    sum += 2.0 * wt[i] * (term1 + term2);
                }
            }
            FamilyType::Poisson => {
                for i in 0..n {
                    let yi = y[i];
                    let mui = mu[i].max(1e-10);

                    let term = if yi > 1e-10 {
                        yi * (yi / mui).ln()
                    } else {
                        0.0
                    };

                    sum += 2.0 * wt[i] * (term - (yi - mui));
                }
            }
        }

        sum
    }

    fn weights(&self, mu: &DVector<f64>, link: LinkFunction) -> DVector<f64> {
        let link_deriv = link.deriv(mu);
        let variance = self.variance(mu);

        DVector::from_fn(mu.len(), |i, _| {
            let d = link_deriv[i];
            let v = variance[i].max(1e-10);
            1.0 / (d * d * v).max(1e-10)
        })
    }

    fn default_link(&self) -> LinkFunction {
        match self {
            FamilyType::Gaussian => LinkFunction::Identity,
            FamilyType::Binomial => LinkFunction::Logit,
            FamilyType::Poisson => LinkFunction::Log,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomEffectStructure {
    pub n_levels: usize,
    pub n_terms: usize,
    pub correlated: bool,
}

fn csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix<f64>, LinalgError> {
    let (nrows, ncols) = shape;
    let indices_usize: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
    let indptr_usize: Vec<usize> = indptr.iter().map(|&i| i as usize).collect();

    CscMatrix::try_from_csc_data(nrows, ncols, indptr_usize, indices_usize, data.to_vec())
        .map_err(|e| LinalgError::InvalidSparseFormat(format!("{:?}", e)))
}

fn build_lambda_dense(theta: &[f64], structures: &[RandomEffectStructure]) -> DMatrix<f64> {
    let mut total_dim = 0;
    for s in structures {
        total_dim += s.n_levels * s.n_terms;
    }

    if total_dim == 0 {
        return DMatrix::zeros(0, 0);
    }

    let mut lambda = DMatrix::zeros(total_dim, total_dim);
    let mut theta_idx = 0;
    let mut block_offset = 0;

    for structure in structures {
        let q = structure.n_terms;
        let n_levels = structure.n_levels;

        let l_block: Vec<Vec<f64>> = if structure.correlated {
            let n_theta = q * (q + 1) / 2;
            let theta_block = &theta[theta_idx..theta_idx + n_theta];
            theta_idx += n_theta;

            let mut l = vec![vec![0.0; q]; q];
            let mut idx = 0;
            for i in 0..q {
                for j in 0..=i {
                    l[i][j] = theta_block[idx];
                    idx += 1;
                }
            }
            l
        } else {
            let theta_block = &theta[theta_idx..theta_idx + q];
            theta_idx += q;

            let mut l = vec![vec![0.0; q]; q];
            for i in 0..q {
                l[i][i] = theta_block[i];
            }
            l
        };

        for level in 0..n_levels {
            let level_offset = block_offset + level * q;
            for i in 0..q {
                for j in 0..=i {
                    lambda[(level_offset + i, level_offset + j)] = l_block[i][j];
                }
            }
        }

        block_offset += n_levels * q;
    }

    lambda
}

fn forward_solve_vec(l: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    let n = l.nrows();
    let mut x = DVector::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[(i, j)] * x[j];
        }
        x[i] = sum / l[(i, i)];
    }
    x
}

fn forward_solve_mat(l: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = l.nrows();
    let ncols = b.ncols();
    let mut result = DMatrix::zeros(n, ncols);
    for col in 0..ncols {
        for i in 0..n {
            let mut sum = b[(i, col)];
            for j in 0..i {
                sum -= l[(i, j)] * result[(j, col)];
            }
            result[(i, col)] = sum / l[(i, i)];
        }
    }
    result
}

#[derive(Debug)]
pub struct PirlsResult {
    pub beta: DVector<f64>,
    pub u: DVector<f64>,
    pub deviance: f64,
    pub converged: bool,
}

pub fn pirls_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix<f64>,
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
    let n = y.len();
    let p = x.ncols();
    let q = z.ncols();

    let mut beta = if let Some(b) = beta_start {
        b.clone()
    } else {
        let eta = x * &DVector::zeros(p) + offset;
        let mu = link.inverse(&eta);
        let link_deriv = link.deriv(&mu);
        let y_work: DVector<f64> =
            DVector::from_fn(n, |i, _| eta[i] + link_deriv[i] * (y[i] - mu[i]));

        let xtx = x.transpose() * x;
        let xty = x.transpose() * &y_work;

        match Cholesky::new(xtx.clone()) {
            Some(chol) => chol.solve(&xty),
            None => xtx
                .try_inverse()
                .map_or(DVector::zeros(p), |inv| inv * &xty),
        }
    };

    let mut u = if let Some(u_init) = u_start {
        u_init.clone()
    } else {
        DVector::zeros(q)
    };

    let lambda = build_lambda_dense(theta, structures);
    let lambda_t_lambda = lambda.transpose() * &lambda;

    let mut converged = false;

    for _iter in 0..maxiter {
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

        for i in 0..n {
            mu[i] = mu[i].clamp(1e-10, 1.0 - 1e-10);
        }

        let mut w_vec = family.weights(&mu, link);
        for i in 0..n {
            w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
        }

        let link_deriv = link.deriv(&mu);
        let z_vec: DVector<f64> = DVector::from_fn(n, |i, _| {
            eta[i] - offset[i] + link_deriv[i] * (y[i] - mu[i])
        });

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

        let xtwz_vec: DVector<f64> = DVector::from_fn(p, |i, _| {
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

        let mut c = &ztwz + &lambda_t_lambda;
        let chol_c = match Cholesky::new(c.clone()) {
            Some(ch) => ch,
            None => {
                for i in 0..q {
                    c[(i, i)] += 1e-6;
                }
                match Cholesky::new(c) {
                    Some(ch) => ch,
                    None => {
                        return PirlsResult {
                            beta,
                            u,
                            deviance: 1e10,
                            converged: false,
                        };
                    }
                }
            }
        };

        let l_c = chol_c.l();
        let rzx = forward_solve_mat(&l_c, &ztwx);
        let cu = forward_solve_vec(&l_c, &ztwz_vec);

        let xtvinvx = &xtwx - &(rzx.transpose() * &rzx);
        let xtvinvz = &xtwz_vec - &(rzx.transpose() * &cu);

        let beta_new = match Cholesky::new(xtvinvx.clone()) {
            Some(chol) => chol.solve(&xtvinvz),
            None => xtvinvx
                .clone()
                .try_inverse()
                .map_or(beta.clone(), |inv| inv * &xtvinvz),
        };

        let u_rhs = &ztwz_vec - &(&ztwx * &beta_new);
        let u_new = chol_c.solve(&u_rhs);

        let delta_beta = (&beta_new - &beta).abs().max();
        let delta_u = if q > 0 {
            (&u_new - &u).abs().max()
        } else {
            0.0
        };

        beta = beta_new;
        u = u_new;

        if delta_beta < tol && delta_u < tol {
            converged = true;
            break;
        }
    }

    let mut eta_final = x * &beta + offset;
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            eta_final[i] += z.values()[idx] * u[j];
        }
    }

    let mut mu_final = link.inverse(&eta_final);
    for i in 0..n {
        mu_final[i] = mu_final[i].clamp(1e-10, 1.0 - 1e-10);
    }

    let dev_resids = family.deviance_resids(y, &mu_final, weights);

    let mut lambda_t_lambda_reg = lambda_t_lambda.clone();
    for i in 0..q {
        lambda_t_lambda_reg[(i, i)] += 1e-10;
    }

    let u_penalty = match Cholesky::new(lambda_t_lambda_reg) {
        Some(chol) => {
            let solved = chol.solve(&u);
            u.dot(&solved)
        }
        None => 0.0,
    };

    let deviance = dev_resids + u_penalty;

    PirlsResult {
        beta,
        u,
        deviance,
        converged,
    }
}

pub fn laplace_deviance_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix<f64>,
    weights: &[f64],
    offset: &DVector<f64>,
    theta: &[f64],
    structures: &[RandomEffectStructure],
    family: FamilyType,
    link: LinkFunction,
    beta_start: Option<&DVector<f64>>,
    u_start: Option<&DVector<f64>>,
) -> (f64, DVector<f64>, DVector<f64>) {
    let n = y.len();
    let q = z.ncols();

    if q == 0 {
        let result = pirls_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start, 25,
            1e-6,
        );
        return (result.deviance, result.beta, result.u);
    }

    let result = pirls_impl(
        y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start, 25, 1e-6,
    );

    let beta = result.beta;
    let u = result.u;

    let lambda = build_lambda_dense(theta, structures);
    let lambda_t_lambda = lambda.transpose() * &lambda;

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
    for i in 0..n {
        mu[i] = mu[i].clamp(1e-10, 1.0 - 1e-10);
    }

    let dev_resids = family.deviance_resids(y, &mu, weights);

    let mut lambda_t_lambda_reg = lambda_t_lambda.clone();
    for i in 0..q {
        lambda_t_lambda_reg[(i, i)] += 1e-10;
    }

    let u_penalty = match Cholesky::new(lambda_t_lambda_reg.clone()) {
        Some(chol) => {
            let solved = chol.solve(&u);
            u.dot(&solved)
        }
        None => 0.0,
    };

    let mut deviance = dev_resids + u_penalty;

    let mut w_vec = family.weights(&mu, link);
    for i in 0..n {
        w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
    }

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

    let h = &ztwz + &lambda_t_lambda_reg;

    let logdet_h = match Cholesky::new(h.clone()) {
        Some(chol) => {
            let l = chol.l();
            2.0 * (0..q).map(|i| l[(i, i)].ln()).sum::<f64>()
        }
        None => {
            let eigvals = h.symmetric_eigenvalues();
            eigvals.iter().map(|&e| e.max(1e-10).ln()).sum::<f64>()
        }
    };

    deviance += logdet_h;

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
pub fn pirls(
    y: numpy::PyReadonlyArray1<'_, f64>,
    x: numpy::PyReadonlyArray2<'_, f64>,
    z_data: numpy::PyReadonlyArray1<'_, f64>,
    z_indices: numpy::PyReadonlyArray1<'_, i64>,
    z_indptr: numpy::PyReadonlyArray1<'_, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyReadonlyArray1<'_, f64>,
    offset: numpy::PyReadonlyArray1<'_, f64>,
    theta: numpy::PyReadonlyArray1<'_, f64>,
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

    let family_type = match family {
        "gaussian" => FamilyType::Gaussian,
        "binomial" => FamilyType::Binomial,
        "poisson" => FamilyType::Poisson,
        _ => FamilyType::Gaussian,
    };

    let link_fn = match link {
        "identity" => LinkFunction::Identity,
        "log" => LinkFunction::Log,
        "logit" => LinkFunction::Logit,
        _ => family_type.default_link(),
    };

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let n = y_arr.len();
    let p = x_arr.ncols();

    let y_vec = DVector::from_iterator(n, y_arr.iter().cloned());
    let x_mat = DMatrix::from_fn(n, p, |i, j| x_arr[[i, j]]);
    let z_mat = csc_from_scipy(
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
    )?;
    let offset_vec = DVector::from_iterator(n, offset.as_array().iter().cloned());

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
        25,
        1e-6,
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
pub fn laplace_deviance(
    y: numpy::PyReadonlyArray1<'_, f64>,
    x: numpy::PyReadonlyArray2<'_, f64>,
    z_data: numpy::PyReadonlyArray1<'_, f64>,
    z_indices: numpy::PyReadonlyArray1<'_, i64>,
    z_indptr: numpy::PyReadonlyArray1<'_, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyReadonlyArray1<'_, f64>,
    offset: numpy::PyReadonlyArray1<'_, f64>,
    theta: numpy::PyReadonlyArray1<'_, f64>,
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

    let family_type = match family {
        "gaussian" => FamilyType::Gaussian,
        "binomial" => FamilyType::Binomial,
        "poisson" => FamilyType::Poisson,
        _ => FamilyType::Gaussian,
    };

    let link_fn = match link {
        "identity" => LinkFunction::Identity,
        "log" => LinkFunction::Log,
        "logit" => LinkFunction::Logit,
        _ => family_type.default_link(),
    };

    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let n = y_arr.len();
    let p = x_arr.ncols();

    let y_vec = DVector::from_iterator(n, y_arr.iter().cloned());
    let x_mat = DMatrix::from_fn(n, p, |i, j| x_arr[[i, j]]);
    let z_mat = csc_from_scipy(
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
    )?;
    let offset_vec = DVector::from_iterator(n, offset.as_array().iter().cloned());

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

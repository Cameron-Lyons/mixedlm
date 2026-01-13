use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, Side};
use nalgebra_sparse::csc::CscMatrix;
use ndarray::{ArrayView1, ArrayView2};
use pyo3::PyResult;
use pyo3::prelude::*;

use crate::linalg::LinalgError;

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

fn build_lambda_faer(theta: &[f64], structures: &[RandomEffectStructure]) -> Mat<f64> {
    let mut total_dim = 0;
    for s in structures {
        total_dim += s.n_levels * s.n_terms;
    }

    if total_dim == 0 {
        return Mat::zeros(0, 0);
    }

    let mut lambda = Mat::zeros(total_dim, total_dim);
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

fn forward_solve_faer(l: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let n = l.nrows();
    let ncols = b.ncols();
    let mut x = b.clone();

    for col in 0..ncols {
        for i in 0..n {
            let mut sum = x[(i, col)];
            for j in 0..i {
                sum -= l[(i, j)] * x[(j, col)];
            }
            x[(i, col)] = sum / l[(i, i)];
        }
    }
    x
}

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
) -> PyResult<f64> {
    let n = y.len();
    let p = x_data.ncols();
    let q = z_shape.1;

    let y_adj: Vec<f64> = y
        .iter()
        .zip(offset.iter())
        .map(|(yi, oi)| yi - oi)
        .collect();
    let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();

    let x = Mat::from_fn(n, p, |i, j| x_data[[i, j]]);

    if q == 0 {
        let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);
        let wy = Mat::from_fn(n, 1, |i, _| sqrt_w[i] * y_adj[i]);

        let xtwx = wx.transpose() * &wx;
        let xtwy = wx.transpose() * &wy;

        let chol = match Llt::new(xtwx.as_ref(), Side::Lower) {
            Ok(c) => c,
            Err(_) => return Ok(1e10),
        };

        let beta = chol.solve(&xtwy);

        let mut wrss = 0.0;
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..p {
                pred += x[(i, j)] * beta[(j, 0)];
            }
            let resid = y_adj[i] - pred;
            wrss += weights[i] * resid * resid;
        }

        let denom = if reml { n - p } else { n } as f64;
        let sigma2 = wrss / denom;

        let logdet_xtwx: f64 = if reml {
            let l = chol.L();
            2.0 * (0..p).map(|i| l[(i, i)].ln()).sum::<f64>()
        } else {
            0.0
        };

        let mut dev = (n as f64) * (2.0 * std::f64::consts::PI * sigma2).ln() + wrss / sigma2;
        if reml {
            dev += logdet_xtwx - (p as f64) * sigma2.ln();
        }

        return Ok(dev);
    }

    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;

    let lambda = build_lambda_faer(theta, structures);

    let mut wz = Mat::zeros(n, q);
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            wz[(i, j)] = sqrt_w[i] * z.values()[idx];
        }
    }

    let ztwz = wz.transpose() * &wz;
    let lambdat_ztwz = lambda.transpose() * &ztwz;
    let lambdat_ztwz_lambda = &lambdat_ztwz * &lambda;

    let mut v_factor = lambdat_ztwz_lambda;
    for i in 0..q {
        v_factor[(i, i)] += 1.0;
    }

    let chol_v = match Llt::new(v_factor.as_ref(), Side::Lower) {
        Ok(c) => c,
        Err(_) => return Ok(1e10),
    };

    let l_v = chol_v.L();
    let logdet_v: f64 = 2.0 * (0..q).map(|i| l_v[(i, i)].ln()).sum::<f64>();

    let mut zt_wy_adj = Mat::zeros(q, 1);
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        let mut sum = 0.0;
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            sum += z.values()[idx] * weights[i] * y_adj[i];
        }
        zt_wy_adj[(j, 0)] = sum;
    }

    let cu = lambda.transpose() * &zt_wy_adj;
    let cu_star = forward_solve_faer(&l_v.to_owned(), &cu);

    let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);

    let mut zt_sqrtw_wx = Mat::zeros(q, p);
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for pj in 0..p {
            let mut sum = 0.0;
            for idx in col_start..col_end {
                let i = z.row_indices()[idx];
                sum += z.values()[idx] * sqrt_w[i] * wx[(i, pj)];
            }
            zt_sqrtw_wx[(j, pj)] = sum;
        }
    }

    let lambdat_ztwx = lambda.transpose() * &zt_sqrtw_wx;
    let rzx = forward_solve_faer(&l_v.to_owned(), &lambdat_ztwx);

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
        Err(_) => return Ok(1e10),
    };

    let l_xtvinvx = chol_xtvinvx.L();
    let logdet_xtvinvx: f64 = 2.0 * (0..p).map(|i| l_xtvinvx[(i, i)].ln()).sum::<f64>();

    let cu_star_rzx_beta_term = rzx.transpose() * &cu_star;
    let xty_adj = &xtwy - &cu_star_rzx_beta_term;
    let beta = chol_xtvinvx.solve(&xty_adj);

    let mut resid = Vec::with_capacity(n);
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += x[(i, j)] * beta[(j, 0)];
        }
        resid.push(y_adj[i] - pred);
    }

    let mut zt_w_resid = Mat::zeros(q, 1);
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        let mut sum = 0.0;
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            sum += z.values()[idx] * weights[i] * resid[i];
        }
        zt_w_resid[(j, 0)] = sum;
    }

    let lambda_t_zt_resid = lambda.transpose() * &zt_w_resid;
    let u_star = chol_v.solve(&lambda_t_zt_resid);

    let w_resid_sq: f64 = (0..n).map(|i| weights[i] * resid[i] * resid[i]).sum();
    let u_star_sq: f64 = (0..q).map(|i| u_star[(i, 0)].powi(2)).sum();
    let pwrss = w_resid_sq + u_star_sq;

    let denom = if reml { n - p } else { n } as f64;
    let sigma2 = pwrss / denom;

    let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v;
    if reml {
        dev += logdet_xtvinvx;
    }

    Ok(dev)
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
pub fn profiled_deviance(
    theta: numpy::PyReadonlyArray1<'_, f64>,
    y: numpy::PyReadonlyArray1<'_, f64>,
    x: numpy::PyReadonlyArray2<'_, f64>,
    z_data: numpy::PyReadonlyArray1<'_, f64>,
    z_indices: numpy::PyReadonlyArray1<'_, i64>,
    z_indptr: numpy::PyReadonlyArray1<'_, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyReadonlyArray1<'_, f64>,
    offset: numpy::PyReadonlyArray1<'_, f64>,
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
    )
}

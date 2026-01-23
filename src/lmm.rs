use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, Side};
use nalgebra_sparse::csc::CscMatrix;
use ndarray::{ArrayView1, ArrayView2};
use numpy::PyArray1;
use pyo3::PyResult;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::linalg::LinalgError;

#[derive(Debug, Clone, Copy)]
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

fn build_lambda_blocks(theta: &[f64], structures: &[RandomEffectStructure]) -> Vec<Mat<f64>> {
    let mut blocks = Vec::new();
    let mut theta_idx = 0;

    for structure in structures {
        let q = structure.n_terms;

        let l_block = if structure.correlated {
            let n_theta = q * (q + 1) / 2;
            let theta_block = &theta[theta_idx..theta_idx + n_theta];
            theta_idx += n_theta;

            let mut l = Mat::zeros(q, q);
            let mut idx = 0;
            for i in 0..q {
                for j in 0..=i {
                    l[(i, j)] = theta_block[idx];
                    idx += 1;
                }
            }
            l
        } else {
            let theta_block = &theta[theta_idx..theta_idx + q];
            theta_idx += q;

            let mut l = Mat::zeros(q, q);
            for i in 0..q {
                l[(i, i)] = theta_block[i];
            }
            l
        };

        blocks.push(l_block);
    }

    blocks
}

fn build_lambda_derivative_blocks(structures: &[RandomEffectStructure]) -> Vec<Vec<Mat<f64>>> {
    let mut all_derivs = Vec::new();

    for structure in structures {
        let q = structure.n_terms;
        let mut block_derivs = Vec::new();

        if structure.correlated {
            let n_theta = q * (q + 1) / 2;
            for k in 0..n_theta {
                let mut dl = Mat::zeros(q, q);
                let mut idx = 0;
                for i in 0..q {
                    for j in 0..=i {
                        if idx == k {
                            dl[(i, j)] = 1.0;
                        }
                        idx += 1;
                    }
                }
                block_derivs.push(dl);
            }
        } else {
            for i in 0..q {
                let mut dl = Mat::zeros(q, q);
                dl[(i, i)] = 1.0;
                block_derivs.push(dl);
            }
        }

        all_derivs.push(block_derivs);
    }

    all_derivs
}

fn compute_dv_dtheta(
    ztwz: &Mat<f64>,
    lambda_blocks: &[Mat<f64>],
    dlambda: &Mat<f64>,
    block_idx: usize,
    structures: &[RandomEffectStructure],
) -> Mat<f64> {
    let q = ztwz.nrows();
    let mut dv = Mat::zeros(q, q);

    let affected_structure = &structures[block_idx];
    let qi = affected_structure.n_terms;
    let ni = affected_structure.n_levels;
    let lambda_i = &lambda_blocks[block_idx];

    let mut affected_block_offset = 0;
    for (idx, s) in structures.iter().enumerate() {
        if idx == block_idx {
            break;
        }
        affected_block_offset += s.n_levels * s.n_terms;
    }

    for level in 0..ni {
        let offset_i = affected_block_offset + level * qi;

        let mut block_ii = Mat::zeros(qi, qi);
        for ii in 0..qi {
            for jj in 0..qi {
                block_ii[(ii, jj)] = ztwz[(offset_i + ii, offset_i + jj)];
            }
        }

        let dlambda_t = dlambda.transpose();
        let lambda_i_t = lambda_i.transpose();

        let term1 = dlambda_t * &block_ii * lambda_i;
        let term2 = lambda_i_t * &block_ii * dlambda;

        for ii in 0..qi {
            for jj in 0..qi {
                dv[(offset_i + ii, offset_i + jj)] += term1[(ii, jj)] + term2[(ii, jj)];
            }
        }
    }

    let mut block_offset_j = 0;
    for (struct_j, (structure_j, lambda_j)) in
        structures.iter().zip(lambda_blocks.iter()).enumerate()
    {
        let qj = structure_j.n_terms;
        let nj = structure_j.n_levels;

        if struct_j != block_idx {
            for level_i in 0..ni {
                let offset_i = affected_block_offset + level_i * qi;
                for level_j in 0..nj {
                    let offset_j = block_offset_j + level_j * qj;

                    let mut block_ij = Mat::zeros(qi, qj);
                    for ii in 0..qi {
                        for jj in 0..qj {
                            block_ij[(ii, jj)] = ztwz[(offset_i + ii, offset_j + jj)];
                        }
                    }

                    let dlambda_t = dlambda.transpose();
                    let term = dlambda_t * &block_ij * lambda_j;

                    for ii in 0..qi {
                        for jj in 0..qj {
                            dv[(offset_i + ii, offset_j + jj)] += term[(ii, jj)];
                            dv[(offset_j + jj, offset_i + ii)] += term[(ii, jj)];
                        }
                    }
                }
            }
        }

        block_offset_j += nj * qj;
    }

    dv
}

fn apply_dlambda_transpose_vector(
    v: &Mat<f64>,
    dlambda: &Mat<f64>,
    block_idx: usize,
    structures: &[RandomEffectStructure],
) -> Mat<f64> {
    let q = v.nrows();
    let mut result = Mat::zeros(q, v.ncols());

    let structure = &structures[block_idx];
    let qi = structure.n_terms;
    let ni = structure.n_levels;
    let dlambda_t = dlambda.transpose();

    let mut block_offset = 0;
    for (idx, s) in structures.iter().enumerate() {
        if idx == block_idx {
            break;
        }
        block_offset += s.n_levels * s.n_terms;
    }

    for level in 0..ni {
        let offset = block_offset + level * qi;

        for col in 0..v.ncols() {
            let mut block_v = Mat::zeros(qi, 1);
            for i in 0..qi {
                block_v[(i, 0)] = v[(offset + i, col)];
            }

            let transformed = dlambda_t * &block_v;

            for i in 0..qi {
                result[(offset + i, col)] = transformed[(i, 0)];
            }
        }
    }

    result
}

fn compute_ztwz_entry(z: &CscMatrix<f64>, weights: &[f64], j: usize, k: usize) -> f64 {
    let col_j_start = z.col_offsets()[j];
    let col_j_end = z.col_offsets()[j + 1];
    let col_k_start = z.col_offsets()[k];
    let col_k_end = z.col_offsets()[k + 1];

    let mut sum = 0.0;
    let mut idx_j = col_j_start;
    let mut idx_k = col_k_start;

    while idx_j < col_j_end && idx_k < col_k_end {
        let row_j = z.row_indices()[idx_j];
        let row_k = z.row_indices()[idx_k];

        if row_j == row_k {
            sum += z.values()[idx_j] * weights[row_j] * z.values()[idx_k];
            idx_j += 1;
            idx_k += 1;
        } else if row_j < row_k {
            idx_j += 1;
        } else {
            idx_k += 1;
        }
    }

    sum
}

fn compute_ztwz_sparse(z: &CscMatrix<f64>, weights: &[f64], q: usize) -> Mat<f64> {
    let mut ztwz = Mat::zeros(q, q);

    if q <= 50 {
        for j in 0..q {
            for k in j..q {
                let val = compute_ztwz_entry(z, weights, j, k);
                ztwz[(j, k)] = val;
                if j != k {
                    ztwz[(k, j)] = val;
                }
            }
        }
    } else {
        let entries: Vec<(usize, usize, f64)> = (0..q)
            .into_par_iter()
            .flat_map(|j| {
                (j..q)
                    .map(|k| (j, k, compute_ztwz_entry(z, weights, j, k)))
                    .collect::<Vec<_>>()
            })
            .collect();

        for (j, k, val) in entries {
            ztwz[(j, k)] = val;
            if j != k {
                ztwz[(k, j)] = val;
            }
        }
    }

    ztwz
}

fn mat_from_flat_array(data: &[f64], q: usize) -> Mat<f64> {
    Mat::from_fn(q, q, |i, j| data[i * q + j])
}

fn apply_lambda_block_transform(
    ztwz: &Mat<f64>,
    lambda_blocks: &[Mat<f64>],
    structures: &[RandomEffectStructure],
) -> Mat<f64> {
    let q = ztwz.nrows();
    let mut result = Mat::zeros(q, q);

    let mut block_offset_i = 0;
    for (struct_i, (structure_i, lambda_i)) in
        structures.iter().zip(lambda_blocks.iter()).enumerate()
    {
        let qi = structure_i.n_terms;
        let ni = structure_i.n_levels;

        let mut block_offset_j = 0;
        for (struct_j, (structure_j, lambda_j)) in
            structures.iter().zip(lambda_blocks.iter()).enumerate()
        {
            let qj = structure_j.n_terms;
            let nj = structure_j.n_levels;

            if struct_i == struct_j {
                for level in 0..ni {
                    let offset = block_offset_i + level * qi;

                    let mut block = Mat::zeros(qi, qi);
                    for ii in 0..qi {
                        for jj in 0..qi {
                            block[(ii, jj)] = ztwz[(offset + ii, offset + jj)];
                        }
                    }

                    let lambda_t = lambda_i.transpose();
                    let temp = lambda_t * &block;
                    let transformed = &temp * lambda_i;

                    for ii in 0..qi {
                        for jj in 0..qi {
                            result[(offset + ii, offset + jj)] = transformed[(ii, jj)];
                        }
                    }
                }
            } else {
                for level_i in 0..ni {
                    let offset_i = block_offset_i + level_i * qi;
                    for level_j in 0..nj {
                        let offset_j = block_offset_j + level_j * qj;

                        let mut block = Mat::zeros(qi, qj);
                        for ii in 0..qi {
                            for jj in 0..qj {
                                block[(ii, jj)] = ztwz[(offset_i + ii, offset_j + jj)];
                            }
                        }

                        let lambda_i_t = lambda_i.transpose();
                        let temp = lambda_i_t * &block;
                        let transformed = &temp * lambda_j;

                        for ii in 0..qi {
                            for jj in 0..qj {
                                result[(offset_i + ii, offset_j + jj)] = transformed[(ii, jj)];
                            }
                        }
                    }
                }
            }

            block_offset_j += nj * qj;
        }

        block_offset_i += ni * qi;
    }

    result
}

fn apply_lambda_transpose_vector(
    v: &Mat<f64>,
    lambda_blocks: &[Mat<f64>],
    structures: &[RandomEffectStructure],
) -> Mat<f64> {
    let q = v.nrows();
    let mut result = Mat::zeros(q, v.ncols());

    let mut block_offset = 0;
    for (structure, lambda) in structures.iter().zip(lambda_blocks.iter()) {
        let qi = structure.n_terms;
        let ni = structure.n_levels;
        let lambda_t = lambda.transpose();

        for level in 0..ni {
            let offset = block_offset + level * qi;

            for col in 0..v.ncols() {
                let mut block_v = Mat::zeros(qi, 1);
                for i in 0..qi {
                    block_v[(i, 0)] = v[(offset + i, col)];
                }

                let transformed = lambda_t * &block_v;

                for i in 0..qi {
                    result[(offset + i, col)] = transformed[(i, 0)];
                }
            }
        }

        block_offset += ni * qi;
    }

    result
}

fn compute_ztwy_sparse(z: &CscMatrix<f64>, w: &[f64], y: &[f64], q: usize) -> Mat<f64> {
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

fn compute_ztwx_sparse(
    z: &CscMatrix<f64>,
    w: &[f64],
    x: &Mat<f64>,
    q: usize,
    p: usize,
) -> Mat<f64> {
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
    let n = y.len();
    let p = x_data.ncols();
    let q = z_shape.1;

    let y_adj: Vec<f64> = y
        .iter()
        .zip(offset.iter())
        .map(|(yi, oi)| yi - oi)
        .collect();
    let w: Vec<f64> = weights.iter().copied().collect();
    let sqrt_w: Vec<f64> = weights.iter().map(|wi| wi.sqrt()).collect();

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

        let mut dev = (n as f64) * (2.0 * std::f64::consts::PI * sigma2).ln() + wrss / sigma2;
        if reml {
            dev += logdet_xtwx - (p as f64) * sigma2.ln();
        }

        return Ok(dev);
    }

    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
    let lambda_blocks = build_lambda_blocks(theta, structures);

    let ztwz = if let Some(cached_data) = ztwz_cache {
        mat_from_flat_array(cached_data, q)
    } else {
        compute_ztwz_sparse(&z, &w, q)
    };
    let lambdat_ztwz_lambda = apply_lambda_block_transform(&ztwz, &lambda_blocks, structures);

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

    let ztwy = compute_ztwy_sparse(&z, &w, &y_adj, q);
    let cu = apply_lambda_transpose_vector(&ztwy, &lambda_blocks, structures);
    let cu_star = chol_v.solve(&cu);

    let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);

    let ztwx = compute_ztwx_sparse(&z, &w, &x, q, p);
    let lambdat_ztwx = apply_lambda_transpose_vector(&ztwx, &lambda_blocks, structures);
    let rzx = chol_v.solve(&lambdat_ztwx);

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

    let zt_w_resid = compute_ztwy_sparse(&z, &w, &resid, q);
    let lambda_t_zt_resid = apply_lambda_transpose_vector(&zt_w_resid, &lambda_blocks, structures);
    let u_star = chol_v.solve(&lambda_t_zt_resid);

    let w_resid_sq: f64 = (0..n).map(|i| w[i] * resid[i] * resid[i]).sum();
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
    let w: Vec<f64> = weights.iter().copied().collect();
    let sqrt_w: Vec<f64> = weights.iter().map(|wi| wi.sqrt()).collect();

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

        let mut dev = (n as f64) * (2.0 * std::f64::consts::PI * sigma2).ln() + wrss / sigma2;
        if reml {
            dev += logdet_xtwx - (p as f64) * sigma2.ln();
        }

        return Ok((dev, vec![0.0; n_theta]));
    }

    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
    let lambda_blocks = build_lambda_blocks(theta, structures);
    let dlambda_blocks = build_lambda_derivative_blocks(structures);

    let ztwz = if let Some(cached_data) = ztwz_cache {
        mat_from_flat_array(cached_data, q)
    } else {
        compute_ztwz_sparse(&z, &w, q)
    };
    let lambdat_ztwz_lambda = apply_lambda_block_transform(&ztwz, &lambda_blocks, structures);

    let mut v_factor = lambdat_ztwz_lambda.clone();
    for i in 0..q {
        v_factor[(i, i)] += 1.0;
    }

    let chol_v = match Llt::new(v_factor.as_ref(), Side::Lower) {
        Ok(c) => c,
        Err(_) => return Ok((1e10, vec![0.0; n_theta])),
    };

    let l_v = chol_v.L();
    let logdet_v: f64 = 2.0 * (0..q).map(|i| l_v[(i, i)].ln()).sum::<f64>();

    let ztwy = compute_ztwy_sparse(&z, &w, &y_adj, q);
    let cu = apply_lambda_transpose_vector(&ztwy, &lambda_blocks, structures);
    let cu_star = chol_v.solve(&cu);

    let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);

    let ztwx = compute_ztwx_sparse(&z, &w, &x, q, p);
    let lambdat_ztwx = apply_lambda_transpose_vector(&ztwx, &lambda_blocks, structures);
    let rzx = chol_v.solve(&lambdat_ztwx);

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

    let mut resid = Vec::with_capacity(n);
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += x[(i, j)] * beta[(j, 0)];
        }
        resid.push(y_adj[i] - pred);
    }

    let zt_w_resid = compute_ztwy_sparse(&z, &w, &resid, q);
    let lambda_t_zt_resid = apply_lambda_transpose_vector(&zt_w_resid, &lambda_blocks, structures);
    let u_star = chol_v.solve(&lambda_t_zt_resid);

    let w_resid_sq: f64 = (0..n).map(|i| w[i] * resid[i] * resid[i]).sum();
    let u_star_sq: f64 = (0..q).map(|i| u_star[(i, 0)].powi(2)).sum();
    let pwrss = w_resid_sq + u_star_sq;

    let denom = if reml { n - p } else { n } as f64;
    let sigma2 = pwrss / denom;

    let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v;
    if reml {
        dev += logdet_xtvinvx;
    }

    let v_inv = chol_v.solve(&Mat::<f64>::identity(q, q));

    let mut gradient = Vec::with_capacity(n_theta);
    let mut theta_idx = 0;

    for (block_idx, (structure, block_derivs)) in
        structures.iter().zip(dlambda_blocks.iter()).enumerate()
    {
        let n_block_theta = if structure.correlated {
            structure.n_terms * (structure.n_terms + 1) / 2
        } else {
            structure.n_terms
        };

        for dlambda in block_derivs.iter().take(n_block_theta) {
            let dv = compute_dv_dtheta(&ztwz, &lambda_blocks, dlambda, block_idx, structures);

            let mut d_logdet_v = 0.0;
            for i in 0..q {
                for j in 0..q {
                    d_logdet_v += v_inv[(i, j)] * dv[(j, i)];
                }
            }

            let dlambdat_zt_w_resid =
                apply_dlambda_transpose_vector(&zt_w_resid, dlambda, block_idx, structures);
            let dlambdat_ztwx =
                apply_dlambda_transpose_vector(&ztwx, dlambda, block_idx, structures);
            let dlambdat_ztwy =
                apply_dlambda_transpose_vector(&ztwy, dlambda, block_idx, structures);

            let v_inv_dv_rzx = chol_v.solve(&(&dv * &rzx));
            let v_inv_dlambdat_ztwx = chol_v.solve(&dlambdat_ztwx);

            let mut d_rzx_t_rzx = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    let mut sum = 0.0;
                    for r in 0..q {
                        sum -=
                            v_inv_dv_rzx[(r, i)] * rzx[(r, j)] + rzx[(r, i)] * v_inv_dv_rzx[(r, j)];
                        sum += v_inv_dlambdat_ztwx[(r, i)] * rzx[(r, j)]
                            + rzx[(r, i)] * v_inv_dlambdat_ztwx[(r, j)];
                    }
                    d_rzx_t_rzx[(i, j)] = sum;
                }
            }

            let v_inv_dv_cu_star = chol_v.solve(&(&dv * &cu_star));
            let v_inv_dlambdat_ztwy = chol_v.solve(&dlambdat_ztwy);

            let mut d_rzx_t_cu_star = Mat::zeros(p, 1);
            for i in 0..p {
                let mut sum = 0.0;
                for r in 0..q {
                    sum -= v_inv_dv_rzx[(r, i)] * cu_star[(r, 0)]
                        + rzx[(r, i)] * v_inv_dv_cu_star[(r, 0)];
                    sum += v_inv_dlambdat_ztwx[(r, i)] * cu_star[(r, 0)]
                        + rzx[(r, i)] * v_inv_dlambdat_ztwy[(r, 0)];
                }
                d_rzx_t_cu_star[(i, 0)] = sum;
            }

            let xtvinvx_inv = chol_xtvinvx.solve(&Mat::<f64>::identity(p, p));
            let d_xtvinvx_beta = &d_rzx_t_rzx * &beta;
            let d_beta = &xtvinvx_inv * &(&d_xtvinvx_beta - &d_rzx_t_cu_star);

            let mut resid_w_x_dbeta = 0.0;
            for i in 0..n {
                let mut x_dbeta = 0.0;
                for j in 0..p {
                    x_dbeta += x[(i, j)] * d_beta[(j, 0)];
                }
                resid_w_x_dbeta += w[i] * resid[i] * x_dbeta;
            }
            let d_w_resid_sq = -2.0 * resid_w_x_dbeta;

            let v_inv_dv_u = chol_v.solve(&(&dv * &u_star));
            let mut u_star_v_inv_dv_u = 0.0;
            for i in 0..q {
                u_star_v_inv_dv_u += u_star[(i, 0)] * v_inv_dv_u[(i, 0)];
            }

            let v_inv_dlambdat = chol_v.solve(&dlambdat_zt_w_resid);
            let mut u_star_v_inv_dlambdat = 0.0;
            for i in 0..q {
                u_star_v_inv_dlambdat += u_star[(i, 0)] * v_inv_dlambdat[(i, 0)];
            }

            let lambdat_ztwx_dbeta =
                apply_lambda_transpose_vector(&ztwx, &lambda_blocks, structures) * &d_beta;
            let v_inv_lambdat_ztwx_dbeta = chol_v.solve(&lambdat_ztwx_dbeta);
            let mut u_star_v_inv_lambdat_ztwx_dbeta = 0.0;
            for i in 0..q {
                u_star_v_inv_lambdat_ztwx_dbeta +=
                    u_star[(i, 0)] * v_inv_lambdat_ztwx_dbeta[(i, 0)];
            }

            let d_u_star_sq = -2.0 * u_star_v_inv_dv_u + 2.0 * u_star_v_inv_dlambdat
                - 2.0 * u_star_v_inv_lambdat_ztwx_dbeta;

            let d_pwrss = d_w_resid_sq + d_u_star_sq;

            let mut grad_k = d_logdet_v + denom / pwrss * d_pwrss;

            if reml {
                let mut d_logdet_xtvinvx = 0.0;
                for i in 0..p {
                    for j in 0..p {
                        d_logdet_xtvinvx += xtvinvx_inv[(i, j)] * d_rzx_t_rzx[(j, i)];
                    }
                }
                grad_k += d_logdet_xtvinvx;
            }

            gradient.push(grad_k);
            theta_idx += 1;
        }
    }

    let _ = theta_idx;

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
    let w: Vec<f64> = weights.as_array().iter().copied().collect();
    let q = z_shape.1;

    let ztwz = compute_ztwz_sparse(&z, &w, q);

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

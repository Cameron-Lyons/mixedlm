use faer::linalg::solvers::{Llt, Solve};
use faer::{Mat, Side};
use nalgebra_sparse::csc::CscMatrix;
use numpy::PyArray1;
use numpy::ndarray::{ArrayView1, ArrayView2};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::blocked_chol::{BlockedCholesky, BlockedMatrix};
use crate::linalg::LinalgError;

#[derive(Debug, Clone, Copy)]
pub struct RandomEffectStructure {
    pub n_levels: usize,
    pub n_terms: usize,
    pub correlated: bool,
}

fn validate_prior_weights(weights: ArrayView1<'_, f64>, n: usize) -> PyResult<(Vec<f64>, f64)> {
    if weights.len() != n {
        return Err(PyValueError::new_err(format!(
            "weights has length {}, expected {n}",
            weights.len()
        )));
    }

    let mut values = Vec::with_capacity(n);
    let mut logdet = 0.0;
    for &weight in weights {
        if !weight.is_finite() {
            return Err(PyValueError::new_err(
                "weights must contain only finite values",
            ));
        }
        if weight <= 0.0 {
            return Err(PyValueError::new_err("weights must be strictly positive"));
        }
        values.push(weight);
        logdet += weight.ln();
    }
    Ok((values, logdet))
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

fn compute_ztwz_sparse(z: &CscMatrix<f64>, weights: &[f64]) -> Mat<f64> {
    let n = z.nrows();
    let q = z.ncols();
    let nnz = z.values().len();
    let mut row_offsets = vec![0_usize; n + 1];

    for &row in z.row_indices() {
        row_offsets[row + 1] += 1;
    }
    for row in 0..n {
        row_offsets[row + 1] += row_offsets[row];
    }

    let mut next_position = row_offsets[..n].to_vec();
    let mut row_columns = vec![0_usize; nnz];
    let mut row_values = vec![0.0; nnz];

    for column in 0..q {
        for index in z.col_offsets()[column]..z.col_offsets()[column + 1] {
            let row = z.row_indices()[index];
            let position = next_position[row];
            row_columns[position] = column;
            row_values[position] = z.values()[index];
            next_position[row] += 1;
        }
    }

    let mut ztwz = Mat::zeros(q, q);

    for row in 0..n {
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        let weight = weights[row];

        for left in start..end {
            let left_column = row_columns[left];
            let weighted_left = weight * row_values[left];

            for right in left..end {
                let right_column = row_columns[right];
                let value = weighted_left * row_values[right];
                ztwz[(left_column, right_column)] += value;
                if left_column != right_column {
                    ztwz[(right_column, left_column)] += value;
                }
            }
        }
    }

    ztwz
}

fn mat_from_flat_array(data: &[f64], q: usize) -> Mat<f64> {
    Mat::from_fn(q, q, |i, j| data[i * q + j])
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
    let (w, logdet_w) = validate_prior_weights(weights, n)?;
    let sqrt_w: Vec<f64> = w.iter().map(|wi| wi.sqrt()).collect();

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

        let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) - logdet_w;
        if reml {
            dev += logdet_xtwx;
        }

        return Ok(dev);
    }

    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
    let lambda_blocks = build_lambda_blocks(theta, structures);

    let ztwz = if let Some(cached_data) = ztwz_cache {
        mat_from_flat_array(cached_data, q)
    } else {
        compute_ztwz_sparse(&z, &w)
    };

    let blocked_v = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, structures, true);
    let chol_v = match BlockedCholesky::factor(&blocked_v) {
        Ok(c) => c,
        Err(_) => return Ok(1e10),
    };

    let logdet_v = chol_v.logdet();

    let ztwy = compute_ztwy_sparse(&z, &w, &y_adj, q);
    let cu = apply_lambda_transpose_vector(&ztwy, &lambda_blocks, structures);
    let cu_star = chol_v.solve_lower(&cu);

    let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);

    let ztwx = compute_ztwx_sparse(&z, &w, &x, q, p);
    let lambdat_ztwx = apply_lambda_transpose_vector(&ztwx, &lambda_blocks, structures);
    let rzx = chol_v.solve_lower(&lambdat_ztwx);

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
    let random_reduction: f64 = (0..q)
        .map(|i| lambda_t_zt_resid[(i, 0)] * u_star[(i, 0)])
        .sum();
    let pwrss = w_resid_sq - random_reduction;

    let denom = if reml { n - p } else { n } as f64;
    let sigma2 = pwrss / denom;

    let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v - logdet_w;
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
    let (w, logdet_w) = validate_prior_weights(weights, n)?;
    let sqrt_w: Vec<f64> = w.iter().map(|wi| wi.sqrt()).collect();

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

        let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) - logdet_w;
        if reml {
            dev += logdet_xtwx;
        }

        return Ok((dev, vec![0.0; n_theta]));
    }

    let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
    let lambda_blocks = build_lambda_blocks(theta, structures);
    let dlambda_blocks = build_lambda_derivative_blocks(structures);

    let ztwz = if let Some(cached_data) = ztwz_cache {
        mat_from_flat_array(cached_data, q)
    } else {
        compute_ztwz_sparse(&z, &w)
    };

    let blocked_v = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, structures, true);
    let chol_v = match BlockedCholesky::factor(&blocked_v) {
        Ok(c) => c,
        Err(_) => return Ok((1e10, vec![0.0; n_theta])),
    };

    let logdet_v = chol_v.logdet();

    let ztwy = compute_ztwy_sparse(&z, &w, &y_adj, q);
    let cu = apply_lambda_transpose_vector(&ztwy, &lambda_blocks, structures);
    let cu_star = chol_v.solve_lower(&cu);

    let wx = Mat::from_fn(n, p, |i, j| sqrt_w[i] * x[(i, j)]);

    let ztwx = compute_ztwx_sparse(&z, &w, &x, q, p);
    let lambdat_ztwx = apply_lambda_transpose_vector(&ztwx, &lambda_blocks, structures);
    let rzx = chol_v.solve_lower(&lambdat_ztwx);

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
    let random_reduction: f64 = (0..q)
        .map(|i| lambda_t_zt_resid[(i, 0)] * u_star[(i, 0)])
        .sum();
    let pwrss = w_resid_sq - random_reduction;

    let denom = if reml { n - p } else { n } as f64;
    let sigma2 = pwrss / denom;

    let mut dev = denom * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + logdet_v - logdet_w;
    if reml {
        dev += logdet_xtvinvx;
    }

    let v_inv = chol_v.solve(&Mat::<f64>::identity(q, q));
    let v_inv_b = chol_v.solve(&lambdat_ztwx);
    let xtvinvx_inv = if reml {
        Some(chol_xtvinvx.solve(&Mat::<f64>::identity(p, p)))
    } else {
        None
    };

    let mut gradient = Vec::with_capacity(n_theta);

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

            let dc = apply_dlambda_transpose_vector(&zt_w_resid, dlambda, block_idx, structures);
            let dv_u = &dv * &u_star;
            let mut d_pwrss = 0.0;
            for i in 0..q {
                d_pwrss += u_star[(i, 0)] * dv_u[(i, 0)];
                d_pwrss -= 2.0 * dc[(i, 0)] * u_star[(i, 0)];
            }

            let mut grad_k = d_logdet_v + denom / pwrss * d_pwrss;

            if let Some(xtvinvx_inv) = &xtvinvx_inv {
                let db = apply_dlambda_transpose_vector(&ztwx, dlambda, block_idx, structures);
                let dv_v_inv_b = &dv * &v_inv_b;
                let mut d_logdet_xtvinvx = 0.0;

                for i in 0..p {
                    for j in 0..p {
                        let mut dm_ji = 0.0;
                        for r in 0..q {
                            dm_ji -= db[(r, j)] * v_inv_b[(r, i)];
                            dm_ji -= v_inv_b[(r, j)] * db[(r, i)];
                            dm_ji += v_inv_b[(r, j)] * dv_v_inv_b[(r, i)];
                        }
                        d_logdet_xtvinvx += xtvinvx_inv[(i, j)] * dm_ji;
                    }
                }

                grad_k += d_logdet_xtvinvx;
            }

            gradient.push(grad_k);
        }
    }

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
    let (w, _) = validate_prior_weights(weights.as_array(), z_shape.0)?;
    let q = z_shape.1;

    let ztwz = compute_ztwz_sparse(&z, &w);

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

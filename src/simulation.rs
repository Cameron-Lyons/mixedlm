use faer::Mat;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::lmm::RandomEffectStructure;

fn simulate_re_single(
    theta: &[f64],
    sigma: f64,
    structures: &[RandomEffectStructure],
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut total_dim = 0;
    for s in structures {
        total_dim += s.n_levels * s.n_terms;
    }

    if total_dim == 0 {
        return vec![];
    }

    let mut u = vec![0.0; total_dim];
    let mut theta_idx = 0;
    let mut u_idx = 0;

    for structure in structures {
        let q = structure.n_terms;
        let n_levels = structure.n_levels;

        let l_block: Mat<f64> = if structure.correlated {
            let n_theta = q * (q + 1) / 2;
            let theta_block = &theta[theta_idx..theta_idx + n_theta];
            theta_idx += n_theta;

            let mut l = Mat::zeros(q, q);
            let mut idx = 0;
            for i in 0..q {
                for j in 0..=i {
                    l[(i, j)] = theta_block[idx] * sigma;
                    idx += 1;
                }
            }
            l
        } else {
            let theta_block = &theta[theta_idx..theta_idx + q];
            theta_idx += q;

            let mut l = Mat::zeros(q, q);
            for i in 0..q {
                l[(i, i)] = theta_block[i] * sigma;
            }
            l
        };

        for _g in 0..n_levels {
            let z: Vec<f64> = (0..q).map(|_| rng.sample(StandardNormal)).collect();

            for i in 0..q {
                let mut sum = 0.0;
                for j in 0..=i {
                    sum += l_block[(i, j)] * z[j];
                }
                u[u_idx + i] = sum;
            }
            u_idx += q;
        }
    }

    u
}

pub fn simulate_re_batch_impl(
    theta: &[f64],
    sigma: f64,
    structures: &[RandomEffectStructure],
    n_sim: usize,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    if n_sim == 0 {
        return vec![];
    }

    let base_seed = seed.unwrap_or_else(|| rand::rng().random());

    (0..n_sim)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            simulate_re_single(theta, sigma, structures, &mut rng)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    theta,
    sigma,
    n_levels,
    n_terms,
    correlated,
    n_sim,
    seed = None
))]
pub fn simulate_re_batch(
    py: Python<'_>,
    theta: PyReadonlyArray1<'_, f64>,
    sigma: f64,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    n_sim: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
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

    let results = simulate_re_batch_impl(
        theta.as_slice()?,
        sigma,
        &structures,
        n_sim,
        seed,
    );

    if results.is_empty() {
        return Ok(PyArray2::from_vec2(py, &vec![])?.into());
    }

    Ok(PyArray2::from_vec2(py, &results)?.into())
}

#[pyfunction]
#[pyo3(signature = (
    u,
    z_data,
    z_indices,
    z_indptr,
    z_shape,
    n_obs
))]
pub fn compute_zu(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    z_data: PyReadonlyArray1<'_, f64>,
    z_indices: PyReadonlyArray1<'_, i64>,
    z_indptr: PyReadonlyArray1<'_, i64>,
    z_shape: (usize, usize),
    n_obs: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let u_slice = u.as_slice()?;
    let z_data_slice = z_data.as_slice()?;
    let z_indices_slice = z_indices.as_slice()?;
    let z_indptr_slice = z_indptr.as_slice()?;
    let (_nrows, ncols) = z_shape;

    let mut result = vec![0.0; n_obs];

    for j in 0..ncols {
        let col_start = z_indptr_slice[j] as usize;
        let col_end = z_indptr_slice[j + 1] as usize;

        for idx in col_start..col_end {
            let i = z_indices_slice[idx] as usize;
            result[i] += z_data_slice[idx] * u_slice[j];
        }
    }

    Ok(PyArray1::from_vec(py, result).into())
}

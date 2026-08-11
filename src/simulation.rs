use faer::Mat;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

struct SimulationBlock {
    n_levels: usize,
    factor: Mat<f64>,
}

fn checked_simulation_sizes(
    n_levels: &[usize],
    n_terms: &[usize],
    correlated: &[bool],
) -> PyResult<(usize, usize)> {
    if n_levels.len() != n_terms.len() || n_levels.len() != correlated.len() {
        return Err(PyValueError::new_err(format!(
            "n_levels, n_terms, and correlated must have the same length, got {}, {}, and {}",
            n_levels.len(),
            n_terms.len(),
            correlated.len()
        )));
    }

    let mut theta_len = 0usize;
    let mut total_dim = 0usize;
    for (index, ((&levels, &terms), &is_correlated)) in
        n_levels.iter().zip(n_terms).zip(correlated).enumerate()
    {
        if levels == 0 {
            return Err(PyValueError::new_err(format!(
                "n_levels[{index}] must be positive"
            )));
        }
        if terms == 0 {
            return Err(PyValueError::new_err(format!(
                "n_terms[{index}] must be positive"
            )));
        }

        let block_theta_len = if is_correlated {
            terms
                .checked_add(1)
                .and_then(|next| terms.checked_mul(next))
                .map(|product| product / 2)
        } else {
            Some(terms)
        }
        .ok_or_else(|| PyValueError::new_err("random-effect dimensions are too large"))?;

        theta_len = theta_len
            .checked_add(block_theta_len)
            .ok_or_else(|| PyValueError::new_err("theta dimension is too large"))?;
        total_dim = total_dim
            .checked_add(
                levels
                    .checked_mul(terms)
                    .ok_or_else(|| PyValueError::new_err("simulation dimension is too large"))?,
            )
            .ok_or_else(|| PyValueError::new_err("simulation dimension is too large"))?;
    }

    Ok((theta_len, total_dim))
}

fn build_simulation_blocks(
    theta: &[f64],
    sigma: f64,
    n_levels: &[usize],
    n_terms: &[usize],
    correlated: &[bool],
) -> Vec<SimulationBlock> {
    let mut blocks = Vec::with_capacity(n_levels.len());
    let mut theta_idx = 0;

    for ((&levels, &q), &is_correlated) in n_levels.iter().zip(n_terms).zip(correlated) {
        let mut factor = Mat::zeros(q, q);
        if is_correlated {
            for i in 0..q {
                for j in 0..=i {
                    factor[(i, j)] = theta[theta_idx] * sigma;
                    theta_idx += 1;
                }
            }
        } else {
            for i in 0..q {
                factor[(i, i)] = theta[theta_idx] * sigma;
                theta_idx += 1;
            }
        }
        blocks.push(SimulationBlock {
            n_levels: levels,
            factor,
        });
    }

    blocks
}

fn simulate_re_single(blocks: &[SimulationBlock], rng: &mut impl Rng, u: &mut [f64]) {
    let mut u_idx = 0;

    for block in blocks {
        let q = block.factor.nrows();
        let mut z = vec![0.0; q];
        for _ in 0..block.n_levels {
            for value in &mut z {
                *value = rng.sample(StandardNormal);
            }

            for i in 0..q {
                let mut sum = 0.0;
                for (j, value) in z.iter().take(i + 1).enumerate() {
                    sum += block.factor[(i, j)] * value;
                }
                u[u_idx + i] = sum;
            }
            u_idx += q;
        }
    }
}

fn simulate_re_batch_impl(
    blocks: &[SimulationBlock],
    total_dim: usize,
    n_sim: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut results = vec![0.0; n_sim * total_dim];
    if results.is_empty() {
        return vec![];
    }

    let base_seed = seed.unwrap_or_else(|| rand::rng().random());

    #[cfg(miri)]
    let iter = results.chunks_mut(total_dim).enumerate();
    #[cfg(not(miri))]
    let iter = results.par_chunks_mut(total_dim).enumerate();
    iter.for_each(|(i, result)| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
        simulate_re_single(blocks, &mut rng, result);
    });
    results
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
#[allow(clippy::too_many_arguments)]
pub fn simulate_re_batch<'py>(
    py: Python<'py>,
    theta: PyArrayLike1<'py, f64>,
    sigma: f64,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    n_sim: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(PyValueError::new_err(
            "sigma must be finite and non-negative",
        ));
    }

    let (expected_theta_len, total_dim) =
        checked_simulation_sizes(&n_levels, &n_terms, &correlated)?;
    let theta = theta.as_slice()?;
    if theta.len() != expected_theta_len {
        return Err(PyValueError::new_err(format!(
            "theta must contain exactly {expected_theta_len} values, got {}",
            theta.len()
        )));
    }
    if let Some((index, _)) = theta
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "theta[{index}] must be finite"
        )));
    }

    n_sim
        .checked_mul(total_dim)
        .ok_or_else(|| PyValueError::new_err("simulation output is too large"))?;

    let blocks = build_simulation_blocks(theta, sigma, &n_levels, &n_terms, &correlated);
    let results = simulate_re_batch_impl(&blocks, total_dim, n_sim, seed);
    let array = Array2::from_shape_vec((n_sim, total_dim), results)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    Ok(PyArray2::from_owned_array(py, array).into())
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
pub fn compute_zu<'py>(
    py: Python<'py>,
    u: PyArrayLike1<'py, f64>,
    z_data: PyArrayLike1<'py, f64>,
    z_indices: PyArrayLike1<'py, i64>,
    z_indptr: PyArrayLike1<'py, i64>,
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

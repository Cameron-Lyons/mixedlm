use faer::Mat;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

struct SimulationBlock {
    n_levels: usize,
    factor: Mat<f64>,
}

#[derive(Clone, Copy)]
enum CovarianceType {
    Unstructured,
    CompoundSymmetry,
    Autoregressive1,
}

const ZIG_NORM_R: f64 = 3.654_152_885_361_009;
const ZIG_NORM_X0: f64 = 3.910_757_959_537_09;

struct NormalZigguratTables {
    x: [f64; 257],
    density: [f64; 257],
}

#[inline(always)]
fn normal_density(value: f64) -> f64 {
    (-0.5 * value * value).exp()
}

fn normal_ziggurat_tables() -> &'static NormalZigguratTables {
    static TABLES: OnceLock<NormalZigguratTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut x = [0.0; 257];
        let mut density = [0.0; 257];
        x[0] = ZIG_NORM_X0;
        x[1] = ZIG_NORM_R;
        let rectangle_area = x[0] * normal_density(x[1]);
        for index in 2..256 {
            let previous = x[index - 1];
            x[index] = (-2.0 * (rectangle_area / previous + normal_density(previous)).ln()).sqrt();
        }
        for (value, output) in x.iter().zip(&mut density) {
            *output = normal_density(*value);
        }
        NormalZigguratTables { x, density }
    })
}

struct StandardNormalSampler {
    tables: &'static NormalZigguratTables,
}

impl Default for StandardNormalSampler {
    fn default() -> Self {
        Self {
            tables: normal_ziggurat_tables(),
        }
    }
}

impl StandardNormalSampler {
    #[inline(always)]
    fn sample(&mut self, rng: &mut impl Rng) -> f64 {
        // Doornik's ZIGNOR variant uses eight index bits and 52 fraction bits
        // from one RNG word. Most samples return after the first comparison.
        loop {
            let bits = rng.next_u64();
            let index = bits as usize & 0xff;
            let uniform = f64::from_bits((bits >> 12) | 0x4000_0000_0000_0000) - 3.0;
            let value = uniform * self.tables.x[index];
            if value.abs() < self.tables.x[index + 1] {
                return value;
            }
            if index == 0 {
                loop {
                    let tail = open_unit_interval(rng).ln() / ZIG_NORM_R;
                    let height = open_unit_interval(rng).ln();
                    if -2.0 * height >= tail * tail {
                        return if uniform < 0.0 {
                            tail - ZIG_NORM_R
                        } else {
                            ZIG_NORM_R - tail
                        };
                    }
                }
            }
            let lower_density = self.tables.density[index + 1];
            if lower_density + (self.tables.density[index] - lower_density) * rng.random::<f64>()
                < normal_density(value)
            {
                return value;
            }
        }
    }
}

#[inline(always)]
fn open_unit_interval(rng: &mut impl Rng) -> f64 {
    const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
    ((rng.next_u64() >> 11) as f64 + 0.5) * SCALE
}

fn checked_simulation_sizes(
    n_levels: &[usize],
    n_terms: &[usize],
    correlated: &[bool],
    covariance_types: &[CovarianceType],
) -> PyResult<(usize, usize)> {
    if n_levels.len() != n_terms.len()
        || n_levels.len() != correlated.len()
        || n_levels.len() != covariance_types.len()
    {
        return Err(PyValueError::new_err(format!(
            "n_levels, n_terms, correlated, and cov_types must have the same length, got {}, {}, {}, and {}",
            n_levels.len(),
            n_terms.len(),
            correlated.len(),
            covariance_types.len()
        )));
    }

    let mut theta_len = 0usize;
    let mut total_dim = 0usize;
    for (index, (((&levels, &terms), &is_correlated), &covariance_type)) in n_levels
        .iter()
        .zip(n_terms)
        .zip(correlated)
        .zip(covariance_types)
        .enumerate()
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

        let block_theta_len = match covariance_type {
            CovarianceType::CompoundSymmetry | CovarianceType::Autoregressive1 => {
                Some(if terms > 1 { 2 } else { 1 })
            }
            CovarianceType::Unstructured if is_correlated => terms
                .checked_add(1)
                .and_then(|next| terms.checked_mul(next))
                .map(|product| product / 2),
            CovarianceType::Unstructured => Some(terms),
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

fn parse_covariance_types(
    cov_types: Option<Vec<String>>,
    n_structures: usize,
) -> PyResult<Vec<CovarianceType>> {
    let Some(cov_types) = cov_types else {
        return Ok(vec![CovarianceType::Unstructured; n_structures]);
    };

    cov_types
        .into_iter()
        .enumerate()
        .map(|(index, covariance_type)| match covariance_type.as_str() {
            "us" => Ok(CovarianceType::Unstructured),
            "cs" => Ok(CovarianceType::CompoundSymmetry),
            "ar1" => Ok(CovarianceType::Autoregressive1),
            _ => Err(PyValueError::new_err(format!(
                "cov_types[{index}] must be one of 'us', 'cs', or 'ar1', got '{covariance_type}'"
            ))),
        })
        .collect()
}

fn structured_correlation_factor(
    n_terms: usize,
    rho: f64,
    covariance_type: CovarianceType,
) -> PyResult<Mat<f64>> {
    if n_terms == 1 {
        return Ok(Mat::identity(1, 1));
    }

    let lower_bound = match covariance_type {
        CovarianceType::CompoundSymmetry => -1.0 / (n_terms - 1) as f64,
        CovarianceType::Autoregressive1 => -1.0,
        CovarianceType::Unstructured => unreachable!(),
    };
    let rho = rho.clamp(lower_bound + 1e-6, 1.0 - 1e-6);
    let correlation = Mat::from_fn(n_terms, n_terms, |row, column| match covariance_type {
        CovarianceType::CompoundSymmetry => {
            if row == column {
                1.0
            } else {
                rho
            }
        }
        CovarianceType::Autoregressive1 => {
            rho.powi(row.abs_diff(column).try_into().unwrap_or(i32::MAX))
        }
        CovarianceType::Unstructured => unreachable!(),
    });

    let mut factor = Mat::zeros(n_terms, n_terms);
    for row in 0..n_terms {
        for column in 0..=row {
            let mut value = correlation[(row, column)];
            for index in 0..column {
                value -= factor[(row, index)] * factor[(column, index)];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(PyValueError::new_err(
                        "structured covariance matrix is not positive definite",
                    ));
                }
                factor[(row, column)] = value.sqrt();
            } else {
                factor[(row, column)] = value / factor[(column, column)];
            }
        }
    }
    Ok(factor)
}

fn build_simulation_blocks(
    theta: &[f64],
    sigma: f64,
    n_levels: &[usize],
    n_terms: &[usize],
    correlated: &[bool],
    covariance_types: &[CovarianceType],
) -> PyResult<Vec<SimulationBlock>> {
    let mut blocks = Vec::with_capacity(n_levels.len());
    let mut theta_idx = 0;

    for (((&levels, &q), &is_correlated), &covariance_type) in n_levels
        .iter()
        .zip(n_terms)
        .zip(correlated)
        .zip(covariance_types)
    {
        let factor = match covariance_type {
            CovarianceType::CompoundSymmetry | CovarianceType::Autoregressive1 => {
                let relative_scale = theta[theta_idx];
                theta_idx += 1;
                let rho = if q > 1 {
                    let value = theta[theta_idx];
                    theta_idx += 1;
                    value
                } else {
                    0.0
                };
                let mut factor = structured_correlation_factor(q, rho, covariance_type)?;
                for row in 0..q {
                    for column in 0..=row {
                        factor[(row, column)] *= relative_scale * sigma;
                    }
                }
                factor
            }
            CovarianceType::Unstructured => {
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
                factor
            }
        };
        blocks.push(SimulationBlock {
            n_levels: levels,
            factor,
        });
    }

    Ok(blocks)
}

fn simulate_re_single(blocks: &[SimulationBlock], rng: &mut impl Rng, u: &mut [f64]) {
    let mut u_idx = 0;
    let mut standard_normal = StandardNormalSampler::default();

    for block in blocks {
        let q = block.factor.nrows();
        let mut z = vec![0.0; q];
        for _ in 0..block.n_levels {
            for value in &mut z {
                *value = standard_normal.sample(rng);
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

#[cfg(test)]
mod tests {
    use super::StandardNormalSampler;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn standard_normal_sampler_is_reproducible_and_well_calibrated() {
        const SAMPLE_COUNT: usize = if cfg!(miri) { 10_000 } else { 200_000 };
        const MEAN_TOLERANCE: f64 = if cfg!(miri) { 0.05 } else { 0.01 };
        const VARIANCE_TOLERANCE: f64 = if cfg!(miri) { 0.05 } else { 0.02 };

        let mut first_rng = StdRng::seed_from_u64(42);
        let mut second_rng = StdRng::seed_from_u64(42);
        let mut first = StandardNormalSampler::default();
        let mut second = StandardNormalSampler::default();
        let mut sum = 0.0;
        let mut sum_squares = 0.0;

        for _ in 0..SAMPLE_COUNT {
            let value = first.sample(&mut first_rng);
            let repeated = second.sample(&mut second_rng);
            let reproducibility_tolerance = 1e-14 * value.abs().max(repeated.abs()).max(1.0);
            assert!(
                (value - repeated).abs() <= reproducibility_tolerance,
                "seeded samples differed: {value} versus {repeated}"
            );
            assert!(value.is_finite());
            sum += value;
            sum_squares += value * value;
        }

        let mean = sum / SAMPLE_COUNT as f64;
        let variance = sum_squares / SAMPLE_COUNT as f64 - mean * mean;
        assert!(mean.abs() < MEAN_TOLERANCE, "sample mean was {mean}");
        assert!(
            (variance - 1.0).abs() < VARIANCE_TOLERANCE,
            "sample variance was {variance}"
        );
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
    seed = None,
    cov_types = None
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
    cov_types: Option<Vec<String>>,
) -> PyResult<Py<PyArray2<f64>>> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(PyValueError::new_err(
            "sigma must be finite and non-negative",
        ));
    }

    let covariance_types = parse_covariance_types(cov_types, n_levels.len())?;
    let (expected_theta_len, total_dim) =
        checked_simulation_sizes(&n_levels, &n_terms, &correlated, &covariance_types)?;
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

    let blocks = build_simulation_blocks(
        theta,
        sigma,
        &n_levels,
        &n_terms,
        &correlated,
        &covariance_types,
    )?;
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

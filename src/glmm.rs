use faer::linalg::solvers::{Llt, Solve, SolveLstsq};
use faer::{Col as DVector, Mat as DMatrix, Side};
use numpy::ndarray::{ArrayView1, ArrayView2};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(not(miri))]
use rayon::prelude::*;

use crate::covariance::CovarianceFactor;
pub use crate::covariance::RandomEffectStructure;
use crate::csc::CscMatrix;
use crate::glmm_sparse::{RandomFactor, WeightedRandomDesign};
use crate::linalg::LinalgError;
use crate::quadrature::gauss_hermite_nodes_weights;

const PIRLS_MAX_ITER: usize = 100;
const PIRLS_TOLERANCE: f64 = 1e-6;

fn validate_pirls_controls(maxiter: usize, tol: f64) -> PyResult<()> {
    if maxiter == 0 {
        return Err(PyValueError::new_err("maxiter must be a positive integer"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be positive and finite"));
    }
    Ok(())
}

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
    fn link(&self, mu: f64) -> f64 {
        match self {
            LinkFunction::Identity => mu,
            LinkFunction::Log => mu.ln(),
            LinkFunction::Logit => (mu / (1.0 - mu)).ln(),
        }
    }

    fn inverse(&self, eta: &DVector<f64>) -> DVector<f64> {
        match self {
            LinkFunction::Identity => eta.clone(),
            LinkFunction::Log => DVector::from_fn(eta.nrows(), |i| eta[i].exp()),
            LinkFunction::Logit => DVector::from_fn(eta.nrows(), |i| 1.0 / (1.0 + (-eta[i]).exp())),
        }
    }

    fn deriv_at(&self, mu: f64) -> f64 {
        match self {
            LinkFunction::Identity => 1.0,
            LinkFunction::Log => 1.0 / mu.max(1e-10),
            LinkFunction::Logit => {
                let m = mu.clamp(1e-10, 1.0 - 1e-10);
                1.0 / (m * (1.0 - m))
            }
        }
    }
}

impl FamilyType {
    fn starting_mean(&self, y: f64, link: LinkFunction) -> f64 {
        // Use the intersection of the family and link mean domains, matching Python.
        if *self == FamilyType::Binomial || link == LinkFunction::Logit {
            ((y + 0.5) / 2.0).clamp(1e-7, 1.0 - 1e-7)
        } else if *self == FamilyType::Poisson || link == LinkFunction::Log {
            y.max(0.1)
        } else {
            y
        }
    }

    fn clamp_mu(&self, mu: &mut DVector<f64>, eps: f64) {
        match self {
            FamilyType::Gaussian => {}
            FamilyType::Binomial => {
                for value in mu.iter_mut() {
                    *value = value.clamp(eps, 1.0 - eps);
                }
            }
            FamilyType::Poisson => {
                for value in mu.iter_mut() {
                    *value = value.max(eps);
                }
            }
        }
    }

    fn variance_at(&self, mu: f64) -> f64 {
        match self {
            FamilyType::Gaussian => 1.0,
            FamilyType::Binomial => {
                let m = mu.clamp(1e-10, 1.0 - 1e-10);
                m * (1.0 - m)
            }
            FamilyType::Poisson => mu.max(1e-10),
        }
    }

    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            FamilyType::Gaussian => (y - mu).powi(2),
            FamilyType::Binomial => {
                let bounded_mu = mu.clamp(1e-10, 1.0 - 1e-10);
                let success = if y > 1e-10 {
                    y * (y / bounded_mu).ln()
                } else {
                    0.0
                };
                let failure = if (1.0 - y) > 1e-10 {
                    (1.0 - y) * ((1.0 - y) / (1.0 - bounded_mu)).ln()
                } else {
                    0.0
                };
                2.0 * (success + failure)
            }
            FamilyType::Poisson => {
                let bounded_mu = mu.max(1e-10);
                let log_term = if y > 1e-10 {
                    y * (y / bounded_mu).ln()
                } else {
                    0.0
                };
                2.0 * (log_term - (y - bounded_mu))
            }
        }
    }

    fn deviance_resids(&self, y: &DVector<f64>, mu: &DVector<f64>, wt: &[f64]) -> f64 {
        (0..y.nrows())
            .map(|i| wt[i] * self.unit_deviance(y[i], mu[i]))
            .sum()
    }

    fn deviance_resids_rows(
        &self,
        y: &DVector<f64>,
        mu: &DVector<f64>,
        wt: &[f64],
        rows: &[usize],
    ) -> f64 {
        rows.iter()
            .map(|&i| wt[i] * self.unit_deviance(y[i], mu[i]))
            .sum()
    }

    fn weights(&self, mu: &DVector<f64>, link: LinkFunction) -> DVector<f64> {
        DVector::from_fn(mu.nrows(), |i| {
            self.weight_from_derivative(mu[i], link.deriv_at(mu[i]))
        })
    }

    fn weight_from_derivative(&self, mu: f64, derivative: f64) -> f64 {
        let variance = self.variance_at(mu).max(1e-10);
        1.0 / (derivative * derivative * variance).max(1e-10)
    }
}

fn parse_family_and_link(family: &str, link: &str) -> PyResult<(FamilyType, LinkFunction)> {
    let family_type = match family {
        "gaussian" => FamilyType::Gaussian,
        "binomial" => FamilyType::Binomial,
        "poisson" => FamilyType::Poisson,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported GLMM family: {family}"
            )));
        }
    };

    let link_function = match link {
        "identity" => LinkFunction::Identity,
        "log" => LinkFunction::Log,
        "logit" => LinkFunction::Logit,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported GLMM link: {link}"
            )));
        }
    };

    Ok((family_type, link_function))
}

fn csc_from_scipy(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    shape: (usize, usize),
) -> Result<CscMatrix, LinalgError> {
    CscMatrix::try_from_i64(data, indices, indptr, shape)
}

/// Validate and prepare the common inputs for native GLMM entry points.
struct GlmmInputs<'a> {
    y: DVector<f64>,
    x: DMatrix<f64>,
    z: CscMatrix,
    weights: &'a [f64],
    offset: DVector<f64>,
    n_theta: usize,
    structures: Vec<RandomEffectStructure>,
    family: FamilyType,
    link: LinkFunction,
}

fn validate_glmm_dimensions(
    n: usize,
    x_rows: usize,
    z_shape: (usize, usize),
    weights_len: usize,
    offset_len: usize,
    theta_len: Option<usize>,
    structures: &[RandomEffectStructure],
) -> Result<usize, String> {
    if n == 0 {
        return Err("y must contain at least one observation".into());
    }
    if x_rows != n {
        return Err(format!("x must have {n} rows to match y, got {x_rows}"));
    }
    if z_shape.0 != n {
        return Err(format!(
            "z must have {n} rows to match y, got {}",
            z_shape.0
        ));
    }
    if weights_len != n {
        return Err(format!("weights must have length {n}, got {weights_len}"));
    }
    if offset_len != n {
        return Err(format!("offset must have length {n}, got {offset_len}"));
    }
    let mut columns = 0usize;
    let mut parameters = 0usize;
    for structure in structures {
        let terms = structure.n_terms;
        if terms == 0 || structure.n_levels == 0 {
            return Err("random-effect structures must have positive dimensions".into());
        }
        columns = structure
            .n_levels
            .checked_mul(terms)
            .and_then(|size| columns.checked_add(size))
            .ok_or("random-effect dimensions overflow")?;
        let count = if structure.correlated {
            terms
                .checked_add(1)
                .and_then(|next| terms.checked_mul(next))
                .map(|product| product / 2)
                .ok_or("random-effect parameter count overflows")?
        } else {
            terms
        };
        parameters = parameters
            .checked_add(count)
            .ok_or("random-effect parameter count overflows")?;
    }
    if columns != z_shape.1 {
        return Err(format!(
            "random-effect structures describe {columns} columns, but z has {}",
            z_shape.1
        ));
    }
    if let Some(length) = theta_len {
        validate_theta_length(length, parameters)?;
    }
    Ok(parameters)
}

impl<'a> GlmmInputs<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        y: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        z_data: &[f64],
        z_indices: &[i64],
        z_indptr: &[i64],
        z_shape: (usize, usize),
        weights: &'a [f64],
        offset: ArrayView1<'_, f64>,
        theta_len: Option<usize>,
        n_levels: Vec<usize>,
        n_terms: Vec<usize>,
        correlated: Vec<bool>,
        family: &str,
        link: &str,
        n_agq: usize,
    ) -> PyResult<Self> {
        if n_agq == 0 {
            return Err(PyValueError::new_err("n_agq must be a positive integer"));
        }
        if n_levels.len() != n_terms.len() || n_levels.len() != correlated.len() {
            return Err(PyValueError::new_err(
                "n_levels, n_terms, and correlated must have equal lengths",
            ));
        }
        let structures: Vec<_> = n_levels
            .into_iter()
            .zip(n_terms)
            .zip(correlated)
            .map(|((n_levels, n_terms), correlated)| RandomEffectStructure {
                n_levels,
                n_terms,
                correlated,
            })
            .collect();
        let n_theta = validate_glmm_dimensions(
            y.len(),
            x.nrows(),
            z_shape,
            weights.len(),
            offset.len(),
            theta_len,
            &structures,
        )
        .map_err(PyValueError::new_err)?;
        validate_agq_structure(n_agq, z_shape.1, &structures)?;
        let (family, link) = parse_family_and_link(family, link)?;
        // Validate before indexing views or allocating the owned dense matrices.
        let z = csc_from_scipy(z_data, z_indices, z_indptr, z_shape)?;
        Ok(Self {
            y: y.iter().copied().collect(),
            x: DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| x[[i, j]]),
            z,
            weights,
            offset: offset.iter().copied().collect(),
            n_theta,
            structures,
            family,
            link,
        })
    }
}

fn validate_theta_length(length: usize, expected: usize) -> Result<(), String> {
    if length != expected {
        return Err(format!("theta must have length {expected}, got {length}"));
    }
    Ok(())
}

fn validate_agq_structure(
    n_agq: usize,
    q: usize,
    structures: &[RandomEffectStructure],
) -> PyResult<()> {
    if n_agq == 0 {
        return Err(PyValueError::new_err("n_agq must be a positive integer"));
    }
    if n_agq > 1 && q > 0 && (structures.len() != 1 || structures[0].n_terms != 1) {
        return Err(PyValueError::new_err(
            "n_agq > 1 requires one random-effect term with one coefficient per group; use n_agq=1 for this model",
        ));
    }
    Ok(())
}

/// An owned, immutable problem reused across likelihood evaluations.
/// Every solve starts from the same coefficients, independent of call order.
#[pyclass(frozen)]
pub struct GlmmProblem {
    y: DVector<f64>,
    x: DMatrix<f64>,
    z: CscMatrix,
    weights: Vec<f64>,
    offset: DVector<f64>,
    structures: Vec<RandomEffectStructure>,
    n_theta: usize,
    family: FamilyType,
    link: LinkFunction,
    beta_start: DVector<f64>,
}

#[pymethods]
impl GlmmProblem {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
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
        family: &str,
        link: &str,
    ) -> PyResult<Self> {
        let inputs = GlmmInputs::new(
            y.as_array(),
            x.as_array(),
            z_data.as_slice()?,
            z_indices.as_slice()?,
            z_indptr.as_slice()?,
            z_shape,
            weights.as_slice()?,
            offset.as_array(),
            None,
            n_levels,
            n_terms,
            correlated,
            family,
            link,
            1,
        )?;
        let beta_start = initial_beta(
            &inputs.y,
            &inputs.x,
            inputs.weights,
            &inputs.offset,
            inputs.family,
            inputs.link,
        );
        Ok(Self {
            y: inputs.y,
            x: inputs.x,
            z: inputs.z,
            weights: inputs.weights.to_vec(),
            offset: inputs.offset,
            structures: inputs.structures,
            n_theta: inputs.n_theta,
            family: inputs.family,
            link: inputs.link,
            beta_start,
        })
    }

    #[pyo3(signature = (theta, n_agq=1, *, offset=None, maxiter=PIRLS_MAX_ITER, tol=PIRLS_TOLERANCE))]
    fn evaluate<'py>(
        &self,
        theta: numpy::PyArrayLike1<'py, f64>,
        n_agq: usize,
        offset: Option<numpy::PyArrayLike1<'py, f64>>,
        maxiter: usize,
        tol: f64,
    ) -> PyResult<(f64, Vec<f64>, Vec<f64>, bool)> {
        validate_pirls_controls(maxiter, tol)?;
        validate_agq_structure(n_agq, self.z.ncols(), &self.structures)?;
        let theta = theta.as_slice()?;
        validate_theta_length(theta.len(), self.n_theta).map_err(PyValueError::new_err)?;
        let override_offset = if let Some(offset) = offset {
            let offset = offset.as_array();
            if offset.len() != self.y.nrows() {
                return Err(PyValueError::new_err(format!(
                    "offset must have length {}, got {}",
                    self.y.nrows(),
                    offset.len(),
                )));
            }
            Some(offset.iter().copied().collect::<DVector<f64>>())
        } else {
            None
        };
        let offset = override_offset.as_ref().unwrap_or(&self.offset);
        // A changed offset needs its own initial coefficients. Mode-only solves
        // have no fixed coefficients and can always use the empty cached start.
        let beta_start = if override_offset.is_none() || self.x.ncols() == 0 {
            Some(&self.beta_start)
        } else {
            None
        };
        let (deviance, beta, u, converged) = adaptive_gh_deviance_impl(
            &self.y,
            &self.x,
            &self.z,
            &self.weights,
            offset,
            theta,
            &self.structures,
            self.family,
            self.link,
            n_agq,
            beta_start,
            None,
            maxiter,
            tol,
        )?;
        Ok((
            deviance,
            beta.iter().copied().collect(),
            u.iter().copied().collect(),
            converged,
        ))
    }
}

fn dense_penalized_crossproduct(
    z: &CscMatrix,
    lambda: &CovarianceFactor,
    w_vec: &DVector<f64>,
) -> DMatrix<f64> {
    let ztwz = z.weighted_crossproduct(
        w_vec
            .try_as_col_major()
            .expect("owned weights are contiguous")
            .as_slice(),
    );
    lambda.penalized_crossproduct(ztwz)
}

fn factor_random_system(
    z: &CscMatrix,
    lambda: &CovarianceFactor,
    prepared: Option<&WeightedRandomDesign>,
    weights: &DVector<f64>,
) -> Result<RandomFactor, LinalgError> {
    if let Some(prepared) = prepared {
        let weights = weights
            .try_as_col_major()
            .expect("owned weights are contiguous");
        return prepared
            .factor(weights.as_slice(), 0.0)
            .or_else(|_| prepared.factor(weights.as_slice(), 1e-6));
    }
    let mut matrix = dense_penalized_crossproduct(z, lambda, weights);
    match Llt::new(matrix.as_ref(), Side::Lower) {
        Ok(factor) => Ok(RandomFactor::Dense(factor)),
        Err(_) => {
            for i in 0..z.ncols() {
                matrix[(i, i)] += 1e-6;
            }
            Llt::new(matrix.as_ref(), Side::Lower)
                .map(RandomFactor::Dense)
                .map_err(|_| LinalgError::NotPositiveDefinite)
        }
    }
}

fn dense_logdet(z: &CscMatrix, lambda: &CovarianceFactor, weights: &DVector<f64>) -> f64 {
    let matrix = dense_penalized_crossproduct(z, lambda, weights);
    match Llt::new(matrix.as_ref(), Side::Lower) {
        Ok(factor) => 2.0 * (0..z.ncols()).map(|i| factor.L()[(i, i)].ln()).sum::<f64>(),
        Err(_) => matrix
            .self_adjoint_eigenvalues(Side::Lower)
            .unwrap_or_else(|_| vec![1e-10; z.ncols()])
            .iter()
            .map(|&value| value.max(1e-10).ln())
            .sum(),
    }
}

fn max_abs_diff(left: &DVector<f64>, right: &DVector<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&lhs, &rhs)| {
            if lhs.is_finite() && rhs.is_finite() {
                (lhs - rhs).abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

fn initial_beta(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    weights: &[f64],
    offset: &DVector<f64>,
    family: FamilyType,
    link: LinkFunction,
) -> DVector<f64> {
    let n = y.nrows();
    let p = x.ncols();
    if p == 0 {
        return DVector::zeros(0);
    }
    let sqrt_weights: Vec<f64> = weights
        .iter()
        .map(|weight| weight.max(1e-10).sqrt())
        .collect();
    let weighted_x = DMatrix::from_fn(n, p, |i, j| sqrt_weights[i] * x[(i, j)]);
    let weighted_eta = DVector::from_fn(n, |i| {
        sqrt_weights[i] * (link.link(family.starting_mean(y[i], link)) - offset[i])
    });
    let xtwx = weighted_x.transpose() * &weighted_x;
    let xtweta = weighted_x.transpose() * &weighted_eta;
    match Llt::new(xtwx.as_ref(), Side::Lower) {
        Ok(chol) => chol.solve(&xtweta),
        Err(_) => xtwx.partial_piv_lu().solve(&xtweta),
    }
}

fn update_fixed_linear_predictor(
    eta: &mut DVector<f64>,
    x: &DMatrix<f64>,
    beta: &DVector<f64>,
    offset: &DVector<f64>,
) {
    if x.ncols() == 0 {
        eta.copy_from(offset);
    } else {
        faer::linalg::matmul::matmul(
            eta.as_mut(),
            faer::Accum::Replace,
            x,
            beta,
            1.0,
            faer::get_global_parallelism(),
        );
        *eta += offset;
    }
}

// Keep the separate mutable slices visible at the function boundary so the
// compiler can vectorize stores without assuming they alias the model inputs.
#[inline(never)]
fn update_binomial_logit_working_values(
    working_weights: &mut [f64],
    working_response: &mut [f64],
    mean: &[f64],
    eta: &[f64],
    y: &[f64],
    offset: &[f64],
    prior_weights: &[f64],
) {
    let n = mean.len();
    assert_eq!(working_weights.len(), n);
    assert_eq!(working_response.len(), n);
    assert_eq!(eta.len(), n);
    assert_eq!(y.len(), n);
    assert_eq!(offset.len(), n);
    assert_eq!(prior_weights.len(), n);
    for i in 0..n {
        let derivative = LinkFunction::Logit.deriv_at(mean[i]);
        working_weights[i] = (FamilyType::Binomial.weight_from_derivative(mean[i], derivative)
            * prior_weights[i])
            .max(1e-10);
        working_response[i] = eta[i] - offset[i] + derivative * (y[i] - mean[i]);
    }
}

#[derive(Debug)]
pub struct PirlsResult {
    pub beta: DVector<f64>,
    pub spherical: DVector<f64>,
    pub u: DVector<f64>,
    pub deviance: f64,
    pub converged: bool,
    // Preserve the final mode state for Laplace and adaptive quadrature.
    mean: DVector<f64>,
    covariance: CovarianceFactor,
    random_system: Option<WeightedRandomDesign>,
}

#[allow(clippy::too_many_arguments)]
pub fn pirls_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
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
    let n = y.nrows();
    let p = x.ncols();
    let q = z.ncols();

    let mut beta = if let Some(b) = beta_start {
        b.clone()
    } else {
        initial_beta(y, x, weights, offset, family, link)
    };

    let lambda = CovarianceFactor::new(theta, structures);
    let random_system = WeightedRandomDesign::new(z, &lambda);
    let mut spherical = if let Some(u_init) = u_start {
        lambda.to_dense().col_piv_qr().solve_lstsq(u_init)
    } else {
        DVector::zeros(q)
    };

    let mut converged = false;
    let mut w_vec = DVector::zeros(n);
    let mut z_vec = DVector::zeros(n);
    let mut eta = DVector::zeros(n);

    for _iter in 0..maxiter {
        let random_effects = lambda.apply(&spherical);
        update_fixed_linear_predictor(&mut eta, x, &beta, offset);
        for j in 0..q {
            let col_start = z.col_offsets()[j];
            let col_end = z.col_offsets()[j + 1];
            for idx in col_start..col_end {
                let i = z.row_indices()[idx];
                eta[i] += z.values()[idx] * random_effects[j];
            }
        }

        let mut mu = link.inverse(&eta);
        family.clamp_mu(&mut mu, 1e-10);

        if family == FamilyType::Binomial && link == LinkFunction::Logit {
            update_binomial_logit_working_values(
                w_vec.try_as_col_major_mut().unwrap().as_slice_mut(),
                z_vec.try_as_col_major_mut().unwrap().as_slice_mut(),
                mu.try_as_col_major().unwrap().as_slice(),
                eta.try_as_col_major().unwrap().as_slice(),
                y.try_as_col_major().unwrap().as_slice(),
                offset.try_as_col_major().unwrap().as_slice(),
                weights,
            );
        } else {
            for i in 0..n {
                // Share each derivative without allocating full derivative and
                // variance vectors. Keep the weight arithmetic and floors intact.
                let derivative = link.deriv_at(mu[i]);
                w_vec[i] =
                    (family.weight_from_derivative(mu[i], derivative) * weights[i]).max(1e-10);
                z_vec[i] = eta[i] - offset[i] + derivative * (y[i] - mu[i]);
            }
        }

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

        let chol_c = match factor_random_system(z, &lambda, random_system.as_ref(), &w_vec) {
            Ok(factor) => factor,
            Err(_) => {
                return PirlsResult {
                    beta,
                    spherical,
                    u: random_effects,
                    deviance: 1e10,
                    converged: false,
                    mean: mu,
                    covariance: lambda,
                    random_system,
                };
            }
        };

        let spherical_ztwz = lambda.transpose_apply_vector(&ztwz_vec);
        let (beta_new, mut spherical_new) = if p == 0 {
            // Joint likelihoods put fixed coefficients into the offset. Their
            // mode solve only needs C u = Lambda' Z' W z; avoid constructing
            // and factoring an empty fixed-effect Schur complement.
            let mut rhs = spherical_ztwz;
            chol_c.solve_lower_in_place(rhs.as_mat_mut());
            (DVector::zeros(0), rhs)
        } else {
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

            let xtwz_vec: DVector<f64> = DVector::from_fn(p, |i| {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += x[(j, i)] * w_vec[j] * z_vec[j];
                }
                sum
            });
            let spherical_ztwx = lambda.transpose_apply(xtwz_mat.transpose());
            // Solve fixed-effect and response columns together, borrowing the
            // Cholesky factor instead of copying its storage each iteration.
            let mut rhs = DMatrix::from_fn(q, p + 1, |i, j| {
                if j == p {
                    spherical_ztwz[i]
                } else {
                    spherical_ztwx[(i, j)]
                }
            });
            chol_c.solve_lower_in_place(rhs.as_mut());
            let rzx = rhs.subcols(0, p);
            let cu = rhs.col(p);

            let xtvinvx = &xtwx - &(rzx.transpose() * rzx);
            let xtvinvz = &xtwz_vec - &(rzx.transpose() * cu);
            let beta_new = match Llt::new(xtvinvx.as_ref(), Side::Lower) {
                Ok(chol) => chol.solve(&xtvinvz),
                Err(_) => xtvinvx.partial_piv_lu().solve(&xtvinvz),
            };
            let spherical_new = cu - rzx * &beta_new;
            (beta_new, spherical_new)
        };
        chol_c.solve_upper_in_place(spherical_new.as_mat_mut());

        let delta_beta = max_abs_diff(&beta_new, &beta);
        let delta_u = if q > 0 {
            max_abs_diff(&spherical_new, &spherical)
        } else {
            0.0
        };

        beta = beta_new;
        spherical = spherical_new;

        if !delta_beta.is_finite() || !delta_u.is_finite() {
            break;
        }
        if delta_beta < tol && delta_u < tol {
            converged = true;
            break;
        }
    }

    let random_effects = lambda.apply(&spherical);
    update_fixed_linear_predictor(&mut eta, x, &beta, offset);
    for j in 0..q {
        let col_start = z.col_offsets()[j];
        let col_end = z.col_offsets()[j + 1];
        for idx in col_start..col_end {
            let i = z.row_indices()[idx];
            eta[i] += z.values()[idx] * random_effects[j];
        }
    }

    let mut mu_final = link.inverse(&eta);
    family.clamp_mu(&mut mu_final, 1e-10);

    let dev_resids = family.deviance_resids(y, &mu_final, weights);

    let deviance = dev_resids + spherical.squared_norm_l2();

    PirlsResult {
        beta,
        spherical,
        u: random_effects,
        deviance,
        converged: converged && deviance.is_finite(),
        mean: mu_final,
        covariance: lambda,
        random_system,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn laplace_deviance_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
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
) -> (f64, DVector<f64>, DVector<f64>, bool) {
    let n = y.nrows();
    let q = z.ncols();

    if q == 0 {
        let result = pirls_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
            maxiter, tol,
        );
        let converged = result.converged && result.deviance.is_finite();
        return (result.deviance, result.beta, result.u, converged);
    }

    let result = pirls_impl(
        y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start, maxiter,
        tol,
    );

    let converged = result.converged && result.deviance.is_finite();
    let beta = result.beta;
    let u = result.u;
    let spherical = result.spherical;

    let lambda = result.covariance;
    let mu = result.mean;
    let mut deviance = if converged {
        result.deviance
    } else {
        // A failed factorization retains PIRLS's sentinel deviance. Recompute
        // the likelihood component as before, while preserving the failure flag.
        family.deviance_resids(y, &mu, weights) + spherical.squared_norm_l2()
    };

    let mut w_vec = family.weights(&mu, link);
    for i in 0..n {
        w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
    }

    let logdet_h = if let Some(prepared) = result.random_system.as_ref() {
        let weights = w_vec
            .try_as_col_major()
            .expect("owned weights are contiguous");
        match prepared.factor(weights.as_slice(), 0.0) {
            Ok(factor) => factor.logdet(),
            Err(_) => dense_logdet(z, &lambda, &w_vec),
        }
    } else {
        dense_logdet(z, &lambda, &w_vec)
    };

    deviance += logdet_h;

    (deviance, beta, u, converged)
}

#[allow(clippy::too_many_arguments)]
fn compute_group_log_integral(
    g: usize,
    spherical: &DVector<f64>,
    relative_scale: f64,
    nodes: &[f64],
    weights: &[f64],
    y: &DVector<f64>,
    z: &CscMatrix,
    eta_fixed: &DVector<f64>,
    working_weights: &DVector<f64>,
    prior_weights: &[f64],
    family: FamilyType,
    link: LinkFunction,
) -> f64 {
    let start = z.col_offsets()[g];
    let end = z.col_offsets()[g + 1];
    let entries: Vec<(usize, f64)> = (start..end)
        .filter(|&i| z.values()[i] != 0.0)
        .map(|i| (z.row_indices()[i], z.values()[i]))
        .collect();
    if entries.is_empty() {
        return 0.0;
    }
    let curvature: f64 = entries
        .iter()
        .map(|&(row, value)| value * working_weights[row] * value)
        .sum();
    let hessian = (relative_scale * curvature) * relative_scale + 1.0;
    let scale = 1.0 / (hessian + 1e-10).sqrt();
    let spherical_mode = spherical[g];
    let group_y = DVector::from_fn(entries.len(), |i| y[entries[i].0]);
    let group_weights: Vec<f64> = entries.iter().map(|&(row, _)| prior_weights[row]).collect();
    let mut eta_quad = DVector::zeros(entries.len());
    let mut log_terms = Vec::with_capacity(nodes.len());
    for (node, weight) in nodes.iter().zip(weights.iter()) {
        if *weight == 0.0 {
            continue;
        }
        let spherical_quad = spherical_mode + std::f64::consts::SQRT_2 * scale * node;
        let random_effect = relative_scale * spherical_quad;
        for (i, &(row, value)) in entries.iter().enumerate() {
            eta_quad[i] = eta_fixed[row] + value * random_effect;
        }
        let mut mu_quad = link.inverse(&eta_quad);
        family.clamp_mu(&mut mu_quad, 1e-10);
        let log_lik_y = -0.5 * family.deviance_resids(&group_y, &mu_quad, &group_weights);
        let log_prior = -0.5 * spherical_quad * spherical_quad;
        log_terms.push(weight.ln() + log_lik_y + log_prior + node * node);
    }
    let max_log = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = max_log
        + log_terms
            .iter()
            .map(|value| (value - max_log).exp())
            .sum::<f64>()
            .ln();
    scale.ln() - 0.5 * std::f64::consts::PI.ln() + log_sum
}

/// Preserve small group contributions beside much larger log likelihoods.
fn sum_log_integrals(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = sum + value;
        if next.is_finite() {
            correction += if sum.abs() >= value.abs() {
                (sum - next) + value
            } else {
                (value - next) + sum
            };
        } else {
            // Keep ordinary infinity/NaN propagation, including overflow.
            correction = 0.0;
        }
        sum = next;
    }
    sum + correction
}

/// Keep group addition order independent of worker count and avoid dispatch
/// when the pool has only one worker. Each group still has its own scratch data.
fn sum_group_log_integrals<F>(n_groups: usize, integrate: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    #[cfg(not(miri))]
    if n_groups > 1 && rayon::current_num_threads() > 1 {
        // Indexed collection preserves group order while allowing uneven groups
        // to be scheduled independently. Only one scalar per group is retained.
        let values = (0..n_groups)
            .into_par_iter()
            .map(integrate)
            .collect::<Vec<_>>();
        return sum_log_integrals(values);
    }
    sum_log_integrals((0..n_groups).map(integrate))
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_gh_deviance_impl(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &CscMatrix,
    weights: &[f64],
    offset: &DVector<f64>,
    theta: &[f64],
    structures: &[RandomEffectStructure],
    family: FamilyType,
    link: LinkFunction,
    n_agq: usize,
    beta_start: Option<&DVector<f64>>,
    u_start: Option<&DVector<f64>>,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, DVector<f64>, DVector<f64>, bool)> {
    let q = z.ncols();

    if n_agq <= 1 || q == 0 {
        return Ok(laplace_deviance_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
            maxiter, tol,
        ));
    }

    if structures.len() != 1 {
        return Ok(laplace_deviance_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
            maxiter, tol,
        ));
    }

    let first_struct = &structures[0];
    let n_terms_first = first_struct.n_terms;
    let n_levels_first = first_struct.n_levels;

    if n_terms_first > 1 {
        return Ok(laplace_deviance_impl(
            y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start,
            maxiter, tol,
        ));
    }

    let mut active_rows = vec![false; y.nrows()];
    for (&row, &value) in z.row_indices().iter().zip(z.values().iter()) {
        if value != 0.0 {
            if active_rows[row] {
                return Err(PyValueError::new_err(
                    "Adaptive quadrature requires at most one nonzero random-effect coefficient per observation",
                ));
            }
            active_rows[row] = true;
        }
    }

    let result = pirls_impl(
        y, x, z, weights, offset, theta, structures, family, link, beta_start, u_start, maxiter,
        tol,
    );

    let converged = result.converged && result.deviance.is_finite();
    let beta = result.beta;
    let u = result.u;
    let spherical = result.spherical;

    let n = y.nrows();
    let relative_scale = theta[0];

    let mut eta_fixed = DVector::zeros(n);
    update_fixed_linear_predictor(&mut eta_fixed, x, &beta, offset);
    let mu = result.mean;

    let mut w_vec = family.weights(&mu, link);
    for i in 0..n {
        w_vec[i] = (w_vec[i] * weights[i]).max(1e-10);
    }

    let rule = gauss_hermite_nodes_weights(n_agq);

    let log_integral = sum_group_log_integrals(n_levels_first, |g| {
        compute_group_log_integral(
            g,
            &spherical,
            relative_scale,
            &rule.nodes,
            &rule.weights,
            y,
            z,
            &eta_fixed,
            &w_vec,
            weights,
            family,
            link,
        )
    });

    let fixed_rows: Vec<usize> = active_rows
        .iter()
        .enumerate()
        .filter_map(|(row, &active)| if active { None } else { Some(row) })
        .collect();
    let fixed_deviance = family.deviance_resids_rows(y, &mu, weights, &fixed_rows);
    let deviance = -2.0 * log_integral + fixed_deviance;

    Ok((deviance, beta, u, converged))
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
    link,
    *,
    maxiter=PIRLS_MAX_ITER,
    tol=PIRLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments)]
pub fn pirls<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
    maxiter: usize,
    tol: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, f64, bool)> {
    validate_pirls_controls(maxiter, tol)?;
    let theta = theta.as_slice()?;
    let inputs = GlmmInputs::new(
        y.as_array(),
        x.as_array(),
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
        weights.as_slice()?,
        offset.as_array(),
        Some(theta.len()),
        n_levels,
        n_terms,
        correlated,
        family,
        link,
        1,
    )?;

    let result = pirls_impl(
        &inputs.y,
        &inputs.x,
        &inputs.z,
        inputs.weights,
        &inputs.offset,
        theta,
        &inputs.structures,
        inputs.family,
        inputs.link,
        None,
        None,
        maxiter,
        tol,
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
    link,
    *,
    maxiter=PIRLS_MAX_ITER,
    tol=PIRLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments)]
pub fn laplace_deviance<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let (deviance, beta, u, _) = glmm_deviance(
        y, x, z_data, z_indices, z_indptr, z_shape, weights, offset, theta, n_levels, n_terms,
        correlated, family, link, 1, maxiter, tol,
    )?;
    Ok((deviance, beta, u))
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
    link,
    n_agq,
    *,
    maxiter=PIRLS_MAX_ITER,
    tol=PIRLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_gh_deviance<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
    n_agq: usize,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let (deviance, beta, u, _) = glmm_deviance(
        y, x, z_data, z_indices, z_indptr, z_shape, weights, offset, theta, n_levels, n_terms,
        correlated, family, link, n_agq, maxiter, tol,
    )?;
    Ok((deviance, beta, u))
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
    link,
    n_agq,
    *,
    maxiter=PIRLS_MAX_ITER,
    tol=PIRLS_TOLERANCE
))]
#[allow(clippy::too_many_arguments)]
pub fn glmm_deviance<'py>(
    y: numpy::PyArrayLike1<'py, f64>,
    x: numpy::PyArrayLike2<'py, f64>,
    z_data: numpy::PyArrayLike1<'py, f64>,
    z_indices: numpy::PyArrayLike1<'py, i64>,
    z_indptr: numpy::PyArrayLike1<'py, i64>,
    z_shape: (usize, usize),
    weights: numpy::PyArrayLike1<'py, f64>,
    offset: numpy::PyArrayLike1<'py, f64>,
    theta: numpy::PyArrayLike1<'py, f64>,
    n_levels: Vec<usize>,
    n_terms: Vec<usize>,
    correlated: Vec<bool>,
    family: &str,
    link: &str,
    n_agq: usize,
    maxiter: usize,
    tol: f64,
) -> PyResult<(f64, Vec<f64>, Vec<f64>, bool)> {
    validate_pirls_controls(maxiter, tol)?;
    let theta = theta.as_slice()?;
    let inputs = GlmmInputs::new(
        y.as_array(),
        x.as_array(),
        z_data.as_slice()?,
        z_indices.as_slice()?,
        z_indptr.as_slice()?,
        z_shape,
        weights.as_slice()?,
        offset.as_array(),
        Some(theta.len()),
        n_levels,
        n_terms,
        correlated,
        family,
        link,
        n_agq,
    )?;

    let (deviance, beta, u, converged) = adaptive_gh_deviance_impl(
        &inputs.y,
        &inputs.x,
        &inputs.z,
        inputs.weights,
        &inputs.offset,
        theta,
        &inputs.structures,
        inputs.family,
        inputs.link,
        n_agq,
        None,
        None,
        maxiter,
        tol,
    )?;

    Ok((
        deviance,
        beta.iter().cloned().collect(),
        u.iter().cloned().collect(),
        converged,
    ))
}

#[cfg(test)]
mod quadrature_tests {
    use super::*;

    #[test]
    #[cfg(not(miri))]
    fn group_sums_preserve_small_contributions_for_every_pool_size() {
        for workers in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for n_groups in [4, 65, 2048] {
                let actual = pool.install(|| {
                    sum_group_log_integrals(n_groups, |group| if group == 0 { -1e16 } else { -1.0 })
                });
                let expected = -1e16 - (n_groups - 1) as f64;
                assert_eq!(actual, expected, "{workers} workers, {n_groups} groups");
            }
        }
    }

    #[test]
    fn group_sum_preserves_nonfinite_results() {
        assert_eq!(sum_log_integrals([]), 0.0);
        assert_eq!(
            sum_log_integrals([f64::NEG_INFINITY, -1.0]),
            f64::NEG_INFINITY
        );
        assert_eq!(sum_log_integrals([f64::MAX, f64::MAX]), f64::INFINITY);
        assert!(sum_log_integrals([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert!(sum_log_integrals([1.0, f64::NAN, 2.0]).is_nan());
    }

    #[test]
    fn local_gaussian_integral_and_empty_group() {
        let z = csc_from_scipy(&[1.0, -0.5], &[0, 2], &[0, 2, 2], (3, 2)).unwrap();
        let y = DVector::from_fn(3, |i| [1.0, 4.0, 2.0][i]);
        let eta_fixed = DVector::from_fn(3, |i| [0.0, 1.0, 0.0][i]);
        let prior_weights = [1.0, 2.0, 3.0];
        let working_weights = DVector::from_fn(3, |i| prior_weights[i]);
        let relative_scale = 0.7;
        let hessian = 1.0 + relative_scale * relative_scale * 1.75;
        let score = -2.0 * relative_scale;
        let spherical = DVector::from_fn(2, |i| if i == 0 { score / hessian } else { 0.0 });
        let rule = gauss_hermite_nodes_weights(9);
        let integral = compute_group_log_integral(
            0,
            &spherical,
            relative_scale,
            &rule.nodes,
            &rule.weights,
            &y,
            &z,
            &eta_fixed,
            &working_weights,
            &prior_weights,
            FamilyType::Gaussian,
            LinkFunction::Identity,
        );
        let expected = -0.5 * (13.0 - score * score / hessian + hessian.ln());
        assert!((integral - expected).abs() < 1e-12);
        assert_eq!(
            compute_group_log_integral(
                1,
                &spherical,
                relative_scale,
                &rule.nodes,
                &rule.weights,
                &y,
                &z,
                &eta_fixed,
                &working_weights,
                &prior_weights,
                FamilyType::Gaussian,
                LinkFunction::Identity,
            ),
            0.0
        );
    }
}

#[cfg(test)]
mod initialization_tests {
    use super::*;

    #[test]
    fn weighted_link_scale_starts_match_python_with_offsets() {
        let y = DVector::from_fn(3, |i| [0.0, 0.25, 1.0][i]);
        let x = DMatrix::from_fn(3, 2, |i, j| if j == 0 { 1.0 } else { i as f64 - 1.0 });
        let weights = [0.5, 1.0, 2.0];
        let offset = DVector::from_fn(3, |i| [-0.4, 0.1, 0.7][i]);
        let cases = [
            (
                FamilyType::Gaussian,
                LinkFunction::Identity,
                [0.2730769230769231, -0.0038461538461538026],
            ),
            (
                FamilyType::Binomial,
                LinkFunction::Logit,
                [-0.327240624525381, 0.6549566633833384],
            ),
            (
                FamilyType::Poisson,
                LinkFunction::Log,
                [-1.372447090582741, 0.6439852729484538],
            ),
        ];
        for (family, link, expected) in cases {
            let beta = initial_beta(&y, &x, &weights, &offset, family, link);
            for i in 0..2 {
                assert!((beta[i] - expected[i]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn poisson_large_counts_converge_in_a_few_iterations() {
        let y = DVector::from_fn(12, |i| [998.0, 1000.0, 1002.0][i % 3]);
        let x = DMatrix::full(12, 1, 1.0);
        let z = csc_from_scipy(
            &[1.0; 12],
            &(0..12).collect::<Vec<_>>(),
            &[0, 3, 6, 9, 12],
            (12, 4),
        )
        .unwrap();
        let structures = [RandomEffectStructure {
            n_levels: 4,
            n_terms: 1,
            correlated: true,
        }];
        let result = pirls_impl(
            &y,
            &x,
            &z,
            &[1.0; 12],
            &DVector::full(12, -20.0),
            &[0.5],
            &structures,
            FamilyType::Poisson,
            LinkFunction::Log,
            None,
            None,
            6,
            1e-6,
        );
        assert!(result.converged);
        assert!(result.deviance.is_finite());
        assert!((result.beta[0] - (1000.0_f64.ln() + 20.0)).abs() < 1e-10);
    }

    #[test]
    fn supplied_starts_are_preserved() {
        let y = DVector::full(3, 10.0);
        let x = DMatrix::full(3, 1, 1.0);
        let z = csc_from_scipy(&[1.0; 3], &[0, 1, 2], &[0, 3], (3, 1)).unwrap();
        let structures = [RandomEffectStructure {
            n_levels: 1,
            n_terms: 1,
            correlated: true,
        }];
        let beta = DVector::full(1, 1.25);
        let u = DVector::full(1, 0.4);
        let result = pirls_impl(
            &y,
            &x,
            &z,
            &[1.0; 3],
            &DVector::zeros(3),
            &[0.5],
            &structures,
            FamilyType::Poisson,
            LinkFunction::Log,
            Some(&beta),
            Some(&u),
            0,
            1e-6,
        );
        assert_eq!(result.beta[0], beta[0]);
        assert!((result.u[0] - u[0]).abs() < 1e-14);
        assert!(!result.converged);
    }

    #[test]
    fn nonfinite_updates_never_appear_stationary() {
        let finite = DVector::full(2, 1.0);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = DVector::full(2, value);
            assert_eq!(max_abs_diff(&invalid, &finite), f64::INFINITY);
            assert_eq!(max_abs_diff(&finite, &invalid), f64::INFINITY);
            assert_eq!(max_abs_diff(&invalid, &invalid), f64::INFINITY);
        }
    }
}

#[cfg(test)]
mod final_mode_tests {
    use super::*;

    #[test]
    fn retained_state_matches_returned_parameters_after_every_exit_iteration() {
        let n = 12;
        for (family, link) in [
            (FamilyType::Gaussian, LinkFunction::Identity),
            (FamilyType::Binomial, LinkFunction::Logit),
            (FamilyType::Poisson, LinkFunction::Log),
        ] {
            for terms in 0..=2 {
                for fixed in [0, 2] {
                    let y = DVector::from_fn(n, |i| {
                        if family == FamilyType::Binomial {
                            [0.0, 0.25, 0.75, 1.0][i % 4]
                        } else {
                            [0.0, 2.0, 1.0, 3.0][i % 4]
                        }
                    });
                    let x = DMatrix::from_fn(
                        n,
                        fixed,
                        |i, j| {
                            if j == 0 { 1.0 } else { i as f64 / 12.0 }
                        },
                    );
                    let weights: Vec<_> = (0..n).map(|i| 0.5 + i as f64 / 10.0).collect();
                    let offset = DVector::from_fn(n, |i| -0.2 + i as f64 / 30.0);
                    let mut values = vec![];
                    let mut rows = vec![];
                    let mut offsets = vec![0];
                    for group in 0..3 {
                        for term in 0..terms {
                            for row in (4 * group)..(4 * group + 4) {
                                rows.push(row);
                                values.push(if term == 0 { 1.0 } else { row as f64 / 12.0 });
                            }
                            offsets.push(rows.len());
                        }
                    }
                    let z = CscMatrix::try_from_usize(&values, &rows, &offsets, (n, 3 * terms))
                        .unwrap();
                    let structures = if terms == 0 {
                        vec![]
                    } else {
                        vec![RandomEffectStructure {
                            n_levels: 3,
                            n_terms: terms,
                            correlated: true,
                        }]
                    };
                    let theta: &[f64] = match terms {
                        0 => &[],
                        1 => &[0.4],
                        _ => &[0.4, 0.15, 0.25],
                    };
                    for maxiter in [0, 1, 100] {
                        let state = pirls_impl(
                            &y,
                            &x,
                            &z,
                            &weights,
                            &offset,
                            theta,
                            &structures,
                            family,
                            link,
                            None,
                            None,
                            maxiter,
                            1e-10,
                        );
                        let mut eta = &x * &state.beta + &offset;
                        for column in 0..z.ncols() {
                            for entry in z.col_offsets()[column]..z.col_offsets()[column + 1] {
                                eta[z.row_indices()[entry]] += z.values()[entry] * state.u[column];
                            }
                        }
                        let mut mean = link.inverse(&eta);
                        family.clamp_mu(&mut mean, 1e-10);
                        for row in 0..n {
                            assert_eq!(state.mean[row].to_bits(), mean[row].to_bits());
                        }
                        assert_eq!(state.covariance.apply(&state.spherical), state.u);
                        assert_eq!(
                            state.deviance,
                            family.deviance_resids(&y, &mean, &weights)
                                + state.spherical.squared_norm_l2(),
                        );
                        if maxiter == 0 {
                            assert!(!state.converged);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn failed_factorization_keeps_its_matching_mean_and_sentinel() {
        let y = DVector::from_fn(4, |i| [1.0, 2.0, 1.0, 2.0][i]);
        let x = DMatrix::full(4, 1, 1.0);
        let z = csc_from_scipy(&[1.0; 4], &[0, 1, 2, 3], &[0, 4], (4, 1)).unwrap();
        let structures = [RandomEffectStructure {
            n_levels: 1,
            n_terms: 1,
            correlated: true,
        }];
        let state = pirls_impl(
            &y,
            &x,
            &z,
            &[1.0; 4],
            &DVector::zeros(4),
            &[1e308],
            &structures,
            FamilyType::Poisson,
            LinkFunction::Log,
            None,
            None,
            100,
            1e-6,
        );
        assert!(!state.converged);
        assert_eq!(state.deviance, 1e10);
        assert_eq!(state.covariance.apply(&state.spherical), state.u);
        let expected_mean = LinkFunction::Log.inverse(&(&x * &state.beta));
        assert_eq!(state.mean, expected_mean);
    }
}

#[cfg(test)]
mod input_dimension_tests {
    use super::*;

    #[test]
    fn mixed_covariance_structures_have_exact_counts() {
        let structures = [
            RandomEffectStructure {
                n_levels: 3,
                n_terms: 2,
                correlated: true,
            },
            RandomEffectStructure {
                n_levels: 4,
                n_terms: 2,
                correlated: false,
            },
        ];
        assert!(validate_glmm_dimensions(8, 8, (8, 14), 8, 8, Some(5), &structures).is_ok());
        assert!(validate_glmm_dimensions(8, 8, (8, 14), 8, 8, Some(4), &structures).is_err());
        assert!(validate_glmm_dimensions(8, 8, (8, 14), 8, 8, Some(6), &structures).is_err());
        assert!(validate_glmm_dimensions(8, 8, (8, 13), 8, 8, Some(5), &structures).is_err());
        assert!(validate_glmm_dimensions(8, 8, (8, 0), 8, 8, Some(0), &[]).is_ok());
        assert_eq!(
            validate_glmm_dimensions(8, 8, (8, 14), 8, 8, None, &structures),
            Ok(5)
        );
    }
}

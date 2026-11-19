from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, sparse

from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure, validate_prior_weights
from mixedlm.models import shared_utils


@dataclass
class EMResult:
    """Result from EM algorithm fitting.

    Attributes
    ----------
    theta : NDArray
        Variance component parameters.
    beta : NDArray
        Fixed effects estimates.
    sigma : float
        Residual standard deviation.
    converged : bool
        Whether the algorithm converged.
    n_iter : int
        Number of iterations performed.
    final_loglik : float
        Restricted log likelihood evaluated at the returned variance estimates.
    """

    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    sigma: float
    converged: bool
    n_iter: int
    final_loglik: float


def _ar1_covariance(moment: NDArray[np.floating], floor: float) -> NDArray[np.floating]:
    """Maximize the expected Gaussian log likelihood for a stationary AR(1) block."""
    q = len(moment)
    trace = float(np.trace(moment))
    adjacent = float(np.diag(moment, 1).sum())
    interior = float(np.diag(moment)[1:-1].sum())
    distances = np.abs(np.arange(q)[:, None] - np.arange(q))

    # R(rho)^-1 is tridiagonal. Profiling out the common variance gives
    # q log(trace - 2 adjacent rho + interior rho^2) - log(1 - rho^2).
    # Every interior stationary point is a real root of this cubic.
    roots = np.roots([(1 - q) * interior, (q - 2) * adjacent, trace + q * interior, -q * adjacent])
    candidates = [0.0, *(float(r.real) for r in roots if r.imag == 0 and abs(r.real) < 1)]

    def innovation_sum(rho: float) -> float:
        return trace - 2 * adjacent * rho + interior * rho**2

    rho = min(candidates, key=lambda r: q * np.log(innovation_sum(r)) - np.log1p(-r * r))

    def scales(r: float) -> tuple[float, float]:
        correlation = r**distances
        minimum = linalg.eigvalsh(correlation, subset_by_index=[0, 0])[0]
        variance = innovation_sum(r) / (q * (1 - r * r))
        return variance, floor / max(minimum, np.finfo(float).tiny)

    variance, lower = scales(rho)
    if variance < lower:
        # On the eigenvalue floor, profile the scale subject to that same bound.
        def objective(r: float) -> float:
            scale = max(scales(r))
            return float(
                q * np.log(scale)
                + (q - 1) * np.log1p(-r * r)
                + innovation_sum(r) / ((1 - r * r) * scale)
            )

        bound = np.copysign(1 - np.finfo(float).eps, adjacent)
        optimum = optimize.minimize_scalar(objective, bounds=sorted((0.0, bound)), method="bounded")
        rho = min((0.0, rho, float(optimum.x)), key=objective)
        variance = max(scales(rho))
    return variance * rho**distances


def _m_step_covariance(
    struct: RandomEffectStructure, moment: NDArray[np.floating], variance_floor: float
) -> NDArray[np.floating]:
    """Fit a structured covariance to the mean posterior second moment."""
    q = struct.n_terms
    cov_type = struct.cov_type
    if q == 1 or (cov_type == "us" and not struct.correlated):
        return np.diag(np.maximum(np.diag(moment), variance_floor))
    if cov_type == "ar1" and q > 2:
        return _ar1_covariance(moment, variance_floor)
    # With two terms, AR(1) and compound symmetry have the same covariance.
    if cov_type in ("cs", "ar1"):
        # The mean direction and its orthogonal contrasts are the two
        # eigenspaces of a CS matrix. Constrain their variances separately.
        common = float(moment.sum() / q)
        contrast = max((np.trace(moment) - common) / (q - 1), variance_floor)
        common = max(common, variance_floor)
        covariance = np.full((q, q), (common - contrast) / q)
        covariance.flat[:: q + 1] += contrast
        return covariance

    eigvals, eigvecs = linalg.eigh(moment)
    return (eigvecs * np.maximum(eigvals, variance_floor)) @ eigvecs.T


def _sigma_to_theta(
    structures: list[RandomEffectStructure],
    sigma_list: list[NDArray[np.floating]],
    sigma2_e: float,
) -> NDArray[np.floating]:
    """Convert per-structure covariance matrices to theta parameters.

    For each structure, compute L_k = cholesky(Sigma_k / sigma2_e) and extract
    parameters in the format expected by _build_lambda().
    """
    theta_parts: list[NDArray[np.floating]] = []

    for struct, sigma_k in zip(structures, sigma_list, strict=True):
        q = struct.n_terms
        cov_type = struct.cov_type

        relative_cov = sigma_k / sigma2_e

        if cov_type in ("cs", "ar1"):
            variance = relative_cov[0, 0]
            theta_k = [np.sqrt(variance)]
            if q > 1:
                theta_k.append(relative_cov[0, 1] / variance)
            theta_parts.append(np.asarray(theta_k, dtype=np.float64))
            continue

        L = linalg.cholesky(relative_cov, lower=True)

        if struct.correlated or q == 1:
            row_indices, col_indices = np.tril_indices(q)
            theta_parts.append(L[row_indices, col_indices])
        else:
            theta_parts.append(np.diag(L))

    return np.concatenate(theta_parts) if theta_parts else np.empty(0)


def _validate_em_controls(
    max_iter: int,
    min_iter_converge: int,
    tol: float,
    variance_floor: float,
    sigma2_u_init_min: float,
    sigma2_u_init_scale: float,
) -> None:
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be at least 1 and an integer")
    if (
        isinstance(min_iter_converge, bool)
        or not isinstance(min_iter_converge, Integral)
        or min_iter_converge < 0
    ):
        raise ValueError("min_iter_converge must be a nonnegative integer")
    for name, value in (("variance_floor", variance_floor), ("tol", tol)):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and strictly positive")
    for name, value in (
        ("sigma2_u_init_min", sigma2_u_init_min),
        ("sigma2_u_init_scale", sigma2_u_init_scale),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")


def _relative_precision(
    structures: list[RandomEffectStructure],
    covariances: list[NDArray[np.floating]],
    sigma2: float,
) -> tuple[NDArray[np.floating], float]:
    """Return the repeated precision blocks and log determinant of G/sigma²."""
    blocks = []
    logdet = 0.0
    for struct, covariance in zip(structures, covariances, strict=True):
        chol = linalg.cholesky(covariance, lower=True)
        inverse = linalg.cho_solve((chol, True), np.eye(struct.n_terms))
        blocks.append(np.tile((sigma2 * inverse).ravel(), struct.n_levels))
        logdet += struct.n_levels * (
            2 * np.log(np.diag(chol)).sum() - struct.n_terms * np.log(sigma2)
        )
    return np.concatenate(blocks), float(logdet)


def em_reml_simple(
    matrices: ModelMatrices,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: int = 0,
    variance_floor: float = 1e-8,
    min_iter_converge: int = 5,
    sigma2_u_init_scale: float = 0.5,
    sigma2_u_init_min: float = 0.1,
) -> EMResult:
    """Fit linear mixed model using EM-REML algorithm.

    This is an EM-REML implementation that supports models with multiple
    random effects and random slopes. It can be used as a robust initialization
    method before switching to direct optimization.

    The algorithm alternates between:
    - E-step: Computing conditional expectations of random effects
    - M-step: Updating variance components and fixed effects

    Parameters
    ----------
    matrices : ModelMatrices
        Model design matrices and data.
    max_iter : int, default 100
        Maximum number of EM iterations.
    tol : float, default 1e-4
        Convergence tolerance for relative log-likelihood change.
    verbose : int, default 0
        Verbosity level (0 = silent, 1 = show iterations).
    variance_floor : float, default 1e-8
        Minimum value for variance components to ensure numerical stability.
    min_iter_converge : int, default 5
        Minimum number of iterations before checking for convergence.
    sigma2_u_init_scale : float, default 0.5
        Scaling factor applied to residual variance for initial random effect
        variance estimate.
    sigma2_u_init_min : float, default 0.1
        Minimum initial value for random effect variance.

    Returns
    -------
    EMResult
        Result containing estimated parameters and convergence info.

    Notes
    -----
    This implementation supports random intercepts, random slopes (correlated
    and uncorrelated), and multiple random effect structures with unstructured
    covariance (cov_type='us'), compound symmetry ('cs'), or AR(1) ('ar1').

    The EM algorithm tends to be more robust to poor starting values but
    converges more slowly than direct optimization methods.

    References
    ----------
    .. [1] Lindstrom & Bates (1988). "Newton-Raphson and EM Algorithms for
           Linear Mixed-Effects Models for Repeated-Measures Data"
    .. [2] Dempster, Laird, & Rubin (1977). "Maximum Likelihood from
           Incomplete Data via the EM Algorithm"

    Examples
    --------
    >>> from mixedlm import lFormula
    >>> from mixedlm.estimation.em_reml import em_reml_simple
    >>> parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)
    >>> result = em_reml_simple(parsed.matrices, max_iter=50, verbose=1)
    """
    _validate_em_controls(
        max_iter, min_iter_converge, tol, variance_floor, sigma2_u_init_min, sigma2_u_init_scale
    )
    structures = matrices.random_structures
    for struct in structures:
        if struct.cov_type not in ("us", "cs", "ar1"):
            raise ValueError(f"Unknown covariance type '{struct.cov_type}'")

    X, Z = matrices.X, matrices.Z
    n, p = X.shape
    q = Z.shape[1]
    if n <= p:
        raise ValueError("REML requires more observations than fixed effects")
    y = matrices.y - matrices.offset
    weights = validate_prior_weights(matrices.weights, n)
    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    # Work in an orthonormal fixed-effect basis. Forming X'WX directly can
    # lose the intercept/slope separation when a predictor has a large origin.
    scales = np.max(np.abs(Xw), axis=0)
    if np.any(scales == 0):
        raise ValueError("EM-REML requires a full-rank fixed-effects design")
    Q, R, pivots = linalg.qr(Xw / scales, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(R))
    if p and np.any(diagonal <= diagonal[0] * n * np.finfo(float).eps):
        raise ValueError("EM-REML requires a full-rank fixed-effects design")
    logdet_fixed = 2 * (np.log(diagonal).sum() + np.log(scales).sum())
    alpha_ols = Q.T @ yw

    def coefficients(alpha: NDArray[np.floating]) -> NDArray[np.floating]:
        beta = np.empty(p)
        beta[pivots] = linalg.solve_triangular(R, alpha) / scales[pivots]
        return beta

    weighted_residual = yw - Q @ alpha_ols
    initial_rss = float(weighted_residual @ weighted_residual)
    logdet_weights = np.log(weights).sum()
    if not q:
        sigma2 = max(initial_rss / (n - p), variance_floor)
        loglik = -0.5 * (
            (n - p) * np.log(2 * np.pi * sigma2)
            + initial_rss / sigma2
            + logdet_fixed
            - logdet_weights
        )
        return EMResult(
            np.empty(0), coefficients(alpha_ols), np.sqrt(sigma2), True, 0, float(loglik)
        )

    sigma2 = max(initial_rss / n, variance_floor)
    initial_variance = max(sigma2 * sigma2_u_init_scale, sigma2_u_init_min, variance_floor)
    covariances = [np.eye(s.n_terms) * initial_variance for s in structures]
    Zw = sparse.csc_matrix(Z).multiply(sqrt_w[:, None]).tocsc()
    cross = sparse.csc_matrix(Zw.T @ Q)
    gram = sparse.bmat(
        [[sparse.eye(p), cross.T], [cross, Zw.T @ Zw]],
        format="csc",
    )
    rhs = np.concatenate([Q.T @ weighted_residual, Zw.T @ weighted_residual])

    # Request only the within-level blocks of the joint posterior covariance.
    rows, columns = [], []
    start = p
    for struct in structures:
        levels, terms = struct.n_levels, struct.n_terms
        indices = np.arange(start, start + levels * terms).reshape(levels, terms)
        rows.append(np.repeat(indices, terms, axis=1).ravel())
        columns.append(np.tile(indices, (1, terms)).ravel())
        start += levels * terms
    block_rows, block_cols = np.concatenate(rows), np.concatenate(columns)
    pattern = None
    if shared_utils._HAS_RUST and p + q >= shared_utils._SPARSE_PROJECTION_MIN_RANDOM:
        structural_prior = sparse.csc_matrix(
            (np.ones(len(block_rows)), (block_rows, block_cols)), shape=gram.shape
        )
        pattern = shared_utils._SparseCholeskyPattern(abs(gram) + structural_prior)

    prev_loglik = -np.inf
    converged = False
    for iteration in range(max_iter + 1):
        # H = D'D + diag(0, sigma^2 G^-1) for D = [Q sqrt(W)Z]. Its factor gives
        # the joint mode, posterior covariance sigma^2 H^-1 and REML determinant.
        precision_values, logdet_relative = _relative_precision(structures, covariances, sigma2)
        prior = sparse.csc_matrix((precision_values, (block_rows, block_cols)), shape=gram.shape)
        factor = shared_utils._RandomEffectFactor(gram + prior, pattern=pattern)
        mode = factor.solve(rhs)
        alpha, u = mode[:p], mode[p:]
        residual = weighted_residual - Q @ alpha - Zw @ u
        wrss = residual @ residual
        penalty = mode @ (prior @ mode)
        loglik = -0.5 * (
            (n - p) * np.log(2 * np.pi * sigma2)
            + factor.logdet
            + logdet_relative
            + logdet_fixed
            - logdet_weights
            + (wrss + penalty) / sigma2
        )
        if not np.isfinite(loglik):
            raise ValueError("EM-REML produced a nonfinite restricted log likelihood")
        if iteration:
            change = abs(loglik - prev_loglik) / max(abs(prev_loglik), variance_floor)
            if verbose >= 1:
                print(
                    f"EM iter {iteration}: loglik={loglik:.4f}, "
                    f"sigma2_e={sigma2:.4f}, rel_change={change:.6f}"
                )
            converged = iteration >= min_iter_converge and bool(change < tol)
            if converged or iteration == max_iter:
                break
        prev_loglik = loglik

        entries = factor.inverse_entries(block_rows, block_cols)
        # tr(D'D H^-1) = p + q - tr(diag(0, sigma^2 G^-1) H^-1).
        # This includes uncertainty in beta and its covariance with u.
        uncertainty = sigma2 * (p + q - precision_values @ entries)
        sigma2_new = max(float((wrss + uncertainty) / n), variance_floor)
        offset = 0
        start = 0
        for k, struct in enumerate(structures):
            levels, terms = struct.n_levels, struct.n_terms
            stop = start + levels * terms
            end = offset + levels * terms**2
            U = u[start:stop].reshape(levels, terms)
            posterior_sum = entries[offset:end].reshape(levels, terms, terms).sum(axis=0)
            moment = (U.T @ U + sigma2 * posterior_sum) / levels
            covariances[k] = _m_step_covariance(struct, moment, variance_floor)
            start, offset = stop, end
        sigma2 = sigma2_new

    theta = _sigma_to_theta(structures, covariances, sigma2)
    return EMResult(
        theta, coefficients(alpha_ols + alpha), np.sqrt(sigma2), converged, iteration, float(loglik)
    )

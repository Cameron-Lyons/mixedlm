from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import ReferenceType, ref

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

if TYPE_CHECKING:
    from mixedlm.models.lmer import LmerResult

_NUMERICAL_EPS = 1e-6
_HESSIAN_EPS = 1e-4
_GRADIENT_ZERO_THRESHOLD = 1e-10
_MIN_DF = 1.0
_CHOLESKY_REGULARIZATION = 1e-6


@dataclass
class _WeightedCrossproducts:
    XtWX: NDArray[np.float64]
    ZtWZ: sparse.csc_matrix
    ZtWX: NDArray[np.float64]


@dataclass
class _VcovGradCacheEntry:
    result_ref: ReferenceType[LmerResult]
    parameter_key: bytes
    gradients: list[NDArray[np.floating]]
    theta_covariance: NDArray[np.float64]


_vcov_grad_cache: dict[int, _VcovGradCacheEntry] = {}
_VCOV_GRAD_CACHE_MAX_SIZE = 4


def clear_vcov_grad_cache() -> None:
    _vcov_grad_cache.clear()


def _solve_linear_system(A: NDArray[np.floating], B: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve A X = B with SPD-fast path and robust fallbacks."""
    try:
        L = linalg.cholesky(A, lower=True)
        return linalg.cho_solve((L, True), B)
    except linalg.LinAlgError:
        try:
            return linalg.solve(A, B)
        except linalg.LinAlgError:
            return linalg.pinv(A) @ B


def _matrix_inverse(A: NDArray[np.floating]) -> NDArray[np.floating]:
    return _solve_linear_system(A, np.eye(A.shape[0], dtype=np.float64))


def _weighted_crossproducts(result: LmerResult) -> _WeightedCrossproducts:
    """Build the weighted cross-products shared by covariance perturbations."""
    sqrt_weights = np.sqrt(result.matrices.weights)
    weighted_X = sqrt_weights[:, None] * result.matrices.X
    weighted_Z = result.matrices.Z.multiply(sqrt_weights[:, None]).tocsc()
    return _WeightedCrossproducts(
        XtWX=np.asarray(weighted_X.T @ weighted_X, dtype=np.float64),
        ZtWZ=(weighted_Z.T @ weighted_Z).tocsc(),
        ZtWX=np.asarray(weighted_Z.T @ weighted_X, dtype=np.float64),
    )


def _xt_vinv_x_from_theta(
    result: LmerResult,
    theta: NDArray[np.floating],
    crossproducts: _WeightedCrossproducts | None = None,
    sigma: float | None = None,
) -> NDArray[np.floating]:
    """Compute X'V^-1X without materializing the n x n marginal covariance."""
    from mixedlm.estimation.reml import _build_lambda

    if crossproducts is None:
        crossproducts = _weighted_crossproducts(result)
    residual_scale = result.sigma if sigma is None else sigma

    q = result.matrices.n_random
    if q == 0:
        return crossproducts.XtWX / residual_scale**2

    Lambda = _build_lambda(theta, result.matrices.random_structures)
    lambdat_ztz_lambda = Lambda.T @ crossproducts.ZtWZ @ Lambda
    v_factor = lambdat_ztz_lambda + sparse.eye(q, format="csc")
    v_factor_dense = v_factor.toarray() if sparse.issparse(v_factor) else v_factor

    try:
        chol = linalg.cholesky(v_factor_dense, lower=True)
    except linalg.LinAlgError:
        v_factor_dense += _CHOLESKY_REGULARIZATION * np.eye(q, dtype=np.float64)
        chol = linalg.cholesky(v_factor_dense, lower=True)

    lambdat_ztx = Lambda.T @ crossproducts.ZtWX
    rzx = linalg.solve_triangular(chol, lambdat_ztx, lower=True)
    information = crossproducts.XtWX - rzx.T @ rzx
    return (information + information.T) / (2.0 * residual_scale**2)


def _finite_difference_step(value: float) -> float:
    return _NUMERICAL_EPS * max(1.0, abs(value))


def _vcov_from_theta(
    result: LmerResult,
    theta: NDArray[np.floating],
    crossproducts: _WeightedCrossproducts,
) -> NDArray[np.floating]:
    from mixedlm.estimation.reml import _profiled_deviance_core

    profiled = _profiled_deviance_core(theta, result.matrices, REML=result.REML)
    sigma = result.sigma if profiled is None else profiled.sigma
    information = _xt_vinv_x_from_theta(result, theta, crossproducts, sigma=sigma)
    return _matrix_inverse(information)


def _discard_vcov_grad_cache_entry(
    result_id: int,
    expired_ref: ReferenceType[LmerResult],
) -> None:
    entry = _vcov_grad_cache.get(result_id)
    if entry is not None and entry.result_ref is expired_ref:
        _vcov_grad_cache.pop(result_id, None)


def _profiled_theta_covariance(
    result: LmerResult,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Approximate covariance of relative covariance parameters from deviance curvature."""
    from mixedlm.estimation.reml import profiled_deviance

    n_theta = len(theta)
    if n_theta == 0:
        return np.zeros((0, 0), dtype=np.float64)

    steps = _HESSIAN_EPS * np.maximum(1.0, np.abs(theta))
    hessian = np.zeros((n_theta, n_theta), dtype=np.float64)
    base_deviance = profiled_deviance(theta, result.matrices, REML=result.REML)

    for i in range(n_theta):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += steps[i]
        theta_minus[i] -= steps[i]
        hessian[i, i] = (
            profiled_deviance(theta_plus, result.matrices, REML=result.REML)
            - 2.0 * base_deviance
            + profiled_deviance(theta_minus, result.matrices, REML=result.REML)
        ) / steps[i] ** 2

        for j in range(i):
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()
            theta_pp[i] += steps[i]
            theta_pp[j] += steps[j]
            theta_pm[i] += steps[i]
            theta_pm[j] -= steps[j]
            theta_mp[i] -= steps[i]
            theta_mp[j] += steps[j]
            theta_mm[i] -= steps[i]
            theta_mm[j] -= steps[j]
            mixed_derivative = (
                profiled_deviance(theta_pp, result.matrices, REML=result.REML)
                - profiled_deviance(theta_pm, result.matrices, REML=result.REML)
                - profiled_deviance(theta_mp, result.matrices, REML=result.REML)
                + profiled_deviance(theta_mm, result.matrices, REML=result.REML)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = mixed_derivative
            hessian[j, i] = mixed_derivative

    eigenvalues, eigenvectors = linalg.eigh((hessian + hessian.T) / 2.0)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-8
    positive = eigenvalues > tolerance
    if not np.any(positive):
        return np.zeros((n_theta, n_theta), dtype=np.float64)
    scaled_vectors = eigenvectors[:, positive] * np.sqrt(2.0 / eigenvalues[positive])
    return np.asarray(scaled_vectors @ scaled_vectors.T, dtype=np.float64)


def _vcov_derivatives(
    result: LmerResult,
) -> tuple[list[NDArray[np.floating]], NDArray[np.float64]]:
    """Differentiate coefficient covariance and estimate covariance-parameter uncertainty."""
    theta = np.asarray(result.theta, dtype=np.float64)
    parameter_key = (
        theta.tobytes()
        + np.array([result.sigma], dtype=np.float64).tobytes()
        + result.matrices.weights.tobytes()
    )
    result_id = id(result)
    cached = _vcov_grad_cache.get(result_id)
    if (
        cached is not None
        and cached.result_ref() is result
        and cached.parameter_key == parameter_key
    ):
        return cached.gradients, cached.theta_covariance

    crossproducts = _weighted_crossproducts(result)
    gradients: list[NDArray[np.floating]] = []
    for k, value in enumerate(theta):
        step = _finite_difference_step(float(value))
        theta_plus = theta.copy()
        theta_plus[k] += step
        vcov_plus = _vcov_from_theta(result, theta_plus, crossproducts)

        theta_minus = theta.copy()
        theta_minus[k] -= step
        vcov_minus = _vcov_from_theta(result, theta_minus, crossproducts)
        gradients.append((vcov_plus - vcov_minus) / (2.0 * step))
    theta_covariance = _profiled_theta_covariance(result, theta)

    if len(_vcov_grad_cache) >= _VCOV_GRAD_CACHE_MAX_SIZE:
        _vcov_grad_cache.pop(next(iter(_vcov_grad_cache)))

    def discard_expired(expired: ReferenceType[LmerResult]) -> None:
        _discard_vcov_grad_cache_entry(result_id, expired)

    result_ref = ref(result, discard_expired)
    _vcov_grad_cache[result_id] = _VcovGradCacheEntry(
        result_ref=result_ref,
        parameter_key=parameter_key,
        gradients=gradients,
        theta_covariance=theta_covariance,
    )
    return gradients, theta_covariance


@dataclass
class DenomDFResult:
    df: NDArray[np.floating]
    method: str
    param_names: list[str]

    def __getitem__(self, key: str) -> float:
        idx = self.param_names.index(key)
        return float(self.df[idx])

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.param_names, self.df, strict=False))


def satterthwaite_df(
    result: LmerResult,
    _param_idx: int | None = None,
) -> DenomDFResult:
    """Compute Satterthwaite denominator degrees of freedom.

    Uses weighted mixed-model projections and numerical differentiation
    to propagate covariance-parameter and residual-scale uncertainty into
    the coefficient variance before applying the Satterthwaite formula.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    param_idx : int, optional
        Index of the fixed effect parameter. If None, computes DF
        for all parameters.

    Returns
    -------
    DenomDFResult
        Object containing denominator degrees of freedom for each parameter.
    """
    vcov = result.vcov()
    p = result.matrices.n_fixed
    vcov_grads, theta_covariance = _vcov_derivatives(result)

    df_values = np.zeros(p, dtype=np.float64)

    for i in range(p):
        var_i = vcov[i, i]

        grad_var_i = np.array([g[i, i] for g in vcov_grads])

        residual_df = result.matrices.n_obs - result.matrices.n_fixed
        residual_scale_uncertainty = 2.0 * var_i**2 / residual_df
        variance_of_variance = (
            float(grad_var_i @ theta_covariance @ grad_var_i) + residual_scale_uncertainty
        )

        if variance_of_variance > _GRADIENT_ZERO_THRESHOLD:
            df_values[i] = 2 * var_i**2 / variance_of_variance
        else:
            df_values[i] = result.matrices.n_obs - result.matrices.n_fixed

        df_values[i] = max(_MIN_DF, min(df_values[i], result.matrices.n_obs - p))

    return DenomDFResult(
        df=df_values,
        method="Satterthwaite",
        param_names=list(result.matrices.fixed_names),
    )


def kenward_roger_df(
    result: LmerResult,
    _param_idx: int | None = None,
) -> DenomDFResult:
    """Compute Kenward-Roger denominator degrees of freedom.

    Applies small-sample correction to the Satterthwaite approximation
    by adjusting the variance-covariance matrix.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    param_idx : int, optional
        Index of the fixed effect parameter. If None, computes DF
        for all parameters.

    Returns
    -------
    DenomDFResult
        Object containing denominator degrees of freedom for each parameter.
    """
    vcov = result.vcov()
    p = result.matrices.n_fixed
    n = result.matrices.n_obs
    n_theta = len(result.theta)

    absolute_theta = result.sigma * result.theta
    crossproducts = _weighted_crossproducts(result)

    satt_result = satterthwaite_df(result)
    df_satt = satt_result.df

    W = np.zeros((p, p), dtype=np.float64)

    def compute_hessian_contribution(absolute_theta_vec: NDArray) -> NDArray:
        if result.matrices.n_random == 0:
            return np.zeros((p, p))

        relative_theta = absolute_theta_vec / result.sigma
        XtVinvX = _xt_vinv_x_from_theta(result, relative_theta, crossproducts)
        return XtVinvX - XtVinvX @ vcov @ XtVinvX

    base_hess = compute_hessian_contribution(absolute_theta)

    for k in range(n_theta):
        step = _finite_difference_step(float(absolute_theta[k]))
        theta_plus = absolute_theta.copy()
        theta_plus[k] += step
        hess_plus = compute_hessian_contribution(theta_plus)

        theta_minus = absolute_theta.copy()
        theta_minus[k] -= step
        hess_minus = compute_hessian_contribution(theta_minus)

        d2_hess = (hess_plus - 2 * base_hess + hess_minus) / (step**2)
        W += d2_hess

    if n_theta > 0:
        W /= 2 * n_theta

    scale_factor = 1.0 + np.trace(W @ vcov) / p if p > 0 else 1.0
    scale_factor = max(_MIN_DF, scale_factor)

    df_kr = df_satt / scale_factor

    df_kr = np.maximum(_MIN_DF, np.minimum(df_kr, n - p))

    return DenomDFResult(
        df=df_kr,
        method="Kenward-Roger",
        param_names=list(result.matrices.fixed_names),
    )


def pvalues_with_ddf(
    result: LmerResult,
    method: str = "Satterthwaite",
) -> dict[str, tuple[float, float, float]]:
    """Compute p-values using specified denominator DF method.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    method : str, default "Satterthwaite"
        Method for computing denominator degrees of freedom.
        Options: "Satterthwaite", "Kenward-Roger".

    Returns
    -------
    dict
        Dictionary mapping parameter names to (estimate, t-value, p-value) tuples.
    """
    from scipy import stats

    normalized_method = method.strip().lower().replace("_", "-")
    if normalized_method in ("satterthwaite", "satt"):
        ddf_result = satterthwaite_df(result)
    elif normalized_method in ("kenward-roger", "kr"):
        ddf_result = kenward_roger_df(result)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'Satterthwaite' or 'Kenward-Roger'.")

    vcov = result.vcov()
    beta = result.beta
    se = np.sqrt(np.diag(vcov))

    results: dict[str, tuple[float, float, float]] = {}

    for i, name in enumerate(result.matrices.fixed_names):
        t_val = beta[i] / se[i] if se[i] > 0 else np.nan
        df = ddf_result.df[i]
        p_val = 2 * stats.t.sf(np.abs(t_val), df)
        results[name] = (float(beta[i]), float(t_val), float(p_val))

    return results

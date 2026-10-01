from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, special

from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.estimation.pirls_control import validate_pirls_controls
from mixedlm.estimation.reml import (
    _build_lambda,
    _build_theta_bounds,
    _count_theta,
)
from mixedlm.estimation.validation import validate_finite_real
from mixedlm.families.base import Family, IdentityLink, LogitLink, LogLink
from mixedlm.families.binomial import Binomial
from mixedlm.families.gaussian import Gaussian
from mixedlm.families.poisson import Poisson
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
from mixedlm.utils.quadrature import _positive_integer, hermite_rule

if TYPE_CHECKING:
    from mixedlm.estimation.joint_glmm import JointGLMMObjective

_ETA_CLIP_MIN = -30.0
_ETA_CLIP_MAX = 30.0
_MU_EPS = 1e-7
_MU_EPS_STRICT = 1e-10
_WEIGHT_CLIP_MIN = 1e-10
_WEIGHT_CLIP_MAX = 1e10
_DERIV_CLIP_MIN = -1e10
_DERIV_CLIP_MAX = 1e10
_SQRT_WEIGHT_CLIP_MAX = 1e5
_CHOLESKY_REGULARIZATION = 1e-6
_EIGENVALUE_FLOOR = 1e-10
_lambda_cache: dict[tuple[bytes, tuple[tuple[int, int, bool, str], ...]], sparse.csc_matrix] = {}
_LAMBDA_CACHE_MAX_SIZE = 8
_NATIVE_FAMILY_LINKS = frozenset(
    {("binomial", "logit"), ("poisson", "log"), ("gaussian", "identity")}
)


def _validate_quadrature(nAGQ: int, matrices: ModelMatrices) -> None:
    """Reject unavailable quadrature requests before fitting or backend dispatch."""
    if isinstance(nAGQ, (bool, np.bool_)) or not isinstance(nAGQ, Integral) or nAGQ < 0:
        raise ValueError("nAGQ must be a nonnegative integer")
    if nAGQ > 1 and matrices.n_random:
        structures = matrices.random_structures
        if len(structures) != 1 or structures[0].n_terms != 1:
            raise ValueError(
                "nAGQ > 1 requires one random-effect term with one coefficient per group; "
                "use nAGQ=1 for this model"
            )


def _scale_sparse_rows(
    Z: sparse.csc_matrix,
    row_scale: NDArray[np.floating],
) -> sparse.csc_matrix:
    """Scale sparse rows without explicitly constructing a diagonal matrix."""
    return Z.multiply(row_scale[:, None]).tocsc()


def _get_lambda_cached(
    theta: NDArray[np.floating], random_structures: list[RandomEffectStructure]
) -> sparse.csc_matrix:
    struct_key = tuple(
        (s.n_levels, s.n_terms, s.correlated, getattr(s, "cov_type", "us"))
        for s in random_structures
    )
    key = (theta.tobytes(), struct_key)
    if key in _lambda_cache:
        return _lambda_cache[key]

    Lambda = _build_lambda(theta, random_structures)

    if len(_lambda_cache) >= _LAMBDA_CACHE_MAX_SIZE:
        _lambda_cache.pop(next(iter(_lambda_cache)))
    _lambda_cache[key] = Lambda

    return Lambda


def clear_lambda_cache() -> None:
    _lambda_cache.clear()


try:
    from mixedlm._rust import GlmmProblem as _RustGlmmProblem
    from mixedlm._rust import adaptive_gh_deviance as _rust_adaptive_gh_deviance
    from mixedlm._rust import glmm_deviance as _rust_glmm_deviance
    from mixedlm._rust import laplace_deviance as _rust_laplace_deviance

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _get_family_name(family: Family) -> str | None:
    if type(family) is Binomial:
        return "binomial"
    if type(family) is Poisson:
        return "poisson"
    if type(family) is Gaussian:
        return "gaussian"
    return None


def _get_link_name(family: Family) -> str | None:
    if type(family.link) is LogitLink:
        return "logit"
    if type(family.link) is LogLink:
        return "log"
    if type(family.link) is IdentityLink:
        return "identity"
    return None


def _native_covariance_supported(matrices: ModelMatrices) -> bool:
    return all(getattr(struct, "cov_type", "us") == "us" for struct in matrices.random_structures)


@dataclass
class GLMMOptimizationResult:
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    u: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int
    pirls_converged: bool = True
    message: str = ""
    joint_fit: bool = False


@dataclass
class _PIRLSState:
    beta: NDArray[np.floating]
    spherical: NDArray[np.floating]
    random_effects: NDArray[np.floating]
    deviance: float
    converged: bool


def _as_dense(matrix: Any) -> NDArray[np.floating]:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _random_effects_to_spherical(
    Lambda: sparse.csc_matrix,
    random_effects: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Recover a stable spherical start without densifying the covariance factor."""
    if Lambda.shape[0] == 0:
        return np.array([], dtype=np.float64)
    solution = sparse.linalg.lsmr(
        Lambda,
        np.asarray(random_effects, dtype=np.float64),
        atol=1e-10,
        btol=1e-10,
    )[0]
    return np.asarray(solution, dtype=np.float64)


def _pirls_state(
    matrices: ModelMatrices,
    family: Family,
    theta: NDArray[np.floating],
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    maxiter: int = 25,
    tol: float = 1e-6,
) -> _PIRLSState:
    validate_pirls_controls(maxiter, tol)
    q = matrices.n_random

    prior_weights = matrices.weights
    offset = matrices.offset

    Zt = matrices.Zt

    beta: NDArray[np.floating]
    if beta_start is None:
        mu_start = family.initialize_mu(matrices.y)
        eta_start = family.link(mu_start) - offset
        sqrt_prior_weights = np.sqrt(np.maximum(prior_weights, _WEIGHT_CLIP_MIN))
        weighted_X = sqrt_prior_weights[:, None] * matrices.X
        weighted_eta = sqrt_prior_weights * eta_start
        XtWX = weighted_X.T @ weighted_X
        XtWeta = weighted_X.T @ weighted_eta
        try:
            beta = linalg.solve(XtWX, XtWeta, assume_a="pos")
        except linalg.LinAlgError:
            beta = linalg.lstsq(weighted_X, weighted_eta)[0]
    else:
        beta = np.asarray(beta_start, dtype=np.float64).copy()

    Lambda = _get_lambda_cached(theta, matrices.random_structures)
    spherical = (
        np.zeros(q, dtype=np.float64)
        if u_start is None
        else _random_effects_to_spherical(Lambda, u_start)
    )

    W = np.empty(matrices.n_obs, dtype=np.float64)
    z = np.empty(matrices.n_obs, dtype=np.float64)

    converged = False
    for _iteration in range(maxiter):
        random_effects = np.asarray(Lambda @ spherical).ravel()
        eta = matrices.X @ beta + matrices.Z @ random_effects + offset
        np.clip(eta, _ETA_CLIP_MIN, _ETA_CLIP_MAX, out=eta)
        mu = family.link.inverse(eta)
        family.clamp_mu(mu, eps=_MU_EPS, out=mu)

        np.multiply(family.weights(mu), prior_weights, out=W)
        np.clip(W, _WEIGHT_CLIP_MIN, _WEIGHT_CLIP_MAX, out=W)

        deriv = family.link.deriv(mu)
        np.clip(deriv, _DERIV_CLIP_MIN, _DERIV_CLIP_MAX, out=deriv)
        np.subtract(eta, offset, out=z)
        z += deriv * (matrices.y - mu)
        np.clip(z, _DERIV_CLIP_MIN, _DERIV_CLIP_MAX, out=z)

        W_sqrt = np.sqrt(W)
        np.clip(W_sqrt, 0, _SQRT_WEIGHT_CLIP_MAX, out=W_sqrt)
        WX = W_sqrt[:, None] * matrices.X
        WZ = _scale_sparse_rows(matrices.Z, W_sqrt)

        XtWX = WX.T @ WX
        ZtWZ = WZ.T @ WZ
        XtWZ = WX.T @ WZ

        if not np.all(np.isfinite(XtWX)):
            XtWX = np.nan_to_num(XtWX, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX)

        Wz = W * z
        if not np.all(np.isfinite(Wz)):
            Wz = np.nan_to_num(Wz, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX)

        XtWz = matrices.X.T @ Wz
        ZtWz = Zt @ Wz

        if q > 0:
            C = _as_dense(Lambda.T @ ZtWZ @ Lambda) + np.eye(q)
            if not np.all(np.isfinite(C)):
                C = np.nan_to_num(C, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX)

            try:
                L_C = linalg.cholesky(C, lower=True)
            except (linalg.LinAlgError, ValueError):
                C += _CHOLESKY_REGULARIZATION * np.eye(q)
                L_C = linalg.cholesky(C, lower=True)

            ZtWX = _as_dense(XtWZ).T
            ZtWX_spherical = np.asarray(Lambda.T @ ZtWX)
            ZtWz_spherical = np.asarray(Lambda.T @ ZtWz).ravel()
            if not np.all(np.isfinite(ZtWX_spherical)):
                ZtWX_spherical = np.nan_to_num(
                    ZtWX_spherical,
                    nan=0.0,
                    posinf=_WEIGHT_CLIP_MAX,
                    neginf=-_WEIGHT_CLIP_MAX,
                )
            if not np.all(np.isfinite(ZtWz_spherical)):
                ZtWz_spherical = np.nan_to_num(
                    ZtWz_spherical,
                    nan=0.0,
                    posinf=_WEIGHT_CLIP_MAX,
                    neginf=-_WEIGHT_CLIP_MAX,
                )

            RZX = linalg.solve_triangular(L_C, ZtWX_spherical, lower=True)
            cu = linalg.solve_triangular(L_C, ZtWz_spherical, lower=True)
            if not np.all(np.isfinite(cu)):
                cu = np.nan_to_num(cu, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX)

            XtVinvX = XtWX - RZX.T @ RZX
            XtVinvz = XtWz - RZX.T @ cu
        else:
            XtVinvX = XtWX
            XtVinvz = XtWz

        if not np.all(np.isfinite(XtVinvX)):
            XtVinvX = np.nan_to_num(
                XtVinvX, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX
            )
        if not np.all(np.isfinite(XtVinvz)):
            XtVinvz = np.nan_to_num(
                XtVinvz, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX
            )

        try:
            beta_new = linalg.solve(XtVinvX, XtVinvz, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            XtVinvX += _CHOLESKY_REGULARIZATION * np.eye(XtVinvX.shape[0])
            beta_new = linalg.lstsq(XtVinvX, XtVinvz)[0]
        if not np.all(np.isfinite(beta_new)):
            beta_new = beta.copy()

        if q > 0:
            spherical_rhs = ZtWz_spherical - ZtWX_spherical @ beta_new
            if not np.all(np.isfinite(spherical_rhs)):
                spherical_rhs = np.nan_to_num(
                    spherical_rhs,
                    nan=0.0,
                    posinf=_WEIGHT_CLIP_MAX,
                    neginf=-_WEIGHT_CLIP_MAX,
                )
            spherical_new = linalg.cho_solve((L_C, True), spherical_rhs)
        else:
            spherical_new = spherical

        delta_beta = np.max(np.abs(beta_new - beta), initial=0.0)
        delta_u = np.max(np.abs(spherical_new - spherical)) if q > 0 else 0.0

        beta = beta_new
        spherical = spherical_new

        if delta_beta < tol and delta_u < tol:
            converged = True
            break

    random_effects = np.asarray(Lambda @ spherical).ravel()
    eta = matrices.X @ beta + matrices.Z @ random_effects + offset
    mu = family.link.inverse(eta)
    family.clamp_mu(mu, eps=_MU_EPS_STRICT, out=mu)

    dev_resids = family.deviance_resids(matrices.y, mu, prior_weights)
    deviance = np.sum(dev_resids)

    deviance += np.dot(spherical, spherical)

    return _PIRLSState(
        beta=beta,
        spherical=spherical,
        random_effects=random_effects,
        deviance=float(deviance),
        converged=converged,
    )


def pirls(
    matrices: ModelMatrices,
    family: Family,
    theta: NDArray[np.floating],
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    maxiter: int = 25,
    tol: float = 1e-6,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, bool]:
    """Solve penalized IRLS and return covariance-scale random effects."""
    state = _pirls_state(matrices, family, theta, beta_start, u_start, maxiter, tol)
    return state.beta, state.random_effects, state.deviance, state.converged


def laplace_deviance(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate deviance and fitted coefficients; use glmm_deviance_with_status for inner status."""
    return _laplace_deviance_with_status(
        theta,
        matrices,
        family,
        beta_start,
        u_start,
        pirls_maxiter=pirls_maxiter,
        pirls_tol=pirls_tol,
    )[:3]


def _laplace_deviance_with_status(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
    q = matrices.n_random

    prior_weights = matrices.weights
    offset = matrices.offset

    if q == 0:
        state = _pirls_state(
            matrices,
            family,
            theta,
            beta_start,
            u_start,
            maxiter=25 if pirls_maxiter is None else pirls_maxiter,
            tol=pirls_tol,
        )
        return (
            state.deviance,
            state.beta,
            state.random_effects,
            bool(state.converged and np.isfinite(state.deviance)),
        )

    state = _pirls_state(
        matrices,
        family,
        theta,
        beta_start,
        u_start,
        maxiter=25 if pirls_maxiter is None else pirls_maxiter,
        tol=pirls_tol,
    )
    beta = state.beta
    spherical = state.spherical
    random_effects = state.random_effects
    Lambda = _get_lambda_cached(theta, matrices.random_structures)

    eta_fixed = matrices.X @ beta + offset
    eta = eta_fixed + matrices.Z @ random_effects
    mu = family.link.inverse(eta)
    mu = family.clamp_mu(mu, eps=_MU_EPS_STRICT)

    dev_resids = family.deviance_resids(matrices.y, mu, prior_weights)
    deviance = np.sum(dev_resids) + np.dot(spherical, spherical)

    W = family.weights(mu) * prior_weights
    W = np.clip(W, _WEIGHT_CLIP_MIN, _WEIGHT_CLIP_MAX)

    W_sqrt = np.sqrt(W)
    np.clip(W_sqrt, 0, _SQRT_WEIGHT_CLIP_MAX, out=W_sqrt)
    WZ = _scale_sparse_rows(matrices.Z, W_sqrt)
    ZtWZ = WZ.T @ WZ
    H = _as_dense(Lambda.T @ ZtWZ @ Lambda) + np.eye(q)
    if not np.all(np.isfinite(H)):
        H = np.nan_to_num(H, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX)

    try:
        L_H = linalg.cholesky(H, lower=True)
        logdet_H = 2.0 * np.sum(np.log(np.diag(L_H)))
    except (linalg.LinAlgError, ValueError):
        H = np.nan_to_num(H, nan=0.0, posinf=_WEIGHT_CLIP_MAX, neginf=-_WEIGHT_CLIP_MAX)
        H += _CHOLESKY_REGULARIZATION * np.eye(q)
        try:
            L_H = linalg.cholesky(H, lower=True)
            logdet_H = 2.0 * np.sum(np.log(np.diag(L_H)))
        except linalg.LinAlgError:
            eigvals = linalg.eigvalsh(H)
            eigvals = np.maximum(eigvals, _EIGENVALUE_FLOOR)
            logdet_H = np.sum(np.log(eigvals))

    deviance += logdet_H

    return (
        float(deviance),
        beta,
        random_effects,
        bool(state.converged and np.isfinite(state.deviance)),
    )


def _get_gh_nodes_weights(n: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get a shared, immutable Gauss-Hermite quadrature rule."""
    return hermite_rule(n)


def _compute_group_quadrature(
    spherical_mode: float,
    scale: float,
    relative_scale: float,
    z_values: NDArray[np.floating],
    y: NDArray[np.floating],
    eta_fixed: NDArray[np.floating],
    prior_weights: NDArray[np.floating],
    nodes: NDArray[np.floating],
    weights: NDArray[np.floating],
    family: Family,
) -> float:
    """Integrate one scalar random effect over only the observations it affects."""
    sqrt2 = np.sqrt(2.0)
    log_terms = np.empty(len(nodes), dtype=np.float64)
    for i, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
        if weight == 0:
            log_terms[i] = -np.inf
            continue
        spherical_quad = spherical_mode + sqrt2 * scale * node
        eta_quad = eta_fixed + z_values * (relative_scale * spherical_quad)
        mu_quad = family.link.inverse(eta_quad)
        mu_quad = family.clamp_mu(mu_quad, eps=_MU_EPS_STRICT)
        log_lik_y = -0.5 * np.sum(family.deviance_resids(y, mu_quad, prior_weights))
        log_prior = -0.5 * spherical_quad**2
        log_terms[i] = np.log(weight) + log_lik_y + log_prior + node**2

    return float(np.log(scale) - 0.5 * np.log(np.pi) + special.logsumexp(log_terms))


def adaptive_gh_deviance(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    n_jobs: int = 1,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Compute deviance using adaptive Gauss-Hermite quadrature.

    For nAGQ=0 or 1, this evaluates the Laplace approximation with beta
    estimated by PIRLS. Joint theta/beta fitting uses JointGLMMObjective.
    For nAGQ>1, uses adaptive GH quadrature for more accurate integration.

    Parameters
    ----------
    theta : NDArray
        Variance component parameters.
    matrices : ModelMatrices
        Model design matrices.
    family : Family
        GLM family with link function.
    nAGQ : int, default 1
        Number of quadrature points.
    beta_start : NDArray, optional
        Starting values for fixed effects.
    u_start : NDArray, optional
        Starting values for random effects.
    n_jobs : int, default 1
        Number of parallel jobs for group quadrature. Use -1 for all CPUs.

    Returns
    -------
    deviance : float
        -2 * log-likelihood approximation.
    beta : NDArray
        Fixed effect estimates.
    u : NDArray
        Random effect estimates.
    """
    return _adaptive_gh_deviance_with_status(
        theta,
        matrices,
        family,
        nAGQ,
        beta_start,
        u_start,
        n_jobs,
        pirls_maxiter=pirls_maxiter,
        pirls_tol=pirls_tol,
    )[:3]


def _adaptive_gh_deviance_with_status(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    n_jobs: int = 1,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
    """Evaluate adaptive quadrature and retain the inner convergence flag."""
    _validate_quadrature(nAGQ, matrices)
    if nAGQ <= 1:
        return _laplace_deviance_with_status(
            theta,
            matrices,
            family,
            beta_start,
            u_start,
            pirls_maxiter=pirls_maxiter,
            pirls_tol=pirls_tol,
        )

    q = matrices.n_random
    prior_weights = matrices.weights
    offset = matrices.offset

    if q == 0:
        state = _pirls_state(
            matrices,
            family,
            theta,
            beta_start,
            u_start,
            maxiter=25 if pirls_maxiter is None else pirls_maxiter,
            tol=pirls_tol,
        )
        return (
            state.deviance,
            state.beta,
            state.random_effects,
            bool(state.converged and np.isfinite(state.deviance)),
        )

    first_struct = matrices.random_structures[0]
    Z = matrices.Z.tocsc()
    if not Z.has_canonical_format or np.any(Z.data == 0):
        Z = Z.copy()
        Z.sum_duplicates()
        Z.eliminate_zeros()
    row_counts = np.bincount(Z.indices, minlength=matrices.n_obs)
    if np.any(row_counts > 1):
        raise ValueError(
            "Adaptive quadrature requires at most one nonzero random-effect "
            "coefficient per observation"
        )

    state = _pirls_state(
        matrices,
        family,
        theta,
        beta_start,
        u_start,
        maxiter=25 if pirls_maxiter is None else pirls_maxiter,
        tol=pirls_tol,
    )
    beta = state.beta
    spherical = state.spherical
    random_effects = state.random_effects
    relative_scale = theta[0]

    eta_fixed = matrices.X @ beta + offset
    eta = eta_fixed + Z @ random_effects
    mu = family.clamp_mu(family.link.inverse(eta), eps=_MU_EPS_STRICT)
    W = np.clip(family.weights(mu) * prior_weights, _WEIGHT_CLIP_MIN, _WEIGHT_CLIP_MAX)
    sqrt_W = np.sqrt(W)
    nodes, weights = _get_gh_nodes_weights(nAGQ)
    n_levels_first = first_struct.n_levels

    def integrate_group(g: int) -> float:
        start, end = Z.indptr[g : g + 2]
        if start == end:
            return 0.0
        rows = Z.indices[start:end]
        z_values = Z.data[start:end]
        weighted_z = z_values * sqrt_W[rows]
        hessian = (relative_scale * np.dot(weighted_z, weighted_z)) * relative_scale + 1.0
        scale = 1.0 / np.sqrt(hessian)
        return _compute_group_quadrature(
            spherical[g],
            scale,
            relative_scale,
            z_values,
            matrices.y[rows],
            eta_fixed[rows],
            prior_weights[rows],
            nodes,
            weights,
            family,
        )

    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and n_levels_first > 2:
        with ThreadPoolExecutor(max_workers=min(n_jobs, n_levels_first)) as executor:
            log_integral = sum(executor.map(integrate_group, range(n_levels_first)))
    else:
        log_integral = sum(integrate_group(g) for g in range(n_levels_first))

    # A zero design row has no random contribution, but its response still contributes.
    fixed_rows = row_counts == 0
    fixed_deviance = (
        np.sum(
            family.deviance_resids(
                matrices.y[fixed_rows], mu[fixed_rows], prior_weights[fixed_rows]
            )
        )
        if np.any(fixed_rows)
        else 0.0
    )
    deviance = float(-2.0 * log_integral + fixed_deviance)
    return deviance, beta, random_effects, bool(state.converged and np.isfinite(deviance))


def _native_glmm_args(
    theta: NDArray[np.floating], matrices: ModelMatrices, family: Family
) -> tuple[Any, ...]:
    z_csc = matrices.Z.tocsc()
    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if family_name is None or link_name is None:
        raise TypeError("Family and link are not supported by the native GLMM backend")
    return (
        np.ascontiguousarray(matrices.y, dtype=np.float64),
        np.ascontiguousarray(matrices.X, dtype=np.float64),
        np.ascontiguousarray(z_csc.data, dtype=np.float64),
        np.ascontiguousarray(z_csc.indices, dtype=np.int64),
        np.ascontiguousarray(z_csc.indptr, dtype=np.int64),
        z_csc.shape,
        np.ascontiguousarray(matrices.weights, dtype=np.float64),
        np.ascontiguousarray(matrices.offset, dtype=np.float64),
        np.ascontiguousarray(theta, dtype=np.float64),
        [s.n_levels for s in matrices.random_structures],
        [s.n_terms for s in matrices.random_structures],
        [s.correlated for s in matrices.random_structures],
        family_name,
        link_name,
    )


def _prepare_native_glmm(matrices: ModelMatrices, family: Family) -> Any | None:
    """Snapshot eligible native inputs once for the lifetime of an objective."""
    if (
        not _HAS_RUST
        or (_get_family_name(family), _get_link_name(family)) not in _NATIVE_FAMILY_LINKS
        or not _native_covariance_supported(matrices)
    ):
        return None
    args = _native_glmm_args(np.empty(0), matrices, family)
    return _RustGlmmProblem(*args[:8], *args[9:])


def _evaluate_native_problem(
    problem: Any,
    theta: NDArray[np.floating],
    nAGQ: int,
    *,
    offset: NDArray[np.floating] | None = None,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
    validate_pirls_controls(pirls_maxiter, pirls_tol)
    deviance, beta, u, converged = problem.evaluate(
        np.ascontiguousarray(theta, dtype=np.float64),
        max(1, nAGQ),
        offset=offset,
        maxiter=100 if pirls_maxiter is None else pirls_maxiter,
        tol=pirls_tol,
    )
    return deviance, np.array(beta), np.array(u), converged


def _laplace_deviance_rust(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    validate_pirls_controls(pirls_maxiter, pirls_tol)
    deviance, beta, u = _rust_laplace_deviance(
        *_native_glmm_args(theta, matrices, family),
        maxiter=100 if pirls_maxiter is None else pirls_maxiter,
        tol=pirls_tol,
    )
    return deviance, np.array(beta), np.array(u)


def _adaptive_gh_deviance_rust(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    validate_pirls_controls(pirls_maxiter, pirls_tol)
    deviance, beta, u = _rust_adaptive_gh_deviance(
        *_native_glmm_args(theta, matrices, family),
        nAGQ,
        maxiter=100 if pirls_maxiter is None else pirls_maxiter,
        tol=pirls_tol,
    )
    return deviance, np.array(beta), np.array(u)


def _native_deviance_with_status(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
    validate_pirls_controls(pirls_maxiter, pirls_tol)
    deviance, beta, u, converged = _rust_glmm_deviance(
        *_native_glmm_args(theta, matrices, family),
        max(1, nAGQ),
        maxiter=100 if pirls_maxiter is None else pirls_maxiter,
        tol=pirls_tol,
    )
    return deviance, np.array(beta), np.array(u), converged


def laplace_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if (
        _HAS_RUST
        and beta_start is None
        and u_start is None
        and (family_name, link_name) in _NATIVE_FAMILY_LINKS
        and _native_covariance_supported(matrices)
    ):
        return _laplace_deviance_rust(
            theta, matrices, family, pirls_maxiter=pirls_maxiter, pirls_tol=pirls_tol
        )
    return laplace_deviance(
        theta,
        matrices,
        family,
        beta_start,
        u_start,
        pirls_maxiter=pirls_maxiter,
        pirls_tol=pirls_tol,
    )


def adaptive_gh_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    _validate_quadrature(nAGQ, matrices)

    if nAGQ <= 1:
        return laplace_deviance_fast(
            theta,
            matrices,
            family,
            beta_start,
            u_start,
            pirls_maxiter=pirls_maxiter,
            pirls_tol=pirls_tol,
        )

    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if (
        _HAS_RUST
        and beta_start is None
        and u_start is None
        and (family_name, link_name) in _NATIVE_FAMILY_LINKS
        and _native_covariance_supported(matrices)
    ):
        first_struct = matrices.random_structures[0] if matrices.random_structures else None
        if first_struct and first_struct.n_terms == 1:
            return _adaptive_gh_deviance_rust(
                theta, matrices, family, nAGQ, pirls_maxiter=pirls_maxiter, pirls_tol=pirls_tol
            )

    return adaptive_gh_deviance(
        theta,
        matrices,
        family,
        nAGQ,
        beta_start,
        u_start,
        pirls_maxiter=pirls_maxiter,
        pirls_tol=pirls_tol,
    )


def glmm_deviance_with_status(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    *,
    pirls_maxiter: int | None = None,
    pirls_tol: float = 1e-6,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
    """Return deviance, fixed effects, random effects, and inner PIRLS convergence.

    Status comes from the same evaluation as the estimates. Outer optimization
    success alone does not establish convergence of the conditional mode.
    """
    _validate_quadrature(nAGQ, matrices)
    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if (
        _HAS_RUST
        and beta_start is None
        and u_start is None
        and (family_name, link_name) in _NATIVE_FAMILY_LINKS
        and _native_covariance_supported(matrices)
        and (
            nAGQ <= 1 or (matrices.random_structures and matrices.random_structures[0].n_terms == 1)
        )
    ):
        return _native_deviance_with_status(
            theta, matrices, family, nAGQ, pirls_maxiter=pirls_maxiter, pirls_tol=pirls_tol
        )
    return _adaptive_gh_deviance_with_status(
        theta,
        matrices,
        family,
        nAGQ,
        beta_start,
        u_start,
        pirls_maxiter=pirls_maxiter,
        pirls_tol=pirls_tol,
    )


class GLMMOptimizer:
    """Optimize a fixed GLMM problem, reusing native input preparation.

    Treat the model arrays and family as immutable for this object's lifetime.
    Construct a new optimizer when the response, design, weights or offsets change.
    """

    def __init__(
        self,
        matrices: ModelMatrices,
        family: Family,
        verbose: int = 0,
        nAGQ: int = 1,
        *,
        pirls_maxiter: int | None = None,
        pirls_tol: float = 1e-6,
        nAGQ0initStep: bool = True,
    ) -> None:
        _validate_quadrature(nAGQ, matrices)
        validate_pirls_controls(pirls_maxiter, pirls_tol)
        self.pirls_maxiter = pirls_maxiter
        self.pirls_tol = pirls_tol
        if not isinstance(nAGQ0initStep, (bool, np.bool_)):
            raise ValueError("nAGQ0initStep must be a boolean")
        self.nAGQ0initStep = bool(nAGQ0initStep)
        self.matrices = matrices
        self.family = family
        self.verbose = verbose
        self.nAGQ = nAGQ
        self.n_theta = _count_theta(matrices.random_structures)
        self._native_problem = _prepare_native_glmm(matrices, family)

    def get_start_theta(self) -> NDArray[np.floating]:
        theta_list: list[float] = []
        for struct in self.matrices.random_structures:
            q = struct.n_terms
            cov_type = getattr(struct, "cov_type", "us")
            if cov_type == "cs" or cov_type == "ar1":
                theta_list.append(1.0)
                if q > 1:
                    theta_list.append(0.0)
            elif struct.correlated:
                for i in range(q):
                    for j in range(i + 1):
                        theta_list.append(1.0 if i == j else 0.0)
            else:
                theta_list.extend([1.0] * q)
        return np.array(theta_list, dtype=np.float64)

    def objective(self, theta: NDArray[np.floating]) -> float:
        if self._native_problem is not None and (self.nAGQ <= 1 or self.matrices.n_random):
            return _evaluate_native_problem(
                self._native_problem,
                theta,
                self.nAGQ,
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )[0]
        if self.nAGQ > 1:
            dev, _, _ = adaptive_gh_deviance_fast(
                theta,
                self.matrices,
                self.family,
                nAGQ=self.nAGQ,
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )
        else:
            dev, _, _ = laplace_deviance_fast(
                theta,
                self.matrices,
                self.family,
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )
        return dev

    def joint_objective(self) -> JointGLMMObjective:
        from mixedlm.estimation.joint_glmm import JointGLMMObjective

        return JointGLMMObjective(
            self.matrices,
            self.family,
            self.nAGQ,
            pirls_maxiter=self.pirls_maxiter,
            pirls_tol=self.pirls_tol,
        )

    def _final_evaluation(
        self,
        theta: NDArray[np.floating],
        *,
        nAGQ: int | None = None,
        beta: NDArray[np.floating] | None = None,
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
        return self._final_evaluation_with_status(theta, nAGQ=nAGQ, beta=beta)[:3]

    def _final_evaluation_with_status(
        self,
        theta: NDArray[np.floating],
        *,
        nAGQ: int | None = None,
        beta: NDArray[np.floating] | None = None,
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
        """Evaluate and validate final estimates and their inner convergence."""
        nAGQ = self.nAGQ if nAGQ is None else nAGQ
        try:
            validate_finite_real("variance parameters", theta, (self.n_theta,))
            if beta is None:
                deviance, beta, u, converged = glmm_deviance_with_status(
                    theta,
                    self.matrices,
                    self.family,
                    nAGQ=nAGQ,
                    pirls_maxiter=self.pirls_maxiter,
                    pirls_tol=self.pirls_tol,
                )
            else:
                objective = self.joint_objective()
                objective.nAGQ = nAGQ
                deviance, beta, u, converged = objective.evaluate(np.r_[theta, beta])
            validate_finite_real("deviance", deviance, ())
            validate_finite_real("fixed effects", beta, (self.matrices.n_fixed,))
            validate_finite_real("random effects", u, (self.matrices.n_random,))
        except (
            FloatingPointError,
            OverflowError,
            TypeError,
            ValueError,
            linalg.LinAlgError,
        ) as exc:
            raise RuntimeError(
                f"Generalized optimization did not produce a valid fit: {type(exc).__name__}: {exc}"
            ) from exc
        return deviance, beta, u, converged

    def optimize(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        options: dict[str, Any] | None = None,
        *,
        restart_edge: bool = True,
    ) -> GLMMOptimizationResult:
        """Optimize theta and beta jointly, or use the nAGQ=0 PIRLS approximation."""
        _validate_quadrature(self.nAGQ, self.matrices)
        if self.nAGQ == 0:
            return self._optimize_pirls(start, method, maxiter, options, restart_edge=restart_edge)
        exact_pirls = self.matrices.n_random == 0 or (
            self.nAGQ == 1
            and (
                self.matrices.n_fixed == 0
                or (type(self.family) is Gaussian and type(self.family.link) is IdentityLink)
            )
        )
        if exact_pirls:
            fitted = self._optimize_pirls(
                start, method, maxiter, options, restart_edge=restart_edge
            )
            return replace(fitted, joint_fit=fitted.pirls_converged)
        if start is None:
            start = self.get_start_theta()
        if self.nAGQ0initStep:
            initial = GLMMOptimizer(
                self.matrices,
                self.family,
                self.verbose,
                nAGQ=0,
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )._optimize_pirls(start, method, maxiter, options, restart_edge=restart_edge)
            theta, beta = initial.theta, initial.beta
            initial_iterations = initial.n_iter
            initial_converged = initial.pirls_converged
        else:
            theta = np.asarray(start, dtype=np.float64)
            _, beta, _, initial_converged = self._final_evaluation_with_status(theta, nAGQ=1)
            initial_iterations = 0
        if not initial_converged:
            # A failed mode solve (including non-finite MLEs for separated data)
            # cannot provide a trustworthy starting point for the joint likelihood.
            deviance, beta, u, _ = self._final_evaluation_with_status(theta)
            return GLMMOptimizationResult(
                theta=theta,
                beta=beta,
                u=u,
                deviance=deviance,
                converged=False,
                pirls_converged=False,
                n_iter=initial_iterations,
                message="initial inner PIRLS solver did not converge",
            )
        objective = self.joint_objective()
        scale = objective.parameter_scale(theta)
        parameters = np.r_[theta, beta] / scale
        bounds = [
            (None if lower is None else lower / step, None if upper is None else upper / step)
            for (lower, upper), step in zip(objective.bounds, scale, strict=True)
        ]

        def scaled_objective(values: NDArray[np.floating]) -> float:
            return objective(values * scale)

        callback = None
        if self.verbose > 0:

            def callback(values: NDArray[np.floating]) -> None:
                print(
                    f"joint parameters = {values * scale}, "
                    f"deviance = {scaled_objective(values):.6f}"
                )

        opt_options = {"maxiter": maxiter}
        if options:
            opt_options.update(options)
        result = run_optimizer(
            scaled_objective,
            parameters,
            method=method,
            bounds=bounds,
            options=opt_options,
            callback=callback,
            jac="3-point"
            if method in {"L-BFGS-B", "BFGS", "TNC", "SLSQP", "trust-constr"}
            else None,
            restart_edge=restart_edge,
        )
        optimum = result.x * scale
        theta, beta = optimum[: self.n_theta], optimum[self.n_theta :]
        deviance, beta, u, pirls_converged = self._final_evaluation_with_status(theta, beta=beta)
        return GLMMOptimizationResult(
            theta=theta,
            beta=beta,
            u=u,
            deviance=deviance,
            converged=bool(result.success and pirls_converged),
            pirls_converged=pirls_converged,
            n_iter=initial_iterations + result.nit,
            joint_fit=True,
            message=str(getattr(result, "message", "")),
        )

    def _optimize_pirls(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        options: dict[str, Any] | None = None,
        *,
        restart_edge: bool = True,
    ) -> GLMMOptimizationResult:
        _validate_quadrature(self.nAGQ, self.matrices)
        if start is None:
            start = self.get_start_theta()

        bounds = _build_theta_bounds(
            self.matrices.random_structures, len(start), eps=_CHOLESKY_REGULARIZATION
        )

        callback: Callable[[NDArray[np.floating]], None] | None = None
        if self.verbose > 0:

            def callback(x: NDArray[np.floating]) -> None:
                dev = self.objective(x)
                print(f"theta = {x}, deviance = {dev:.6f}")

        opt_options = {"maxiter": maxiter}
        if options:
            opt_options.update(options)

        result = run_optimizer(
            self.objective,
            start,
            method=method,
            bounds=bounds,
            options=opt_options,
            callback=callback,
            restart_edge=restart_edge,
        )

        theta_opt = result.x

        final_dev, beta, u, pirls_converged = self._final_evaluation_with_status(theta_opt)

        return GLMMOptimizationResult(
            theta=theta_opt,
            beta=beta,
            u=u,
            deviance=final_dev,
            converged=bool(result.success and pirls_converged),
            pirls_converged=pirls_converged,
            n_iter=result.nit,
            message=str(getattr(result, "message", "")),
        )


def GQdk(d: int, k: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate d-dimensional Gauss-Hermite quadrature rule with k points per dimension.

    Creates a tensor product grid of Gauss-Hermite quadrature nodes and weights
    for integration over R^d with respect to the multivariate normal distribution.

    Parameters
    ----------
    d : int
        Dimension of the integration domain.
    k : int
        Number of quadrature points per dimension.

    Returns
    -------
    nodes : ndarray
        Quadrature nodes with shape (k^d, d).
    weights : ndarray
        Quadrature weights with shape (k^d,).

    Examples
    --------
    >>> nodes, weights = GQdk(2, 3)
    >>> nodes.shape
    (9, 2)
    >>> weights.shape
    (9,)

    Notes
    -----
    The nodes and weights are scaled for integration with respect to
    the standard multivariate normal distribution N(0, I).

    For 1D integration of f(x) * phi(x) where phi is the standard normal pdf:
        integral ≈ sum(weights * f(nodes))

    For d > 1, the total number of nodes is k^d, which grows exponentially.
    For high dimensions, consider sparse grids or other methods.
    """
    d = _positive_integer(d, "d")
    k = _positive_integer(k, "k")
    nodes_1d, weights_1d = _get_gh_nodes_weights(k)

    nodes_1d = nodes_1d * np.sqrt(2)
    weights_1d = weights_1d / np.sqrt(np.pi)

    if d == 1:
        return nodes_1d.reshape(-1, 1), weights_1d

    grids = [nodes_1d] * d
    weight_grids = [weights_1d] * d

    mesh = np.meshgrid(*grids, indexing="ij")
    weight_mesh = np.meshgrid(*weight_grids, indexing="ij")

    nodes = np.column_stack([m.ravel() for m in mesh])
    weights = np.prod(np.column_stack([w.ravel() for w in weight_mesh]), axis=1)

    return nodes, weights


def GQN(n: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate normalized Gauss-Hermite quadrature rule for N(0,1).

    Returns quadrature nodes and weights scaled for integration with
    respect to the standard normal distribution. This is a convenience
    wrapper around GHrule with proper scaling.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    nodes : ndarray
        Quadrature nodes of shape (n,).
    weights : ndarray
        Quadrature weights of shape (n,), sum to 1.

    Examples
    --------
    >>> nodes, weights = GQN(5)
    >>> np.sum(weights)  # Should be approximately 1
    1.0

    >>> # Approximate E[X^2] for X ~ N(0,1)
    >>> nodes, weights = GQN(10)
    >>> np.sum(weights * nodes**2)  # Should be approximately 1
    1.0

    Notes
    -----
    For integration of f(x) with respect to the standard normal:
        integral of f(x) * phi(x) dx ≈ sum(weights * f(nodes))

    where phi(x) is the standard normal density.
    """
    nodes, weights = _get_gh_nodes_weights(n)

    nodes = nodes * np.sqrt(2)
    weights = weights / np.sqrt(np.pi)

    return nodes, weights

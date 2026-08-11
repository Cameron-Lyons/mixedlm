from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, special

from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.estimation.reml import (
    _build_lambda,
    _build_theta_bounds,
    _count_theta,
)
from mixedlm.families.base import Family
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure

_ETA_CLIP_MIN = -30.0
_ETA_CLIP_MAX = 30.0
_MU_CLIP_EPS = 1e-7
_MU_CLIP_EPS_STRICT = 1e-10
_WEIGHT_CLIP_MIN = 1e-10
_WEIGHT_CLIP_MAX = 1e10
_DERIV_CLIP_MIN = -1e10
_DERIV_CLIP_MAX = 1e10
_SQRT_WEIGHT_CLIP_MAX = 1e5
_CHOLESKY_REGULARIZATION = 1e-6
_EIGENVALUE_FLOOR = 1e-10
_lambda_cache: dict[tuple[bytes, tuple[tuple[int, int, bool, str], ...]], sparse.csc_matrix] = {}
_LAMBDA_CACHE_MAX_SIZE = 8


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
    from mixedlm._rust import adaptive_gh_deviance as _rust_adaptive_gh_deviance
    from mixedlm._rust import laplace_deviance as _rust_laplace_deviance

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _get_family_name(family: Family) -> str | None:
    class_name = family.__class__.__name__.lower()
    return {
        "binomial": "binomial",
        "gaussian": "gaussian",
        "poisson": "poisson",
    }.get(class_name)


def _get_link_name(family: Family) -> str | None:
    link_name = family.link.__class__.__name__.lower()
    return {
        "identitylink": "identity",
        "loglink": "log",
        "logitlink": "logit",
    }.get(link_name)


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
    p = matrices.n_fixed
    q = matrices.n_random

    prior_weights = matrices.weights
    offset = matrices.offset

    Zt = matrices.Zt

    beta: NDArray[np.floating]
    if beta_start is None:
        beta = np.zeros(p, dtype=np.float64)
        eta = matrices.X @ beta + offset
        mu = family.link.inverse(eta)
        y_work = eta + family.link.deriv(mu) * (matrices.y - mu)
        XtWX = matrices.X.T @ matrices.X
        XtWy = matrices.X.T @ y_work
        try:
            beta = linalg.solve(XtWX, XtWy, assume_a="pos")
        except linalg.LinAlgError:
            beta = linalg.lstsq(matrices.X, y_work)[0]
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
        family.clip_mu(mu, eps=_MU_CLIP_EPS)

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

        delta_beta = np.max(np.abs(beta_new - beta))
        delta_u = np.max(np.abs(spherical_new - spherical)) if q > 0 else 0.0

        beta = beta_new
        spherical = spherical_new

        if delta_beta < tol and delta_u < tol:
            converged = True
            break

    random_effects = np.asarray(Lambda @ spherical).ravel()
    eta = matrices.X @ beta + matrices.Z @ random_effects + offset
    mu = family.link.inverse(eta)
    family.clip_mu(mu, eps=_MU_CLIP_EPS_STRICT)

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
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    q = matrices.n_random

    prior_weights = matrices.weights
    offset = matrices.offset

    if q == 0:
        state = _pirls_state(matrices, family, theta, beta_start, u_start)
        return state.deviance, state.beta, state.random_effects

    state = _pirls_state(matrices, family, theta, beta_start, u_start)
    beta = state.beta
    spherical = state.spherical
    random_effects = state.random_effects
    Lambda = _get_lambda_cached(theta, matrices.random_structures)

    eta_fixed = matrices.X @ beta + offset
    eta = eta_fixed + matrices.Z @ random_effects
    mu = family.link.inverse(eta)
    family.clip_mu(mu, eps=_MU_CLIP_EPS_STRICT)

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

    return float(deviance), beta, random_effects


def _get_gh_nodes_weights(n: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get Gauss-Hermite quadrature nodes and weights."""
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(n)
    return np.asarray(nodes), np.asarray(weights)


def _compute_group_quadrature(
    g: int,
    n_terms_first: int,
    spherical: NDArray[np.floating],
    H: NDArray[np.floating],
    Lambda: sparse.csc_matrix,
    nodes: NDArray[np.floating],
    weights: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    eta_fixed: NDArray[np.floating],
    prior_weights: NDArray[np.floating],
) -> float:
    """Compute quadrature contribution for a single group."""
    sqrt2 = np.sqrt(2.0)

    idx_start = g * n_terms_first
    idx_end = idx_start + n_terms_first

    spherical_mode = spherical[idx_start:idx_end].copy()
    H_block = H[idx_start:idx_end, idx_start:idx_end]

    try:
        L_block = linalg.cholesky(H_block, lower=True)
        scale = 1.0 / L_block[0, 0]
    except linalg.LinAlgError:
        scale = 1.0 / np.sqrt(H_block[0, 0] + _EIGENVALUE_FLOOR)

    group_rows = matrices.Z[:, idx_start:idx_end].getnnz(axis=1) > 0
    if not np.any(group_rows):
        return 0.0

    log_terms = np.empty(len(nodes), dtype=np.float64)
    for i, (node, weight) in enumerate(zip(nodes, weights, strict=False)):
        spherical_block = spherical_mode + sqrt2 * scale * node
        spherical[idx_start:idx_end] = spherical_block
        random_effects = np.asarray(Lambda @ spherical).ravel()
        eta_quad = eta_fixed + matrices.Z @ random_effects
        mu_quad = family.link.inverse(eta_quad)
        family.clip_mu(mu_quad, eps=_MU_CLIP_EPS_STRICT)
        log_lik_y = -0.5 * np.sum(
            family.deviance_resids(
                matrices.y[group_rows], mu_quad[group_rows], prior_weights[group_rows]
            )
        )
        log_prior = -0.5 * np.dot(spherical_block, spherical_block)
        log_terms[i] = np.log(weight) + log_lik_y + log_prior + node**2

    spherical[idx_start:idx_end] = spherical_mode
    return float(np.log(scale) - 0.5 * np.log(np.pi) + special.logsumexp(log_terms))


def adaptive_gh_deviance(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    n_jobs: int = 1,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Compute deviance using adaptive Gauss-Hermite quadrature.

    For nAGQ=1, this is equivalent to the Laplace approximation.
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
    if nAGQ == 1:
        return laplace_deviance(theta, matrices, family, beta_start, u_start)

    q = matrices.n_random
    prior_weights = matrices.weights
    offset = matrices.offset

    if q == 0:
        state = _pirls_state(matrices, family, theta, beta_start, u_start)
        return state.deviance, state.beta, state.random_effects

    first_struct = matrices.random_structures[0]
    if len(matrices.random_structures) != 1 or first_struct.n_terms > 1:
        return laplace_deviance(theta, matrices, family, beta_start, u_start)

    state = _pirls_state(matrices, family, theta, beta_start, u_start)
    beta = state.beta
    spherical = state.spherical
    random_effects = state.random_effects
    Lambda = _get_lambda_cached(theta, matrices.random_structures)

    eta_fixed = matrices.X @ beta + offset
    eta = eta_fixed + matrices.Z @ random_effects
    mu = family.link.inverse(eta)
    family.clip_mu(mu, eps=_MU_CLIP_EPS_STRICT)

    W = np.clip(family.weights(mu) * prior_weights, _WEIGHT_CLIP_MIN, _WEIGHT_CLIP_MAX)

    W_sqrt = np.sqrt(W)
    WZ = _scale_sparse_rows(matrices.Z, W_sqrt)
    ZtWZ = WZ.T @ WZ
    H = _as_dense(Lambda.T @ ZtWZ @ Lambda) + np.eye(q)

    try:
        linalg.cholesky(H, lower=True)
    except linalg.LinAlgError:
        H = H + _CHOLESKY_REGULARIZATION * np.eye(q)

    n_terms_first = first_struct.n_terms
    n_levels_first = first_struct.n_levels

    nodes, weights = _get_gh_nodes_weights(nAGQ)

    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and n_levels_first > 2:
        with ThreadPoolExecutor(max_workers=min(n_jobs, n_levels_first)) as executor:
            futures = [
                executor.submit(
                    _compute_group_quadrature,
                    g,
                    n_terms_first,
                    spherical.copy(),
                    H,
                    Lambda,
                    nodes,
                    weights,
                    matrices,
                    family,
                    eta_fixed,
                    prior_weights,
                )
                for g in range(n_levels_first)
            ]
            log_integral = sum(f.result() for f in futures)
    else:
        log_integral = 0.0
        for g in range(n_levels_first):
            log_integral += _compute_group_quadrature(
                g,
                n_terms_first,
                spherical,
                H,
                Lambda,
                nodes,
                weights,
                matrices,
                family,
                eta_fixed,
                prior_weights,
            )

    deviance = -2.0 * log_integral

    return float(deviance), beta, random_effects


def _laplace_deviance_rust(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    n_levels = [s.n_levels for s in matrices.random_structures]
    n_terms = [s.n_terms for s in matrices.random_structures]
    correlated = [s.correlated for s in matrices.random_structures]

    z_csc = matrices.Z.tocsc()

    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if family_name is None or link_name is None:
        raise ValueError("Family and link combination is not supported by the Rust backend")

    deviance, beta, u = _rust_laplace_deviance(
        np.ascontiguousarray(matrices.y, dtype=np.float64),
        np.ascontiguousarray(matrices.X, dtype=np.float64),
        np.ascontiguousarray(z_csc.data, dtype=np.float64),
        np.ascontiguousarray(z_csc.indices, dtype=np.int64),
        np.ascontiguousarray(z_csc.indptr, dtype=np.int64),
        (z_csc.shape[0], z_csc.shape[1]),
        np.ascontiguousarray(matrices.weights, dtype=np.float64),
        np.ascontiguousarray(matrices.offset, dtype=np.float64),
        np.ascontiguousarray(theta, dtype=np.float64),
        n_levels,
        n_terms,
        correlated,
        family_name,
        link_name,
    )

    return deviance, np.array(beta), np.array(u)


def _adaptive_gh_deviance_rust(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    n_levels = [s.n_levels for s in matrices.random_structures]
    n_terms = [s.n_terms for s in matrices.random_structures]
    correlated = [s.correlated for s in matrices.random_structures]

    z_csc = matrices.Z.tocsc()

    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if family_name is None or link_name is None:
        raise ValueError("Family and link combination is not supported by the Rust backend")

    deviance, beta, u = _rust_adaptive_gh_deviance(
        np.ascontiguousarray(matrices.y, dtype=np.float64),
        np.ascontiguousarray(matrices.X, dtype=np.float64),
        np.ascontiguousarray(z_csc.data, dtype=np.float64),
        np.ascontiguousarray(z_csc.indices, dtype=np.int64),
        np.ascontiguousarray(z_csc.indptr, dtype=np.int64),
        (z_csc.shape[0], z_csc.shape[1]),
        np.ascontiguousarray(matrices.weights, dtype=np.float64),
        np.ascontiguousarray(matrices.offset, dtype=np.float64),
        np.ascontiguousarray(theta, dtype=np.float64),
        n_levels,
        n_terms,
        correlated,
        family_name,
        link_name,
        nAGQ,
    )

    return deviance, np.array(beta), np.array(u)


def laplace_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if (
        _HAS_RUST
        and beta_start is None
        and u_start is None
        and family_name is not None
        and link_name is not None
        and _native_covariance_supported(matrices)
    ):
        return _laplace_deviance_rust(theta, matrices, family)
    return laplace_deviance(theta, matrices, family, beta_start, u_start)


def adaptive_gh_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    if nAGQ == 1:
        return laplace_deviance_fast(theta, matrices, family, beta_start, u_start)

    family_name = _get_family_name(family)
    link_name = _get_link_name(family)
    if (
        _HAS_RUST
        and beta_start is None
        and u_start is None
        and family_name is not None
        and link_name is not None
        and _native_covariance_supported(matrices)
    ):
        first_struct = matrices.random_structures[0] if matrices.random_structures else None
        if first_struct and first_struct.n_terms == 1:
            return _adaptive_gh_deviance_rust(theta, matrices, family, nAGQ)

    return adaptive_gh_deviance(theta, matrices, family, nAGQ, beta_start, u_start)


class GLMMOptimizer:
    def __init__(
        self,
        matrices: ModelMatrices,
        family: Family,
        verbose: int = 0,
        nAGQ: int = 1,
    ) -> None:
        self.matrices = matrices
        self.family = family
        self.verbose = verbose
        self.nAGQ = nAGQ
        self.n_theta = _count_theta(matrices.random_structures)

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
        if self.nAGQ > 1:
            dev, _, _ = adaptive_gh_deviance_fast(
                theta,
                self.matrices,
                self.family,
                nAGQ=self.nAGQ,
            )
        else:
            dev, _, _ = laplace_deviance_fast(theta, self.matrices, self.family)
        return dev

    def optimize(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        options: dict[str, Any] | None = None,
    ) -> GLMMOptimizationResult:
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
        )

        theta_opt = result.x

        if self.nAGQ > 1:
            final_dev, beta, u = adaptive_gh_deviance_fast(
                theta_opt,
                self.matrices,
                self.family,
                nAGQ=self.nAGQ,
            )
        else:
            final_dev, beta, u = laplace_deviance_fast(theta_opt, self.matrices, self.family)

        return GLMMOptimizationResult(
            theta=theta_opt,
            beta=beta,
            u=u,
            deviance=final_dev,
            converged=result.success,
            n_iter=result.nit,
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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.estimation.reml import (
    _build_lambda,
    _count_theta,
)
from mixedlm.families.base import Family
from mixedlm.matrices.design import ModelMatrices

try:
    from mixedlm._rust import laplace_deviance as _rust_laplace_deviance

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _get_family_name(family: Family) -> str:
    class_name = family.__class__.__name__.lower()
    if "binomial" in class_name:
        return "binomial"
    elif "poisson" in class_name:
        return "poisson"
    elif "gaussian" in class_name:
        return "gaussian"
    return "gaussian"


def _get_link_name(family: Family) -> str:
    link_name = family.link.__class__.__name__.lower()
    if "logit" in link_name:
        return "logit"
    elif "log" in link_name and "logit" not in link_name:
        return "log"
    elif "identity" in link_name:
        return "identity"
    return "identity"


@dataclass
class GLMMOptimizationResult:
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    u: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int


def pirls(
    matrices: ModelMatrices,
    family: Family,
    theta: NDArray[np.floating],
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    maxiter: int = 25,
    tol: float = 1e-6,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, bool]:
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
        beta = beta_start.copy()

    u = np.zeros(q, dtype=np.float64) if u_start is None else u_start.copy()

    Lambda = _build_lambda(theta, matrices.random_structures)

    LambdatLambda = Lambda.T @ Lambda
    if sparse.issparse(LambdatLambda):
        LambdatLambda = LambdatLambda.toarray()

    converged = False
    for _iteration in range(maxiter):
        eta = matrices.X @ beta + matrices.Z @ u + offset
        mu = family.link.inverse(eta)

        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = family.weights(mu) * prior_weights
        W = np.maximum(W, 1e-10)

        z = eta - offset + family.link.deriv(mu) * (matrices.y - mu)

        W_diag = sparse.diags(W, format="csc")

        XtWX = matrices.X.T @ W_diag @ matrices.X
        XtWZ = matrices.X.T @ W_diag @ matrices.Z
        ZtWZ = Zt @ W_diag @ matrices.Z

        XtWz = matrices.X.T @ (W * z)
        ZtWz = Zt @ (W * z)

        ZtWZ_dense = ZtWZ.toarray() if sparse.issparse(ZtWZ) else ZtWZ

        C = ZtWZ_dense + LambdatLambda

        try:
            L_C = linalg.cholesky(C, lower=True)
        except linalg.LinAlgError:
            C += 1e-6 * np.eye(q)
            L_C = linalg.cholesky(C, lower=True)

        ZtWX_dense = XtWZ.T.toarray() if sparse.issparse(XtWZ) else XtWZ.T

        RZX = linalg.solve_triangular(L_C, ZtWX_dense, lower=True)
        cu = linalg.solve_triangular(L_C, ZtWz, lower=True)

        XtVinvX = XtWX - RZX.T @ RZX
        XtVinvz = XtWz - RZX.T @ cu

        try:
            beta_new = linalg.solve(XtVinvX, XtVinvz, assume_a="pos")
        except linalg.LinAlgError:
            beta_new = linalg.lstsq(XtVinvX, XtVinvz)[0]

        u_rhs = ZtWz - ZtWX_dense @ beta_new
        u_new = linalg.cho_solve((L_C, True), u_rhs)

        delta_beta = np.max(np.abs(beta_new - beta))
        delta_u = np.max(np.abs(u_new - u)) if q > 0 else 0.0

        beta = beta_new
        u = u_new

        if delta_beta < tol and delta_u < tol:
            converged = True
            break

    eta = matrices.X @ beta + matrices.Z @ u + offset
    mu = family.link.inverse(eta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    dev_resids = family.deviance_resids(matrices.y, mu, prior_weights)
    deviance = np.sum(dev_resids)

    deviance += np.dot(u, linalg.solve(LambdatLambda + 1e-10 * np.eye(q), u))

    return beta, u, deviance, converged


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
        beta, _, deviance, _ = pirls(matrices, family, theta, beta_start, u_start)
        return deviance, beta, np.array([])

    beta, u, _, _ = pirls(matrices, family, theta, beta_start, u_start)

    Lambda = _build_lambda(theta, matrices.random_structures)

    eta = matrices.X @ beta + matrices.Z @ u + offset
    mu = family.link.inverse(eta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    dev_resids = family.deviance_resids(matrices.y, mu, prior_weights)
    deviance = np.sum(dev_resids)

    LambdatLambda = Lambda.T @ Lambda
    if sparse.issparse(LambdatLambda):
        LambdatLambda = LambdatLambda.toarray()

    u_penalty = np.dot(u, linalg.solve(LambdatLambda + 1e-10 * np.eye(q), u))
    deviance += u_penalty

    W = family.weights(mu) * prior_weights
    W = np.maximum(W, 1e-10)
    W_diag = sparse.diags(W, format="csc")

    Zt = matrices.Zt
    ZtWZ = Zt @ W_diag @ matrices.Z
    if sparse.issparse(ZtWZ):
        ZtWZ = ZtWZ.toarray()

    H = ZtWZ + LambdatLambda

    try:
        L_H = linalg.cholesky(H, lower=True)
        logdet_H = 2.0 * np.sum(np.log(np.diag(L_H)))
    except linalg.LinAlgError:
        eigvals = linalg.eigvalsh(H)
        eigvals = np.maximum(eigvals, 1e-10)
        logdet_H = np.sum(np.log(eigvals))

    deviance += logdet_H

    return deviance, beta, u


def _get_gh_nodes_weights(n: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get Gauss-Hermite quadrature nodes and weights.

    Returns nodes and weights for integrating functions of the form
    f(x) * exp(-x^2) over (-inf, inf).
    """
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(n)
    return np.array(nodes), np.array(weights)


def adaptive_gh_deviance(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    nAGQ: int = 1,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Compute deviance using adaptive Gauss-Hermite quadrature.

    For nAGQ=1, this is equivalent to the Laplace approximation.
    For nAGQ>1, uses adaptive GH quadrature for more accurate integration
    over the random effects.

    Parameters
    ----------
    theta : NDArray
        Variance component parameters.
    matrices : ModelMatrices
        Model design matrices.
    family : Family
        GLM family with link function.
    nAGQ : int, default 1
        Number of quadrature points. 1 gives Laplace approximation.
    beta_start : NDArray, optional
        Starting values for fixed effects.
    u_start : NDArray, optional
        Starting values for random effects.

    Returns
    -------
    deviance : float
        -2 * log-likelihood approximation.
    beta : NDArray
        Fixed effect estimates.
    u : NDArray
        Random effect estimates (conditional modes).

    Notes
    -----
    Adaptive GH quadrature centers the quadrature nodes at the conditional
    modes and scales them by the Cholesky factor of the conditional variance.
    This works well for scalar random effects or models with a single random
    effect per group.

    For models with crossed or nested random effects with multiple terms per
    grouping factor, the integration becomes high-dimensional and nAGQ > 1
    may not provide benefits (the integral is approximated group-by-group
    for the first grouping factor only, matching lme4's behavior).
    """
    if nAGQ == 1:
        return laplace_deviance(theta, matrices, family, beta_start, u_start)

    q = matrices.n_random
    prior_weights = matrices.weights
    offset = matrices.offset

    if q == 0:
        beta, _, deviance, _ = pirls(matrices, family, theta, beta_start, u_start)
        return deviance, beta, np.array([])

    beta, u, _, _ = pirls(matrices, family, theta, beta_start, u_start)

    Lambda = _build_lambda(theta, matrices.random_structures)
    LambdatLambda = Lambda.T @ Lambda
    if sparse.issparse(LambdatLambda):
        LambdatLambda = LambdatLambda.toarray()

    eta = matrices.X @ beta + matrices.Z @ u + offset
    mu = family.link.inverse(eta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    W = family.weights(mu) * prior_weights
    W = np.maximum(W, 1e-10)
    W_diag = sparse.diags(W, format="csc")

    Zt = matrices.Zt
    ZtWZ = Zt @ W_diag @ matrices.Z
    if sparse.issparse(ZtWZ):
        ZtWZ = ZtWZ.toarray()

    H = ZtWZ + LambdatLambda

    try:
        linalg.cholesky(H, lower=True)
    except linalg.LinAlgError:
        H = H + 1e-6 * np.eye(q)

    first_struct = matrices.random_structures[0]
    n_terms_first = first_struct.n_terms
    n_levels_first = first_struct.n_levels

    if n_terms_first > 1:
        return laplace_deviance(theta, matrices, family, beta_start, u_start)

    nodes, weights = _get_gh_nodes_weights(nAGQ)
    sqrt2 = np.sqrt(2.0)

    log_integral = 0.0

    for g in range(n_levels_first):
        idx_start = g * n_terms_first
        idx_end = idx_start + n_terms_first

        u_mode = u[idx_start:idx_end]
        H_block = H[idx_start:idx_end, idx_start:idx_end]

        try:
            L_block = linalg.cholesky(H_block, lower=True)
            scale = 1.0 / L_block[0, 0]
        except linalg.LinAlgError:
            scale = 1.0 / np.sqrt(H_block[0, 0] + 1e-10)

        group_contrib = 0.0

        for _k, (node, weight) in enumerate(zip(nodes, weights, strict=False)):
            u_quad = u.copy()
            u_quad[idx_start:idx_end] = u_mode + sqrt2 * scale * node

            eta_quad = matrices.X @ beta + matrices.Z @ u_quad + offset
            mu_quad = family.link.inverse(eta_quad)
            mu_quad = np.clip(mu_quad, 1e-10, 1 - 1e-10)

            log_lik_y = -0.5 * np.sum(family.deviance_resids(matrices.y, mu_quad, prior_weights))

            u_block = u_quad[idx_start:idx_end]
            Lambda_block = LambdatLambda[idx_start:idx_end, idx_start:idx_end]
            log_prior = -0.5 * np.dot(
                u_block, linalg.solve(Lambda_block + 1e-10 * np.eye(n_terms_first), u_block)
            )

            integrand = np.exp(log_lik_y + log_prior)
            group_contrib += weight * integrand

        group_contrib *= scale * np.sqrt(np.pi)
        log_integral += np.log(max(group_contrib, 1e-300))

    other_u_start = first_struct.n_levels * first_struct.n_terms
    if other_u_start < q:
        u_other = u[other_u_start:]
        Lambda_other = LambdatLambda[other_u_start:, other_u_start:]
        u_penalty_other = np.dot(
            u_other,
            linalg.solve(Lambda_other + 1e-10 * np.eye(q - other_u_start), u_other),
        )
        H_other = H[other_u_start:, other_u_start:]
        try:
            L_H_other = linalg.cholesky(H_other, lower=True)
            logdet_other = 2.0 * np.sum(np.log(np.diag(L_H_other)))
        except linalg.LinAlgError:
            eigvals = linalg.eigvalsh(H_other)
            eigvals = np.maximum(eigvals, 1e-10)
            logdet_other = np.sum(np.log(eigvals))
        log_integral -= 0.5 * (u_penalty_other + logdet_other)

    deviance = -2.0 * log_integral

    return deviance, beta, u


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

    deviance, beta, u = _rust_laplace_deviance(
        matrices.y,
        matrices.X,
        z_csc.data,
        z_csc.indices.astype(np.int64),
        z_csc.indptr.astype(np.int64),
        (z_csc.shape[0], z_csc.shape[1]),
        matrices.weights,
        matrices.offset,
        theta,
        n_levels,
        n_terms,
        correlated,
        family_name,
        link_name,
    )

    return deviance, np.array(beta), np.array(u)


def laplace_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    family: Family,
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    if _HAS_RUST and beta_start is None and u_start is None:
        family_name = _get_family_name(family)
        if family_name in ("binomial", "poisson", "gaussian"):
            return _laplace_deviance_rust(theta, matrices, family)
    return laplace_deviance(theta, matrices, family, beta_start, u_start)


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
        self._beta_cache: NDArray[np.floating] | None = None
        self._u_cache: NDArray[np.floating] | None = None

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
                        if i == j:
                            theta_list.append(1.0)
                        else:
                            theta_list.append(0.0)
            else:
                theta_list.extend([1.0] * q)
        return np.array(theta_list, dtype=np.float64)

    def objective(self, theta: NDArray[np.floating]) -> float:
        if self.nAGQ > 1:
            dev, beta, u = adaptive_gh_deviance(
                theta,
                self.matrices,
                self.family,
                nAGQ=self.nAGQ,
                beta_start=self._beta_cache,
                u_start=self._u_cache,
            )
        else:
            dev, beta, u = laplace_deviance(
                theta, self.matrices, self.family, self._beta_cache, self._u_cache
            )
        self._beta_cache = beta
        self._u_cache = u
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

        self._beta_cache = None
        self._u_cache = None

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * len(start)
        idx = 0
        for struct in self.matrices.random_structures:
            q = struct.n_terms
            cov_type = getattr(struct, "cov_type", "us")
            if cov_type == "cs":
                bounds[idx] = (0.0, None)
                idx += 1
                if q > 1:
                    bounds[idx] = (-1.0 / (q - 1) + 1e-6, 1.0 - 1e-6)
                    idx += 1
            elif cov_type == "ar1":
                bounds[idx] = (0.0, None)
                idx += 1
                if q > 1:
                    bounds[idx] = (-1.0 + 1e-6, 1.0 - 1e-6)
                    idx += 1
            elif struct.correlated:
                for i in range(q):
                    for j in range(i + 1):
                        if i == j:
                            bounds[idx] = (0.0, None)
                        idx += 1
            else:
                for _ in range(q):
                    bounds[idx] = (0.0, None)
                    idx += 1

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
            final_dev, beta, u = adaptive_gh_deviance(
                theta_opt,
                self.matrices,
                self.family,
                nAGQ=self.nAGQ,
                beta_start=self._beta_cache,
                u_start=self._u_cache,
            )
        else:
            final_dev, beta, u = laplace_deviance(
                theta_opt, self.matrices, self.family, self._beta_cache, self._u_cache
            )

        return GLMMOptimizationResult(
            theta=theta_opt,
            beta=beta,
            u=u,
            deviance=final_dev,
            converged=result.success,
            n_iter=result.nit,
        )

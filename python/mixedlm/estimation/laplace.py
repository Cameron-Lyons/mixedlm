from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.optimize import minimize

from mixedlm.families.base import Family
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure


@dataclass
class GLMMOptimizationResult:
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    u: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int


def _build_lambda(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
) -> sparse.csc_matrix:
    blocks: list[sparse.csc_matrix] = []
    theta_idx = 0

    for struct in structures:
        q = struct.n_terms
        n_levels = struct.n_levels

        if struct.correlated:
            n_theta = q * (q + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta

            L_block = np.zeros((q, q), dtype=np.float64)
            idx = 0
            for i in range(q):
                for j in range(i + 1):
                    L_block[i, j] = theta_block[idx]
                    idx += 1

            block = sparse.kron(
                sparse.eye(n_levels, format="csc"),
                sparse.csc_matrix(L_block),
            )
        else:
            theta_block = theta[theta_idx : theta_idx + q]
            theta_idx += q

            L_diag = np.diag(theta_block)
            block = sparse.kron(
                sparse.eye(n_levels, format="csc"),
                sparse.csc_matrix(L_diag),
            )

        blocks.append(block)

    if not blocks:
        return sparse.csc_matrix((0, 0), dtype=np.float64)

    return sparse.block_diag(blocks, format="csc")


def _count_theta(structures: list[RandomEffectStructure]) -> int:
    count = 0
    for struct in structures:
        q = struct.n_terms
        if struct.correlated:
            count += q * (q + 1) // 2
        else:
            count += q
    return count


def pirls(
    matrices: ModelMatrices,
    family: Family,
    theta: NDArray[np.floating],
    beta_start: NDArray[np.floating] | None = None,
    u_start: NDArray[np.floating] | None = None,
    maxiter: int = 25,
    tol: float = 1e-6,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, bool]:
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    if beta_start is None:
        beta = np.zeros(p, dtype=np.float64)
        eta = matrices.X @ beta
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

    if u_start is None:
        u = np.zeros(q, dtype=np.float64)
    else:
        u = u_start.copy()

    Lambda = _build_lambda(theta, matrices.random_structures)

    converged = False
    for iteration in range(maxiter):
        eta = matrices.X @ beta + matrices.Z @ u
        mu = family.link.inverse(eta)

        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = family.weights(mu)
        W = np.maximum(W, 1e-10)

        z = eta + family.link.deriv(mu) * (matrices.y - mu)

        W_diag = sparse.diags(W, format="csc")
        sqrt_W = sparse.diags(np.sqrt(W), format="csc")

        XtWX = matrices.X.T @ W_diag @ matrices.X
        XtWZ = matrices.X.T @ W_diag @ matrices.Z
        ZtWX = XtWZ.T
        ZtWZ = matrices.Z.T @ W_diag @ matrices.Z

        LambdatLambda = Lambda.T @ Lambda
        if sparse.issparse(LambdatLambda):
            LambdatLambda = LambdatLambda.toarray()

        XtWz = matrices.X.T @ (W * z)
        ZtWz = matrices.Z.T @ (W * z)

        if sparse.issparse(ZtWZ):
            ZtWZ_dense = ZtWZ.toarray()
        else:
            ZtWZ_dense = ZtWZ

        C = ZtWZ_dense + LambdatLambda

        try:
            L_C = linalg.cholesky(C, lower=True)
        except linalg.LinAlgError:
            C += 1e-6 * np.eye(q)
            L_C = linalg.cholesky(C, lower=True)

        if sparse.issparse(ZtWX):
            ZtWX_dense = ZtWX.toarray()
        else:
            ZtWX_dense = ZtWX

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

    eta = matrices.X @ beta + matrices.Z @ u
    mu = family.link.inverse(eta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    dev_resids = family.deviance_resids(matrices.y, mu, np.ones(n))
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
    n = matrices.n_obs
    q = matrices.n_random

    if q == 0:
        beta, _, deviance, _ = pirls(matrices, family, theta, beta_start, u_start)
        return deviance, beta, np.array([])

    beta, u, _, converged = pirls(matrices, family, theta, beta_start, u_start)

    Lambda = _build_lambda(theta, matrices.random_structures)

    eta = matrices.X @ beta + matrices.Z @ u
    mu = family.link.inverse(eta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    dev_resids = family.deviance_resids(matrices.y, mu, np.ones(n))
    deviance = np.sum(dev_resids)

    LambdatLambda = Lambda.T @ Lambda
    if sparse.issparse(LambdatLambda):
        LambdatLambda = LambdatLambda.toarray()

    u_penalty = np.dot(u, linalg.solve(LambdatLambda + 1e-10 * np.eye(q), u))
    deviance += u_penalty

    W = family.weights(mu)
    W = np.maximum(W, 1e-10)
    W_diag = sparse.diags(W, format="csc")

    ZtWZ = matrices.Z.T @ W_diag @ matrices.Z
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


class GLMMOptimizer:
    def __init__(
        self,
        matrices: ModelMatrices,
        family: Family,
        verbose: int = 0,
    ) -> None:
        self.matrices = matrices
        self.family = family
        self.verbose = verbose
        self.n_theta = _count_theta(matrices.random_structures)
        self._beta_cache: NDArray[np.floating] | None = None
        self._u_cache: NDArray[np.floating] | None = None

    def get_start_theta(self) -> NDArray[np.floating]:
        return np.ones(self.n_theta, dtype=np.float64)

    def objective(self, theta: NDArray[np.floating]) -> float:
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
    ) -> GLMMOptimizationResult:
        if start is None:
            start = self.get_start_theta()

        self._beta_cache = None
        self._u_cache = None

        bounds = [(None, None)] * len(start)
        idx = 0
        for struct in self.matrices.random_structures:
            q = struct.n_terms
            if struct.correlated:
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

        result = minimize(
            self.objective,
            start,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter},
            callback=callback,
        )

        theta_opt = result.x

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

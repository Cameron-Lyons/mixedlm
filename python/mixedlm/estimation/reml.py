from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.optimize import minimize

from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure


@dataclass
class OptimizationResult:
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    sigma: float
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


def profiled_deviance(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> float:
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    w = matrices.weights
    y_adj = matrices.y - matrices.offset

    sqrtW = sparse.diags(np.sqrt(w), format="csc")

    if q == 0:
        WX = sqrtW @ matrices.X
        Wy = sqrtW @ y_adj
        XtWX = WX.T @ WX
        XtWy = WX.T @ Wy
        try:
            beta = linalg.solve(XtWX, XtWy, assume_a="pos")
        except linalg.LinAlgError:
            beta = linalg.lstsq(WX, Wy)[0]

        resid = y_adj - matrices.X @ beta
        wrss = np.dot(w * resid, resid)
        sigma2 = wrss / (n - p if REML else n)

        logdet_XtWX = np.linalg.slogdet(XtWX)[1] if REML else 0.0
        dev = n * np.log(2 * np.pi * sigma2) + wrss / sigma2
        if REML:
            dev += logdet_XtWX - p * np.log(sigma2)
        return float(dev)

    Lambda = _build_lambda(theta, matrices.random_structures)

    Zt = matrices.Zt
    WZ = sqrtW @ matrices.Z
    ZtWZ = WZ.T @ WZ
    LambdatZtWZLambda = Lambda.T @ ZtWZ @ Lambda

    I_q = sparse.eye(q, format="csc")
    V_factor = LambdatZtWZLambda + I_q

    try:
        V_factor_dense = V_factor.toarray()
        L_V = linalg.cholesky(V_factor_dense, lower=True)
    except linalg.LinAlgError:
        return 1e10

    logdet_V = 2.0 * np.sum(np.log(np.diag(L_V)))

    ZtWy = Zt @ (w * y_adj)
    cu = Lambda.T @ ZtWy
    cu_star = linalg.solve_triangular(L_V, cu, lower=True)

    WX = sqrtW @ matrices.X
    ZtWX = Zt @ (sqrtW.T @ WX)
    Lambdat_ZtWX = Lambda.T @ ZtWX
    RZX = linalg.solve_triangular(L_V, Lambdat_ZtWX, lower=True)

    XtWX = WX.T @ WX
    XtWy = WX.T @ (sqrtW @ y_adj)

    RZX_tRZX = RZX.T @ RZX
    XtVinvX = XtWX - RZX_tRZX

    try:
        L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)
    except linalg.LinAlgError:
        return 1e10

    logdet_XtVinvX = 2.0 * np.sum(np.log(np.diag(L_XtVinvX)))

    cu_star_RZX_beta_term = RZX.T @ cu_star
    Xty_adj = XtWy - cu_star_RZX_beta_term
    beta = linalg.cho_solve((L_XtVinvX, True), Xty_adj)

    resid = y_adj - matrices.X @ beta
    Zt_resid = Zt @ (w * resid)
    Lambda_t_Zt_resid = Lambda.T @ Zt_resid
    u_star = linalg.cho_solve((L_V, True), Lambda_t_Zt_resid)

    pwrss = np.dot(w * resid, resid) + np.dot(u_star, u_star)

    denom = n - p if REML else n

    sigma2 = pwrss / denom

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_V
    if REML:
        dev += logdet_XtVinvX

    return float(dev)


def profiled_reml(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
) -> float:
    return profiled_deviance(theta, matrices, REML=True)


class LMMOptimizer:
    def __init__(
        self,
        matrices: ModelMatrices,
        REML: bool = True,
        verbose: int = 0,
    ) -> None:
        self.matrices = matrices
        self.REML = REML
        self.verbose = verbose
        self.n_theta = _count_theta(matrices.random_structures)

    def get_start_theta(self) -> NDArray[np.floating]:
        theta = np.ones(self.n_theta, dtype=np.float64)
        return theta

    def objective(self, theta: NDArray[np.floating]) -> float:
        return profiled_deviance(theta, self.matrices, self.REML)

    def optimize(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
    ) -> OptimizationResult:
        if start is None:
            start = self.get_start_theta()

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
        beta, sigma, u = self._extract_estimates(theta_opt)

        return OptimizationResult(
            theta=theta_opt,
            beta=beta,
            sigma=sigma,
            u=u,
            deviance=result.fun,
            converged=result.success,
            n_iter=result.nit,
        )

    def _extract_estimates(
        self, theta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, NDArray[np.floating]]:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        q = self.matrices.n_random

        w = self.matrices.weights
        y_adj = self.matrices.y - self.matrices.offset
        sqrtW = sparse.diags(np.sqrt(w), format="csc")

        if q == 0:
            WX = sqrtW @ self.matrices.X
            Wy = sqrtW @ y_adj
            XtWX = WX.T @ WX
            XtWy = WX.T @ Wy
            try:
                beta = linalg.solve(XtWX, XtWy, assume_a="pos")
            except linalg.LinAlgError:
                beta = linalg.lstsq(WX, Wy)[0]
            resid = y_adj - self.matrices.X @ beta
            wrss = np.dot(w * resid, resid)
            sigma = np.sqrt(wrss / (n - p if self.REML else n))
            return beta, sigma, np.array([])

        Lambda = _build_lambda(theta, self.matrices.random_structures)

        Zt = self.matrices.Zt
        WZ = sqrtW @ self.matrices.Z
        ZtWZ = WZ.T @ WZ
        LambdatZtWZLambda = Lambda.T @ ZtWZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V_factor = LambdatZtWZLambda + I_q

        V_factor_dense = V_factor.toarray()
        L_V = linalg.cholesky(V_factor_dense, lower=True)

        ZtWy = Zt @ (w * y_adj)
        cu = Lambda.T @ ZtWy
        cu_star = linalg.solve_triangular(L_V, cu, lower=True)

        WX = sqrtW @ self.matrices.X
        ZtWX = Zt @ (sqrtW.T @ WX)
        Lambdat_ZtWX = Lambda.T @ ZtWX
        RZX = linalg.solve_triangular(L_V, Lambdat_ZtWX, lower=True)

        XtWX = WX.T @ WX
        XtWy = WX.T @ (sqrtW @ y_adj)

        RZX_tRZX = RZX.T @ RZX
        XtVinvX = XtWX - RZX_tRZX
        L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)

        cu_star_RZX_beta_term = RZX.T @ cu_star
        Xty_adj = XtWy - cu_star_RZX_beta_term
        beta = linalg.cho_solve((L_XtVinvX, True), Xty_adj)

        resid = y_adj - self.matrices.X @ beta
        Zt_resid = Zt @ (w * resid)
        Lambda_t_Zt_resid = Lambda.T @ Zt_resid
        u_star = linalg.cho_solve((L_V, True), Lambda_t_Zt_resid)

        u = Lambda @ u_star

        pwrss = np.dot(w * resid, resid) + np.dot(u_star, u_star)
        denom = n - p if self.REML else n
        sigma = np.sqrt(pwrss / denom)

        return beta, sigma, u

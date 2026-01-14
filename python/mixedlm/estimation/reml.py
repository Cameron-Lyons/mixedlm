from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure

try:
    from mixedlm._rust import profiled_deviance as _rust_profiled_deviance

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class OptimizationResult:
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    sigma: float
    u: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int


@dataclass
class DevianceComponents:
    """Components of the deviance calculation for linear mixed models.

    This class provides a breakdown of the deviance into its constituent
    parts, useful for diagnostics and understanding model fit.
    """

    total: float
    ldL2: float
    ldRX2: float
    wrss: float
    ussq: float
    pwrss: float
    sigma2: float
    REML: bool

    def __str__(self) -> str:
        lines = []
        lines.append("Deviance Components:")
        lines.append(f"  Total deviance:     {self.total:.4f}")
        lines.append(f"  log|L|^2 (ldL2):    {self.ldL2:.4f}")
        lines.append(f"  log|RX|^2 (ldRX2):  {self.ldRX2:.4f}")
        lines.append(f"  WRSS:               {self.wrss:.4f}")
        lines.append(f"  u'u (ussq):         {self.ussq:.4f}")
        lines.append(f"  PWRSS:              {self.pwrss:.4f}")
        lines.append(f"  sigma^2:            {self.sigma2:.4f}")
        lines.append(f"  REML:               {self.REML}")
        return "\n".join(lines)


def _build_cs_cholesky(q: int, rho: float) -> NDArray[np.floating]:
    """Build Cholesky factor for compound symmetry correlation matrix."""
    if q == 1:
        return np.array([[1.0]])
    R = np.full((q, q), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    try:
        return linalg.cholesky(R, lower=True)
    except linalg.LinAlgError:
        rho_safe = np.clip(rho, -1.0 / (q - 1) + 1e-6, 1.0 - 1e-6)
        R = np.full((q, q), rho_safe, dtype=np.float64)
        np.fill_diagonal(R, 1.0)
        return linalg.cholesky(R, lower=True)


def _build_ar1_cholesky(q: int, rho: float) -> NDArray[np.floating]:
    """Build Cholesky factor for AR(1) correlation matrix."""
    if q == 1:
        return np.array([[1.0]])
    R = np.zeros((q, q), dtype=np.float64)
    for i in range(q):
        for j in range(q):
            R[i, j] = rho ** abs(i - j)
    try:
        return linalg.cholesky(R, lower=True)
    except linalg.LinAlgError:
        rho_safe = np.clip(rho, -1.0 + 1e-6, 1.0 - 1e-6)
        for i in range(q):
            for j in range(q):
                R[i, j] = rho_safe ** abs(i - j)
        return linalg.cholesky(R, lower=True)


def _build_lambda(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
) -> sparse.csc_matrix:
    blocks: list[sparse.csc_matrix] = []
    theta_idx = 0

    for struct in structures:
        q = struct.n_terms
        n_levels = struct.n_levels
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type == "cs":
            sigma_rel = theta[theta_idx]
            rho = theta[theta_idx + 1] if q > 1 else 0.0
            theta_idx += 2 if q > 1 else 1
            L_corr = _build_cs_cholesky(q, rho)
            L_block = sigma_rel * L_corr
            block = sparse.kron(
                sparse.eye(n_levels, format="csc"),
                sparse.csc_matrix(L_block),
            )
        elif cov_type == "ar1":
            sigma_rel = theta[theta_idx]
            rho = theta[theta_idx + 1] if q > 1 else 0.0
            theta_idx += 2 if q > 1 else 1
            L_corr = _build_ar1_cholesky(q, rho)
            L_block = sigma_rel * L_corr
            block = sparse.kron(
                sparse.eye(n_levels, format="csc"),
                sparse.csc_matrix(L_block),
            )
        elif struct.correlated:
            n_theta = q * (q + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta

            L_block = np.zeros((q, q), dtype=np.float64)
            row_indices, col_indices = np.tril_indices(q)
            L_block[row_indices, col_indices] = theta_block

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
        cov_type = getattr(struct, "cov_type", "us")
        if cov_type == "cs" or cov_type == "ar1":
            count += 2 if q > 1 else 1
        elif struct.correlated:
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


def profiled_deviance_components(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> DevianceComponents:
    """Compute deviance and return all components.

    This function returns the same deviance as profiled_deviance() but also
    returns the individual components for diagnostic purposes.

    Parameters
    ----------
    theta : NDArray
        Variance component parameters.
    matrices : ModelMatrices
        Model design matrices.
    REML : bool, default True
        Whether to compute REML (True) or ML (False) deviance.

    Returns
    -------
    DevianceComponents
        Object containing deviance and all component values.
    """
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
        denom = n - p if REML else n
        sigma2 = wrss / denom

        ldRX2 = np.linalg.slogdet(XtWX)[1] if REML else 0.0
        dev = n * np.log(2 * np.pi * sigma2) + wrss / sigma2
        if REML:
            dev += ldRX2 - p * np.log(sigma2)

        return DevianceComponents(
            total=float(dev),
            ldL2=0.0,
            ldRX2=float(ldRX2),
            wrss=float(wrss),
            ussq=0.0,
            pwrss=float(wrss),
            sigma2=float(sigma2),
            REML=REML,
        )

    Lambda = _build_lambda(theta, matrices.random_structures)

    Zt = matrices.Zt
    WZ = sqrtW @ matrices.Z
    ZtWZ = WZ.T @ WZ
    LambdatZtWZLambda = Lambda.T @ ZtWZ @ Lambda

    I_q = sparse.eye(q, format="csc")
    V_factor = LambdatZtWZLambda + I_q

    V_factor_dense = V_factor.toarray()
    L_V = linalg.cholesky(V_factor_dense, lower=True)

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

    L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)
    logdet_XtVinvX = 2.0 * np.sum(np.log(np.diag(L_XtVinvX)))

    cu_star_RZX_beta_term = RZX.T @ cu_star
    Xty_adj = XtWy - cu_star_RZX_beta_term
    beta = linalg.cho_solve((L_XtVinvX, True), Xty_adj)

    resid = y_adj - matrices.X @ beta
    wrss = np.dot(w * resid, resid)

    Zt_resid = Zt @ (w * resid)
    Lambda_t_Zt_resid = Lambda.T @ Zt_resid
    u_star = linalg.cho_solve((L_V, True), Lambda_t_Zt_resid)

    ussq = np.dot(u_star, u_star)
    pwrss = wrss + ussq

    denom = n - p if REML else n
    sigma2 = pwrss / denom

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_V
    if REML:
        dev += logdet_XtVinvX

    return DevianceComponents(
        total=float(dev),
        ldL2=float(logdet_V),
        ldRX2=float(logdet_XtVinvX) if REML else 0.0,
        wrss=float(wrss),
        ussq=float(ussq),
        pwrss=float(pwrss),
        sigma2=float(sigma2),
        REML=REML,
    )


def _profiled_deviance_rust(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> float:
    n_levels = [s.n_levels for s in matrices.random_structures]
    n_terms = [s.n_terms for s in matrices.random_structures]
    correlated = [s.correlated for s in matrices.random_structures]

    z_csc = matrices.Z.tocsc()

    return _rust_profiled_deviance(
        theta,
        matrices.y,
        matrices.X,
        z_csc.data,
        z_csc.indices.astype(np.int64),
        z_csc.indptr.astype(np.int64),
        (z_csc.shape[0], z_csc.shape[1]),
        matrices.weights,
        matrices.offset,
        n_levels,
        n_terms,
        correlated,
        REML,
    )


def profiled_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> float:
    if _HAS_RUST:
        return _profiled_deviance_rust(theta, matrices, REML)
    return profiled_deviance(theta, matrices, REML)


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
        use_rust: bool = False,
    ) -> None:
        self.matrices = matrices
        self.REML = REML
        self.verbose = verbose
        self.n_theta = _count_theta(matrices.random_structures)
        has_special_cov = any(
            getattr(s, "cov_type", "us") in ("cs", "ar1") for s in matrices.random_structures
        )
        self.use_rust = use_rust and _HAS_RUST and not has_special_cov

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
        if self.use_rust:
            return _profiled_deviance_rust(theta, self.matrices, self.REML)
        return profiled_deviance(theta, self.matrices, self.REML)

    def optimize(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        options: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        if start is None:
            start = self.get_start_theta()

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

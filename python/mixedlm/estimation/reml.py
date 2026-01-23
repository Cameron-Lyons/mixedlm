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
    gradient_norm: float | None = None
    at_boundary: bool = False
    message: str = ""
    function_evals: int = 0


@dataclass
class DevianceComponents:
    """Components of the deviance calculation for linear mixed models."""

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


@dataclass
class _DevianceCoreResult:
    """Internal result from core deviance computation."""

    deviance: float
    beta: NDArray[np.floating]
    sigma: float
    u: NDArray[np.floating]
    ldL2: float
    ldRX2: float
    wrss: float
    ussq: float
    pwrss: float


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
    """Build Cholesky factor for AR(1) correlation matrix (vectorized)."""
    if q == 1:
        return np.array([[1.0]])
    indices = np.arange(q)
    R = rho ** np.abs(indices[:, None] - indices[None, :])
    try:
        return linalg.cholesky(R, lower=True)
    except linalg.LinAlgError:
        rho_safe = np.clip(rho, -1.0 + 1e-6, 1.0 - 1e-6)
        R = rho_safe ** np.abs(indices[:, None] - indices[None, :])
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


def _profiled_deviance_core(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> _DevianceCoreResult | None:
    """Core deviance computation returning all components.

    This unified function computes the profiled deviance and all intermediate
    values needed for both the deviance itself and for extracting estimates.
    Returns None if the Cholesky decomposition fails.
    """
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    w = matrices.weights
    y_adj = matrices.y - matrices.offset
    sqrt_w = np.sqrt(w)

    if q == 0:
        WX = sqrt_w[:, None] * matrices.X
        Wy = sqrt_w * y_adj
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

        return _DevianceCoreResult(
            deviance=float(dev),
            beta=beta,
            sigma=np.sqrt(sigma2),
            u=np.array([]),
            ldL2=0.0,
            ldRX2=float(ldRX2),
            wrss=float(wrss),
            ussq=0.0,
            pwrss=float(wrss),
        )

    Lambda = _build_lambda(theta, matrices.random_structures)

    Zt = matrices.Zt
    WZ = (
        sqrt_w[:, None] * matrices.Z
        if not sparse.issparse(matrices.Z)
        else sparse.diags(sqrt_w, format="csc") @ matrices.Z
    )
    ZtWZ = WZ.T @ WZ
    LambdatZtWZLambda = Lambda.T @ ZtWZ @ Lambda

    I_q = sparse.eye(q, format="csc")
    V_factor = LambdatZtWZLambda + I_q

    try:
        V_factor_dense = V_factor.toarray() if sparse.issparse(V_factor) else V_factor
        L_V = linalg.cholesky(V_factor_dense, lower=True)
    except linalg.LinAlgError:
        return None

    ldL2 = 2.0 * np.sum(np.log(np.diag(L_V)))

    ZtWy = Zt @ (w * y_adj)
    cu = Lambda.T @ ZtWy
    cu_star = linalg.solve_triangular(L_V, cu, lower=True)

    WX = sqrt_w[:, None] * matrices.X
    sqrtW_sparse = sparse.diags(sqrt_w, format="csc")
    ZtWX = Zt @ (sqrtW_sparse.T @ WX)
    Lambdat_ZtWX = Lambda.T @ ZtWX
    RZX = linalg.solve_triangular(L_V, Lambdat_ZtWX, lower=True)

    XtWX = WX.T @ WX
    XtWy = WX.T @ (sqrt_w * y_adj)

    RZX_tRZX = RZX.T @ RZX
    XtVinvX = XtWX - RZX_tRZX

    try:
        L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)
    except linalg.LinAlgError:
        return None

    ldRX2 = 2.0 * np.sum(np.log(np.diag(L_XtVinvX)))

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

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + ldL2
    if REML:
        dev += ldRX2

    u = Lambda @ u_star

    return _DevianceCoreResult(
        deviance=float(dev),
        beta=beta,
        sigma=np.sqrt(sigma2),
        u=u,
        ldL2=float(ldL2),
        ldRX2=float(ldRX2) if REML else 0.0,
        wrss=float(wrss),
        ussq=float(ussq),
        pwrss=float(pwrss),
    )


def profiled_deviance(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> float:
    """Compute profiled deviance for linear mixed model."""
    result = _profiled_deviance_core(theta, matrices, REML)
    if result is None:
        return 1e10
    return result.deviance


def profiled_deviance_components(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> DevianceComponents:
    """Compute deviance and return all components."""
    result = _profiled_deviance_core(theta, matrices, REML)
    if result is None:
        return DevianceComponents(
            total=1e10,
            ldL2=0.0,
            ldRX2=0.0,
            wrss=0.0,
            ussq=0.0,
            pwrss=0.0,
            sigma2=1.0,
            REML=REML,
        )
    return DevianceComponents(
        total=result.deviance,
        ldL2=result.ldL2,
        ldRX2=result.ldRX2,
        wrss=result.wrss,
        ussq=result.ussq,
        pwrss=result.pwrss,
        sigma2=result.sigma**2,
        REML=REML,
    )


@dataclass
class _RustMatrixCache:
    """Cached data for Rust profiled_deviance calls."""

    y: NDArray[np.floating]
    X: NDArray[np.floating]
    z_data: NDArray[np.floating]
    z_indices: NDArray[np.int64]
    z_indptr: NDArray[np.int64]
    z_shape: tuple[int, int]
    weights: NDArray[np.floating]
    offset: NDArray[np.floating]
    n_levels: list[int]
    n_terms: list[int]
    correlated: list[bool]

    @classmethod
    def from_matrices(cls, matrices: ModelMatrices) -> _RustMatrixCache:
        z_csc = matrices.Z.tocsc()
        return cls(
            y=np.ascontiguousarray(matrices.y),
            X=np.ascontiguousarray(matrices.X),
            z_data=np.ascontiguousarray(z_csc.data),
            z_indices=np.ascontiguousarray(z_csc.indices.astype(np.int64)),
            z_indptr=np.ascontiguousarray(z_csc.indptr.astype(np.int64)),
            z_shape=(z_csc.shape[0], z_csc.shape[1]),
            weights=np.ascontiguousarray(matrices.weights),
            offset=np.ascontiguousarray(matrices.offset),
            n_levels=[s.n_levels for s in matrices.random_structures],
            n_terms=[s.n_terms for s in matrices.random_structures],
            correlated=[s.correlated for s in matrices.random_structures],
        )


def _profiled_deviance_rust_cached(
    theta: NDArray[np.floating],
    cache: _RustMatrixCache,
    REML: bool = True,
) -> float:
    return _rust_profiled_deviance(
        theta,
        cache.y,
        cache.X,
        cache.z_data,
        cache.z_indices,
        cache.z_indptr,
        cache.z_shape,
        cache.weights,
        cache.offset,
        cache.n_levels,
        cache.n_terms,
        cache.correlated,
        REML,
    )


def _profiled_deviance_rust(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
) -> float:
    cache = _RustMatrixCache.from_matrices(matrices)
    return _profiled_deviance_rust_cached(theta, cache, REML)


def profiled_deviance_fast(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
    use_rust: bool = False,
) -> float:
    if use_rust and _HAS_RUST:
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
        use_rust: bool | None = None,
    ) -> None:
        self.matrices = matrices
        self.REML = REML
        self.verbose = verbose
        self.n_theta = _count_theta(matrices.random_structures)
        has_special_cov = any(
            getattr(s, "cov_type", "us") in ("cs", "ar1") for s in matrices.random_structures
        )
        if use_rust is None:
            use_rust = _HAS_RUST and not has_special_cov and matrices.n_random < 50
        self.use_rust = use_rust and _HAS_RUST and not has_special_cov
        self._rust_cache: _RustMatrixCache | None = None
        if self.use_rust:
            self._rust_cache = _RustMatrixCache.from_matrices(matrices)

    def get_start_theta(self) -> NDArray[np.floating]:
        theta_list: list[float] = []
        try:
            beta_ols, sigma_ols = self._fit_ols()
            residuals = self._compute_ols_residuals(beta_ols)
        except Exception:
            beta_ols = None
            sigma_ols = 1.0
            residuals = None

        for struct in self.matrices.random_structures:
            q = struct.n_terms
            cov_type = getattr(struct, "cov_type", "us")

            if residuals is not None and beta_ols is not None:
                try:
                    theta_struct = self._get_adaptive_start_for_structure(
                        struct, residuals, sigma_ols
                    )
                    theta_list.extend(theta_struct)
                    continue
                except Exception:
                    pass

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

    def _fit_ols(self) -> tuple[NDArray[np.floating], float]:
        """Fit OLS regression to get initial estimates."""
        from scipy import linalg

        y = self.matrices.y
        X = self.matrices.X
        w = self.matrices.weights
        offset = self.matrices.offset

        y_adj = y - offset
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = y_adj * sqrt_w

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw

        beta = linalg.solve(XtX, Xty, assume_a="pos")

        fitted = X @ beta
        residuals = y_adj - fitted
        wrss = np.sum(w * residuals**2)
        sigma = np.sqrt(wrss / (len(y) - len(beta)))

        return beta, sigma

    def _compute_ols_residuals(self, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute residuals from OLS fit."""
        y = self.matrices.y
        X = self.matrices.X
        offset = self.matrices.offset

        y_adj = y - offset
        fitted = X @ beta
        return y_adj - fitted

    def _get_adaptive_start_for_structure(
        self, struct, residuals: NDArray[np.floating], sigma_ols: float
    ) -> list[float]:
        """Get data-driven starting values for a random effect structure."""
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type in ("cs", "ar1"):
            sigma_start = max(0.5 * sigma_ols, 0.1)
            rho_start = 0.3 if cov_type == "cs" else 0.5

            if q > 1:
                return [sigma_start, rho_start]
            return [sigma_start]

        if not struct.correlated:
            sigma_start = max(0.5 * sigma_ols, 0.1)
            return [sigma_start] * q

        Z_block_start = 0
        for s in self.matrices.random_structures:
            if s is struct:
                break
            Z_block_start += s.n_levels * s.n_terms

        Z_block_end = Z_block_start + struct.n_levels * struct.n_terms
        Z_block = self.matrices.Z[:, Z_block_start:Z_block_end].toarray()

        group_residuals = []
        for level_idx in range(struct.n_levels):
            level_cols = range(level_idx * q, (level_idx + 1) * q)
            level_mask = np.asarray(Z_block[:, level_cols].sum(axis=1)).ravel() > 0
            if np.any(level_mask):
                level_resid_mean = residuals[level_mask].mean()
                group_residuals.append(level_resid_mean)

        if len(group_residuals) > 1:
            group_var = max(np.var(group_residuals, ddof=1), 0.01)
            relative_sd = np.sqrt(group_var) / max(sigma_ols, 0.1)
            theta_diag = max(min(relative_sd, 3.0), 0.2)
        else:
            theta_diag = 0.5

        theta_struct = []
        for i in range(q):
            for j in range(i + 1):
                if i == j:
                    theta_struct.append(theta_diag)
                else:
                    theta_struct.append(0.0)

        return theta_struct

    def objective(self, theta: NDArray[np.floating]) -> float:
        if self.use_rust and self._rust_cache is not None:
            return _profiled_deviance_rust_cached(theta, self._rust_cache, self.REML)
        return profiled_deviance(theta, self.matrices, self.REML)

    def _extract_estimates(
        self, theta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, NDArray[np.floating]]:
        """Extract beta, sigma, and u from fitted theta."""
        result = _profiled_deviance_core(theta, self.matrices, self.REML)
        if result is None:
            return np.zeros(self.matrices.n_fixed), 1.0, np.zeros(self.matrices.n_random)
        return result.beta, result.sigma, result.u

    def _check_at_boundary(
        self, theta: NDArray[np.floating], bounds: list[tuple[float | None, float | None]]
    ) -> bool:
        """Check if any parameters are at their bounds."""
        tol = 1e-6
        for theta_i, (lb, ub) in zip(theta, bounds, strict=True):
            if lb is not None and abs(theta_i - lb) < tol:
                return True
            if ub is not None and abs(theta_i - ub) < tol:
                return True
        return False

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
        core_result = _profiled_deviance_core(theta_opt, self.matrices, self.REML)

        gradient_norm = None
        if result.jac is not None:
            gradient_norm = float(np.linalg.norm(result.jac))

        at_boundary = self._check_at_boundary(theta_opt, bounds)

        if core_result is None:
            return OptimizationResult(
                theta=theta_opt,
                beta=np.zeros(self.matrices.n_fixed),
                sigma=1.0,
                u=np.zeros(self.matrices.n_random),
                deviance=result.fun,
                converged=False,
                n_iter=result.nit,
                gradient_norm=gradient_norm,
                at_boundary=at_boundary,
                message=result.message,
                function_evals=result.nfev,
            )

        return OptimizationResult(
            theta=theta_opt,
            beta=core_result.beta,
            sigma=core_result.sigma,
            u=core_result.u,
            deviance=result.fun,
            converged=result.success,
            n_iter=result.nit,
            gradient_norm=gradient_norm,
            at_boundary=at_boundary,
            message=result.message,
            function_evals=result.nfev,
        )

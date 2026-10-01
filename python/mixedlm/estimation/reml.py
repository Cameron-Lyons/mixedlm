from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.estimation.validation import validate_finite_real
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure, validate_prior_weights

try:
    from mixedlm import _rust as _rust

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


_SPARSE_PROFILE_MIN_RANDOM = 256


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
    fixed_information: NDArray[np.floating]


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


def _build_lambda_blocks(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
) -> list[NDArray[np.floating]]:
    """Build one relative covariance factor per random-effect structure."""
    blocks: list[NDArray[np.floating]] = []
    theta_idx = 0

    for struct in structures:
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type in ("cs", "ar1"):
            sigma_rel = theta[theta_idx]
            rho = theta[theta_idx + 1] if q > 1 else 0.0
            theta_idx += 2 if q > 1 else 1
            build_correlation = _build_cs_cholesky if cov_type == "cs" else _build_ar1_cholesky
            L_block = sigma_rel * build_correlation(q, rho)
        elif struct.correlated:
            n_theta = q * (q + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta
            L_block = np.zeros((q, q), dtype=np.float64)
            row_indices, col_indices = np.tril_indices(q)
            L_block[row_indices, col_indices] = theta_block
        else:
            L_block = np.diag(theta[theta_idx : theta_idx + q])
            theta_idx += q

        blocks.append(L_block)

    return blocks


def _build_lambda(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
) -> sparse.csc_matrix:
    """Assemble repeated covariance factors directly in column-compressed form."""
    data_blocks: list[NDArray[np.floating]] = []
    row_blocks: list[NDArray[np.intp]] = []
    column_counts: list[NDArray[np.intp]] = []
    offset = 0

    for struct, factor in zip(structures, _build_lambda_blocks(theta, structures), strict=True):
        # Transposing before nonzero orders entries by column, then by row.
        columns, rows = np.nonzero(factor.T)
        level_offsets = offset + np.arange(struct.n_levels) * struct.n_terms
        row_blocks.append((level_offsets[:, None] + rows).ravel())
        data_blocks.append(np.tile(factor[rows, columns], struct.n_levels))
        counts = np.bincount(columns, minlength=struct.n_terms)
        column_counts.append(np.tile(counts, struct.n_levels))
        offset += struct.n_levels * struct.n_terms

    if not structures:
        return sparse.csc_matrix((0, 0), dtype=np.float64)

    indptr = np.empty(offset + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(np.concatenate(column_counts), out=indptr[1:])
    return sparse.csc_matrix(
        (np.concatenate(data_blocks), np.concatenate(row_blocks), indptr),
        shape=(offset, offset),
        dtype=np.float64,
    )


def _diagonal_covariance_factor(
    theta: NDArray[np.floating], structures: list[RandomEffectStructure]
) -> NDArray[np.floating] | None:
    """Return repeated diagonal factors only when every block is diagonal."""
    diagonals = []
    for structure, block in zip(structures, _build_lambda_blocks(theta, structures), strict=True):
        diagonal = np.diag(block)
        if np.count_nonzero(block) != np.count_nonzero(diagonal):
            return None
        diagonals.append(np.tile(diagonal, structure.n_levels))
    return np.concatenate(diagonals) if diagonals else np.empty(0)


def _diagonal_entries(
    matrix: sparse.spmatrix | NDArray[np.floating],
) -> NDArray[np.floating] | None:
    """Identify an exactly diagonal product, including stored sparse zeros."""
    if sparse.issparse(matrix):
        entries = sparse.coo_matrix(matrix, copy=False)
        if not entries.has_canonical_format:
            entries = entries.copy()
            entries.sum_duplicates()
        if np.any((entries.row != entries.col) & (entries.data != 0)):
            return None
        return np.asarray(entries.diagonal())
    diagonal = np.diag(matrix)
    return diagonal if np.count_nonzero(matrix) == np.count_nonzero(diagonal) else None


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


def _build_theta_bounds(
    structures: list[RandomEffectStructure],
    n_theta: int,
    eps: float = 1e-6,
) -> list[tuple[float | None, float | None]]:
    """Build parameter bounds for theta optimization.

    Parameters
    ----------
    structures : list[RandomEffectStructure]
        Random effect structures from the model.
    n_theta : int
        Total number of theta parameters.
    eps : float, default 1e-6
        Small epsilon for correlation parameter bounds.

    Returns
    -------
    list[tuple[float | None, float | None]]
        List of (lower, upper) bound tuples for each theta parameter.
    """
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * n_theta
    idx = 0
    for struct in structures:
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")
        if cov_type == "cs":
            bounds[idx] = (0.0, None)
            idx += 1
            if q > 1:
                bounds[idx] = (-1.0 / (q - 1) + eps, 1.0 - eps)
                idx += 1
        elif cov_type == "ar1":
            bounds[idx] = (0.0, None)
            idx += 1
            if q > 1:
                bounds[idx] = (-1.0 + eps, 1.0 - eps)
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
    return bounds


def _scale_sparse_rows(
    Z: sparse.csc_matrix,
    row_scale: NDArray[np.floating],
) -> sparse.csc_matrix:
    """Scale sparse rows without explicitly constructing a diagonal matrix."""
    return Z.multiply(row_scale[:, None]).tocsc()


@dataclass
class _LMMCrossproducts:
    """Weighted products for one fixed set of model data, independent of theta."""

    weights: NDArray[np.float64]
    sqrt_weights: NDArray[np.floating]
    weighted_X: NDArray[np.floating]
    y_adj: NDArray[np.floating]
    logdet_w: float
    XtWX: NDArray[np.floating]
    XtWy: NDArray[np.floating]
    ZtWZ: sparse.csc_matrix | NDArray[np.floating]
    ZtWX: NDArray[np.floating]
    ZtWy: NDArray[np.floating]
    ZtWZ_diagonal: NDArray[np.floating] | None = None

    @classmethod
    def from_matrices(cls, matrices: ModelMatrices) -> _LMMCrossproducts:
        weights = validate_prior_weights(matrices.weights, matrices.n_obs)
        y_adj = matrices.y - matrices.offset
        sqrt_w = np.sqrt(weights)
        WX = sqrt_w[:, None] * matrices.X
        WZ = (
            _scale_sparse_rows(matrices.Z, sqrt_w)
            if sparse.issparse(matrices.Z)
            else sqrt_w[:, None] * matrices.Z
        )
        ZtWZ = WZ.T @ WZ
        return cls(
            weights=weights,
            sqrt_weights=sqrt_w,
            weighted_X=WX,
            y_adj=y_adj,
            logdet_w=float(np.sum(np.log(weights))),
            XtWX=WX.T @ WX,
            XtWy=WX.T @ (sqrt_w * y_adj),
            ZtWZ=ZtWZ,
            ZtWX=matrices.Zt @ (weights[:, None] * matrices.X),
            ZtWy=matrices.Zt @ (weights * y_adj),
            ZtWZ_diagonal=_diagonal_entries(ZtWZ),
        )

    def with_response(self, matrices: ModelMatrices) -> _LMMCrossproducts:
        y_adj = matrices.y - matrices.offset
        return replace(
            self,
            y_adj=y_adj,
            XtWy=self.weighted_X.T @ (self.sqrt_weights * y_adj),
            ZtWy=matrices.Zt @ (self.weights * y_adj),
        )


def _profiled_deviance_core(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
    REML: bool = True,
    *,
    crossproducts: _LMMCrossproducts | None = None,
) -> _DevianceCoreResult | None:
    """Core deviance computation returning all components.

    This unified function computes the profiled deviance and all intermediate
    values needed for both the deviance itself and for extracting estimates.
    Large sparse random-effect systems retain their sparse representation.
    Diagonal random-effect systems use scalar precision operations.
    Returns None if a factorization fails.
    """
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    if crossproducts is None:
        crossproducts = _LMMCrossproducts.from_matrices(matrices)
    w = crossproducts.weights
    y_adj = crossproducts.y_adj

    if q == 0:
        logdet_w = crossproducts.logdet_w
        WX = crossproducts.weighted_X
        Wy = crossproducts.sqrt_weights * y_adj
        XtWX = crossproducts.XtWX
        XtWy = crossproducts.XtWy
        try:
            beta = linalg.solve(XtWX, XtWy, assume_a="pos")
        except linalg.LinAlgError:
            beta = linalg.lstsq(WX, Wy)[0]

        resid = y_adj - matrices.X @ beta
        wrss = np.dot(w * resid, resid)
        denom = n - p if REML else n
        sigma2 = wrss / denom

        ldRX2 = np.linalg.slogdet(XtWX)[1] if REML else 0.0
        dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) - logdet_w
        if REML:
            dev += ldRX2

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
            fixed_information=XtWX,
        )

    diagonal_factor = (
        _diagonal_covariance_factor(theta, matrices.random_structures)
        if crossproducts.ZtWZ_diagonal is not None
        else None
    )
    Lambda = None
    solve_random: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    if diagonal_factor is not None:
        assert crossproducts.ZtWZ_diagonal is not None
        information = diagonal_factor * (crossproducts.ZtWZ_diagonal * diagonal_factor)
        precision = 1.0 + information
        if np.any(~np.isfinite(precision)) or np.any(precision <= 0):
            return None

        def solve_random(rhs: NDArray[np.floating]) -> NDArray[np.floating]:
            return rhs / precision

        ldL2 = np.sum(np.log1p(information))
        scaled_factor = diagonal_factor / np.sqrt(precision)
        cu_star = scaled_factor * crossproducts.ZtWy
        RZX = scaled_factor[:, None] * crossproducts.ZtWX
        XtVinvX = crossproducts.XtWX - RZX.T @ RZX
        Xty_adj = crossproducts.XtWy - RZX.T @ cu_star
    else:
        Lambda = _build_lambda(theta, matrices.random_structures)
        LambdatZtWZLambda = Lambda.T @ crossproducts.ZtWZ @ Lambda
        V_factor = LambdatZtWZLambda + sparse.eye(q, format="csc")
        cu = Lambda.T @ crossproducts.ZtWy
        Lambdat_ZtWX = Lambda.T @ crossproducts.ZtWX
        if sparse.issparse(V_factor) and q >= _SPARSE_PROFILE_MIN_RANDOM:
            try:
                factor = sparse_linalg.splu(V_factor.tocsc())
            except RuntimeError:
                return None
            solve_random = factor.solve
            # The precision is positive definite. Permutation signs do not affect
            # its log determinant, obtained from the absolute LU diagonal.
            ldL2 = np.sum(np.log(np.abs(factor.U.diagonal())))
            solved = solve_random(np.column_stack((cu, Lambdat_ZtWX)))
            XtVinvX = crossproducts.XtWX - Lambdat_ZtWX.T @ solved[:, 1:]
            XtVinvX = (XtVinvX + XtVinvX.T) * 0.5
            Xty_adj = crossproducts.XtWy - Lambdat_ZtWX.T @ solved[:, 0]
        else:
            try:
                V_factor_dense = V_factor.toarray() if sparse.issparse(V_factor) else V_factor
                L_V = linalg.cholesky(V_factor_dense, lower=True)
            except linalg.LinAlgError:
                return None

            def solve_random(rhs: NDArray[np.floating]) -> NDArray[np.floating]:
                return linalg.cho_solve((L_V, True), rhs)

            ldL2 = 2.0 * np.sum(np.log(np.diag(L_V)))
            cu_star = linalg.solve_triangular(L_V, cu, lower=True)
            RZX = linalg.solve_triangular(L_V, Lambdat_ZtWX, lower=True)
            XtVinvX = crossproducts.XtWX - RZX.T @ RZX
            Xty_adj = crossproducts.XtWy - RZX.T @ cu_star

    try:
        L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)
    except linalg.LinAlgError:
        return None

    ldRX2 = 2.0 * np.sum(np.log(np.diag(L_XtVinvX)))

    beta = linalg.cho_solve((L_XtVinvX, True), Xty_adj)

    marginal_resid = y_adj - matrices.X @ beta

    Zt_resid = matrices.Zt @ (w * marginal_resid)
    if diagonal_factor is not None:
        u_star = solve_random(diagonal_factor * Zt_resid)
        u = diagonal_factor * u_star
    else:
        assert Lambda is not None
        u_star = solve_random(Lambda.T @ Zt_resid)
        u = np.asarray(Lambda @ u_star).reshape(-1)
    conditional_resid = marginal_resid - matrices.Z @ u
    wrss = np.dot(w * conditional_resid, conditional_resid)
    ussq = np.dot(u_star, u_star)
    pwrss = wrss + ussq

    denom = n - p if REML else n
    sigma2 = pwrss / denom

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + ldL2 - crossproducts.logdet_w
    if REML:
        dev += ldRX2

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
        fixed_information=XtVinvX,
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
    """Owned native design and response products, independent of theta."""

    design: Any
    response: Any

    @classmethod
    def from_matrices(cls, matrices: ModelMatrices) -> _RustMatrixCache:
        from mixedlm._rust import LmmDesign

        z_csc = matrices.Z.tocsc()
        design = LmmDesign(
            matrices.X,
            np.ascontiguousarray(z_csc.data),
            np.ascontiguousarray(z_csc.indices, dtype=np.int64),
            np.ascontiguousarray(z_csc.indptr, dtype=np.int64),
            z_csc.shape,
            matrices.weights,
            matrices.offset,
            [s.n_levels for s in matrices.random_structures],
            [s.n_terms for s in matrices.random_structures],
            [s.correlated for s in matrices.random_structures],
        )
        return cls(design, design.with_response(matrices.y))

    def with_response(self, response: NDArray[np.floating]) -> _RustMatrixCache:
        return type(self)(self.design, self.design.with_response(response))


def _profiled_deviance_rust_cached(
    theta: NDArray[np.floating],
    cache: _RustMatrixCache,
    REML: bool = True,
) -> float:
    return cache.response.deviance(theta, REML)


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
    """Optimize theta for a fixed design, optionally sharing it with new responses.

    Treat the design matrices, weights and offsets as immutable for the lifetime
    of this optimizer and any optimizers returned by :meth:`with_response`.
    Construct a new optimizer if any of those inputs change.
    """

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

    def with_response(self, response: NDArray[np.floating]) -> LMMOptimizer:
        """Create an independent fit sharing this optimizer's prepared design.

        Only response-dependent products are rebuilt. The response is copied;
        starting values and optimization results are not shared between fits.
        """
        validate_finite_real("response", response, (self.matrices.n_obs,))
        matrices = replace(self.matrices, y=np.array(response, dtype=np.float64, copy=True))
        matrices.Zt = self.matrices.Zt
        optimizer = copy(self)
        optimizer.matrices = matrices
        if not self.use_rust or not self.matrices.n_random or "_crossproducts" in self.__dict__:
            optimizer._crossproducts = self._crossproducts.with_response(matrices)
        if self._rust_cache is not None:
            optimizer._rust_cache = self._rust_cache.with_response(matrices.y)
        return optimizer

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

        level_indices = getattr(struct, "level_indices", None)
        if level_indices is not None and len(level_indices) == len(residuals):
            valid = (level_indices >= 0) & (level_indices < struct.n_levels)
            counts = np.bincount(level_indices[valid], minlength=struct.n_levels)
            sums = np.bincount(
                level_indices[valid],
                weights=residuals[valid],
                minlength=struct.n_levels,
            )
            populated = counts > 0
            group_residual_means = sums[populated] / counts[populated]
        else:
            Z_block_start = 0
            for s in self.matrices.random_structures:
                if s is struct:
                    break
                Z_block_start += s.n_levels * s.n_terms

            Z_block_end = Z_block_start + struct.n_levels * struct.n_terms
            Z_block = self.matrices.Z[:, Z_block_start:Z_block_end]

            residual_means: list[float] = []
            for level_idx in range(struct.n_levels):
                level_cols = range(level_idx * q, (level_idx + 1) * q)
                level_block = Z_block[:, level_cols]
                if sparse.issparse(level_block):
                    level_mask = level_block.getnnz(axis=1) > 0
                else:
                    level_mask = np.asarray(level_block != 0).any(axis=1)
                if np.any(level_mask):
                    level_resid_mean = residuals[level_mask].mean()
                    residual_means.append(float(level_resid_mean))
            group_residual_means = np.asarray(residual_means)

        if len(group_residual_means) > 1:
            group_var = max(float(np.var(group_residual_means, ddof=1)), 0.01)
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

    @cached_property
    def _crossproducts(self) -> _LMMCrossproducts:
        return _LMMCrossproducts.from_matrices(self.matrices)

    def _evaluate_core(self, theta: NDArray[np.floating]) -> _DevianceCoreResult | None:
        # Fixed-only fits retain the existing least-squares fallback and rank
        # behavior. Mixed models can extract estimates from their native products.
        if self.use_rust and self._rust_cache is not None and self.matrices.n_random:
            native = self._rust_cache.response.evaluate(theta, self.REML)
            if native is not None:
                deviance, beta, sigma, u, ldL2, ldRX2, wrss, ussq, pwrss, information = native
                p = self.matrices.n_fixed
                return _DevianceCoreResult(
                    deviance=deviance,
                    beta=np.asarray(beta),
                    sigma=sigma,
                    u=np.asarray(u),
                    ldL2=ldL2,
                    ldRX2=ldRX2,
                    wrss=wrss,
                    ussq=ussq,
                    pwrss=pwrss,
                    fixed_information=np.asarray(information).reshape(p, p),
                )
            return None
        return _profiled_deviance_core(
            theta,
            self.matrices,
            self.REML,
            crossproducts=self._crossproducts,
        )

    def objective(self, theta: NDArray[np.floating]) -> float:
        if self.use_rust and self._rust_cache is not None:
            return _profiled_deviance_rust_cached(theta, self._rust_cache, self.REML)
        result = self._evaluate_core(theta)
        return 1e10 if result is None else result.deviance

    def _final_evaluation(self, theta: NDArray[np.floating]) -> _DevianceCoreResult:
        """Extract and validate the estimates at the final parameter vector."""
        try:
            validate_finite_real("variance parameters", theta, (self.n_theta,))
            result = self._evaluate_core(theta)
            if result is None:
                raise ValueError("final covariance factorization failed")
            validate_finite_real("deviance", result.deviance, ())
            validate_finite_real("fixed effects", result.beta, (self.matrices.n_fixed,))
            validate_finite_real("random effects", result.u, (self.matrices.n_random,))
            validate_finite_real("residual scale", result.sigma, ())
            if result.sigma <= 0:
                raise ValueError("residual scale must be strictly positive")
        except (
            FloatingPointError,
            OverflowError,
            TypeError,
            ValueError,
            linalg.LinAlgError,
        ) as exc:
            raise RuntimeError(
                f"Linear optimization did not produce a valid fit: {type(exc).__name__}: {exc}"
            ) from exc
        return result

    def _extract_estimates(
        self, theta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, NDArray[np.floating]]:
        """Extract beta, sigma, and u from valid fitted theta."""
        result = self._final_evaluation(theta)
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
        *,
        restart_edge: bool = True,
    ) -> OptimizationResult:
        if start is None:
            start = self.get_start_theta()

        bounds = _build_theta_bounds(self.matrices.random_structures, len(start))

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
        core_result = self._final_evaluation(theta_opt)

        gradient_norm = None
        if result.jac is not None:
            gradient_norm = float(np.linalg.norm(result.jac))

        at_boundary = self._check_at_boundary(theta_opt, bounds)

        return OptimizationResult(
            theta=theta_opt,
            beta=core_result.beta,
            sigma=core_result.sigma,
            u=core_result.u,
            deviance=core_result.deviance,
            converged=result.success,
            n_iter=result.nit,
            gradient_norm=gradient_norm,
            at_boundary=at_boundary,
            message=result.message,
            function_evals=result.nfev,
        )

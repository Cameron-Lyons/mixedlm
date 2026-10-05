from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from mixedlm.estimation.optimizers import (
    OptimizeResult,
    _near_zero_variances,
    _rounding_tolerance,
    run_optimizer,
)
from mixedlm.estimation.validation import validate_finite_real
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure, validate_prior_weights

if TYPE_CHECKING:
    from mixedlm.models.shared_utils import _SparseCholeskyPattern

try:
    from mixedlm import _rust as _rust

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
    optimizer: str = ""


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
    return _assemble_lambda(_build_lambda_blocks(theta, structures), structures)


def _lambda_pattern(structures: list[RandomEffectStructure]) -> sparse.csc_matrix:
    """Mark every entry a covariance factor can store, whatever its parameters."""
    blocks = [
        np.eye(s.n_terms)
        if not s.correlated and getattr(s, "cov_type", "us") not in ("cs", "ar1")
        else np.tril(np.ones((s.n_terms, s.n_terms)))
        for s in structures
    ]
    return _assemble_lambda(blocks, structures)


def _assemble_lambda(
    factors: Sequence[NDArray[np.floating]],
    structures: list[RandomEffectStructure],
) -> sparse.csc_matrix:
    data_blocks: list[NDArray[np.floating]] = []
    row_blocks: list[NDArray[np.intp]] = []
    column_counts: list[NDArray[np.intp]] = []
    offset = 0

    for struct, factor in zip(structures, factors, strict=True):
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


def _in_correlated_block(structures: list[RandomEffectStructure]) -> list[bool]:
    """Flag the theta entries of correlated covariances of two or more terms."""
    flags: list[bool] = []
    for struct in structures:
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")
        if cov_type == "cs" or cov_type == "ar1":
            flags.extend([False] * (2 if q > 1 else 1))
        elif struct.correlated:
            flags.extend([q > 1] * (q * (q + 1) // 2))
        else:
            flags.extend([False] * q)
    return flags


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
    # Symbolic analysis of the random-effect precision, shared by every theta.
    precision_pattern: _SparseCholeskyPattern | None = None

    @classmethod
    def from_matrices(cls, matrices: ModelMatrices) -> _LMMCrossproducts:
        from mixedlm.models import shared_utils

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
        precision_pattern = None
        q = matrices.n_random
        if shared_utils._HAS_RUST and q >= shared_utils._SPARSE_PROJECTION_MIN_RANDOM:
            # Absolute values keep products of the structural pattern from cancelling.
            factor = _lambda_pattern(matrices.random_structures)
            pattern = factor.T @ abs(sparse.csc_matrix(ZtWZ)) @ factor
            precision_pattern = shared_utils._SparseCholeskyPattern(
                pattern + sparse.eye(q, format="csc")
            )
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
            precision_pattern=precision_pattern,
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
        from mixedlm.models.shared_utils import _RandomEffectFactor

        Lambda = _build_lambda(theta, matrices.random_structures)
        random_precision = Lambda.T @ crossproducts.ZtWZ @ Lambda + sparse.eye(q, format="csc")
        try:
            factor = _RandomEffectFactor(random_precision, pattern=crossproducts.precision_pattern)
        except (linalg.LinAlgError, RuntimeError):
            return None
        solve_random = factor.solve
        ldL2 = factor.logdet
        projected = factor.crossproduct(
            Lambda.T @ crossproducts.ZtWX, Lambda.T @ crossproducts.ZtWy
        )
        XtVinvX = crossproducts.XtWX - projected[:, :p]
        XtVinvX = (XtVinvX + XtVinvX.T) * 0.5
        Xty_adj = crossproducts.XtWy - projected[:, p]

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


def profiled_reml(
    theta: NDArray[np.floating],
    matrices: ModelMatrices,
) -> float:
    """Deprecated alias for ``profiled_deviance(theta, matrices, REML=True)``."""
    warnings.warn(
        "profiled_reml is deprecated; use profiled_deviance(theta, matrices, REML=True).",
        DeprecationWarning,
        stacklevel=2,
    )
    return profiled_deviance(theta, matrices, REML=True)


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


# The default lmer optimizer: L-BFGS-B with exact native gradients, falling back
# to COBYQA, which also fits covariance structures without native gradients.
AUTO_OPTIMIZER = "auto"
# Final gradient tolerances per observation, as the deviance sums observation terms.
_AUTO_GRADIENT_TOL = 1e-8
_AUTO_FALLBACK_GRADIENT_TOL = 1e-6


class _LMMGradientObjective:
    """Keep one owned value/gradient pair local to an optimization run."""

    def __init__(
        self,
        evaluate: Callable[[NDArray[np.floating]], tuple[float, NDArray[np.floating]]],
    ) -> None:
        self.evaluate = evaluate
        self.theta: NDArray[np.floating] | None = None
        self.value = 0.0
        self.derivative: NDArray[np.floating] = np.empty(0)

    def __call__(self, theta: NDArray[np.floating]) -> float:
        if self.theta is None or not np.array_equal(theta, self.theta):
            snapshot = np.array(theta, dtype=np.float64, copy=True)
            value, derivative = self.evaluate(snapshot)
            self.theta, self.value, self.derivative = snapshot, value, derivative
        return self.value

    def gradient(self, theta: NDArray[np.floating]) -> NDArray[np.floating]:
        self(theta)
        return self.derivative.copy()


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
            use_rust = _HAS_RUST
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
        """Start each covariance scale at the relative spread of group residual means.

        Theta is relative to the residual scale, so every structure starts from
        the same dimensionless estimate.
        """
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

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

        if len(group_residual_means) > 1 and sigma_ols > 0:
            relative_sd = np.sqrt(float(np.var(group_residual_means, ddof=1))) / sigma_ols
            theta_diag = max(min(relative_sd, 3.0), 0.2)
        else:
            theta_diag = 0.5

        if cov_type in ("cs", "ar1"):
            rho_start = 0.3 if cov_type == "cs" else 0.5
            return [theta_diag, rho_start] if q > 1 else [theta_diag]
        if not struct.correlated:
            return [theta_diag] * q

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

    def _optimization_functions(
        self, method: str, use_analytic_gradient: bool
    ) -> tuple[
        Callable[[NDArray[np.floating]], float],
        Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
    ]:
        if not isinstance(use_analytic_gradient, (bool, np.bool_)):
            raise ValueError("use_analytic_gradient must be a boolean")
        if (
            use_analytic_gradient
            and method in {"L-BFGS-B", "BFGS", "TNC", "SLSQP", "trust-constr"}
            and self.use_rust
            and self._rust_cache is not None
            and self.n_theta
        ):
            response, reml = self._rust_cache.response, self.REML
            objective = _LMMGradientObjective(
                lambda theta: response.deviance_with_gradient(theta, reml)
            )
            return objective, objective.gradient
        return self.objective, None

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
        use_analytic_gradient: bool = False,
    ) -> OptimizationResult:
        """Fit covariance parameters with optional prepared native gradients.

        Analytic gradients support L-BFGS-B, BFGS, TNC, SLSQP, and trust-constr.
        They share value/gradient evaluations within this call; other backends
        and covariance structures retain the solver's numerical derivatives.
        ``method="auto"`` always uses native gradients where available and
        passes ``options`` other than ``maxiter`` and evaluation limits only
        to its COBYQA stage.
        """
        if start is None:
            start = self.get_start_theta()

        bounds = _build_theta_bounds(self.matrices.random_structures, len(start))
        objective, gradient = self._optimization_functions(method, use_analytic_gradient)

        callback: Callable[[NDArray[np.floating]], None] | None = None
        if self.verbose > 0:

            def callback(x: NDArray[np.floating]) -> None:
                dev = objective(x)
                print(f"theta = {x}, deviance = {dev:.6f}")

        opt_options = {"maxiter": maxiter}
        if options:
            opt_options.update(options)

        if method == AUTO_OPTIMIZER:
            result, method = self._optimize_auto(start, bounds, opt_options, callback, restart_edge)
        else:
            result = run_optimizer(
                objective,
                start,
                method=method,
                bounds=bounds,
                options=opt_options,
                callback=callback,
                jac=gradient,
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
            optimizer=method,
        )

    def _optimize_auto(
        self,
        start: NDArray[np.floating],
        bounds: list[tuple[float | None, float | None]],
        options: dict[str, Any],
        callback: Callable[[NDArray[np.floating]], None] | None,
        restart_edge: bool,
    ) -> tuple[OptimizeResult, str]:
        """Run exact-gradient L-BFGS-B, falling back to COBYQA from the same start.

        The fallback covers unconverged fits and large final gradients. It also
        covers variance scales left near zero, where the gradient vanishes by
        symmetry, unless boundary probes checked them: scales of correlated
        covariances always fall back, as such singular fits can have several
        boundary optima. The fallback keeps the lower of the two deviances,
        preferring COBYQA's within rounding.
        """
        gradient_fit = None
        if self.use_rust and self._rust_cache is not None and self.n_theta:
            response, reml = self._rust_cache.response, self.REML

            def evaluate(theta: NDArray[np.floating]) -> tuple[float, NDArray[np.floating]]:
                # A failed line search can propose non-finite steps, which the
                # native evaluator rejects; report them as infeasible instead.
                if np.all(np.isfinite(theta)):
                    value, derivative = response.deviance_with_gradient(theta, reml)
                    if np.isfinite(value) and np.all(np.isfinite(derivative)):
                        return value, derivative
                return np.inf, np.zeros_like(theta)

            objective = _LMMGradientObjective(evaluate)
            n_obs = self.matrices.n_obs
            gradient_options = {
                "maxiter": options["maxiter"],
                "ftol": 1e-15,
                "gtol": _AUTO_GRADIENT_TOL * n_obs,
            }
            # Evaluation limits bound each stage.
            limit = options.get("maxfev", options.get("maxfun"))
            if limit is not None:
                gradient_options["maxfun"] = limit
            gradient_fit = run_optimizer(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options=gradient_options,
                callback=callback,
                jac=objective.gradient,
                restart_edge=restart_edge,
            )
            correlated = _in_correlated_block(self.matrices.random_structures)
            near_zero = [
                i
                for i in _near_zero_variances(gradient_fit.x, start, bounds)
                if correlated[i] or not restart_edge
            ]
            if (
                gradient_fit.success
                and np.isfinite(gradient_fit.fun)
                and not near_zero
                and gradient_fit.jac is not None
                and np.max(np.abs(gradient_fit.jac)) <= _AUTO_FALLBACK_GRADIENT_TOL * n_obs
            ):
                return gradient_fit, "L-BFGS-B"

        result = run_optimizer(
            self.objective,
            start,
            method="COBYQA",
            bounds=bounds,
            options=options,
            callback=callback,
            restart_edge=restart_edge,
        )
        if gradient_fit is None:
            return result, "COBYQA"
        nit, nfev = gradient_fit.nit + result.nit, gradient_fit.nfev + result.nfev
        if gradient_fit.success and gradient_fit.fun < result.fun - _rounding_tolerance(result.fun):
            return replace(gradient_fit, nit=nit, nfev=nfev), "L-BFGS-B"
        message = f"{result.message} (after L-BFGS-B: {gradient_fit.message})"
        return replace(result, nit=nit, nfev=nfev, message=message), "COBYQA"

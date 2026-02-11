from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure


@dataclass
class EMResult:
    """Result from EM algorithm fitting.

    Attributes
    ----------
    theta : NDArray
        Variance component parameters.
    beta : NDArray
        Fixed effects estimates.
    sigma : float
        Residual standard deviation.
    converged : bool
        Whether the algorithm converged.
    n_iter : int
        Number of iterations performed.
    final_loglik : float
        Final log-likelihood value.
    """

    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    sigma: float
    converged: bool
    n_iter: int
    final_loglik: float


def _init_sigma_k(
    struct: RandomEffectStructure,
    sigma2_e: float,
    sigma2_u_init_scale: float,
    sigma2_u_init_min: float,
) -> NDArray[np.floating]:
    """Initialize per-structure covariance matrix."""
    q = struct.n_terms
    init_var = max(sigma2_e * sigma2_u_init_scale, sigma2_u_init_min)
    return np.eye(q, dtype=np.float64) * init_var


def _build_block_diag_D_inv(
    structures: list[RandomEffectStructure],
    sigma_list: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Build block-diagonal D_inv = block_diag(kron(I_{n_k}, inv(Sigma_k)))."""
    blocks: list[NDArray[np.floating]] = []
    for struct, sigma_k in zip(structures, sigma_list, strict=True):
        q = struct.n_terms
        n_levels = struct.n_levels
        sigma_k_inv = linalg.inv(sigma_k) if q > 1 else np.array([[1.0 / sigma_k[0, 0]]])
        block = np.kron(np.eye(n_levels), sigma_k_inv)
        blocks.append(block)
    return linalg.block_diag(*blocks)


def _m_step_update_sigma(
    struct: RandomEffectStructure,
    u_block: NDArray[np.floating],
    Var_u_block: NDArray[np.floating],
    variance_floor: float,
) -> NDArray[np.floating]:
    """M-step update for a single structure's covariance matrix.

    Parameters
    ----------
    struct : RandomEffectStructure
        The random effect structure.
    u_block : NDArray
        Random effects for this structure, shape (n_levels * q,).
    Var_u_block : NDArray
        Posterior variance block for this structure, shape (n_levels*q, n_levels*q).
    variance_floor : float
        Minimum eigenvalue for numerical stability.
    """
    q = struct.n_terms
    n_levels = struct.n_levels

    U = u_block.reshape(n_levels, q)
    S = U.T @ U

    var_sum = np.zeros((q, q), dtype=np.float64)
    for i in range(n_levels):
        start = i * q
        end = start + q
        var_sum += Var_u_block[start:end, start:end]

    Sigma_new = (S + var_sum) / n_levels

    if not struct.correlated and q > 1:
        Sigma_new = np.diag(np.diag(Sigma_new))

    eigvals, eigvecs = linalg.eigh(Sigma_new)
    eigvals = np.maximum(eigvals, variance_floor)
    Sigma_new = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return Sigma_new


def _sigma_to_theta(
    structures: list[RandomEffectStructure],
    sigma_list: list[NDArray[np.floating]],
    sigma2_e: float,
) -> NDArray[np.floating]:
    """Convert per-structure covariance matrices to theta parameters.

    For each structure, compute L_k = cholesky(Sigma_k / sigma2_e) and extract
    parameters in the format expected by _build_lambda().
    """
    theta_parts: list[NDArray[np.floating]] = []

    for struct, sigma_k in zip(structures, sigma_list, strict=True):
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type not in ("us",):
            raise NotImplementedError(
                f"EM-REML does not support cov_type='{cov_type}'. "
                "Use direct optimization via lmer() with optimizer='bobyqa' instead."
            )

        relative_cov = sigma_k / sigma2_e

        eigvals, eigvecs = linalg.eigh(relative_cov)
        eigvals = np.maximum(eigvals, 1e-10)
        relative_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        L = linalg.cholesky(relative_cov, lower=True)

        if struct.correlated or q == 1:
            row_indices, col_indices = np.tril_indices(q)
            theta_parts.append(L[row_indices, col_indices])
        else:
            theta_parts.append(np.diag(L))

    return np.concatenate(theta_parts)


def em_reml_simple(
    matrices: ModelMatrices,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: int = 0,
    variance_floor: float = 1e-8,
    min_iter_converge: int = 5,
    sigma2_u_init_scale: float = 0.5,
    sigma2_u_init_min: float = 0.1,
) -> EMResult:
    """Fit linear mixed model using EM-REML algorithm.

    This is an EM-REML implementation that supports models with multiple
    random effects and random slopes. It can be used as a robust initialization
    method before switching to direct optimization.

    The algorithm alternates between:
    - E-step: Computing conditional expectations of random effects
    - M-step: Updating variance components and fixed effects

    Parameters
    ----------
    matrices : ModelMatrices
        Model design matrices and data.
    max_iter : int, default 100
        Maximum number of EM iterations.
    tol : float, default 1e-4
        Convergence tolerance for relative log-likelihood change.
    verbose : int, default 0
        Verbosity level (0 = silent, 1 = show iterations).
    variance_floor : float, default 1e-8
        Minimum value for variance components to ensure numerical stability.
    min_iter_converge : int, default 5
        Minimum number of iterations before checking for convergence.
    sigma2_u_init_scale : float, default 0.5
        Scaling factor applied to residual variance for initial random effect
        variance estimate.
    sigma2_u_init_min : float, default 0.1
        Minimum initial value for random effect variance.

    Returns
    -------
    EMResult
        Result containing estimated parameters and convergence info.

    Notes
    -----
    This implementation supports random intercepts, random slopes (correlated
    and uncorrelated), and multiple random effect structures with unstructured
    covariance (cov_type='us').

    The EM algorithm tends to be more robust to poor starting values but
    converges more slowly than direct optimization methods.

    References
    ----------
    .. [1] Lindstrom & Bates (1988). "Newton-Raphson and EM Algorithms for
           Linear Mixed-Effects Models for Repeated-Measures Data"
    .. [2] Dempster, Laird, & Rubin (1977). "Maximum Likelihood from
           Incomplete Data via the EM Algorithm"

    Examples
    --------
    >>> from mixedlm import lFormula
    >>> from mixedlm.estimation.em_reml import em_reml_simple
    >>> parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)
    >>> result = em_reml_simple(parsed.matrices, max_iter=50, verbose=1)
    """
    structures = matrices.random_structures

    for struct in structures:
        cov_type = getattr(struct, "cov_type", "us")
        if cov_type not in ("us",):
            raise NotImplementedError(
                f"EM-REML does not support cov_type='{cov_type}'. "
                "Use direct optimization via lmer() with optimizer='bobyqa' instead."
            )

    y = matrices.y
    X = matrices.X
    Z = matrices.Z.toarray() if hasattr(matrices.Z, "toarray") else matrices.Z
    weights = matrices.weights
    offset = matrices.offset

    y_adj = y - offset
    n = len(y)
    p = X.shape[1]

    try:
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        yw = y_adj * sqrt_w
        beta = linalg.lstsq(Xw, yw)[0]
        residuals = y_adj - X @ beta
        sigma2_e = float(np.sum(weights * residuals**2) / n)
    except Exception:
        beta = np.zeros(p)
        sigma2_e = 1.0

    sigma_list = [
        _init_sigma_k(struct, sigma2_e, sigma2_u_init_scale, sigma2_u_init_min)
        for struct in structures
    ]

    prev_loglik = -np.inf
    converged = False

    col_offsets: list[int] = []
    offset_val = 0
    for struct in structures:
        col_offsets.append(offset_val)
        offset_val += struct.n_levels * struct.n_terms

    for iteration in range(max_iter):
        W = weights / sigma2_e
        XtWX = X.T @ (W[:, None] * X)
        XtWZ = X.T @ (W[:, None] * Z)
        ZtWZ = Z.T @ (W[:, None] * Z)
        D_inv = _build_block_diag_D_inv(structures, sigma_list)

        XtWy = X.T @ (W * y_adj)
        ZtWy = Z.T @ (W * y_adj)

        LHS_top = np.hstack([XtWX, XtWZ])
        LHS_bot = np.hstack([XtWZ.T, ZtWZ + D_inv])
        LHS = np.vstack([LHS_top, LHS_bot])
        RHS = np.concatenate([XtWy, ZtWy])

        try:
            solution = linalg.solve(LHS, RHS, assume_a="pos")
        except linalg.LinAlgError:
            solution = linalg.lstsq(LHS, RHS)[0]

        beta_new = solution[:p]
        u_hat = solution[p:]

        try:
            XtWX_inv_XtWZ = linalg.solve(XtWX, XtWZ, assume_a="pos")
            schur_complement = (ZtWZ + D_inv) - XtWZ.T @ XtWX_inv_XtWZ
            Var_u = linalg.inv(schur_complement)
        except linalg.LinAlgError:
            Var_u = linalg.pinv(ZtWZ + D_inv)

        sigma_list_new: list[NDArray[np.floating]] = []
        for k, struct in enumerate(structures):
            block_size = struct.n_levels * struct.n_terms
            start = col_offsets[k]
            end = start + block_size
            u_block = u_hat[start:end]
            Var_u_block = Var_u[start:end, start:end]
            sigma_k_new = _m_step_update_sigma(struct, u_block, Var_u_block, variance_floor)
            sigma_list_new.append(sigma_k_new)

        residuals = y_adj - X @ beta_new - Z @ u_hat
        wrss = float(np.sum(weights * residuals**2))
        uncertainty_term = float(np.trace(ZtWZ @ Var_u))
        sigma2_e_new = max((wrss + uncertainty_term) / n, variance_floor)

        loglik = -0.5 * (
            (n - p) * np.log(2 * np.pi * sigma2_e_new)
            + wrss / sigma2_e_new
            + np.linalg.slogdet(ZtWZ + D_inv)[1]
            + np.linalg.slogdet(XtWX)[1]
        )

        if not np.isfinite(loglik):
            if verbose >= 1:
                print(f"Warning: Non-finite log-likelihood at iteration {iteration + 1}")
            break

        loglik_change = abs(loglik - prev_loglik)
        rel_change = (
            loglik_change / (abs(prev_loglik) + variance_floor) if prev_loglik != -np.inf else 1.0
        )

        if verbose >= 1:
            print(
                f"EM iter {iteration + 1}: loglik={loglik:.4f}, "
                f"sigma2_e={sigma2_e_new:.4f}, "
                f"rel_change={rel_change:.6f}"
            )

        if rel_change < tol and iteration >= min_iter_converge:
            converged = True
            if verbose >= 1:
                print(f"Converged after {iteration + 1} iterations")
            break

        beta = beta_new
        sigma2_e = sigma2_e_new
        sigma_list = sigma_list_new
        prev_loglik = loglik

    sigma = np.sqrt(sigma2_e)
    theta = _sigma_to_theta(structures, sigma_list, sigma2_e)

    return EMResult(
        theta=theta,
        beta=beta,
        sigma=sigma,
        converged=converged,
        n_iter=iteration + 1,
        final_loglik=prev_loglik,
    )

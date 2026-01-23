from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from mixedlm.matrices.design import ModelMatrices


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

    This is a simplified EM-REML implementation suitable for models with
    a single random intercept. It can be used as a robust initialization
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
    This implementation currently supports only simple random intercept models.
    For more complex random effect structures, direct optimization via BOBYQA
    or L-BFGS-B is recommended.

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
    >>> parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)
    >>> result = em_reml_simple(parsed.matrices, max_iter=50, verbose=1)
    """
    if len(matrices.random_structures) != 1:
        raise NotImplementedError(
            "EM-REML currently only supports models with a single random effect term. "
            "Use direct optimization via lmer() with optimizer='bobyqa' instead."
        )

    struct = matrices.random_structures[0]
    if struct.n_terms != 1:
        raise NotImplementedError(
            "EM-REML currently only supports random intercept models (1|group). "
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
    q = Z.shape[1]

    try:
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        yw = y_adj * sqrt_w
        beta = linalg.lstsq(Xw, yw)[0]
        residuals = y_adj - X @ beta
        sigma2_e = np.sum(weights * residuals**2) / n
    except Exception:
        beta = np.zeros(p)
        sigma2_e = 1.0

    sigma2_u = max(sigma2_e * sigma2_u_init_scale, sigma2_u_init_min)
    prev_loglik = -np.inf
    converged = False

    for iteration in range(max_iter):
        W = weights / sigma2_e
        XtWX = X.T @ (W[:, None] * X)
        XtWZ = X.T @ (W[:, None] * Z)
        ZtWZ = Z.T @ (W[:, None] * Z)
        D_inv = np.eye(q) / sigma2_u

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

        E_utu = u_hat @ u_hat + np.trace(Var_u)
        sigma2_u_new = E_utu / q

        residuals = y_adj - X @ beta_new - Z @ u_hat
        wrss = np.sum(weights * residuals**2)

        uncertainty_term = np.trace(ZtWZ @ Var_u)

        sigma2_e_new = (wrss + uncertainty_term) / n

        sigma2_u_new = max(sigma2_u_new, variance_floor)
        sigma2_e_new = max(sigma2_e_new, variance_floor)

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
                f"sigma2_e={sigma2_e_new:.4f}, sigma2_u={sigma2_u_new:.4f}, "
                f"rel_change={rel_change:.6f}"
            )

        if rel_change < tol and iteration >= min_iter_converge:
            converged = True
            if verbose >= 1:
                print(f"Converged after {iteration + 1} iterations")
            break

        beta = beta_new
        sigma2_e = sigma2_e_new
        sigma2_u = sigma2_u_new
        prev_loglik = loglik

    sigma = np.sqrt(sigma2_e)
    theta = np.array([np.sqrt(sigma2_u / sigma2_e)])

    return EMResult(
        theta=theta,
        beta=beta,
        sigma=sigma,
        converged=converged,
        n_iter=iteration + 1,
        final_loglik=prev_loglik,
    )

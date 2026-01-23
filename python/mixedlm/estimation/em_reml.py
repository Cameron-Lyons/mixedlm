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
        Convergence tolerance for log-likelihood change.
    verbose : int, default 0
        Verbosity level (0 = silent, 1 = show iterations).

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
    # Check if model structure is supported
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

    # Extract data
    y = matrices.y.copy()
    X = matrices.X
    Z = matrices.Z.toarray() if hasattr(matrices.Z, "toarray") else matrices.Z
    weights = matrices.weights
    offset = matrices.offset

    y_adj = y - offset
    n = len(y)
    p = X.shape[1]
    q = Z.shape[1]

    # Initialize parameters
    sigma2_e = 1.0  # Residual variance
    sigma2_u = 1.0  # Random effect variance
    beta = np.zeros(p)
    u = np.zeros(q)

    prev_loglik = -np.inf

    for iteration in range(max_iter):
        # E-step: Compute conditional expectations
        # For simplified random intercept: V = sigma2_e * W^{-1} + sigma2_u * Z Z'

        residuals = y_adj - X @ beta

        # Compute V^{-1} * residuals using Woodbury identity
        # V^{-1} = W / sigma2_e - W / sigma2_e * Z * (I / sigma2_u + Z' W Z / sigma2_e)^{-1} * Z' W / sigma2_e

        # Simpler approach for random intercepts: work with grouped structure
        W_sqrt = np.sqrt(weights)
        ZtWZ = Z.T @ (weights[:, None] * Z) / sigma2_e  # Shape: (q, q)
        ZtWr = Z.T @ (weights * residuals) / sigma2_e  # Shape: (q,)

        # (I / sigma2_u + Z' W Z / sigma2_e)
        Gamma_inv = np.eye(q) / sigma2_u + ZtWZ

        try:
            Gamma = linalg.inv(Gamma_inv)
        except linalg.LinAlgError:
            Gamma = linalg.pinv(Gamma_inv)

        # Conditional mean: u_hat = Gamma * Z' W (y - X beta) / sigma2_e
        u_new = Gamma @ ZtWr

        # Conditional variance: Var(u|y) = Gamma
        var_u = np.diag(Gamma)
        var_u = np.maximum(var_u, 1e-10)  # Ensure positive

        # M-step: Update parameters

        # Update fixed effects
        fitted_random = Z @ u_new
        pseudo_y = y_adj - fitted_random
        XtWX = X.T @ (weights[:, None] * X)
        XtWy = X.T @ (weights * pseudo_y)

        try:
            beta_new = linalg.solve(XtWX, XtWy, assume_a="pos")
        except linalg.LinAlgError:
            beta_new = linalg.lstsq(XtWX, XtWy)[0]

        # Update variance components
        residuals_new = y_adj - X @ beta_new - fitted_random
        sse = np.sum(weights * residuals_new**2)

        # Update residual variance (including trace of conditional variance)
        trace_term = np.sum(var_u)  # tr(Gamma)
        sigma2_e_new = (sse + trace_term) / n

        # Update random effect variance
        # E[u'u] = u_hat' u_hat + tr(Gamma)
        u_sq_sum = np.sum(u_new**2) + trace_term
        sigma2_u_new = u_sq_sum / q

        # Ensure positive variance estimates
        sigma2_e_new = max(sigma2_e_new, 1e-10)
        sigma2_u_new = max(sigma2_u_new, 1e-10)

        # Compute log-likelihood (approximate REML criterion)
        residuals_new = y_adj - X @ beta_new - Z @ u_new
        wrss = np.sum(weights * residuals_new**2)
        logdet_Gamma_inv = np.linalg.slogdet(Gamma_inv)[1]

        loglik = -0.5 * (
            (n - p) * np.log(2 * np.pi * sigma2_e_new)
            + wrss / sigma2_e_new
            + logdet_Gamma_inv
            + np.linalg.slogdet(XtWX)[1]  # REML adjustment
        )

        # Check convergence
        loglik_change = abs(loglik - prev_loglik)
        rel_change = loglik_change / (abs(prev_loglik) + 1e-10)

        if verbose >= 1:
            print(
                f"EM iter {iteration + 1}: loglik={loglik:.4f}, "
                f"sigma2_e={sigma2_e_new:.4f}, sigma2_u={sigma2_u_new:.4f}"
            )

        if rel_change < tol and iteration > 5:
            converged = True
            if verbose >= 1:
                print(f"Converged after {iteration + 1} iterations")
            break

        # Update for next iteration
        beta = beta_new
        u = u_new
        sigma2_e = sigma2_e_new
        sigma2_u = sigma2_u_new
        prev_loglik = loglik
    else:
        converged = False
        if verbose >= 1:
            print(f"Did not converge after {max_iter} iterations")

    # Convert to theta parameterization (theta = sigma_u / sigma_e)
    sigma = np.sqrt(sigma2_e)
    theta = np.array([sigma2_u / sigma2_e])

    return EMResult(
        theta=theta,
        beta=beta,
        sigma=sigma,
        converged=converged,
        n_iter=iteration + 1,
        final_loglik=prev_loglik,
    )

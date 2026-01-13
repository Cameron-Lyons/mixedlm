from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

if TYPE_CHECKING:
    from mixedlm.models.lmer import LmerResult

_NUMERICAL_EPS = 1e-6
_GRADIENT_ZERO_THRESHOLD = 1e-10
_MIN_DF = 1.0
_CHOLESKY_REGULARIZATION = 1e-6


@dataclass
class DenomDFResult:
    df: NDArray[np.floating]
    method: str
    param_names: list[str]

    def __getitem__(self, key: str) -> float:
        idx = self.param_names.index(key)
        return float(self.df[idx])

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.param_names, self.df, strict=False))


def _build_lambda_from_theta(
    theta: NDArray[np.floating],
    random_structures: list,
) -> NDArray[np.floating]:
    total_q = sum(s.n_levels * s.n_terms for s in random_structures)
    Lambda = np.zeros((total_q, total_q), dtype=np.float64)

    theta_idx = 0
    u_idx = 0

    for struct in random_structures:
        n_terms = struct.n_terms
        n_levels = struct.n_levels

        if struct.correlated:
            n_theta = n_terms * (n_terms + 1) // 2
        else:
            n_theta = n_terms

        theta_block = theta[theta_idx : theta_idx + n_theta]

        if struct.correlated:
            L_block = np.zeros((n_terms, n_terms), dtype=np.float64)
            idx = 0
            for i in range(n_terms):
                for j in range(i + 1):
                    L_block[i, j] = theta_block[idx]
                    idx += 1
        else:
            L_block = np.diag(theta_block)

        for g in range(n_levels):
            start = u_idx + g * n_terms
            end = start + n_terms
            Lambda[start:end, start:end] = L_block

        theta_idx += n_theta
        u_idx += n_levels * n_terms

    return Lambda


def _compute_vcov_gradient(
    result: LmerResult,
    param_idx: int,
    eps: float = _NUMERICAL_EPS,
) -> NDArray[np.floating]:
    theta = result.theta.copy()
    n_theta = len(theta)
    sigma = result.sigma
    X = result.matrices.X
    Z = result.matrices.Z if result.matrices.Z is not None else None
    n = result.matrices.n_obs
    p = result.matrices.n_fixed
    q = result.matrices.n_random

    def compute_vcov_beta(theta_vec: NDArray, sigma_val: float) -> NDArray:
        if q == 0 or Z is None:
            XtX = X.T @ X
            try:
                return sigma_val**2 * linalg.inv(XtX)
            except linalg.LinAlgError:
                return sigma_val**2 * linalg.pinv(XtX)

        Lambda = _build_lambda_from_theta(theta_vec, result.matrices.random_structures)

        LambdatLambda = Lambda.T @ Lambda
        sigma2 = sigma_val**2

        if sparse.issparse(Z):
            Z_dense = Z.toarray()
        else:
            Z_dense = Z

        ZLambda = Z_dense @ Lambda
        V = np.eye(n) * sigma2 + ZLambda @ LambdatLambda @ ZLambda.T * sigma2

        try:
            L_V = linalg.cholesky(V, lower=True)
            V_inv = linalg.cho_solve((L_V, True), np.eye(n))
        except linalg.LinAlgError:
            V_inv = linalg.pinv(V)

        XtVinvX = X.T @ V_inv @ X
        try:
            return linalg.inv(XtVinvX)
        except linalg.LinAlgError:
            return linalg.pinv(XtVinvX)

    vcov_base = compute_vcov_beta(theta, sigma)

    grad = np.zeros((p, p), dtype=np.float64)

    for k in range(n_theta):
        theta_plus = theta.copy()
        theta_plus[k] += eps
        vcov_plus = compute_vcov_beta(theta_plus, sigma)

        theta_minus = theta.copy()
        theta_minus[k] -= eps
        vcov_minus = compute_vcov_beta(theta_minus, sigma)

        grad += (vcov_plus - vcov_minus) / (2 * eps)

    return grad


def satterthwaite_df(
    result: LmerResult,
    param_idx: int | None = None,
) -> DenomDFResult:
    """Compute Satterthwaite denominator degrees of freedom.

    Uses numerical differentiation to compute the gradient of the
    variance-covariance matrix with respect to variance parameters,
    then applies the Satterthwaite formula.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    param_idx : int, optional
        Index of the fixed effect parameter. If None, computes DF
        for all parameters.

    Returns
    -------
    DenomDFResult
        Object containing denominator degrees of freedom for each parameter.
    """
    vcov = result.vcov()
    p = result.matrices.n_fixed
    n_theta = len(result.theta)

    eps = _NUMERICAL_EPS
    theta = result.theta.copy()
    sigma = result.sigma

    def compute_vcov_for_theta(theta_vec: NDArray) -> NDArray:
        from mixedlm.estimation.reml import _build_lambda

        q = result.matrices.n_random

        if q == 0:
            XtX = result.matrices.X.T @ result.matrices.X
            try:
                return sigma**2 * linalg.inv(XtX)
            except linalg.LinAlgError:
                return sigma**2 * linalg.pinv(XtX)

        Lambda = _build_lambda(theta_vec, result.matrices.random_structures)
        Z = result.matrices.Z
        X = result.matrices.X
        n = result.matrices.n_obs

        if sparse.issparse(Z):
            Zt = Z.T
            ZtZ = Zt @ Z
            LambdatLambda = Lambda.T @ Lambda

            C = ZtZ + LambdatLambda / sigma**2
            if sparse.issparse(C):
                C = C.toarray()

            try:
                L_C = linalg.cholesky(C, lower=True)
            except linalg.LinAlgError:
                C += _CHOLESKY_REGULARIZATION * np.eye(q)
                L_C = linalg.cholesky(C, lower=True)

            XtZ = X.T @ Z
            if sparse.issparse(XtZ):
                XtZ = XtZ.toarray()

            RZX = linalg.solve_triangular(L_C, XtZ.T, lower=True)
            XtX = X.T @ X
            XtVinvX = XtX / sigma**2 - RZX.T @ RZX / sigma**4
        else:
            Lambda_dense = Lambda.toarray() if sparse.issparse(Lambda) else Lambda
            ZLambda = Z @ Lambda_dense
            V = np.eye(n) * sigma**2 + ZLambda @ ZLambda.T * sigma**2

            try:
                V_inv = linalg.inv(V)
            except linalg.LinAlgError:
                V_inv = linalg.pinv(V)

            XtVinvX = X.T @ V_inv @ X

        try:
            return linalg.inv(XtVinvX)
        except linalg.LinAlgError:
            return linalg.pinv(XtVinvX)

    vcov_grads = []
    for k in range(n_theta):
        theta_plus = theta.copy()
        theta_plus[k] += eps
        vcov_plus = compute_vcov_for_theta(theta_plus)

        theta_minus = theta.copy()
        theta_minus[k] -= eps
        vcov_minus = compute_vcov_for_theta(theta_minus)

        vcov_grads.append((vcov_plus - vcov_minus) / (2 * eps))

    df_values = np.zeros(p, dtype=np.float64)

    for i in range(p):
        var_i = vcov[i, i]

        grad_var_i = np.array([g[i, i] for g in vcov_grads])

        grad_sum_sq = np.sum(grad_var_i**2)

        if grad_sum_sq > _GRADIENT_ZERO_THRESHOLD:
            df_values[i] = 2 * var_i**2 / grad_sum_sq
        else:
            df_values[i] = result.matrices.n_obs - result.matrices.n_fixed

        df_values[i] = max(_MIN_DF, min(df_values[i], result.matrices.n_obs - p))

    return DenomDFResult(
        df=df_values,
        method="Satterthwaite",
        param_names=list(result.matrices.fixed_names),
    )


def kenward_roger_df(
    result: LmerResult,
    param_idx: int | None = None,
) -> DenomDFResult:
    """Compute Kenward-Roger denominator degrees of freedom.

    Applies small-sample correction to the Satterthwaite approximation
    by adjusting the variance-covariance matrix.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    param_idx : int, optional
        Index of the fixed effect parameter. If None, computes DF
        for all parameters.

    Returns
    -------
    DenomDFResult
        Object containing denominator degrees of freedom for each parameter.
    """
    vcov = result.vcov()
    p = result.matrices.n_fixed
    n = result.matrices.n_obs
    n_theta = len(result.theta)

    eps = _NUMERICAL_EPS
    theta = result.theta.copy()
    sigma = result.sigma

    satt_result = satterthwaite_df(result)
    df_satt = satt_result.df

    W = np.zeros((p, p), dtype=np.float64)

    def compute_hessian_contribution(theta_vec: NDArray) -> NDArray:
        from mixedlm.estimation.reml import _build_lambda

        q = result.matrices.n_random

        if q == 0:
            return np.zeros((p, p))

        Lambda = _build_lambda(theta_vec, result.matrices.random_structures)
        Z = result.matrices.Z
        X = result.matrices.X

        if sparse.issparse(Z):
            Z_dense = Z.toarray()
        else:
            Z_dense = Z

        if sparse.issparse(Lambda):
            Lambda_dense = Lambda.toarray()
        else:
            Lambda_dense = Lambda

        ZLambda = Z_dense @ Lambda_dense
        V = np.eye(n) * sigma**2 + ZLambda @ ZLambda.T * sigma**2

        try:
            V_inv = linalg.inv(V)
        except linalg.LinAlgError:
            V_inv = linalg.pinv(V)

        P = V_inv - V_inv @ X @ vcov @ X.T @ V_inv

        return X.T @ P @ X

    base_hess = compute_hessian_contribution(theta)

    for k in range(n_theta):
        theta_plus = theta.copy()
        theta_plus[k] += eps
        hess_plus = compute_hessian_contribution(theta_plus)

        theta_minus = theta.copy()
        theta_minus[k] -= eps
        hess_minus = compute_hessian_contribution(theta_minus)

        d2_hess = (hess_plus - 2 * base_hess + hess_minus) / (eps**2)
        W += d2_hess

    W = W / (2 * n_theta) if n_theta > 0 else W

    scale_factor = 1.0 + np.trace(W @ vcov) / p if p > 0 else 1.0
    scale_factor = max(_MIN_DF, scale_factor)

    vcov_adj = vcov * scale_factor

    df_kr = df_satt / scale_factor

    df_kr = np.maximum(_MIN_DF, np.minimum(df_kr, n - p))

    return DenomDFResult(
        df=df_kr,
        method="Kenward-Roger",
        param_names=list(result.matrices.fixed_names),
    )


def pvalues_with_ddf(
    result: LmerResult,
    method: str = "Satterthwaite",
) -> dict[str, tuple[float, float, float]]:
    """Compute p-values using specified denominator DF method.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    method : str, default "Satterthwaite"
        Method for computing denominator degrees of freedom.
        Options: "Satterthwaite", "Kenward-Roger".

    Returns
    -------
    dict
        Dictionary mapping parameter names to (estimate, t-value, p-value) tuples.
    """
    from scipy import stats

    if method == "Satterthwaite":
        ddf_result = satterthwaite_df(result)
    elif method == "Kenward-Roger":
        ddf_result = kenward_roger_df(result)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'Satterthwaite' or 'Kenward-Roger'.")

    vcov = result.vcov()
    beta = result.beta
    se = np.sqrt(np.diag(vcov))

    results: dict[str, tuple[float, float, float]] = {}

    for i, name in enumerate(result.matrices.fixed_names):
        t_val = beta[i] / se[i] if se[i] > 0 else np.nan
        df = ddf_result.df[i]
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_val), df))
        results[name] = (float(beta[i]), float(t_val), float(p_val))

    return results

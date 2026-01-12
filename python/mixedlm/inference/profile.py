from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


@dataclass
class ProfileResult:
    parameter: str
    values: NDArray[np.floating]
    zeta: NDArray[np.floating]
    mle: float
    ci_lower: float
    ci_upper: float
    level: float


def profile_lmer(
    result: LmerResult,
    which: str | list[str] | None = None,
    n_points: int = 20,
    level: float = 0.95,
) -> dict[str, ProfileResult]:
    if which is None:
        which = result.matrices.fixed_names
    elif isinstance(which, str):
        which = [which]

    profiles: dict[str, ProfileResult] = {}
    alpha = 1 - level
    z_crit = stats.norm.ppf(1 - alpha / 2)

    dev_mle = result.deviance

    for param in which:
        if param not in result.matrices.fixed_names:
            continue

        idx = result.matrices.fixed_names.index(param)
        mle = result.beta[idx]

        vcov = result.vcov()
        se = np.sqrt(vcov[idx, idx])

        range_low = mle - 4 * se
        range_high = mle + 4 * se

        param_values = np.linspace(range_low, range_high, n_points)
        zeta_values = np.zeros(n_points)

        for i, val in enumerate(param_values):
            dev = _profile_deviance_at_beta(result, idx, val)
            sign = 1 if val >= mle else -1
            zeta_values[i] = sign * np.sqrt(max(0, dev - dev_mle))

        def zeta_func(val: float, idx: int = idx, mle: float = mle) -> float:
            dev = _profile_deviance_at_beta(result, idx, val)
            sign = 1 if val >= mle else -1
            return sign * np.sqrt(max(0, dev - dev_mle))

        try:
            ci_lower = brentq(
                lambda x: zeta_func(x) + z_crit,
                range_low,
                mle,
            )
        except ValueError:
            ci_lower = mle - z_crit * se

        try:
            ci_upper = brentq(
                lambda x: zeta_func(x) - z_crit,
                mle,
                range_high,
            )
        except ValueError:
            ci_upper = mle + z_crit * se

        profiles[param] = ProfileResult(
            parameter=param,
            values=param_values,
            zeta=zeta_values,
            mle=mle,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            level=level,
        )

    return profiles


def _profile_deviance_at_beta(
    result: LmerResult,
    idx: int,
    value: float,
) -> float:
    from scipy import linalg, sparse

    from mixedlm.estimation.reml import _build_lambda

    matrices = result.matrices
    theta = result.theta
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    X_reduced = np.delete(matrices.X, idx, axis=1)
    y_adjusted = matrices.y - value * matrices.X[:, idx]

    if q == 0:
        XtX = X_reduced.T @ X_reduced
        Xty = X_reduced.T @ y_adjusted
        try:
            beta_reduced = linalg.solve(XtX, Xty, assume_a="pos")
        except linalg.LinAlgError:
            beta_reduced = linalg.lstsq(X_reduced, y_adjusted)[0]

        resid = y_adjusted - X_reduced @ beta_reduced
        rss = np.dot(resid, resid)

        if result.REML:
            sigma2 = rss / (n - p)
            logdet_XtX = np.linalg.slogdet(XtX)[1]
            dev = (n - p) * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_XtX
        else:
            sigma2 = rss / n
            dev = n * (1.0 + np.log(2.0 * np.pi * sigma2))

        return float(dev)

    Lambda = _build_lambda(theta, matrices.random_structures)

    Zt = matrices.Zt
    ZtZ = Zt @ Zt.T
    LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda

    I_q = sparse.eye(q, format="csc")
    V_factor = LambdatZtZLambda + I_q

    V_factor_dense = V_factor.toarray()
    L_V = linalg.cholesky(V_factor_dense, lower=True)

    logdet_V = 2.0 * np.sum(np.log(np.diag(L_V)))

    Zty = Zt @ y_adjusted
    cu = Lambda.T @ Zty
    cu_star = linalg.solve_triangular(L_V, cu, lower=True)

    ZtX = Zt @ X_reduced
    Lambdat_ZtX = Lambda.T @ ZtX
    RZX = linalg.solve_triangular(L_V, Lambdat_ZtX, lower=True)

    XtX = X_reduced.T @ X_reduced
    Xty = X_reduced.T @ y_adjusted

    RZX_tRZX = RZX.T @ RZX
    XtVinvX = XtX - RZX_tRZX

    try:
        L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)
    except linalg.LinAlgError:
        return 1e10

    logdet_XtVinvX = 2.0 * np.sum(np.log(np.diag(L_XtVinvX)))

    cu_star_RZX_beta_term = RZX.T @ cu_star
    Xty_adj = Xty - cu_star_RZX_beta_term
    beta_reduced = linalg.cho_solve((L_XtVinvX, True), Xty_adj)

    resid = y_adjusted - X_reduced @ beta_reduced
    Zt_resid = Zt @ resid
    Lambda_t_Zt_resid = Lambda.T @ Zt_resid
    u_star = linalg.cho_solve((L_V, True), Lambda_t_Zt_resid)

    pwrss = np.dot(resid, resid) + np.dot(u_star, u_star)

    denom = n - p if result.REML else n

    sigma2 = pwrss / denom

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_V
    if result.REML:
        dev += logdet_XtVinvX

    return float(dev)


def profile_glmer(
    result: GlmerResult,
    which: str | list[str] | None = None,
    n_points: int = 20,
    level: float = 0.95,
) -> dict[str, ProfileResult]:
    if which is None:
        which = result.matrices.fixed_names
    elif isinstance(which, str):
        which = [which]

    profiles: dict[str, ProfileResult] = {}
    alpha = 1 - level
    z_crit = stats.norm.ppf(1 - alpha / 2)

    vcov = result.vcov()

    for param in which:
        if param not in result.matrices.fixed_names:
            continue

        idx = result.matrices.fixed_names.index(param)
        mle = result.beta[idx]
        se = np.sqrt(vcov[idx, idx])

        ci_lower = mle - z_crit * se
        ci_upper = mle + z_crit * se

        param_values = np.linspace(mle - 3 * se, mle + 3 * se, n_points)
        zeta_values = (param_values - mle) / se

        profiles[param] = ProfileResult(
            parameter=param,
            values=param_values,
            zeta=zeta_values,
            mle=mle,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            level=level,
        )

    return profiles

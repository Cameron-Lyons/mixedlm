from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq

from mixedlm.inference.profile_types import Profile2DResult, ProfileResult

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

_SLICE2D_PARALLEL_MIN_TASKS = 400


def plot_profiles(
    profiles: dict[str, ProfileResult],
    plot_type: str = "zeta",
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot multiple profile results in a grid.

    Parameters
    ----------
    profiles : dict[str, ProfileResult]
        Dictionary of profile results from profile_lmer or profile_glmer.
    plot_type : str, default "zeta"
        Type of plot: "zeta" for signed sqrt deviance, "density" for density.
    ncols : int, default 2
        Number of columns in the plot grid.
    figsize : tuple, optional
        Figure size. If None, computed automatically.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all profile plots.

    Examples
    --------
    >>> result = lmer("y ~ x1 + x2 + (1 | group)", data)
    >>> profiles = profile_lmer(result)
    >>> plot_profiles(profiles)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting") from None

    n_profiles = len(profiles)
    nrows = (n_profiles + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_profiles == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (_name, profile) in enumerate(profiles.items()):
        if plot_type == "density":
            profile.plot_density(ax=axes[i])
        else:
            profile.plot(ax=axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def splom_profiles(
    profiles: dict[str, ProfileResult],
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Create a scatter plot matrix (pairs plot) of profile zeta values.

    This creates a matrix of plots showing the relationships between
    profile zeta values for different parameters, which can reveal
    correlations and non-linearities in the likelihood surface.

    Parameters
    ----------
    profiles : dict[str, ProfileResult]
        Dictionary of profile results from profile_lmer or profile_glmer.
    figsize : tuple, optional
        Figure size. If None, computed automatically.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the scatter plot matrix.

    Examples
    --------
    >>> result = lmer("y ~ x1 + x2 + (1 | group)", data)
    >>> profiles = profile_lmer(result)
    >>> splom_profiles(profiles)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting") from None

    names = list(profiles.keys())
    n = len(names)

    if n < 2:
        raise ValueError("Need at least 2 profiles for splom plot")

    if figsize is None:
        figsize = (3 * n, 3 * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axes[i, j]

            if i == j:
                profiles[name_i].plot(ax=ax, show_ci=False, show_mle=False)
                ax.set_title("")
                if i == 0:
                    ax.set_title(name_i)
                if j == n - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(name_i)
                else:
                    ax.set_ylabel("")
            else:
                p_i = profiles[name_i]
                p_j = profiles[name_j]

                from scipy.interpolate import interp1d

                try:
                    f_i = interp1d(
                        p_i.zeta,
                        p_i.values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    f_j = interp1d(
                        p_j.zeta,
                        p_j.values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )

                    zeta_common = np.linspace(
                        max(p_i.zeta.min(), p_j.zeta.min()), min(p_i.zeta.max(), p_j.zeta.max()), 50
                    )

                    vals_i = f_i(zeta_common)
                    vals_j = f_j(zeta_common)

                    ax.plot(vals_j, vals_i, "b-", linewidth=1.5)
                    ax.axhline(p_i.mle, color="gray", linestyle="--", alpha=0.3)
                    ax.axvline(p_j.mle, color="gray", linestyle="--", alpha=0.3)
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

            if i < n - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(name_j)

            if j > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(name_i)

    plt.tight_layout()
    return fig


def _profile_param_worker(
    args: tuple[Any, ...],
) -> tuple[str, ProfileResult | None]:
    (
        param,
        idx,
        mle,
        se,
        z_crit,
        level,
        n_points,
        theta,
        y,
        weights,
        X,
        Zt_data,
        Zt_indices,
        Zt_indptr,
        Zt_shape,
        random_structures,
        n,
        p,
        q,
        REML,
    ) = args

    range_low = mle - 4 * se
    range_high = mle + 4 * se

    param_values = np.linspace(range_low, range_high, n_points)
    zeta_values = np.zeros(n_points)

    param_cache = _ProfileCache.from_components(
        idx,
        theta,
        y,
        weights,
        X,
        Zt_data,
        Zt_indices,
        Zt_indptr,
        Zt_shape,
        random_structures,
        n,
        p,
        q,
        REML,
    )
    dev_mle = _profile_deviance_cached(param_cache, mle)

    for i, val in enumerate(param_values):
        dev = _profile_deviance_cached(param_cache, val)
        sign = 1 if val >= mle else -1
        zeta_values[i] = sign * np.sqrt(max(0, dev - dev_mle))

    def zeta_func(val: float) -> float:
        dev = _profile_deviance_cached(param_cache, val)
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

    return (
        param,
        ProfileResult(
            parameter=param,
            values=param_values,
            zeta=zeta_values,
            mle=mle,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            level=level,
        ),
    )


def profile_lmer(
    result: LmerResult,
    which: str | list[str] | None = None,
    n_points: int = 20,
    level: float = 0.95,
    n_jobs: int = 1,
) -> dict[str, ProfileResult]:
    if which is None:
        which = result.matrices.fixed_names
    elif isinstance(which, str):
        which = [which]

    profiles: dict[str, ProfileResult] = {}
    alpha = 1 - level
    z_crit = stats.norm.ppf(1 - alpha / 2)

    vcov = result.vcov()

    matrices = result.matrices
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    if q > 0:
        Zt = matrices.Zt
        Zt_data = np.array(Zt.data)
        Zt_indices = np.array(Zt.indices)
        Zt_indptr = np.array(Zt.indptr)
        Zt_shape = Zt.shape
    else:
        Zt_data = np.array([])
        Zt_indices = np.array([])
        Zt_indptr = np.array([0])
        Zt_shape = (0, n)

    if n_jobs == 1:
        for param in which:
            if param not in result.matrices.fixed_names:
                continue

            idx = result.matrices.fixed_names.index(param)
            mle = result.beta[idx]
            se = np.sqrt(vcov[idx, idx])

            range_low = mle - 4 * se
            range_high = mle + 4 * se

            cache = _ProfileCache.build(result, idx)
            profile_dev_mle = _profile_deviance_cached(cache, mle)

            param_values = np.linspace(range_low, range_high, n_points)
            zeta_values = np.zeros(n_points)

            for i, val in enumerate(param_values):
                dev = _profile_deviance_cached(cache, val)
                sign = 1 if val >= mle else -1
                zeta_values[i] = sign * np.sqrt(max(0, dev - profile_dev_mle))

            def zeta_func(
                val: float,
                c: _ProfileCache = cache,
                m: float = mle,
                d0: float = profile_dev_mle,
            ) -> float:
                dev = _profile_deviance_cached(c, val)
                sign = 1 if val >= m else -1
                return sign * np.sqrt(max(0, dev - d0))

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
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        tasks = []
        for param in which:
            if param not in result.matrices.fixed_names:
                continue

            idx = result.matrices.fixed_names.index(param)
            mle = result.beta[idx]
            se = np.sqrt(vcov[idx, idx])

            tasks.append(
                (
                    param,
                    idx,
                    mle,
                    se,
                    z_crit,
                    level,
                    n_points,
                    result.theta.copy(),
                    (matrices.y - matrices.offset).copy(),
                    matrices.weights.copy(),
                    matrices.X.copy(),
                    Zt_data,
                    Zt_indices,
                    Zt_indptr,
                    Zt_shape,
                    matrices.random_structures,
                    n,
                    p,
                    q,
                    result.REML,
                )
            )

        try:
            executor = ProcessPoolExecutor(max_workers=n_jobs)
        except (NotImplementedError, OSError) as exc:
            warnings.warn(
                "Process-based parallel profiling is unavailable; falling back to serial "
                f"execution ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            for task in tasks:
                param, profile_result = _profile_param_worker(task)
                if profile_result is not None:
                    profiles[param] = profile_result
        else:
            with executor:
                futures = {executor.submit(_profile_param_worker, task): task[0] for task in tasks}

                for future in as_completed(futures):
                    param, profile_result = future.result()
                    if profile_result is not None:
                        profiles[param] = profile_result

    return profiles


@dataclass
class _ProfileProjection:
    """Weighted projection shared by one- and two-parameter profiles."""

    n: int
    q: int
    REML: bool
    y: NDArray[np.floating]
    X_reduced: NDArray[np.floating]
    sqrt_weights: NDArray[np.floating]
    logdet_weights: float
    weighted_X: NDArray[np.floating]
    weighted_Zt: Any | None
    Lambda_T: Any | None
    L_V: NDArray[np.floating] | None
    logdet_V: float
    RZX: NDArray[np.floating] | None
    L_XtVinvX: NDArray[np.floating] | None
    logdet_XtVinvX: float

    @classmethod
    def from_result(cls, result: LmerResult, keep_idx: list[int]) -> _ProfileProjection:
        matrices = result.matrices
        weighted = result._weighted_projection
        information = weighted.XtVinvX[np.ix_(keep_idx, keep_idx)]
        L_XtVinvX, logdet_XtVinvX = _factor_profile_information(
            information,
            result.REML,
        )

        if matrices.n_random == 0:
            weighted_Zt = None
            Lambda_T = None
            L_V = None
            logdet_V = 0.0
            RZX = None
        else:
            if weighted.lambda_matrix is None or weighted.L_V is None:
                raise ValueError("fitted random-effect projection is incomplete")
            weighted_Zt = weighted.weighted_Z.T.tocsc()
            Lambda_T = weighted.lambda_matrix.T
            L_V = weighted.L_V
            logdet_V = float(2.0 * np.sum(np.log(np.diag(L_V))))
            RZX = weighted.RZX[:, keep_idx]

        weights = np.asarray(matrices.weights, dtype=np.float64)
        return cls(
            n=matrices.n_obs,
            q=matrices.n_random,
            REML=result.REML,
            y=matrices.y - matrices.offset,
            X_reduced=matrices.X[:, keep_idx],
            sqrt_weights=weighted.sqrt_weights,
            logdet_weights=float(np.sum(np.log(weights))),
            weighted_X=weighted.weighted_X[:, keep_idx],
            weighted_Zt=weighted_Zt,
            Lambda_T=Lambda_T,
            L_V=L_V,
            logdet_V=logdet_V,
            RZX=RZX,
            L_XtVinvX=L_XtVinvX,
            logdet_XtVinvX=logdet_XtVinvX,
        )

    @classmethod
    def from_components(
        cls,
        theta: NDArray[np.floating],
        y: NDArray[np.floating],
        weights: NDArray[np.floating],
        X: NDArray[np.floating],
        keep_idx: list[int],
        Zt_data: NDArray[np.floating],
        Zt_indices: NDArray[np.int64],
        Zt_indptr: NDArray[np.int64],
        Zt_shape: tuple[int, int],
        random_structures: list[Any],
        n: int,
        q: int,
        REML: bool,
    ) -> _ProfileProjection:
        from scipy import linalg, sparse

        from mixedlm.estimation.reml import _build_lambda
        from mixedlm.matrices.design import validate_prior_weights

        validated_weights = validate_prior_weights(weights, n)
        sqrt_weights = np.sqrt(validated_weights)
        X_reduced = X[:, keep_idx]
        weighted_X = sqrt_weights[:, None] * X_reduced

        if q == 0:
            information = weighted_X.T @ weighted_X
            L_XtVinvX, logdet_XtVinvX = _factor_profile_information(information, REML)
            return cls(
                n=n,
                q=q,
                REML=REML,
                y=y,
                X_reduced=X_reduced,
                sqrt_weights=sqrt_weights,
                logdet_weights=float(np.sum(np.log(validated_weights))),
                weighted_X=weighted_X,
                weighted_Zt=None,
                Lambda_T=None,
                L_V=None,
                logdet_V=0.0,
                RZX=None,
                L_XtVinvX=L_XtVinvX,
                logdet_XtVinvX=logdet_XtVinvX,
            )

        Zt = sparse.csc_matrix((Zt_data, Zt_indices, Zt_indptr), shape=Zt_shape)
        weighted_Zt = Zt.multiply(sqrt_weights[None, :]).tocsc()
        Lambda_T = _build_lambda(theta, random_structures).T
        V_factor = Lambda_T @ (weighted_Zt @ weighted_Zt.T) @ Lambda_T.T
        V_factor = V_factor + sparse.eye(q, format="csc")

        try:
            L_V = linalg.cholesky(V_factor.toarray(), lower=True)
        except linalg.LinAlgError:
            L_V = None

        if L_V is None:
            RZX = None
            L_XtVinvX = None
            logdet_V = 0.0
            logdet_XtVinvX = 0.0
        else:
            logdet_V = float(2.0 * np.sum(np.log(np.diag(L_V))))
            Lambdat_ZtWX = Lambda_T @ (weighted_Zt @ weighted_X)
            RZX = linalg.solve_triangular(L_V, Lambdat_ZtWX, lower=True)
            information = weighted_X.T @ weighted_X - RZX.T @ RZX
            information = (information + information.T) / 2.0
            L_XtVinvX, logdet_XtVinvX = _factor_profile_information(information, REML)

        return cls(
            n=n,
            q=q,
            REML=REML,
            y=y,
            X_reduced=X_reduced,
            sqrt_weights=sqrt_weights,
            logdet_weights=float(np.sum(np.log(validated_weights))),
            weighted_X=weighted_X,
            weighted_Zt=weighted_Zt,
            Lambda_T=Lambda_T,
            L_V=L_V,
            logdet_V=logdet_V,
            RZX=RZX,
            L_XtVinvX=L_XtVinvX,
            logdet_XtVinvX=logdet_XtVinvX,
        )

    def deviance(self, y_adjusted: NDArray[np.floating]) -> float:
        from scipy import linalg

        weighted_y = self.sqrt_weights * y_adjusted
        if self.q == 0:
            if self.X_reduced.shape[1] > 0:
                rhs = self.weighted_X.T @ weighted_y
                if self.L_XtVinvX is not None:
                    beta_reduced = linalg.cho_solve((self.L_XtVinvX, True), rhs)
                else:
                    beta_reduced = linalg.lstsq(self.weighted_X, weighted_y)[0]
            else:
                beta_reduced = np.empty(0, dtype=np.float64)
        else:
            if (
                self.L_XtVinvX is None
                or self.RZX is None
                or self.L_V is None
                or self.weighted_Zt is None
                or self.Lambda_T is None
            ):
                return 1e10

            cu = self.Lambda_T @ (self.weighted_Zt @ weighted_y)
            cu_star = linalg.solve_triangular(self.L_V, cu, lower=True)
            rhs = self.weighted_X.T @ weighted_y - self.RZX.T @ cu_star
            if self.X_reduced.shape[1] > 0:
                beta_reduced = linalg.cho_solve((self.L_XtVinvX, True), rhs)
            else:
                beta_reduced = np.empty(0, dtype=np.float64)

        residual = y_adjusted - self.X_reduced @ beta_reduced
        weighted_residual = self.sqrt_weights * residual
        pwrss = float(np.dot(weighted_residual, weighted_residual))

        if self.q > 0:
            Lambda_t_ZtW_resid = self.Lambda_T @ (self.weighted_Zt @ weighted_residual)
            u_star = linalg.cho_solve((self.L_V, True), Lambda_t_ZtW_resid)
            pwrss -= float(np.dot(Lambda_t_ZtW_resid, u_star))

        denom = self.n - self.X_reduced.shape[1] if self.REML else self.n
        sigma2 = pwrss / denom
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            return 1e10

        deviance = (
            denom * (1.0 + np.log(2.0 * np.pi * sigma2))
            + self.logdet_V
            - self.logdet_weights
        )
        if self.REML:
            deviance += self.logdet_XtVinvX
        return float(deviance)


def _factor_profile_information(
    information: NDArray[np.floating],
    REML: bool,
) -> tuple[NDArray[np.floating] | None, float]:
    from scipy import linalg

    if information.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64), 0.0

    information = (information + information.T) / 2.0
    try:
        factor = linalg.cholesky(information, lower=True)
    except linalg.LinAlgError:
        logdet = float(np.linalg.slogdet(information)[1]) if REML else 0.0
        return None, logdet

    logdet = float(2.0 * np.sum(np.log(np.diag(factor)))) if REML else 0.0
    return factor, logdet


@dataclass
class _ProfileCache:
    """Cache invariant terms for repeated one-parameter profile evaluations."""

    projection: _ProfileProjection
    X_col: NDArray[np.floating]

    @classmethod
    def build(cls, result: LmerResult, idx: int) -> _ProfileCache:
        keep_idx = [column for column in range(result.matrices.n_fixed) if column != idx]
        return cls(
            projection=_ProfileProjection.from_result(result, keep_idx),
            X_col=result.matrices.X[:, idx],
        )

    @classmethod
    def from_components(
        cls,
        idx: int,
        theta: NDArray[np.floating],
        y: NDArray[np.floating],
        weights: NDArray[np.floating],
        X: NDArray[np.floating],
        Zt_data: NDArray[np.floating],
        Zt_indices: NDArray[np.int64],
        Zt_indptr: NDArray[np.int64],
        Zt_shape: tuple[int, int],
        random_structures: list[Any],
        n: int,
        p: int,
        q: int,
        REML: bool,
    ) -> _ProfileCache:
        keep_idx = [column for column in range(p) if column != idx]
        return cls(
            projection=_ProfileProjection.from_components(
                theta,
                y,
                weights,
                X,
                keep_idx,
                Zt_data,
                Zt_indices,
                Zt_indptr,
                Zt_shape,
                random_structures,
                n,
                q,
                REML,
            ),
            X_col=X[:, idx],
        )


def _profile_deviance_cached(cache: _ProfileCache, value: float) -> float:
    return cache.projection.deviance(cache.projection.y - value * cache.X_col)


def _profile_deviance_at_beta(
    result: LmerResult,
    idx: int,
    value: float,
) -> float:
    cache = _ProfileCache.build(result, idx)
    return _profile_deviance_cached(cache, value)


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


def logProf(profile: ProfileResult) -> ProfileResult:
    """Transform profile to log scale for variance components.

    This transformation is useful for variance components, which are
    always positive and often better represented on a log scale.
    The transformation is: log_value = log(value)

    Parameters
    ----------
    profile : ProfileResult
        Original profile result.

    Returns
    -------
    ProfileResult
        Profile with values transformed to log scale.

    Examples
    --------
    >>> profiles = profile_lmer(result)
    >>> log_profile = logProf(profiles["sigma"])
    """
    log_values = np.log(np.maximum(profile.values, 1e-10))
    log_mle = np.log(max(profile.mle, 1e-10))
    log_ci_lower = np.log(max(profile.ci_lower, 1e-10))
    log_ci_upper = np.log(max(profile.ci_upper, 1e-10))

    return ProfileResult(
        parameter=f"log({profile.parameter})",
        values=log_values,
        zeta=profile.zeta,
        mle=log_mle,
        ci_lower=log_ci_lower,
        ci_upper=log_ci_upper,
        level=profile.level,
    )


def varianceProf(profile: ProfileResult) -> ProfileResult:
    """Transform profile to variance scale.

    This transformation squares the values, which is useful when
    the original profile is on the standard deviation scale but
    the variance is desired.

    Parameters
    ----------
    profile : ProfileResult
        Original profile result (typically on SD scale).

    Returns
    -------
    ProfileResult
        Profile with values transformed to variance scale.

    Examples
    --------
    >>> profiles = profile_lmer(result)
    >>> var_profile = varianceProf(profiles["sigma"])
    """
    var_values = profile.values**2
    var_mle = profile.mle**2
    var_ci_lower = profile.ci_lower**2
    var_ci_upper = profile.ci_upper**2

    if var_ci_lower > var_ci_upper:
        var_ci_lower, var_ci_upper = var_ci_upper, var_ci_lower

    return ProfileResult(
        parameter=f"{profile.parameter}²",
        values=var_values,
        zeta=profile.zeta,
        mle=var_mle,
        ci_lower=var_ci_lower,
        ci_upper=var_ci_upper,
        level=profile.level,
    )


def sdProf(profile: ProfileResult) -> ProfileResult:
    """Transform profile to standard deviation scale.

    This transformation takes the square root of the values,
    which is useful when the original profile is on the variance
    scale but the standard deviation is desired.

    Parameters
    ----------
    profile : ProfileResult
        Original profile result (typically on variance scale).

    Returns
    -------
    ProfileResult
        Profile with values transformed to SD scale.

    Examples
    --------
    >>> profiles = profile_lmer(result)
    >>> sd_profile = sdProf(var_profile)
    """
    sd_values = np.sqrt(np.maximum(profile.values, 0))
    sd_mle = np.sqrt(max(profile.mle, 0))
    sd_ci_lower = np.sqrt(max(profile.ci_lower, 0))
    sd_ci_upper = np.sqrt(max(profile.ci_upper, 0))

    return ProfileResult(
        parameter=f"sqrt({profile.parameter})",
        values=sd_values,
        zeta=profile.zeta,
        mle=sd_mle,
        ci_lower=sd_ci_lower,
        ci_upper=sd_ci_upper,
        level=profile.level,
    )


def as_dataframe(
    profiles: dict[str, ProfileResult] | ProfileResult,
) -> pd.DataFrame:
    """Export profile(s) as a pandas DataFrame.

    Parameters
    ----------
    profiles : dict[str, ProfileResult] or ProfileResult
        Either a single ProfileResult or a dictionary of profile results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for parameter, value, and zeta.
        For multiple profiles, includes all parameters stacked.

    Examples
    --------
    >>> profiles = profile_lmer(result)
    >>> df = as_dataframe(profiles)
    >>> # Export to CSV
    >>> df.to_csv("profiles.csv", index=False)
    """
    import pandas as pd

    if isinstance(profiles, ProfileResult):
        profiles = {profiles.parameter: profiles}

    rows = []
    for param, profile in profiles.items():
        for val, zeta in zip(profile.values, profile.zeta, strict=False):
            rows.append(
                {
                    "parameter": param,
                    "value": val,
                    "zeta": zeta,
                    "mle": profile.mle,
                    "ci_lower": profile.ci_lower,
                    "ci_upper": profile.ci_upper,
                    "level": profile.level,
                }
            )

    return pd.DataFrame(rows)


def confint_profile(
    profiles: dict[str, ProfileResult],
    level: float | None = None,
) -> pd.DataFrame:
    """Extract confidence intervals from profile results.

    Parameters
    ----------
    profiles : dict[str, ProfileResult]
        Dictionary of profile results.
    level : float, optional
        Confidence level. If None, uses the level from the profiles.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for parameter, lower, upper, and level.

    Examples
    --------
    >>> profiles = profile_lmer(result)
    >>> ci = confint_profile(profiles)
    """
    import pandas as pd

    rows = []
    for param, profile in profiles.items():
        if level is not None and level != profile.level:
            alpha = 1 - level
            z_crit = stats.norm.ppf(1 - alpha / 2)

            from scipy.interpolate import interp1d

            try:
                f = interp1d(
                    profile.zeta,
                    profile.values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                ci_lower = float(f(-z_crit))
                ci_upper = float(f(z_crit))
            except Exception:
                z_scale = 2 * stats.norm.ppf((1 + profile.level) / 2)
                se = (profile.ci_upper - profile.ci_lower) / z_scale
                ci_lower = profile.mle - z_crit * se
                ci_upper = profile.mle + z_crit * se
            use_level = level
        else:
            ci_lower = profile.ci_lower
            ci_upper = profile.ci_upper
            use_level = profile.level

        rows.append(
            {
                "parameter": param,
                "estimate": profile.mle,
                "lower": ci_lower,
                "upper": ci_upper,
                "level": use_level,
            }
        )

    return pd.DataFrame(rows)


def slice2D(
    result: LmerResult,
    param1: str,
    param2: str,
    n_points: int = 15,
    level: float = 0.95,
    n_jobs: int = 1,
) -> Profile2DResult:
    """Compute 2D profile likelihood slice for two parameters.

    This function evaluates the profile deviance over a grid of values
    for two parameters, while optimizing over all other parameters.
    The result can be used to visualize joint confidence regions and
    parameter correlations.

    Parameters
    ----------
    result : LmerResult
        A fitted linear mixed model.
    param1 : str
        Name of the first parameter.
    param2 : str
        Name of the second parameter.
    n_points : int, default 15
        Number of grid points in each dimension.
    level : float, default 0.95
        Confidence level for the joint region.
    n_jobs : int, default 1
        Number of parallel jobs. Use -1 for all available cores.

    Returns
    -------
    Profile2DResult
        Object containing the 2D profile surface.

    Examples
    --------
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    >>> slice2d = slice2D(result, "(Intercept)", "Days", n_points=10)
    >>> slice2d.plot()
    """
    if param1 not in result.matrices.fixed_names:
        raise ValueError(f"Parameter '{param1}' not found in fixed effects")
    if param2 not in result.matrices.fixed_names:
        raise ValueError(f"Parameter '{param2}' not found in fixed effects")

    idx1 = result.matrices.fixed_names.index(param1)
    idx2 = result.matrices.fixed_names.index(param2)

    vcov = result.vcov()
    mle1 = result.beta[idx1]
    mle2 = result.beta[idx2]
    se1 = np.sqrt(vcov[idx1, idx1])
    se2 = np.sqrt(vcov[idx2, idx2])

    values1 = np.linspace(mle1 - 3 * se1, mle1 + 3 * se1, n_points)
    values2 = np.linspace(mle2 - 3 * se2, mle2 + 3 * se2, n_points)

    reference_cache = _Slice2DCache.build(result, idx1, idx2)
    dev_mle = _profile_deviance_2d_cached(reference_cache, mle1, mle2)

    zeta = np.zeros((n_points, n_points))
    if n_jobs == -1:
        n_jobs_actual = os.cpu_count() or 1
    elif n_jobs > 1:
        n_jobs_actual = n_jobs
    else:
        n_jobs_actual = 1

    use_parallel = n_jobs_actual > 1 and (n_points * n_points) >= _SLICE2D_PARALLEL_MIN_TASKS

    if not use_parallel:
        cache = reference_cache
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                dev = _profile_deviance_2d_cached(cache, v1, v2)
                diff = dev - dev_mle
                sign = 1 if diff >= 0 else -1
                zeta[i, j] = sign * np.sqrt(abs(diff))
    else:
        matrices = result.matrices
        if matrices.n_random > 0:
            Zt = matrices.Zt
            Zt_data = np.array(Zt.data)
            Zt_indices = np.array(Zt.indices)
            Zt_indptr = np.array(Zt.indptr)
            Zt_shape = Zt.shape
        else:
            Zt_data = np.array([])
            Zt_indices = np.array([])
            Zt_indptr = np.array([0])
            Zt_shape = (0, matrices.n_obs)

        tasks = []
        for i, v1 in enumerate(values1):
            tasks.append(
                (
                    i,
                    float(v1),
                    values2,
                    dev_mle,
                    idx1,
                    idx2,
                    result.theta.copy(),
                    (matrices.y - matrices.offset).copy(),
                    matrices.weights.copy(),
                    matrices.X.copy(),
                    Zt_data,
                    Zt_indices,
                    Zt_indptr,
                    Zt_shape,
                    matrices.random_structures,
                    matrices.n_obs,
                    matrices.n_fixed,
                    matrices.n_random,
                    result.REML,
                )
            )

        with ProcessPoolExecutor(max_workers=n_jobs_actual) as executor:
            futures = {executor.submit(_slice2d_row_worker, task): task[0] for task in tasks}
            for future in as_completed(futures):
                i, zeta_row = future.result()
                zeta[i, :] = zeta_row

    return Profile2DResult(
        param1=param1,
        param2=param2,
        values1=values1,
        values2=values2,
        zeta=zeta,
        mle1=mle1,
        mle2=mle2,
        level=level,
    )


def _slice2d_row_worker(
    args: tuple[Any, ...],
) -> tuple[int, NDArray[np.floating]]:
    (
        i,
        v1,
        values2,
        dev_mle,
        idx1,
        idx2,
        theta,
        y,
        weights,
        X,
        Zt_data,
        Zt_indices,
        Zt_indptr,
        Zt_shape,
        random_structures,
        n,
        p,
        q,
        REML,
    ) = args

    cache = _Slice2DCache.from_components(
        idx1,
        idx2,
        theta,
        y,
        weights,
        X,
        Zt_data,
        Zt_indices,
        Zt_indptr,
        Zt_shape,
        random_structures,
        n,
        p,
        q,
        REML,
    )

    row = np.zeros(len(values2), dtype=np.float64)
    for j, v2 in enumerate(values2):
        dev = _profile_deviance_2d_cached(cache, float(v1), float(v2))
        diff = dev - dev_mle
        sign = 1 if diff >= 0 else -1
        row[j] = sign * np.sqrt(abs(diff))

    return i, row


@dataclass
class _Slice2DCache:
    """Cache invariant terms for repeated 2D profile deviance evaluations."""

    projection: _ProfileProjection
    X_col1: NDArray[np.floating]
    X_col2: NDArray[np.floating]

    @classmethod
    def build(cls, result: LmerResult, idx1: int, idx2: int) -> _Slice2DCache:
        matrices = result.matrices
        keep_idx = [column for column in range(matrices.n_fixed) if column not in (idx1, idx2)]
        return cls(
            projection=_ProfileProjection.from_result(result, keep_idx),
            X_col1=matrices.X[:, idx1],
            X_col2=matrices.X[:, idx2],
        )

    @classmethod
    def from_components(
        cls,
        idx1: int,
        idx2: int,
        theta: NDArray[np.floating],
        y: NDArray[np.floating],
        weights: NDArray[np.floating],
        X: NDArray[np.floating],
        Zt_data: NDArray[np.floating],
        Zt_indices: NDArray[np.int64],
        Zt_indptr: NDArray[np.int64],
        Zt_shape: tuple[int, int],
        random_structures: list[Any],
        n: int,
        p: int,
        q: int,
        REML: bool,
    ) -> _Slice2DCache:
        keep_idx = [column for column in range(p) if column not in (idx1, idx2)]
        return cls(
            projection=_ProfileProjection.from_components(
                theta,
                y,
                weights,
                X,
                keep_idx,
                Zt_data,
                Zt_indices,
                Zt_indptr,
                Zt_shape,
                random_structures,
                n,
                q,
                REML,
            ),
            X_col1=X[:, idx1],
            X_col2=X[:, idx2],
        )


def _profile_deviance_2d_cached(
    cache: _Slice2DCache,
    value1: float,
    value2: float,
) -> float:
    y_adjusted = cache.projection.y - value1 * cache.X_col1 - value2 * cache.X_col2
    return cache.projection.deviance(y_adjusted)


def _profile_deviance_2d(
    result: LmerResult,
    idx1: int,
    value1: float,
    idx2: int,
    value2: float,
) -> float:
    """Compute deviance with two fixed effects parameters held constant."""
    cache = _Slice2DCache.build(result, idx1, idx2)
    return _profile_deviance_2d_cached(cache, value1, value2)

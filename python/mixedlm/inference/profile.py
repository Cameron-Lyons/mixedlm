from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq

if TYPE_CHECKING:
    import pandas as pd

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

    def plot(
        self,
        ax: Any | None = None,
        show_ci: bool = True,
        show_mle: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot the profile likelihood.

        Creates a plot of the signed square root deviance (zeta)
        against the parameter values. This is useful for assessing
        the symmetry of the likelihood and identifying non-normality.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show_ci : bool, default True
            Whether to show confidence interval lines.
        show_mle : bool, default True
            Whether to show vertical line at MLE.
        **kwargs
            Additional arguments passed to plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the profile plot.

        Examples
        --------
        >>> result = lmer("y ~ x + (1 | group)", data)
        >>> profiles = profile_lmer(result, which=["x"])
        >>> profiles["x"].plot()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting") from None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(self.values, self.zeta, "b-", linewidth=2, **kwargs)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        if show_mle:
            ax.axvline(self.mle, color="red", linestyle="--", alpha=0.7, label="MLE")

        if show_ci:
            z_crit = stats.norm.ppf((1 + self.level) / 2)
            ax.axhline(z_crit, color="green", linestyle=":", alpha=0.7)
            ax.axhline(-z_crit, color="green", linestyle=":", alpha=0.7)
            ax.axvline(self.ci_lower, color="green", linestyle=":", alpha=0.5)
            ax.axvline(self.ci_upper, color="green", linestyle=":", alpha=0.5)

        ax.set_xlabel(self.parameter)
        ax.set_ylabel("ζ (signed sqrt deviance)")
        ax.set_title(f"Profile: {self.parameter}")

        return ax

    def plot_density(
        self,
        ax: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the profile-based density.

        Creates a density plot derived from the profile likelihood,
        which can show deviations from normality.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional arguments passed to plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the density plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting") from None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        density = np.exp(-0.5 * self.zeta**2)
        density = density / np.trapezoid(density, self.values)

        ax.plot(self.values, density, "b-", linewidth=2, **kwargs)
        ax.fill_between(self.values, density, alpha=0.3)

        ax.axvline(self.mle, color="red", linestyle="--", alpha=0.7, label="MLE")
        ax.axvline(self.ci_lower, color="green", linestyle=":", alpha=0.5)
        ax.axvline(self.ci_upper, color="green", linestyle=":", alpha=0.5)

        ax.set_xlabel(self.parameter)
        ax.set_ylabel("Density")
        ax.set_title(f"Profile density: {self.parameter}")

        return ax


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
        dev_mle,
        z_crit,
        level,
        n_points,
        theta,
        y,
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

    for i, val in enumerate(param_values):
        dev = _profile_deviance_at_beta_direct(
            idx,
            val,
            theta,
            y,
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
        sign = 1 if val >= mle else -1
        zeta_values[i] = sign * np.sqrt(max(0, dev - dev_mle))

    def zeta_func(val: float) -> float:
        dev = _profile_deviance_at_beta_direct(
            idx,
            val,
            theta,
            y,
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


def _profile_deviance_at_beta_direct(
    idx: int,
    value: float,
    theta: NDArray,
    y: NDArray,
    X: NDArray,
    Zt_data: NDArray,
    Zt_indices: NDArray,
    Zt_indptr: NDArray,
    Zt_shape: tuple[int, int],
    random_structures: list,
    n: int,
    p: int,
    q: int,
    REML: bool,
) -> float:
    from scipy import linalg, sparse

    from mixedlm.estimation.reml import _build_lambda

    X_reduced = np.delete(X, idx, axis=1)
    y_adjusted = y - value * X[:, idx]

    if q == 0:
        XtX = X_reduced.T @ X_reduced
        Xty = X_reduced.T @ y_adjusted
        try:
            beta_reduced = linalg.solve(XtX, Xty, assume_a="pos")
        except linalg.LinAlgError:
            beta_reduced = linalg.lstsq(X_reduced, y_adjusted)[0]

        resid = y_adjusted - X_reduced @ beta_reduced
        rss = np.dot(resid, resid)

        if REML:
            sigma2 = rss / (n - p)
            logdet_XtX = np.linalg.slogdet(XtX)[1]
            dev = (n - p) * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_XtX
        else:
            sigma2 = rss / n
            dev = n * (1.0 + np.log(2.0 * np.pi * sigma2))

        return float(dev)

    Zt = sparse.csc_matrix((Zt_data, Zt_indices, Zt_indptr), shape=Zt_shape)
    Lambda = _build_lambda(theta, random_structures)

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

    denom = n - p if REML else n

    sigma2 = pwrss / denom

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_V
    if REML:
        dev += logdet_XtVinvX

    return float(dev)


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

    dev_mle = result.deviance
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

            param_values = np.linspace(range_low, range_high, n_points)
            zeta_values = np.zeros(n_points)

            for i, val in enumerate(param_values):
                dev = _profile_deviance_cached(cache, val)
                sign = 1 if val >= mle else -1
                zeta_values[i] = sign * np.sqrt(max(0, dev - dev_mle))

            def zeta_func(val: float, c: _ProfileCache = cache, m: float = mle) -> float:
                dev = _profile_deviance_cached(c, val)
                sign = 1 if val >= m else -1
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
                    dev_mle,
                    z_crit,
                    level,
                    n_points,
                    result.theta.copy(),
                    matrices.y.copy(),
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

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_profile_param_worker, task): task[0] for task in tasks}

            for future in as_completed(futures):
                param, profile_result = future.result()
                if profile_result is not None:
                    profiles[param] = profile_result

    return profiles


@dataclass
class _ProfileCache:
    """Cache for invariant computations in profile likelihood."""

    n: int
    p: int
    q: int
    REML: bool
    X_reduced: NDArray[np.floating]
    X_col: NDArray[np.floating]
    y: NDArray[np.floating]
    logdet_V: float
    L_V: NDArray[np.floating] | None
    Lambda_T: Any | None
    Zt: Any | None
    RZX: NDArray[np.floating] | None
    L_XtVinvX: NDArray[np.floating] | None
    logdet_XtVinvX: float
    XtX_chol: NDArray[np.floating] | None
    logdet_XtX: float

    @classmethod
    def build(cls, result: LmerResult, idx: int) -> _ProfileCache:
        from scipy import linalg, sparse

        from mixedlm.estimation.reml import _build_lambda

        matrices = result.matrices
        theta = result.theta
        n = matrices.n_obs
        p = matrices.n_fixed
        q = matrices.n_random

        X_reduced = np.delete(matrices.X, idx, axis=1)
        X_col = matrices.X[:, idx]

        if q == 0:
            XtX = X_reduced.T @ X_reduced
            try:
                XtX_chol = linalg.cholesky(XtX, lower=True)
            except linalg.LinAlgError:
                XtX_chol = None
            logdet_XtX = np.linalg.slogdet(XtX)[1] if result.REML else 0.0

            return cls(
                n=n,
                p=p,
                q=q,
                REML=result.REML,
                X_reduced=X_reduced,
                X_col=X_col,
                y=matrices.y,
                logdet_V=0.0,
                L_V=None,
                Lambda_T=None,
                Zt=None,
                RZX=None,
                L_XtVinvX=None,
                logdet_XtVinvX=0.0,
                XtX_chol=XtX_chol,
                logdet_XtX=logdet_XtX,
            )

        Lambda = _build_lambda(theta, matrices.random_structures)
        Zt = matrices.Zt
        ZtZ = Zt @ Zt.T
        LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V_factor = LambdatZtZLambda + I_q
        V_factor_dense = V_factor.toarray()
        L_V = linalg.cholesky(V_factor_dense, lower=True)
        logdet_V = 2.0 * np.sum(np.log(np.diag(L_V)))

        ZtX = Zt @ X_reduced
        Lambdat_ZtX = Lambda.T @ ZtX
        RZX = linalg.solve_triangular(L_V, Lambdat_ZtX, lower=True)

        XtX = X_reduced.T @ X_reduced
        RZX_tRZX = RZX.T @ RZX
        XtVinvX = XtX - RZX_tRZX

        try:
            L_XtVinvX = linalg.cholesky(XtVinvX, lower=True)
            logdet_XtVinvX = 2.0 * np.sum(np.log(np.diag(L_XtVinvX)))
        except linalg.LinAlgError:
            L_XtVinvX = None
            logdet_XtVinvX = 0.0

        return cls(
            n=n,
            p=p,
            q=q,
            REML=result.REML,
            X_reduced=X_reduced,
            X_col=X_col,
            y=matrices.y,
            logdet_V=logdet_V,
            L_V=L_V,
            Lambda_T=Lambda.T,
            Zt=Zt,
            RZX=RZX,
            L_XtVinvX=L_XtVinvX,
            logdet_XtVinvX=logdet_XtVinvX,
            XtX_chol=None,
            logdet_XtX=0.0,
        )


def _profile_deviance_cached(cache: _ProfileCache, value: float) -> float:
    from scipy import linalg

    y_adjusted = cache.y - value * cache.X_col

    if cache.q == 0:
        Xty = cache.X_reduced.T @ y_adjusted
        if cache.XtX_chol is not None:
            beta_reduced = linalg.cho_solve((cache.XtX_chol, True), Xty)
        else:
            beta_reduced = linalg.lstsq(cache.X_reduced, y_adjusted)[0]

        resid = y_adjusted - cache.X_reduced @ beta_reduced
        rss = np.dot(resid, resid)

        if cache.REML:
            sigma2 = rss / (cache.n - cache.p)
            dev = (cache.n - cache.p) * (1.0 + np.log(2.0 * np.pi * sigma2)) + cache.logdet_XtX
        else:
            sigma2 = rss / cache.n
            dev = cache.n * (1.0 + np.log(2.0 * np.pi * sigma2))

        return float(dev)

    if cache.L_XtVinvX is None or cache.RZX is None or cache.L_V is None:
        return 1e10

    Zty = cache.Zt @ y_adjusted
    cu = cache.Lambda_T @ Zty
    cu_star = linalg.solve_triangular(cache.L_V, cu, lower=True)

    Xty = cache.X_reduced.T @ y_adjusted
    cu_star_RZX_beta_term = cache.RZX.T @ cu_star
    Xty_adj = Xty - cu_star_RZX_beta_term
    beta_reduced = linalg.cho_solve((cache.L_XtVinvX, True), Xty_adj)

    resid = y_adjusted - cache.X_reduced @ beta_reduced
    Zt_resid = cache.Zt @ resid
    Lambda_t_Zt_resid = cache.Lambda_T @ Zt_resid
    u_star = linalg.cho_solve((cache.L_V, True), Lambda_t_Zt_resid)

    pwrss = np.dot(resid, resid) + np.dot(u_star, u_star)
    denom = cache.n - cache.p if cache.REML else cache.n
    sigma2 = pwrss / denom

    dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + cache.logdet_V
    if cache.REML:
        dev += cache.logdet_XtVinvX

    return float(dev)


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


@dataclass
class Profile2DResult:
    """Result of 2D profile likelihood slice.

    Represents the profile likelihood surface over a 2D grid of
    parameter values, useful for visualizing parameter correlations
    and joint confidence regions.
    """

    param1: str
    param2: str
    values1: NDArray[np.floating]
    values2: NDArray[np.floating]
    zeta: NDArray[np.floating]
    mle1: float
    mle2: float
    level: float

    def plot(
        self,
        ax: Any | None = None,
        show_ci: bool = True,
        show_mle: bool = True,
        n_levels: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Plot the 2D profile likelihood surface as contours.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show_ci : bool, default True
            Whether to highlight the confidence region.
        show_mle : bool, default True
            Whether to show the MLE point.
        n_levels : int, default 10
            Number of contour levels.
        **kwargs
            Additional arguments passed to contour().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the 2D profile plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting") from None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        V1, V2 = np.meshgrid(self.values1, self.values2, indexing="ij")

        contour = ax.contour(V2, V1, self.zeta, levels=n_levels, **kwargs)
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

        if show_ci:
            z_crit_sq = stats.chi2.ppf(self.level, df=2)
            ax.contour(
                V2,
                V1,
                self.zeta**2,
                levels=[z_crit_sq],
                colors="red",
                linewidths=2,
                linestyles="--",
            )

        if show_mle:
            ax.plot(self.mle2, self.mle1, "ro", markersize=8, label="MLE")

        ax.set_xlabel(self.param2)
        ax.set_ylabel(self.param1)
        ax.set_title(f"2D Profile: {self.param1} vs {self.param2}")

        return ax

    def plot_filled(
        self,
        ax: Any | None = None,
        show_ci: bool = True,
        show_mle: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot the 2D profile as a filled contour plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show_ci : bool, default True
            Whether to highlight the confidence region boundary.
        show_mle : bool, default True
            Whether to show the MLE point.
        **kwargs
            Additional arguments passed to contourf().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the filled contour plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting") from None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        V1, V2 = np.meshgrid(self.values1, self.values2, indexing="ij")

        lik_surface = np.exp(-0.5 * self.zeta**2)

        contourf = ax.contourf(V2, V1, lik_surface, levels=20, cmap="viridis", **kwargs)
        plt.colorbar(contourf, ax=ax, label="Relative likelihood")

        if show_ci:
            z_crit_sq = stats.chi2.ppf(self.level, df=2)
            ax.contour(
                V2,
                V1,
                self.zeta**2,
                levels=[z_crit_sq],
                colors="white",
                linewidths=2,
                linestyles="--",
            )

        if show_mle:
            ax.plot(self.mle2, self.mle1, "w*", markersize=12, label="MLE")

        ax.set_xlabel(self.param2)
        ax.set_ylabel(self.param1)
        ax.set_title(f"2D Profile Likelihood: {self.param1} vs {self.param2}")

        return ax


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

    dev_mle = result.deviance

    zeta = np.zeros((n_points, n_points))

    if n_jobs == 1:
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                dev = _profile_deviance_2d(result, idx1, v1, idx2, v2)
                diff = dev - dev_mle
                sign = 1 if diff >= 0 else -1
                zeta[i, j] = sign * np.sqrt(abs(diff))
    else:
        n_jobs_actual = os.cpu_count() or 1 if n_jobs == -1 else n_jobs

        tasks = []
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                tasks.append((i, j, v1, v2, idx1, idx2, result))

        with ProcessPoolExecutor(max_workers=n_jobs_actual) as executor:
            futures = {executor.submit(_slice2d_worker, task): task[:2] for task in tasks}
            for future in as_completed(futures):
                i, j, dev = future.result()
                diff = dev - dev_mle
                sign = 1 if diff >= 0 else -1
                zeta[i, j] = sign * np.sqrt(abs(diff))

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


def _slice2d_worker(
    args: tuple[Any, ...],
) -> tuple[int, int, float]:
    i, j, v1, v2, idx1, idx2, result = args
    dev = _profile_deviance_2d(result, idx1, v1, idx2, v2)
    return i, j, dev


def _profile_deviance_2d(
    result: LmerResult,
    idx1: int,
    value1: float,
    idx2: int,
    value2: float,
) -> float:
    """Compute deviance with two fixed effects parameters held constant."""
    from scipy import linalg, sparse

    from mixedlm.estimation.reml import _build_lambda

    matrices = result.matrices
    theta = result.theta
    n = matrices.n_obs
    p = matrices.n_fixed
    q = matrices.n_random

    keep_idx = [i for i in range(p) if i not in (idx1, idx2)]
    X_reduced = matrices.X[:, keep_idx]
    y_adjusted = matrices.y - value1 * matrices.X[:, idx1] - value2 * matrices.X[:, idx2]

    p_reduced = len(keep_idx)

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
            denom = n - p
            sigma2 = rss / denom
            logdet_XtX = np.linalg.slogdet(XtX)[1] if p_reduced > 0 else 0.0
            dev = denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + logdet_XtX
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
    try:
        L_V = linalg.cholesky(V_factor_dense, lower=True)
    except linalg.LinAlgError:
        return 1e10

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

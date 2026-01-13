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


def _profile_grid_worker(
    args: tuple[Any, ...],
) -> tuple[int, float]:
    from scipy import linalg, sparse

    from mixedlm.estimation.reml import _build_lambda

    (
        grid_idx,
        val,
        idx,
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

    X_reduced = np.delete(X, idx, axis=1)
    y_adjusted = y - val * X[:, idx]

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

        return (grid_idx, float(dev))

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
        return (grid_idx, 1e10)

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

    return (grid_idx, float(dev))


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

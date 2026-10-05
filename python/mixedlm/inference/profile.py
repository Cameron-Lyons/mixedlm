from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mixedlm._parallel import process_pool, resolve_n_jobs
from mixedlm.inference.profile_types import Profile2DResult, ProfileResult
from mixedlm.models.shared_utils import _RandomEffectFactor
from mixedlm.utils.validation import _validate_confidence_level

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

# Starting worker processes can take a second. A conditional slice point is one
# linear solve whose cost depends on the data, the random effects and the BLAS
# threads of this process, so slice2D times its first row and sends the other
# rows to workers only when they would take at least this long serially.
_SLICE2D_PARALLEL_MIN_SECONDS = 1.0


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


def profile_lmer(
    result: LmerResult,
    which: str | list[str] | None = None,
    n_points: int = 20,
    level: float = 0.95,
    n_jobs: int = 1,
) -> dict[str, ProfileResult]:
    """Profile fixed coefficients by re-optimizing covariance and residual scale.

    Use an ML reference fit for both ML and REML inputs. Plotting resolution does
    not control endpoint accuracy. Failed fits or unbracketed intervals raise.
    ``n_jobs`` worker processes, or -1 for all CPUs, profile coefficients
    concurrently. Workers are started without forking, so scripts need an
    ``if __name__ == "__main__":`` guard.
    """
    from mixedlm.inference.lmm_profile import likelihood_profiles

    return likelihood_profiles(result, which, n_points, level, n_jobs)


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
    random_factor: _RandomEffectFactor | None
    logdet_V: float
    random_fixed_map: NDArray[np.floating] | None
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
            random_factor = None
            logdet_V = 0.0
            random_fixed_map = None
        else:
            if weighted.lambda_matrix is None or weighted.random_factor is None:
                raise ValueError("fitted random-effect projection is incomplete")
            weighted_Zt = weighted.weighted_Z.T.tocsc()
            Lambda_T = weighted.lambda_matrix.T
            random_factor = weighted.random_factor
            logdet_V = random_factor.logdet
            random_fixed_map = weighted.random_fixed_map[:, keep_idx]

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
            random_factor=random_factor,
            logdet_V=logdet_V,
            random_fixed_map=random_fixed_map,
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
                or self.random_fixed_map is None
                or self.random_factor is None
                or self.weighted_Zt is None
                or self.Lambda_T is None
            ):
                return 1e10

            if self.X_reduced.shape[1] > 0:
                cu = self.Lambda_T @ (self.weighted_Zt @ weighted_y)
                rhs = self.weighted_X.T @ weighted_y - self.random_fixed_map.T @ cu
                beta_reduced = linalg.cho_solve((self.L_XtVinvX, True), rhs)
            else:
                beta_reduced = np.empty(0, dtype=np.float64)

        residual = y_adjusted - self.X_reduced @ beta_reduced
        weighted_residual = self.sqrt_weights * residual
        pwrss = float(np.dot(weighted_residual, weighted_residual))

        if self.q > 0:
            assert self.random_factor is not None
            Lambda_t_ZtW_resid = self.Lambda_T @ (self.weighted_Zt @ weighted_residual)
            u_star = self.random_factor.solve(Lambda_t_ZtW_resid)
            pwrss -= float(np.dot(Lambda_t_ZtW_resid, u_star))

        denom = self.n - self.X_reduced.shape[1] if self.REML else self.n
        sigma2 = pwrss / denom
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            return 1e10

        deviance = (
            denom * (1.0 + np.log(2.0 * np.pi * sigma2)) + self.logdet_V - self.logdet_weights
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


def _profile_deviance_at_beta(result: LmerResult, idx: int, value: float) -> float:
    """Evaluate a conditional slice at the fitted covariance parameters."""
    keep = [column for column in range(result.matrices.n_fixed) if column != idx]
    projection = _ProfileProjection.from_result(result, keep)
    return projection.deviance(projection.y - value * result.matrices.X[:, idx])


def profile_glmer(
    result: GlmerResult,
    which: str | list[str] | None = None,
    n_points: int = 20,
    level: float = 0.95,
) -> dict[str, ProfileResult]:
    """Profile fixed effects using constrained integrated-likelihood fits.

    Other fixed coefficients and covariance parameters are re-optimized at
    every point. The joint optimum is refined before profiling, so a profile's
    ``mle`` can differ from the original fit's PIRLS coefficient estimate.
    Quadrature and inner solver controls are inherited from the fitted model.
    ``n_points`` controls plotting resolution; interval endpoints are solved
    independently using the likelihood-ratio cutoff. Failed fits or unbracketed
    intervals raise an error instead of substituting a Wald interval.
    """
    from mixedlm.inference.glmm_profile import likelihood_profiles

    return likelihood_profiles(result, which, n_points, level)


def _transform_profile(
    profile: ProfileResult, transform: Callable[[Any], Any], parameter: str
) -> ProfileResult:
    """Map a profile's points and bounds through a monotone transform, keeping zeta."""
    lower, upper = float(transform(profile.ci_lower)), float(transform(profile.ci_upper))
    if lower > upper:
        lower, upper = upper, lower
    return ProfileResult(
        parameter=parameter,
        values=transform(profile.values),
        zeta=profile.zeta,
        mle=float(transform(profile.mle)),
        ci_lower=lower,
        ci_upper=upper,
        level=profile.level,
    )


def _profile_points(profile: ProfileResult) -> NDArray[np.floating]:
    return np.append(profile.values, [profile.mle, profile.ci_lower, profile.ci_upper])


def logProf(profile: ProfileResult) -> ProfileResult:
    """Transform a profile of a positive parameter to the log scale.

    ``profile_lmer`` and ``profile_glmer`` profile fixed effects, which can be
    negative. These scale transforms apply to profiles of positive scale
    parameters, such as a ``ProfileResult`` built for a standard deviation.

    Parameters
    ----------
    profile : ProfileResult
        Profile whose values, MLE and confidence bounds are all positive.

    Returns
    -------
    ProfileResult
        Profile of ``log(parameter)`` with the original zeta values.

    Raises
    ------
    ValueError
        If any value, the MLE or a confidence bound is not positive.

    Examples
    --------
    >>> sd = ProfileResult(
    ...     parameter="sigma",
    ...     values=np.array([1.5, 2.0, 2.5]),
    ...     zeta=np.array([-1.0, 0.0, 1.0]),
    ...     mle=2.0,
    ...     ci_lower=1.6,
    ...     ci_upper=2.4,
    ...     level=0.95,
    ... )
    >>> logProf(sd).parameter
    'log(sigma)'
    """
    if np.any(_profile_points(profile) <= 0):
        raise ValueError("logProf requires a profile of a positive parameter")
    return _transform_profile(profile, np.log, f"log({profile.parameter})")


def varianceProf(profile: ProfileResult) -> ProfileResult:
    """Transform a standard-deviation profile to the variance scale.

    Squaring is monotone only on one side of zero, so the profile must not
    change sign. See :func:`logProf` for the profiles these transforms suit.

    Parameters
    ----------
    profile : ProfileResult
        Profile on the standard-deviation scale.

    Returns
    -------
    ProfileResult
        Profile of the squared parameter with the original zeta values.

    Raises
    ------
    ValueError
        If the values, MLE and confidence bounds include both signs.

    Examples
    --------
    >>> var_profile = varianceProf(sd)
    >>> var_profile.mle
    4.0
    """
    points = _profile_points(profile)
    if np.any(points < 0) and np.any(points > 0):
        raise ValueError("varianceProf requires a profile that does not change sign")
    return _transform_profile(profile, np.square, f"{profile.parameter}²")


def sdProf(profile: ProfileResult) -> ProfileResult:
    """Transform a variance profile to the standard-deviation scale.

    See :func:`logProf` for the profiles these transforms suit.

    Parameters
    ----------
    profile : ProfileResult
        Profile on the variance scale, with nonnegative values, MLE and
        confidence bounds.

    Returns
    -------
    ProfileResult
        Profile of ``sqrt(parameter)`` with the original zeta values.

    Raises
    ------
    ValueError
        If any value, the MLE or a confidence bound is negative.

    Examples
    --------
    >>> sd_profile = sdProf(var_profile)
    """
    if np.any(_profile_points(profile) < 0):
        raise ValueError("sdProf requires a profile of a nonnegative parameter")
    return _transform_profile(profile, np.sqrt, f"sqrt({profile.parameter})")


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

    if level is not None:
        level = _validate_confidence_level(level)
    rows = []
    for param, profile in profiles.items():
        if level is not None and level != profile.level:
            alpha = 1 - level
            z_crit = stats.norm.isf(alpha / 2)

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
                z_scale = 2 * stats.norm.isf((1 - float(profile.level)) / 2)
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
    *,
    profile_covariance: bool = False,
) -> Profile2DResult:
    """Compute a conditional slice or full likelihood profile for two parameters.

    This function evaluates the profile deviance over a grid of values
    for two fixed effects, while recomputing the remaining fixed effects and
    residual scale at the fitted covariance parameters. Set profile_covariance
    to True to also re-optimize covariance parameters using ML, including for
    REML inputs. Use that full profile for likelihood-ratio confidence regions.

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
        Number of worker processes, or -1 for all available cores. Conditional
        slices that would take less than about a second in this process run
        serially. Workers are started without forking, so scripts need an
        ``if __name__ == "__main__":`` guard.
    profile_covariance : bool, default False
        Re-optimize covariance parameters at each pair of fixed coefficients.
        True uses an ML reference and ranges covering the requested joint
        likelihood-ratio region. False retains the faster conditional slice.

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
    if not isinstance(profile_covariance, (bool, np.bool_)):
        raise ValueError("profile_covariance must be a boolean")
    if param1 == param2:
        raise ValueError("The two profile parameters must be distinct")
    if profile_covariance:
        from mixedlm.inference.lmm_profile import likelihood_surface

        return likelihood_surface(result, param1, param2, n_points, level, n_jobs)

    if param1 not in result.matrices.fixed_names:
        raise ValueError(f"Parameter '{param1}' not found in fixed effects")
    if param2 not in result.matrices.fixed_names:
        raise ValueError(f"Parameter '{param2}' not found in fixed effects")

    from mixedlm.utils.names import _check_unique_coefficient_names

    _check_unique_coefficient_names(
        result.matrices.fixed_names,
        [param1, param2],
        alternative="Rename colliding formula variables before requesting named profile slices.",
    )

    workers = resolve_n_jobs(n_jobs, max_tasks=n_points - 1)
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
    zeta_row = partial(_slice2d_zeta_row, reference_cache, dev_mle, values2)

    started = time.perf_counter()
    rows = list(map(zeta_row, values1[:1]))
    remaining = values1[1:]
    serial_seconds = (time.perf_counter() - started) * len(remaining)
    if workers > 1 and serial_seconds >= _SLICE2D_PARALLEL_MIN_SECONDS:
        # One chunk of rows per worker sends the cached projection once to each.
        with process_pool(workers) as executor:
            chunksize = -(-len(remaining) // workers)
            rows.extend(executor.map(zeta_row, remaining, chunksize=chunksize))
    else:
        rows.extend(map(zeta_row, remaining))
    zeta = np.array(rows, dtype=np.float64).reshape(n_points, n_points)

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


def _slice2d_zeta_row(
    cache: _Slice2DCache,
    dev_mle: float,
    values2: NDArray[np.floating],
    value1: float,
) -> NDArray[np.floating]:
    row = np.empty(len(values2), dtype=np.float64)
    for j, value2 in enumerate(values2):
        diff = _profile_deviance_2d_cached(cache, float(value1), float(value2)) - dev_mle
        sign = 1 if diff >= 0 else -1
        row[j] = sign * np.sqrt(abs(diff))
    return row


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

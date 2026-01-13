from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install mixedlm[plots]"
        )


def plot_resid_fitted(
    result: LmerResult | GlmerResult,
    ax: Axes | None = None,
    lowess: bool = True,
    point_kws: dict[str, Any] | None = None,
    line_kws: dict[str, Any] | None = None,
) -> Axes:
    """Plot residuals vs fitted values.

    Parameters
    ----------
    result : LmerResult or GlmerResult
        Fitted model result.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new axes.
    lowess : bool, default True
        Whether to add a lowess smoothing line.
    point_kws : dict, optional
        Keyword arguments passed to scatter plot.
    line_kws : dict, optional
        Keyword arguments passed to lowess line plot.

    Returns
    -------
    Axes
        The matplotlib axes with the plot.
    """
    _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fitted = result.fitted()
    resid = result.residuals(type="response")

    point_defaults: dict[str, Any] = {"alpha": 0.6, "s": 30, "edgecolors": "none"}
    if point_kws:
        point_defaults.update(point_kws)

    ax.scatter(fitted, resid, **point_defaults)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    if lowess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

            sorted_idx = np.argsort(fitted)
            smoothed = sm_lowess(resid[sorted_idx], fitted[sorted_idx], frac=0.6)

            line_defaults: dict[str, Any] = {"color": "red", "linewidth": 2}
            if line_kws:
                line_defaults.update(line_kws)

            ax.plot(smoothed[:, 0], smoothed[:, 1], **line_defaults)
        except ImportError:
            pass

    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    return ax


def plot_qq(
    result: LmerResult | GlmerResult,
    ax: Axes | None = None,
    standardize: bool = True,
    point_kws: dict[str, Any] | None = None,
    line_kws: dict[str, Any] | None = None,
) -> Axes:
    """Q-Q plot of residuals.

    Parameters
    ----------
    result : LmerResult or GlmerResult
        Fitted model result.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new axes.
    standardize : bool, default True
        Whether to standardize residuals before plotting.
    point_kws : dict, optional
        Keyword arguments passed to scatter plot.
    line_kws : dict, optional
        Keyword arguments passed to reference line plot.

    Returns
    -------
    Axes
        The matplotlib axes with the plot.
    """
    _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    resid = result.residuals(type="pearson" if standardize else "response")

    n = len(resid)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sorted_resid = np.sort(resid)

    point_defaults: dict[str, Any] = {"alpha": 0.6, "s": 30, "edgecolors": "none"}
    if point_kws:
        point_defaults.update(point_kws)

    ax.scatter(theoretical_quantiles, sorted_resid, **point_defaults)

    line_defaults: dict[str, Any] = {"color": "red", "linestyle": "--", "linewidth": 1}
    if line_kws:
        line_defaults.update(line_kws)

    q25, q75 = np.percentile(sorted_resid, [25, 75])
    t25, t75 = stats.norm.ppf([0.25, 0.75])
    slope = (q75 - q25) / (t75 - t25)
    intercept = q25 - slope * t25

    xlim = ax.get_xlim()
    x_line = np.array(xlim)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, **line_defaults)
    ax.set_xlim(xlim)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Normal Q-Q")

    return ax


def plot_scale_location(
    result: LmerResult | GlmerResult,
    ax: Axes | None = None,
    lowess: bool = True,
    point_kws: dict[str, Any] | None = None,
    line_kws: dict[str, Any] | None = None,
) -> Axes:
    """Scale-location plot (sqrt of standardized residuals vs fitted).

    Parameters
    ----------
    result : LmerResult or GlmerResult
        Fitted model result.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new axes.
    lowess : bool, default True
        Whether to add a lowess smoothing line.
    point_kws : dict, optional
        Keyword arguments passed to scatter plot.
    line_kws : dict, optional
        Keyword arguments passed to lowess line plot.

    Returns
    -------
    Axes
        The matplotlib axes with the plot.
    """
    _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fitted = result.fitted()
    resid = result.residuals(type="pearson")
    sqrt_abs_resid = np.sqrt(np.abs(resid))

    point_defaults: dict[str, Any] = {"alpha": 0.6, "s": 30, "edgecolors": "none"}
    if point_kws:
        point_defaults.update(point_kws)

    ax.scatter(fitted, sqrt_abs_resid, **point_defaults)

    if lowess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

            sorted_idx = np.argsort(fitted)
            smoothed = sm_lowess(sqrt_abs_resid[sorted_idx], fitted[sorted_idx], frac=0.6)

            line_defaults: dict[str, Any] = {"color": "red", "linewidth": 2}
            if line_kws:
                line_defaults.update(line_kws)

            ax.plot(smoothed[:, 0], smoothed[:, 1], **line_defaults)
        except ImportError:
            pass

    ax.set_xlabel("Fitted values")
    ax.set_ylabel(r"$\sqrt{|Standardized\ residuals|}$")
    ax.set_title("Scale-Location")

    return ax


def plot_resid_group(
    result: LmerResult | GlmerResult,
    group: str | None = None,
    ax: Axes | None = None,
    point_kws: dict[str, Any] | None = None,
) -> Axes:
    """Boxplot of residuals by group.

    Parameters
    ----------
    result : LmerResult or GlmerResult
        Fitted model result.
    group : str, optional
        Name of grouping factor. If None, uses the first random effect grouping.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new axes.
    point_kws : dict, optional
        Keyword arguments passed to boxplot.

    Returns
    -------
    Axes
        The matplotlib axes with the plot.
    """
    _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if group is None:
        if not result.matrices.random_structures:
            raise ValueError("No random effects in model")
        group = result.matrices.random_structures[0].grouping_factor

    struct = None
    for s in result.matrices.random_structures:
        if s.grouping_factor == group:
            struct = s
            break

    if struct is None:
        raise ValueError(f"Grouping factor '{group}' not found in model")

    resid = result.residuals(type="response")

    z_start = 0
    for s in result.matrices.random_structures:
        if s.grouping_factor == group:
            break
        z_start += s.n_levels * s.n_terms

    n_terms = struct.n_terms
    Z_slice = result.matrices.Z[:, z_start : z_start + struct.n_levels * n_terms]

    if hasattr(Z_slice, "toarray"):
        Z_slice = Z_slice.toarray()

    group_resids: list[NDArray[np.floating]] = []
    group_labels: list[str] = []

    for level_name, level_idx in struct.level_map.items():
        col_idx = level_idx * n_terms
        mask = Z_slice[:, col_idx] != 0
        if np.any(mask):
            group_resids.append(resid[mask])
            group_labels.append(str(level_name))

    box_defaults: dict[str, Any] = {"patch_artist": True}
    if point_kws:
        box_defaults.update(point_kws)

    bp = ax.boxplot(group_resids, labels=group_labels, **box_defaults)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Group ({group})")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals by {group}")

    if len(group_labels) > 10:
        ax.tick_params(axis="x", rotation=45)

    return ax


def plot_ranef(
    result: LmerResult | GlmerResult,
    group: str | None = None,
    term: str | None = None,
    ax: Axes | None = None,
    condVar: bool = True,
    order: bool = True,
    point_kws: dict[str, Any] | None = None,
) -> Axes:
    """Caterpillar plot of random effects with confidence intervals.

    Parameters
    ----------
    result : LmerResult or GlmerResult
        Fitted model result.
    group : str, optional
        Name of grouping factor. If None, uses the first random effect grouping.
    term : str, optional
        Name of random effect term. If None, uses the first term.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new axes.
    condVar : bool, default True
        Whether to show conditional variance intervals.
    order : bool, default True
        Whether to order by random effect value.
    point_kws : dict, optional
        Keyword arguments passed to scatter plot.

    Returns
    -------
    Axes
        The matplotlib axes with the plot.
    """
    _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 10))

    ranef_result = result.ranef(condVar=condVar)

    from mixedlm.models.lmer import RanefResult

    ranef_values = ranef_result.values if isinstance(ranef_result, RanefResult) else ranef_result

    if group is None:
        group = list(ranef_values.keys())[0]

    if group not in ranef_values:
        raise ValueError(f"Grouping factor '{group}' not found")

    group_ranef = ranef_values[group]

    if term is None:
        term = list(group_ranef.keys())[0]

    if term not in group_ranef:
        raise ValueError(f"Term '{term}' not found in group '{group}'")

    values = group_ranef[term]

    struct = None
    for s in result.matrices.random_structures:
        if s.grouping_factor == group:
            struct = s
            break

    levels = list(struct.level_map.keys()) if struct else [f"Level {i}" for i in range(len(values))]

    if order:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        levels = [levels[i] for i in sort_idx]

    y_pos = np.arange(len(values))

    point_defaults: dict[str, Any] = {"s": 40, "zorder": 3}
    if point_kws:
        point_defaults.update(point_kws)

    ax.scatter(values, y_pos, **point_defaults)

    if (
        condVar
        and isinstance(ranef_result, RanefResult)
        and ranef_result.condVar is not None
        and group in ranef_result.condVar
        and term in ranef_result.condVar[group]
    ):
        cond_var = ranef_result.condVar[group][term]
        if order:
            cond_var = cond_var[sort_idx]
        se = np.sqrt(cond_var)

        for i, (v, s) in enumerate(zip(values, se, strict=True)):
            ax.plot([v - 1.96 * s, v + 1.96 * s], [i, i], color="gray", linewidth=1, zorder=2)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(levels)
    ax.set_xlabel(f"Random effect: {term}")
    ax.set_ylabel(f"Group: {group}")
    ax.set_title(f"Random Effects: {term} | {group}")

    return ax


def plot_diagnostics(
    result: LmerResult | GlmerResult,
    which: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a panel of diagnostic plots.

    Parameters
    ----------
    result : LmerResult or GlmerResult
        Fitted model result.
    which : list of int, optional
        Which plots to include. Default is [1, 2, 3, 4].
        1 = Residuals vs Fitted
        2 = Normal Q-Q
        3 = Scale-Location
        4 = Residuals by Group (if random effects exist)
    figsize : tuple, optional
        Figure size. Default is (12, 10).

    Returns
    -------
    Figure
        The matplotlib figure with diagnostic plots.
    """
    _check_matplotlib()

    if which is None:
        which = [1, 2, 3, 4]

    has_random = bool(result.matrices.random_structures)
    if 4 in which and not has_random:
        which = [w for w in which if w != 4]

    n_plots = len(which)
    if n_plots == 0:
        raise ValueError("No plots to display")

    if n_plots <= 2:
        nrows, ncols = 1, n_plots
    elif n_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (n_plots + 1) // 2
        ncols = 2

    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    axes = [axes] if n_plots == 1 else axes.flatten()

    plot_idx = 0
    for plot_num in which:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        if plot_num == 1:
            plot_resid_fitted(result, ax=ax)
        elif plot_num == 2:
            plot_qq(result, ax=ax)
        elif plot_num == 3:
            plot_scale_location(result, ax=ax)
        elif plot_num == 4:
            plot_resid_group(result, ax=ax)

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig

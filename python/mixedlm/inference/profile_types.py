from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats


def _get_plot_axes(ax: Any | None, figsize: tuple[float, float]) -> Any:
    if ax is not None:
        return ax

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting") from None

    _, ax = plt.subplots(figsize=figsize)
    return ax


def _profile_density(
    values: NDArray[np.floating],
    zeta: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if values.ndim != 1 or zeta.ndim != 1 or values.shape != zeta.shape:
        raise ValueError("profile values and zeta must be one-dimensional arrays of equal length")

    finite = np.isfinite(values) & np.isfinite(zeta)
    sorted_values = values[finite]
    sorted_zeta = zeta[finite]

    if len(sorted_values) < 2:
        raise ValueError("profile density requires at least two finite points")

    order = np.argsort(sorted_values, kind="stable")
    sorted_values = sorted_values[order]
    sorted_zeta = sorted_zeta[order]
    density = np.exp(-0.5 * np.square(sorted_zeta))
    normalization = integrate.trapezoid(density, sorted_values)

    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("profile density requires at least two distinct parameter values")

    return sorted_values, density / normalization


def _validate_profile_grid(
    values1: NDArray[np.floating],
    values2: NDArray[np.floating],
    zeta: NDArray[np.floating],
) -> None:
    if values1.ndim != 1 or values2.ndim != 1:
        raise ValueError("profile grid coordinates must be one-dimensional")

    expected_shape = (len(values1), len(values2))
    if zeta.shape != expected_shape:
        raise ValueError(f"zeta must have shape {expected_shape}, got {zeta.shape}")


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
        ax = _get_plot_axes(ax, (6, 4))

        line_kwargs = {"color": "blue", "linestyle": "-", "linewidth": 2}
        line_kwargs.update(kwargs)
        ax.plot(self.values, self.zeta, **line_kwargs)
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
        ax = _get_plot_axes(ax, (6, 4))
        values, density = _profile_density(self.values, self.zeta)

        line_kwargs = {"color": "blue", "linestyle": "-", "linewidth": 2}
        line_kwargs.update(kwargs)
        line = ax.plot(values, density, **line_kwargs)[0]
        ax.fill_between(values, density, color=line.get_color(), alpha=0.3)

        ax.axvline(self.mle, color="red", linestyle="--", alpha=0.7, label="MLE")
        ax.axvline(self.ci_lower, color="green", linestyle=":", alpha=0.5)
        ax.axvline(self.ci_upper, color="green", linestyle=":", alpha=0.5)

        ax.set_xlabel(self.parameter)
        ax.set_ylabel("Density")
        ax.set_title(f"Profile density: {self.parameter}")

        return ax


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
        ax = _get_plot_axes(ax, (8, 6))
        _validate_profile_grid(self.values1, self.values2, self.zeta)

        contour_kwargs = {"levels": n_levels}
        contour_kwargs.update(kwargs)
        contour = ax.contour(self.values2, self.values1, self.zeta, **contour_kwargs)
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

        if show_ci:
            z_crit_sq = stats.chi2.ppf(self.level, df=2)
            ax.contour(
                self.values2,
                self.values1,
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
        ax = _get_plot_axes(ax, (8, 6))
        _validate_profile_grid(self.values1, self.values2, self.zeta)
        squared_zeta = np.square(self.zeta)
        lik_surface = np.exp(-0.5 * squared_zeta)

        contour_kwargs = {"levels": 20, "cmap": "viridis"}
        contour_kwargs.update(kwargs)
        contourf = ax.contourf(self.values2, self.values1, lik_surface, **contour_kwargs)
        ax.figure.colorbar(contourf, ax=ax, label="Relative likelihood")

        if show_ci:
            z_crit_sq = stats.chi2.ppf(self.level, df=2)
            ax.contour(
                self.values2,
                self.values1,
                squared_zeta,
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

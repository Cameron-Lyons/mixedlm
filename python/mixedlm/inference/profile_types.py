from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


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

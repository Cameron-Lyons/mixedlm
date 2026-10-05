"""Diagnostic, random-effect and profile plots, checked against the plotted data.

The autouse fixture in conftest.py selects the Agg backend and closes figures.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from numpy.testing import assert_allclose
from scipy import integrate, stats

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from mixedlm.diagnostics.plots import (
    plot_diagnostics,
    plot_qq,
    plot_ranef,
    plot_resid_fitted,
    plot_resid_group,
    plot_scale_location,
)
from mixedlm.inference.profile import (
    Profile2DResult,
    ProfileResult,
    plot_profiles,
    profile_lmer,
    splom_profiles,
)

from tests._datasets import SLEEPSTUDY

# The residual smoothers are drawn only when statsmodels is installed.
SMOOTHER_LINES = int(find_spec("statsmodels") is not None)
DIAGNOSTIC_TITLES = ["Residuals vs Fitted", "Normal Q-Q", "Scale-Location", "Residuals by group"]


def grouped_frame(n_groups, *, seed=42, group_sd=2.0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"G{i}" for i in range(n_groups)], 20)
    x = rng.standard_normal(len(groups))
    effects = np.repeat(rng.normal(0.0, group_sd, n_groups), 20)
    y = 5.0 + 2.0 * x + effects + rng.normal(0.0, 0.5, len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def points(ax):
    return ax.collections[0].get_offsets()


def visible_titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.get_visible()]


@pytest.fixture(scope="module")
def lmer_result():
    return lmer("y ~ x + (1|group)", grouped_frame(6))


@pytest.fixture(scope="module")
def split_terms():
    return lmer("Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)", SLEEPSTUDY)


@pytest.fixture(scope="module")
def sleepstudy_profiles(sleepstudy_lmm):
    return profile_lmer(sleepstudy_lmm, which=["(Intercept)", "Days"], n_points=10)


class TestPlotResidFitted:
    def test_points_are_fitted_values_and_residuals(self, lmer_result):
        ax = plot_resid_fitted(lmer_result)

        expected = np.column_stack((lmer_result.fitted(), lmer_result.residuals()))
        assert_allclose(points(ax), expected)
        assert ax.get_xlabel() == "Fitted values"
        assert ax.get_ylabel() == "Residuals"
        assert ax.get_title() == "Residuals vs Fitted"
        # The zero reference line, plus the smoother when available.
        assert len(ax.lines) == 1 + SMOOTHER_LINES

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_resid_fitted(lmer_result, ax=ax)
        assert result_ax is ax

    def test_without_lowess(self, lmer_result):
        ax = plot_resid_fitted(lmer_result, lowess=False)
        assert len(ax.lines) == 1

    def test_custom_point_kws(self, lmer_result):
        ax = plot_resid_fitted(lmer_result, point_kws={"color": "red", "alpha": 0.3})

        assert_allclose(ax.collections[0].get_facecolor()[0], [1.0, 0.0, 0.0, 0.3])

    def test_weighted_fit_plots_response_residuals(self):
        data = grouped_frame(5).assign(w=np.r_[np.full(40, 2.0), np.full(60, 0.5)])
        result = lmer("y ~ x + (1|group)", data, weights="w")

        ax = plot_resid_fitted(result)

        assert_allclose(points(ax)[:, 1], data["y"] - result.fitted())


class TestPlotQQ:
    def test_standardized_residuals_against_normal_quantiles(self, lmer_result):
        ax = plot_qq(lmer_result)

        n = lmer_result.nobs()
        quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        expected = np.sort(lmer_result.residuals(type="pearson"))
        assert_allclose(points(ax), np.column_stack((quantiles, expected)))
        assert ax.get_xlabel() == "Theoretical Quantiles"
        assert ax.get_ylabel() == "Sample Quantiles"

    def test_standardize_false(self, lmer_result):
        ax = plot_qq(lmer_result, standardize=False)

        assert_allclose(points(ax)[:, 1], np.sort(lmer_result.residuals(type="response")))

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_qq(lmer_result, ax=ax)
        assert result_ax is ax


class TestPlotScaleLocation:
    def test_points_are_root_absolute_standardized_residuals(self, lmer_result):
        ax = plot_scale_location(lmer_result)

        expected = np.sqrt(np.abs(lmer_result.residuals(type="pearson")))
        assert_allclose(points(ax), np.column_stack((lmer_result.fitted(), expected)))
        assert ax.get_title() == "Scale-Location"
        assert len(ax.lines) == SMOOTHER_LINES

    def test_without_lowess(self, lmer_result):
        ax = plot_scale_location(lmer_result, lowess=False)
        assert len(ax.lines) == 0

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_scale_location(lmer_result, ax=ax)
        assert result_ax is ax


class TestPlotResidGroup:
    def test_one_box_per_group(self, lmer_result):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatplotlibDeprecationWarning)
            ax = plot_resid_group(lmer_result)

        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["G0", "G1", "G2", "G3", "G4", "G5"]
        assert ax.get_title() == "Residuals by group"

    def test_with_specified_group(self, lmer_result):
        ax = plot_resid_group(lmer_result, group="group")
        assert len(ax.get_xticklabels()) == 6

    def test_invalid_group_raises(self, lmer_result):
        with pytest.raises(ValueError, match="not found"):
            plot_resid_group(lmer_result, group="nonexistent")

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_resid_group(lmer_result, ax=ax)
        assert result_ax is ax

    def test_large_number_of_groups(self):
        result = lmer("y ~ x + (1|group)", grouped_frame(15))

        ax = plot_resid_group(result)

        assert len(ax.get_xticklabels()) == 15


class TestPlotRanef:
    def test_sorted_effects_with_conditional_intervals(self, lmer_result):
        ranefs = lmer_result.ranef(condVar=True)
        values = ranefs["group"]["(Intercept)"]
        order = np.argsort(values)
        half_widths = 1.96 * np.sqrt(ranefs.condVar["group"]["(Intercept)"][order])

        ax = plot_ranef(lmer_result)

        assert_allclose(points(ax)[:, 0], values[order])
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert labels == [f"G{i}" for i in order]
        intervals = [line.get_xdata() for line in ax.lines[:-1]]
        assert_allclose(
            intervals,
            np.column_stack((values[order], values[order])) + np.outer(half_widths, [-1.0, 1.0]),
        )
        assert ax.get_title() == "Random Effects: (Intercept) | group"

    def test_with_specified_group(self, lmer_result):
        ax = plot_ranef(lmer_result, group="group")
        assert len(points(ax)) == 6

    def test_without_condvar(self, lmer_result):
        ax = plot_ranef(lmer_result, condVar=False)
        # Only the zero reference line remains.
        assert len(ax.lines) == 1

    def test_without_order(self, lmer_result):
        ax = plot_ranef(lmer_result, order=False)
        assert_allclose(points(ax)[:, 0], lmer_result.ranef()["group"]["(Intercept)"])

    def test_selected_term(self, sleepstudy_slopes_lmm):
        ax = plot_ranef(sleepstudy_slopes_lmm, term="Days")

        assert_allclose(points(ax)[:, 0], np.sort(sleepstudy_slopes_lmm.ranef()["Subject"]["Days"]))

    def test_invalid_group_or_term_raises(self, lmer_result):
        with pytest.raises(ValueError, match="not found"):
            plot_ranef(lmer_result, group="nonexistent")
        with pytest.raises(ValueError, match="Term 'x' not found"):
            plot_ranef(lmer_result, term="x")


class TestPlotDiagnostics:
    def test_default_panels(self, lmer_result, cbpp_glmm):
        for result in (lmer_result, cbpp_glmm):
            fig = plot_diagnostics(result)

            assert [ax.get_title() for ax in fig.axes] == [
                *DIAGNOSTIC_TITLES[:3],
                f"Residuals by {result.matrices.random_structures[0].grouping_factor}",
            ]
            assert_allclose(fig.get_size_inches(), (12, 10))

    def test_method_matches_function(self, lmer_result):
        assert visible_titles(lmer_result.plot()) == visible_titles(plot_diagnostics(lmer_result))

    @pytest.mark.parametrize(
        ("which", "size"), [([1, 2], (12, 5)), ([1], (6, 5)), ([1, 2, 3], (12, 10))]
    )
    def test_which_selects_panels(self, lmer_result, which, size):
        fig = plot_diagnostics(lmer_result, which=which)

        assert visible_titles(fig) == [DIAGNOSTIC_TITLES[i - 1] for i in which]
        assert_allclose(fig.get_size_inches(), size)

    def test_custom_figsize(self, lmer_result):
        fig = plot_diagnostics(lmer_result, figsize=(10, 8))
        assert_allclose(fig.get_size_inches(), (10, 8))

    def test_models_without_random_effects_omit_the_group_panel(self, lmer_result):
        matrices = lmer_result.matrices
        fixed_only = replace(
            lmer_result,
            matrices=replace(matrices, Z=matrices.Z[:, :0], random_structures=[], n_random=0),
            theta=np.empty(0),
            u=np.empty(0),
        )

        fig = plot_diagnostics(fixed_only)

        assert visible_titles(fig) == DIAGNOSTIC_TITLES[:3]

    def test_empty_which_raises(self, lmer_result):
        with pytest.raises(ValueError, match="No plots"):
            plot_diagnostics(lmer_result, which=[])


class TestRandomEffectPanels:
    def test_dotplot_draws_every_term_of_a_split_grouping_factor(self, split_terms):
        ranefs = split_terms.ranef()["Subject"]
        fig = split_terms.dotplot()

        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert [ax.get_title() for ax in axes] == [
            "Random Effects: (Intercept) | Subject",
            "Random Effects: Days | Subject",
        ]
        for ax, term in zip(axes, ["(Intercept)", "Days"], strict=True):
            points = ax.collections[0].get_offsets()[:, 0]
            np.testing.assert_allclose(points, np.sort(ranefs[term]))

    def test_dotplot_selected_term_and_glmm(self, sleepstudy_slopes_lmm, cbpp_glmm):
        single = sleepstudy_slopes_lmm.dotplot(term="(Intercept)")
        herds = cbpp_glmm.dotplot()

        assert visible_titles(single) == ["Random Effects: (Intercept) | Subject"]
        assert visible_titles(herds) == ["Random Effects: (Intercept) | herd"]
        assert_allclose(points(herds.axes[0])[:, 0], np.sort(cbpp_glmm.getME("b")))

    def test_qqmath_plots_sorted_effects_against_normal_quantiles(self, split_terms):
        values = np.sort(split_terms.ranef()["Subject"]["Days"])
        fig = split_terms.qqmath(term="Days")

        (ax,) = fig.axes
        points = ax.collections[0].get_offsets()
        quantiles = stats.norm.ppf((np.arange(1, len(values) + 1) - 0.5) / len(values))
        np.testing.assert_allclose(points[:, 0], quantiles)
        np.testing.assert_allclose(points[:, 1], values)
        assert ax.get_title() == "QQ Plot: Subject / Days"

    def test_qqmath_panels_terms_and_figsize(self, sleepstudy_slopes_lmm, cbpp_glmm):
        both = sleepstudy_slopes_lmm.qqmath()
        sized = sleepstudy_slopes_lmm.qqmath(term="(Intercept)", figsize=(8, 6))
        herds = cbpp_glmm.qqmath()

        assert visible_titles(both) == ["QQ Plot: Subject / (Intercept)", "QQ Plot: Subject / Days"]
        assert visible_titles(sized) == ["QQ Plot: Subject / (Intercept)"]
        assert_allclose(sized.get_size_inches(), (8, 6))
        assert_allclose(points(herds.axes[0])[:, 1], np.sort(cbpp_glmm.getME("b")))

    @pytest.mark.parametrize("method", ["dotplot", "qqmath"])
    def test_unknown_term_raises_before_drawing(self, split_terms, method):
        plt.close("all")
        with pytest.raises(ValueError, match="Term 'x' not found in group 'Subject'"):
            getattr(split_terms, method)(term="x")
        assert plt.get_fignums() == []

    def test_qqmath_unknown_group_raises(self, sleepstudy_lmm):
        with pytest.raises(ValueError, match="Grouping factor 'InvalidGroup' not found"):
            sleepstudy_lmm.qqmath(group="InvalidGroup")


class TestProfilePlots:
    @staticmethod
    def profile(values=None, zeta=None, mle=2.0, ci_lower=1.5, ci_upper=2.5) -> ProfileResult:
        return ProfileResult(
            parameter="test",
            values=np.array([1.0, 2.0, 3.0]) if values is None else values,
            zeta=np.array([-1.0, 0.0, 1.0]) if zeta is None else zeta,
            mle=mle,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            level=0.95,
        )

    @staticmethod
    def profile_2d() -> Profile2DResult:
        return Profile2DResult(
            param1="a",
            param2="b",
            values1=np.array([0.0, 1.0, 2.0]),
            values2=np.array([0.0, 1.0, 2.0, 3.0]),
            zeta=np.array(
                [
                    [2.0, 1.5, 1.0, 1.5],
                    [1.5, 0.5, 0.0, 1.0],
                    [2.0, 1.5, 1.0, 1.5],
                ]
            ),
            mle1=1.0,
            mle2=2.0,
            level=0.95,
        )

    def test_profile_line_with_mle_and_interval_markers(self, sleepstudy_profiles):
        profile = sleepstudy_profiles["Days"]
        z_crit = stats.norm.isf(0.025)

        ax = profile.plot()

        profile_line, zero, mle, upper, lower, ci_lower, ci_upper = ax.lines
        assert_allclose(profile_line.get_xdata(), profile.values)
        assert_allclose(profile_line.get_ydata(), profile.zeta)
        assert_allclose(zero.get_ydata(), [0.0, 0.0])
        assert_allclose(mle.get_xdata(), [profile.mle, profile.mle])
        assert_allclose([upper.get_ydata()[0], lower.get_ydata()[0]], [z_crit, -z_crit])
        assert_allclose(ci_lower.get_xdata(), [profile.ci_lower] * 2)
        assert_allclose(ci_upper.get_xdata(), [profile.ci_upper] * 2)
        assert ax.get_title() == "Profile: Days"

    def test_profile_without_markers(self, sleepstudy_profiles):
        ax = sleepstudy_profiles["Days"].plot(show_ci=False, show_mle=False)
        # The profile and its zero reference line.
        assert len(ax.lines) == 2

    def test_plot_allows_style_overrides(self):
        ax = self.profile().plot(color="black", linewidth=1)

        assert ax.lines[0].get_color() == "black"
        assert ax.lines[0].get_linewidth() == 1

    def test_density_of_a_fitted_profile_integrates_to_one(self, sleepstudy_profiles):
        ax = sleepstudy_profiles["Days"].plot_density()

        line = ax.lines[0]
        assert integrate.trapezoid(line.get_ydata(), line.get_xdata()) == pytest.approx(1.0)
        assert ax.get_title() == "Profile density: Days"

    def test_density_sorts_and_normalizes_profile_points(self):
        result = self.profile(
            values=np.array([3.0, np.nan, 1.0, 2.0]),
            zeta=np.array([1.0, 0.0, -1.0, 0.0]),
        )

        ax = result.plot_density(color="purple", linewidth=1)
        values = ax.lines[0].get_xdata()
        density = ax.lines[0].get_ydata()

        assert np.all(np.diff(values) >= 0.0)
        assert np.all(density >= 0.0)
        assert integrate.trapezoid(density, values) == pytest.approx(1.0)
        assert ax.lines[0].get_color() == "purple"

    def test_density_works_without_numpy_trapezoid(self, monkeypatch):
        monkeypatch.delattr(np, "trapezoid", raising=False)

        ax = self.profile().plot_density()

        values, density = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        area = np.sum(np.diff(values) * (density[1:] + density[:-1]) / 2)
        assert area == pytest.approx(1.0)

    def test_density_rejects_degenerate_profile(self):
        result = self.profile(
            values=np.array([1.0, 1.0]),
            zeta=np.array([-1.0, 1.0]),
            mle=1.0,
            ci_lower=1.0,
            ci_upper=1.0,
        )

        with pytest.raises(ValueError, match="distinct parameter values"):
            result.plot_density()

    def test_plot_profiles_draws_one_panel_per_parameter(self, sleepstudy_profiles):
        fig = plot_profiles(sleepstudy_profiles)
        density = plot_profiles(sleepstudy_profiles, plot_type="density")

        assert visible_titles(fig) == ["Profile: (Intercept)", "Profile: Days"]
        assert visible_titles(density) == ["Profile density: (Intercept)", "Profile density: Days"]

    def test_splom_profiles_draws_a_parameter_grid(self, sleepstudy_profiles):
        fig = splom_profiles(sleepstudy_profiles)

        assert len(fig.axes) == 4
        assert fig.axes[0].get_title() == "(Intercept)"
        assert_allclose(fig.axes[3].lines[0].get_ydata(), sleepstudy_profiles["Days"].zeta)
        with pytest.raises(ValueError, match="at least 2 profiles"):
            splom_profiles({"Days": sleepstudy_profiles["Days"]})

    def test_plot_filled_allows_contour_style_overrides(self):
        ax = self.profile_2d().plot_filled(levels=5, cmap="plasma")

        # The filled contours and their colorbar.
        assert len(ax.figure.axes) == 2
        assert ax.collections[0].get_cmap().name == "plasma"

    def test_plot_rejects_mismatched_grid_shape(self):
        profile = self.profile_2d()
        profile.zeta = np.zeros((4, 3))

        with pytest.raises(ValueError, match=r"zeta must have shape \(3, 4\)"):
            profile.plot()

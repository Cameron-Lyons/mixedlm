from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mixedlm.diagnostics.plots import (
    _check_matplotlib,
    plot_diagnostics,
    plot_qq,
    plot_ranef,
    plot_resid_fitted,
    plot_resid_group,
    plot_scale_location,
)


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n_groups = 6
    n_per_group = 20
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    x = np.random.randn(n)
    group_effects = np.repeat(np.random.randn(n_groups) * 2, n_per_group)
    y = 5.0 + 2.0 * x + group_effects + np.random.randn(n) * 0.5

    return pd.DataFrame({"y": y, "x": x, "group": groups})


@pytest.fixture
def lmer_result(simple_data):
    return lmer("y ~ x + (1|group)", simple_data)


class TestCheckMatplotlib:
    def test_does_not_raise_when_available(self):
        _check_matplotlib()


class TestPlotResidFitted:
    def test_basic_plot(self, lmer_result):
        ax = plot_resid_fitted(lmer_result)
        assert ax is not None
        plt.close("all")

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_resid_fitted(lmer_result, ax=ax)
        assert result_ax is ax
        plt.close("all")

    def test_without_lowess(self, lmer_result):
        ax = plot_resid_fitted(lmer_result, lowess=False)
        assert ax is not None
        plt.close("all")

    def test_custom_point_kws(self, lmer_result):
        ax = plot_resid_fitted(lmer_result, point_kws={"color": "red", "alpha": 0.3})
        assert ax is not None
        plt.close("all")

    def test_plot_labels(self, lmer_result):
        ax = plot_resid_fitted(lmer_result)
        assert ax.get_xlabel() == "Fitted values"
        assert ax.get_ylabel() == "Residuals"
        plt.close("all")


class TestPlotQQ:
    def test_basic_plot(self, lmer_result):
        ax = plot_qq(lmer_result)
        assert ax is not None
        plt.close("all")

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_qq(lmer_result, ax=ax)
        assert result_ax is ax
        plt.close("all")

    def test_standardize_false(self, lmer_result):
        ax = plot_qq(lmer_result, standardize=False)
        assert ax is not None
        plt.close("all")

    def test_plot_labels(self, lmer_result):
        ax = plot_qq(lmer_result)
        assert "Quantiles" in ax.get_xlabel()
        assert "Quantiles" in ax.get_ylabel()
        plt.close("all")


class TestPlotScaleLocation:
    def test_basic_plot(self, lmer_result):
        ax = plot_scale_location(lmer_result)
        assert ax is not None
        plt.close("all")

    def test_without_lowess(self, lmer_result):
        ax = plot_scale_location(lmer_result, lowess=False)
        assert ax is not None
        plt.close("all")

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_scale_location(lmer_result, ax=ax)
        assert result_ax is ax
        plt.close("all")


class TestPlotResidGroup:
    def test_basic_plot(self, lmer_result):
        ax = plot_resid_group(lmer_result)
        assert ax is not None
        plt.close("all")

    def test_with_specified_group(self, lmer_result):
        ax = plot_resid_group(lmer_result, group="group")
        assert ax is not None
        plt.close("all")

    def test_invalid_group_raises(self, lmer_result):
        with pytest.raises(ValueError, match="not found"):
            plot_resid_group(lmer_result, group="nonexistent")
        plt.close("all")

    def test_custom_ax(self, lmer_result):
        fig, ax = plt.subplots()
        result_ax = plot_resid_group(lmer_result, ax=ax)
        assert result_ax is ax
        plt.close("all")


class TestPlotRanef:
    def test_basic_plot(self, lmer_result):
        ax = plot_ranef(lmer_result)
        assert ax is not None
        plt.close("all")

    def test_with_specified_group(self, lmer_result):
        ax = plot_ranef(lmer_result, group="group")
        assert ax is not None
        plt.close("all")

    def test_without_condvar(self, lmer_result):
        ax = plot_ranef(lmer_result, condVar=False)
        assert ax is not None
        plt.close("all")

    def test_without_order(self, lmer_result):
        ax = plot_ranef(lmer_result, order=False)
        assert ax is not None
        plt.close("all")

    def test_invalid_group_raises(self, lmer_result):
        with pytest.raises(ValueError, match="not found"):
            plot_ranef(lmer_result, group="nonexistent")
        plt.close("all")


class TestPlotDiagnostics:
    def test_default_plots(self, lmer_result):
        fig = plot_diagnostics(lmer_result)
        assert fig is not None
        plt.close("all")

    def test_subset_of_plots(self, lmer_result):
        fig = plot_diagnostics(lmer_result, which=[1, 2])
        assert fig is not None
        plt.close("all")

    def test_single_plot(self, lmer_result):
        fig = plot_diagnostics(lmer_result, which=[1])
        assert fig is not None
        plt.close("all")

    def test_custom_figsize(self, lmer_result):
        fig = plot_diagnostics(lmer_result, figsize=(10, 8))
        assert fig is not None
        plt.close("all")

    def test_empty_which_raises(self, lmer_result):
        with pytest.raises(ValueError, match="No plots"):
            plot_diagnostics(lmer_result, which=[])
        plt.close("all")


class TestPlotWithNoRandomEffects:
    def test_no_random_effects_excludes_plot4(self):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 5.0 + 2.0 * x + np.random.randn(n) * 0.5
        groups = np.repeat([f"G{i}" for i in range(5)], 20)
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = lmer("y ~ x + (1|group)", data)
        fig = plot_diagnostics(result, which=[1, 2, 3])
        assert fig is not None
        plt.close("all")


class TestPlotEdgeCases:
    def test_large_number_of_groups(self):
        np.random.seed(42)
        n_groups = 15
        n_per_group = 10
        n = n_groups * n_per_group

        groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
        x = np.random.randn(n)
        y = 5.0 + 2.0 * x + np.random.randn(n)
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = lmer("y ~ x + (1|group)", data)
        ax = plot_resid_group(result)
        assert ax is not None
        plt.close("all")

    def test_plot_with_weights(self):
        np.random.seed(42)
        n = 80
        groups = np.repeat([f"G{i}" for i in range(4)], 20)
        x = np.random.randn(n)
        y = 5.0 + 2.0 * x + np.random.randn(n)
        weights = np.ones(n) * 0.5
        weights[:40] = 2.0
        data = pd.DataFrame({"y": y, "x": x, "group": groups, "w": weights})

        result = lmer("y ~ x + (1|group)", data, weights="w")
        ax = plot_resid_fitted(result)
        assert ax is not None
        plt.close("all")

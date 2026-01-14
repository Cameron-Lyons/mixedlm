from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference.bootstrap import (
    BootstrapResult,
    bootMer,
    bootstrap_glmer,
    bootstrap_lmer,
)
from numpy.testing import assert_allclose


@pytest.fixture
def simple_lmer_data():
    np.random.seed(42)
    n_groups = 8
    n_per_group = 15
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    x = np.random.randn(n)
    group_effects = np.repeat(np.random.randn(n_groups) * 2, n_per_group)
    y = 5.0 + 2.0 * x + group_effects + np.random.randn(n) * 0.5

    return pd.DataFrame({"y": y, "x": x, "group": groups})


@pytest.fixture
def simple_glmer_data():
    np.random.seed(42)
    n_groups = 8
    n_per_group = 20
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    x = np.random.randn(n)
    group_effects = np.repeat(np.random.randn(n_groups) * 0.5, n_per_group)
    eta = -0.5 + 0.8 * x + group_effects
    prob = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, prob)

    return pd.DataFrame({"y": y, "x": x, "group": groups})


@pytest.fixture
def lmer_result(simple_lmer_data):
    return lmer("y ~ x + (1|group)", simple_lmer_data)


@pytest.fixture
def glmer_result(simple_glmer_data):
    return glmer("y ~ x + (1|group)", simple_glmer_data, family=Binomial())


class TestBootstrapResult:
    def test_ci_percentile(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=20, seed=42)
        ci = boot.ci(level=0.95, method="percentile")
        assert "(Intercept)" in ci
        assert "x" in ci
        for _name, (lower, upper) in ci.items():
            assert lower < upper

    def test_ci_basic(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=20, seed=42)
        ci = boot.ci(level=0.95, method="basic")
        assert "(Intercept)" in ci

    def test_ci_normal(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=20, seed=42)
        ci = boot.ci(level=0.95, method="normal")
        assert "(Intercept)" in ci

    def test_ci_invalid_method_raises(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        with pytest.raises(ValueError, match="Unknown method"):
            boot.ci(method="invalid")

    def test_se_method(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=20, seed=42)
        se = boot.se()
        assert "(Intercept)" in se
        assert "x" in se
        for _name, val in se.items():
            assert val > 0

    def test_summary_method(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=20, seed=42)
        summary = boot.summary()
        assert "Parametric bootstrap" in summary
        assert "Fixed effects" in summary


class TestBootstrapLmer:
    def test_basic_bootstrap(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        assert boot.n_boot == 10
        assert boot.beta_samples.shape[0] == 10
        assert boot.sigma_samples is not None

    def test_reproducibility(self, lmer_result, simple_lmer_data):
        boot1 = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        boot2 = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        assert_allclose(boot1.beta_samples, boot2.beta_samples, rtol=1e-10)

    def test_different_seeds(self, lmer_result, simple_lmer_data):
        boot1 = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        boot2 = bootstrap_lmer(lmer_result, n_boot=10, seed=123)
        assert not np.allclose(boot1.beta_samples, boot2.beta_samples)

    def test_original_values_stored(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        assert_allclose(boot.original_beta, lmer_result.beta)
        assert_allclose(boot.original_theta, lmer_result.theta)
        assert boot.original_sigma == pytest.approx(lmer_result.sigma)


class TestBootstrapGlmer:
    def test_basic_bootstrap(self, glmer_result, simple_glmer_data):
        boot = bootstrap_glmer(glmer_result, n_boot=10, seed=42)
        assert boot.n_boot == 10
        assert boot.beta_samples.shape[0] == 10
        assert boot.sigma_samples is None

    def test_reproducibility(self, glmer_result, simple_glmer_data):
        boot1 = bootstrap_glmer(glmer_result, n_boot=10, seed=42)
        boot2 = bootstrap_glmer(glmer_result, n_boot=10, seed=42)
        assert_allclose(boot1.beta_samples, boot2.beta_samples, rtol=1e-10)


class TestBootMer:
    def test_lmer_dispatch(self, lmer_result, simple_lmer_data):
        boot = bootMer(lmer_result, nsim=10, seed=42)
        assert isinstance(boot, BootstrapResult)
        assert boot.n_boot == 10

    def test_glmer_dispatch(self, glmer_result, simple_glmer_data):
        boot = bootMer(glmer_result, nsim=10, seed=42)
        assert isinstance(boot, BootstrapResult)
        assert boot.n_boot == 10

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="not supported"):
            bootMer("not a model", nsim=10)

    def test_invalid_bootstrap_type_raises(self, lmer_result, simple_lmer_data):
        with pytest.raises(ValueError, match="not supported"):
            bootMer(lmer_result, nsim=10, bootstrap_type="nonparametric")


class TestBootstrapEdgeCases:
    def test_bootstrap_with_failures(self, simple_lmer_data):
        np.random.seed(42)
        simple_lmer_data_copy = simple_lmer_data.copy()
        simple_lmer_data_copy.loc[0:5, "y"] = np.nan

        result = lmer("y ~ x + (1|group)", simple_lmer_data)
        boot = bootstrap_lmer(result, n_boot=5, seed=42)
        assert boot.n_boot == 5

    def test_bootstrap_handles_nan_samples(self, lmer_result, simple_lmer_data):
        boot = bootstrap_lmer(lmer_result, n_boot=10, seed=42)
        non_nan_mask = ~np.isnan(boot.beta_samples[:, 0])
        assert np.sum(non_nan_mask) >= 5

    def test_ci_with_all_nan(self):
        boot = BootstrapResult(
            n_boot=10,
            beta_samples=np.full((10, 2), np.nan),
            theta_samples=np.full((10, 1), np.nan),
            sigma_samples=np.full(10, np.nan),
            fixed_names=["a", "b"],
            original_beta=np.array([1.0, 2.0]),
            original_theta=np.array([0.5]),
            original_sigma=1.0,
            n_failed=10,
        )
        ci = boot.ci()
        assert np.isnan(ci["a"][0])
        assert np.isnan(ci["a"][1])

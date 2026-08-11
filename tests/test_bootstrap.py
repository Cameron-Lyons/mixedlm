from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference.bootstrap import (
    BootstrapResult,
    NlmerBootstrapResult,
    _glmer_bootstrap_worker,
    _lmer_bootstrap_worker,
    _prepare_glmer_worker_data,
    _prepare_lmer_worker_data,
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
def categorical_lmer_data():
    rng = np.random.default_rng(47)
    n_groups = 8
    n_per_group = 12
    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    treatment = np.tile(np.repeat(["control", "treated"], n_per_group // 2), n_groups)
    group_effects = np.repeat(rng.normal(0, 0.7, n_groups), n_per_group)
    y = 2.0 + 1.5 * (treatment == "treated") + group_effects + rng.normal(0, 0.5, len(groups))

    return pd.DataFrame({"y": y, "treatment": treatment, "group": groups})


@pytest.fixture
def categorical_glmer_data():
    rng = np.random.default_rng(53)
    n_groups = 8
    n_per_group = 16
    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    treatment = np.tile(np.repeat(["control", "treated"], n_per_group // 2), n_groups)
    group_effects = np.repeat(rng.normal(0, 0.5, n_groups), n_per_group)
    eta = -0.5 + 1.0 * (treatment == "treated") + group_effects
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

    return pd.DataFrame({"y": y, "treatment": treatment, "group": groups})


@pytest.fixture
def lmer_result(simple_lmer_data):
    return lmer("y ~ x + (1|group)", simple_lmer_data)


@pytest.fixture
def glmer_result(simple_glmer_data):
    return glmer("y ~ x + (1|group)", simple_glmer_data, family=Binomial())


class TestBootstrapResult:
    @staticmethod
    def result_from_samples(samples: list[float], original: float = 1.5) -> BootstrapResult:
        beta_samples = np.asarray(samples, dtype=np.float64)[:, None]
        return BootstrapResult(
            n_boot=len(samples),
            beta_samples=beta_samples,
            theta_samples=np.empty((len(samples), 0)),
            sigma_samples=None,
            fixed_names=["x"],
            original_beta=np.array([original]),
            original_theta=np.empty(0),
            original_sigma=None,
            n_failed=int(np.count_nonzero(~np.isfinite(beta_samples))),
        )

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

    def test_se_uses_sample_standard_deviation(self):
        boot = self.result_from_samples([1.0, 3.0])

        assert boot.se()["x"] == pytest.approx(np.sqrt(2.0))
        assert "1.4142" in boot.summary()

    def test_statistics_ignore_all_nonfinite_samples(self):
        boot = self.result_from_samples([1.0, np.nan, np.inf, 3.0])

        assert boot.se()["x"] == pytest.approx(np.sqrt(2.0))
        assert boot.ci(method="percentile")["x"] == pytest.approx((1.05, 2.95))

    def test_normal_ci_corrects_bootstrap_bias(self):
        boot = self.result_from_samples([1.0, 3.0], original=1.5)

        lower, upper = boot.ci(level=0.95, method="normal")["x"]
        expected_half_width = 1.959963984540054 * np.sqrt(2.0)
        assert lower == pytest.approx(1.0 - expected_half_width)
        assert upper == pytest.approx(1.0 + expected_half_width)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, np.nan, -np.inf, np.inf])
    def test_ci_rejects_invalid_level(self, level):
        boot = self.result_from_samples([1.0, 3.0])

        with pytest.raises(ValueError, match="level must be strictly between"):
            boot.ci(level=level)

    def test_single_finite_sample_has_undefined_standard_error(self):
        boot = self.result_from_samples([2.0, np.nan])

        assert np.isnan(boot.se()["x"])
        lower, upper = boot.ci(method="normal")["x"]
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_nlmer_result_uses_same_statistics(self):
        boot = NlmerBootstrapResult(
            n_boot=2,
            phi_samples=np.array([[1.0], [3.0]]),
            theta_samples=np.empty((2, 0)),
            sigma_samples=np.ones(2),
            param_names=["Asym"],
            original_phi=np.array([1.5]),
            original_theta=np.empty(0),
            original_sigma=1.0,
            n_failed=0,
        )

        assert boot.se()["Asym"] == pytest.approx(np.sqrt(2.0))


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

    def test_categorical_predictor_uses_original_model_frame(self, categorical_lmer_data):
        result = lmer("y ~ treatment + (1 | group)", categorical_lmer_data)

        boot = bootstrap_lmer(result, n_boot=5, seed=42)

        assert boot.n_failed == 0
        assert np.isfinite(boot.beta_samples).all()

    def test_parallel_worker_uses_original_model_frame(self, categorical_lmer_data):
        result = lmer("y ~ treatment + (1 | group)", categorical_lmer_data)
        worker_data = _prepare_lmer_worker_data(result)
        task = (
            0,
            42,
            worker_data["formula"],
            worker_data["X"],
            worker_data["Z"],
            worker_data["model_frame"],
            worker_data["random_structures_data"],
            worker_data["response_name"],
            worker_data["beta"],
            worker_data["theta"],
            worker_data["sigma"],
            worker_data["REML"],
        )

        _, beta, theta, sigma = _lmer_bootstrap_worker(task)

        assert beta is not None
        assert theta is not None
        assert sigma is not None

    def test_polars_categorical_predictor(self, categorical_lmer_data):
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(categorical_lmer_data.to_dict(orient="list"))
        result = lmer("y ~ treatment + (1 | group)", data)

        boot = bootstrap_lmer(result, n_boot=3, seed=42)

        assert boot.n_failed == 0
        assert np.isfinite(boot.beta_samples).all()


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

    def test_categorical_predictor_uses_original_model_frame(self, categorical_glmer_data):
        result = glmer("y ~ treatment + (1 | group)", categorical_glmer_data, family=Binomial())

        boot = bootstrap_glmer(result, n_boot=3, seed=42)

        assert boot.n_failed == 0
        assert np.isfinite(boot.beta_samples).all()

    def test_parallel_worker_uses_original_model_frame(self, categorical_glmer_data):
        result = glmer("y ~ treatment + (1 | group)", categorical_glmer_data, family=Binomial())
        worker_data = _prepare_glmer_worker_data(result)
        task = (
            0,
            42,
            worker_data["formula"],
            worker_data["X"],
            worker_data["Z"],
            worker_data["model_frame"],
            worker_data["random_structures_data"],
            worker_data["response_name"],
            worker_data["response_denominator"],
            worker_data["trials"],
            worker_data["beta"],
            worker_data["theta"],
            worker_data["family"],
        )

        _, beta, theta = _glmer_bootstrap_worker(task)

        assert beta is not None
        assert theta is not None


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

        with pytest.raises(ValueError, match="Unknown method"):
            boot.ci(method="invalid")

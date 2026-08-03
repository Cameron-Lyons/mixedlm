from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference.bootstrap import (
    BootstrapResult,
    _lmer_bootstrap_worker,
    _prepare_glmer_worker_data,
    _prepare_lmer_worker_data,
    _simulate_glmer,
    _simulate_lmer,
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

    def test_bootstrap_preserves_categorical_predictors(self):
        rng = np.random.default_rng(20260803)
        n_groups = 8
        n_per_group = 12
        n = n_groups * n_per_group
        group = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
        condition = pd.Categorical(np.tile(["control", "treated"], n // 2))
        condition_effect = (condition == "treated").astype(float)
        group_effect = np.repeat(rng.normal(scale=0.5, size=n_groups), n_per_group)
        y = 1.0 + 0.75 * condition_effect + group_effect + rng.normal(scale=0.3, size=n)
        data = pd.DataFrame({"y": y, "condition": condition, "group": group})

        result = lmer("y ~ condition + (1 | group)", data)
        boot = bootstrap_lmer(result, n_boot=3, seed=42)

        assert boot.n_failed == 0
        assert np.all(np.isfinite(boot.beta_samples))

    def test_bootstrap_does_not_rebuild_validated_design(self, monkeypatch, lmer_result):
        def fail_rebuild(*args, **kwargs):
            raise AssertionError("bootstrap should reuse the fitted design matrices")

        monkeypatch.setattr("mixedlm.models.lmer_fit.build_model_matrices", fail_rebuild)

        boot = bootstrap_lmer(lmer_result, n_boot=2, seed=42)

        assert boot.n_failed == 0

    def test_parallel_payload_reuses_validated_matrices(self, lmer_result):
        payload = _prepare_lmer_worker_data(lmer_result)

        assert set(payload) == {"matrices", "beta", "theta", "sigma", "REML"}
        assert payload["matrices"].X is lmer_result.matrices.X
        assert payload["matrices"].Z is lmer_result.matrices.Z
        assert payload["matrices"].frame is None
        assert payload["matrices"].na_info is None

    def test_parallel_worker_reuses_validated_matrices(self, lmer_result):
        payload = _prepare_lmer_worker_data(lmer_result)
        args = (
            0,
            42,
            payload["matrices"],
            payload["beta"],
            payload["theta"],
            payload["sigma"],
            payload["REML"],
        )

        boot_idx, beta, theta, sigma = _lmer_bootstrap_worker(args)

        assert boot_idx == 0
        assert beta is not None
        assert theta is not None
        assert sigma is not None

    def test_simulation_preserves_offset_and_inverse_variance_weights(self, lmer_result):
        n = lmer_result.matrices.n_obs
        matrices = replace(
            lmer_result.matrices,
            Z=lmer_result.matrices.Z[:, :0],
            random_structures=[],
            n_random=0,
            weights=np.full(n, 4.0),
            offset=np.full(n, 3.0),
        )
        weighted_result = replace(
            lmer_result,
            matrices=matrices,
            theta=np.empty(0),
            sigma=2.0,
        )

        np.random.seed(123)
        simulated = _simulate_lmer(weighted_result)
        np.random.seed(123)
        expected = matrices.X @ weighted_result.beta + 3.0 + np.random.randn(n)

        assert_allclose(simulated, expected)


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

    def test_bootstrap_does_not_rebuild_validated_design(self, monkeypatch, glmer_result):
        def fail_rebuild(*args, **kwargs):
            raise AssertionError("bootstrap should reuse the fitted design matrices")

        monkeypatch.setattr("mixedlm.models.glmer_fit.build_model_matrices", fail_rebuild)

        boot = bootstrap_glmer(glmer_result, n_boot=2, seed=42)

        assert boot.n_failed == 0

    def test_parallel_payload_preserves_quadrature_order(self, glmer_result):
        result = replace(glmer_result, nAGQ=7)

        payload = _prepare_glmer_worker_data(result)

        assert payload["nAGQ"] == 7
        assert payload["matrices"].frame is None

    def test_simulation_preserves_offset(self, glmer_result):
        n = glmer_result.matrices.n_obs
        matrices = replace(
            glmer_result.matrices,
            Z=glmer_result.matrices.Z[:, :0],
            random_structures=[],
            n_random=0,
            offset=np.full(n, 0.75),
        )
        offset_result = replace(glmer_result, matrices=matrices, theta=np.empty(0))

        np.random.seed(123)
        simulated = _simulate_glmer(offset_result)
        np.random.seed(123)
        eta = matrices.X @ offset_result.beta + matrices.offset
        mu = np.clip(offset_result.family.link.inverse(eta), 1e-6, 1 - 1e-6)
        expected = np.random.binomial(1, mu).astype(np.float64)

        assert_allclose(simulated, expected)


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

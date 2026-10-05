from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer, load_cbpp
from mixedlm.families import Binomial
from mixedlm.inference.bootstrap import (
    BootstrapResult,
    NlmerBootstrapResult,
    _lmer_bootstrap_worker,
    _prepare_glmer_worker_data,
    _prepare_lmer_worker_data,
    _simulate_glmer,
    _simulate_lmer,
    bootMer,
    bootstrap_glmer,
    bootstrap_lmer,
)
from mixedlm.utils.random import random_seeds, random_stream
from numpy.testing import assert_allclose


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
    rng = np.random.default_rng(55)
    n_groups = 8
    n_per_group = 16
    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    treatment = np.tile(np.repeat(["control", "treated"], n_per_group // 2), n_groups)
    group_effects = np.repeat(rng.normal(0, 0.5, n_groups), n_per_group)
    eta = -0.5 + 1.0 * (treatment == "treated") + group_effects
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

    return pd.DataFrame({"y": y, "treatment": treatment, "group": groups})


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

    def test_ci_invalid_method_raises(self, grouped_lmm):
        boot = bootstrap_lmer(grouped_lmm, n_boot=10, seed=42)
        with pytest.raises(ValueError, match="Unknown method"):
            boot.ci(method="invalid")

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

        with pytest.raises(ValueError, match="level must be a finite number strictly between"):
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
    def test_basic_bootstrap(self, grouped_lmm):
        boot = bootstrap_lmer(grouped_lmm, n_boot=10, seed=42)
        assert boot.n_boot == 10
        assert boot.n_failed == 0
        assert boot.beta_samples.shape == (10, 2)
        assert boot.theta_samples.shape == (10, 1)
        assert boot.sigma_samples.shape == (10,)
        assert np.all(np.isfinite(boot.beta_samples))

    def test_reproducibility(self, grouped_lmm):
        boot1 = bootstrap_lmer(grouped_lmm, n_boot=10, seed=42)
        boot2 = bootstrap_lmer(grouped_lmm, n_boot=10, seed=42)
        assert_allclose(boot1.beta_samples, boot2.beta_samples, rtol=1e-10)

    def test_different_seeds(self, grouped_lmm):
        boot1 = bootstrap_lmer(grouped_lmm, n_boot=10, seed=42)
        boot2 = bootstrap_lmer(grouped_lmm, n_boot=10, seed=123)
        assert not np.allclose(boot1.beta_samples, boot2.beta_samples)

    def test_original_values_stored(self, grouped_lmm):
        boot = bootstrap_lmer(grouped_lmm, n_boot=10, seed=42)
        assert_allclose(boot.original_beta, grouped_lmm.beta)
        assert_allclose(boot.original_theta, grouped_lmm.theta)
        assert boot.original_sigma == pytest.approx(grouped_lmm.sigma)

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

    def test_bootstrap_does_not_rebuild_validated_design(self, monkeypatch, grouped_lmm):
        from unittest.mock import Mock

        from mixedlm.inference import bootstrap

        rebuild = Mock(side_effect=AssertionError("bootstrap should reuse the fitted design"))
        refit = Mock(wraps=bootstrap._refit_lmer_response)
        monkeypatch.setattr("mixedlm.models.lmer_fit.build_model_matrices", rebuild)
        monkeypatch.setattr(bootstrap, "_refit_lmer_response", refit)

        boot = bootstrap_lmer(grouped_lmm, n_boot=2, seed=42)

        assert boot.n_boot == refit.call_count == 2
        rebuild.assert_not_called()

    def test_parallel_payload_reuses_validated_matrices(self, grouped_lmm):
        payload = _prepare_lmer_worker_data(grouped_lmm)

        assert set(payload) == {"matrices", "beta", "theta", "sigma", "REML"}
        assert payload["matrices"].X is grouped_lmm.matrices.X
        assert payload["matrices"].Z is grouped_lmm.matrices.Z
        assert payload["matrices"].frame is None
        assert payload["matrices"].na_info is None

    def test_parallel_worker_reproduces_the_serial_sample(self, grouped_lmm):
        payload = _prepare_lmer_worker_data(grouped_lmm)
        # Serial and parallel runs draw sample b from the b-th derived seed.
        seed = int(random_seeds(random_stream(42, legacy=False), 1)[0])
        args = (
            0,
            seed,
            payload["matrices"],
            payload["beta"],
            payload["theta"],
            payload["sigma"],
            payload["REML"],
        )

        sample = _lmer_bootstrap_worker(args)
        serial = bootstrap_lmer(grouped_lmm, n_boot=1, seed=42)

        assert sample.index == 0
        assert_allclose(sample.fixed, serial.beta_samples[0])
        assert_allclose(sample.theta, serial.theta_samples[0])
        assert sample.sigma == pytest.approx(serial.sigma_samples[0])

    def test_simulation_preserves_offset_and_inverse_variance_weights(self, grouped_lmm):
        n = grouped_lmm.matrices.n_obs
        matrices = replace(
            grouped_lmm.matrices,
            Z=grouped_lmm.matrices.Z[:, :0],
            random_structures=[],
            n_random=0,
            weights=np.full(n, 4.0),
            offset=np.full(n, 3.0),
        )
        weighted_result = replace(
            grouped_lmm,
            matrices=matrices,
            theta=np.empty(0),
            sigma=2.0,
        )

        np.random.seed(123)
        simulated = _simulate_lmer(weighted_result)
        np.random.seed(123)
        expected = matrices.X @ weighted_result.beta + 3.0 + np.random.randn(n)

        assert_allclose(simulated, expected)

    def test_polars_categorical_predictor(self, categorical_lmer_data):
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(categorical_lmer_data.to_dict(orient="list"))
        result = lmer("y ~ treatment + (1 | group)", data)

        boot = bootstrap_lmer(result, n_boot=3, seed=42)

        assert boot.n_failed == 0
        assert np.isfinite(boot.beta_samples).all()


class TestBootstrapGlmer:
    def test_basic_bootstrap(self, grouped_glmm):
        boot = bootstrap_glmer(grouped_glmm, n_boot=10, seed=42)
        assert boot.n_boot == 10
        assert boot.n_failed == 0
        assert boot.beta_samples.shape == (10, 2)
        assert np.all(np.isfinite(boot.beta_samples))
        assert boot.sigma_samples is None

    def test_reproducibility(self, grouped_glmm):
        boot1 = bootstrap_glmer(grouped_glmm, n_boot=10, seed=42)
        boot2 = bootstrap_glmer(grouped_glmm, n_boot=10, seed=42)
        assert_allclose(boot1.beta_samples, boot2.beta_samples, rtol=1e-10)

    def test_categorical_predictor(self, categorical_glmer_data):
        result = glmer("y ~ treatment + (1 | group)", categorical_glmer_data, family=Binomial())

        boot = bootstrap_glmer(result, n_boot=3, seed=42)

        assert boot.n_failed == 0
        assert np.isfinite(boot.beta_samples).all()

    def test_grouped_binomial_bootstrap_uses_proportion_scale(self, monkeypatch):
        from mixedlm.inference import bootstrap

        data = load_cbpp()
        result = glmer("incidence / size ~ period + (1 | herd)", data, family=Binomial())

        np.random.seed(42)
        simulated = _simulate_glmer(result)
        trials = result.matrices.trials

        assert trials is not None
        assert np.all((simulated >= 0.0) & (simulated <= 1.0))
        assert_allclose(simulated * trials, np.round(simulated * trials))

        refit = bootstrap._refit_glmer_response
        refits = []

        def record_refit(*args, **kwargs):
            fitted = refit(*args, **kwargs)
            refits.append(fitted)
            return fitted

        monkeypatch.setattr(bootstrap, "_refit_glmer_response", record_refit)
        boot = bootstrap_glmer(result, n_boot=2, seed=42)
        successful = np.array([fit.converged and fit.pirls_converged for fit in refits])

        assert len(refits) == 2
        assert boot.n_failed == np.count_nonzero(~successful)
        assert np.isnan(boot.beta_samples[~successful]).all()
        assert np.isnan(boot.theta_samples[~successful]).all()
        for row in np.flatnonzero(successful):
            assert_allclose(boot.beta_samples[row], refits[row].beta)
            assert_allclose(boot.theta_samples[row], refits[row].theta)

    def test_bootstrap_does_not_rebuild_validated_design(self, monkeypatch, grouped_glmm):
        def fail_rebuild(*args, **kwargs):
            raise AssertionError("bootstrap should reuse the fitted design matrices")

        monkeypatch.setattr("mixedlm.models.glmer_fit.build_model_matrices", fail_rebuild)

        boot = bootstrap_glmer(grouped_glmm, n_boot=2, seed=42)

        assert boot.n_failed == 0

    def test_parallel_payload_preserves_quadrature_order(self, grouped_glmm):
        result = replace(grouped_glmm, nAGQ=7)

        payload = _prepare_glmer_worker_data(result)

        assert payload["nAGQ"] == 7
        assert payload["matrices"].frame is None

    def test_simulation_preserves_offset(self, grouped_glmm):
        n = grouped_glmm.matrices.n_obs
        matrices = replace(
            grouped_glmm.matrices,
            Z=grouped_glmm.matrices.Z[:, :0],
            random_structures=[],
            n_random=0,
            offset=np.full(n, 0.75),
        )
        offset_result = replace(grouped_glmm, matrices=matrices, theta=np.empty(0))

        np.random.seed(123)
        simulated = _simulate_glmer(offset_result)
        np.random.seed(123)
        eta = matrices.X @ offset_result.beta + matrices.offset
        mu = np.clip(offset_result.family.link.inverse(eta), 1e-6, 1 - 1e-6)
        expected = np.random.binomial(1, mu).astype(np.float64)

        assert_allclose(simulated, expected)


class TestBootMer:
    def test_lmer_dispatch(self, grouped_lmm):
        boot = bootMer(grouped_lmm, nsim=10, seed=42)
        assert isinstance(boot, BootstrapResult)
        assert boot.n_boot == 10

    def test_glmer_dispatch(self, grouped_glmm):
        boot = bootMer(grouped_glmm, nsim=10, seed=42)
        assert isinstance(boot, BootstrapResult)
        assert boot.n_boot == 10

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="not supported"):
            bootMer("not a model", nsim=10)

    def test_invalid_bootstrap_type_raises(self, grouped_lmm):
        with pytest.raises(ValueError, match="not supported"):
            bootMer(grouped_lmm, nsim=10, bootstrap_type="nonparametric")


class TestBootstrapEdgeCases:
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

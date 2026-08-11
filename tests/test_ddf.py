from __future__ import annotations

from dataclasses import replace

import mixedlm.inference.ddf as ddf_module
import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer, load_sleepstudy
from mixedlm.inference.ddf import (
    DenomDFResult,
    _vcov_derivatives,
    _xt_vinv_x_from_theta,
    clear_vcov_grad_cache,
    kenward_roger_df,
    pvalues_with_ddf,
    satterthwaite_df,
)
from mixedlm.models.control import LmerControl
from numpy.testing import assert_allclose
from scipy import linalg


def create_ddf_data(n_groups: int = 10, n_per_group: int = 5, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data_rows = []
    for g in range(n_groups):
        group_effect = np.random.randn() * 0.5
        for _ in range(n_per_group):
            x = np.random.randn()
            y = 1.0 + 0.5 * x + group_effect + np.random.randn() * 0.5
            data_rows.append({"group": f"G{g + 1}", "x": x, "y": y})
    return pd.DataFrame(data_rows)


DDF_DATA = create_ddf_data()


def create_weighted_ddf_data(seed: int = 70) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_groups = 12
    observations_per_group = 8
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = np.tile(np.linspace(-1.2, 1.2, observations_per_group), n_groups)
    group_effects = np.linspace(-2.0, 2.0, n_groups)
    y = 2.0 + 0.7 * x + group_effects[group] + rng.normal(0.0, 0.18, len(group))
    weights = np.linspace(0.4, 2.2, len(group))
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
    return data, weights


def create_weighted_slope_ddf_data(seed: int = 124) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_groups = 12
    observations_per_group = 10
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = np.tile(np.linspace(-1.5, 1.5, observations_per_group), n_groups)
    intercepts = np.array([-2.0, -1.0, 0.5, 1.5, 2.5, -0.5, 1.0, -2.5, 0.0, 2.0, -1.5, 0.8])
    slopes = np.array([1.4, -0.8, 0.5, -1.3, 1.0, 0.1, -0.4, 1.2, -1.1, 0.8, -0.2, -1.5])
    y = 1.5 + 0.8 * x + intercepts[group] + slopes[group] * x
    y += rng.normal(0.0, 0.12, len(group))
    weights = np.linspace(0.4, 2.0, len(group))
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
    return data, weights


class TestDenomDFResult:
    def test_getitem(self) -> None:
        result = DenomDFResult(
            df=np.array([10.0, 15.0]),
            method="Satterthwaite",
            param_names=["(Intercept)", "x"],
        )
        assert result["(Intercept)"] == 10.0
        assert result["x"] == 15.0

    def test_getitem_invalid_raises(self) -> None:
        result = DenomDFResult(
            df=np.array([10.0, 15.0]),
            method="Satterthwaite",
            param_names=["(Intercept)", "x"],
        )
        with pytest.raises(ValueError):
            result["nonexistent"]

    def test_as_dict(self) -> None:
        result = DenomDFResult(
            df=np.array([10.0, 15.0]),
            method="Satterthwaite",
            param_names=["(Intercept)", "x"],
        )
        d = result.as_dict()
        assert d == {"(Intercept)": 10.0, "x": 15.0}


class TestSatterthwaiteDF:
    def test_satterthwaite_df_basic(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = satterthwaite_df(model)

        assert isinstance(result, DenomDFResult)
        assert result.method == "Satterthwaite"
        assert len(result.df) == 2
        assert "(Intercept)" in result.param_names
        assert "x" in result.param_names

    def test_satterthwaite_df_positive(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = satterthwaite_df(model)

        assert np.all(result.df > 0)

    def test_satterthwaite_df_bounded(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = satterthwaite_df(model)

        n = model.matrices.n_obs
        p = model.matrices.n_fixed
        assert np.all(result.df >= 1.0)
        assert np.all(result.df <= n - p)

    def test_satterthwaite_df_multiple_fixed(self) -> None:
        data = DDF_DATA.copy()
        data["z"] = np.random.randn(len(data))
        model = lmer("y ~ x + z + (1|group)", data)
        result = satterthwaite_df(model)

        assert len(result.df) == 3
        assert "z" in result.param_names

    def test_satterthwaite_df_no_random_effects(self) -> None:
        data = DDF_DATA.copy()
        model = lmer("y ~ x + (1|group)", data)
        result = satterthwaite_df(model)

        assert result.df is not None

    def test_sleepstudy_matches_reference_denominator_df(self) -> None:
        model = lmer("Reaction ~ Days + (Days | Subject)", load_sleepstudy())

        result = satterthwaite_df(model)

        assert_allclose(result.df, [17.0, 17.0], rtol=0, atol=1e-3)

    def test_gradient_cache_is_scoped_to_fitted_result(self) -> None:
        other_data = DDF_DATA.copy()
        other_data["x"] = np.linspace(-2.0, 2.0, len(other_data))

        model = replace(
            lmer("y ~ x + (1|group)", DDF_DATA),
            theta=np.array([0.8]),
            sigma=1.0,
        )
        other_model = replace(
            lmer("y ~ x + (1|group)", other_data),
            theta=model.theta.copy(),
            sigma=model.sigma,
        )

        clear_vcov_grad_cache()
        satterthwaite_df(model)
        cached_entry = next(iter(ddf_module._vcov_grad_cache.values()))

        satterthwaite_df(model)
        assert len(ddf_module._vcov_grad_cache) == 1
        assert next(iter(ddf_module._vcov_grad_cache.values())) is cached_entry

        satterthwaite_df(other_model)
        assert len(ddf_module._vcov_grad_cache) == 2
        assert set(ddf_module._vcov_grad_cache) == {id(model), id(other_model)}
        clear_vcov_grad_cache()


class TestKenwardRogerDF:
    def test_kenward_roger_df_basic(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = kenward_roger_df(model)

        assert isinstance(result, DenomDFResult)
        assert result.method == "Kenward-Roger"
        assert len(result.df) == 2
        assert "(Intercept)" in result.param_names
        assert "x" in result.param_names

    def test_kenward_roger_df_positive(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = kenward_roger_df(model)

        assert np.all(result.df > 0)

    def test_kenward_roger_df_bounded(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        result = kenward_roger_df(model)

        n = model.matrices.n_obs
        p = model.matrices.n_fixed
        assert np.all(result.df >= 1.0)
        assert np.all(result.df <= n - p)

    def test_kenward_roger_smaller_than_satterthwaite(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert np.all(kr.df <= satt.df + 1e-6)


class TestPvaluesWithDDF:
    def test_pvalues_satterthwaite(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Satterthwaite")

        assert "(Intercept)" in results
        assert "x" in results

        for _name, (estimate, t_val, p_val) in results.items():
            assert isinstance(estimate, float)
            assert isinstance(t_val, float)
            assert isinstance(p_val, float)
            assert 0.0 <= p_val <= 1.0

    def test_pvalues_kenward_roger(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Kenward-Roger")

        assert "(Intercept)" in results
        assert "x" in results

        for _name, (estimate, t_val, p_val) in results.items():
            assert isinstance(estimate, float)
            assert isinstance(t_val, float)
            assert isinstance(p_val, float)
            assert 0.0 <= p_val <= 1.0

    def test_pvalues_estimates_match_model(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Satterthwaite")

        for i, name in enumerate(model.matrices.fixed_names):
            estimate, _, _ = results[name]
            assert np.isclose(estimate, model.beta[i])

    def test_pvalues_invalid_method_raises(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)

        with pytest.raises(ValueError, match="Unknown method"):
            pvalues_with_ddf(model, method="invalid")

    @pytest.mark.parametrize("method", ["satt", "satterthwaite", "kr", "kenward_roger"])
    def test_pvalues_method_aliases(self, method: str) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)

        results = pvalues_with_ddf(model, method=method)

        assert set(results) == {"(Intercept)", "x"}

    def test_pvalues_t_values_consistent(self) -> None:
        model = lmer("y ~ x + (1|group)", DDF_DATA)
        results = pvalues_with_ddf(model, method="Satterthwaite")
        vcov = model.vcov()

        for i, name in enumerate(model.matrices.fixed_names):
            estimate, t_val, _ = results[name]
            se = np.sqrt(vcov[i, i])
            expected_t = estimate / se if se > 0 else 0.0
            assert np.isclose(t_val, expected_t, rtol=1e-5) or np.isnan(t_val)


class TestDDFCache:
    def test_cache_is_isolated_per_fitted_model(self) -> None:
        first = lmer("y ~ x + (1|group)", DDF_DATA)
        second = lmer("y ~ x + (1|group)", DDF_DATA)
        ddf_module.clear_vcov_grad_cache()

        try:
            satterthwaite_df(first)
            satterthwaite_df(second)

            assert len(ddf_module._vcov_grad_cache) == 2
        finally:
            ddf_module.clear_vcov_grad_cache()


class TestDDFWithDifferentModels:
    def test_ddf_random_slope(self) -> None:
        model = lmer("y ~ x + (x|group)", DDF_DATA)
        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert len(satt.df) == 2
        assert len(kr.df) == 2
        assert np.all(satt.df > 0)
        assert np.all(kr.df > 0)

    def test_ddf_multiple_random_effects(self) -> None:
        data = DDF_DATA.copy()
        data["group2"] = np.tile([f"H{i}" for i in range(5)], len(data) // 5 + 1)[: len(data)]
        model = lmer("y ~ x + (1|group) + (1|group2)", data)

        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert len(satt.df) == 2
        assert len(kr.df) == 2

    def test_ddf_larger_dataset(self) -> None:
        large_data = create_ddf_data(n_groups=30, n_per_group=10)
        model = lmer("y ~ x + (1|group)", large_data)

        satt = satterthwaite_df(model)
        kr = kenward_roger_df(model)

        assert np.all(satt.df > 0)
        assert np.all(kr.df > 0)
        assert np.all(satt.df <= 300 - 2)


class TestWeightedDDF:
    def test_information_matches_direct_weighted_marginal_covariance(self) -> None:
        from mixedlm.estimation.reml import _build_lambda

        data, weights = create_weighted_ddf_data()
        model = lmer(
            "y ~ x + (1 | group)",
            data,
            weights=weights,
            control=LmerControl(use_rust=False, em_init=False),
        )
        lambda_matrix = _build_lambda(model.theta, model.matrices.random_structures).toarray()
        z_lambda = model.matrices.Z @ lambda_matrix
        marginal_covariance = model.sigma**2 * (np.diag(1.0 / weights) + z_lambda @ z_lambda.T)
        expected = model.matrices.X.T @ linalg.solve(
            marginal_covariance, model.matrices.X, assume_a="pos"
        )

        information = _xt_vinv_x_from_theta(model, model.theta)

        assert_allclose(information, expected, rtol=0, atol=2e-10)

    def test_global_weight_rescaling_preserves_ddf_and_pvalues(self) -> None:
        data, weights = create_weighted_ddf_data()
        scale = 25.0
        control = LmerControl(use_rust=False, em_init=False)
        model = lmer("y ~ x + (1 | group)", data, weights=weights, control=control)
        scaled = lmer("y ~ x + (1 | group)", data, weights=scale * weights, control=control)

        clear_vcov_grad_cache()
        satterthwaite = satterthwaite_df(model)
        kenward_roger = kenward_roger_df(model)
        pvalues = pvalues_with_ddf(model)
        scaled_satterthwaite = satterthwaite_df(scaled)
        scaled_kenward_roger = kenward_roger_df(scaled)
        scaled_pvalues = pvalues_with_ddf(scaled)

        assert 5.0 < satterthwaite["(Intercept)"] < 20.0
        assert satterthwaite["x"] > 50.0
        assert_allclose(scaled_satterthwaite.df, satterthwaite.df, rtol=2e-6, atol=5e-5)
        assert_allclose(scaled_kenward_roger.df, kenward_roger.df, rtol=2e-6, atol=5e-5)
        for name in model.matrices.fixed_names:
            assert_allclose(scaled_pvalues[name], pvalues[name], rtol=2e-7, atol=2e-9)

    def test_correlated_random_slope_ddf_is_weight_scale_invariant(self) -> None:
        data, weights = create_weighted_slope_ddf_data()
        scale = 25.0
        control = LmerControl(use_rust=False, em_init=False)
        model = lmer("y ~ x + (x | group)", data, weights=weights, control=control)
        scaled = lmer("y ~ x + (x | group)", data, weights=scale * weights, control=control)

        clear_vcov_grad_cache()
        model_df = satterthwaite_df(model).df
        scaled_df = satterthwaite_df(scaled).df

        assert_allclose(scaled_df, model_df, rtol=5e-6, atol=5e-5)
        assert_allclose(model_df, [11.0, 11.0], rtol=0, atol=5e-3)

    def test_gradient_cache_is_isolated_between_fitted_models(self) -> None:
        data, weights = create_weighted_ddf_data()
        shifted_data = data.copy()
        shifted_data["x"] = np.square(shifted_data["x"]) + np.linspace(0.0, 1.0, len(data))
        control = LmerControl(use_rust=False, em_init=False)
        first = lmer("y ~ x + (1 | group)", data, weights=weights, control=control)
        second_fit = lmer(
            "y ~ x + (1 | group)",
            shifted_data,
            weights=weights[::-1].copy(),
            control=control,
        )
        second = replace(second_fit, theta=first.theta.copy(), sigma=first.sigma)

        clear_vcov_grad_cache()
        first_gradients, _ = _vcov_derivatives(first)
        cached_second_gradients, cached_second_covariance = _vcov_derivatives(second)
        clear_vcov_grad_cache()
        fresh_second_gradients, fresh_second_covariance = _vcov_derivatives(second)

        assert not np.allclose(first_gradients[0], fresh_second_gradients[0])
        assert_allclose(cached_second_gradients[0], fresh_second_gradients[0], atol=1e-12)
        assert_allclose(cached_second_covariance, fresh_second_covariance, atol=1e-12)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference.drop1 import Drop1Result, _likelihood_ratio, drop1_glmer, drop1_lmer


@pytest.fixture
def multi_predictor_data():
    np.random.seed(42)
    n_groups = 8
    n_per_group = 20
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    group_effects = np.repeat(np.random.randn(n_groups) * 2, n_per_group)
    y = 5.0 + 2.0 * x1 + 1.5 * x2 + 0.5 * x3 + group_effects + np.random.randn(n) * 0.5

    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "group": groups})


@pytest.fixture
def binomial_data():
    np.random.seed(42)
    n_groups = 8
    n_per_group = 25
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    group_effects = np.repeat(np.random.randn(n_groups) * 0.3, n_per_group)
    eta = -0.5 + 0.5 * x1 + 0.3 * x2 + group_effects
    prob = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, prob)

    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})


class TestDrop1Result:
    def test_str_method(self):
        result = Drop1Result(
            terms=["x1", "x2"],
            df=[3, 3],
            aic=[100.0, 105.0],
            lrt=[5.0, 2.0],
            p_value=[0.01, 0.10],
            full_model_aic=98.0,
            full_model_df=4,
        )
        s = str(result)
        assert "Single term deletions" in s
        assert "x1" in s
        assert "x2" in s
        assert "AIC" in s

    def test_repr_method(self):
        result = Drop1Result(
            terms=["x1", "x2"],
            df=[3, 3],
            aic=[100.0, 105.0],
            lrt=[5.0, 2.0],
            p_value=[0.01, 0.10],
            full_model_aic=98.0,
            full_model_df=4,
        )
        r = repr(result)
        assert "Drop1Result" in r
        assert "n_terms=2" in r

    def test_none_lrt_pvalue(self):
        result = Drop1Result(
            terms=["x1"],
            df=[3],
            aic=[100.0],
            lrt=[None],
            p_value=[None],
            full_model_aic=98.0,
            full_model_df=4,
        )
        s = str(result)
        assert "x1" in s

    def test_extreme_likelihood_ratio_keeps_nonzero_tail_probability(self):
        lrt, p_value = _likelihood_ratio(2, 1, 50.0, 0.0, "Chisq")

        assert lrt == 100.0
        assert p_value is not None
        assert 0.0 < p_value < 1e-20


class TestDrop1Lmer:
    def test_basic_drop1(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data)

        assert isinstance(result, Drop1Result)
        assert len(result.terms) >= 1
        assert all(aic > 0 for aic in result.aic)

    def test_terms_are_predictors(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data)

        for term in result.terms:
            assert term in ["x1", "x2", "x3"]

    def test_lrt_positive(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data, test="Chisq")

        for lrt in result.lrt:
            if lrt is not None:
                assert lrt >= 0

    def test_pvalue_valid(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data, test="Chisq")

        for p in result.p_value:
            if p is not None:
                assert 0 <= p <= 1

    def test_no_test(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data, test="none")

        for lrt, p in zip(result.lrt, result.p_value, strict=True):
            assert lrt is None
            assert p is None

    def test_reml_model_is_compared_with_ml(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data)
        ml_model = model.refitML()
        expected = drop1_lmer(ml_model, multi_predictor_data)

        assert model.REML
        assert result.terms == expected.terms
        assert result.full_model_aic == pytest.approx(ml_model.AIC())
        assert result.aic == pytest.approx(expected.aic)
        assert result.lrt == pytest.approx(expected.lrt)
        assert result.p_value == pytest.approx(expected.p_value)

    @pytest.mark.parametrize("test", ["invalid", "F", "LRT"])
    def test_invalid_test_raises(self, multi_predictor_data, test):
        model = lmer("y ~ x1 + x2 + (1|group)", multi_predictor_data)

        with pytest.raises(ValueError, match="test must be"):
            drop1_lmer(model, multi_predictor_data, test=test)

    @pytest.mark.parametrize("n_jobs", [0, -2])
    def test_invalid_worker_count_raises(self, multi_predictor_data, n_jobs):
        model = lmer("y ~ x1 + x2 + (1|group)", multi_predictor_data)

        with pytest.raises(ValueError, match="n_jobs"):
            drop1_lmer(model, multi_predictor_data, n_jobs=n_jobs)


class TestDrop1Glmer:
    def test_basic_drop1(self, binomial_data):
        model = glmer("y ~ x1 + x2 + (1|group)", binomial_data, family=Binomial())
        result = drop1_glmer(model, binomial_data)

        assert isinstance(result, Drop1Result)
        assert len(result.terms) >= 1

    def test_terms_are_predictors(self, binomial_data):
        model = glmer("y ~ x1 + x2 + (1|group)", binomial_data, family=Binomial())
        result = drop1_glmer(model, binomial_data)

        for term in result.terms:
            assert term in ["x1", "x2"]

    def test_lrt_values(self, binomial_data):
        model = glmer("y ~ x1 + x2 + (1|group)", binomial_data, family=Binomial())
        result = drop1_glmer(model, binomial_data, test="Chisq")

        for lrt in result.lrt:
            if lrt is not None:
                assert lrt >= 0


class TestDrop1AIC:
    def test_aic_ordering(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data)

        for aic in result.aic:
            assert aic > 0

    def test_full_model_aic(self, multi_predictor_data):
        model = lmer("y ~ x1 + x2 + (1|group)", multi_predictor_data)
        result = drop1_lmer(model, multi_predictor_data)

        assert result.full_model_aic > 0
        assert result.full_model_df > 0


class TestDrop1EdgeCases:
    def test_intercept_only_model_has_no_deletions(self):
        np.random.seed(42)
        groups = np.repeat([f"G{i}" for i in range(5)], 20)
        data = pd.DataFrame({"y": np.random.randn(100), "group": groups})
        model = lmer("y ~ 1 + (1|group)", data)

        result = drop1_lmer(model, data, n_jobs=2)

        assert result.terms == []
        assert result.aic == []

    def test_single_predictor(self):
        np.random.seed(42)
        n = 100
        groups = np.repeat([f"G{i}" for i in range(5)], 20)
        x = np.random.randn(n)
        y = 5.0 + 2.0 * x + np.random.randn(n)
        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        model = lmer("y ~ x + (1|group)", data)
        result = drop1_lmer(model, data)

        assert len(result.terms) == 1
        assert "x" in result.terms

    def test_with_interaction(self):
        np.random.seed(42)
        n = 100
        groups = np.repeat([f"G{i}" for i in range(5)], 20)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 5.0 + x1 + x2 + 0.5 * x1 * x2 + np.random.randn(n)
        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})

        model = lmer("y ~ x1 * x2 + (1|group)", data)
        result = drop1_lmer(model, data)

        assert result.terms == ["x1:x2"]

    def test_marginality_keeps_unrelated_main_effect(self):
        np.random.seed(42)
        n = 100
        groups = np.repeat([f"G{i}" for i in range(5)], 20)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = 5.0 + x1 + x2 + x3 + x1 * x2 + np.random.randn(n)
        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "group": groups})

        model = lmer("y ~ x1 * x2 + x3 + (1|group)", data)
        result = drop1_lmer(model, data)

        assert result.terms == ["x1:x2", "x3"]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference.drop1 import Drop1Result, drop1_glmer, drop1_lmer


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

        term_set = set(result.terms)
        assert len(term_set) >= 1

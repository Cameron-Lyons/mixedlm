from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference.drop1 import Drop1Result, _likelihood_ratio, drop1_glmer, drop1_lmer
from mixedlm.models.control import lmerControl
from mixedlm.models.lmer import LmerMod
from scipy import stats


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
    rng = np.random.default_rng(42)
    n_groups = 8
    n_per_group = 25
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    group_effects = np.repeat(rng.normal(size=n_groups), n_per_group)
    eta = -0.5 + 0.5 * x1 + 0.3 * x2 + group_effects
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})


def assert_matches_explicit_refits(model, result, refit, terms):
    """Each deletion is an ML refit without that term, compared by AIC and LRT."""
    assert result.terms == terms
    assert result.full_model_aic == pytest.approx(model.AIC(), rel=1e-10)
    for term, aic, lrt, p_value in zip(terms, result.aic, result.lrt, result.p_value, strict=True):
        reduced = refit(" + ".join(other for other in terms if other != term))
        statistic = 2 * (model.logLik().value - reduced.logLik().value)
        assert aic == pytest.approx(reduced.AIC(), abs=1e-4)
        assert lrt == pytest.approx(statistic, abs=1e-4)
        assert p_value == pytest.approx(stats.chi2.sf(lrt, 1), rel=1e-10)


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
    def test_each_deletion_matches_an_explicit_ml_refit(self, multi_predictor_data):
        data = multi_predictor_data
        model = lmer("y ~ x1 + x2 + x3 + (1|group)", data, REML=False)

        result = drop1_lmer(model, data, test="Chisq")

        assert_matches_explicit_refits(
            model,
            result,
            lambda fixed: lmer(f"y ~ {fixed} + (1|group)", data, REML=False),
            ["x1", "x2", "x3"],
        )

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

    def test_reduced_models_reuse_the_fitted_control(self, multi_predictor_data):
        # COBYQA counts one iteration per evaluation limit; "auto" adds a second stage.
        control = lmerControl(optimizer="COBYQA", maxiter=2, check_conv=False)
        model = replace(lmer("y ~ x1 + x2 + (1|group)", multi_predictor_data), control=control)
        result = drop1_lmer(model, multi_predictor_data)

        comparison = model.refitML()
        assert comparison.control is control
        for term, aic in zip(result.terms, result.aic, strict=True):
            reduced = LmerMod(
                f"y ~ {'x2' if term == 'x1' else 'x1'} + (1|group)",
                multi_predictor_data,
                REML=False,
                control=control,
            ).fit(start=comparison.theta)
            assert reduced.n_iter <= 2
            assert aic == reduced.AIC()
        default = drop1_lmer(replace(model, control=None), multi_predictor_data)
        assert result.aic != pytest.approx(default.aic, rel=1e-6)

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

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_failed_deletion_is_reported_and_other_deletions_kept(
        self, multi_predictor_data, n_jobs
    ):
        model = lmer("y ~ x1 + x2 + (1|group)", multi_predictor_data, REML=False)
        # Only the model without x2 needs the missing x1 column.
        data = multi_predictor_data.drop(columns="x1")

        with pytest.warns(RuntimeWarning, match="failed for 1 term") as record:
            result = drop1_lmer(model, data, n_jobs=n_jobs)

        assert record[0].filename == __file__
        assert "x2: " in str(record[0].message)
        expected = lmer("y ~ x2 + (1|group)", data, REML=False)
        assert result.terms == ["x1"]
        assert result.df == [4]
        assert result.aic == pytest.approx([expected.AIC()])
        assert result.lrt == pytest.approx([2 * (model.logLik().value - expected.logLik().value)])


class TestDrop1Glmer:
    def test_each_deletion_matches_an_explicit_refit(self, binomial_data):
        model = glmer("y ~ x1 + x2 + (1|group)", binomial_data, family=Binomial())

        result = drop1_glmer(model, binomial_data, test="Chisq")

        assert_matches_explicit_refits(
            model,
            result,
            lambda fixed: glmer(f"y ~ {fixed} + (1|group)", binomial_data, family=Binomial()),
            ["x1", "x2"],
        )


def grouped_data(*predictors):
    """Five groups with a clear group effect, so no fit is on the variance boundary."""
    rng = np.random.default_rng(42)
    groups = np.repeat([f"G{i}" for i in range(5)], 20)
    data = pd.DataFrame({name: rng.normal(size=100) for name in predictors})
    data["y"] = 5.0 + np.repeat(rng.normal(scale=2.0, size=5), 20) + rng.normal(size=100)
    data["group"] = groups
    return data


class TestDrop1EdgeCases:
    def test_intercept_only_model_has_no_deletions(self):
        data = grouped_data()
        model = lmer("y ~ 1 + (1|group)", data)

        result = drop1_lmer(model, data, n_jobs=2)

        assert result.terms == []
        assert result.aic == []

    def test_single_predictor(self):
        data = grouped_data("x")
        data["y"] += 2.0 * data["x"]

        model = lmer("y ~ x + (1|group)", data)
        result = drop1_lmer(model, data)

        assert result.terms == ["x"]

    def test_with_interaction(self):
        data = grouped_data("x1", "x2")
        data["y"] += data["x1"] + data["x2"] + 0.5 * data["x1"] * data["x2"]

        model = lmer("y ~ x1 * x2 + (1|group)", data)
        result = drop1_lmer(model, data)

        assert result.terms == ["x1:x2"]

    def test_marginality_keeps_unrelated_main_effect(self):
        data = grouped_data("x1", "x2", "x3")
        data["y"] += data["x1"] + data["x2"] + data["x3"] + data["x1"] * data["x2"]

        model = lmer("y ~ x1 * x2 + x3 + (1|group)", data)
        result = drop1_lmer(model, data)

        assert result.terms == ["x1:x2", "x3"]

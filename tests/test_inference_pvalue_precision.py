from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm.families import Gaussian
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.anova import anova, anova_type3
from mixedlm.inference.emmeans import EmmeanResult, Emmeans, _adjust_pvalues
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special


def _linear_result(formula_text="y ~ x", deviance=0.0):
    data = pd.DataFrame({"y": np.ones(102), "x": np.linspace(-1.0, 1.0, 102)})
    formula = parse_formula(formula_text)
    matrices = build_model_matrices(formula, data)
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.empty(0),
        beta=np.zeros(matrices.n_fixed),
        sigma=1.0,
        u=np.empty(0),
        deviance=deviance,
        REML=False,
        converged=True,
        n_iter=0,
    )


@pytest.mark.parametrize("statistic", [0.0, 4.0, 100.0, 400.0])
def test_likelihood_ratio_preserves_upper_tail_probability(statistic):
    reduced = _linear_result("y ~ 1", deviance=1000.0)
    full = _linear_result(deviance=1000.0 - statistic)

    actual = anova(reduced, full)

    expected = special.erfc(np.sqrt(statistic / 2.0))
    assert actual.chi_df[1] == 1
    assert actual.chi_sq[1] == statistic
    assert_allclose(actual.p_value[1], expected, rtol=1e-12, atol=0.0)
    assert actual.p_value[1] > 0.0


@pytest.mark.parametrize("statistic", [0.0, 4.0, 100.0, 400.0])
def test_type3_f_test_preserves_upper_tail_probability(statistic):
    result = _linear_result()
    result.beta[1] = np.sqrt(statistic * result.vcov()[1, 1])

    actual = anova_type3(result)

    assert actual.terms == ["x"]
    assert_allclose(actual.f_value, [statistic])
    assert_array_equal(actual.den_df, [100.0])
    expected = special.betainc(50.0, 0.5, 100.0 / (100.0 + statistic))
    assert_allclose(actual.p_value, [expected], rtol=1e-12, atol=0.0)
    assert actual.p_value[0] > 0.0


def _means():
    estimates = np.array([-15.0, 0.0, 15.0]) * np.sqrt(2.0)
    levels = ["A", "B", "C"]
    result = EmmeanResult(
        emmean=estimates,
        se=np.ones(3),
        df=100.0,
        lower=estimates - 2.0,
        upper=estimates + 2.0,
        grid=pd.DataFrame({"treatment": levels}),
        level=0.95,
    )
    return Emmeans(
        result=result,
        _L=np.eye(3),
        _vcov=np.eye(3),
        _beta=estimates,
        _df=100.0,
        _specs=["treatment"],
        _levels=[levels],
    )


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
@pytest.mark.parametrize("adjustment", ["none", "bonferroni"])
def test_marginal_mean_contrasts_preserve_two_sided_t_tails(kind, adjustment):
    means = _means()
    if kind == "pairs":
        actual = means.pairs(adjust=adjustment)
    elif kind == "control":
        actual = means.contrast("trt.vs.ctrl", adjust=adjustment)
    else:
        actual = means.contrast(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]), adjust=adjustment)

    expected = special.betainc(50.0, 0.5, 100.0 / (100.0 + actual.t_ratio**2))
    if adjustment == "bonferroni":
        expected = expected * len(expected)
    assert np.all(actual.p_value > 0.0)
    assert_allclose(actual.p_value, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "z_value, formatted",
    [
        (0.0, "1.0000"),
        (3.0, "0.0027"),
        (4.0, "6.33e-05"),
        (9.0, "< 2e-16"),
        (-9.0, "< 2e-16"),
        (np.sqrt(2.0) * special.erfcinv(2.1e-16), "2.10e-16"),
    ],
)
def test_glmm_summary_keeps_small_pvalues_and_readable_format(z_value, formatted, monkeypatch):
    from importlib import import_module

    glmer_module = import_module("mixedlm.models.glmer")

    linear = _linear_result("y ~ 1")
    result = GlmerResult(
        formula=linear.formula,
        matrices=linear.matrices,
        family=Gaussian(),
        theta=np.empty(0),
        beta=np.array([z_value / np.sqrt(102)]),
        u=np.empty(0),
        deviance=0.0,
        converged=True,
        n_iter=0,
        nAGQ=1,
    )
    probabilities = []
    original_significance = glmer_module._get_signif_code

    def capture_probability(p):
        probabilities.append(p)
        return original_significance(p)

    monkeypatch.setattr(glmer_module, "_get_signif_code", capture_probability)
    summary = result.summary()

    expected = special.erfc(abs(z_value) / np.sqrt(2))
    assert_allclose(probabilities, [expected], rtol=1e-12, atol=0.0)
    line = next(line for line in summary.splitlines() if line.startswith("(Intercept)"))
    assert line.split(maxsplit=4)[4].startswith(formatted)


@pytest.mark.parametrize("method, multiplier", [("holm", 3.0), ("fdr", 1.5)])
def test_adjusted_contrasts_keep_undefined_test_missing(method, multiplier):
    means = _means()
    constraints = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    with np.errstate(invalid="ignore"):
        actual = means.contrast(constraints, adjust=method)

    expected = multiplier * special.betainc(50.0, 0.5, 100.0 / (100.0 + 15.0**2))
    assert np.isnan(actual.p_value[1])
    assert_allclose(actual.p_value[[0, 2]], [expected, expected], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("method", ["holm", "fdr"])
@pytest.mark.parametrize(
    "probabilities, holm, fdr",
    [
        ([], [], []),
        ([0.02], [0.02], [0.02]),
        ([0.02] * 4, [0.08] * 4, [0.02] * 4),
        (
            [1e-300, 0.01, 0.03, 0.04, 1.0],
            [5e-300, 0.04, 0.09, 0.09, 1.0],
            [5e-300, 0.025, 0.05, 0.05, 1.0],
        ),
        (
            [0.01, 0.04, 0.03, 0.01, 0.9, 0.0, 1e-25],
            [0.05, 0.09, 0.09, 0.05, 0.9, 0.0, 6e-25],
            [0.0175, 7 * 0.04 / 6, 0.042, 0.0175, 0.9, 0.0, 3.5e-25],
        ),
        ([np.nan], [np.nan], [np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]),
        (
            [np.nan, 0.01, 0.04, 0.03, np.nan],
            [np.nan, 0.05, 0.12, 0.12, np.nan],
            [np.nan, 0.05, 0.2 / 3, 0.2 / 3, np.nan],
        ),
    ],
)
def test_rank_adjustments_preserve_tails_ties_order_and_missing_values(
    method, probabilities, holm, fdr
):
    p = np.asarray(probabilities)
    original = p.copy()
    expected = np.asarray(holm if method == "holm" else fdr)
    order = np.random.default_rng(42).permutation(len(p))

    actual = _adjust_pvalues(p, method, len(p), 100.0)
    reordered = _adjust_pvalues(p[order], method, len(p), 100.0)

    assert_allclose(actual, expected, rtol=1e-14, atol=0.0, equal_nan=True)
    assert_allclose(reordered, expected[order], rtol=1e-14, atol=0.0, equal_nan=True)
    assert_array_equal(p, original)

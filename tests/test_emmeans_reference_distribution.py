from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.families import Binomial, Poisson
from mixedlm.inference.emmeans import ContrastResult, EmmeanResult, Emmeans, emmeans
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special, stats


def _data():
    rng = np.random.default_rng(811)
    group = np.repeat(np.arange(12), 12)
    treatment = np.tile(np.arange(3), len(group) // 3)
    eta = -0.2 + np.take([-0.7, 0.1, 0.8], treatment)
    eta += rng.normal(scale=0.8, size=12)[group]
    return (
        pd.DataFrame(
            {"group": group.astype(str), "treatment": np.take(["A", "B", "C"], treatment)}
        ),
        eta,
        rng,
    )


@pytest.fixture(scope="module", params=["binomial", "poisson"])
def generalized_model(request):
    data, eta, rng = _data()
    if request.param == "binomial":
        family = Binomial()
        data["y"] = rng.binomial(1, special.expit(eta))
    else:
        family = Poisson()
        data["y"] = rng.poisson(np.exp(eta))
    return glmer("y ~ treatment + (1 | group)", data, family=family)


@pytest.fixture(scope="module")
def linear_model():
    data, eta, rng = _data()
    data["y"] = eta + rng.normal(scale=0.5, size=len(eta))
    return lmer("y ~ treatment + (1 | group)", data)


def _contrast(means, kind, adjust):
    if kind == "pairs":
        return means.pairs(adjust=adjust)
    if kind == "control":
        return means.contrast("trt.vs.ctrl", adjust=adjust)
    return means.contrast(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]), adjust=adjust)


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
@pytest.mark.parametrize("scale", ["link", "response"])
@pytest.mark.parametrize("adjust", ["none", "bonferroni"])
def test_glmm_contrasts_use_normal_reference(generalized_model, kind, scale, adjust):
    means = emmeans(generalized_model, "treatment", type=scale)

    actual = _contrast(means, kind, adjust)

    expected = special.erfc(np.abs(actual.t_ratio) / np.sqrt(2))
    if adjust == "bonferroni":
        expected = np.minimum(expected * len(expected), 1.0)
    assert means.result.df == np.inf
    assert actual.df == np.inf
    assert_allclose(actual.p_value, expected, rtol=1e-12, atol=0)
    assert "z.ratio" in str(actual)
    assert "t.ratio" not in str(actual)

    link_contrasts = _contrast(emmeans(generalized_model, "treatment", type="link"), kind, adjust)
    assert_allclose(actual.estimate, link_contrasts.estimate, rtol=1e-13, atol=0)
    assert_allclose(actual.se, link_contrasts.se, rtol=1e-13, atol=0)


def test_glmm_tukey_uses_infinite_df(generalized_model):
    means = emmeans(generalized_model, "treatment")

    actual = means.pairs(adjust="tukey")

    expected = stats.studentized_range.sf(np.abs(actual.t_ratio) * np.sqrt(2), 3, np.inf)
    assert actual.df == np.inf
    assert_allclose(actual.p_value, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("scale", ["link", "response"])
def test_glmm_interval_and_test_reference_agree(generalized_model, scale):
    means = emmeans(generalized_model, "treatment", type=scale, level=0.9)
    link = emmeans(generalized_model, "treatment", type="link", level=0.9)
    critical = stats.norm.ppf(0.95)
    expected_lower = link.result.emmean - critical * link.result.se
    expected_upper = link.result.emmean + critical * link.result.se
    if scale == "response":
        expected_lower = generalized_model.family.link.inverse(expected_lower)
        expected_upper = generalized_model.family.link.inverse(expected_upper)

    assert means.result.df == np.inf
    assert_allclose(means.result.lower, expected_lower, rtol=1e-13, atol=0)
    assert_allclose(means.result.upper, expected_upper, rtol=1e-13, atol=0)


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
def test_lmm_contrasts_keep_finite_student_t_reference(linear_model, kind):
    means = emmeans(linear_model, "treatment")

    actual = _contrast(means, kind, "none")

    df = float(linear_model.df_residual())
    expected = special.betainc(df / 2, 0.5, df / (df + actual.t_ratio**2))
    assert means.result.df == df
    assert actual.df == df
    assert_allclose(actual.p_value, expected, rtol=1e-12, atol=0)
    assert "t.ratio" in str(actual)
    assert "z.ratio" not in str(actual)


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
def test_normal_contrast_tails_remain_representable(kind):
    estimates = np.array([-10.0, 0.0, 10.0]) * np.sqrt(2)
    means = Emmeans(
        result=EmmeanResult(
            estimates,
            np.ones(3),
            np.inf,
            estimates - 2,
            estimates + 2,
            pd.DataFrame({"treatment": ["A", "B", "C"]}),
            0.95,
        ),
        _L=np.eye(3),
        _vcov=np.eye(3),
        _beta=estimates,
        _df=np.inf,
        _specs=["treatment"],
        _levels=[["A", "B", "C"]],
    )

    actual = _contrast(means, kind, "none")

    assert np.all(actual.p_value > 0)
    assert_allclose(
        actual.p_value, special.erfc(np.abs(actual.t_ratio) / np.sqrt(2)), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("df", [20.0, np.inf])
def test_contrast_table_preserves_undefined_and_small_probabilities(df):
    result = ContrastResult(
        contrast=["undefined", "ordinary", "small", "tiny"],
        estimate=np.array([0.0, 0.5, 1.0, 2.0]),
        se=np.array([0.0, 0.2, 0.1, 0.1]),
        df=df,
        t_ratio=np.array([np.nan, 2.5, 10.0, 20.0]),
        p_value=np.array([np.nan, 0.03, 1e-6, 1e-30]),
        adjust="none",
    )

    rows = {line.split()[0]: line.split()[1:] for line in str(result).splitlines() if line.split()}

    assert rows["undefined"][-1] == "nan"
    assert rows["ordinary"][-1] == "0.0300"
    assert rows["small"][-1] == "1.00e-06"
    assert rows["tiny"][-2:] == ["<", "2e-16"]


def test_undefined_public_contrast_prints_missing_probability(linear_model):
    means = emmeans(linear_model, "treatment")
    with np.errstate(invalid="ignore"):
        actual = means.contrast(np.zeros((1, 3)), adjust="none")

    assert np.isnan(actual.p_value[0])
    row = next(line for line in str(actual).splitlines() if line.startswith("C1"))
    assert row.split()[-1] == "nan"


def test_empty_contrast_table_prints(linear_model):
    actual = emmeans(linear_model, "treatment").contrast(np.empty((0, 3)))

    assert_array_equal(actual.estimate, np.empty(0))
    assert "p.value" in str(actual)


@pytest.mark.parametrize("columns", [[], ["treatment"]])
def test_empty_marginal_mean_table_prints(columns):
    empty = np.empty(0)
    result = EmmeanResult(empty, empty, np.inf, empty, empty, pd.DataFrame(columns=columns), 0.95)

    assert "Estimated Marginal Means" in str(result)
    assert "Confidence level: 95%" in str(result)

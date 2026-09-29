from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm.inference.emmeans import EmmeanResult, Emmeans, _adjust_pvalues
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

emmeans_module = import_module("mixedlm.inference.emmeans")
_OMITTED = object()
_KINDS = ["pairs", "pairwise", "control", "custom"]


@pytest.fixture
def means():
    estimates = np.array([0.0, 0.5, 1.5, 2.0])
    covariance = np.full((4, 4), 0.2)
    np.fill_diagonal(covariance, 1.0)
    levels = ["A", "B", "C", "D"]
    return Emmeans(
        result=EmmeanResult(
            estimates,
            np.ones(4),
            80.0,
            estimates - 2.0,
            estimates + 2.0,
            pd.DataFrame({"treatment": levels}),
            0.95,
        ),
        _L=np.eye(4),
        _vcov=covariance,
        _beta=estimates,
        _df=80.0,
        _specs=["treatment"],
        _levels=[levels],
    )


def _evaluate(means, kind, adjustment=_OMITTED):
    kwargs = {} if adjustment is _OMITTED else {"adjust": adjustment}
    if kind == "pairs":
        return means.pairs(**kwargs)
    if kind == "pairwise":
        return means.contrast("pairwise", **kwargs)
    if kind == "control":
        return means.contrast("trt.vs.ctrl", **kwargs)
    return means.contrast(np.diff(np.eye(4), axis=0), **kwargs)


def test_explicit_none_is_honored_for_pairwise_contrast(means):
    actual = means.contrast("pairwise", adjust="none")
    direct = means.pairs(adjust="none")

    assert actual.adjust == "none"
    assert_allclose(actual.p_value, 2 * stats.t.sf(np.abs(actual.t_ratio), 80.0), rtol=1e-12)
    assert_array_equal(actual.p_value, direct.p_value)
    assert np.any(actual.p_value < means.pairs().p_value)


@pytest.mark.parametrize("adjustment", [_OMITTED, None], ids=["omitted", "none-default"])
@pytest.mark.parametrize(
    "kind, default", [("pairwise", "tukey"), ("control", "none"), ("custom", "none")]
)
def test_method_defaults_remain_intact(means, kind, default, adjustment):
    actual = _evaluate(means, kind, adjustment)
    expected = _evaluate(means, kind, default)

    assert actual.adjust == default
    assert_array_equal(actual.p_value, expected.p_value)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("adjustment", ["none", "bonferroni", "holm", "fdr", "tukey"])
def test_adjustment_names_ignore_case_and_surrounding_whitespace(means, kind, adjustment):
    actual = _evaluate(means, kind, f"  {adjustment.upper()} \t")
    expected = _evaluate(means, kind, adjustment)

    assert actual.adjust == adjustment
    assert_array_equal(actual.p_value, expected.p_value)
    if adjustment != "none":
        assert f"P-value adjustment: {adjustment}" in str(actual)


@pytest.mark.parametrize("kind", _KINDS)
def test_bh_alias_uses_fdr_and_reports_canonical_name(means, kind):
    actual = _evaluate(means, kind, " BH ")
    expected = _evaluate(means, kind, "fdr")

    assert actual.adjust == "fdr"
    assert_array_equal(actual.p_value, expected.p_value)


def test_bh_alias_matches_known_adjustment():
    probabilities = np.array([0.01, 0.04, 0.03])

    actual = _adjust_pvalues(probabilities, "BH", 3, 80.0)

    assert_allclose(actual, [0.03, 0.04, 0.04], rtol=1e-14, atol=0)


def test_legacy_control_approximation_accepts_normalized_name(means):
    actual = means.contrast("trt.vs.ctrl", adjust=" Dunnett ")
    expected = means.contrast("trt.vs.ctrl", adjust="dunnett")

    assert actual.adjust == "dunnett"
    assert_array_equal(actual.p_value, expected.p_value)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("adjustment", ["holn", "", "unadjusted"])
def test_unknown_adjustment_fails_before_covariance_work(means, monkeypatch, kind, adjustment):
    def unexpected_work(*args, **kwargs):
        raise AssertionError("invalid adjustment should be rejected before projection")

    monkeypatch.setattr(emmeans_module, "_rowwise_quadratic_form", unexpected_work)

    with pytest.raises(ValueError, match="Unknown p-value adjustment") as error:
        _evaluate(means, kind, adjustment)

    assert "holm" in str(error.value)
    assert "BH" in str(error.value)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("adjustment", [False, 1, ["holm"]])
def test_non_string_adjustment_has_clear_error(means, kind, adjustment):
    with pytest.raises(TypeError, match="adjust must be a string"):
        _evaluate(means, kind, adjustment)


def test_pairs_requires_an_explicit_name_when_argument_is_provided(means):
    with pytest.raises(TypeError, match="adjust must be a string"):
        means.pairs(adjust=None)


@pytest.mark.parametrize("probabilities", [np.empty(0), np.array([0.01, 0.04])])
def test_tukey_cannot_silently_return_raw_values_without_statistics(probabilities):
    with pytest.raises(ValueError, match="t_ratio is required"):
        _adjust_pvalues(probabilities, "tukey", 3, 80.0)


def test_adjustment_helper_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown p-value adjustment"):
        _adjust_pvalues(np.array([0.01]), "holn", 3, 80.0)

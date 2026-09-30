from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm.inference.emmeans import ContrastResult, EmmeanResult, Emmeans
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats

module = import_module("mixedlm.inference.emmeans")


def _means(df=17.0, n=4):
    beta = np.linspace(-0.7, 1.0, n)
    covariance = np.diag(np.linspace(0.2, 0.5, n)) + 0.08
    zeros = np.zeros(n)
    return Emmeans(
        EmmeanResult(beta, zeros, df, zeros, zeros, pd.DataFrame({"group": list(range(n))}), 0.95),
        np.eye(n),
        covariance,
        beta,
        df,
        ["group"],
        [list(range(n))],
    )


def _contrasts(means, kind, adjust="none", level=0.9):
    if kind == "pairs":
        return means.pairs(adjust=adjust, level=level)
    if kind == "control":
        return means.contrast("trt.vs.ctrl", adjust=adjust, level=level)
    return means.contrast(
        np.array([[1.0, -0.5, -0.5, 0.0], [0.0, 1.0, -1.0, 0.0]]),
        adjust=adjust,
        level=level,
    )


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
@pytest.mark.parametrize("df", [17.0, np.inf])
@pytest.mark.parametrize("adjust", ["none", "bonferroni", "holm", "fdr", "dunnett"])
def test_contrast_intervals_use_the_requested_level_and_adjustment(kind, df, adjust):
    result = _contrasts(_means(df), kind, adjust)

    interval = result.confint()

    actual_adjust = "none" if adjust == "none" else "bonferroni"
    count = 1 if actual_adjust == "none" else len(result.estimate)
    critical = stats.t.isf(0.1 / (2 * count), df)
    assert_allclose(interval.lower, result.estimate - critical * result.se, rtol=1e-12)
    assert_allclose(interval.upper, result.estimate + critical * result.se, rtol=1e-12)
    assert_array_equal(interval.estimate, result.estimate)
    assert_array_equal(interval.SE, result.se)
    assert_array_equal(interval.df, np.full(len(result.estimate), df))
    assert interval.contrast.tolist() == result.contrast
    assert interval.attrs == {"level": 0.9, "adjust": actual_adjust, "requested_adjust": adjust}
    assert result.level == 0.9


@pytest.mark.parametrize("kind", ["pairs", "control"])
@pytest.mark.parametrize("df", [17.0, np.inf])
@pytest.mark.parametrize("level", [0.8, 0.95])
def test_tukey_intervals_invert_the_studentized_range_test(kind, df, level):
    result = _contrasts(_means(df), kind, "tukey", level)

    interval = result.confint()

    critical = (interval.upper - result.estimate) / result.se
    tail = stats.studentized_range.sf(critical * np.sqrt(2), 4, df)
    assert_allclose(tail, np.full(len(tail), 1 - level), rtol=1e-10, atol=1e-12)
    assert_allclose(interval.lower, result.estimate - critical * result.se)
    assert_array_equal((interval.lower > 0) | (interval.upper < 0), result.p_value < 1 - level)
    assert interval.attrs["adjust"] == "tukey"


@pytest.mark.parametrize("df", [17.0, np.inf])
def test_two_mean_tukey_intervals_match_pointwise_intervals(df):
    result = _means(df, 2).pairs(level=0.9)

    tukey = result.confint()
    pointwise = result.confint(adjust="none")

    assert_array_equal(tukey.lower, pointwise.lower)
    assert_array_equal(tukey.upper, pointwise.upper)


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
def test_interval_overrides_do_not_change_estimates_tests_or_defaults(kind):
    result = _contrasts(_means(), kind, "holm", 0.9)
    original = [array.copy() for array in (result.estimate, result.se, result.p_value)]

    default = result.confint()
    wider = result.confint(level=0.99)
    pointwise = result.confint(adjust="none")

    assert np.all(wider.lower < default.lower)
    assert np.all(wider.upper > default.upper)
    assert np.all(pointwise.lower > default.lower)
    assert np.all(pointwise.upper < default.upper)
    assert_frame_equal(result.confint(), default)
    assert result.level == 0.9
    assert result.adjust == "holm"
    for array, expected in zip((result.estimate, result.se, result.p_value), original, strict=True):
        assert_array_equal(array, expected)


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 2.0, np.nan, np.inf])
def test_invalid_confidence_levels_are_rejected(kind, level):
    with pytest.raises(ValueError, match="level"):
        _contrasts(_means(), kind, level=level)
    with pytest.raises(ValueError, match="level"):
        _contrasts(_means(), kind).confint(level=level)


@pytest.mark.parametrize("level", [True, np.bool_(False), "0.9", [0.9], np.array([0.9]), 0.9j])
def test_confidence_level_requires_a_real_scalar(level):
    with pytest.raises(TypeError, match="level"):
        _means().pairs(level=level)
    with pytest.raises(TypeError, match="level"):
        _means().pairs().confint(level=level)


@pytest.mark.parametrize("adjust", ["invalid", "", "tukye"])
def test_unknown_interval_adjustment_is_rejected(adjust):
    with pytest.raises(ValueError, match="interval adjustment"):
        _means().pairs().confint(adjust=adjust)


def test_interval_adjustment_requires_a_name():
    with pytest.raises(TypeError, match="adjust"):
        _means().pairs().confint(adjust=42)


def test_interval_adjustment_aliases_report_the_actual_method():
    result = _means().pairs()

    interval = result.confint(adjust=" BH ")

    assert interval.attrs["requested_adjust"] == "fdr"
    assert interval.attrs["adjust"] == "bonferroni"
    assert_allclose(interval.lower, result.confint(adjust="bonferroni").lower)


def test_arbitrary_custom_contrasts_require_a_supported_interval_method():
    result = _contrasts(_means(), "custom", "tukey")

    with pytest.raises(ValueError, match="Tukey intervals require pairwise"):
        result.confint()
    assert np.all(np.isfinite(result.confint(adjust="bonferroni").lower))


def test_intervals_are_lazy_and_reuse_scalar_quantiles(monkeypatch):
    module._contrast_critical_value.cache_clear()
    calls = []

    def quantile(tail, n_means, df):
        calls.append((tail, n_means, df))
        return 4.0

    monkeypatch.setattr(stats.studentized_range, "isf", quantile)
    try:
        result = _means().pairs(level=0.9)
        assert calls == []
        first = result.confint()
        assert_frame_equal(result.confint(), first)
        assert len(calls) == 1
        result.confint(level=0.8)
        assert len(calls) == 2
        result.df = 20.0
        result.confint()
        assert len(calls) == 3
    finally:
        module._contrast_critical_value.cache_clear()


def test_returned_intervals_are_independent_of_result_arrays():
    result = _means().pairs(adjust="none")
    original = result.confint()

    modified = result.confint()
    modified.loc[:, ["estimate", "SE", "lower", "upper"]] = -100.0

    assert_frame_equal(result.confint(), original)
    result.estimate += 2.0
    shifted = result.confint()
    assert_allclose(shifted.lower, original.lower + 2.0)
    assert_allclose(shifted.upper, original.upper + 2.0)


@pytest.mark.parametrize("kind", ["control", "custom"])
def test_empty_contrast_families_have_empty_interval_tables(kind):
    result = (
        _means(n=1).contrast("trt.vs.ctrl")
        if kind == "control"
        else _means().contrast(np.empty((0, 4)))
    )

    interval = result.confint()

    assert interval.shape == (0, 6)
    assert interval.columns.tolist() == ["contrast", "estimate", "SE", "df", "lower", "upper"]
    assert interval.attrs["level"] == 0.95


def test_exact_zero_contrast_has_a_point_interval():
    with np.errstate(invalid="ignore"):
        result = _means().contrast(np.zeros((1, 4)))

    interval = result.confint()

    assert_array_equal(interval.lower, [0.0])
    assert_array_equal(interval.upper, [0.0])
    assert np.isnan(result.p_value[0])


def test_extreme_confidence_level_does_not_round_the_tail_to_zero():
    result = _means().pairs(adjust="bonferroni", level=np.nextafter(1.0, 0.0))

    interval = result.confint()

    assert np.all(np.isfinite(interval.lower))
    assert np.all(np.isfinite(interval.upper))


def test_missing_values_remain_missing_and_count_in_the_interval_family():
    result = ContrastResult(
        ["missing", "second", "third"],
        np.array([np.nan, 1.0, 2.0]),
        np.array([np.nan, 0.2, 0.5]),
        17.0,
        np.array([np.nan, 5.0, 4.0]),
        np.array([np.nan, 0.001, 0.003]),
        "bonferroni",
    )

    interval = result.confint()

    assert np.isnan(interval.lower.iloc[0])
    assert np.isnan(interval.upper.iloc[0])
    critical = stats.t.isf(0.05 / 6, 17.0)
    assert_allclose(interval.lower.iloc[1:], [1.0 - critical * 0.2, 2.0 - critical * 0.5])


@pytest.mark.parametrize("adjust", ["none", "bonferroni", "tukey"])
def test_different_sized_families_keep_separate_adjustments(adjust):
    first = _means(n=2).pairs(adjust=adjust, level=0.9)
    second = _means(n=4).pairs(adjust=adjust, level=0.9)

    def join(name):
        return np.concatenate([getattr(first, name), getattr(second, name)])

    result = ContrastResult(
        first.contrast + second.contrast,
        join("estimate"),
        join("se"),
        first.df,
        join("t_ratio"),
        join("p_value"),
        adjust,
        level=0.9,
        _families=((0, 1, 2), (1, 7, 4)),
    )

    interval = result.confint()

    expected = pd.concat([first.confint(), second.confint()], ignore_index=True)
    assert_frame_equal(interval, expected)


def test_existing_result_constructor_supports_pointwise_intervals():
    result = ContrastResult(
        ["A - B"],
        np.array([1.0]),
        np.array([0.2]),
        17.0,
        np.array([5.0]),
        np.array([0.001]),
        "none",
    )

    interval = result.confint()

    assert interval.attrs["level"] == 0.95
    assert_allclose(interval.lower, 1.0 - stats.t.ppf(0.975, 17.0) * 0.2)

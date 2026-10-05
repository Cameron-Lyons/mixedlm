from __future__ import annotations

import itertools
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from tests._inference_results import identity_emmeans

module = import_module("mixedlm.inference.emmeans")


def _pair_rows(n_means=4):
    return np.array(
        [
            np.eye(n_means)[i] - np.eye(n_means)[j]
            for i, j in itertools.combinations(range(n_means), 2)
        ]
    )


@pytest.mark.parametrize("df", [17.0, np.inf])
@pytest.mark.parametrize("rows", [[0], [0, 2, 5], list(range(6))])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("limit", [1, 17, 1_000_000])
def test_custom_tukey_uses_mean_count_for_tests_and_intervals(monkeypatch, df, rows, scaled, limit):
    means = identity_emmeans(df)
    coefficients = _pair_rows()[rows]
    scale = np.resize([-2.0, 0.5, 4.0], len(rows)) if scaled else np.ones(len(rows))
    coefficients *= scale[:, None]
    original = coefficients.copy()
    monkeypatch.setattr(module, "_MAX_CONTRAST_ELEMENTS", limit, raising=False)

    result = means.contrast(coefficients, adjust="tukey", level=0.9)
    intervals = result.confint()

    expected = means.pairs(adjust="tukey", level=0.9)
    assert_allclose(result.estimate, expected.estimate[rows] * scale, rtol=1e-12)
    assert_allclose(result.se, expected.se[rows] * np.abs(scale), rtol=1e-12)
    assert_allclose(result.p_value, expected.p_value[rows], rtol=1e-12)
    assert_allclose(
        result.p_value,
        stats.studentized_range.sf(np.abs(result.t_ratio) * np.sqrt(2), 4, df),
        rtol=1e-12,
    )
    half_width = (intervals.upper - result.estimate) / result.se
    assert_allclose(stats.studentized_range.sf(half_width * np.sqrt(2), 4, df), 0.1, rtol=1e-10)
    assert_array_equal(coefficients, original)


@pytest.mark.parametrize(
    "representation", ["list", "frame", "readonly", "strided", "integer", "float32"]
)
def test_array_like_pairwise_coefficients_keep_the_same_inference(representation):
    coefficients = _pair_rows()
    if representation == "list":
        coefficients = coefficients.tolist()
    elif representation == "frame":
        coefficients = pd.DataFrame(coefficients)
    elif representation == "readonly":
        coefficients.setflags(write=False)
    elif representation == "strided":
        coefficients = np.repeat(coefficients, 2, axis=0)[::2]
    elif representation == "integer":
        coefficients = coefficients.astype(np.int64)
    elif representation == "float32":
        coefficients = coefficients.astype(np.float32)
    means = identity_emmeans()

    result = means.contrast(coefficients, adjust="tukey")

    expected = means.pairs()
    assert_allclose(result.estimate, expected.estimate, rtol=1e-13)
    assert_allclose(result.se, expected.se, rtol=1e-13)
    assert_allclose(result.p_value, expected.p_value, rtol=1e-13)


def test_tukey_intervals_can_be_requested_after_unadjusted_custom_tests():
    means = identity_emmeans()
    result = means.contrast(_pair_rows(), adjust="none", level=0.9)

    intervals = result.confint(adjust="tukey")

    expected = means.pairs(level=0.9).confint()
    assert_allclose(intervals.lower, expected.lower)
    assert_allclose(intervals.upper, expected.upper)
    assert result.adjust == "none"


@pytest.mark.parametrize("df", [17.0, np.inf])
@pytest.mark.parametrize("kind", ["pairs", "control", "single", "multiple", "empty"])
def test_dunnett_approximation_counts_comparisons_not_means(df, kind):
    means = identity_emmeans(df)
    if kind == "pairs":
        result = means.pairs(adjust="dunnett", level=0.9)
    elif kind == "control":
        result = means.contrast("trt.vs.ctrl", adjust="dunnett", level=0.9)
    else:
        coefficients = _pair_rows()[[0]] if kind == "single" else _pair_rows()
        if kind == "empty":
            coefficients = np.empty((0, 4))
        result = means.contrast(coefficients, adjust="dunnett", level=0.9)

    raw = 2 * stats.t.sf(np.abs(result.t_ratio), df)
    assert_allclose(result.p_value, np.minimum(raw * len(raw), 1.0), rtol=1e-12)
    if kind == "single":
        assert 0 < result.p_value[0] < 1
    intervals = result.confint()
    assert intervals.attrs["adjust"] == "bonferroni"
    assert_array_equal((intervals.lower > 0) | (intervals.upper < 0), result.p_value < 0.1)


@pytest.mark.parametrize(
    "coefficients",
    [
        [[1.0, -0.5, -0.5, 0.0]],
        [[1.0, -1.0, 1e-20, 0.0]],
        [[1.0, 0.0, 0.0, 0.0]],
        [[1.0, -2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0]],
        [[True, True, False, False]],
        np.array([[1, np.iinfo(np.uint64).max, 0, 0]], dtype=np.uint64),
        np.array([[np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, 0]], dtype=np.int64),
    ],
)
def test_non_pairwise_tukey_requests_fail_before_covariance_work(monkeypatch, coefficients):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid Tukey request reached covariance work")

    monkeypatch.setattr(module, "_rowwise_quadratic_form", unexpected)

    with pytest.raises(ValueError, match="Tukey adjustment requires pairwise"):
        identity_emmeans().contrast(coefficients, adjust="tukey")


@pytest.mark.parametrize(
    "coefficients, error, message",
    [
        (1.0, ValueError, "2-D"),
        ([1.0, -1.0, 0.0, 0.0], ValueError, "2-D"),
        (np.ones((1, 1, 4)), ValueError, "2-D"),
        ([[1.0, -1.0]], ValueError, "expected 4"),
        (np.empty((0, 3)), ValueError, "expected 4"),
        ([[1.0, -1.0], [1.0]], ValueError, "rectangular"),
        (np.array([[1.0, -1.0], [1.0]], dtype=object), ValueError, "rectangular"),
        ([[1.0, np.nan, 0.0, 0.0]], ValueError, "finite"),
        ([[1.0, np.inf, 0.0, 0.0]], ValueError, "finite"),
        ([[1.0, -np.inf, 0.0, 0.0]], ValueError, "finite"),
        ([[1.0, 2j, 0.0, 0.0]], TypeError, "real numeric"),
        ([[1.0, "bad", 0.0, 0.0]], TypeError, "real numeric"),
        (np.zeros((1, 4), dtype="datetime64[D]"), TypeError, "real numeric"),
        (
            np.ma.array([[1.0, -1.0, 0.0, 0.0]], mask=[[True, False, False, False]]),
            ValueError,
            "masked values",
        ),
    ],
)
def test_invalid_custom_matrices_have_clear_errors(monkeypatch, coefficients, error, message):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid coefficients reached covariance work")

    monkeypatch.setattr(module, "_rowwise_quadratic_form", unexpected)

    with pytest.raises(error, match=message):
        identity_emmeans().contrast(coefficients)


@pytest.mark.parametrize("limit", [1, 13, 1_000_000])
def test_validation_bounds_its_temporary_arrays(monkeypatch, limit):
    coefficients = np.tile(_pair_rows(), (40, 1))
    original_isfinite = np.isfinite
    observed = []

    def finite(values):
        observed.append(values.size)
        assert values.size <= max(limit, coefficients.shape[1])
        return original_isfinite(values)

    monkeypatch.setattr(module, "_MAX_CONTRAST_ELEMENTS", limit)
    monkeypatch.setattr(module.np, "isfinite", finite)

    validated, n_means = module._validate_custom_contrasts(coefficients, 4)

    assert validated is coefficients
    assert n_means == 4
    assert sum(observed) == coefficients.size


def test_late_nonfinite_coefficients_are_checked_after_a_nonpairwise_row(monkeypatch):
    coefficients = np.tile(_pair_rows(), (2, 1))
    coefficients[0] = [1.0, 0.0, 0.0, 0.0]
    coefficients[-1, -1] = np.nan
    monkeypatch.setattr(module, "_MAX_CONTRAST_ELEMENTS", 4)

    with pytest.raises(ValueError, match="finite"):
        identity_emmeans().contrast(coefficients)


def test_valid_nonpairwise_lists_preserve_linear_combination_results():
    means = identity_emmeans()
    coefficients = [[1.0, -0.5, -0.5, 0.0], [True, False, False, False]]

    result = means.contrast(coefficients, adjust="bonferroni")

    matrix = np.asarray(coefficients)
    assert_allclose(result.estimate, matrix @ means._beta)
    assert_allclose(result.se, np.sqrt(np.diag(matrix @ means._vcov @ matrix.T)))
    with pytest.raises(ValueError, match="Tukey intervals require pairwise"):
        result.confint(adjust="tukey")


@pytest.mark.parametrize("n_means", [0, 1, 4])
def test_empty_custom_sets_keep_their_shape(n_means):
    means = identity_emmeans(n=n_means)

    result = means.contrast(np.empty((0, n_means)), adjust="tukey")

    assert result.estimate.shape == (0,)
    assert result.confint().shape == (0, 6)

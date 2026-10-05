"""Independent arithmetic oracles for cross-validation scoring at extreme units."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from mixedlm.inference.cross_validation import (
    CrossValidationResult,
    _score_metrics,
    weighted_mae,
    weighted_mse,
    weighted_r2,
    weighted_rmse,
)

pytestmark = pytest.mark.installed_wheel

SCORERS = (weighted_mse, weighted_rmse, weighted_mae, weighted_r2)


def _reference_scores(observed, predicted, weights):
    # Decimal.from_float uses the actual binary inputs. Enough precision to
    # retain contributions across the full float64 response and weight range.
    with localcontext() as context:
        context.prec = 2200
        y = [Decimal.from_float(float(value)) for value in observed]
        p = [Decimal.from_float(float(value)) for value in predicted]
        w = [Decimal.from_float(float(value)) for value in weights]
        weight_sum = sum(w)
        squared_error = sum(weight * (a - b) ** 2 for a, b, weight in zip(y, p, w, strict=True))
        mse = squared_error / weight_sum
        mae = sum(weight * abs(a - b) for a, b, weight in zip(y, p, w, strict=True)) / weight_sum
        mean = sum(a * weight for a, weight in zip(y, w, strict=True)) / weight_sum
        variance = sum(weight * (a - mean) ** 2 for a, weight in zip(y, w, strict=True))
        r2 = Decimal(1) - squared_error / variance if variance else Decimal(y == p)
        return tuple(float(value) for value in (mse, mse.sqrt(), mae, r2))


@pytest.mark.parametrize(
    "observed,predicted,weights",
    [
        ([1.0, 2.0, 5.0, 8.0], [1.5, 1.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0]),
        ([0.0, 0.0], [1e200, 2e200], [1.0, 1.0]),
        ([0.0, 0.0], [1e-200, 2e-200], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, 2.0], [1e308, 1e308]),
        ([0.0, 0.0], [1e308, 1.0], [1e-308, 1e308]),
        ([-1e308, 1e308], [1e308, 1e308], [1e-308, 1e308]),
        ([-1e308, 1e308], [0.0, 1e308], [1e-308, 1e308]),
        ([1e16, 1e16 + 2, 1e16 + 4], [1e16 + 2, 1e16 + 4, 1e16 + 6], [1.0, 2.0, 3.0]),
        ([0.0, 2e-200, 4e-200], [1e100, 2e-200, 4e-200], [1e-308, 1e308, 1e308]),
        ([0.0, 1e-308, 1e308], [0.0, 0.0, 0.0], [1e308, 1e308, 1e-308]),
        ([0.0, 0.0], [np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)], [1.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0], [1e-308, 1e308]),
    ],
)
def test_scores_match_high_precision_arithmetic_without_runtime_warnings(
    observed, predicted, weights
):
    observed, predicted, weights = map(np.asarray, (observed, predicted, weights))
    expected = _reference_scores(observed, predicted, weights)

    with np.errstate(all="raise"):
        result = [scorer(observed, predicted, weights) for scorer in SCORERS]

    for actual, reference in zip(result, expected, strict=True):
        assert actual == pytest.approx(reference, rel=5e-15, abs=0.0)


@pytest.mark.parametrize("response_scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("weight_scale", [1e-300, 1.0, 1e300])
def test_error_scores_keep_response_units_and_ignore_common_weight_units(
    response_scale, weight_scale
):
    observed = np.array([0.0, 2.0, 4.0]) * response_scale
    predicted = np.array([1.0, 2.0, 6.0]) * response_scale
    weights = np.array([1.0, 2.0, 3.0]) * weight_scale

    assert weighted_rmse(observed, predicted, weights) == pytest.approx(
        np.sqrt(13.0 / 6.0) * response_scale, rel=1e-14, abs=0.0
    )
    assert weighted_mae(observed, predicted, weights) == pytest.approx(
        7.0 / 6.0 * response_scale, rel=1e-14, abs=0.0
    )


@pytest.mark.parametrize("scorer", SCORERS)
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("invalid", [np.array([1.0 + 1j, 2.0]), np.ma.array([1, 2], mask=[0, 1])])
def test_scores_reject_values_whose_information_would_be_silently_discarded(
    scorer, position, invalid
):
    arguments = [np.array([1.0, 2.0]), np.array([1.5, 2.5]), np.array([1.0, 1.0])]
    arguments[position] = invalid

    with pytest.raises(ValueError, match="unmasked real"):
        scorer(*arguments)


@pytest.mark.parametrize("scorer", SCORERS)
def test_score_inputs_are_not_mutated(scorer):
    observed = np.array([-1e308, 1e308])
    predicted = np.array([0.0, 1e308])
    weights = np.array([1e-308, 1e308])
    original = [array.copy() for array in (observed, predicted, weights)]
    for array in (observed, predicted, weights):
        array.setflags(write=False)

    scorer(observed, predicted, weights)

    for actual, expected in zip((observed, predicted, weights), original, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_mixed_exponent_scores_match_independent_arithmetic():
    rng = np.random.default_rng(7721)
    for _ in range(40):
        arrays = []
        for index in range(3):
            mantissa = rng.uniform(0.5, 1.0, size=4)
            if index < 2:
                mantissa *= rng.choice([-1.0, 1.0], size=4)
            arrays.append(np.ldexp(mantissa, rng.integers(-1070, 1024, size=4)))
        expected = _reference_scores(*arrays)

        with np.errstate(all="raise"):
            result = [scorer(*arrays) for scorer in SCORERS]

        for index, (actual, reference) in enumerate(zip(result, expected, strict=True)):
            tolerance = 3e-14 if index == 3 else 0.0
            assert actual == pytest.approx(reference, rel=3e-14, abs=tolerance)


def test_shared_cross_validation_scoring_preserves_finite_error_scores():
    observed = np.zeros(2)
    predicted = np.array([1e200, 2e200])
    weights = np.full(2, 1e308)

    with np.errstate(all="raise"):
        scores = _score_metrics(
            (("rmse", "rmse"), ("mae", "mae")), observed, predicted, weights, None
        )

    assert scores["rmse"] == pytest.approx(np.sqrt(2.5) * 1e200, rel=1e-14)
    assert scores["mae"] == pytest.approx(1.5e200, rel=1e-14)


@pytest.mark.parametrize(
    "values",
    [
        [1e308, 1e308],
        [1e200, 2e200, 3e200],
        [-1e308, -5e307],
        [-1e308, 1e308],
        [1e308, np.nextafter(1e308, 0.0), np.nextafter(np.nextafter(1e308, 0.0), 0.0)],
        [np.nextafter(0.0, 1.0), 2 * np.nextafter(0.0, 1.0), 3 * np.nextafter(0.0, 1.0)],
        [0.0, 0.0],
        [1e308],
    ],
)
def test_fold_summaries_match_high_precision_mean_and_sample_standard_deviation(values):
    with localcontext() as context:
        context.prec = 2200
        exact = [Decimal.from_float(float(value)) for value in values]
        mean = sum(exact) / len(exact)
        standard_deviation = (
            (sum((value - mean) ** 2 for value in exact) / (len(exact) - 1)).sqrt()
            if len(exact) > 1
            else Decimal(0)
        )
    result = CrossValidationResult(
        scores={"custom": float(mean)},
        fold_scores=pd.DataFrame({"custom": values}),
        predictions=np.empty(0),
        fold_ids=np.empty(0, dtype=np.int64),
        folds=(),
        group=None,
        metric_names=("custom",),
    )

    with np.errstate(all="raise"):
        summary = result.summary().iloc[0]

    assert summary.fold_mean == pytest.approx(float(mean), rel=3e-15, abs=0.0)
    assert summary.fold_std == pytest.approx(float(standard_deviation), rel=3e-15, abs=0.0)
    assert summary.fold_min == min(values)
    assert summary.fold_max == max(values)

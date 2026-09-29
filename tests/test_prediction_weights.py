from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.lmer import LmerResult
from scipy import linalg, stats


@pytest.fixture
def result() -> LmerResult:
    x = np.tile([-1.0, 0.0, 1.0, 2.0], 6)
    data = pd.DataFrame({"y": 0.5 + 0.7 * x, "x": x, "group": np.repeat(list("abcdef"), 4)})
    formula = parse_formula("y ~ x + (x | group)")
    matrices = build_model_matrices(formula, data)
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.8, -0.2, 0.5]),
        beta=np.array([0.5, 0.7]),
        sigma=0.6,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )


@pytest.fixture
def newdata(result) -> pd.DataFrame:
    data = result.matrices.frame.iloc[[8, 2, 5]].drop(columns="y").copy()
    return data.assign(precision=[0.5, 2.0, 4.0], shift=[0.1, 0.2, -0.3])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["scalar", "array", "list", "column"])
@pytest.mark.parametrize("re_form", [None, "NA"])
def test_new_prediction_weights_control_residual_variance(
    result, newdata, backend, kind, re_form
) -> None:
    expected_weights = newdata["precision"].to_numpy().copy()
    if kind == "scalar":
        weights = 2.0
        expected_weights[:] = 2.0
    elif kind == "array":
        weights = expected_weights.copy()
    elif kind == "list":
        weights = expected_weights.tolist()
    else:
        weights = "precision"
    if backend == "polars":
        pl = pytest.importorskip("polars")
        newdata = pl.DataFrame(newdata.to_dict(orient="list"))

    baseline = result.predict(newdata, re_form=re_form, interval="prediction", offset="shift")
    actual = result.predict(
        newdata, re_form=re_form, interval="prediction", offset="shift", weights=weights
    )
    critical = stats.norm.ppf(0.975)
    baseline_variance = ((baseline.upper - baseline.lower) / (2 * critical)) ** 2
    actual_variance = ((actual.upper - actual.lower) / (2 * critical)) ** 2

    np.testing.assert_allclose(actual.fit, baseline.fit, rtol=0, atol=0)
    np.testing.assert_allclose(actual.se_fit, baseline.se_fit, rtol=0, atol=0)
    np.testing.assert_allclose(
        actual_variance - baseline_variance,
        result.sigma**2 * (1.0 / expected_weights - 1.0),
        rtol=1e-12,
        atol=1e-12,
    )
    if kind == "array":
        np.testing.assert_array_equal(weights, expected_weights)


def test_weighted_fixed_model_intervals_match_direct_gls(result, newdata) -> None:
    data = result.matrices.frame
    formula = parse_formula("y ~ x")
    weights = np.geomspace(0.2, 4.0, len(data))
    matrices = build_model_matrices(formula, data, weights=weights)
    result = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.empty(0),
        beta=result.beta,
        sigma=result.sigma,
        u=np.empty(0),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )
    X = matrices.X
    covariance = result.sigma**2 * linalg.inv(X.T @ (weights[:, None] * X))
    design = np.column_stack([np.ones(len(newdata)), newdata["x"]])
    mean_variance = np.diag(design @ covariance @ design.T)
    expected_se = np.sqrt(mean_variance + result.sigma**2 / newdata["precision"].to_numpy())
    expected_mean = design @ result.beta
    actual = result.predict(newdata, interval="prediction", weights="precision", level=0.9)
    critical = stats.norm.ppf(0.95)

    np.testing.assert_allclose(actual.se_fit**2, mean_variance, rtol=1e-12)
    np.testing.assert_allclose(actual.lower, expected_mean - critical * expected_se, rtol=1e-12)
    np.testing.assert_allclose(actual.upper, expected_mean + critical * expected_se, rtol=1e-12)


def test_prediction_weight_default_is_one(result, newdata) -> None:
    default = result.predict(newdata, interval="prediction")
    explicit = result.predict(newdata, interval="prediction", weights=1.0)

    np.testing.assert_array_equal(explicit.lower, default.lower)
    np.testing.assert_array_equal(explicit.upper, default.upper)


def test_newdata_with_fitted_weights_matches_in_sample_intervals(result) -> None:
    weights = np.geomspace(0.2, 4.0, result.matrices.n_obs)
    result = replace(result, matrices=replace(result.matrices, weights=weights))
    original = result.predict(interval="prediction")
    new = result.predict(result.matrices.frame, interval="prediction", weights=weights)

    np.testing.assert_allclose(new.fit, original.fit, rtol=0, atol=1e-12)
    np.testing.assert_allclose(new.se_fit, original.se_fit, rtol=0, atol=1e-12)
    np.testing.assert_allclose(new.lower, original.lower, rtol=0, atol=1e-12)
    np.testing.assert_allclose(new.upper, original.upper, rtol=0, atol=1e-12)


@pytest.mark.parametrize("new_groups", [False, True])
def test_equivalent_weight_scaling_preserves_newdata_intervals(result, newdata, new_groups) -> None:
    scale = 25.0
    scaled = replace(
        result,
        matrices=replace(result.matrices, weights=result.matrices.weights * scale),
        theta=result.theta / np.sqrt(scale),
        sigma=result.sigma * np.sqrt(scale),
    )
    if new_groups:
        newdata = newdata.assign(group=["new-a", "new-b", "new-c"])
    baseline = result.predict(
        newdata, allow_new_levels=True, interval="prediction", weights="precision"
    )
    actual = scaled.predict(
        newdata,
        allow_new_levels=True,
        interval="prediction",
        weights=newdata["precision"].to_numpy() * scale,
    )

    np.testing.assert_allclose(actual.fit, baseline.fit, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.se_fit, baseline.se_fit, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.lower, baseline.lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.upper, baseline.upper, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "weights,message",
    [
        (0.0, "strictly positive"),
        (-1.0, "strictly positive"),
        ([1.0, 0.0, 2.0], "strictly positive"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        ([1.0, np.nan, 2.0], "finite"),
        ([1.0, 2.0], "length 2; expected 3"),
        (np.ones((3, 1)), "scalar or one-dimensional"),
        (["bad", "values", "here"], "numeric"),
        ("missing", "missing weights column"),
    ],
)
def test_prediction_rejects_invalid_weights(result, newdata, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        result.predict(newdata, interval="prediction", weights=weights)


@pytest.mark.parametrize("interval", ["none", "confidence"])
def test_prediction_weights_require_prediction_intervals(result, newdata, interval) -> None:
    with pytest.raises(ValueError, match="require interval='prediction'"):
        result.predict(newdata, interval=interval, weights=2.0)


def test_prediction_weights_require_newdata(result) -> None:
    with pytest.raises(ValueError, match="only be supplied with newdata"):
        result.predict(interval="prediction", weights=2.0)

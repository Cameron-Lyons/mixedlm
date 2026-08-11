from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import set_cov_type
from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.lmer import LmerResult
from mixedlm.models.lmer_types import PredictResult
from numpy.typing import NDArray
from scipy import linalg, stats


def _prediction_data() -> pd.DataFrame:
    groups = np.repeat([f"g{i}" for i in range(6)], 4)
    x = np.tile([-1.0, -0.25, 0.5, 1.25], 6)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.5 * x,
            "x": x,
            "group": groups,
        }
    )


def _make_result(
    formula: Formula,
    data: pd.DataFrame,
    theta: NDArray[np.floating],
    *,
    sigma: float = 0.8,
) -> LmerResult:
    matrices = build_model_matrices(formula, data)
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.asarray(theta, dtype=np.float64),
        beta=np.linspace(0.4, 1.0, matrices.n_fixed),
        sigma=sigma,
        u=np.zeros(matrices.n_random, dtype=np.float64),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )


@pytest.fixture
def slope_result() -> LmerResult:
    formula = parse_formula("y ~ x + (x | group)")
    return _make_result(formula, _prediction_data(), np.array([1.1, 0.35, 0.7]))


def _joint_covariance(result: LmerResult) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    matrices = result.matrices
    lambda_matrix = _build_lambda(result.theta, matrices.random_structures).toarray()
    transformed_Z = matrices.Z @ lambda_matrix
    precision = np.block(
        [
            [matrices.X.T @ matrices.X, matrices.X.T @ transformed_Z],
            [
                transformed_Z.T @ matrices.X,
                transformed_Z.T @ transformed_Z + np.eye(matrices.n_random),
            ],
        ]
    )
    covariance = result.sigma**2 * linalg.inv(precision)
    return covariance, lambda_matrix


def _known_random_design(result: LmerResult, newdata: pd.DataFrame) -> NDArray[np.floating]:
    structure = result.matrices.random_structures[0]
    design = np.zeros((len(newdata), result.matrices.n_random), dtype=np.float64)
    for row, (group, x) in enumerate(zip(newdata["group"], newdata["x"], strict=True)):
        level = structure.level_map[group]
        start = level * structure.n_terms
        design[row, start] = 1.0
        design[row, start + 1] = x
    return design


def _assert_prediction_result(value: object) -> PredictResult:
    assert isinstance(value, PredictResult)
    assert value.se_fit is not None
    return value


def test_in_sample_uncertainty_matches_joint_mixed_model_covariance(
    slope_result: LmerResult,
) -> None:
    covariance, lambda_matrix = _joint_covariance(slope_result)
    design = np.hstack([slope_result.matrices.X, slope_result.matrices.Z @ lambda_matrix])
    expected = np.sum((design @ covariance) * design, axis=1)

    predicted = _assert_prediction_result(slope_result.predict(se_fit=True))

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_known_level_uncertainty_includes_full_fixed_random_covariance(
    slope_result: LmerResult,
) -> None:
    covariance, lambda_matrix = _joint_covariance(slope_result)
    newdata = slope_result.matrices.frame.iloc[[0, 5, 10]].copy()
    newdata["x"] = [0.7, -0.8, 1.4]
    new_matrices = build_model_matrices(slope_result.formula, newdata)
    random_design = _known_random_design(slope_result, newdata)
    design = np.hstack([new_matrices.X, random_design @ lambda_matrix])
    expected = np.sum((design @ covariance) * design, axis=1)

    predicted = _assert_prediction_result(slope_result.predict(newdata=newdata, se_fit=True))

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_new_level_uncertainty_uses_full_prior_covariance(
    slope_result: LmerResult,
) -> None:
    covariance, lambda_matrix = _joint_covariance(slope_result)
    p = slope_result.matrices.n_fixed
    newdata = pd.DataFrame(
        {
            "y": [0.0, 0.0, 0.0],
            "x": [0.8, -0.9, 1.5],
            "group": ["new-a", "new-b", "new-c"],
        }
    )
    new_matrices = build_model_matrices(slope_result.formula, newdata)
    random_terms = np.column_stack([np.ones(len(newdata)), newdata["x"]])
    prior_factor = slope_result.sigma * lambda_matrix[:2, :2]
    prior_covariance = prior_factor @ prior_factor.T
    expected = np.sum(
        (new_matrices.X @ covariance[:p, :p]) * new_matrices.X,
        axis=1,
    )
    expected += np.sum((random_terms @ prior_covariance) * random_terms, axis=1)

    predicted = _assert_prediction_result(
        slope_result.predict(newdata=newdata, allow_new_levels=True, se_fit=True)
    )

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_fixed_only_uncertainty_excludes_random_effect_terms(
    slope_result: LmerResult,
) -> None:
    covariance, _lambda_matrix = _joint_covariance(slope_result)
    p = slope_result.matrices.n_fixed
    newdata = slope_result.matrices.frame.iloc[[2, 9, 16]].copy()
    new_matrices = build_model_matrices(slope_result.formula, newdata)
    expected = np.sum(
        (new_matrices.X @ covariance[:p, :p]) * new_matrices.X,
        axis=1,
    )

    predicted = _assert_prediction_result(
        slope_result.predict(newdata=newdata, re_form="NA", se_fit=True)
    )

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_confidence_and_prediction_intervals_use_correct_variance(
    slope_result: LmerResult,
) -> None:
    newdata = slope_result.matrices.frame.iloc[[1, 6, 11]].copy()
    confidence = _assert_prediction_result(
        slope_result.predict(newdata=newdata, interval="confidence", level=0.9)
    )
    prediction = _assert_prediction_result(
        slope_result.predict(newdata=newdata, interval="prediction", level=0.9)
    )
    assert confidence.lower is not None and confidence.upper is not None
    assert prediction.lower is not None and prediction.upper is not None
    critical = stats.norm.ppf(0.95)
    confidence_variance = ((confidence.upper - confidence.lower) / (2 * critical)) ** 2
    prediction_variance = ((prediction.upper - prediction.lower) / (2 * critical)) ** 2

    np.testing.assert_allclose(confidence_variance, confidence.se_fit**2, rtol=1e-12)
    np.testing.assert_allclose(
        prediction_variance,
        prediction.se_fit**2 + slope_result.sigma**2,
        rtol=1e-12,
    )


def test_random_interaction_uncertainty_uses_encoded_design() -> None:
    data = _prediction_data()
    data["z"] = np.tile([0.5, -1.0, 1.5, 0.25], 6)
    formula = parse_formula("y ~ x * z + (0 + x:z | group)")
    result = _make_result(formula, data, np.array([0.9]))
    covariance, lambda_matrix = _joint_covariance(result)
    newdata = data.iloc[[0, 7, 14]].copy()
    new_matrices = build_model_matrices(formula, newdata)
    random_design = np.zeros((len(newdata), result.matrices.n_random))
    structure = result.matrices.random_structures[0]
    for row, values in enumerate(newdata.itertuples(index=False)):
        level = structure.level_map[values.group]
        random_design[row, level] = values.x * values.z
    design = np.hstack([new_matrices.X, random_design @ lambda_matrix])
    expected = np.sum((design @ covariance) * design, axis=1)

    predicted = _assert_prediction_result(result.predict(newdata=newdata, se_fit=True))

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_crossed_structure_uncertainty_includes_conditional_cross_covariance() -> None:
    g = np.repeat([f"g{i}" for i in range(4)], 5)
    h = np.tile([f"h{i}" for i in range(5)], 4)
    x = np.linspace(-1.0, 1.0, len(g))
    data = pd.DataFrame({"y": 1.0 + x, "x": x, "g": g, "h": h})
    formula = parse_formula("y ~ x + (1 | g) + (1 | h)")
    result = _make_result(formula, data, np.array([0.9, 0.6]))
    covariance, lambda_matrix = _joint_covariance(result)
    design = np.hstack([result.matrices.X, result.matrices.Z @ lambda_matrix])
    expected = np.sum((design @ covariance) * design, axis=1)

    predicted = _assert_prediction_result(result.predict(se_fit=True))

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_structured_new_level_uncertainty_uses_structured_covariance() -> None:
    formula = set_cov_type("y ~ x + (x | group)", "ar1")
    result = _make_result(formula, _prediction_data(), np.array([1.2, 0.4]))
    covariance, lambda_matrix = _joint_covariance(result)
    p = result.matrices.n_fixed
    newdata = pd.DataFrame({"y": [0.0], "x": [1.3], "group": ["new"]})
    new_matrices = build_model_matrices(formula, newdata)
    random_terms = np.array([[1.0, 1.3]])
    prior_factor = result.sigma * lambda_matrix[:2, :2]
    expected = np.sum((new_matrices.X @ covariance[:p, :p]) * new_matrices.X, axis=1)
    expected += np.sum(
        (random_terms @ prior_factor @ prior_factor.T) * random_terms,
        axis=1,
    )

    predicted = _assert_prediction_result(
        result.predict(newdata=newdata, allow_new_levels=True, se_fit=True)
    )

    np.testing.assert_allclose(predicted.se_fit**2, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_prediction_rejects_invalid_interval_level(
    slope_result: LmerResult,
    level: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        slope_result.predict(interval="confidence", level=level)


def test_prediction_rejects_unknown_interval_before_returning_fitted_values(
    slope_result: LmerResult,
) -> None:
    with pytest.raises(ValueError, match="Unknown interval type"):
        slope_result.predict(interval="simultaneous")

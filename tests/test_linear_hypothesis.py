from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, lmer
from mixedlm.inference import LinearHypothesisResult, linear_hypothesis
from numpy.testing import assert_allclose
from scipy import stats


@pytest.fixture(scope="module")
def lmm_result():
    rng = np.random.default_rng(20260803)
    n_groups = 16
    observations_per_group = 12
    group_index = np.repeat(np.arange(n_groups), observations_per_group)
    x = rng.normal(size=len(group_index))
    z = rng.normal(size=len(group_index))
    random_intercept = rng.normal(scale=2.0, size=n_groups)
    y = (
        1.5
        + 0.8 * x
        - 0.35 * z
        + random_intercept[group_index]
        + rng.normal(scale=0.3, size=len(group_index))
    )
    data = pd.DataFrame({"y": y, "x": x, "z": z, "group": [f"G{value}" for value in group_index]})
    return lmer("y ~ x + z + (1 | group)", data)


@pytest.fixture(scope="module")
def glmm_result():
    rng = np.random.default_rng(20260804)
    n_groups = 20
    observations_per_group = 12
    group_index = np.repeat(np.arange(n_groups), observations_per_group)
    x = rng.normal(size=len(group_index))
    random_intercept = rng.normal(scale=0.55, size=n_groups)
    eta = -0.4 + 0.75 * x + random_intercept[group_index]
    probability = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, probability)
    data = pd.DataFrame({"y": y, "x": x, "group": [f"G{value}" for value in group_index]})
    return glmer("y ~ x + (1 | group)", data, family=families.Binomial())


def _coefficient_index(model, name: str) -> int:
    return model.matrices.fixed_names.index(name)


def test_single_named_hypothesis_matches_manual_f_test(lmm_result) -> None:
    result = linear_hypothesis(lmm_result, {"x": 1.0})
    index = _coefficient_index(lmm_result, "x")
    estimate = lmm_result.beta[index]
    std_error = np.sqrt(lmm_result.vcov()[index, index])
    t_value = estimate / std_error
    denominator_df = lmm_result.df_residual()

    assert isinstance(result, LinearHypothesisResult)
    assert result.test == "F"
    assert result.numerator_df == 1
    assert result.denominator_df == denominator_df
    assert_allclose(result.estimate, [estimate])
    assert_allclose(result.std_error, [std_error])
    assert_allclose(result.row_statistic, [t_value])
    assert_allclose(result.statistic, t_value**2)
    assert_allclose(result.p_value, stats.f.sf(t_value**2, 1, denominator_df))


def test_nonzero_rhs_tests_the_requested_value(lmm_result) -> None:
    rhs = 1.25
    result = linear_hypothesis(lmm_result, {"x": 1, "z": -1}, rhs=rhs)
    indices = [_coefficient_index(lmm_result, name) for name in ("x", "z")]
    expected = lmm_result.beta[indices[0]] - lmm_result.beta[indices[1]]

    assert result.labels == ("x - z",)
    assert_allclose(result.estimate, [expected])
    assert_allclose(result.rhs, [rhs])
    assert_allclose(result.difference, [expected - rhs])
    assert result.conf_low[0] < result.estimate[0] < result.conf_high[0]


def test_joint_matrix_hypothesis_matches_manual_wald_test(lmm_result) -> None:
    constraints = np.array([[0.0, 1.0, -1.0], [0.0, 1.0, 1.0]])
    result = linear_hypothesis(
        lmm_result,
        constraints,
        rhs=[0.0, 0.5],
        labels=["equal slopes", "slope sum"],
    )

    covariance = constraints @ lmm_result.vcov() @ constraints.T
    difference = constraints @ lmm_result.beta - np.array([0.0, 0.5])
    wald = float(difference @ np.linalg.solve(covariance, difference))
    expected_f = wald / 2

    assert result.labels == ("equal slopes", "slope sum")
    assert result.numerator_df == 2
    assert_allclose(result.covariance, covariance)
    assert_allclose(result.statistic, expected_f)
    assert_allclose(
        result.p_value,
        stats.f.sf(expected_f, 2, lmm_result.df_residual()),
    )


def test_nested_mapping_preserves_labels(lmm_result) -> None:
    result = linear_hypothesis(
        lmm_result,
        {
            "equal slopes": {"x": 1, "z": -1},
            "target sum": {"x": 1, "z": 1},
        },
        rhs=[0, 0.5],
    )

    assert result.labels == ("equal slopes", "target sum")
    assert result.constraints.shape == (2, 3)


def test_sequence_of_named_rows_generates_readable_labels(lmm_result) -> None:
    result = linear_hypothesis(
        lmm_result,
        [{"x": 1, "z": -1}, {"x": 2, "z": 1}],
    )

    assert result.labels == ("x - z", "2*x + z")


def test_dataframe_constraints_use_columns_and_index(lmm_result) -> None:
    constraints = pd.DataFrame(
        {"x": [1.0, 1.0], "z": [-1.0, 1.0]},
        index=["difference", "sum"],
    )
    result = linear_hypothesis(lmm_result, constraints)

    assert result.labels == ("difference", "sum")
    assert_allclose(result.constraints[:, 0], 0)
    assert_allclose(result.constraints[:, 1:], constraints.to_numpy())


def test_glmm_defaults_to_chi_square(glmm_result) -> None:
    result = linear_hypothesis(glmm_result, {"x": 1})
    index = _coefficient_index(glmm_result, "x")
    z_value = glmm_result.beta[index] / np.sqrt(glmm_result.vcov()[index, index])

    assert result.test == "Chisq"
    assert result.denominator_df is None
    assert_allclose(result.statistic, z_value**2)
    assert_allclose(result.p_value, stats.chi2.sf(z_value**2, 1))
    assert "z_value" in result.table


def test_test_type_can_be_overridden(lmm_result) -> None:
    chi_square = linear_hypothesis(lmm_result, {"x": 1}, test="chi-square")
    custom_f = linear_hypothesis(lmm_result, {"x": 1}, test="F", denominator_df=12.5)

    assert chi_square.test == "Chisq"
    assert chi_square.denominator_df is None
    assert custom_f.test == "F"
    assert custom_f.denominator_df == 12.5
    assert custom_f.p_value != chi_square.p_value


def test_result_table_summary_repr_and_readonly_arrays(lmm_result) -> None:
    result = linear_hypothesis(lmm_result, {"x": 1})
    table = result.table

    assert list(table.columns) == [
        "hypothesis",
        "estimate",
        "null_value",
        "difference",
        "std_error",
        "t_value",
        "p_value",
        "conf_low",
        "conf_high",
    ]
    assert "Joint test: F" in result.summary()
    assert "Confidence level: 95.0%" in str(result)
    assert "LinearHypothesisResult" in repr(result)
    with pytest.raises(ValueError):
        result.estimate[0] = 0


def test_inference_namespace_exports_public_api() -> None:
    from mixedlm import inference

    assert inference.linear_hypothesis is linear_hypothesis
    assert inference.LinearHypothesisResult is LinearHypothesisResult
    assert "linear_hypothesis" in inference.__all__


@pytest.mark.parametrize(
    ("hypothesis", "match"),
    [
        (np.array([1.0, 0.0]), "columns"),
        (np.zeros(3), "all-zero"),
        (np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]), "linearly independent"),
        (np.array([0.0, np.nan, 0.0]), "finite"),
        ({"missing": 1.0}, "Unknown coefficient"),
        ({1: 1.0}, "coefficient names must be strings"),
        ({}, "must not be empty"),
        ({"x": 1.0, "nested": {"z": 1.0}}, "cannot mix"),
    ],
)
def test_invalid_hypotheses_raise(lmm_result, hypothesis, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        linear_hypothesis(lmm_result, hypothesis)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rhs": [0.0, 1.0]}, "rhs has length"),
        ({"labels": ["one", "two"]}, "labels has length"),
        ({"labels": "one"}, "sequence of strings"),
        ({"level": 1.0}, "strictly between"),
        ({"level": "bad"}, "level must be numeric"),
        ({"test": "invalid"}, "test must be"),
        ({"test": None}, "test must be a string"),
        ({"test": "F", "denominator_df": 0}, "positive finite"),
        ({"test": "F", "denominator_df": "bad"}, "denominator_df must be numeric"),
        ({"test": "Chisq", "denominator_df": 10}, "only valid"),
    ],
)
def test_invalid_options_raise(lmm_result, kwargs, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        linear_hypothesis(lmm_result, {"x": 1}, **kwargs)


def test_duplicate_labels_raise(lmm_result) -> None:
    with pytest.raises(ValueError, match="unique"):
        linear_hypothesis(
            lmm_result,
            [{"x": 1}, {"z": 1}],
            labels=["duplicate", "duplicate"],
        )


def test_invalid_dataframe_constraints_raise(lmm_result) -> None:
    with pytest.raises(ValueError, match="Unknown coefficient"):
        linear_hypothesis(lmm_result, pd.DataFrame({"missing": [1.0]}))
    with pytest.raises(TypeError, match="must be numeric"):
        linear_hypothesis(lmm_result, pd.DataFrame({"x": ["not numeric"]}))
    with pytest.raises(TypeError, match="column names must be strings"):
        linear_hypothesis(lmm_result, pd.DataFrame({1: [1.0]}))


def test_invalid_model_raises() -> None:
    with pytest.raises(TypeError, match="fitted LmerResult or GlmerResult"):
        linear_hypothesis(object(), np.array([1.0]))  # type: ignore[arg-type]

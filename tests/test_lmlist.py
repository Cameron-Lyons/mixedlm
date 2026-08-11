from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmList


@pytest.fixture
def grouped_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_groups = 4
    rows_per_group = 18
    group_index = np.repeat(np.arange(n_groups), rows_per_group)
    x1 = rng.normal(size=n_groups * rows_per_group)
    x2 = rng.normal(size=n_groups * rows_per_group)
    category = np.tile(["low", "mid", "high"], n_groups * rows_per_group // 3)
    intercept = np.array([1.0, 2.0, -1.0, 0.5])[group_index]
    y_numeric = intercept + 2.0 * x1 - 0.5 * x2
    y = (
        y_numeric
        + 1.25 * (category == "mid")
        - 0.75 * (category == "high")
        + 0.4 * x1 * (category == "high")
    )
    return pd.DataFrame(
        {
            "y": y,
            "y_numeric": y_numeric,
            "x1": x1,
            "x2": x2,
            "category": category,
            "group": [f"G{index}" for index in group_index],
        }
    )


def test_multiple_numeric_predictors(grouped_data: pd.DataFrame) -> None:
    result = lmList("y_numeric ~ x1 + x2", grouped_data, group="group")

    assert len(result["fits"]) == 4
    assert list(result["coef"].columns) == ["(Intercept)", "x1", "x2"]
    np.testing.assert_allclose(result["coef"]["x1"], 2.0, atol=0.2)
    np.testing.assert_allclose(result["coef"]["x2"], -0.5, atol=0.2)


def test_categorical_interaction_formula(grouped_data: pd.DataFrame) -> None:
    result = lmList("y ~ x1 * category", grouped_data, group="group", pool=False)

    expected_names = [
        "(Intercept)",
        "x1",
        "category.1",
        "category.2",
        "x1:category.1",
        "x1:category.2",
    ]
    assert list(result["coef"].columns) == expected_names
    assert "pooled" not in result
    assert np.isfinite(result["coef"].to_numpy()).all()


def test_grouped_formula_syntax_matches_explicit_group(grouped_data: pd.DataFrame) -> None:
    explicit = lmList("y_numeric ~ x1 + x2", grouped_data, group="group")
    grouped = lmList("y_numeric ~ x1 + x2 | group", grouped_data)

    pd.testing.assert_frame_equal(grouped["coef"], explicit["coef"])


def test_random_effect_formula_infers_group(grouped_data: pd.DataFrame) -> None:
    result = lmList("y_numeric ~ x1 + x2 + (1 | group)", grouped_data)

    assert len(result["fits"]) == 4
    assert list(result["coef"].columns) == ["(Intercept)", "x1", "x2"]


def test_mismatched_formula_and_argument_groups_raise(grouped_data: pd.DataFrame) -> None:
    grouped_data["other_group"] = grouped_data["group"]

    with pytest.raises(ValueError, match="does not match"):
        lmList("y_numeric ~ x1 | group", grouped_data, group="other_group")


def test_fit_metadata_and_pooled_result(grouped_data: pd.DataFrame) -> None:
    result = lmList("y_numeric ~ x1 + x2", grouped_data, group="group")
    fit = result["fits"]["G0"]
    pooled = result["pooled"]

    assert fit["coef_names"] == ["(Intercept)", "x1", "x2"]
    assert list(fit["params"]) == fit["coef_names"]
    assert fit["rank"] == 3
    assert fit["n"] == 18
    assert len(fit["fitted"]) == 18
    assert len(fit["residuals"]) == 18
    assert pooled["n"] == len(grouped_data)


def test_missing_formula_values_keep_group_rows_aligned(grouped_data: pd.DataFrame) -> None:
    grouped_data.loc[0, "x1"] = np.nan

    result = lmList("y_numeric ~ x1 + x2", grouped_data, group="group")

    assert result["fits"]["G0"]["n"] == 17
    assert result["fits"]["G1"]["n"] == 18
    assert result["pooled"]["n"] == len(grouped_data) - 1

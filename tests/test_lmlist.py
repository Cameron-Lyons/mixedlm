from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmList, load_sleepstudy


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


@pytest.mark.parametrize("pool", [False, True])
def test_sleepstudy_fits_match_per_subject_least_squares(pool: bool) -> None:
    data = load_sleepstudy()
    result = lmList("Reaction ~ Days", data, group="Subject", pool=pool)
    expected = {
        subject: np.polyfit(rows["Days"], rows["Reaction"], 1)[::-1]
        for subject, rows in data.groupby("Subject")
    }

    assert set(result["fits"]) == set(expected)
    for subject, coefficients in expected.items():
        np.testing.assert_allclose(result["fits"][subject]["coef"], coefficients, rtol=1e-10)
        np.testing.assert_allclose(
            result["coef"].loc[subject, ["(Intercept)", "Days"]], coefficients, rtol=1e-10
        )
    if pool:
        overall = np.polyfit(data["Days"], data["Reaction"], 1)[::-1]
        np.testing.assert_allclose(result["pooled"]["coef"], overall, rtol=1e-10)
    else:
        assert "pooled" not in result


def test_lmlist_learns_stateful_transforms_within_each_group():
    rng = np.random.default_rng(5)
    data = pd.DataFrame(
        {"g": np.repeat(["a", "b"], 20), "x": np.r_[rng.normal(0, 1, 20), rng.normal(5, 3, 20)]}
    )
    data["y"] = 1 + 2 * data["x"] + rng.normal(scale=0.5, size=40)
    # lme4 2.1.0: coef(lmList(y ~ scale(x) | g, d)) and coef(lmList(y ~ poly(x, 2) | g, d))
    np.testing.assert_allclose(
        lmList("y ~ scale(x) | g", data)["coef"].to_numpy(),
        [[0.6125, 1.8189], [8.2817, 4.2442]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        lmList("y ~ poly(x, 2) | g", data)["coef"].to_numpy(),
        [[0.6125, 7.9283, -0.7084], [8.2817, 18.5002, 0.1767]],
        atol=1e-4,
    )


def test_lmlist_reports_nan_for_coefficients_a_group_cannot_estimate():
    rng = np.random.default_rng(2)
    data = pd.DataFrame(
        {
            "g": np.repeat(["a", "b"], 12),
            "k": ["p", "q", "r"] * 4 + ["p", "q"] * 6,
            "x": rng.normal(size=24),
        }
    )
    data["y"] = 1 + data["x"] + (data["k"] == "r") * 3 + rng.normal(scale=0.2, size=24)
    result = lmList("y ~ x + k | g", data)
    # lme4 2.1.0: coef(lmList(y ~ x + k | g, d))["b", ] is 0.8875 1.0766 0.2507 NA
    np.testing.assert_allclose(
        result["coef"].loc["b"].iloc[:3], [0.8875, 1.0766, 0.2507], atol=1e-4
    )
    assert np.isnan(result["coef"].loc["b"].iloc[3])
    assert [fit["rank"] for fit in result["fits"].values()] == [4, 3]


def test_lmlist_codes_every_group_with_the_levels_of_all_the_data():
    from mixedlm import load_grouseticks

    # As lme4 does, the 42 locations observed in a single year cannot be fitted.
    with pytest.warns(UserWarning, match="fitting failed for 42 group"):
        coef = lmList("TICKS ~ YEAR | LOCATION", load_grouseticks())["coef"]
    # lme4 2.1.0: coef(lmList(TICKS ~ YEAR | LOCATION, g))[c("11", "22", "1"), ] with
    # columns YEAR96 (YEAR.1) and YEAR97 (YEAR.2). Location 22 has no 1996 rows, and
    # location 1 has no 1995 rows, so lm() codes it relative to 1996.
    assert list(coef.columns) == ["(Intercept)", "YEAR.1", "YEAR.2"]
    np.testing.assert_allclose(
        coef.loc[["11", "22", "1"]].to_numpy(),
        [[9.75, 14.25, -6.75], [4.0, np.nan, -2.0], [17.6, np.nan, -16.6]],
        atol=1e-10,
    )


def test_lmlist_rereferences_interactions_as_lm_does():
    data = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 7, 6, 8.0],
            "x": [1, 2, 3, 4, 1, 2, 3, 5.0],
            "k": ["a", "b", "c", "a", "b", "c", "b", "c"],
            "g": ["A"] * 4 + ["B"] * 4,
        }
    )
    coef = lmList("y ~ x * k | g", data)["coef"]
    # lme4 2.1.0: coef(lmList(y ~ x * k | g, d)), columns kb kc x:kb x:kc.
    np.testing.assert_allclose(
        coef.to_numpy(),
        [[0, 1, 0, 0, np.nan, np.nan], [4.5, 0.5, np.nan, 11 / 6, np.nan, -1 / 6]],
        atol=1e-10,
    )
    data["kk"] = data["k"].map({"a": 1, "b": 2, "c": 3})
    by_number = lmList("y ~ x + factor(kk) | g", data)["coef"]
    by_label = lmList("y ~ x + k | g", data)["coef"]
    np.testing.assert_allclose(by_number.to_numpy(), by_label.to_numpy(), atol=1e-10)


def test_lmlist_accepts_polars_data():
    pl = pytest.importorskip("polars")

    data = load_sleepstudy()
    expected = lmList("Reaction ~ Days | Subject", data)["coef"]
    for result in (
        lmList("Reaction ~ Days | Subject", pl.from_pandas(data)),
        lmList("Reaction ~ Days", pl.from_pandas(data), group="Subject"),
    ):
        np.testing.assert_allclose(result["coef"].to_numpy(), expected.to_numpy())


@pytest.mark.parametrize(
    "formula", ["y ~ k | g", "y ~ C(k) | g", "y ~ C(k, treatment) | g", "y ~ factor(k) | g"]
)
def test_lmlist_codes_a_group_without_the_reference_level_as_lm_does(formula):
    data = pd.DataFrame(
        {
            "g": ["A"] * 6 + ["B"] * 6,
            "k": ["a", "b", "c"] * 2 + ["b", "c"] * 3,
            "y": [1.0, 3.1, 0.2, 1.2, 2.9, 0.0, 5.0, 1.0, 5.2, 1.1, 4.9, 0.8],
        }
    )
    # lme4 2.1.0: coef(lmList(y ~ C(factor(k), treatment) | g, d)).
    np.testing.assert_allclose(
        lmList(formula, data)["coef"].to_numpy(),
        [[1.1, 1.9, -1.0], [5.0333, np.nan, -4.0667]],
        atol=1e-4,
    )


def test_lmlist_numbers_other_contrasts_by_position_as_r_does():
    data = pd.DataFrame(
        {
            "g": ["A"] * 6 + ["B"] * 6,
            "k": ["a", "b", "c"] * 2 + ["b", "c"] * 3,
            "y": [1.0, 3.1, 0.2, 1.2, 2.9, 0.0, 5.0, 1.0, 5.2, 1.1, 4.9, 0.8],
        }
    )
    # lme4 2.1.0: coef(lmList(y ~ C(k, sum) | g, d)), with k a character column.
    np.testing.assert_allclose(
        lmList("y ~ C(k, sum) | g", data)["coef"].to_numpy(),
        [[1.4, -0.3, 1.6], [3.0, 2.0333, np.nan]],
        atol=1e-4,
    )


def test_lmlist_keeps_level_columns_apart_from_confounded_covariates():
    data = pd.DataFrame(
        {
            "g": ["A"] * 6 + ["B"] * 6,
            "k": ["a", "b", "c", "a", "b", "c", "b", "c", "b", "c", "b", "c"],
            "z": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0.0],
            "y": [1.0, 3.2, 0.1, 1.4, 2.9, 0.3, 5.0, 1.0, 5.4, 1.2, 4.8, 0.9],
        }
    )
    # lme4 2.1.0: in group B, z equals the indicator of b, the group's reference.
    np.testing.assert_allclose(
        lmList("y ~ k + z | g", data)["coef"].to_numpy(),
        [[1.05, 1.85, -1.0, 0.3], [5.0667, np.nan, -4.0333, np.nan]],
        atol=1e-4,
    )


def test_lmlist_reports_nan_for_groups_that_cannot_be_fitted():
    data = pd.DataFrame(
        {
            "g": ["A"] * 5 + ["B"] * 5 + ["C"] * 4,
            "x": [1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 1, 2, 2.0],
            "y": [1.0, 2.1, 2.9, 4.2, 5.1, 0.5, 1.9, 3.2, 3.8, 6.4, 1.1, 0.9, 2.2, 1.8],
        }
    )
    # lme4 2.1.0 returns an NA row and a warning for group C.
    with pytest.warns(UserWarning, match="fitting failed for 1 group"):
        coef = lmList("y ~ poly(x, 2) | g", data)["coef"]
    np.testing.assert_allclose(
        coef.to_numpy(),
        [[3.06, 3.2571, 0.0267], [3.16, 4.4033, -0.0683], [np.nan] * 3],
        atol=1e-4,
    )


def test_lmlist_names_interactions_of_other_contrasts_with_shared_levels():
    data = pd.DataFrame(
        {
            "g": ["A"] * 8 + ["B"] * 8,
            "k": ["u", "v"] * 8,
            "cat": ["p", "p", "q", "q"] * 2 + ["q", "q", "r", "r"] * 2,
            "y": [1.0, 2.0, 4.0, 3.5, 1.5, 2.0, 4.5, 4.0, 3.0, 4.0, 5.0, 7.5, 3.5, 4.0, 5.5, 7.0],
        }
    )
    coef = lmList("y ~ C(k, sum)*cat | g", data)["coef"]
    # Group B lacks p, so its q and r columns are cat.1 and cat.2, as for group A.
    b = coef.loc["B"]
    assert np.isnan(b["cat.1"]) == np.isnan(b["C(k, sum).1:cat.1"])
    assert np.isfinite(b["cat.2"]) and np.isfinite(b["C(k, sum).1:cat.2"])


def test_lmlist_raises_for_ambiguous_names_and_fits_small_groups():
    data = pd.DataFrame(
        {
            "g": np.repeat(["1", "2", "3", "4"], [3, 1, 2, 4]),
            "x": [0.0, 1.0, 2.0, 5.0, np.nan, np.nan, 0.0, 1.0, 2.0, 3.0],
            "y": [0.1, 0.8, 1.9, 3.0, 1.0, 2.0, 0.9, 1.7, 2.8, 3.7],
        }
    )
    with pytest.warns(UserWarning, match="fitting failed for 1 group"):
        coef = lmList("y ~ x | g", data)["coef"]
    # lme4 2.1.0: a one-row group has only its intercept; one without rows is NA.
    assert coef.loc["2", "(Intercept)"] == pytest.approx(3.0)
    assert np.isnan(coef.loc["2", "x"]) and coef.loc["3"].isna().all()
    clash = pd.DataFrame(
        {"g": ["A", "A", "B", "B"] * 3, "k": ["a", "b", "c"] * 4, "k.1": np.arange(12.0)}
    )
    clash["y"] = np.arange(12.0)
    with pytest.raises(ValueError, match="Ambiguous coefficient name"):
        lmList("y ~ k + `k.1` | g", clash, pool=False)

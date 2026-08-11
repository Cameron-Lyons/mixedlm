from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, lmer
from numpy.testing import assert_allclose


def _random_intercept_data() -> pd.DataFrame:
    rng = np.random.default_rng(2468)
    n_groups = 8
    n_per_group = 15
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=len(groups))
    group_effects = np.repeat(rng.normal(0, 2.5, n_groups), n_per_group)
    y = 1.5 + 0.6 * x + group_effects + rng.normal(0, 0.2, len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def _nested_data() -> pd.DataFrame:
    rng = np.random.default_rng(1357)
    schools = np.repeat(["A", "B", "C"], 80)
    classrooms = np.tile(np.repeat(["1", "2", "3", "4"], 20), 3)
    x = rng.normal(size=len(schools))
    nested_index = (
        np.array([ord(school) - ord("A") for school in schools]) * 4 + classrooms.astype(int) - 1
    )
    nested_effects = rng.normal(0, 3, 12)
    y = 2 + 0.4 * x + nested_effects[nested_index] + rng.normal(0, 0.15, len(schools))
    return pd.DataFrame({"y": y, "x": x, "school": schools, "classroom": classrooms})


def _binomial_data() -> pd.DataFrame:
    rng = np.random.default_rng(9753)
    n_groups = 10
    n_per_group = 20
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=len(groups))
    group_effects = np.repeat(rng.normal(0, 0.6, n_groups), n_per_group)
    eta = -0.5 + 0.8 * x + group_effects
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_predict_matches_fitted_for_numeric_group_levels() -> None:
    data = _random_intercept_data()
    result = lmer("y ~ x + (1 | group)", data)

    predicted = result.predict(newdata=data)

    assert_allclose(predicted, result.fitted())


def test_predict_rejects_unknown_numeric_group_level() -> None:
    data = _random_intercept_data()
    result = lmer("y ~ x + (1 | group)", data)
    newdata = data.iloc[[0]].copy()
    newdata["group"] = 999

    with pytest.raises(ValueError, match="New level '999'"):
        result.predict(newdata=newdata)


def test_glmer_predict_matches_fitted_for_numeric_group_levels() -> None:
    data = _binomial_data()
    result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

    predicted = result.predict(newdata=data)

    assert_allclose(predicted, result.fitted())


def test_predict_accepts_polars_newdata() -> None:
    pl = pytest.importorskip("polars")
    data = _random_intercept_data()
    data["group"] = data["group"].astype(str)
    polars_data = pl.DataFrame(data.to_dict(orient="list"))
    result = lmer("y ~ x + (1 | group)", polars_data)

    predicted = result.predict(newdata=polars_data)

    assert_allclose(predicted, result.fitted())


def test_predict_preserves_nested_random_effects() -> None:
    data = _nested_data()
    result = lmer("y ~ x + (1 | school/classroom)", data)

    predicted = result.predict(newdata=data)
    fixed_only = result.matrices.X @ result.beta + result.matrices.offset

    assert np.max(np.abs(result.u)) > 1
    assert np.max(np.abs(predicted - fixed_only)) > 1
    assert_allclose(predicted, result.fitted())


def test_predict_rejects_unknown_nested_group_level() -> None:
    data = _nested_data()
    result = lmer("y ~ x + (1 | school/classroom)", data)
    newdata = data.iloc[[0]].copy()
    newdata["classroom"] = "new"

    with pytest.raises(ValueError, match="New level 'A/new'"):
        result.predict(newdata=newdata)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.estimation.reml import _count_theta
from mixedlm.families import Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(412)
    n = 60
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "category": np.tile(["low", "medium", "high"], n // 3),
            "group": np.repeat(["a", "b", "c", "d", "e"], n // 5),
            "item": np.tile(["one", "two", "three", "four"], n // 4),
            "shift": np.linspace(-0.3, 0.3, n),
        }
    )


def _result(kind, formula, data, contrasts=None):
    formula = parse_formula(formula)
    matrices = build_model_matrices(
        formula, data, contrasts=contrasts, offset=data["shift"].to_numpy()
    )
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.full(_count_theta(matrices.random_structures), 0.4),
        beta=np.linspace(0.2, 0.3, matrices.n_fixed),
        u=np.linspace(-0.5, 0.8, matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    if kind == "lmm":
        return LmerResult(**common, sigma=0.6, REML=True)
    return GlmerResult(**common, family=Poisson(), nAGQ=1)


def _predict(result, data, **kwargs):
    if isinstance(result, GlmerResult):
        kwargs["type"] = "link"
    return result.predict(data, **kwargs)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    "random",
    [
        "(0 + x:z | group)",
        "(I(x**2) | group)",
        "(0 + I(x**2):z | group)",
        "(category | group)",
        "(0 + category | group)",
        "(x:category | group)",
        "(x:z | group/item) + (I(x**2) | item)",
        "(x:z || group) + (category | item)",
    ],
)
def test_encoded_random_predictions_match_fitted_design(data, kind, backend, random) -> None:
    result = _result(kind, f"y ~ x + {random}", data, contrasts={"category": "sum"})
    # Shuffle and repeat rows, and retain just one factor level in the new data.
    rows = np.array([56, 2, 20, 8, 56])
    newdata = data.iloc[rows].drop(columns="y")
    if backend == "polars":
        pl = pytest.importorskip("polars")
        newdata = pl.DataFrame(newdata.to_dict(orient="list"))
    matrices = result.matrices
    expected = (matrices.X @ result.beta + matrices.Z @ result.u + matrices.offset)[rows]

    actual = _predict(result, newdata, offset="shift")
    with_se = _predict(result, newdata, offset="shift", se_fit=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(with_se.fit, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_custom_random_contrasts_use_training_schema(data, kind) -> None:
    contrasts = np.array([[1.0, 2.0], [-2.0, 1.0], [0.5, -1.5]])
    result = _result(kind, "y ~ x + (category | group)", data, {"category": contrasts})
    rows = np.array([0, 12, 24, 36, 48])
    expected = (result.matrices.X @ result.beta + result.matrices.Z @ result.u)[rows]
    contrasts[:] = 0.0

    actual = _predict(result, data.iloc[rows].drop(columns="y"))

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_crossed_predictions_keep_known_contributions_with_new_groups(data, kind) -> None:
    result = _result(kind, "y ~ x + (x:z | group) + (category | item)", data)
    rows = np.array([2, 15, 37])
    newdata = data.iloc[rows].drop(columns="y").copy()
    newdata.loc[:, "group"] = "new"
    first_width = (
        result.matrices.random_structures[0].n_levels * result.matrices.random_structures[0].n_terms
    )
    expected = result.matrices.X[rows] @ result.beta
    expected += result.matrices.Z[rows, first_width:] @ result.u[first_width:]

    actual = _predict(result, newdata, allow_new_levels=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="New level 'new'"):
        _predict(result, newdata)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("missing", ["z", "group", "item"])
def test_conditional_prediction_rejects_missing_random_inputs(data, kind, missing) -> None:
    result = _result(kind, "y ~ x + (x:z | group/item)", data)
    newdata = data.iloc[:3].drop(columns=["y", missing])

    with pytest.raises(ValueError, match=f"missing random-effect variable.*'{missing}'"):
        _predict(result, newdata)

    fixed = _predict(result, newdata, re_form="NA")
    np.testing.assert_allclose(fixed, result.matrices.X[:3] @ result.beta)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_unknown_random_predictor_category_is_not_a_new_group(data, kind) -> None:
    result = _result(kind, "y ~ x + (category | group)", data)
    newdata = data.iloc[:3].drop(columns="y").assign(category="unseen")

    with pytest.raises(ValueError, match="New level.*'unseen'.*random-effect factor 'category'"):
        _predict(result, newdata, allow_new_levels=True)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_formula_operator_in_column_name_is_distinct_from_interaction(data, kind) -> None:
    data = data.assign(**{"x:z": np.linspace(2.0, 3.0, len(data))})
    result = _result(kind, "y ~ x + (0 + x:z | group) + (0 + `x:z` | item)", data)
    expected = result.matrices.X @ result.beta + result.matrices.Z @ result.u

    np.testing.assert_allclose(_predict(result, data), expected, rtol=1e-12, atol=1e-12)


def test_fitted_interaction_model_reproduces_conditional_means(data) -> None:
    rng = np.random.default_rng(623)
    group_effects = data["group"].map({"a": -1.5, "b": 0.5, "c": 1.3, "d": -0.7, "e": 0.8})
    data["y"] = 2.0 + 0.3 * data["x"] + group_effects * data["x"] * data["z"]
    data["y"] += rng.normal(scale=0.1, size=len(data))
    result = lmer("y ~ x + (0 + x:z | group)", data)

    assert np.max(np.abs(result.u)) > 0.5
    np.testing.assert_allclose(result.predict(data.drop(columns="y")), result.fitted(), atol=1e-12)

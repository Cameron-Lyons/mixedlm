from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, parse_formula
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal

pl = pytest.importorskip("polars")


def _model(kind="lmm", formula="y ~ x + (1 + x | g)"):
    rng = np.random.default_rng(533)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=48),
            "z": rng.normal(size=48),
            "y": np.ones(48),
            "g": np.repeat(np.arange(6), 8),
            "site": np.tile(["east", "west"], 24),
            "treatment": np.tile(["a", "b", "c"], 16),
        }
    )
    formula = parse_formula(formula)
    matrices = build_model_matrices(formula, frame)
    n_theta = sum(s.n_terms * (s.n_terms + 1) // 2 for s in matrices.random_structures)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.full(n_theta, 0.2),
        beta=np.linspace(0.1, 0.3, matrices.n_fixed),
        u=rng.normal(scale=0.1, size=matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.8, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    model.vcov = lambda: np.eye(matrices.n_fixed) * 0.02
    return model


def _data():
    return pl.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, -0.5, 0.5, 1.5],
            "z": [0.5, -0.5, 1.0, 0.5, -0.5, 1.0],
            "g": [0, 1, 8, 0, 3, 5],
            "site": ["east", "west", "east", "east", "west", "west"],
            "treatment": ["c", "a", "b", "a", "b", "c"],
            "off": [0.1, 0.2, -0.3, 0.5, 0.0, -0.2],
            "row": [4, 3, 2, 1, 0, 5],
        }
    )


def _assert_result(actual, expected):
    if isinstance(expected, np.ndarray):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        return
    for name in ["fit", "se_fit", "lower", "upper"]:
        left, right = getattr(actual, name), getattr(expected, name)
        if right is None:
            assert left is None
        else:
            assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    assert actual.interval == expected.interval
    assert actual.level == expected.level


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("uncertainty", ["point", "se", "confidence"])
@pytest.mark.parametrize("offset_form", ["none", "scalar", "array", "column"])
def test_lazy_prediction_evaluates_predictors_once_and_matches_eager(
    kind, conditional, uncertainty, offset_form
):
    model = _model(kind)
    data = _data()
    calls = []

    def observe(values):
        calls.append(len(values))
        return values

    lazy = data.lazy().with_columns(pl.col("x").map_batches(observe, return_dtype=pl.Float64))
    lazy = lazy.filter(pl.col("row") != 2).sort("row")
    eager = data.filter(pl.col("row") != 2).sort("row")
    offset = {
        "none": None,
        "scalar": 0.2,
        "array": np.linspace(-0.2, 0.3, len(eager)),
        "column": "off",
    }[offset_form]
    options = dict(re_form=None if conditional else "NA", allow_new_levels=True, offset=offset)
    if uncertainty == "se":
        options["se_fit"] = True
    elif uncertainty == "confidence":
        options.update(interval="confidence", level=0.9)
    expected = model.predict(eager, **options)

    actual = model.predict(lazy, **options)

    _assert_result(actual, expected)
    assert len(calls) == 1
    model.predict(lazy, **options)
    assert len(calls) == 2  # No persistent model cache hides later query evaluations.


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("unused", ["y", "z", "g", "unrelated"])
def test_fixed_predictions_do_not_evaluate_unused_columns(kind, unused):
    model = _model(kind, "y ~ x + (1 + z | g)")
    data = _data().with_columns(pl.lit(1.0).alias(unused))

    def unexpected(values):
        raise AssertionError("An unused prediction column must not be evaluated")

    lazy = data.lazy().with_columns(pl.col(unused).map_batches(unexpected, return_dtype=pl.Float64))
    expected = model.predict(data, re_form="NA", offset="off")

    _assert_result(model.predict(lazy, re_form="NA", offset="off"), expected)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("n_rows", [0, 1, 7])
@pytest.mark.parametrize("offset", [None, 0.4])
def test_intercept_only_predictions_preserve_lazy_row_count_with_one_query(
    kind, n_rows, offset, monkeypatch
):
    model = _model(kind, "y ~ 1 + (1 | g)")
    data = pl.DataFrame({"unused": np.arange(n_rows, dtype=float)})
    lazy = data.lazy()
    expected = model.predict(pd.DataFrame(index=pd.RangeIndex(n_rows)), re_form="NA", offset=offset)
    calls = []
    original_collect = pl.LazyFrame.collect

    def observe_collect(query, *args, **kwargs):
        calls.append(query)
        return original_collect(query, *args, **kwargs)

    # Row-count queries can optimize away unused expressions, so count collections
    # directly instead of observing a Python expression that may never execute.
    monkeypatch.setattr(pl.LazyFrame, "collect", observe_collect)

    actual = model.predict(lazy, re_form="NA", offset=offset)

    assert actual.shape == (n_rows,)
    _assert_result(actual, expected)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize(
    "formula",
    [
        "y ~ treatment*x + (1 | g)",
        "y ~ x + I(x**2) + (1 | site/g)",
        "y ~ x + (1 | g) + (1 | site)",
    ],
)
def test_categorical_polynomial_nested_and_crossed_predictions_match_eager(kind, formula):
    model = _model(kind, formula)
    data = _data()
    options = dict(offset="off", allow_new_levels=True, interval="confidence", level=0.9)
    _assert_result(model.predict(data.lazy(), **options), model.predict(data, **options))


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_unknown_fixed_factor_levels_retain_their_error(kind):
    model = _model(kind, "y ~ treatment + (1 | g)")
    data = _data().with_columns(pl.lit("new").alias("treatment"))
    with pytest.raises(ValueError, match="New level.*fixed-effect factor"):
        model.predict(data.lazy(), re_form="NA")


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("missing", ["x", "off"])
def test_missing_prediction_columns_keep_specific_errors(kind, missing):
    data = _data().drop(missing)
    match = "missing fixed-effect variable" if missing == "x" else "missing offset column"
    with pytest.raises(ValueError, match=match):
        _model(kind).predict(data.lazy(), re_form="NA", offset="off")


def test_intercept_only_filtered_query_uses_the_resulting_row_count():
    model = _model(formula="y ~ 1 + (1 | g)")
    query = _data().lazy().filter(pl.col("x") > 0).sort("row").slice(0, 2)
    expected = model.predict(pd.DataFrame(index=pd.RangeIndex(2)), re_form="NA")
    assert_array_equal(model.predict(query, re_form="NA"), expected)


def test_polars_lazy_schema_access_does_not_collect_data():
    from mixedlm.utils.dataframe import get_columns

    def unexpected(values):
        raise AssertionError("Schema discovery must not evaluate data")

    query = (
        _data().lazy().with_columns(pl.col("x").map_batches(unexpected, return_dtype=pl.Float64))
    )
    assert get_columns(query) == _data().columns


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_lazy_file_predictions_project_away_unused_failing_expressions(
    tmp_path, kind, conditional, format
):
    model = _model(kind)
    data = _data().with_columns(pl.lit("not-a-number").alias("unused"))
    path = tmp_path / f"newdata.{format}"
    if format == "csv":
        data.write_csv(path)
        query = pl.scan_csv(path)
    else:
        data.write_parquet(path)
        query = pl.scan_parquet(path)
    query = query.with_columns(pl.col("unused").cast(pl.Int64))
    options = dict(re_form=None if conditional else "NA", allow_new_levels=True, offset="off")

    _assert_result(model.predict(query, **options), model.predict(data, **options))

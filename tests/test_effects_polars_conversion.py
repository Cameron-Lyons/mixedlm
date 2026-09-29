from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, ggpredict, parse_formula
from mixedlm.families import Poisson
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from mixedlm.utils.dataframe import get_column_numpy, get_column_values
from mixedlm.utils.na_action import _get_na_mask_polars, handle_na
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

pl = pytest.importorskip("polars")


def _model(kind="lmm", dtype=None, factor="enum", chunked=False):
    x = pl.Series("x", np.tile([-1.5, 0.2, 2.0], 12), dtype=dtype or pl.Float64)
    treatment = pl.Series("treatment", np.repeat(["C", "A", "B"], 12))
    if factor == "enum":
        treatment = treatment.cast(pl.Enum(["C", "A", "B"]))
    elif factor == "categorical":
        treatment = treatment.cast(pl.Categorical)
    data = pl.DataFrame(
        {
            "x": x,
            "treatment": treatment,
            "g": np.arange(36) % 6,
            "y": np.linspace(0.5, 2.0, 36),
        }
    )
    if chunked:
        data = pl.concat([data[:18], data[18:]], rechunk=False)
    formula = parse_formula("y ~ x * treatment + I(x**2) + (1 | g)")
    matrices = build_model_matrices(formula, data, contrasts={"treatment": "sum"})
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.5]),
        beta=np.linspace(-0.2, 0.3, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    return (
        LmerResult(**common, sigma=1.2, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=Poisson(), nAGQ=1)
    )


def _list_based_frame(frame):
    result = pd.DataFrame(frame.to_dict(as_series=False))
    for name in frame.columns:
        column = frame.get_column(name)
        if "Categorical" in str(column.dtype) or "Enum" in str(column.dtype):
            result[name] = pd.Categorical(
                result[name], categories=column.cat.get_categories().to_list()
            )
    return result


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32])
@pytest.mark.parametrize("factor", ["enum", "string"])
@pytest.mark.parametrize("chunked", [False, True])
def test_array_conversion_preserves_adjusted_predictions(kind, dtype, factor, chunked):
    model = _model(kind, dtype, factor, chunked)
    original = model.matrices.frame.clone()
    reference = replace(
        model, matrices=replace(model.matrices, frame=_list_based_frame(model.matrices.frame))
    )
    options = {"n_points": 3, "level": 0.9, "offset": 0.3, "contrasts": model.matrices.contrasts}

    actual = allEffects(model, **options)
    expected = allEffects(reference, **options)

    for variable in actual:
        assert_frame_equal(actual[variable], expected[variable], check_exact=True)
        assert actual[variable].attrs == expected[variable].attrs
    assert model.matrices.frame.equals(original)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Float32, [1e8, 1.0, -1e8, None]),
        (pl.Float64, [1e8, 1.0, -1e8, None]),
        (pl.Int8, [1, 3, -2, None]),
        (pl.Int32, [2**24 + 1, 2**24 + 3, -2, None]),
        (pl.Int64, [2**53 + 1, 2**53 + 3, -2, None]),
        (pl.UInt64, [2**63 + 1, 2**63 + 2, 3, None]),
        (pl.Boolean, [True, False, None, True]),
        (pl.Utf8, ["B", "A", None, "C"]),
    ],
)
def test_nullable_conversion_preserves_values_and_reference_precision(dtype, values):
    effects = import_module("mixedlm.inference.effects")
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    expected = _list_based_frame(frame)

    actual = effects._as_pandas_frame(frame)

    assert_frame_equal(actual, expected, check_dtype=False)
    if actual.x.dtype.kind == "f":
        assert actual.x.dtype == np.float64
        assert actual.x.mean() == expected.x.mean()


@pytest.mark.parametrize("dtype", [pl.Enum(["C", "A", "B", "unused"]), pl.Categorical])
def test_category_conversion_preserves_levels_order_and_missing_values(dtype):
    effects = import_module("mixedlm.inference.effects")
    frame = pl.DataFrame({"treatment": pl.Series(["B", "A", None, "C"]).cast(dtype)})

    actual = effects._as_pandas_frame(frame)

    assert_frame_equal(actual, _list_based_frame(frame))


def test_float32_conditioning_mean_is_computed_in_float64():
    effects = import_module("mixedlm.inference.effects")
    frame = pl.DataFrame({"x": pl.Series([1e8, 1.0, -1e8], dtype=pl.Float32)})

    converted = effects._as_pandas_frame(frame)

    assert effects._reference_value(converted.x) == pytest.approx(1.0 / 3.0)


def test_contiguous_float64_predictors_are_borrowed_without_copying():
    effects = import_module("mixedlm.inference.effects")
    frame = pl.DataFrame({"x": np.linspace(-1, 1, 1000), "z": np.arange(1000, dtype=float)})

    converted = effects._as_pandas_frame(frame)

    for name in frame.columns:
        assert np.shares_memory(converted[name].to_numpy(), frame.get_column(name).to_numpy())
        assert_array_equal(converted[name], frame.get_column(name).to_numpy())


@pytest.mark.parametrize("function", [ggpredict, allEffects])
def test_prediction_grids_do_not_materialize_python_row_objects(monkeypatch, function):
    model = _model()

    def unexpected_dictionary(self, *args, **kwargs):
        raise AssertionError(
            "prediction grids must not materialize the model frame as Python lists"
        )

    monkeypatch.setattr(pl.DataFrame, "to_dict", unexpected_dictionary)
    args = ("treatment",) if function is ggpredict else ()
    result = function(model, *args, contrasts=model.matrices.contrasts)

    assert len(result) > 0


@pytest.mark.parametrize("function", [ggpredict, allEffects])
def test_only_fixed_predictors_are_converted(monkeypatch, function):
    effects = import_module("mixedlm.inference.effects")
    model = _model()
    # Unused nested metadata should never enter the temporary pandas frame.
    model.matrices.frame = model.matrices.frame.with_columns(
        pl.struct("x", "y").alias("unused_metadata")
    )
    original_conversion = effects._as_pandas_frame
    observed = []

    def convert(frame):
        observed.append(frame.columns)
        assert frame.columns == ["x", "treatment"]
        return original_conversion(frame)

    monkeypatch.setattr(effects, "_as_pandas_frame", convert)
    args = ("treatment",) if function is ggpredict else ()
    result = function(model, *args, contrasts=model.matrices.contrasts)

    assert observed
    assert len(result) > 0


def test_prediction_output_can_be_edited_without_changing_borrowed_model_data():
    model = _model()
    original = model.matrices.frame.clone()

    result = ggpredict(model, "x", contrasts=model.matrices.contrasts)
    result.loc[:, "x"] = -100.0
    result.loc[:, "predicted"] = 0.0

    assert model.matrices.frame.equals(original)
    assert_allclose(result.x, -100.0)


@pytest.mark.parametrize("dtype", [pl.Enum(["C", "A", "B"]), pl.Categorical])
@pytest.mark.parametrize("extract", [get_column_numpy, get_column_values])
def test_older_categorical_export_preserves_values_and_nulls(monkeypatch, dtype, extract):
    frame = pl.DataFrame({"treatment": pl.Series(["B", None, "A", "C"]).cast(dtype)})
    original = pl.Series.to_numpy

    def export(self, *args, **kwargs):
        if "Categorical" in str(self.dtype) or "Enum" in str(self.dtype):
            raise pl.exceptions.ComputeError("categorical export unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_numpy", export)

    assert_array_equal(extract(frame, "treatment"), ["B", None, "A", "C"])


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_prediction_works_when_native_categorical_export_is_unavailable(monkeypatch, kind):
    original = pl.Series.to_numpy

    def export(self, *args, **kwargs):
        if "Categorical" in str(self.dtype) or "Enum" in str(self.dtype):
            raise pl.exceptions.ComputeError("categorical export unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_numpy", export)
    model = _model(kind)
    reference = replace(
        model, matrices=replace(model.matrices, frame=_list_based_frame(model.matrices.frame))
    )

    actual = ggpredict(model, "treatment", contrasts=model.matrices.contrasts)
    expected = ggpredict(reference, "treatment", contrasts=model.matrices.contrasts)

    assert_frame_equal(actual, expected, check_exact=True)


def test_numeric_export_errors_are_not_converted_to_strings(monkeypatch):
    frame = pl.DataFrame({"x": [1.0, 2.0]})

    def failed_export(self, *args, **kwargs):
        raise pl.exceptions.ComputeError("numeric export failed")

    monkeypatch.setattr(pl.Series, "to_numpy", failed_export)
    with pytest.raises(pl.exceptions.ComputeError, match="numeric export failed"):
        get_column_numpy(frame, "x")


@pytest.fixture
def legacy_boolean_export(monkeypatch):
    original = pl.Series.to_numpy

    def export(self, *args, **kwargs):
        values = original(self, *args, **kwargs)
        return values.astype(object) if self.dtype == pl.Boolean else values

    monkeypatch.setattr(pl.Series, "to_numpy", export)


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("extract", [get_column_numpy, get_column_values])
def test_older_boolean_export_preserves_dtype_and_nulls(legacy_boolean_export, nullable, extract):
    values = [True, False, None if nullable else True]
    frame = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})

    actual = extract(frame, "flag")

    assert actual.dtype == (object if nullable else np.bool_)
    assert_array_equal(actual, values)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_older_boolean_export_preserves_adjusted_predictions(legacy_boolean_export, kind):
    model = _model(kind)
    data = model.matrices.frame.with_columns(pl.Series("flag", np.arange(36) % 2 == 0))
    formula = parse_formula("y ~ x * flag + (1 | g)")
    matrices = build_model_matrices(formula, data)
    model = replace(
        model, formula=formula, matrices=matrices, beta=np.linspace(0.1, 0.4, matrices.n_fixed)
    )
    reference = replace(model, matrices=replace(matrices, frame=_list_based_frame(data)))

    actual = allEffects(model)
    expected = allEffects(reference)

    for variable in actual:
        assert_frame_equal(actual[variable], expected[variable], check_exact=True)


@pytest.mark.parametrize("action", ["omit", "exclude"])
def test_older_boolean_export_keeps_missing_rows_and_vectors_aligned(legacy_boolean_export, action):
    data = pl.DataFrame(
        {
            "x": [0.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "flag": [True, False, None, True, False, True, False, True],
            "y": [0.0, 1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0],
            "g": ["A", "A", "A", "B", "B", "B", "C", "C"],
        }
    )
    weights = np.arange(1.0, 9.0)
    weights[4] = np.nan
    offset = np.arange(8.0) / 10.0
    offset[5] = np.nan
    formula = parse_formula("y ~ x + flag + (1 | g)")

    mask = _get_na_mask_polars(data, ["x", "flag", "y", "g"])
    clean, info, clean_weights, clean_offset = handle_na(
        data, formula, action, weights=weights, offset=offset
    )

    assert mask.dtype == np.bool_
    assert_array_equal(mask, [False, True, True, True, False, False, False, False])
    assert clean.get_column("x").to_list() == [0.0, 6.0, 7.0]
    assert_array_equal(info.omitted_indices, [1, 2, 3, 4, 5])
    assert_array_equal(clean_weights, weights[[0, 6, 7]])
    assert_array_equal(clean_offset, offset[[0, 6, 7]])
    assert_array_equal(
        info.expand_to_original(np.array([1.0, 2.0, 3.0])),
        [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0, 3.0],
    )

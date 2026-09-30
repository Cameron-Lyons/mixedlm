from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, nlme, nlmer, parse_formula
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from mixedlm.models.nlmer import NlmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal


def _model(kind):
    data = pd.DataFrame(
        {
            "x": np.tile([0.5, 1.0], 6),
            "y": np.ones(12),
            "g": np.repeat(["a", "b", "c"], 4),
        }
    )
    if kind == "nlmm":
        return NlmerResult(
            model=nlme.SSasymp(),
            group_var="g",
            phi=np.array([5.0, 2.0, -1.0]),
            theta=np.array([0.2]),
            sigma=1.0,
            b=np.array([[0.3], [-0.2], [0.1]]),
            random_params=[0],
            deviance=0.0,
            converged=True,
            n_iter=0,
            x=data.x.to_numpy(),
            y=data.y.to_numpy(),
            groups=np.repeat(np.arange(3), 4),
            group_levels=["a", "b", "c"],
            _x_var="x",
            _offset=np.full(12, 2.0),
        )
    formula = parse_formula("y ~ x + (1 | g)")
    matrices = build_model_matrices(formula, data, offset=np.full(12, 2.0))
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.2]),
        beta=np.array([0.3, 0.2]),
        u=np.array([0.3, -0.2, 0.1]),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    return (
        LmerResult(**common, sigma=1.0, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )


def _predict(model, kind, data, *, conditional=False, **kwargs):
    if kind == "nlmm":
        return model.predict(data, group_var="g" if conditional else None, **kwargs)
    return model.predict(
        data, re_form=None if conditional else "NA", allow_new_levels=True, **kwargs
    )


@pytest.mark.parametrize("kind", ["lmm", "glmm", "nlmm"])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize(
    "form", ["none", "scalar", "zero-d", "array", "list", "series", "column", "unmasked"]
)
def test_prediction_offsets_match_the_requested_scale_and_row_order(kind, conditional, form):
    model = _model(kind)
    data = pd.DataFrame(
        {"x": [1.0, 0.5, 1.5], "g": ["b", "new", "a"], "off": [0.2, -0.3, 0.7]}, index=[9, 2, 9]
    )
    original = data.copy(deep=True)
    vector = data.off.to_numpy(copy=True)
    vector.setflags(write=False)
    alternatives = {
        "none": None,
        "scalar": 0.4,
        "zero-d": np.array(0.4),
        "array": vector,
        "list": vector.tolist(),
        "series": pd.Series(vector, index=[10, 20, 30]),
        "column": "off",
        "unmasked": np.ma.array(vector, mask=False),
    }
    offset = alternatives[form]
    expected_offset = 0 if form == "none" else 0.4 if form in {"scalar", "zero-d"} else vector
    baseline = _predict(model, kind, data, conditional=conditional)

    actual = _predict(model, kind, data, conditional=conditional, offset=offset)

    expected = baseline * np.exp(expected_offset) if kind == "glmm" else baseline + expected_offset
    assert_allclose(actual, expected, rtol=5e-14, atol=1e-14)
    assert_frame_equal(data, original)
    if kind == "glmm":
        link = _predict(model, kind, data, conditional=conditional, type="link", offset=offset)
        assert_allclose(actual, np.exp(link), rtol=5e-14)


INVALID = [
    np.nan,
    np.inf,
    -np.inf,
    1 + 0j,
    np.complex128(1 + 2j),
    np.array([1 + 2j, 0.0, 0.0]),
    np.array([np.complex128(1 + 2j), 0.0, 0.0], dtype=object),
    np.ma.masked,
    np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False]),
    np.array([np.ma.masked, 0.0, 0.0], dtype=object),
    np.datetime64("2026-01-01"),
    np.timedelta64(1, "D"),
    2**2000,
    [],
    [1.0],
    [1.0, 2.0],
    [[1.0, 2.0, 3.0]],
    [1.0, np.nan, 3.0],
    "missing",
]


@pytest.mark.parametrize("kind", ["lmm", "glmm", "nlmm"])
@pytest.mark.parametrize("offset", INVALID)
def test_invalid_offsets_fail_before_design_or_nonlinear_evaluation(monkeypatch, kind, offset):
    model = _model(kind)
    data = pd.DataFrame({"x": [0.1, 0.2, 0.3], "g": ["a", "b", "c"]})

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid offsets must be checked before prediction work")

    if kind == "nlmm":
        monkeypatch.setattr(model.model, "predict", unexpected)
    else:
        monkeypatch.setattr(model, "_prediction_fixed_matrix", unexpected)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="offset"):
            _predict(model, kind, data, offset=offset)


@pytest.mark.parametrize("kind", ["lmm", "glmm", "nlmm"])
@pytest.mark.parametrize("offset", [np.nan, np.inf, np.complex128(1 + 2j), np.ma.masked])
def test_invalid_scalar_offsets_are_rejected_for_empty_newdata(kind, offset):
    data = pd.DataFrame({"x": pd.Series(dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="offset"):
            _predict(_model(kind), kind, data, offset=offset)


@pytest.mark.parametrize("kind", ["lmm", "glmm", "nlmm"])
@pytest.mark.parametrize("offset", [None, 0.5, np.array([]), "off"])
def test_empty_prediction_inputs_preserve_the_empty_shape(kind, offset):
    data = pd.DataFrame({"x": pd.Series(dtype=float), "off": pd.Series(dtype=float)})
    result = _predict(_model(kind), kind, data, offset=offset)
    assert result.shape == (0,)
    assert result.dtype == np.float64


@pytest.mark.parametrize("kind", ["lmm", "glmm", "nlmm"])
def test_in_sample_predictions_retain_fitted_offsets_and_reject_overrides(kind):
    model = _model(kind)
    assert_array_equal(model.predict(), model.fitted())
    with pytest.raises(ValueError, match="only be supplied with newdata"):
        model.predict(offset=1.0)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("dtype", ["int64", "Int64", "float64", "Float64", "string"])
def test_pandas_offset_column_dtypes_keep_numeric_conversion(kind, dtype):
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "off": pd.Series([1, 2, 3], dtype=dtype)})
    model = _model(kind)
    assert_allclose(
        _predict(model, kind, data, offset="off"), _predict(model, kind, data, offset=[1, 2, 3])
    )


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("lazy", [False, True])
def test_polars_offset_columns_preserve_order(kind, lazy):
    pl = pytest.importorskip("polars")
    data = pl.DataFrame({"x": [0.5, 1.0, 1.5], "off": [0.1, 0.2, 0.3]})
    if lazy:
        data = data.lazy()
    model = _model(kind)
    assert_allclose(
        _predict(model, kind, data, offset="off"),
        _predict(model, kind, data, offset=[0.1, 0.2, 0.3]),
    )


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("offset", [None, 0.4, np.array(0.4)])
def test_constant_offset_storage_does_not_scale_with_prediction_rows(kind, offset):
    data = pd.DataFrame(index=pd.RangeIndex(1_000_000))
    resolved = _model(kind)._prediction_offset(data, offset)
    assert resolved.shape == (len(data),)
    assert resolved.strides == (0,)
    assert not resolved.flags.writeable
    assert resolved.base.nbytes == 8


@pytest.mark.parametrize(
    "kind,scale", [("lmm", "response"), ("glmm", "link"), ("glmm", "response")]
)
@pytest.mark.parametrize("interval", ["none", "confidence"])
def test_offsets_preserve_prediction_uncertainty(kind, scale, interval):
    model = _model(kind)
    model.vcov = lambda: np.eye(2) * 0.02
    data = pd.DataFrame({"x": [0.5, 1.0, 1.5]})
    offset = np.array([0.2, -0.3, 0.7])
    options = dict(se_fit=True, interval=interval)
    if kind == "glmm":
        options["type"] = scale
    baseline = _predict(model, kind, data, **options)
    actual = _predict(model, kind, data, offset=offset, **options)
    response = kind == "glmm" and scale == "response"
    for name in ["fit", "lower", "upper"]:
        before = getattr(baseline, name)
        if before is not None:
            expected = before * np.exp(offset) if response else before + offset
            assert_allclose(getattr(actual, name), expected, rtol=5e-14, atol=1e-14)
    expected_se = baseline.se_fit * np.exp(offset) if response else baseline.se_fit
    assert_allclose(actual.se_fit, expected_se, rtol=5e-14, atol=1e-14)


def test_nonlinear_offsets_do_not_modify_cached_model_predictions(monkeypatch):
    model = _model("nlmm")
    cached = np.array([1.0, 2.0, 3.0])
    cached.setflags(write=False)
    monkeypatch.setattr(model.model, "predict", lambda params, x: cached)
    result = model.predict(pd.DataFrame({"x": [0.5, 1.0, 1.5]}), offset=0.5)
    assert_array_equal(result, [1.5, 2.5, 3.5])
    assert_array_equal(cached, [1.0, 2.0, 3.0])
    assert not np.shares_memory(result, cached)


def test_nonlinear_fitted_offset_prediction_round_trip():
    from tests.test_nlmer_methods import create_offset_nlme_data

    data = create_offset_nlme_data()
    offset = np.linspace(-2.0, 2.0, len(data))
    result = nlmer(
        nlme.SSasymp(),
        data,
        x_var="time",
        y_var="y",
        group_var="subject",
        random_params=["Asym"],
        offset=offset,
    )

    actual = result.predict(data.assign(off=offset), group_var="subject", offset="off")

    assert_allclose(actual, result.fitted(), rtol=1e-12, atol=1e-12)
    assert_allclose(
        result.predict(data, group_var="subject"), result.fitted() - offset, rtol=1e-12, atol=1e-12
    )

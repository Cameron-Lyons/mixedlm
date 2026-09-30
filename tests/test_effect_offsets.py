from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, families, ggpredict, glmer, lmer, parse_formula
from mixedlm.matrices import build_model_matrices
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats


@pytest.fixture(
    scope="module",
    params=list(itertools.product(["lmm", "glmm"], ["pandas", "polars"], ["omit", "exclude"])),
)
def fitted_model(request):
    kind, backend, action = request.param
    rng = np.random.default_rng(530)
    x = rng.normal(size=120)
    treatment = np.tile(["a", "b"], 60)
    groups = np.repeat(np.arange(10), 12)
    offset = np.log(np.linspace(2.0, 20.0, len(x)))
    eta = 0.2 + 0.2 * x + 0.3 * (treatment == "b") + offset
    eta += np.repeat(rng.normal(scale=0.3, size=10), 12)
    y = eta + rng.normal(scale=0.2, size=len(x)) if kind == "lmm" else rng.poisson(np.exp(eta))
    weights = np.linspace(0.5, 2.0, len(x))
    data = pd.DataFrame({"y": y, "x": x, "treatment": treatment, "g": groups})
    # Removed rows have distinct offsets; the default must use the fitted subset.
    data.loc[[0, 1, 2, 3], "x"] = np.nan
    weights[4] = np.nan
    offset[5] = np.nan
    if backend == "polars":
        polars = pytest.importorskip("polars")
        data = polars.DataFrame(data.to_dict(orient="list"))
    options = dict(offset=offset, weights=weights, na_action=action)
    model = (
        lmer("y ~ x * treatment + (1 | g)", data, **options)
        if kind == "lmm"
        else glmer("y ~ x * treatment + (1 | g)", data, family=families.Poisson(), **options)
    )
    assert_array_equal(model.matrices.offset, offset[6:])
    return model


def _assert_prediction(model, result, grid, offset, scale):
    options = dict(re_form="NA", offset=offset, interval="confidence", se_fit=True, level=0.9)
    if model.isGLMM():
        options["type"] = scale
    direct = model.predict(grid, **options)
    if not model.isGLMM():
        # Adjusted LMM effects use residual-df t intervals; predict() uses normal intervals.
        margin = stats.t.isf(0.05, model.df_residual()) * direct.se_fit
        direct.lower = direct.fit - margin
        direct.upper = direct.fit + margin
    for column, field in [
        ("predicted", "fit"),
        ("std.error", "se_fit"),
        ("conf.low", "lower"),
        ("conf.high", "upper"),
    ]:
        assert_allclose(result[column], getattr(direct, field), rtol=5e-12, atol=1e-12)


@pytest.mark.parametrize("scale", ["link", "response"])
def test_default_effects_use_the_mean_fitted_offset(fitted_model, scale):
    model = fitted_model
    mean = float(np.mean(model.matrices.offset))
    assert not np.isclose(mean, np.average(model.matrices.offset, weights=model.matrices.weights))
    options = dict(at={"x": [-0.5, 0.5]}, type=scale, level=0.9)

    result = ggpredict(model, ["x", "treatment"], **options)

    _assert_prediction(model, result, result[["x", "treatment"]], mean, scale)
    assert result.attrs["offset"] == mean
    assert_frame_equal(result, ggpredict(model, ["x", "treatment"], offset=None, **options))
    assert_frame_equal(result, ggpredict(model, ["x", "treatment"], offset=mean, **options))
    rates = ggpredict(model, ["x", "treatment"], offset=0, **options)
    _assert_prediction(model, rates, rates[["x", "treatment"]], 0, scale)
    if model.isGLMM() and scale == "response":
        assert_allclose(result["predicted"], rates["predicted"] * np.exp(mean))
        assert_allclose(result["std.error"], rates["std.error"] * np.exp(mean))
    else:
        assert_allclose(result["predicted"], rates["predicted"] + mean)
        assert_allclose(result["std.error"], rates["std.error"])


@pytest.mark.parametrize("scale", ["link", "response"])
def test_row_offsets_follow_product_order_and_leave_inputs_unchanged(fitted_model, scale):
    model = fitted_model
    values = np.log(np.arange(2.0, 8.0))
    values.setflags(write=False)
    # Series indexes are deliberately unrelated to positional grid row numbers.
    offsets = pd.Series(values, index=[10, 8, 7, 5, 3, 1])
    original = offsets.copy()
    options = dict(at={"x": [0.5, -0.5, 0.5], "treatment": ["b", "a"]}, type=scale, level=0.9)

    result = ggpredict(model, ["x", "treatment"], offset=offsets, **options)

    assert result[["x", "treatment"]].to_records(index=False).tolist() == list(
        itertools.product([0.5, -0.5, 0.5], ["b", "a"])
    )
    _assert_prediction(model, result, result[["x", "treatment"]], values, scale)
    assert_frame_equal(result, ggpredict(model, ["x", "treatment"], offset=values, **options))
    pd.testing.assert_series_equal(offsets, original)
    assert result.attrs["offset"] == tuple(values)
    pd.concat([result, result.copy()])  # Attribute comparison must remain unambiguous.
    offsets = offsets.copy()
    offsets.iloc[0] = 0
    assert result.attrs["offset"] == tuple(values)


def test_all_effects_share_the_resolved_reference_offset(fitted_model):
    mean = float(np.mean(fitted_model.matrices.offset))
    result = allEffects(fitted_model, n_points=3, level=0.9)
    rates = allEffects(fitted_model, n_points=3, level=0.9, offset=0)

    assert list(result) == ["x", "treatment"]
    for variable in result:
        assert_frame_equal(
            result[variable], ggpredict(fitted_model, variable, n_points=3, level=0.9)
        )
        assert result[variable].attrs["offset"] == mean
        assert rates[variable].attrs["offset"] == 0


def _model(offset=None, formula="y ~ x + z + (1 | g)"):
    data = pd.DataFrame(
        {
            "y": np.ones(12),
            "x": np.tile([-1.0, 0.0, 1.0], 4),
            "z": np.tile([-0.5, 0.5], 6),
            "g": np.repeat(np.arange(6), 2),
        }
    )
    formula = parse_formula(formula)
    matrices = build_model_matrices(formula, data, offset=offset)
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        beta=np.zeros(matrices.n_fixed),
        theta=np.array([0.2]),
        sigma=1.0,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
        REML=True,
    )
    model.vcov = lambda: np.eye(matrices.n_fixed)
    return model


INVALID = [
    np.datetime64("2026-01-01"),
    np.timedelta64(2, "D"),
    2**2000,
    np.nan,
    np.inf,
    -np.inf,
    "bad",
    [0, np.nan],
    [[0, 1]],
    [],
    1 + 0j,
    np.complex128(1 + 2j),
    np.array([1 + 2j]),
    np.array([np.complex128(1 + 2j)], dtype=object),
    np.ma.masked,
    np.ma.array([1.0], mask=True),
    np.array([np.ma.masked], dtype=object),
]


@pytest.mark.parametrize("offset", INVALID)
@pytest.mark.parametrize("api", ["ggpredict", "allEffects"])
def test_invalid_offsets_fail_before_covariance(api, offset):
    model = _model()

    def unexpected():
        raise AssertionError("Invalid offsets must not request covariance")

    model.vcov = unexpected
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises((TypeError, ValueError), match="offset"):
            if api == "ggpredict":
                ggpredict(model, "x", offset=offset)
            else:
                allEffects(model, offset=offset)


@pytest.mark.parametrize("offset", [[0.0], [0.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
def test_row_offset_length_is_exact_and_checked_before_covariance(offset):
    model = _model()

    def unexpected():
        raise AssertionError("Wrong-length offsets must not request covariance")

    model.vcov = unexpected
    with pytest.raises(ValueError, match="3 values"):
        ggpredict(model, "x", offset=offset)


@pytest.mark.parametrize("formula", ["y ~ 1 + (1 | g)", "y ~ x + (1 | g)", "y ~ x + z + (1 | g)"])
def test_all_effects_rejects_row_offsets_even_for_empty_or_equal_length_grids(formula):
    model = _model(formula=formula)
    with pytest.raises(ValueError, match="allEffects offset must be a scalar"):
        allEffects(model, offset=[1.0, 2.0, 3.0])


@pytest.mark.parametrize("value", [1e308, -1e308, np.nextafter(0.0, 1.0)])
def test_finite_fitted_offsets_have_a_finite_reference_even_when_the_sum_overflows(value):
    model = _model(offset=np.full(12, value))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ggpredict(model, "x")
    assert result.attrs["offset"] == value
    assert_array_equal(result["predicted"], np.full(len(result), value))


@pytest.mark.parametrize(
    "offset", [np.array(0.5), "0.5", [0.5, 0.5, 0.5], np.ma.array([0.5] * 3, mask=False)]
)
def test_supported_numeric_offsets_match_scalar_predictions(offset):
    model = _model()
    result = ggpredict(model, "x", offset=offset)
    expected = ggpredict(model, "x", offset=0.5)
    assert_frame_equal(result, expected)


def test_scalar_offset_metadata_stays_independent_of_input():
    model = _model()
    offset = np.array(0.5)
    result = ggpredict(model, "x", offset=offset)
    offset[...] = 2.0
    assert result.attrs["offset"] == 0.5


def test_models_without_fitted_offsets_keep_zero_reference():
    model = _model()
    for variable, result in allEffects(model).items():
        assert result.attrs["offset"] == 0
        assert_frame_equal(result, ggpredict(model, variable, offset=0))

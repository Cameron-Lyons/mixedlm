from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import replace
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, ggpredict, lmer, lmerControl, parse_formula
from mixedlm.families import Poisson
from mixedlm.matrices import build_model_matrices
from mixedlm.matrices.design import build_fixed_matrix
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats


def _model(contrast="sum", kind="lmm", backend="pandas", intercept=True):
    data = pd.DataFrame(
        itertools.product(["high", "low", "middle"], ["west", "east"], [-1.5, 0.0, 2.0]),
        columns=["treatment", "site", "x"],
    )
    data["treatment"] = pd.Categorical(
        data.treatment, categories=["high", "low", "middle"], ordered=True
    )
    data["g"] = np.arange(len(data)) % 6
    data["y"] = 1.0 + data.x + np.sin(np.arange(len(data)))
    if backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(data.to_dict("list")).with_columns(
            pl.col("treatment").cast(pl.Enum(["high", "low", "middle"]))
        )
    formula = parse_formula(
        "y ~ "
        + ("" if intercept else "0 + ")
        + "treatment * site + x + I(x**2) + treatment:x + (1 | g)"
    )
    matrices = build_model_matrices(formula, data, contrasts={"treatment": contrast, "site": "sum"})
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.5]),
        beta=np.linspace(-0.3, 0.5, matrices.n_fixed),
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


def _contrast(name):
    return np.array([[0.2, 0.8], [-0.7, 0.1], [0.5, -0.9]]) if name == "custom" else name


def _expected_grid(result, model, at=None):
    grid = result.drop(columns=["predicted", "std.error", "conf.low", "conf.high"]).copy()
    defaults = {"x": 1.0 / 6.0, "treatment": "high", "site": "east"}
    defaults.update(at or {})
    for name, value in defaults.items():
        if name not in grid:
            grid[name] = value
    return grid


def _assert_predictions(result, model, grid, contrasts=None):
    contrasts = model.matrices.contrasts if contrasts is None else contrasts
    matrix, names = build_fixed_matrix(
        model.formula, grid, contrasts=contrasts, category_levels=model.matrices.category_levels
    )
    rows = matrix[:, [names.index(name) for name in model.matrices.fixed_names]]
    eta = rows @ model.beta
    se = np.sqrt(np.diag(rows @ model.vcov() @ rows.T))
    critical = stats.norm.ppf(0.975) if model.isGLMM() else stats.t.ppf(0.975, model.df_residual())
    lower, upper = eta - critical * se, eta + critical * se
    if model.isGLMM() and result.attrs["type"] == "response":
        expected = np.exp(eta)
        lower, upper = np.exp(lower), np.exp(upper)
        se = se * expected
    else:
        expected = eta
    assert_allclose(result.predicted, expected, atol=1e-13)
    assert_allclose(result["std.error"], se, atol=1e-13)
    assert_allclose(result["conf.low"], lower, atol=1e-13)
    assert_allclose(result["conf.high"], upper, atol=1e-13)


@pytest.mark.parametrize("contrast", ["sum", "helmert", "poly", "custom"])
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("terms", "at"),
    [
        ("treatment", {}),
        (["x", "treatment"], {"x": [2.0, -1.5]}),
        ("x", {"x": [0.5, -1.0], "treatment": "low"}),
    ],
)
def test_default_effect_grid_uses_fitted_encoding(contrast, kind, backend, terms, at):
    model = _model(_contrast(contrast), kind, backend)

    result = ggpredict(model, terms, at=at)

    _assert_predictions(result, model, _expected_grid(result, model, at))


@pytest.mark.parametrize("contrast", ["sum", "helmert", "poly", "custom"])
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_all_effects_reuses_encoding_for_focal_and_conditioning_factors(contrast, kind):
    model = _model(_contrast(contrast), kind)

    result = allEffects(model, n_points=4)

    assert list(result) == ["treatment", "site", "x"]
    for effect in result.values():
        _assert_predictions(effect, model, _expected_grid(effect, model))


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("contrast", ["sum", "helmert", "poly", "custom"])
def test_explicit_contrast_mapping_remains_supported(backend, contrast):
    model = _model(_contrast(contrast), backend=backend)
    explicit = ggpredict(model, "treatment", contrasts=model.matrices.contrasts)
    automatic = ggpredict(model, "treatment")

    assert_frame_equal(automatic, explicit)
    _assert_predictions(explicit, model, _expected_grid(explicit, model))


def test_explicit_override_does_not_modify_fitted_contrasts():
    model = _model()
    original = dict(model.matrices.contrasts)
    overrides = {"treatment": "helmert", "site": "treatment"}

    result = ggpredict(model, "treatment", contrasts=overrides)

    _assert_predictions(result, model, _expected_grid(result, model), contrasts=overrides)
    assert model.matrices.contrasts == original


def test_fitted_custom_contrasts_are_independent_of_original_input():
    contrast = _contrast("custom")
    model = _model(contrast)
    expected = ggpredict(model, "treatment", contrasts=model.matrices.contrasts)

    contrast[:] = 100.0
    result = ggpredict(model, "treatment")

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("representation", ["object", "reordered", "additional_level"])
def test_fitted_category_schema_controls_grid_and_reference_levels(representation):
    model = _model()
    original = allEffects(model, contrasts=model.matrices.contrasts)
    frame = model.matrices.frame
    if representation == "object":
        frame["treatment"] = frame.treatment.astype(object)
    elif representation == "reordered":
        frame["treatment"] = frame.treatment.cat.reorder_categories(["low", "middle", "high"])
    else:
        frame["treatment"] = frame.treatment.cat.add_categories(["unused_after_fit"])

    actual = allEffects(model)

    assert list(actual) == list(original)
    for name in actual:
        assert_array_equal(actual[name].predicted, original[name].predicted)
        if name == "treatment":
            assert actual[name].treatment.tolist() == ["high", "low", "middle"]


def test_new_frame_categories_cannot_be_requested_as_fitted_levels():
    model = _model()
    model.matrices.frame["treatment"] = model.matrices.frame.treatment.cat.add_categories(
        ["unused_after_fit"]
    )

    with pytest.raises(ValueError, match="Unknown level.*unused_after_fit"):
        ggpredict(model, "treatment", at={"treatment": "unused_after_fit"})


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_no_intercept_grid_preserves_indicator_and_contrast_columns(kind):
    model = _model("helmert", kind, intercept=False)
    result = ggpredict(model, ["treatment", "site"], at={"x": 0.4}, type="link")

    _assert_predictions(result, model, _expected_grid(result, model, {"x": 0.4}))


def test_grid_preserves_retained_columns_after_rank_deficiency():
    rng = np.random.default_rng(533)
    data = pd.DataFrame(
        {"treatment": np.tile(["C", "B", "A"], 20), "g": np.repeat(np.arange(10), 6)}
    )
    data["x"] = rng.normal(size=len(data))
    data["alias"] = 2 * data.x
    data["y"] = data.x + (data.treatment == "A") + rng.normal(size=len(data))
    with pytest.warns(UserWarning, match="rank deficient"):
        model = lmer(
            "y ~ treatment + x + alias + (1 | g)",
            data,
            contrasts={"treatment": "sum"},
            control=lmerControl(check_singular=False, check_rankX="warning+drop.cols"),
        )
    result = ggpredict(model, "treatment", at={"x": 0.5, "alias": 1.0})
    grid = result[["treatment"]].assign(x=0.5, alias=1.0)

    assert len(model.beta) == 4
    assert_allclose(result.predicted, model.predict(grid, re_form="~0"))
    _assert_predictions(result, model, grid)


def test_effect_calculations_leave_the_fitted_frame_unchanged():
    model = _model()
    frame = model.matrices.frame.copy(deep=True)

    effects = allEffects(model)
    effects["treatment"]["treatment"] = "edited_output"

    assert_frame_equal(model.matrices.frame, frame)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("type", ["link", "response"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"n_points": 4, "level": 0.8, "offset": 0.7},
        {"at": {"x": 0.5, "treatment": "low"}, "level": 0.99},
    ],
)
def test_shared_effect_setup_matches_independent_grids(kind, backend, type, options):
    model = _model("helmert", kind, backend)

    actual = allEffects(model, type=type, **options)

    for variable, result in actual.items():
        expected = ggpredict(model, variable, type=type, **options)
        assert_frame_equal(result, expected)
        assert result.attrs == expected.attrs


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_all_effects_converts_frame_and_computes_shared_statistics_once(monkeypatch, kind, backend):
    effects = import_module("mixedlm.inference.effects")
    model = _model("sum", kind, backend)
    calls = Counter()
    original_frame = effects._as_pandas_frame
    original_reference = effects._reference_value
    original_vcov = type(model).vcov

    def frame(source):
        calls["frame"] += 1
        return original_frame(source)

    def reference(series, levels=None):
        calls[series.name] += 1
        return original_reference(series, levels)

    def covariance(self):
        calls["covariance"] += 1
        return original_vcov(self)

    monkeypatch.setattr(effects, "_as_pandas_frame", frame)
    monkeypatch.setattr(effects, "_reference_value", reference)
    monkeypatch.setattr(type(model), "vcov", covariance)

    result = allEffects(model)

    assert list(result) == ["treatment", "site", "x"]
    assert calls == {"frame": 1, "covariance": 1, "treatment": 1, "site": 1, "x": 1}


def test_all_effects_consumes_each_override_iterable_once():
    model = _model()
    expected = allEffects(model, at={"x": [0.5], "site": ["west"]})

    actual = allEffects(model, at={"x": iter([0.5]), "site": iter(["west"])})

    for variable in actual:
        assert_frame_equal(actual[variable], expected[variable])


def test_shared_effect_statistics_are_fresh_on_each_call():
    model = _model()
    original = allEffects(model)
    model.beta = model.beta + 0.2
    model.sigma *= 1.5
    model.matrices.frame["x"] = model.matrices.frame.x + 2.0

    actual = allEffects(model)

    for variable in actual:
        expected = ggpredict(model, variable)
        assert_frame_equal(actual[variable], expected)
        assert not np.array_equal(actual[variable].predicted, original[variable].predicted)
        assert not np.array_equal(actual[variable]["std.error"], original[variable]["std.error"])


def _intercept_model():
    model = _model()
    formula = parse_formula("y ~ 1 + (1 | g)")
    matrices = build_model_matrices(formula, model.matrices.frame)
    return replace(model, formula=formula, matrices=matrices, beta=np.array([0.3]))


def test_all_effects_without_predictors_needs_no_covariance(monkeypatch):
    model = _intercept_model()

    def unexpected_covariance(self):
        raise AssertionError("an empty effects collection needs no covariance calculation")

    monkeypatch.setattr(type(model), "vcov", unexpected_covariance)
    assert allEffects(model) == {}


@pytest.mark.parametrize("intercept_only", [False, True])
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"type": "unknown"}, "response.*link"),
        ({"level": np.nan}, "between 0 and 1"),
        ({"n_points": True}, "at least 2"),
        ({"offset": np.inf}, "offset must be finite"),
        ({"at": []}, "at must be a mapping"),
        ({"at": {"missing": 1}}, "Unknown variable.*missing"),
    ],
)
def test_all_effects_validates_options_before_covariance(
    monkeypatch, intercept_only, options, message
):
    model = _intercept_model() if intercept_only else _model()

    def unexpected_covariance(self):
        raise AssertionError("invalid options must fail before covariance calculation")

    monkeypatch.setattr(type(model), "vcov", unexpected_covariance)
    with pytest.raises((TypeError, ValueError), match=message):
        allEffects(model, **options)


def test_all_effects_validates_every_conditioning_grid_before_covariance(monkeypatch):
    model = _model()

    def unexpected_covariance(self):
        raise AssertionError("invalid conditioning must fail before covariance calculation")

    monkeypatch.setattr(type(model), "vcov", unexpected_covariance)
    with pytest.raises(ValueError, match="Non-focal variable 'treatment'.*one value"):
        allEffects(model, at={"treatment": ["high", "low"]})


@pytest.mark.parametrize("function", [ggpredict, allEffects])
def test_invalid_contrast_overrides_fail_before_covariance(monkeypatch, function):
    model = _model()

    def unexpected_covariance(self):
        raise AssertionError("invalid contrast overrides need no covariance calculation")

    monkeypatch.setattr(type(model), "vcov", unexpected_covariance)
    args = ("treatment",) if function is ggpredict else ()
    with pytest.raises(ValueError, match="Unknown contrast type"):
        function(model, *args, contrasts={"treatment": "unknown"})


@pytest.mark.parametrize(
    ("n_points", "expected"),
    [(2, [-1.5, 2.0]), (3, [-1.5, 0.0, 2.0]), (4, [-1.5, 0.0, 2.0])],
)
def test_automatic_numeric_grids_preserve_observed_values_or_use_even_spacing(n_points, expected):
    model = _model()

    result = allEffects(model, n_points=n_points)

    assert_array_equal(result["x"].x, expected)

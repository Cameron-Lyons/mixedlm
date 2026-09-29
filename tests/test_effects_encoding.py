from __future__ import annotations

import itertools

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

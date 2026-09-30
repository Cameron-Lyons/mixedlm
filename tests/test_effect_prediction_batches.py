from __future__ import annotations

import gc
import itertools
import math
import weakref
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, families, ggpredict, parse_formula
from mixedlm.matrices import build_model_matrices
from mixedlm.matrices.design import build_fixed_matrix
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats

effects = import_module("mixedlm.inference.effects")
design = import_module("mixedlm.matrices.design")


def _model(kind="lmm", contrast="treatment", backend="pandas", intercept=True, levels=None):
    levels = ["middle", "low", "high"] if levels is None else levels
    frame = pd.DataFrame(
        itertools.product(levels, ["west", "east"], [-1.0, 0.25, 1.5]),
        columns=["treatment", "site", "x"],
    )
    frame["treatment"] = pd.Categorical(frame.treatment, categories=levels, ordered=True)
    frame["y"] = np.ones(len(frame))
    frame["g"] = np.arange(len(frame)) % 3
    if backend == "polars":
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame(frame.to_dict("list"))
    formula = parse_formula(
        "y ~ "
        + ("" if intercept else "0 + ")
        + "treatment * site + x + I(x**2) + treatment:x + (1 | g)"
    )
    matrices = build_model_matrices(formula, frame, contrasts={"treatment": contrast})
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.3]),
        beta=np.linspace(-0.1, 0.15, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.8, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    rng = np.random.default_rng(622)
    factor = rng.normal(size=(matrices.n_fixed, matrices.n_fixed)) / 20
    covariance = factor @ factor.T
    model.vcov = lambda: covariance
    return model


def _assert_dense_prediction(model, result, grid, *, scale="response", offset=0.4, level=0.9):
    matrix, names = build_fixed_matrix(
        model.formula,
        grid,
        contrasts=model.matrices.contrasts,
        category_levels=model.matrices.category_levels,
    )
    matrix = matrix[:, [names.index(name) for name in model.matrices.fixed_names]]
    eta = matrix @ model.beta + offset
    se = np.sqrt(np.diag(matrix @ model.vcov() @ matrix.T))
    critical = (
        stats.norm.ppf((1 + level) / 2)
        if model.isGLMM()
        else stats.t.ppf((1 + level) / 2, model.df_residual())
    )
    lower, upper = eta - critical * se, eta + critical * se
    if model.isGLMM() and scale == "response":
        eta, lower, upper = np.exp(eta), np.exp(lower), np.exp(upper)
        se *= eta
    assert_allclose(result.predicted, eta, rtol=1e-12, atol=1e-14)
    assert_allclose(result["std.error"], se, rtol=1e-12, atol=1e-14)
    assert_allclose(result["conf.low"], lower, rtol=1e-12, atol=1e-14)
    assert_allclose(result["conf.high"], upper, rtol=1e-12, atol=1e-14)
    assert result.attrs == {
        "type": scale,
        "level": level,
        "offset": offset,
        "adjustment": "numeric means and categorical reference levels",
    }


@pytest.mark.parametrize("limit", [1, 47, 1_000_000])
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("scale", ["link", "response"])
@pytest.mark.parametrize(
    "contrast", ["treatment", "sum", np.array([[0.2, 0.8], [-0.4, 0.3], [0.6, -0.7]])]
)
def test_effect_batches_match_dense_covariance_and_preserve_product_order(
    monkeypatch, limit, kind, scale, contrast
):
    model = _model(kind, contrast)
    original = model.matrices.frame.copy(deep=True)
    monkeypatch.setattr(effects, "_MAX_EFFECT_MATRIX_ELEMENTS", limit, raising=False)
    values = [[0.75, -1.2, 0.75], ["low", "high"], ["west", "east"]]
    terms = ["x", "treatment", "site"]

    actual = ggpredict(
        model,
        terms,
        at=dict(zip(terms, values, strict=True)),
        type=scale,
        offset=0.4,
        level=0.9,
        contrasts=model.matrices.contrasts,
    )

    grid = pd.DataFrame(itertools.product(*values), columns=terms)
    for name, categories in model.matrices.category_levels.items():
        if name in terms:
            grid[name] = pd.Categorical(
                grid[name], categories=categories, ordered=name == "treatment"
            )
    assert_frame_equal(actual[terms], grid)
    _assert_dense_prediction(model, actual, grid, scale=scale)
    assert_frame_equal(model.matrices.frame, original)


@pytest.mark.parametrize("limit", [1, 43, 1_000_000])
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_public_effect_prediction_bounds_matrix_batches_and_reuses_covariance(
    monkeypatch, limit, kind
):
    model = _model(kind)
    original = design.build_fixed_matrix
    calls = []
    covariance_calls = []
    covariance = model.vcov()
    max_rows = max(1, limit // max(len(model.beta), 3))

    def build(formula, frame, **kwargs):
        assert len(frame) <= max_rows
        calls.extend(frame.index.tolist())
        return original(formula, frame, **kwargs)

    def vcov():
        covariance_calls.append(True)
        return covariance

    model.vcov = vcov
    monkeypatch.setattr(effects, "_MAX_EFFECT_MATRIX_ELEMENTS", limit, raising=False)
    monkeypatch.setattr(design, "build_fixed_matrix", build)
    if hasattr(effects, "build_fixed_matrix"):
        monkeypatch.setattr(effects, "build_fixed_matrix", build)

    result = ggpredict(model, ["x", "treatment", "site"], at={"x": np.linspace(-1, 1, 5)})

    assert calls == list(range(30))
    assert covariance_calls == [True]
    assert np.isfinite(result.predicted).all()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("limit", [1, 37])
def test_all_effects_matches_independent_grids_across_batches(monkeypatch, backend, kind, limit):
    model = _model(kind, "sum", backend)
    options = dict(n_points=4, contrasts=model.matrices.contrasts, offset=0.4, level=0.9)
    expected = {name: ggpredict(model, name, **options) for name in ["treatment", "site", "x"]}
    monkeypatch.setattr(effects, "_MAX_EFFECT_MATRIX_ELEMENTS", limit, raising=False)

    actual = allEffects(model, **options)

    assert list(actual) == list(expected)
    for name in actual:
        assert_frame_equal(actual[name], expected[name], rtol=1e-12, atol=1e-14)
        assert actual[name].attrs == expected[name].attrs


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("drop_column", [False, True])
def test_batch_encoding_preserves_retained_columns_and_no_intercept_models(
    monkeypatch, kind, intercept, drop_column
):
    model = _model(kind, "sum", intercept=intercept)
    if drop_column:
        model.matrices.X = model.matrices.X[:, :-1]
        model.matrices.fixed_names = model.matrices.fixed_names[:-1]
        model.matrices.n_fixed -= 1
        model.beta = model.beta[:-1]
        covariance = model.vcov()[:-1, :-1]
        model.vcov = lambda: covariance
    monkeypatch.setattr(effects, "_MAX_EFFECT_MATRIX_ELEMENTS", 1, raising=False)

    result = ggpredict(
        model,
        ["treatment", "site"],
        at={"x": 0.2},
        contrasts=model.matrices.contrasts,
        offset=0.4,
        level=0.9,
    )

    _assert_dense_prediction(model, result, result[["treatment", "site"]].assign(x=0.2))


@pytest.mark.parametrize("levels", [[30, 10, 20], ["third", "first", "second"]])
def test_product_columns_preserve_factor_values_and_declared_order(levels):
    model = _model(levels=levels)

    result = ggpredict(model, "treatment", at={"treatment": [levels[2], levels[0], levels[2]]})

    assert result.treatment.tolist() == [levels[2], levels[0], levels[2]]
    assert result.treatment.cat.categories.tolist() == levels
    assert result.treatment.cat.ordered
    assert_array_equal(result.index, np.arange(3))


@pytest.mark.parametrize("function", [ggpredict, allEffects])
def test_invalid_contrast_fails_before_covariance(monkeypatch, function):
    model = _model()
    monkeypatch.setattr(effects, "_MAX_EFFECT_MATRIX_ELEMENTS", 1, raising=False)

    def unexpected():
        raise AssertionError("Invalid contrasts need no covariance calculation")

    model.vcov = unexpected
    args = ("treatment",) if function is ggpredict else ()
    with pytest.raises(ValueError, match="Unknown contrast type"):
        function(model, *args, contrasts={"treatment": "invalid"})


def test_numeric_product_does_not_materialize_python_row_tuples(monkeypatch):
    model = _model()
    product = itertools.product

    def bounded_product(*iterables, **kwargs):
        # Formula construction may enumerate short tuples of term names.
        if math.prod(map(len, iterables)) > 20:
            raise AssertionError("The effect grid must construct columns directly")
        return product(*iterables, **kwargs)

    monkeypatch.setattr(itertools, "product", bounded_product)
    result = ggpredict(model, ["x", "treatment", "site"], at={"x": np.linspace(-1, 1, 10)})
    assert len(result) == 60


def test_invalid_conditioning_precedes_product_allocation(monkeypatch):
    model = _model()

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid conditioning needs no product grid")

    monkeypatch.setattr(effects, "_cartesian_grid", unexpected, raising=False)
    with pytest.raises(ValueError, match="Non-focal variable 'site'.*one value"):
        ggpredict(model, "treatment", at={"site": ["east", "west"]})


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("limit", [1, 1_000_000])
def test_interaction_buffers_are_released_without_cyclic_garbage_collection(
    monkeypatch, kind, limit
):
    model = _model(kind)
    original = design._encode_interaction
    references = []

    def encode(*args, **kwargs):
        columns, names = original(*args, **kwargs)
        references.extend(weakref.ref(column) for column in columns)
        return columns, names

    monkeypatch.setattr(design, "_encode_interaction", encode)
    monkeypatch.setattr(effects, "_MAX_EFFECT_MATRIX_ELEMENTS", limit, raising=False)
    collecting = gc.isenabled()
    gc.disable()
    try:
        result = ggpredict(model, ["x", "treatment", "site"])
        assert np.isfinite(result.predicted).all()
        assert references
        assert all(reference() is None for reference in references)
    finally:
        if collecting:
            gc.enable()

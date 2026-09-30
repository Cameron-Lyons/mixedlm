from __future__ import annotations

import itertools
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import emmeans, parse_formula
from mixedlm.families import Poisson
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats

module = import_module("mixedlm.inference.emmeans")


def _model(kind="lmm", custom=False, dropped=False):
    levels = {"a": ["C", "A", "B"], "b": [20, 10, 40, 30], "c": ["high", "low"]}
    data = pd.DataFrame(itertools.product(*levels.values(), range(6)), columns=[*levels, "g"])
    for name, categories in levels.items():
        data[name] = pd.Categorical(data[name], categories=categories, ordered=True)
    rng = np.random.default_rng(712)
    data["x"] = rng.normal(size=len(data))
    data["y"] = rng.uniform(0.5, 2.0, size=len(data))
    formula = parse_formula("y ~ a * b + c + x + I(x**2) + (1 | g)")
    contrasts = {"a": "sum", "b": np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])}
    matrices = build_model_matrices(formula, data, contrasts=contrasts if custom else None)
    if dropped:
        keep = np.arange(matrices.n_fixed) % 3 != 2
        matrices = replace(
            matrices,
            X=matrices.X[:, keep],
            fixed_names=[
                name for name, retained in zip(matrices.fixed_names, keep, strict=True) if retained
            ],
            n_fixed=int(keep.sum()),
        )
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.5]),
        beta=np.linspace(-0.3, 0.4, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.8, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=Poisson(), nAGQ=1)
    )
    return model, levels


def _direct_reference(model, specs, levels, at):
    reference = {**levels, "x": [float(np.median(model.matrices.frame.x))], **at}
    grid = pd.DataFrame(itertools.product(*reference.values()), columns=list(reference))
    design = model._prediction_fixed_matrix(grid)
    combinations = list(itertools.product(*(reference[name] for name in specs)))
    result_grid = pd.DataFrame(combinations, columns=specs)
    rows = []
    for values in combinations:
        selected = np.ones(len(grid), dtype=bool)
        for name, value in zip(specs, values, strict=True):
            selected &= np.asarray(grid[name] == value)
        rows.append(design[selected].mean(axis=0))
    return np.asarray(rows), result_grid, grid


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("specs", [["a"], ["c", "a"], ["b", "c", "a"]])
@pytest.mark.parametrize("row_limit", [1, 5, 17, 1_000_000])
def test_streamed_means_preserve_encoding_order_and_uncertainty(
    monkeypatch, kind, custom, specs, row_limit
):
    model, levels = _model(kind, custom)
    expected_L, expected_grid, full_grid = _direct_reference(model, specs, levels, {})
    original_frame = model.matrices.frame.copy(deep=True)
    original_prediction = model._prediction_fixed_matrix
    batches = []

    def predict(grid):
        assert len(grid) <= row_limit
        batches.append(grid.copy())
        return original_prediction(grid)

    budget = row_limit * max(len(model.beta), len(full_grid.columns))
    monkeypatch.setattr(module, "_MAX_REFERENCE_GRID_ELEMENTS", budget, raising=False)
    monkeypatch.setattr(model, "_prediction_fixed_matrix", predict)

    result = emmeans(model, specs, cov_reduce=np.median, level=0.9)

    assert_frame_equal(result.result.grid, expected_grid)
    assert_allclose(result._L, expected_L, rtol=1e-13, atol=1e-13)
    expected = expected_L @ model.beta
    se = np.sqrt(np.diag(expected_L @ model.vcov() @ expected_L.T))
    critical = stats.norm.ppf(0.95) if kind == "glmm" else stats.t.ppf(0.95, model.df_residual())
    lower, upper = expected - critical * se, expected + critical * se
    if kind == "glmm":
        expected = np.exp(expected)
        se *= expected
        lower, upper = np.exp(lower), np.exp(upper)
    assert_allclose(result.result.emmean, expected, rtol=1e-12, atol=1e-13)
    assert_allclose(result.result.se, se, rtol=1e-12, atol=1e-13)
    assert_allclose(result.result.lower, lower, rtol=1e-12, atol=1e-13)
    assert_allclose(result.result.upper, upper, rtol=1e-12, atol=1e-13)
    combined = pd.concat(batches, ignore_index=True)
    assert len(combined) == len(full_grid)
    assert not combined.duplicated().any()
    assert_frame_equal(model.matrices.frame, original_frame)


@pytest.mark.parametrize("dropped", [False, True])
@pytest.mark.parametrize("row_limit", [1, 3, 9])
def test_streamed_subsets_keep_fitted_columns_and_user_level_order(monkeypatch, dropped, row_limit):
    model, levels = _model(custom=True, dropped=dropped)
    at = {"a": ["B", "C"], "b": [30, 10], "x": [0.75]}
    expected_L, expected_grid, _ = _direct_reference(model, ["a"], levels, at)
    budget = row_limit * max(len(model.beta), 4)
    monkeypatch.setattr(module, "_MAX_REFERENCE_GRID_ELEMENTS", budget, raising=False)

    result = emmeans(model, "a", at=at)

    assert_frame_equal(result.result.grid, expected_grid)
    assert_allclose(result._L, expected_L, rtol=1e-13, atol=1e-13)
    assert_allclose(result.result.emmean, expected_L @ model.beta, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("row_limit", [1, 5, 100])
def test_overall_mean_has_one_result_row(monkeypatch, row_limit):
    model, levels = _model(custom=True)
    expected_L, expected_grid, _ = _direct_reference(model, [], levels, {})
    monkeypatch.setattr(
        module, "_MAX_REFERENCE_GRID_ELEMENTS", row_limit * len(model.beta), raising=False
    )

    result = emmeans(model, [], cov_reduce=np.median)

    assert_frame_equal(result.result.grid, expected_grid)
    assert len(result.result.grid) == len(result.result.emmean) == 1
    assert_allclose(result._L, expected_L, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("n_beta", [0, 1, 8])
@pytest.mark.parametrize("limit", [1, 13, 1_000_000])
@pytest.mark.parametrize("specs", [[], ["a"], ["b", "a"]])
def test_batch_budget_handles_narrow_and_empty_designs(monkeypatch, n_beta, limit, specs):
    levels = {"a": [3.0, -1.0, 2.0], "b": [-2.0, 0.5, 1.0, 4.0]}
    observed = []
    max_rows = max(1, limit // max(n_beta, len(levels)))

    def predict(grid):
        assert len(grid) <= max_rows
        observed.append(grid.copy())
        value = grid.a.to_numpy() * grid.b.to_numpy()
        return value[:, None] * np.arange(1, n_beta + 1)[None, :]

    model = SimpleNamespace(beta=np.zeros(n_beta), _prediction_fixed_matrix=predict)
    monkeypatch.setattr(module, "_MAX_REFERENCE_GRID_ELEMENTS", limit)

    coefficients, grid = module._marginal_mean_coefficients(model, specs, levels)

    expected = []
    for index in range(len(grid)):
        row = grid.iloc[index].to_dict()
        a = row.get("a", np.mean(levels["a"]))
        b = row.get("b", np.mean(levels["b"]))
        expected.append(a * b * np.arange(1, n_beta + 1))
    assert_allclose(coefficients, expected, rtol=1e-13, atol=1e-13)
    assert sum(map(len, observed)) == 12
    if max_rows >= 12:
        assert len(observed) == 1


def test_intercept_only_reference_grid_retains_its_single_row():
    model = SimpleNamespace(
        beta=np.array([2.0]), _prediction_fixed_matrix=lambda grid: np.ones((len(grid), 1))
    )

    coefficients, grid = module._marginal_mean_coefficients(model, [], {})

    assert grid.shape == (1, 0)
    assert_array_equal(coefficients, [[1.0]])

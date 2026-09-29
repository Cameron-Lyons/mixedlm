from __future__ import annotations

import itertools
from functools import partial
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.emmeans import EmmeanResult, Emmeans, emmeans
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats

emmeans_module = import_module("mixedlm.inference.emmeans")


def _means(coefficients, covariance, beta):
    labels = [f"L{i}" for i in range(len(coefficients))]
    values = coefficients @ beta
    zeros = np.zeros(len(values))
    return Emmeans(
        result=EmmeanResult(
            emmean=values,
            se=zeros,
            df=80.0,
            lower=zeros,
            upper=zeros,
            grid=pd.DataFrame({"treatment": labels}),
            level=0.95,
        ),
        _L=coefficients,
        _vcov=covariance,
        _beta=beta,
        _df=80.0,
        _specs=["treatment"],
        _levels=[labels],
    )


@pytest.mark.parametrize("kind", ["pairs", "control", "custom"])
@pytest.mark.parametrize("rank", [1, 7])
@pytest.mark.parametrize("limit", [1, 19, 1_000_000])
def test_contrasts_match_direct_covariance_with_bounded_buffers(monkeypatch, kind, rank, limit):
    rng = np.random.default_rng(534)
    n_levels, n_beta = 15, 7
    coefficients = rng.normal(size=(n_levels, n_beta))
    factor = rng.normal(size=(n_beta, rank))
    covariance = factor @ factor.T
    beta = rng.normal(size=n_beta)
    means = _means(coefficients, covariance, beta)
    custom = rng.normal(size=(13, n_levels))
    originals = [array.copy() for array in (coefficients, covariance, beta, custom)]
    if kind == "pairs":
        pairs = list(itertools.combinations(range(n_levels), 2))
        expected_coefficients = np.array([coefficients[i] - coefficients[j] for i, j in pairs])
        expected_labels = [f"L{i} - L{j}" for i, j in pairs]
        evaluate = partial(means.pairs, adjust="none")
    elif kind == "control":
        treatments = [i for i in range(n_levels) if i != 3]
        expected_coefficients = coefficients[treatments] - coefficients[3]
        expected_labels = [f"L{i} - L3" for i in treatments]
        evaluate = partial(means._trt_vs_ctrl, ctrl_idx=3, adjust="none")
    else:
        expected_coefficients = custom @ coefficients
        expected_labels = [f"C{i + 1}" for i in range(len(custom))]
        evaluate = partial(means.contrast, custom)

    original_quadratic_form = emmeans_module._rowwise_quadratic_form
    processed_rows = 0

    def bounded_quadratic_form(chunk, vcov):
        nonlocal processed_rows
        assert chunk.size <= max(limit, n_beta)
        processed_rows += len(chunk)
        return original_quadratic_form(chunk, vcov)

    monkeypatch.setattr(emmeans_module, "_MAX_CONTRAST_ELEMENTS", limit, raising=False)
    monkeypatch.setattr(emmeans_module, "_rowwise_quadratic_form", bounded_quadratic_form)

    actual = evaluate()

    expected_variance = np.diag(expected_coefficients @ covariance @ expected_coefficients.T)
    assert_allclose(actual.estimate, expected_coefficients @ beta, rtol=1e-12, atol=1e-13)
    assert_allclose(actual.se, np.sqrt(np.maximum(expected_variance, 0)), rtol=1e-10, atol=1e-12)
    assert actual.contrast == expected_labels
    assert processed_rows == len(expected_coefficients)
    for array, original in zip((coefficients, covariance, beta, custom), originals, strict=True):
        assert_array_equal(array, original)


def test_pairwise_differences_preserve_small_changes_in_large_common_terms(monkeypatch):
    coefficients = np.full((8, 3), 2.0**45)
    coefficients[:, 0] += np.arange(8) * 0.125
    means = _means(coefficients, np.eye(3), np.array([1.0, 1e6, 1e6]))
    monkeypatch.setattr(emmeans_module, "_MAX_CONTRAST_ELEMENTS", 10, raising=False)

    actual = means.pairs(adjust="none")

    expected = np.array([(i - j) * 0.125 for i, j in itertools.combinations(range(8), 2)])
    assert_array_equal(actual.estimate, expected)
    assert_array_equal(actual.se, np.abs(expected))


def test_empty_custom_contrasts():
    means = _means(np.eye(3), np.eye(3), np.ones(3))

    actual = means.contrast(np.empty((0, 3)))

    assert actual.contrast == []
    for values in (actual.estimate, actual.se, actual.t_ratio, actual.p_value):
        assert values.shape == (0,)


def test_contrasts_without_fixed_coefficients(monkeypatch):
    means = _means(np.empty((3, 0)), np.empty((0, 0)), np.empty(0))
    monkeypatch.setattr(emmeans_module, "_MAX_CONTRAST_ELEMENTS", 1, raising=False)

    with np.errstate(invalid="ignore"):
        actual = means.pairs(adjust="none")

    assert_array_equal(actual.estimate, np.zeros(3))
    assert_array_equal(actual.se, np.zeros(3))


@pytest.fixture(params=[False, True])
def grid_model(request):
    levels = {"a": ["L3", "L1", "L4", "L2"], "b": [20, 10, 30], "c": ["B", "A"]}
    data = pd.DataFrame(itertools.product(*levels.values()), columns=list(levels))
    data = data.loc[data.index.repeat(np.arange(len(data)) % 3 + 1)].reset_index(drop=True)
    for name, categories in levels.items():
        data[name] = pd.Categorical(data[name], categories=categories)
    rng = np.random.default_rng(522)
    data["x"] = rng.normal(size=len(data))
    data["y"] = rng.normal(size=len(data))
    data["group"] = (np.arange(len(data)) % 6).astype(str)
    formula = parse_formula("y ~ a*b + b:c + x + (1 | group)")
    contrasts = {"a": np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])}
    matrices = build_model_matrices(formula, data, contrasts=contrasts if request.param else None)
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.5]),
        beta=rng.normal(size=matrices.n_fixed),
        sigma=0.7,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )
    return model, data, levels


@pytest.mark.parametrize("specs", [["a"], ["b", "a"], ["c", "b", "a"]])
@pytest.mark.parametrize("at", [None, {"b": [30, 10], "x": 0.7}, {"a": ["L2"], "c": ["A"]}])
def test_grid_reduction_matches_explicit_equal_weight_averaging(grid_model, specs, at):
    model, data, all_levels = grid_model
    at = {} if at is None else at
    levels = {name: at.get(name, values) for name, values in all_levels.items()}
    grid = pd.DataFrame(itertools.product(*levels.values()), columns=list(levels))
    grid["x"] = at.get("x", float(np.median(data["x"])))
    expected_design = model._prediction_fixed_matrix(grid)
    combinations = list(itertools.product(*(levels[name] for name in specs)))
    rows = []
    for values in combinations:
        selected = np.ones(len(grid), dtype=bool)
        for name, value in zip(specs, values, strict=True):
            selected &= np.asarray(grid[name] == value)
        rows.append(expected_design[selected].mean(axis=0))
    expected_L = np.array(rows)
    expected_mean = expected_L @ model.beta
    expected_se = np.sqrt(np.diag(expected_L @ model.vcov() @ expected_L.T))

    actual = emmeans(model, specs, at=at, cov_reduce=np.median, level=0.9)

    assert_frame_equal(actual.result.grid, pd.DataFrame(combinations, columns=specs))
    assert_allclose(actual._L, expected_L, rtol=1e-13, atol=1e-13)
    assert_allclose(actual.result.emmean, expected_mean, rtol=1e-13, atol=1e-13)
    assert_allclose(actual.result.se, expected_se, rtol=1e-13, atol=1e-13)
    critical = stats.t.ppf(0.95, model.df_residual())
    assert_allclose(actual.result.lower, expected_mean - critical * expected_se)
    assert_allclose(actual.result.upper, expected_mean + critical * expected_se)

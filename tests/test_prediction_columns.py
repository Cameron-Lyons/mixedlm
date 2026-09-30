import gc
import pickle
import tracemalloc
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, glmerControl, lmer, lmerControl, parse_formula
from mixedlm.matrices.design import build_fixed_matrix, build_model_matrices
from mixedlm.models.checks import run_model_checks
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult


def _data(collision, reduction):
    rng = np.random.default_rng(918)
    n = 120
    data = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "a": pd.Categorical(np.arange(n) % 3, categories=range(3)),
            "g": np.arange(n) % 10,
            "y": np.ones(n),
        }
    )
    name = {"factor": "a.1", "power": "I(x**2)", "interaction": "x:z", "intercept": "(Intercept)"}[
        collision
    ]
    data[name] = rng.normal(size=n)
    if reduction == "zero":
        data[name] = 0.0
    elif reduction == "constant":
        data[name] = 2.0
    rhs = {"factor": "a", "power": "I(x**2)", "interaction": "x*z", "intercept": "x"}[collision]
    formula = parse_formula(f"y ~ {rhs} + `{name}` + (1 | g)")
    return formula, data, name


def _model(kind="lmm", collision="factor", reduction="none"):
    formula, data, name = _data(collision, reduction)
    matrices = build_model_matrices(formula, data)
    width = matrices.n_fixed
    control = lmerControl(check_rankX="message+drop.cols", check_singular=False)
    matrices, dropped = run_model_checks(matrices, control)
    retained = [index for index in range(width) if index not in (dropped or [])]
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.3]),
        beta=np.linspace(0.2, 0.8, matrices.n_fixed),
        u=np.linspace(-0.1, 0.1, matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.7, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    return model, data, name, retained


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("collision", ["factor", "power", "interaction", "intercept"])
@pytest.mark.parametrize("reduction", ["none", "zero", "constant"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_prediction_keeps_distinct_columns_with_the_same_display_name(
    kind, collision, reduction, backend
):
    model, data, name, retained = _model(kind, collision, reduction)
    newdata = data.iloc[[0, 1, 5, 9, 14]].copy()
    newdata[name] = [-2.0, 3.0, 1.0, -1.0, 4.0]
    complete, _ = build_fixed_matrix(
        model.formula, newdata, category_levels=model.matrices.category_levels
    )
    expected_matrix = complete[:, retained]
    expected_link = expected_matrix @ model.beta
    if backend == "polars":
        pl = pytest.importorskip("polars")
        newdata = pl.DataFrame({column: newdata[column].to_numpy() for column in newdata})
    np.testing.assert_array_equal(model._prediction_fixed_matrix(newdata), expected_matrix)
    result = model.predict(newdata, re_form="NA", se_fit=True, interval="confidence", level=0.9)
    expected = expected_link if kind == "lmm" else np.exp(expected_link)
    np.testing.assert_allclose(result.fit, expected, rtol=1e-12, atol=1e-12)
    variance = np.einsum("ij,jk,ik->i", expected_matrix, model.vcov(), expected_matrix)
    expected_se = np.sqrt(np.maximum(variance, 0))
    if kind == "glmm":
        expected_se *= expected
    np.testing.assert_allclose(result.se_fit, expected_se, rtol=1e-11, atol=1e-12)
    assert np.all(result.lower <= result.fit)
    assert np.all(result.upper >= result.fit)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("reduction", ["none", "zero", "constant"])
def test_fitted_prediction_matches_training_design_and_survives_refit(kind, reduction):
    formula, data, _ = _data("factor", reduction)
    rng = np.random.default_rng(927)
    linear = (
        1.0
        + 0.7 * (data["a"].astype(int) == 1)
        - 0.4 * (data["a"].astype(int) == 2)
        + 0.3 * data["a.1"]
    )
    data["y"] = (
        linear + rng.normal(scale=0.1, size=len(data))
        if kind == "lmm"
        else rng.poisson(np.exp(linear * 0.5))
    )
    if kind == "lmm":
        model = lmer(
            formula,
            data,
            control=lmerControl(check_rankX="message+drop.cols", check_singular=False),
        )
    else:
        model = glmer(
            formula,
            data,
            family=families.Poisson(),
            control=glmerControl(check_rankX="message+drop.cols", check_singular=False),
        )
    for result in [model, model.refit(newresp=data["y"].to_numpy())]:
        link = result.matrices.X @ result.beta
        expected = link if kind == "lmm" else np.exp(link)
        np.testing.assert_allclose(
            result.predict(data, re_form="NA"), expected, rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(result.predict(data), result.fitted(), rtol=1e-11, atol=1e-12)
        assert result.matrices.fixed_source_names == model.matrices.fixed_source_names
        assert result.matrices.fixed_column_indices == model.matrices.fixed_column_indices


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_fitted_positions_survive_pickling_and_response_replacement(kind):
    model, data, _, retained = _model(kind, reduction="zero")
    assert model.matrices.fixed_column_indices == tuple(retained)
    clone = model._clone_matrices_with_response_base(np.full(len(data), 2.0))
    assert clone is not model.matrices
    assert clone.fixed_column_indices == model.matrices.fixed_column_indices
    assert clone.fixed_source_names == model.matrices.fixed_source_names
    assert clone.X is model.matrices.X
    assert clone.Z is model.matrices.Z
    np.testing.assert_array_equal(model.matrices.y, np.ones(len(data)))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.predict(data, re_form="NA"), model.predict(data, re_form="NA")
    )


def test_successive_rank_reductions_compose_original_positions():
    model, _, _, _ = _model(reduction="constant")
    original = model.matrices
    assert original.fixed_column_indices == (1, 2, 3)
    second_X = original.X.copy()
    second_X[:, 0] = 0
    reduced, dropped = run_model_checks(
        replace(original, X=second_X), lmerControl(check_rankX="message+drop.cols")
    )
    assert dropped == [0]
    assert reduced.fixed_column_indices == (2, 3)
    assert reduced.fixed_source_names == original.fixed_source_names
    checked, dropped = run_model_checks(reduced, lmerControl(check_rankX="message+drop.cols"))
    assert dropped is None
    assert checked.fixed_column_indices == (2, 3)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_legacy_ambiguous_reduced_schema_requires_refitting(kind):
    model, data, _, _ = _model(kind, reduction="zero")
    model.matrices = replace(model.matrices, fixed_source_names=None, fixed_column_indices=None)
    with pytest.raises(ValueError, match="ambiguous fixed-effect column name 'a.1'.*Refit"):
        model.predict(data, re_form="NA")


def test_legacy_unique_names_still_support_reordered_and_dropped_columns():
    model, data, _, _ = _model()
    model.formula = parse_formula("y ~ x + z + (1 | g)")
    matrices = build_model_matrices(model.formula, data)
    model.matrices = replace(
        matrices, X=matrices.X[:, [2, 0]], fixed_names=["z", "(Intercept)"], n_fixed=2
    )
    model.beta = np.array([2.0, 3.0])
    np.testing.assert_allclose(model.predict(data, re_form="NA"), 2 * data["z"] + 3)


def test_missing_fitted_column_still_has_a_clear_error():
    model, data, _, _ = _model()
    model.matrices = replace(model.matrices, fixed_names=["missing"])
    with pytest.raises(ValueError, match="missing fitted fixed-effect column 'missing'"):
        model.predict(data, re_form="NA")


@pytest.mark.parametrize("positions", [(-1, 2, 3), (1, 2, 4), (0, 1, 2)])
def test_inconsistent_stored_positions_are_rejected(positions):
    model, data, _, _ = _model(reduction="constant")
    model.matrices = replace(model.matrices, fixed_column_indices=positions)
    with pytest.raises(ValueError, match="column positions do not match"):
        model.predict(data, re_form="NA")


def test_aligned_matrix_is_independent_of_prediction_and_training_frames():
    model, data, _, _ = _model()
    expected_frame = data.copy(deep=True)
    expected_training = model.matrices.X.copy()
    matrix = model._prediction_fixed_matrix(data)
    matrix[:] = -999
    pd.testing.assert_frame_equal(data, expected_frame)
    np.testing.assert_array_equal(model.matrices.X, expected_training)


def test_prediction_alignment_does_not_allocate_a_second_full_numeric_matrix():
    rng = np.random.default_rng(338)
    data = pd.DataFrame(rng.normal(size=(4_000, 64)), columns=[f"x{i}" for i in range(64)])
    formula = parse_formula("y ~ " + " + ".join(data.columns))
    training = data.iloc[:160].copy()
    training["y"] = 1.0
    matrices = build_model_matrices(formula, training)
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.empty(0),
        beta=np.ones(65),
        u=np.empty(0),
        sigma=0.7,
        REML=True,
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model.predict(data.iloc[:1], re_form="NA")
    gc.collect()
    tracemalloc.start()
    try:
        predicted = model.predict(data, re_form="NA")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_allclose(predicted, 1 + data.to_numpy().sum(axis=1), rtol=1e-12, atol=1e-12)
    assert peak < len(data) * 65 * 8 * 1.6


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("n_rows", [0, 5])
def test_rank_reduction_can_retain_no_fixed_columns(kind, n_rows):
    data = pd.DataFrame({"y": np.ones(60), "x": np.zeros(60), "g": np.arange(60) % 10})
    formula = parse_formula("y ~ 0 + x + (1 | g)")
    matrices = build_model_matrices(formula, data)
    matrices, dropped = run_model_checks(matrices, lmerControl(check_rankX="message+drop.cols"))
    assert dropped == [0]
    assert matrices.fixed_column_indices == ()
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.3]),
        beta=np.empty(0),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.7, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    newdata = pd.DataFrame({"x": np.arange(n_rows, dtype=float)})
    assert model._prediction_fixed_matrix(newdata).shape == (n_rows, 0)
    np.testing.assert_array_equal(
        model.predict(newdata, re_form="NA"), np.full(n_rows, 0 if kind == "lmm" else 1)
    )


def test_unique_generated_column_cannot_fill_two_distinct_fitted_positions():
    model, data, _, _ = _model()
    model.formula = parse_formula("y ~ x + z + (1 | g)")
    matrices = build_model_matrices(model.formula, data)
    model.matrices = replace(matrices, X=matrices.X[:, [1, 2]], fixed_names=["x", "x"], n_fixed=2)
    model.beta = np.ones(2)
    with pytest.raises(ValueError, match="ambiguous fixed-effect column name 'x'"):
        model.predict(data, re_form="NA")

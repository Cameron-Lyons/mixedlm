"""Prediction uncertainty agrees with an independent joint working Hessian."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, parse_formula, set_cov_type
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from scipy import linalg, sparse, stats


def _model(family, *, weighted=False, crossed=False, structured=False):
    x = np.tile([-0.8, -0.1, 0.5, 1.2], 6)
    frame = pd.DataFrame(
        {"y": np.ones(24), "x": x, "g": np.repeat(np.arange(6), 4), "h": np.tile(range(4), 6)}
    )
    formula = parse_formula("y ~ x + (x | g)" + (" + (1 | h)" if crossed else ""))
    formula = set_cov_type(formula, {"g": "ar1"}) if structured else formula
    weights = np.geomspace(0.2, 4, len(frame)) if weighted else None
    matrices = build_model_matrices(formula, frame, weights=weights)
    theta = np.array([0.9, 0.4] if structured else [0.9, 0.25, 0.6])
    if crossed:
        theta = np.append(theta, 0.5)
    model = GlmerResult(
        formula=formula,
        matrices=matrices,
        family=family,
        theta=theta,
        beta=np.array([0.2, 0.3]),
        u=np.linspace(-0.2, 0.25, matrices.n_random),
        deviance=0,
        converged=True,
        n_iter=0,
        nAGQ=1,
    )
    # Build factors independently from the documented covariance parametrization.
    if structured:
        rho = 0.4
        factor = linalg.cholesky(0.9**2 * np.array([[1, rho], [rho, 1]]), lower=True)
    else:
        factor = np.array([[0.9, 0], [0.25, 0.6]])
    blocks = [factor] * 6
    if crossed:
        blocks.extend([np.array([[0.5]])] * 4)
    return model, linalg.block_diag(*blocks)


def _covariance(model, factor):
    matrices = model.matrices
    eta = matrices.X @ model.beta + matrices.Z @ model.u + matrices.offset
    if isinstance(model.family, families.Poisson):
        working_weights = np.exp(eta)
    elif isinstance(model.family, families.Binomial):
        probability = 1 / (1 + np.exp(-eta))
        working_weights = probability * (1 - probability)
    else:
        working_weights = np.ones(len(eta))
    working_weights *= matrices.weights
    design = np.column_stack([matrices.X, matrices.Z @ factor])
    penalty = np.diag(np.r_[np.zeros(matrices.n_fixed), np.ones(matrices.n_random)])
    return linalg.inv(design.T @ (working_weights[:, None] * design) + penalty)


def _known_design(model, frame):
    """Direct row construction in the fitted coefficient order, without alignment helpers."""
    random = np.zeros((len(frame), model.matrices.n_random))
    offset = 0
    for structure in model.matrices.random_structures:
        for row, values in enumerate(frame.itertuples(index=False)):
            level = structure.level_map.get(getattr(values, structure.grouping_factor))
            if level is not None:
                random[row, offset + level * structure.n_terms] = 1
                if structure.n_terms == 2:
                    random[row, offset + level * structure.n_terms + 1] = values.x
        offset += structure.n_levels * structure.n_terms
    return random


@pytest.fixture(params=["dense", "sparse"])
def projection_backend(request, monkeypatch):
    from mixedlm.models import shared_utils

    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if request.param == "sparse" else np.inf
    )


@pytest.mark.parametrize("family", [families.Poisson(), families.Binomial(), families.Gaussian()])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("crossed", [False, True])
def test_conditional_uncertainty_matches_joint_working_hessian(
    family, weighted, crossed, projection_backend
):
    model, factor = _model(family, weighted=weighted, crossed=crossed)
    covariance = _covariance(model, factor)
    design = np.column_stack([model.matrices.X, model.matrices.Z @ factor])
    expected = np.einsum("ij,jk,ik->i", design, covariance, design)

    prediction = model.predict(type="link", se_fit=True)

    np.testing.assert_allclose(prediction.fit, model.linear_predictor())
    np.testing.assert_allclose(prediction.se_fit**2, expected, rtol=1e-11, atol=1e-12)
    # Reconstructing the training rows must produce exactly the same uncertainty.
    rebuilt = model.predict(model.matrices.frame, type="link", se_fit=True)
    np.testing.assert_allclose(rebuilt.fit, prediction.fit, atol=1e-14)
    np.testing.assert_allclose(rebuilt.se_fit, prediction.se_fit, atol=1e-14)


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("family", [families.Poisson(), families.Binomial()])
def test_known_and_new_level_variance_include_cross_terms_and_prior(
    family, structured, projection_backend
):
    model, factor = _model(family, weighted=True, crossed=True, structured=structured)
    covariance = _covariance(model, factor)
    frame = pd.DataFrame({"x": [0.7, -0.9, 1.4, 0.2], "g": [0, 80, 2, 90], "h": [99, 1, 2, 99]})
    fixed = np.column_stack([np.ones(len(frame)), frame.x])
    random = _known_design(model, frame)
    design = np.column_stack([fixed, random @ factor])
    expected = np.einsum("ij,jk,ik->i", design, covariance, design)
    prior_covariance = factor[:2, :2] @ factor[:2, :2].T
    for row, values in enumerate(frame.itertuples(index=False)):
        if values.g not in model.matrices.random_structures[0].level_map:
            terms = np.array([1, values.x])
            expected[row] += terms @ prior_covariance @ terms
        if values.h not in model.matrices.random_structures[1].level_map:
            expected[row] += 0.5**2

    prediction = model.predict(frame, allow_new_levels=True, type="link", interval="confidence")

    np.testing.assert_allclose(prediction.fit, fixed @ model.beta + random @ model.u)
    np.testing.assert_allclose(prediction.se_fit**2, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        prediction.lower, prediction.fit - stats.norm.ppf(0.975) * np.sqrt(expected)
    )
    np.testing.assert_allclose(
        prediction.upper, prediction.fit + stats.norm.ppf(0.975) * np.sqrt(expected)
    )
    with pytest.raises(ValueError, match="New level"):
        model.predict(frame, type="link", se_fit=True)


@pytest.mark.parametrize("re_form", ["NA", "~0"])
def test_fixed_only_prediction_uses_marginal_fixed_covariance(re_form, projection_backend):
    model, factor = _model(families.Poisson(), weighted=True)
    fixed = np.array([[1, 0.9], [1, -1.2]])
    covariance = _covariance(model, factor)[:2, :2]
    prediction = model.predict(pd.DataFrame({"x": fixed[:, 1]}), re_form=re_form, se_fit=True)
    means = np.exp(fixed @ model.beta)
    expected = np.einsum("ij,jk,ik->i", fixed, covariance, fixed)

    np.testing.assert_allclose(prediction.fit, means)
    np.testing.assert_allclose(prediction.se_fit**2, means**2 * expected, rtol=1e-11)


def test_random_only_glmm_has_positive_conditional_and_new_level_uncertainty():
    frame = pd.DataFrame({"y": [1, 2, 1, 3, 2, 1], "g": [0, 0, 0, 1, 1, 1]})
    formula = parse_formula("y ~ 0 + (1 | g)")
    matrices = build_model_matrices(formula, frame)
    model = GlmerResult(
        formula=formula,
        matrices=matrices,
        family=families.Poisson(),
        theta=np.array([0.8]),
        beta=np.empty(0),
        u=np.array([0.2, -0.1]),
        deviance=0,
        converged=True,
        n_iter=0,
        nAGQ=1,
    )
    prediction = model.predict(
        pd.DataFrame({"g": [0, 99]}), type="link", allow_new_levels=True, se_fit=True
    )
    expected = [0.8**2 / (1 + 3 * np.exp(0.2) * 0.8**2), 0.8**2]
    np.testing.assert_allclose(prediction.fit, [0.2, 0])
    np.testing.assert_allclose(prediction.se_fit**2, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_colliding_random_column_names_retain_position_in_mean_and_variance(kind):
    frame = pd.DataFrame(
        {
            "y": np.ones(36),
            "a": np.tile(["a", "b", "c"], 12),
            "a.1": np.linspace(-1, 1, 36),
            "g": np.repeat(range(6), 6),
        }
    )
    formula = parse_formula("y ~ 1 + (a + `a.1` | g)")
    matrices = build_model_matrices(formula, frame)
    assert matrices.random_structures[0].term_names == ["(Intercept)", "a.1", "a.2", "a.1"]
    level_factor = np.array(
        [[0.8, 0, 0, 0], [0.1, 0.6, 0, 0], [-0.2, 0.15, 0.5, 0], [0.2, 0.1, -0.1, 0.7]]
    )
    # Theta packs the lower triangle in row-major order.
    theta = level_factor[np.tril_indices(4)]
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=theta,
        beta=np.array([0.2]),
        u=np.linspace(-0.4, 0.5, matrices.n_random),
        deviance=0,
        converged=True,
        n_iter=0,
    )
    model = (
        GlmerResult(**common, family=families.Poisson(), nAGQ=1)
        if kind == "glmm"
        else LmerResult(**common, sigma=1, REML=True)
    )
    factor = linalg.block_diag(*([level_factor] * 6))
    design = np.column_stack([matrices.X, matrices.Z @ factor])
    weights = (
        np.exp(matrices.X @ model.beta + matrices.Z @ model.u) if kind == "glmm" else np.ones(36)
    )
    covariance = linalg.inv(design.T @ (weights[:, None] * design) + np.diag(np.r_[0, np.ones(24)]))
    expected = np.einsum("ij,jk,ik->i", design, covariance, design)

    prediction = model.predict(frame, se_fit=True, **({"type": "link"} if kind == "glmm" else {}))

    np.testing.assert_allclose(prediction.fit, matrices.X @ model.beta + matrices.Z @ model.u)
    np.testing.assert_allclose(prediction.se_fit**2, expected, rtol=1e-11, atol=1e-12)


def test_conditional_prediction_reuses_factor_and_encodes_random_terms_once(monkeypatch):
    from mixedlm.matrices import design
    from mixedlm.models import shared_utils

    model, _ = _model(families.Binomial())
    model.vcov()

    def reject_refactor(*args, **kwargs):
        raise AssertionError("prediction must reuse the fitted working factorization")

    original = design._random_term_columns
    calls = []

    def count_encodings(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_utils._RandomEffectFactor, "__init__", reject_refactor)
    monkeypatch.setattr(design, "_random_term_columns", count_encodings)
    first = model.predict(model.matrices.frame, se_fit=True)
    second = model.predict(model.matrices.frame, interval="confidence")

    assert len(calls) == 2
    np.testing.assert_array_equal(first.fit, second.fit)
    np.testing.assert_array_equal(first.se_fit, second.se_fit)


def test_large_sparse_prediction_does_not_materialize_random_precision(monkeypatch):
    frame = pd.DataFrame({"y": np.ones(800), "g": np.repeat(range(400), 2)})
    formula = parse_formula("y ~ 1 + (1 | g)")
    matrices = build_model_matrices(formula, frame)
    model = GlmerResult(
        formula=formula,
        matrices=matrices,
        family=families.Poisson(),
        theta=np.array([0.7]),
        beta=np.array([0.4]),
        u=np.zeros(400),
        deviance=0,
        converged=True,
        n_iter=0,
        nAGQ=1,
    )
    original = sparse.csc_matrix.toarray

    def reject_dense_precision(matrix, *args, **kwargs):
        if matrix.shape == (400, 400):
            raise AssertionError("sparse random precision must stay sparse")
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense_precision)
    prediction = model.predict(
        pd.DataFrame({"g": [0, 100, 900]}), allow_new_levels=True, type="link", se_fit=True
    )
    tau2 = 0.7**2
    noise = 1 / (2 * np.exp(0.4))
    fixed_var = (tau2 + noise) / 400
    shrinkage = tau2 / (tau2 + noise)
    expected_known = (1 - shrinkage) ** 2 * fixed_var + tau2 * noise / (tau2 + noise)
    np.testing.assert_allclose(
        prediction.se_fit**2, [expected_known, expected_known, fixed_var + tau2], rtol=1e-11
    )


def test_fitted_binomial_model_uncertainty_matches_joint_curvature():
    from mixedlm.datasets import load_cbpp

    frame = load_cbpp()
    model = glmer("incidence / size ~ period + (1 | herd)", frame, family=families.Binomial())
    matrices = model.matrices
    factor = model.theta[0] * np.eye(matrices.n_random)
    covariance = _covariance(model, factor)
    design = np.column_stack([matrices.X, matrices.Z @ factor])
    expected = np.einsum("ij,jk,ik->i", design, covariance, design)

    prediction = model.predict(type="link", interval="confidence")
    response = model.predict(interval="confidence")

    np.testing.assert_allclose(prediction.se_fit**2, expected, rtol=1e-10)
    np.testing.assert_allclose(
        response.se_fit, prediction.se_fit * response.fit * (1 - response.fit)
    )
    np.testing.assert_allclose(response.lower, 1 / (1 + np.exp(-prediction.lower)))
    np.testing.assert_allclose(response.upper, 1 / (1 + np.exp(-prediction.upper)))


def test_zero_random_covariance_reduces_to_fixed_only_uncertainty(projection_backend):
    model, _ = _model(families.Binomial(), weighted=True)
    model.theta[:] = 0
    model.u[:] = 0
    frame = pd.DataFrame({"x": [0.4, -0.7], "g": [0, 90]})

    conditional = model.predict(frame, allow_new_levels=True, type="link", se_fit=True)
    fixed = model.predict(frame, re_form="NA", type="link", se_fit=True)

    np.testing.assert_array_equal(conditional.fit, fixed.fit)
    np.testing.assert_array_equal(conditional.se_fit, fixed.se_fit)


@pytest.mark.parametrize("backend", ["pandas", "polars", "lazy"])
def test_prediction_backends_preserve_known_and_new_level_uncertainty(backend):
    model, _ = _model(families.Binomial(), weighted=True, crossed=True)
    frame = pd.DataFrame({"x": [0.7, -0.9, 1.4], "g": [0, 80, 2], "h": [99, 1, 2]})
    expected = model.predict(frame, allow_new_levels=True, interval="confidence")
    if backend != "pandas":
        pl = pytest.importorskip("polars")
        frame = pl.from_pandas(frame)
        if backend == "lazy":
            frame = frame.lazy()

    actual = model.predict(frame, allow_new_levels=True, interval="confidence")

    for name in ("fit", "se_fit", "lower", "upper"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("allow_new_levels", [False, True])
@pytest.mark.parametrize("se_fit", [False, True])
def test_missing_groups_fail_with_a_consistent_informative_error(kind, allow_new_levels, se_fit):
    model, _ = _model(families.Poisson())
    if kind == "lmm":
        model = LmerResult(
            formula=model.formula,
            matrices=model.matrices,
            theta=model.theta,
            beta=model.beta,
            u=model.u,
            sigma=1,
            REML=True,
            deviance=0,
            converged=True,
            n_iter=0,
        )
    with pytest.raises(ValueError, match="missing values in grouping factor.*'g'"):
        model.predict(
            pd.DataFrame({"x": [0.4], "g": [None]}),
            allow_new_levels=allow_new_levels,
            se_fit=se_fit,
        )


def test_sparse_prior_projection_is_bounded_and_matches_dense_covariance(monkeypatch):
    from mixedlm.models import shared_utils

    rng = np.random.default_rng(27391)
    design = sparse.random(17, 4, density=0.4, random_state=rng, format="csr")
    factor = np.tril(rng.normal(size=(4, 4)))
    factor[:, 2] = 0  # Boundary covariance also has to work without an inverse.
    dense = design.toarray()
    expected = np.einsum("ij,jk,ik->i", dense, factor @ factor.T, dense)
    monkeypatch.setattr(shared_utils, "_MAX_QUADRATIC_FORM_ELEMENTS", 8)
    original = sparse.csr_matrix.__matmul__
    chunk_rows = []

    def check_buffer(matrix, right):
        chunk_rows.append(matrix.shape[0])
        assert matrix.shape[0] * matrix.shape[1] <= 8
        return original(matrix, right)

    monkeypatch.setattr(sparse.csr_matrix, "__matmul__", check_buffer)
    actual = shared_utils.sparse_covariance_factor_diagonal(design, factor)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert sum(chunk_rows) == len(expected)


def test_empty_conditional_prediction_returns_empty_interval_arrays():
    model, _ = _model(families.Poisson())

    prediction = model.predict(pd.DataFrame({"x": [], "g": []}), interval="confidence")

    for name in ("fit", "se_fit", "lower", "upper"):
        assert getattr(prediction, name).shape == (0,)

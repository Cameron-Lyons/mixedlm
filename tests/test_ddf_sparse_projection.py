from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import set_cov_type
from mixedlm.estimation import reml
from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.ddf import _vcov_from_theta, _weighted_crossproducts, _xt_vinv_x_from_theta
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models import shared_utils
from mixedlm.models.lmer import LmerResult
from mixedlm.models.shared_utils import _RandomEffectFactor
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, sparse


def _result(kind="intercept", n_groups=6):
    rng = np.random.default_rng(826)
    n = 6 * n_groups
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "group": np.arange(n) % n_groups,
            "other": np.arange(n) % 4,
        }
    )
    formulas = {
        "intercept": ("y ~ x + z + (1 | group)", [0.6]),
        "zero": ("y ~ x + z + (1 | group)", [0.0]),
        "slopes": ("y ~ x + z + (1 + x + z | group)", [0.6, 0.1, 0.5, -0.12, 0.08, 0.7]),
        "diagonal": ("y ~ x + z + (1 + x + z || group)", [0.6, 0.5, 0.7]),
        "cs": ("y ~ x + z + (1 + x + z | group)", [0.6, 0.2]),
        "ar1": ("y ~ x + z + (1 + x + z | group)", [0.6, 0.2]),
        "crossed": ("y ~ x + z + (1 | group) + (1 | other)", [0.6, 0.3]),
        "no_random": ("y ~ x + z", []),
        "no_fixed": ("y ~ 0 + (1 | group)", [0.6]),
    }
    text, theta = formulas[kind]
    formula = parse_formula(text)
    if kind in ("cs", "ar1"):
        formula = set_cov_type(formula, kind)
    matrices = build_model_matrices(formula, data, weights=np.linspace(0.4, 2.0, n))
    if kind == "no_fixed":
        matrices = replace(matrices, X=np.empty((n, 0)), fixed_names=[], n_fixed=0)
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array(theta),
        beta=np.zeros(matrices.n_fixed),
        sigma=0.7,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )


def _direct_information(result, sigma):
    matrices = result.matrices
    factor = _build_lambda(result.theta, matrices.random_structures).toarray()
    random_design = matrices.Z @ factor
    marginal_covariance = np.diag(1.0 / matrices.weights) + random_design @ random_design.T
    return matrices.X.T @ linalg.solve(marginal_covariance, matrices.X, assume_a="pos") / sigma**2


@pytest.mark.parametrize(
    "kind", ["intercept", "zero", "slopes", "diagonal", "cs", "ar1", "crossed"]
)
@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("sigma", [None, 1.6])
def test_information_matches_weighted_marginal_covariance(
    kind, backend, cached, sigma, monkeypatch
):
    result = _result(kind)
    expected = _direct_information(result, result.sigma if sigma is None else sigma)
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    crossproducts = _weighted_crossproducts(result) if cached else None

    actual = _xt_vinv_x_from_theta(result, result.theta, crossproducts, sigma=sigma)

    assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert_array_equal(actual, actual.T)


def _intercept_information(result):
    matrices = result.matrices
    groups = np.asarray(matrices.frame["group"])
    group_weights = np.bincount(groups, weights=matrices.weights)
    group_sums = np.column_stack(
        [np.bincount(groups, weights=matrices.weights * column) for column in matrices.X.T]
    )
    multiplier = result.theta[0] ** 2 / (1.0 + result.theta[0] ** 2 * group_weights)
    return (
        matrices.X.T @ (matrices.weights[:, None] * matrices.X)
        - group_sums.T @ (multiplier[:, None] * group_sums)
    ) / result.sigma**2


def _forbid_dense_precision(monkeypatch, q):
    for cls in (sparse.csc_matrix, sparse.csr_matrix):
        original = cls.toarray

        def guarded(matrix, *args, original=original, **kwargs):
            assert matrix.shape != (q, q), "random-effect precision must remain sparse"
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(cls, "toarray", guarded)


@pytest.mark.parametrize("n_groups", [255, 256, 300])
def test_information_uses_sparse_factor_at_size_boundary(n_groups, monkeypatch):
    result = _result(n_groups=n_groups)
    expected = _intercept_information(result)
    if n_groups >= 256:
        _forbid_dense_precision(monkeypatch, n_groups)

    actual = _xt_vinv_x_from_theta(result, result.theta)

    assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("cached", [False, True])
def test_no_random_effects_bypasses_factorization(cached, monkeypatch):
    result = _result("no_random")
    expected = _direct_information(result, result.sigma)
    crossproducts = _weighted_crossproducts(result) if cached else None

    def unexpected_factor(*args, **kwargs):
        raise AssertionError("no random effects should require no precision factor")

    monkeypatch.setattr(_RandomEffectFactor, "_factorize", unexpected_factor)

    actual = _xt_vinv_x_from_theta(result, result.theta, crossproducts)

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_empty_fixed_information(backend, monkeypatch):
    result = _result("no_fixed")
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )

    actual = _xt_vinv_x_from_theta(result, result.theta)

    assert_array_equal(actual, np.empty((0, 0)))


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("n_columns", [0, 1, 3])
def test_factor_crossproduct_matches_direct_solve(backend, n_columns, monkeypatch):
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    rng = np.random.default_rng(448)
    root = rng.normal(size=(9, 9))
    precision = root @ root.T + np.eye(9)
    rhs = rng.normal(size=(9, n_columns))
    original_rhs = rhs.copy()
    expected = rhs.T @ linalg.solve(precision, rhs, assume_a="pos")
    factor = _RandomEffectFactor(sparse.csc_matrix(precision))
    triangular_calls = []
    original_triangular = linalg.solve_triangular

    def counted(*args, **kwargs):
        triangular_calls.append(kwargs.get("lower", False))
        return original_triangular(*args, **kwargs)

    monkeypatch.setattr(linalg, "solve_triangular", counted)

    actual = factor.crossproduct(rhs)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert_array_equal(rhs, original_rhs)
    assert triangular_calls == ([True] if backend == "dense" else [])


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_information_retains_regularization_fallback(backend, monkeypatch):
    result = _result()
    crossproducts = _weighted_crossproducts(result)
    covariance_factor = _build_lambda(result.theta, result.matrices.random_structures)
    q = result.matrices.n_random
    precision = (covariance_factor.T @ crossproducts.ZtWZ @ covariance_factor).toarray()
    precision += (1.0 + 1e-6) * np.eye(q)
    rhs = covariance_factor.T @ crossproducts.ZtWX
    expected = (crossproducts.XtWX - rhs.T @ linalg.solve(precision, rhs)) / result.sigma**2
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    if backend == "dense":
        target, name, failure = linalg, "cholesky", linalg.LinAlgError("not positive definite")
    elif shared_utils._HAS_RUST:
        target, name = shared_utils._SparseCholeskyPattern, "factor"
        failure = ValueError("not positive definite")
    else:
        target, name, failure = sparse.linalg, "splu", RuntimeError("singular precision")
    original = getattr(target, name)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, fail_once)

    actual = _xt_vinv_x_from_theta(result, result.theta, crossproducts)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert calls == 2


def test_covariance_fallback_avoids_dense_random_precision(monkeypatch):
    result = _result(n_groups=300)
    expected = linalg.inv(_intercept_information(result))
    evaluator = reml.LMMOptimizer(result.matrices, REML=result.REML)
    monkeypatch.setattr(evaluator, "_evaluate_core", lambda theta: None)
    _forbid_dense_precision(monkeypatch, result.matrices.n_random)

    actual = _vcov_from_theta(result, result.theta, evaluator)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

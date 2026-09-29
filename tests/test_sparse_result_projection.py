from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation.reml import _build_lambda
from mixedlm.families import Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.profile import _ProfileProjection
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models import shared_utils
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from mixedlm.models.shared_utils import _RandomEffectFactor
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, sparse


def _result(kind, n_groups=8):
    n = 4 * n_groups
    data = pd.DataFrame({"y": np.linspace(0.0, 2.0, n), "group": np.arange(n) % n_groups})
    formula = parse_formula("y ~ 1 + (1 | group)")
    matrices = build_model_matrices(formula, data)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.8]),
        beta=np.array([0.3]),
        u=np.zeros(n_groups),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    if kind == "lmm":
        return LmerResult(**common, sigma=0.7, REML=True)
    return GlmerResult(**common, family=Poisson(), nAGQ=1)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_large_projection_avoids_dense_random_precision_and_reuses_factor(kind, monkeypatch):
    result = _result(kind, n_groups=300)
    q = result.matrices.n_random
    calls = []
    original_splu = sparse.linalg.splu

    def counted_splu(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return original_splu(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.linalg, "splu", counted_splu)
    monkeypatch.setattr(shared_utils, "_MAX_QUADRATIC_FORM_ELEMENTS", 2 * q)
    for cls in (sparse.csc_matrix, sparse.csr_matrix):
        original_toarray = cls.toarray

        def guarded_toarray(matrix, *args, original=original_toarray, **kwargs):
            assert matrix.shape != (q, q), "random precision should stay sparse"
            assert np.prod(matrix.shape) <= 2 * q, "solve buffers should stay bounded"
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(cls, "toarray", guarded_toarray)

    weight = 1.0 if kind == "lmm" else np.exp(0.3)
    scale = 0.7**2 if kind == "lmm" else 1.0
    denominator = 1.0 + 4 * weight * 0.8**2
    expected_covariance = scale * denominator / (result.matrices.n_obs * weight)
    expected_hat = (weight * 0.8**2 + 1.0 / result.matrices.n_obs) / denominator

    covariance = result.vcov()
    assert_allclose(covariance, [[expected_covariance]])
    assert_allclose(result.hatvalues(), expected_hat)
    assert_allclose(result.getME("RX").T @ result.getME("RX"), [[scale / expected_covariance]])
    if kind == "lmm":
        assert_allclose(result.predict(se_fit=True).se_fit ** 2, scale * expected_hat)
    covariance[:] = np.nan
    assert_allclose(result.vcov(), [[expected_covariance]])
    assert calls == [(q, q)]


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_sparse_projection_keeps_lazy_rzx_and_pickle_compatibility(kind, monkeypatch):
    monkeypatch.setattr(shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0)
    result = _result(kind)
    matrices = result.matrices
    weight = 1.0 if kind == "lmm" else np.exp(0.3)
    factor = _build_lambda(result.theta, matrices.random_structures).toarray()
    weighted_random = np.sqrt(weight) * (matrices.Z @ factor)
    precision = weighted_random.T @ weighted_random + np.eye(matrices.n_random)
    expected_rzx = linalg.solve_triangular(
        linalg.cholesky(precision, lower=True),
        weighted_random.T @ (np.sqrt(weight) * matrices.X),
        lower=True,
    )
    cholesky_calls = []
    original_cholesky = linalg.cholesky

    def counted_cholesky(matrix, *args, **kwargs):
        if matrix.shape == precision.shape:
            cholesky_calls.append(matrix.shape)
        return original_cholesky(matrix, *args, **kwargs)

    monkeypatch.setattr(linalg, "cholesky", counted_cholesky)
    covariance = result.vcov()
    assert cholesky_calls == []
    restored = pickle.loads(pickle.dumps(result))
    assert_allclose(restored.vcov(), covariance)
    assert_allclose(restored.hatvalues(), result.hatvalues())
    assert cholesky_calls == []
    actual = result.getME("RZX")
    assert_allclose(actual, expected_rzx)
    actual[:] = np.nan
    assert_allclose(result.getME("RZX"), expected_rzx)
    assert cholesky_calls == [precision.shape]


@pytest.mark.parametrize("keep", [[], [0]])
def test_profile_reuses_sparse_factor_without_materializing_cholesky(keep, monkeypatch):
    monkeypatch.setattr(shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0)
    result = _result("lmm")
    sparse_covariance = result.vcov()
    projection = _ProfileProjection.from_result(result, keep)
    assert projection.random_factor is result._weighted_projection.random_factor
    assert "cholesky" not in projection.random_factor.__dict__
    adjusted_y = result.matrices.y - 0.2
    actual = projection.deviance(adjusted_y)
    assert "cholesky" not in projection.random_factor.__dict__

    monkeypatch.setattr(shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", np.inf)
    reference = _result("lmm")
    expected = _ProfileProjection.from_result(reference, keep).deviance(adjusted_y)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert_allclose(sparse_covariance, reference.vcov(), rtol=1e-13)


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_factor_regularization_and_empty_rhs_preserve_precision(backend, monkeypatch):
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    precision = sparse.csc_matrix((3, 3))
    with pytest.raises((linalg.LinAlgError, RuntimeError)):
        _RandomEffectFactor(precision)
    factor = _RandomEffectFactor(precision, jitter=1e-6)
    rhs = np.arange(6.0).reshape(3, 2)

    assert_allclose(factor.solve(rhs), 1e6 * rhs)
    assert_allclose(factor.cholesky @ factor.cholesky.T, 1e-6 * np.eye(3))
    assert_array_equal(factor.solve(np.empty((3, 0))), np.empty((3, 0)))
    assert precision.nnz == 0
    restored = pickle.loads(pickle.dumps(factor))
    assert_allclose(restored.solve(rhs), 1e6 * rhs)

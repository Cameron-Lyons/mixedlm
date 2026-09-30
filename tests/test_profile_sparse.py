from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import reml as reml_module
from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.inference import profile as profile_module
from mixedlm.inference.profile import _ProfileProjection, profile_lmer, slice2D
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models import shared_utils
from mixedlm.models.lmer import LmerResult
from mixedlm.models.shared_utils import _RandomEffectFactor
from numpy.testing import assert_allclose
from scipy import linalg, sparse


def _result(structure="slope", reml=True, n_groups=8):
    rng = np.random.default_rng(473)
    n = 6 * n_groups
    x, z = rng.normal(size=(2, n))
    groups = np.repeat(np.arange(n_groups), 6)
    data = pd.DataFrame(
        {
            "y": 0.4 + 0.2 * x - 0.1 * z + rng.normal(size=n),
            "x": x,
            "z": z,
            "group": groups,
            "other": np.arange(n) % 6,
        }
    )
    formulas = {
        "fixed": ("y ~ x + z", []),
        "slope": ("y ~ x + z + (x | group)", [0.8, -0.15, 0.4]),
        "crossed": ("y ~ x + z + (x | group) + (1 | other)", [0.8, -0.15, 0.4, 0.6]),
        "independent": ("y ~ x + z + (x || group)", [0.8, 0.4]),
        "boundary": ("y ~ x + z + (x | group)", [0.8, -0.15, 0.0]),
        "cs": ("y ~ x + z + (x | group)", [0.8, -0.3]),
        "ar1": ("y ~ x + z + (x | group)", [0.8, 0.3]),
    }
    formula_text, theta = formulas[structure]
    formula = (
        set_cov_type(formula_text, structure)
        if structure in ("cs", "ar1")
        else parse_formula(formula_text)
    )
    matrices = build_model_matrices(
        formula, data, weights=np.geomspace(0.2, 3.0, n), offset=0.2 * np.sin(x)
    )
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.asarray(theta),
        beta=np.array([0.4, 0.2, -0.1]),
        sigma=0.7,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=reml,
        converged=True,
        n_iter=0,
    )


def _from_components(result, keep):
    matrices = result.matrices
    zt = matrices.Zt
    return _ProfileProjection.from_components(
        result.theta,
        matrices.y - matrices.offset,
        matrices.weights,
        matrices.X,
        keep,
        zt.data,
        zt.indices,
        zt.indptr,
        zt.shape,
        matrices.random_structures,
        matrices.n_obs,
        matrices.n_random,
        result.REML,
    )


def _direct_deviance(result, adjusted_y, keep):
    matrices = result.matrices
    transformed = (matrices.Z @ _build_lambda(result.theta, matrices.random_structures)).toarray()
    covariance = np.diag(1.0 / matrices.weights) + transformed @ transformed.T
    factor = linalg.cho_factor(covariance, lower=True)
    x = matrices.X[:, keep]
    information = x.T @ linalg.cho_solve(factor, x)
    rhs = x.T @ linalg.cho_solve(factor, adjusted_y)
    beta = linalg.solve(information, rhs, assume_a="pos") if keep else np.empty(0)
    residual = adjusted_y - x @ beta
    pwrss = residual @ linalg.cho_solve(factor, residual)
    denominator = matrices.n_obs - len(keep) if result.REML else matrices.n_obs
    deviance = denominator * (1.0 + np.log(2.0 * np.pi * pwrss / denominator))
    deviance += np.linalg.slogdet(covariance)[1]
    if result.REML and keep:
        deviance += np.linalg.slogdet(information)[1]
    return deviance


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize(
    "structure", ["fixed", "slope", "crossed", "independent", "boundary", "cs", "ar1"]
)
@pytest.mark.parametrize("held", [[1], [0, 1], [0, 1, 2]])
def test_profiles_match_direct_marginal_covariance(backend, reml, structure, held, monkeypatch):
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    result = _result(structure, reml)
    keep = [i for i in range(result.matrices.n_fixed) if i not in held]
    adjusted_y = result.matrices.y - result.matrices.offset
    adjusted_y = adjusted_y - result.matrices.X[:, held] @ (result.beta[held] + 0.25)
    expected = _direct_deviance(result, adjusted_y, keep)

    for projection in (
        _ProfileProjection.from_result(result, keep),
        _from_components(result, keep),
    ):
        assert_allclose(projection.deviance(adjusted_y), expected, rtol=1e-12, atol=1e-11)
        if backend == "sparse" and projection.random_factor is not None:
            assert "cholesky" not in projection.random_factor.__dict__


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_factor_logdet_matches_correlated_precision(backend, monkeypatch):
    monkeypatch.setattr(
        shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0 if backend == "sparse" else np.inf
    )
    rng = np.random.default_rng(692)
    raw = rng.normal(size=(9, 9))
    scales = np.geomspace(0.01, 100.0, 9)
    precision = (raw @ raw.T + np.eye(9)) * np.outer(scales, scales)
    factor = _RandomEffectFactor(sparse.csc_matrix(precision))

    assert_allclose(factor.logdet, np.linalg.slogdet(precision)[1], rtol=1e-12)
    if backend == "sparse":
        assert "cholesky" not in factor.__dict__


@pytest.mark.parametrize("builder", ["result", "components"])
def test_large_profile_never_densifies_random_precision(builder, monkeypatch):
    result = _result(n_groups=150)
    q = result.matrices.n_random
    adjusted_y = result.matrices.y - result.matrices.offset - 0.5 * result.matrices.X[:, 1]
    expected = _direct_deviance(result, adjusted_y, [0, 2])
    calls = []
    original_splu = sparse.linalg.splu

    def counted_splu(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return original_splu(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.linalg, "splu", counted_splu)
    for cls in (sparse.csc_matrix, sparse.csr_matrix):
        original_toarray = cls.toarray

        def guarded_toarray(matrix, *args, original=original_toarray, **kwargs):
            assert matrix.shape != (q, q), "profiling must keep random precision sparse"
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(cls, "toarray", guarded_toarray)

    if builder == "result":
        result.vcov()
        projection = _ProfileProjection.from_result(result, [0, 2])
    else:
        projection = _from_components(result, [0, 2])
    for _ in range(3):
        assert_allclose(projection.deviance(adjusted_y), expected, rtol=1e-12, atol=1e-10)
    assert calls == [(q, q)]


def test_sparse_serial_and_parallel_profiles_match_dense_profiles(monkeypatch):
    result = _result("crossed")
    monkeypatch.setattr(shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", np.inf)
    monkeypatch.setattr(reml_module, "_SPARSE_PROFILE_MIN_RANDOM", np.inf)
    expected_profiles = profile_lmer(replace(result), n_points=7)
    expected_slice = slice2D(replace(result), "(Intercept)", "x", n_points=5)
    monkeypatch.setattr(shared_utils, "_SPARSE_PROJECTION_MIN_RANDOM", 0)
    monkeypatch.setattr(reml_module, "_SPARSE_PROFILE_MIN_RANDOM", 0)
    monkeypatch.setattr(profile_module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(profile_module, "_SLICE2D_PARALLEL_MIN_TASKS", 0)

    for jobs in (1, 2):
        actual = profile_lmer(replace(result), n_points=7, n_jobs=jobs)
        for name, reference in expected_profiles.items():
            # Nuisance fits and interval roots have optimization tolerance;
            # the conditional slices below still agree to linear-solve precision.
            assert_allclose(actual[name].values, reference.values, rtol=1e-8, atol=1e-8)
            assert_allclose(actual[name].zeta, reference.zeta, rtol=1e-10, atol=1e-7)
            assert_allclose(
                [actual[name].ci_lower, actual[name].ci_upper],
                [reference.ci_lower, reference.ci_upper],
                rtol=1e-8,
                atol=1e-8,
            )
        actual_slice = slice2D(replace(result), "(Intercept)", "x", n_points=5, n_jobs=jobs)
        assert_allclose(actual_slice.values1, expected_slice.values1, rtol=1e-12, atol=1e-12)
        assert_allclose(actual_slice.values2, expected_slice.values2, rtol=1e-12, atol=1e-12)
        assert_allclose(actual_slice.zeta, expected_slice.zeta, rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize("error", [linalg.LinAlgError, RuntimeError])
def test_failed_profile_factorization_returns_penalty(error, monkeypatch):
    result = _result()

    def failed_factor(*args, **kwargs):
        raise error("factorization failed")

    monkeypatch.setattr(profile_module, "_RandomEffectFactor", failed_factor)
    projection = _from_components(result, [0, 2])
    assert projection.deviance(result.matrices.y) == 1e10

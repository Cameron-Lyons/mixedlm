from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import reml
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose
from scipy import sparse


def _matrices(n_groups=20):
    rng = np.random.default_rng(127)
    n = n_groups * 5
    x = rng.normal(size=n)
    group = np.repeat(np.arange(n_groups), 5)
    group2 = np.arange(n) % (n_groups - 1)
    y = 1.0 + 0.6 * x + rng.normal(size=n_groups)[group] + rng.normal(size=n)
    data = pd.DataFrame(dict(y=y, x=x, group=group, group2=group2))
    return build_model_matrices(
        parse_formula("y ~ x + (x | group) + (1 | group2)"),
        data,
        weights=rng.uniform(0.2, 3.0, size=n),
        offset=rng.normal(scale=0.3, size=n),
    )


@pytest.mark.parametrize("reml_fit", [False, True])
@pytest.mark.parametrize("theta", [[0.8, -0.2, 0.4, 0.6], [0.0, 0.0, 0.4, 0.0]])
def test_sparse_crossed_profile_matches_dense_estimates(monkeypatch, reml_fit, theta):
    matrices = _matrices()
    theta = np.asarray(theta)
    monkeypatch.setattr(reml, "_SPARSE_PROFILE_MIN_RANDOM", np.inf)
    expected = reml._profiled_deviance_core(theta, matrices, REML=reml_fit)
    monkeypatch.setattr(reml, "_SPARSE_PROFILE_MIN_RANDOM", 0)

    actual = reml._profiled_deviance_core(theta, matrices, REML=reml_fit)

    assert actual is not None and expected is not None
    for name in ("deviance", "beta", "sigma", "u", "ldL2", "ldRX2", "wrss", "ussq", "pwrss"):
        assert_allclose(getattr(actual, name), getattr(expected, name), rtol=2e-12, atol=2e-12)


def test_large_random_system_never_densifies(monkeypatch):
    matrices = _matrices(n_groups=400)
    theta = np.array([0.8, -0.2, 0.4, 0.6])

    def reject_dense_matrix(self, *args, **kwargs):
        raise AssertionError("large sparse likelihood systems must not be densified")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense_matrix)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense_matrix)
    core = reml._profiled_deviance_core(theta, matrices)
    optimizer = reml.LMMOptimizer(matrices, use_rust=False)

    assert core is not None
    assert np.isfinite(core.deviance)
    assert optimizer.objective(theta) == pytest.approx(core.deviance)
    beta, sigma, u = optimizer._extract_estimates(theta)
    assert_allclose(beta, core.beta)
    assert sigma == pytest.approx(core.sigma)
    assert_allclose(u, core.u)


def test_sparse_factorization_is_reused_for_all_solves(monkeypatch):
    matrices = _matrices()
    theta = np.array([0.8, -0.2, 0.4, 0.6])
    monkeypatch.setattr(reml, "_SPARSE_PROFILE_MIN_RANDOM", 0)
    original = reml.sparse_linalg.splu
    factorizations = 0

    def count_factorizations(matrix):
        nonlocal factorizations
        factorizations += 1
        return original(matrix)

    monkeypatch.setattr(reml.sparse_linalg, "splu", count_factorizations)

    result = reml._profiled_deviance_core(theta, matrices)

    assert result is not None
    assert factorizations == 1


def test_failed_sparse_factorization_preserves_invalid_objective(monkeypatch):
    matrices = _matrices()
    theta = np.array([0.8, -0.2, 0.4, 0.6])
    monkeypatch.setattr(reml, "_SPARSE_PROFILE_MIN_RANDOM", 0)

    def fail_factorization(matrix):
        raise RuntimeError("factor is singular")

    monkeypatch.setattr(reml.sparse_linalg, "splu", fail_factorization)

    assert reml._profiled_deviance_core(theta, matrices) is None
    assert reml.profiled_deviance(theta, matrices) == 1e10


def test_small_system_keeps_dense_cholesky(monkeypatch):
    matrices = _matrices()

    def reject_sparse_factorization(matrix):
        raise AssertionError("small systems should use dense Cholesky")

    monkeypatch.setattr(reml.sparse_linalg, "splu", reject_sparse_factorization)

    assert reml._profiled_deviance_core(np.array([0.8, -0.2, 0.4, 0.6]), matrices) is not None


def test_sparse_profile_preserves_weight_rescaling(monkeypatch):
    matrices = _matrices()
    theta = np.array([0.8, -0.2, 0.4, 0.6])
    scale = 7.0
    monkeypatch.setattr(reml, "_SPARSE_PROFILE_MIN_RANDOM", 0)

    result = reml._profiled_deviance_core(theta, matrices)
    scaled = reml._profiled_deviance_core(
        theta / np.sqrt(scale), replace(matrices, weights=matrices.weights * scale)
    )

    assert result is not None and scaled is not None
    assert_allclose(scaled.deviance, result.deviance, atol=1e-10)
    assert_allclose(scaled.beta, result.beta, atol=1e-12)
    assert_allclose(scaled.u, result.u, atol=1e-12)
    assert scaled.sigma == pytest.approx(result.sigma * np.sqrt(scale))

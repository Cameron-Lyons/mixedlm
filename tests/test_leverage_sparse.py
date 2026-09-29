from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation.reml import _build_lambda
from mixedlm.families import Binomial, Poisson
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models import shared_utils
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from mixedlm.models.shared_utils import sparse_quadratic_form_diagonal
from scipy import linalg, sparse


@pytest.mark.parametrize("shape", [(17, 7), (0, 7), (17, 0), (0, 0)])
@pytest.mark.parametrize("budget", [1, 32, 1_000_000])
def test_sparse_quadratic_diagonal_matches_dense_covariance(shape, budget, monkeypatch) -> None:
    rng = np.random.default_rng(823)
    design = sparse.random(*shape, density=0.4, format="csc", random_state=rng)
    dense_design = design.toarray()
    raw = rng.normal(size=(shape[1], shape[1]))
    precision = raw @ raw.T + np.eye(shape[1])
    factor = linalg.cholesky(precision, lower=True)
    expected = np.diag(dense_design @ linalg.inv(precision) @ dense_design.T)
    original_factor = factor.copy()
    original_toarray = sparse.csr_matrix.toarray

    def bounded_toarray(matrix, *args, **kwargs):
        assert np.prod(matrix.shape) <= max(budget, shape[1])
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(shared_utils, "_MAX_QUADRATIC_FORM_ELEMENTS", budget)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", bounded_toarray)

    actual = sparse_quadratic_form_diagonal(design, factor)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(factor, original_factor)
    np.testing.assert_array_equal(design.toarray(), dense_design)


def _result(kind: str, structure: str) -> LmerResult | GlmerResult:
    rng = np.random.default_rng(923)
    n = 42
    x = rng.normal(size=n)
    data = pd.DataFrame({"y": 1.0 + x, "x": x, "g": np.arange(n) % 7, "h": np.arange(n) % 6})
    formulas = {
        "fixed": ("y ~ x", []),
        "slope": ("y ~ x + (x | g)", [0.9, -0.2, 0.5]),
        "uncorrelated": ("y ~ x + (x || g)", [0.9, 0.5]),
        "crossed": ("y ~ x + (x | g) + (1 | h)", [0.9, -0.2, 0.5, 0.4]),
        "boundary": ("y ~ x + (x | g)", [0.9, -0.2, 0.0]),
        "ar1": ("y ~ x + (x | g)", [0.9, 0.4]),
    }
    formula_string, theta = formulas[structure]
    formula = (
        set_cov_type(formula_string, "ar1") if structure == "ar1" else parse_formula(formula_string)
    )
    matrices = build_model_matrices(
        formula, data, weights=np.geomspace(0.2, 4.0, n), offset=np.linspace(-0.3, 0.3, n)
    )
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.asarray(theta),
        beta=np.array([0.2, 0.3]),
        u=rng.normal(scale=0.2, size=matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    if kind == "lmm":
        return LmerResult(**common, sigma=0.7, REML=True)
    return GlmerResult(**common, family=Binomial() if kind == "binomial" else Poisson(), nAGQ=1)


def _dense_hat_diagonal(result: LmerResult | GlmerResult) -> np.ndarray:
    matrices = result.matrices
    weights = matrices.weights.copy()
    if isinstance(result, GlmerResult):
        mu = result.family.clamp_mu(result.fitted(na_expand=False))
        weights = np.clip(weights * result.family.weights(mu), 1e-10, 1e10)
    factor = _build_lambda(result.theta, matrices.random_structures).toarray()
    design = np.column_stack([matrices.X, matrices.Z @ factor])
    design *= np.sqrt(weights)[:, None]
    penalty = np.diag(np.r_[np.zeros(matrices.n_fixed), np.ones(matrices.n_random)])
    information = design.T @ design + penalty
    return np.diag(design @ linalg.solve(information, design.T, assume_a="pos"))


@pytest.mark.parametrize("kind", ["lmm", "binomial", "poisson"])
@pytest.mark.parametrize(
    "structure", ["fixed", "slope", "uncorrelated", "crossed", "boundary", "ar1"]
)
def test_leverage_matches_joint_system_without_densifying_random_design(
    kind, structure, monkeypatch
) -> None:
    result = _result(kind, structure)
    expected = _dense_hat_diagonal(result)
    shape = (result.matrices.n_obs, result.matrices.n_random)
    monkeypatch.setattr(shared_utils, "_MAX_QUADRATIC_FORM_ELEMENTS", 64)

    for matrix_class in (sparse.csc_matrix, sparse.csr_matrix):
        original_toarray = matrix_class.toarray

        def guarded_toarray(matrix, *args, original_toarray=original_toarray, **kwargs):
            if matrix.shape == shape:
                raise AssertionError("leverage should not densify the full random design")
            return original_toarray(matrix, *args, **kwargs)

        monkeypatch.setattr(matrix_class, "toarray", guarded_toarray)

    actual = result.hatvalues()

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    actual[:] = np.nan
    np.testing.assert_allclose(result.hatvalues(), expected, rtol=1e-11, atol=1e-12)

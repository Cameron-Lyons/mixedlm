from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import reml
from mixedlm.formula.parser import parse_formula
from mixedlm.inference import ddf
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, sparse


def _model(cov_type="us", reml_fit=True):
    rng = np.random.default_rng(752)
    n = 32
    x = np.tile(np.linspace(-1.0, 1.0, 8), 4)
    z = rng.normal(size=n)
    offset = rng.normal(scale=0.3, size=n)
    y = 1.0 + 0.4 * x - 0.2 * z + offset + rng.normal(scale=0.5, size=n)
    data = pd.DataFrame(dict(y=y, x=x, z=z, group=np.repeat(np.arange(4), 8)))
    formula = parse_formula("y ~ x + z + (x + z | group)")
    matrices = build_model_matrices(formula, data, weights=np.linspace(0.5, 2.0, n), offset=offset)
    if cov_type == "fixed":
        matrices = replace(matrices, Z=sparse.csc_matrix((n, 0)), random_structures=[], n_random=0)
        theta = np.array([])
    elif cov_type == "diagonal":
        matrices.random_structures[0].correlated = False
        theta = np.array([0.8, 0.0, 0.6])
    elif cov_type in ("cs", "ar1"):
        matrices.random_structures[0].cov_type = cov_type
        theta = np.array([0.9, -0.2])
    else:
        theta = np.array([0.8, 0.1, 0.5, -0.2, 0.3, 0.6])
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=theta,
        beta=np.zeros(matrices.n_fixed),
        sigma=1.3,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=reml_fit,
        converged=True,
        n_iter=0,
    )


def _direct_information(model, theta):
    matrices = model.matrices
    marginal = np.diag(1.0 / matrices.weights)
    if matrices.n_random:
        structure = matrices.random_structures[0]
        if structure.cov_type in ("cs", "ar1"):
            scale, rho = theta
            covariance = (
                np.full((3, 3), rho)
                if structure.cov_type == "cs"
                else rho ** np.abs(np.arange(3)[:, None] - np.arange(3))
            )
            np.fill_diagonal(covariance, 1.0)
            covariance *= scale**2
        elif structure.correlated:
            factor = np.array(
                [[theta[0], 0, 0], [theta[1], theta[2], 0], [theta[3], theta[4], theta[5]]]
            )
            covariance = factor @ factor.T
        else:
            covariance = np.diag(theta**2)
        Z = matrices.Z.toarray()
        marginal += Z @ np.kron(np.eye(structure.n_levels), covariance) @ Z.T
    solved_x = linalg.solve(marginal, matrices.X, assume_a="pos")
    information = matrices.X.T @ solved_x
    adjusted_y = matrices.y - matrices.offset
    beta = linalg.solve(information, solved_x.T @ adjusted_y, assume_a="pos")
    residual = adjusted_y - matrices.X @ beta
    denom = matrices.n_obs - matrices.n_fixed if model.REML else matrices.n_obs
    sigma2 = residual @ linalg.solve(marginal, residual, assume_a="pos") / denom
    return information, sigma2


@pytest.mark.parametrize("cov_type", ["fixed", "us", "diagonal", "cs", "ar1"])
@pytest.mark.parametrize("reml_fit", [False, True])
def test_reused_information_matches_direct_weighted_covariance(cov_type, reml_fit):
    model = _model(cov_type, reml_fit)
    products = reml._LMMCrossproducts.from_matrices(model.matrices)
    information, sigma2 = _direct_information(model, model.theta)

    actual = ddf._vcov_from_theta(model, model.theta, products)
    profiled = reml._profiled_deviance_core(model.theta, model.matrices, REML=reml_fit)

    assert profiled is not None
    assert_allclose(profiled.fixed_information, information, atol=2e-13)
    assert_allclose(actual, sigma2 * linalg.inv(information), atol=2e-14)


@pytest.mark.parametrize("cov_type", ["us", "diagonal", "cs", "ar1"])
@pytest.mark.parametrize("reml_fit", [False, True])
def test_covariance_derivatives_match_direct_marginal_reference(cov_type, reml_fit):
    model = _model(cov_type, reml_fit)
    ddf.clear_vcov_grad_cache()
    actual, _ = ddf._vcov_derivatives(model)

    for index, value in enumerate(model.theta):
        step = 1e-6 * max(1.0, abs(value))
        plus = model.theta.copy()
        minus = model.theta.copy()
        plus[index] += step
        minus[index] -= step
        plus_info, plus_sigma2 = _direct_information(model, plus)
        minus_info, minus_sigma2 = _direct_information(model, minus)
        expected = (plus_sigma2 * linalg.inv(plus_info) - minus_sigma2 * linalg.inv(minus_info)) / (
            2 * step
        )
        assert_allclose(actual[index], expected, rtol=1e-6, atol=2e-10)


def test_derivatives_and_curvature_share_one_set_of_weighted_products(monkeypatch):
    model = _model()
    builds = 0
    original = reml._LMMCrossproducts.from_matrices

    def count_build(cls, matrices):
        nonlocal builds
        builds += 1
        return original(matrices)

    monkeypatch.setattr(reml._LMMCrossproducts, "from_matrices", classmethod(count_build))
    ddf.clear_vcov_grad_cache()

    first_gradients, first_covariance = ddf._vcov_derivatives(model)
    repeated_gradients, repeated_covariance = ddf._vcov_derivatives(model)

    assert builds == 1
    assert_array_equal(first_gradients, repeated_gradients)
    assert_array_equal(first_covariance, repeated_covariance)


def test_coefficient_covariance_factors_random_system_only_once(monkeypatch):
    model = _model()
    products = reml._LMMCrossproducts.from_matrices(model.matrices)
    random_factorizations = 0
    cholesky = linalg.cholesky

    def count_cholesky(matrix, *args, **kwargs):
        nonlocal random_factorizations
        if matrix.shape == (model.matrices.n_random, model.matrices.n_random):
            random_factorizations += 1
        return cholesky(matrix, *args, **kwargs)

    monkeypatch.setattr(linalg, "cholesky", count_cholesky)

    ddf._vcov_from_theta(model, model.theta, products)

    assert random_factorizations == 1


def test_failed_profile_retains_fitted_scale_fallback(monkeypatch):
    model = _model()
    products = reml._LMMCrossproducts.from_matrices(model.matrices)
    information, _ = _direct_information(model, model.theta)
    monkeypatch.setattr(reml, "_profiled_deviance_core", lambda *args, **kwargs: None)

    actual = ddf._vcov_from_theta(model, model.theta, products)

    assert_allclose(actual, model.sigma**2 * linalg.inv(information), atol=1e-13)
    assert_array_equal(
        ddf._profiled_theta_covariance(model, model.theta, products),
        np.zeros((len(model.theta), len(model.theta))),
    )

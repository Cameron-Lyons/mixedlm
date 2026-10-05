"""Validate covariance transforms against full, independently assembled systems."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import _rust
from mixedlm.estimation.laplace import (
    _laplace_deviance_with_status,
    adaptive_gh_deviance,
    laplace_deviance,
)
from mixedlm.families import Binomial, Gaussian, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose

from tests._glmm_oracles import covariance_problem, glmm_deviance_args


@pytest.mark.parametrize(
    "layout",
    ["intercept", "correlated", "diagonal", "mixed", "crossed_slopes", "no_fixed", "fixed"],
)
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_gaussian_transform_matches_dense_system(layout, variance, weighted, overlap):
    matrices, theta, factor = covariance_problem(layout, variance, weighted, overlap=overlap)
    p, q = matrices.n_fixed, matrices.n_random
    design = np.column_stack((matrices.X, matrices.Z @ factor))
    information = design.T @ (matrices.weights[:, None] * design)
    information[p:, p:] += np.eye(q)
    coefficients = np.linalg.solve(
        information, design.T @ (matrices.weights * (matrices.y - matrices.offset))
    )
    beta, spherical = coefficients[:p], coefficients[p:]
    random = factor @ spherical
    residual = matrices.y - matrices.offset - design @ coefficients
    conditional = np.dot(matrices.weights * residual, residual) + spherical @ spherical
    logdet = np.linalg.slogdet(information[p:, p:])[1]
    args = glmm_deviance_args(matrices, theta, "gaussian")

    actual = _rust.glmm_deviance(*args, 1)
    assert actual[3]
    assert actual[0] == pytest.approx(conditional + logdet, rel=1e-11, abs=1e-11)
    assert_allclose(actual[1], beta, rtol=1e-11, atol=1e-11)
    assert_allclose(actual[2], random, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("layout", ["correlated", "diagonal", "mixed", "crossed_slopes"])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
def test_iterative_transforms_match_python(layout, variance, weighted, family_name):
    matrices, theta, _ = covariance_problem(layout, variance, weighted, family_name)
    family = Poisson() if family_name == "poisson" else Binomial()
    args = glmm_deviance_args(matrices, theta, family_name)
    actual = _rust.glmm_deviance(*args, 1)
    expected = _laplace_deviance_with_status(theta, matrices, family)
    assert actual[3] and expected[3]
    for left, right in zip(actual[:3], expected[:3], strict=True):
        assert_allclose(left, right, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("scale", [0.0, 0.45, 1.2])
@pytest.mark.parametrize("weighted", [False, True])
def test_quadrature_transforms_match_python(family_name, scale, weighted):
    matrices, _, _ = covariance_problem("intercept", "regular", weighted, family_name)
    theta = np.array([scale])
    family = {"gaussian": Gaussian, "poisson": Poisson, "binomial": Binomial}[family_name]()
    actual = _rust.glmm_deviance(*glmm_deviance_args(matrices, theta, family_name), 9)
    expected = adaptive_gh_deviance(theta, matrices, family, nAGQ=9)
    for left, right in zip(actual[:3], expected, strict=True):
        assert_allclose(left, right, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("n_terms", [16, 32])
@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("singular", [False, True])
def test_wide_correlated_blocks_match_python(n_terms, family_name, singular):
    rng = np.random.default_rng(642)
    n = 320
    predictors = rng.normal(scale=0.1, size=(n, n_terms))
    names = [f"x{i}" for i in range(n_terms)]
    data = pd.DataFrame(predictors, columns=names)
    data["g"] = np.arange(n) % 2
    if family_name == "gaussian":
        data["y"] = rng.normal(loc=0.4, size=n)
    elif family_name == "poisson":
        data["y"] = rng.poisson(1.5, size=n).astype(float)
    else:
        data["y"] = rng.binomial(1, 0.4, size=n).astype(float)
    matrices = build_model_matrices(
        parse_formula("y ~ 1 + (0 + " + " + ".join(names) + " | g)"),
        data,
        weights=np.linspace(0.5, 1.5, n),
        offset=np.linspace(-0.1, 0.1, n),
    )
    lower = np.diag(np.full(n_terms, 0.45))
    lower[np.tril_indices(n_terms, -1)] = 0.015
    if singular:
        lower[:, -1] = 0.0
    theta = lower[np.tril_indices(n_terms)]
    family = {"gaussian": Gaussian, "poisson": Poisson, "binomial": Binomial}[family_name]()
    actual = _rust.glmm_deviance(*glmm_deviance_args(matrices, theta, family_name), 1)
    expected = laplace_deviance(theta, matrices, family)
    for left, right in zip(actual[:3], expected, strict=True):
        assert_allclose(left, right, rtol=2e-7, atol=2e-7)

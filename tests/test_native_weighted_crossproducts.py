"""Check native crossproducts and GLMM solves against independent dense systems."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import _rust
from mixedlm.estimation.laplace import (
    _build_lambda,
    adaptive_gh_deviance,
    laplace_deviance,
    pirls,
)
from mixedlm.families import Binomial, Gaussian, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose
from scipy import sparse


def _noncanonical(z):
    """Split each value between duplicates, reverse rows, and store explicit zeros."""
    values, rows, offsets = [], [], [0]
    for column in range(z.shape[1]):
        for index in range(z.indptr[column + 1] - 1, z.indptr[column] - 1, -1):
            values.extend([0.25 * z.data[index], 0.75 * z.data[index]])
            rows.extend([z.indices[index], z.indices[index]])
        if z.shape[0]:
            values.append(0.0)
            rows.append(column % z.shape[0])
        offsets.append(len(values))
    return sparse.csc_matrix((values, rows, offsets), shape=z.shape)


def _matrices(layout, weighted, noncanonical, family="gaussian"):
    rng = np.random.default_rng(951)
    n = 96
    x = rng.uniform(-1.0, 1.0, n)
    group = np.arange(n) % 8
    offset = np.linspace(-0.2, 0.2, n)
    eta = 0.2 + 0.3 * x + 0.15 * np.sin(group) + offset
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.normal(scale=0.3, size=n)
    data = pd.DataFrame({"y": y, "x": x, "g": group, "h": np.arange(n) % 5})
    if layout == "dense":
        data["g"] = 0
    formulas = {
        "intercept": "y ~ x + (1 | g)",
        "slopes": "y ~ x + (x | g)",
        "crossed": "y ~ x + (1 | g) + (1 | h)",
        "fixed": "y ~ x",
        "dense": "y ~ 1 + (x | g)",
    }
    matrices = build_model_matrices(
        parse_formula(formulas[layout]),
        data,
        weights=np.linspace(0.4, 2.0, n) if weighted else np.ones(n),
        offset=offset,
    )
    if noncanonical:
        matrices = replace(matrices, Z=_noncanonical(matrices.Z.tocsc()))
    theta = np.array(
        {
            "intercept": [0.6],
            "slopes": [0.6, 0.15, 0.4],
            "crossed": [0.6, 0.4],
            "fixed": [],
            "dense": [0.6, 0.15, 0.4],
        }[layout]
    )
    return matrices, theta


def _args(matrices, theta, family):
    z = matrices.Z.tocsc()
    return (
        matrices.y,
        matrices.X,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        matrices.weights,
        matrices.offset,
        theta,
        [s.n_levels for s in matrices.random_structures],
        [s.n_terms for s in matrices.random_structures],
        [s.correlated for s in matrices.random_structures],
        family,
        {"gaussian": "identity", "poisson": "log", "binomial": "logit"}[family],
    )


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (7, 0), (7, 9), (31, 18)])
@pytest.mark.parametrize("noncanonical", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("zero_fraction", [0.0, 0.7])
def test_crossproduct_matches_dense_reference(shape, noncanonical, weighted, zero_fraction):
    rng = np.random.default_rng(913)
    dense = rng.normal(size=shape)
    dense[rng.random(shape) < zero_fraction] = 0.0
    if all(shape) and zero_fraction:
        dense[0, :] = 0.0
        dense[:, -1] = 0.0
    z = sparse.csc_matrix(dense)
    if noncanonical:
        z = _noncanonical(z)
    weights = np.linspace(0.1, 2.0, shape[0]) if weighted else np.ones(shape[0])
    actual = _rust.compute_ztwz(
        z.data, z.indices.astype(np.int64), z.indptr.astype(np.int64), z.shape, weights
    ).reshape(shape[1], shape[1])
    assert_allclose(actual, dense.T @ (weights[:, None] * dense), rtol=1e-13, atol=1e-13)
    assert_allclose(actual, actual.T, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["intercept", "slopes", "crossed", "fixed", "dense"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("noncanonical", [False, True])
@pytest.mark.parametrize("zero_variance", [False, True])
def test_gaussian_glmm_matches_dense_penalized_solve(layout, weighted, noncanonical, zero_variance):
    matrices, theta = _matrices(layout, weighted, noncanonical)
    if zero_variance:
        theta[:] = 0.0
    lam = _build_lambda(theta, matrices.random_structures).toarray()
    design = np.column_stack((matrices.X, matrices.Z @ lam))
    p, q = matrices.n_fixed, matrices.n_random
    information = design.T @ (matrices.weights[:, None] * design)
    information[p:, p:] += np.eye(q)
    rhs = design.T @ (matrices.weights * (matrices.y - matrices.offset))
    coefficients = np.linalg.solve(information, rhs)
    beta, spherical = coefficients[:p], coefficients[p:]
    random = lam @ spherical
    residual = matrices.y - matrices.offset - design @ coefficients
    conditional = np.dot(matrices.weights * residual, residual) + spherical @ spherical
    logdet = np.linalg.slogdet(information[p:, p:])[1]

    args = _args(matrices, theta, "gaussian")
    actual_beta, actual_random, deviance, converged = _rust.pirls(*args)
    assert converged
    assert_allclose(actual_beta, beta, rtol=1e-11, atol=1e-11)
    assert_allclose(actual_random, random, rtol=1e-11, atol=1e-11)
    assert deviance == pytest.approx(conditional, rel=1e-11, abs=1e-11)
    laplace = _rust.laplace_deviance(*args)
    assert laplace[0] == pytest.approx(conditional + logdet, rel=1e-11, abs=1e-11)
    assert_allclose(laplace[1], beta, rtol=1e-11, atol=1e-11)
    assert_allclose(laplace[2], random, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("layout", ["intercept", "slopes", "crossed", "dense"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("noncanonical", [False, True])
@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
def test_changing_pirls_weights_match_python(layout, weighted, noncanonical, family_name):
    matrices, theta = _matrices(layout, weighted, noncanonical, family_name)
    family = Poisson() if family_name == "poisson" else Binomial()
    args = _args(matrices, theta, family_name)
    actual = _rust.pirls(*args)
    expected = pirls(matrices, family, theta, maxiter=100, tol=1e-9)
    assert actual[3] and expected[3]
    for left, right in zip(actual[:3], expected[:3], strict=True):
        assert_allclose(left, right, rtol=1e-7, atol=1e-7)
    actual = _rust.laplace_deviance(*args)
    expected = laplace_deviance(theta, matrices, family)
    for left, right in zip(actual, expected, strict=True):
        assert_allclose(left, right, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_quadrature_with_weighted_noncanonical_design(family_name, noncanonical):
    matrices, theta = _matrices("intercept", True, noncanonical, family_name)
    family = {"gaussian": Gaussian, "poisson": Poisson, "binomial": Binomial}[family_name]()
    actual = _rust.adaptive_gh_deviance(*_args(matrices, theta, family_name), 9)
    expected = adaptive_gh_deviance(theta, matrices, family, nAGQ=9)
    for left, right in zip(actual, expected, strict=True):
        assert_allclose(left, right, rtol=1e-7, atol=1e-7)

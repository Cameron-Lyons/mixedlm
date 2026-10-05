"""Validate covariance transforms against full, independently assembled systems."""

from dataclasses import replace

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
from scipy import linalg, sparse


def _problem(layout, variance, weighted, family_name="gaussian", overlap=False):
    rng = np.random.default_rng(924)
    n = 120
    rows = rng.permutation(n)
    x, z, w = rng.uniform(-1, 1, (3, n))
    groups, other = rows % 6, rows % 5
    offset = 0.1 * np.sin(rows)
    eta = 0.3 + 0.2 * x - 0.1 * z + 0.15 * np.cos(groups) + offset
    if family_name == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family_name == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = eta + rng.normal(scale=0.3, size=n)
    data = pd.DataFrame(dict(y=y, x=x, z=z, w=w, g=groups, h=other))
    formulas = {
        "intercept": "y ~ x + z + (1 | g)",
        "correlated": "y ~ x + z + (x + z | g)",
        "diagonal": "y ~ x + z + (x + z || g)",
        "mixed": "y ~ x + z + (x + z | g) + (w || h)",
        "crossed_slopes": "y ~ x + z + (x | g) + (0 + z | h)",
        "no_fixed": "y ~ 0 + (x + z | g)",
        "fixed": "y ~ x + z",
    }
    matrices = build_model_matrices(
        parse_formula(formulas[layout]),
        data,
        weights=np.linspace(0.4, 2.0, n) if weighted else np.ones(n),
        offset=offset,
    )
    theta, blocks = [], []
    for structure in matrices.random_structures:
        width = structure.n_terms
        lower = np.diag(np.linspace(0.45, 0.85, width))
        if structure.correlated:
            for i in range(width):
                for j in range(i):
                    lower[i, j] = 0.1 * (i + 1) * (-1) ** j
        if variance == "singular":
            lower[:, -1] = 0.0
        elif variance == "zero":
            lower[:] = 0.0
        theta.extend(lower[np.tril_indices(width)] if structure.correlated else lower.diagonal())
        blocks.extend([lower] * structure.n_levels)
    factor = linalg.block_diag(*blocks) if blocks else np.zeros((0, 0))
    if overlap and matrices.n_random:
        # Advanced designs can overlap multiple levels. A block covariance
        # factor must preserve the off-diagonal entries of their crossproduct.
        extra = rng.normal(scale=0.05, size=matrices.Z.shape)
        extra[rng.random(extra.shape) < 0.9] = 0.0
        matrices = replace(matrices, Z=(matrices.Z + sparse.csc_matrix(extra)).tocsc())
    return matrices, np.asarray(theta), factor


def _args(matrices, theta, family_name):
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
        family_name,
        {"gaussian": "identity", "poisson": "log", "binomial": "logit"}[family_name],
    )


@pytest.mark.parametrize(
    "layout",
    ["intercept", "correlated", "diagonal", "mixed", "crossed_slopes", "no_fixed", "fixed"],
)
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_gaussian_transform_matches_dense_system(layout, variance, weighted, overlap):
    matrices, theta, factor = _problem(layout, variance, weighted, overlap=overlap)
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
    args = _args(matrices, theta, "gaussian")

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
    matrices, theta, _ = _problem(layout, variance, weighted, family_name)
    family = Poisson() if family_name == "poisson" else Binomial()
    args = _args(matrices, theta, family_name)
    actual = _rust.glmm_deviance(*args, 1)
    expected = _laplace_deviance_with_status(theta, matrices, family)
    assert actual[3] and expected[3]
    for left, right in zip(actual[:3], expected[:3], strict=True):
        assert_allclose(left, right, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("scale", [0.0, 0.45, 1.2])
@pytest.mark.parametrize("weighted", [False, True])
def test_quadrature_transforms_match_python(family_name, scale, weighted):
    matrices, _, _ = _problem("intercept", "regular", weighted, family_name)
    theta = np.array([scale])
    family = {"gaussian": Gaussian, "poisson": Poisson, "binomial": Binomial}[family_name]()
    actual = _rust.glmm_deviance(*_args(matrices, theta, family_name), 9)
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
    actual = _rust.glmm_deviance(*_args(matrices, theta, family_name), 1)
    expected = laplace_deviance(theta, matrices, family)
    for left, right in zip(actual[:3], expected, strict=True):
        assert_allclose(left, right, rtol=2e-7, atol=2e-7)

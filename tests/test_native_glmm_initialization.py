from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import laplace
from mixedlm.families import Binomial, Gaussian, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices

native = pytest.importorskip("mixedlm._rust")


def native_pirls(matrices, family, theta):
    z = matrices.Z.tocsc()
    return native.pirls(
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
        family.__class__.__name__.lower(),
        family.link.name,
    )


def poisson_matrices(mean, offset=0.0, weighted=False, formula="y ~ 1 + (1 | g)"):
    group = np.repeat(np.arange(4), 3)
    data = pd.DataFrame({"y": np.tile([mean - 2, mean, mean + 2], 4), "g": group})
    weights = np.tile([0.5, 1.0, 2.0], 4) if weighted else np.ones(12)
    return build_model_matrices(
        parse_formula(formula), data, weights=weights, offset=np.full(12, offset)
    )


@pytest.mark.parametrize("mean", [10, 100, 200, 1000, 10000])
@pytest.mark.parametrize("offset", [0.0, -20.0, 20.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_poisson_pirls_converges_to_weighted_mean(mean, offset, weighted):
    matrices = poisson_matrices(mean, offset, weighted)
    theta = np.array([0.5])
    beta, u, deviance, converged = native_pirls(matrices, Poisson(), theta)
    expected_mean = np.average(matrices.y, weights=matrices.weights)
    assert converged
    assert np.isfinite(deviance)
    np.testing.assert_allclose(beta, [np.log(expected_mean) - offset], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(u, 0, atol=1e-8)
    expected = np.sum(
        Poisson().deviance_resids(matrices.y, np.full(12, expected_mean), matrices.weights)
    )
    assert deviance == pytest.approx(expected, rel=1e-7, abs=1e-8)


@pytest.mark.parametrize("mean", [200, 1000, 10000])
@pytest.mark.parametrize("order", [1, 9])
def test_poisson_objectives_agree_with_python_at_large_counts(mean, order):
    matrices = poisson_matrices(mean, offset=4.0, weighted=True)
    theta = np.array([0.7])
    actual = laplace.adaptive_gh_deviance_fast(theta, matrices, Poisson(), nAGQ=order)
    expected = laplace.adaptive_gh_deviance(theta, matrices, Poisson(), nAGQ=order)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("mean", [200, 1000])
def test_fixed_only_poisson_initialization(mean):
    matrices = poisson_matrices(mean, offset=-5.0, weighted=True, formula="y ~ 1")
    beta, u, deviance, converged = native_pirls(matrices, Poisson(), np.array([]))
    assert converged
    assert np.isfinite(deviance)
    assert len(u) == 0
    assert beta[0] == pytest.approx(np.log(np.average(matrices.y, weights=matrices.weights)) + 5)


@pytest.mark.parametrize("family", [Gaussian(), Binomial(), Poisson()])
@pytest.mark.parametrize("weighted", [False, True])
def test_native_and_python_modes_agree_with_offsets_and_slopes(family, weighted):
    rng = np.random.default_rng(713)
    group = np.repeat(np.arange(5), 12)
    x = rng.uniform(-0.7, 0.7, len(group))
    offset = np.linspace(-0.4, 0.5, len(group))
    eta = 0.6 + 0.4 * x + rng.normal(0, 0.2, 5)[group] + offset
    if isinstance(family, Gaussian):
        y = eta + rng.normal(0, 0.4, len(group))
    elif isinstance(family, Binomial):
        y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    else:
        y = rng.poisson(np.exp(eta))
    data = pd.DataFrame({"y": y, "x": x, "g": group})
    weights = np.linspace(0.2, 3.0, len(group)) if weighted else None
    matrices = build_model_matrices(
        parse_formula("y ~ x + (x | g)"), data, weights=weights, offset=offset
    )
    theta = np.array([0.6, 0.1, 0.3])
    actual = native_pirls(matrices, family, theta)
    expected = laplace.pirls(matrices, family, theta)
    assert actual[3] and expected[3]
    for value, reference in zip(actual[:3], expected[:3], strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("field", ["y", "offset"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_results_are_not_reported_as_converged(field, value):
    matrices = poisson_matrices(10)
    invalid = getattr(matrices, field).copy()
    invalid[0] = value
    matrices = replace(matrices, **{field: invalid})
    _, _, deviance, converged = native_pirls(matrices, Poisson(), np.array([0.5]))
    assert not converged
    assert not np.isfinite(deviance)


@pytest.mark.parametrize("mean", [200, 1000])
@pytest.mark.parametrize("order", [1, 9])
def test_public_poisson_fit_retains_large_response_means(mean, order):
    from mixedlm import glmer, glmerControl

    data = pd.DataFrame(
        {"y": np.tile([mean - 2, mean, mean + 2], 5), "g": np.repeat(np.arange(5), 3)}
    )
    result = glmer(
        "y ~ 1 + (1 | g)",
        data,
        family=Poisson(),
        nAGQ=order,
        control=glmerControl(check_singular=False),
    )
    assert result.converged
    assert np.isfinite(result.deviance)
    np.testing.assert_allclose(result.fitted(), mean, rtol=1e-7)

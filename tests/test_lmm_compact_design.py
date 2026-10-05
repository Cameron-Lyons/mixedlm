"""Compact prepared products retain groupwise likelihoods and cached inputs."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from tests.test_lmm_gradient_contractions import observation_gradient
from tests.test_lmm_prepared_design import native_arguments, observation_likelihood
from tests.test_lmm_reml_contractions import fixed_effect_problem


def large_intercepts(levels, fixed):
    n = 3 * levels
    rows = np.arange(n)
    groups = rows % levels
    offset = 0.1 * np.cos(rows)
    matrices = ModelMatrices(
        y=offset + 0.3 + 0.4 * np.sin(groups) + 0.1 * np.sin(rows * 0.137),
        X=np.ones((n, 1)) if fixed else np.empty((n, 0)),
        Z=sparse.csc_matrix((np.ones(n), (rows, groups)), shape=(n, levels)),
        fixed_names=["Intercept"] if fixed else [],
        random_structures=[RandomEffectStructure("g", ["Intercept"], levels, 1, True, {})],
        n_obs=n,
        n_fixed=int(fixed),
        n_random=levels,
        weights=np.geomspace(0.5, 2.0, n),
        offset=offset,
    )
    return matrices, groups


def grouped_intercept_oracle(matrices, groups, theta, reml):
    """Within/between-group decomposition, without a square random-effect matrix."""
    y, w = matrices.y - matrices.offset, matrices.weights
    totals = np.bincount(groups, weights=w)
    means = np.bincount(groups, weights=w * y) / totals
    precision = 1 + theta**2 * totals
    information = np.sum(totals / precision)
    beta = np.sum(totals * means / precision) / information if matrices.n_fixed else 0.0
    centered = means - beta
    pwrss = np.sum(w * (y - means[groups]) ** 2) + np.sum(totals * centered**2 / precision)
    df = matrices.n_obs - (matrices.n_fixed if reml else 0)
    value = df * (1 + np.log(2 * np.pi * pwrss / df))
    value += np.log(precision).sum() - np.log(w).sum()
    gradient = np.sum(2 * theta * totals / precision)
    gradient -= df / pwrss * np.sum(2 * theta * (totals * centered / precision) ** 2)
    if reml and matrices.n_fixed:
        value += np.log(information)
        gradient -= np.sum(2 * theta * (totals / precision) ** 2) / information
    return value, gradient, beta, np.sqrt(pwrss / df)


def large_slopes(levels, width):
    matrices, groups = large_intercepts(levels, True)
    terms = np.random.default_rng(714).normal(scale=0.3, size=(matrices.n_obs, width))
    terms[:, 0] = 1.0
    columns = groups[:, None] * width + np.arange(width)
    z = sparse.csc_matrix(
        (terms.ravel(), (np.repeat(np.arange(matrices.n_obs), width), columns.ravel())),
        shape=(matrices.n_obs, levels * width),
    )
    structure = replace(
        matrices.random_structures[0], n_terms=width, term_names=[f"x{i}" for i in range(width)]
    )
    matrices = replace(matrices, Z=z, n_random=z.shape[1], random_structures=[structure])
    lower = np.diag(np.linspace(0.3, 0.7, width))
    lower[np.tril_indices(width, -1)] = 0.03
    return matrices, terms, lower[np.tril_indices(width)]


def grouped_slope_oracle(matrices, terms, theta, reml):
    """Independent three-observation covariance systems, with no q-by-q array."""
    levels, width = matrices.random_structures[0].n_levels, terms.shape[1]
    design = terms.reshape(3, levels, width).transpose(1, 0, 2)
    lower = np.zeros((width, width))
    lower[np.tril_indices(width)] = theta
    transformed = design @ lower
    covariance = transformed @ transformed.transpose(0, 2, 1)
    diagonal = np.arange(3)
    covariance[:, diagonal, diagonal] += 1 / matrices.weights.reshape(3, levels).T
    precision = np.linalg.inv(covariance)
    y = (matrices.y - matrices.offset).reshape(3, levels).T
    information = precision.sum()
    beta = np.sum(np.einsum("gij,gj->gi", precision, y)) / information
    residual = y - beta
    projected = np.einsum("gij,gj->gi", precision, residual)
    pwrss = np.sum(residual * projected)
    df = matrices.n_obs - int(reml)
    sign, logdet = np.linalg.slogdet(covariance)
    assert_array_equal(sign, np.ones(levels))
    value = df * (1 + np.log(2 * np.pi * pwrss / df)) + logdet.sum()
    if reml:
        value += np.log(information)
    fixed_projection = precision.sum(axis=2)
    gradient = []
    for row, column in zip(*np.tril_indices(width), strict=True):
        basis = np.zeros((width, width))
        basis[row, column] = 1.0
        changed = design @ basis
        derivative = changed @ transformed.transpose(0, 2, 1)
        derivative += derivative.transpose(0, 2, 1).copy()
        score = np.einsum("gij,gji->", precision, derivative)
        score -= df / pwrss * np.einsum("gi,gij,gj->", projected, derivative, projected)
        if reml:
            score -= (
                np.einsum("gi,gij,gj->", fixed_projection, derivative, fixed_projection)
                / information
            )
        gradient.append(score)
    return value, gradient, beta, np.sqrt(pwrss / df)


@pytest.mark.parametrize("q", [512, 4096])
@pytest.mark.parametrize("width", [2, 8])
@pytest.mark.parametrize("scale", [0.0, 1.0])
@pytest.mark.parametrize("reml", [False, True])
def test_many_independent_slopes_match_group_observation_systems(q, width, scale, reml):
    matrices, terms, theta = large_slopes(q // width, width)
    theta *= scale
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    expected, expected_gradient, beta, sigma = grouped_slope_oracle(matrices, terms, theta, reml)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert value == response.deviance(theta, reml)
    assert_allclose(value, expected, rtol=2e-12, atol=2e-8)
    assert_allclose(gradient, expected_gradient, rtol=2e-11, atol=2e-8)
    final = response.evaluate(theta, reml)
    assert final[0] == value
    assert_allclose(final[1], [beta], rtol=2e-12, atol=2e-12)
    assert_allclose(final[2], sigma, rtol=2e-12)


@pytest.mark.parametrize("levels", [512, 4096])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("theta_value", [0.0, 0.4, 1.2])
def test_many_independent_levels_match_groupwise_likelihood(levels, fixed, reml, theta_value):
    matrices, groups = large_intercepts(levels, fixed)
    design = _rust.LmmDesign(**native_arguments(matrices))
    for y in [matrices.y, matrices.y[::-1] + matrices.offset]:
        current = replace(matrices, y=y)
        response = design.with_response(y)
        expected, expected_gradient, beta, sigma = grouped_intercept_oracle(
            current, groups, theta_value, reml
        )
        value, gradient = response.deviance_with_gradient(np.array([theta_value]), reml)
        assert value == response.deviance(np.array([theta_value]), reml)
        assert_allclose(value, expected, rtol=2e-12, atol=2e-8)
        assert_allclose(gradient, [expected_gradient], rtol=2e-11, atol=2e-8)
        final = response.evaluate(np.array([theta_value]), reml)
        assert final[0] == value
        assert_allclose(final[1], [beta] if fixed else [], rtol=2e-12, atol=2e-12)
        assert_allclose(final[2], sigma, rtol=2e-12)


@pytest.mark.parametrize("widths", [(3, 2), (17, 16)])
@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("reml", [False, True])
def test_compact_preparation_matches_observation_system(widths, coupled, diagonal, variance, reml):
    matrices, theta = fixed_effect_problem(widths, 3, coupled, diagonal, variance)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert value == response.deviance(theta, reml)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)

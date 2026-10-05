"""Compact prepared products retain groupwise likelihoods and cached inputs."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose

from tests._lmm_oracles import (
    fixed_effect_problem,
    grouped_slope_oracle,
    large_intercepts,
    large_slopes,
    native_arguments,
    observation_gradient,
    observation_likelihood,
)


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

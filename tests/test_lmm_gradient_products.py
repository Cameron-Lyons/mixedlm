"""Shared fixed-effect products preserve wide and cancellation-sensitive gradients."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose

from tests._lmm_oracles import (
    dominant_random_effects,
    fixed_effect_problem,
    groupwise_likelihood,
    native_arguments,
    observation_gradient,
    observation_likelihood,
)


@pytest.mark.parametrize("widths", [(16, 15), (17, 16)])
@pytest.mark.parametrize("fixed", [1, 16, 64])
@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
def test_wide_random_and_fixed_effect_products_match_observation_space(
    widths, fixed, coupled, diagonal, variance
):
    matrices, theta = fixed_effect_problem(widths, fixed, coupled, diagonal, variance)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, True)
    assert value == response.deviance(theta, True)
    assert_allclose(value, observation_likelihood(matrices, theta, True), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, True), rtol=2e-10, atol=2e-9)


@pytest.mark.parametrize("group_component", [0.1, 1.0, 100.0])
@pytest.mark.parametrize("theta_value", [1.0, 1e2, 1e4, 1e6, 1e8])
def test_fixed_effect_gradient_with_large_random_variance_matches_decimal_likelihood(
    group_component, theta_value
):
    matrices, groups = dominant_random_effects(True)
    # Give the fixed column a group component as well as within-group variation,
    # so its projected crossproducts can cancel when the variance is large.
    x = matrices.X[:, 0] + group_component * np.cos(groups)
    matrices = replace(matrices, X=x[:, None])
    theta = np.array([theta_value])
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, True)
    expected, _ = groupwise_likelihood(matrices, groups, theta_value, True)
    assert_allclose(value, expected, rtol=0, atol=2e-5)
    step = theta_value * 1e-4
    upper = groupwise_likelihood(matrices, groups, theta_value + step, True)[0]
    lower = groupwise_likelihood(matrices, groups, theta_value - step, True)[0]
    assert_allclose(theta_value * gradient[0], (upper - lower) / 2e-4, rtol=3e-6, atol=5e-7)

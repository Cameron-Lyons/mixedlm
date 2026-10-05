"""Wide fixed-effect corrections agree with observation-space derivatives."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose, assert_array_equal

from tests._lmm_oracles import (
    fixed_effect_problem,
    native_arguments,
    observation_gradient,
    observation_likelihood,
)


@pytest.mark.parametrize("widths", [(3, 2), (8, 5)])
@pytest.mark.parametrize("fixed", [0, 1, 8, 32, 64])
@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("reml", [False, True])
def test_fixed_effect_corrections_match_observation_space(
    widths, fixed, coupled, diagonal, variance, reml
):
    matrices, theta = fixed_effect_problem(widths, fixed, coupled, diagonal, variance)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert value == response.deviance(theta, reml)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_gradient_is_invariant_to_fixed_effect_rotation_and_scaling(coupled, reml):
    matrices, theta = fixed_effect_problem((8, 5), 32, coupled, False, "regular")
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    rotation, _ = np.linalg.qr(np.random.default_rng(912).normal(size=(32, 32)))
    scales = np.geomspace(1e-3, 1e3, 32)
    transformed = replace(matrices, X=(matrices.X @ rotation) * scales)
    other = _rust.LmmDesign(**native_arguments(transformed)).with_response(matrices.y)
    actual_value, actual_gradient = other.deviance_with_gradient(theta, reml)
    # The rotation has unit absolute determinant, and the scale product is one.
    assert_allclose(actual_value, value, rtol=2e-12, atol=2e-10)
    assert_allclose(actual_gradient, gradient, rtol=2e-10, atol=2e-9)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_wide_fixed_effect_gradients_match_scalar_finite_differences(coupled, reml):
    matrices, theta = fixed_effect_problem((3, 2), 32, coupled, False, "regular")
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    _, gradient = response.deviance_with_gradient(theta, reml)
    difference = np.empty_like(theta)
    for index in range(len(theta)):
        step = 1e-5 * max(1.0, abs(theta[index]))
        plus, minus = theta.copy(), theta.copy()
        plus[index] += step
        minus[index] -= step
        difference[index] = (response.deviance(plus, reml) - response.deviance(minus, reml)) / (
            2 * step
        )
    assert_allclose(gradient, difference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("coupled", [False, True])
def test_shared_design_keeps_wide_fixed_effect_workspaces_local(coupled):
    matrices, theta = fixed_effect_problem((8, 5), 32, coupled, False, "regular")
    design = _rust.LmmDesign(**native_arguments(matrices))
    responses = [design.with_response(y) for y in [matrices.y, matrices.y[::-1] + 0.2]]
    candidates = [theta * scale for scale in [0.0, 0.7, 1.3]]
    original = [candidate.copy() for candidate in candidates]

    def evaluate(index):
        return responses[index % 2].deviance_with_gradient(candidates[index % 3], True)

    expected = [evaluate(index) for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, range(12)))
    for (value, gradient), (reference_value, reference_gradient) in zip(
        actual, expected, strict=True
    ):
        assert value == reference_value
        assert_array_equal(gradient, reference_gradient)
    for candidate, initial in zip(candidates, original, strict=True):
        assert_array_equal(candidate, initial)

"""Shared mode adjoints preserve wide covariance gradients and local state."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_lmm_gradient_contractions import observation_gradient
from tests.test_lmm_prepared_design import native_arguments, observation_likelihood
from tests.test_lmm_reml_contractions import fixed_effect_problem


@pytest.mark.parametrize("fixed", [0, 8])
@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("reml", [False, True])
def test_wide_mode_gradients_match_observation_space(fixed, coupled, diagonal, variance, reml):
    matrices, theta = fixed_effect_problem((32, 24), fixed, coupled, diagonal, variance)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert value == response.deviance(theta, reml)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)


@pytest.mark.parametrize("coupled", [False, True])
def test_wide_mode_adjoints_are_local_to_each_response_and_call(coupled):
    matrices, theta = fixed_effect_problem((32, 24), 8, coupled, False, "regular")
    design = _rust.LmmDesign(**native_arguments(matrices))
    responses = [design.with_response(y) for y in [matrices.y, matrices.y[::-1] + 0.2]]
    candidates = [theta * scale for scale in [0.0, 0.7, 1.3]]
    original = [candidate.copy() for candidate in candidates]

    def evaluate(index):
        return responses[index % 2].deviance_with_gradient(candidates[index % 3], index % 4 < 2)

    expected = [evaluate(index) for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, range(12)))
    for (value, gradient), (expected_value, expected_gradient) in zip(
        actual, expected, strict=True
    ):
        assert value == expected_value
        assert_array_equal(gradient, expected_gradient)
    for candidate, unchanged in zip(candidates, original, strict=True):
        assert_array_equal(candidate, unchanged)
    assert np.all(np.isfinite([value for value, _ in actual]))

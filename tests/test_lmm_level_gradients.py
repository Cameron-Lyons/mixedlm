"""Independent grouping structures retain full ML and REML covariance gradients."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _profiled_deviance_core
from numpy.testing import assert_allclose, assert_array_equal

from tests._lmm_oracles import (
    native_arguments,
    observation_gradient,
    observation_likelihood,
    separate_structures,
)


@pytest.mark.parametrize("widths", [(1, 3), (3, 1), (15, 3), (16, 3), (17, 16)])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_separate_structures_match_observation_space(widths, independent, variance, fixed, reml):
    matrices, theta = separate_structures(widths, independent, variance, fixed)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    expected = _profiled_deviance_core(theta, matrices, reml)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)
    assert value == response.deviance(theta, reml)
    for actual, reference in zip(
        response.evaluate(theta, reml)[:4],
        [expected.deviance, expected.beta, expected.sigma, expected.u],
        strict=True,
    ):
        assert_allclose(actual, reference, rtol=2e-10, atol=2e-9)


@pytest.mark.parametrize("reml", [False, True])
def test_shared_independent_design_keeps_gradient_state_local(reml):
    matrices, theta = separate_structures((3, 2), False, "regular", True)
    design = _rust.LmmDesign(**native_arguments(matrices))
    responses = [design.with_response(y) for y in [matrices.y, matrices.y[::-1] + 0.2]]
    candidates = [theta * scale for scale in [0.0, 0.7, 1.3]]
    original = [candidate.copy() for candidate in candidates]

    def evaluate(index):
        return responses[index % 2].deviance_with_gradient(candidates[index % 3], reml)

    expected = [evaluate(index) for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, range(12)))
    for (value, gradient), (expected_value, expected_gradient) in zip(
        actual, expected, strict=True
    ):
        assert value == expected_value
        assert_array_equal(gradient, expected_gradient)
    for candidate, initial in zip(candidates, original, strict=True):
        assert_array_equal(candidate, initial)

"""Independent grouping structures retain full ML and REML covariance gradients."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _profiled_deviance_core
from mixedlm.matrices.design import RandomEffectStructure
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from tests.test_lmm_covariance_transforms import wide_problem
from tests.test_lmm_gradient_contractions import observation_gradient
from tests.test_lmm_prepared_design import native_arguments, observation_likelihood


def separate_structures(widths, independent, variance, fixed):
    left_width, right_width = widths
    matrices, theta, _ = wide_problem(left_width, independent, variance == "singular", fixed)
    rows = np.arange(matrices.n_obs)
    left = matrices.Z.toarray()
    left *= (rows[:, None] < matrices.n_obs // 2) & (
        rows[:, None] % 2 == np.arange(left.shape[1]) // left_width
    )
    right = np.random.default_rng(710).normal(scale=0.15, size=(matrices.n_obs, 3 * right_width))
    right *= (rows[:, None] >= matrices.n_obs // 2) & (
        rows[:, None] % 3 == np.arange(right.shape[1]) // right_width
    )
    lower = np.diag(np.linspace(0.3, 0.7, right_width))
    if independent:
        lower[np.tril_indices(right_width, -1)] = -0.02
    if variance == "singular":
        lower[:, -1] = 0
    theta = np.concatenate(
        (theta, lower[np.tril_indices(right_width)] if independent else lower.diagonal())
    )
    if variance == "zero":
        theta[:] = 0
    other = RandomEffectStructure(
        "other",
        [f"z{i}" for i in range(right_width)],
        3,
        right_width,
        independent,
        {"a": 0, "b": 1, "c": 2},
    )
    design = sparse.csc_matrix(np.column_stack((left, right)))
    matrices = replace(
        matrices,
        Z=design,
        n_random=design.shape[1],
        random_structures=[*matrices.random_structures, other],
    )
    return matrices, theta


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

"""Shared mode adjoints preserve wide covariance gradients and local state."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _build_lambda
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from tests._glmm_oracles import mode_problem
from tests._lmm_oracles import (
    decimal_mode_likelihood,
    fixed_effect_problem,
    native_arguments,
    observation_gradient,
    observation_likelihood,
)


@pytest.mark.parametrize(
    "parameters",
    [[1e8, 3e7, 0.0], [1e8, 3e7, 1e-4], [0.0, 1e8, 0.0], [1e4, -3e3, 1e2]],
)
@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("response_scale", [1e-6, 1.0, 1e6])
def test_shared_mode_products_retain_decimal_accuracy(parameters, width, coupled, response_scale):
    matrices, _, _ = mode_problem("gaussian", "slope", n_obs=32, n_groups=2)
    theta = np.array(parameters)
    if width == 3:
        z = matrices.Z.toarray()
        z = np.column_stack((z[:, :2], z[:, 1] ** 2, z[:, 2:], z[:, 3] ** 2))
        structure = replace(
            matrices.random_structures[0], n_terms=3, term_names=["Intercept", "x", "x2"]
        )
        matrices = replace(
            matrices, Z=sparse.csc_matrix(z), n_random=6, random_structures=[structure]
        )
        theta = np.r_[theta, [0.0, 0.0, 0.0]]
    if coupled:
        columns = np.roll(np.arange(matrices.n_random), width)
        matrices = replace(matrices, Z=(matrices.Z + 0.15 * matrices.Z[:, columns]).tocsc())
    factor = _build_lambda(theta, matrices.random_structures).toarray()
    signal = matrices.Z @ (factor @ np.sin(np.arange(matrices.n_random) + 1))
    signal *= 1e6 / max(np.max(np.abs(signal)), 1.0)
    matrices = replace(
        matrices,
        X=np.empty((matrices.n_obs, 0)),
        n_fixed=0,
        y=response_scale * (signal + 1e-3 * np.cos(np.arange(matrices.n_obs))) + matrices.offset,
    )
    expected, expected_gradient = decimal_mode_likelihood(matrices, theta)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    for reml in [False, True]:
        value, gradient = response.deviance_with_gradient(theta, reml)
        assert value == response.deviance(theta, reml)
        assert_allclose(value, expected, rtol=0, atol=2e-5)
        scale = np.maximum(np.abs(theta), 1)
        assert_allclose(scale * gradient, scale * expected_gradient, rtol=3e-6, atol=5e-7)


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

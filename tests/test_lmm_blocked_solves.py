"""Blocked native solves preserve wide gradients and difficult covariance scales."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _profiled_deviance_core
from numpy.testing import assert_allclose
from scipy import linalg, sparse

from tests._glmm_oracles import mode_problem
from tests._lmm_oracles import (
    dominant_random_effects,
    groupwise_likelihood,
    native_arguments,
    observation_gradient,
    observation_likelihood,
    parameters,
)


@pytest.mark.parametrize("layout", ["intercept", "mode_only", "slope", "crossed"])
@pytest.mark.parametrize("groups", [16, 64, 256])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_wide_blocked_solves_match_observation_space_and_python_estimates(
    layout, groups, overlap, singular, reml
):
    matrices, _, _ = mode_problem("gaussian", layout, n_obs=256, n_groups=groups)
    if overlap:
        columns = np.roll(np.arange(matrices.n_random), 2)
        matrices = replace(matrices, Z=(matrices.Z + 0.15 * matrices.Z[:, columns]).tocsc())
    theta = parameters(matrices)
    if singular:
        position = 0
        for structure in matrices.random_structures:
            width = structure.n_terms
            count = width * (width + 1) // 2 if structure.correlated else width
            theta[position + count - 1] = 0.0
            position += count
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    final = response.evaluate(theta, reml)
    expected = _profiled_deviance_core(theta, matrices, reml)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(value, response.deviance(theta, reml), rtol=0, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)
    for actual, reference in zip(
        final[:4], [expected.deviance, expected.beta, expected.sigma, expected.u], strict=True
    ):
        assert_allclose(actual, reference, rtol=2e-10, atol=2e-9)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("scale", [0.0, 1e2, 1e4, 1e6, 1e8])
def test_rotated_group_design_keeps_extreme_variance_profiles_accurate(fixed, reml, scale):
    matrices, groups = dominant_random_effects(fixed)
    # Orthogonal rotation preserves the observation covariance while making
    # unequal weighted group crossproducts dense in the latent coordinates.
    rotated = replace(matrices, Z=sparse.csc_matrix(matrices.Z @ (linalg.hadamard(4) / 2)))
    crossproduct = (rotated.Z.T @ rotated.Z.multiply(rotated.weights[:, None])).toarray()
    assert np.any(np.abs(crossproduct - np.diag(crossproduct.diagonal())) > 1e-6)
    response = _rust.LmmDesign(**native_arguments(rotated)).with_response(rotated.y)
    theta = np.array([scale])
    value, gradient = response.deviance_with_gradient(theta, reml)
    expected, sigma = groupwise_likelihood(matrices, groups, scale, reml)
    assert_allclose(value, expected, rtol=0, atol=2e-5)
    final = response.evaluate(theta, reml)
    assert_allclose(final[0], expected, rtol=0, atol=2e-5)
    assert_allclose(final[2], sigma, rtol=2e-7)
    if scale == 0:
        assert gradient[0] == 0.0
    else:
        upper = groupwise_likelihood(matrices, groups, scale * (1 + 1e-4), reml)[0]
        lower = groupwise_likelihood(matrices, groups, scale * (1 - 1e-4), reml)[0]
        assert_allclose(scale * gradient[0], (upper - lower) / 2e-4, rtol=3e-6, atol=5e-7)

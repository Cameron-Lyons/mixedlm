"""Mixed-width covariance assembly retains full cross-structure likelihoods."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _profiled_deviance_core
from mixedlm.matrices.design import RandomEffectStructure
from numpy.testing import assert_allclose
from scipy import sparse

from tests.test_lmm_covariance_transforms import wide_problem
from tests.test_lmm_gradient_contractions import observation_gradient
from tests.test_lmm_prepared_design import native_arguments, observation_likelihood


@pytest.mark.parametrize("widths", [(1, 3), (3, 1), (15, 3), (16, 3), (17, 16)])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("reml", [False, True])
def test_mixed_width_assembly_matches_observation_likelihood_and_gradient(
    widths, independent, overlap, variance, reml
):
    left_width, right_width = widths
    matrices, theta, _ = wide_problem(left_width, independent, variance == "singular")
    rng = np.random.default_rng(208)
    left = matrices.Z.toarray()
    right = rng.normal(scale=0.15, size=(matrices.n_obs, 3 * right_width))
    if not overlap:
        rows = np.arange(matrices.n_obs)
        left *= rows[:, None] % 2 == np.arange(left.shape[1]) // left_width
        right *= rows[:, None] % 3 == np.arange(right.shape[1]) // right_width
    lower = np.diag(np.linspace(0.3, 0.7, right_width))
    if independent:
        lower[np.tril_indices(right_width, -1)] = -0.02
    if variance == "singular":
        lower[:, -1] = 0
    other_theta = lower[np.tril_indices(right_width)] if independent else lower.diagonal()
    theta = np.concatenate((theta, other_theta))
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

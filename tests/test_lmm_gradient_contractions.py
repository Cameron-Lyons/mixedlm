"""Gradient contractions agree with independent observation-space derivatives."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose

from tests._glmm_oracles import mode_problem
from tests._lmm_oracles import (
    native_arguments,
    observation_gradient,
    observation_likelihood,
    parameters,
)


@pytest.mark.parametrize("layout", ["intercept", "mode_only", "slope", "crossed"])
@pytest.mark.parametrize("groups", [8, 33])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("reml", [False, True])
def test_contracted_gradients_preserve_all_levels_and_structures(
    layout, groups, overlap, variance, reml
):
    matrices, _, _ = mode_problem("gaussian", layout, n_obs=256, n_groups=groups)
    if overlap:
        columns = np.roll(np.arange(matrices.n_random), 2)
        matrices = replace(matrices, Z=(matrices.Z + 0.15 * matrices.Z[:, columns]).tocsc())
    theta = parameters(matrices)
    if variance == "zero":
        theta[:] = 0
    elif variance == "singular":
        position = 0
        for structure in matrices.random_structures:
            width = structure.n_terms
            count = width * (width + 1) // 2 if structure.correlated else width
            theta[position + count - 1] = 0
            position += count
    arguments = native_arguments(matrices)
    response = _rust.LmmDesign(**arguments).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert value == response.deviance(theta, reml)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)

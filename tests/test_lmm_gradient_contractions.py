"""Gradient contractions agree with independent observation-space derivatives."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _build_lambda
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

from tests.test_glmm_final_state import mode_problem
from tests.test_lmm_prepared_design import native_arguments, observation_likelihood, parameters


def observation_gradient(matrices, theta, reml):
    """Differentiate the observation covariance, including every grouping factor."""
    z = matrices.Z.toarray()
    transformed = z @ _build_lambda(theta, matrices.random_structures).toarray()
    covariance = np.diag(1 / matrices.weights) + transformed @ transformed.T
    precision = linalg.cho_solve(linalg.cho_factor(covariance), np.eye(matrices.n_obs))
    x, y = matrices.X, matrices.y - matrices.offset
    projector = precision
    if matrices.n_fixed:
        information = x.T @ precision @ x
        beta = linalg.solve(information, x.T @ precision @ y, assume_a="pos")
        residual = y - x @ beta
        if reml:
            projector = precision - precision @ x @ linalg.solve(information, x.T @ precision)
    else:
        residual = y
    projected = precision @ residual
    pwrss = residual @ projected
    df = matrices.n_obs - (matrices.n_fixed if reml else 0)
    score = projector - df / pwrss * np.outer(projected, projected)
    factor_score = 2 * z.T @ score @ transformed
    gradient, offset = [], 0
    for structure in matrices.random_structures:
        width = structure.n_terms
        positions = (
            [(i, j) for i in range(width) for j in range(i + 1)]
            if structure.correlated
            else [(i, i) for i in range(width)]
        )
        gradient.extend(
            sum(
                factor_score[offset + level * width + i, offset + level * width + j]
                for level in range(structure.n_levels)
            )
            for i, j in positions
        )
        offset += structure.n_levels * width
    return np.array(gradient)


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
    raw_value, raw_gradient = _rust.profiled_deviance_with_gradient(
        theta=theta, y=matrices.y, reml=reml, **arguments
    )
    assert value == raw_value == response.deviance(theta, reml)
    assert_array_equal(gradient, raw_gradient)
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)

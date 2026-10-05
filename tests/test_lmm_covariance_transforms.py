"""Repeated factor transforms retain likelihoods and observation-space gradients."""

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import LMMOptimizer
from numpy.testing import assert_allclose
from scipy import linalg

from tests._lmm_oracles import direct_profiled_likelihood, native_arguments, wide_problem


def observation_gradient(matrices, factor, reml):
    """Differentiate the full observation covariance, independent of block solves."""
    z = matrices.Z.toarray()
    transformed = z @ factor
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
    structure = matrices.random_structures[0]
    width = structure.n_terms
    positions = (
        [(i, j) for i in range(width) for j in range(i + 1)]
        if structure.correlated
        else [(i, i) for i in range(width)]
    )
    return np.array(
        [
            sum(factor_score[level * width + i, level * width + j] for level in range(2))
            for i, j in positions
        ]
    )


@pytest.mark.parametrize("width", [3, 15, 16, 17, 32])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_wide_lmm_transforms_match_observation_system(width, independent, singular, fixed, reml):
    matrices, theta, factor = wide_problem(width, independent, singular, fixed)
    expected = direct_profiled_likelihood(theta, matrices, reml)
    prepared = LMMOptimizer(matrices, REML=reml, use_rust=True)
    actual = prepared._final_evaluation(theta)
    for field, value in expected.items():
        assert_allclose(getattr(actual, field), value, rtol=2e-11, atol=2e-10)
    assert prepared.objective(theta) == actual.deviance
    value, gradient = (
        _rust.LmmDesign(**native_arguments(matrices))
        .with_response(matrices.y)
        .deviance_with_gradient(theta, reml)
    )
    assert_allclose(value, expected["deviance"], rtol=2e-12, atol=2e-11)
    assert_allclose(gradient, observation_gradient(matrices, factor, reml), rtol=2e-10, atol=2e-10)

"""Overlapping level columns retain their covariance in native LMM profiles."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import LMMOptimizer, _profiled_deviance_core
from numpy.testing import assert_allclose

from tests._glmm_oracles import covariance_problem, mode_problem
from tests._lmm_oracles import direct_profiled_likelihood, native_arguments, parameters


@pytest.mark.parametrize(
    "layout", ["intercept", "correlated", "diagonal", "mixed", "crossed_slopes", "no_fixed"]
)
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_overlapping_levels_match_observation_covariance(layout, variance, weighted, reml):
    matrices, theta, _ = covariance_problem(layout, variance, weighted, overlap=True)
    prepared = LMMOptimizer(matrices, REML=reml, use_rust=True)
    for y in [matrices.y, matrices.y[::-1] + 0.3 * matrices.weights]:
        response = prepared.with_response(y)
        expected = direct_profiled_likelihood(theta, replace(matrices, y=y), reml)
        assert_allclose(response.objective(theta), expected["deviance"], rtol=1e-12, atol=1e-11)
        actual = response._final_evaluation(theta)
        for field, value in expected.items():
            assert_allclose(getattr(actual, field), value, rtol=2e-12, atol=2e-11)
        assert response._rust_cache.design is prepared._rust_cache.design


@pytest.mark.parametrize(
    "layout", ["intercept", "correlated", "diagonal", "mixed", "crossed_slopes", "no_fixed"]
)
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("reml", [False, True])
def test_overlapping_level_gradients_match_independent_likelihood(layout, variance, reml):
    matrices, theta, _ = covariance_problem(layout, variance, True, overlap=True)
    value, gradient = (
        _rust.LmmDesign(**native_arguments(matrices))
        .with_response(matrices.y)
        .deviance_with_gradient(theta, reml)
    )
    expected = direct_profiled_likelihood(theta, matrices, reml)["deviance"]
    assert_allclose(value, expected, rtol=1e-12, atol=1e-11)
    step = 2e-5
    difference = []
    for index in range(len(theta)):
        upper, lower = theta.copy(), theta.copy()
        upper[index] += step
        lower[index] -= step
        difference.append(
            (
                direct_profiled_likelihood(upper, matrices, reml)["deviance"]
                - direct_profiled_likelihood(lower, matrices, reml)["deviance"]
            )
            / (2 * step)
        )
    assert_allclose(gradient, difference, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("reml", [False, True])
def test_overlapping_levels_do_not_create_spurious_factorization_failure(reml):
    matrices, _, _ = mode_problem("gaussian", "slope", n_obs=8192, n_groups=64)
    columns = np.roll(np.arange(matrices.n_random), 2)
    matrices = replace(matrices, Z=(matrices.Z + 0.15 * matrices.Z[:, columns]).tocsc())
    theta = parameters(matrices)
    expected = _profiled_deviance_core(theta, matrices, reml)
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    actual = optimizer._final_evaluation(theta)
    for field in vars(expected):
        assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-11, atol=2e-10)
    assert_allclose(optimizer.objective(theta), expected.deviance, rtol=2e-12)
    value, gradient = (
        _rust.LmmDesign(**native_arguments(matrices))
        .with_response(matrices.y)
        .deviance_with_gradient(theta, reml)
    )
    assert_allclose(value, expected.deviance, rtol=2e-12)
    assert np.all(np.isfinite(gradient))

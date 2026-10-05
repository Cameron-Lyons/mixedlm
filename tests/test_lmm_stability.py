"""Stable profiled objectives agree with groupwise Gaussian likelihoods."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import LMMOptimizer, _build_lambda
from numpy.testing import assert_allclose
from scipy import linalg

from tests._glmm_oracles import mode_problem
from tests._lmm_oracles import (
    decimal_mode_likelihood,
    dominant_random_effects,
    groupwise_likelihood,
    native_arguments,
)


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("response_scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("theta_value", [0.0, 1e2, 1e4, 1e6, 1e8])
def test_dominant_random_effects_retain_finite_accurate_profile(
    reml, fixed, response_scale, theta_value
):
    matrices, groups = dominant_random_effects(fixed, response_scale)
    theta = np.array([theta_value])
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    expected, sigma = groupwise_likelihood(matrices, groups, theta_value, reml)
    actual = optimizer.objective(theta)
    assert np.isfinite(actual)
    assert_allclose(actual, expected, rtol=0, atol=2e-5)
    final = optimizer._final_evaluation(theta)
    assert actual == final.deviance
    assert_allclose(final.sigma, sigma, rtol=2e-7)
    gradient_value, gradient = (
        _rust.LmmDesign(**native_arguments(matrices))
        .with_response(matrices.y)
        .deviance_with_gradient(theta, reml)
    )
    assert_allclose(gradient_value, actual, rtol=0, atol=1e-10)
    assert np.all(np.isfinite(gradient))
    if theta_value == 0:
        assert gradient[0] == 0.0
    else:
        step = theta_value * 1e-4
        upper = groupwise_likelihood(matrices, groups, theta_value + step, reml)[0]
        lower = groupwise_likelihood(matrices, groups, theta_value - step, reml)[0]
        assert_allclose(theta_value * gradient[0], (upper - lower) / 2e-4, rtol=3e-6, atol=5e-7)


@pytest.mark.parametrize("variance_scale", [1e2, 1e4, 1e6, 1e8])
@pytest.mark.parametrize("independent", [False, True])
def test_large_random_slopes_match_augmented_least_squares(variance_scale, independent):
    matrices, _, _ = mode_problem("gaussian", "slope", n_obs=64, n_groups=4)
    for structure in matrices.random_structures:
        structure.correlated = not independent
    theta = variance_scale * np.array([1.0, 0.7] if independent else [1.0, -0.3, 0.7])
    matrices = replace(matrices, X=np.empty((matrices.n_obs, 0)), n_fixed=0)
    signal = matrices.Z @ (1e6 * np.sin(np.arange(matrices.n_random)))
    matrices = replace(
        matrices, y=signal + 1e-3 * np.cos(np.arange(matrices.n_obs)) + matrices.offset
    )
    factor = _build_lambda(theta, matrices.random_structures).toarray()
    weighted_design = np.sqrt(matrices.weights)[:, None] * (matrices.Z @ factor)
    augmented = np.vstack((weighted_design, np.eye(matrices.n_random)))
    target = np.concatenate(
        (np.sqrt(matrices.weights) * (matrices.y - matrices.offset), np.zeros(matrices.n_random))
    )
    # QR of the augmented observation design avoids the marginal quadratic
    # subtraction and the normal-equation factorization used by the native path.
    q, r = linalg.qr(augmented, mode="economic")
    spherical = linalg.solve_triangular(r, q.T @ target)
    residual = target - augmented @ spherical
    pwrss = np.dot(residual, residual)
    logdet = 2 * np.log(np.abs(r.diagonal())).sum()
    expected = matrices.n_obs * (1 + np.log(2 * np.pi * pwrss / matrices.n_obs))
    expected += logdet - np.log(matrices.weights).sum()
    optimizer = LMMOptimizer(matrices, use_rust=True)
    actual = optimizer.objective(theta)
    assert np.isfinite(actual)
    assert_allclose(actual, expected, rtol=0, atol=2e-5)
    assert actual == optimizer._final_evaluation(theta).deviance
    gradient_value, gradient = (
        _rust.LmmDesign(**native_arguments(matrices))
        .with_response(matrices.y)
        .deviance_with_gradient(theta)
    )
    assert_allclose(gradient_value, actual, rtol=0, atol=1e-10)
    # Here the factor is invertible and well conditioned. Recover the projected
    # conditional residual from the mode normal equation, avoiding subtraction
    # when the random effects explain almost the entire response.
    projected = linalg.solve_triangular(factor.T, spherical)
    expected_gradient = []
    positions = [(0, 0), (1, 1)] if independent else [(0, 0), (1, 0), (1, 1)]
    for i, j in positions:
        derivative = np.zeros_like(factor)
        for group in range(4):
            derivative[2 * group + i, 2 * group + j] = 1.0
        d_design = np.sqrt(matrices.weights)[:, None] * (matrices.Z @ derivative)
        d_precision = d_design.T @ weighted_design + weighted_design.T @ d_design
        d_logdet = np.trace(linalg.cho_solve((r, False), d_precision))
        d_pwrss = -2 * projected @ derivative @ spherical
        expected_gradient.append(d_logdet + matrices.n_obs / pwrss * d_pwrss)
    assert_allclose(
        variance_scale * gradient,
        variance_scale * np.asarray(expected_gradient),
        rtol=3e-6,
        atol=5e-7,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        [0.0, 0.0, 0.0],
        [0.8, 0.3, 0.0],
        [1e8, 0.0, 0.0],
        [1e8, 3e7, 0.0],
        [1e8, 3e7, 1e-4],
        [0.0, 1e8, 0.0],
    ],
)
def test_singular_and_near_singular_factors_match_decimal_gradient(parameters):
    matrices, _, _ = mode_problem("gaussian", "slope", n_obs=32, n_groups=2)
    theta = np.array(parameters)
    factor = _build_lambda(theta, matrices.random_structures).toarray()
    signal = matrices.Z @ (factor @ np.sin(np.arange(matrices.n_random) + 1))
    signal *= 1e6 / max(np.max(np.abs(signal)), 1.0)
    matrices = replace(
        matrices,
        X=np.empty((matrices.n_obs, 0)),
        n_fixed=0,
        y=signal + 1e-3 * np.cos(np.arange(matrices.n_obs)) + matrices.offset,
    )
    expected, expected_gradient = decimal_mode_likelihood(matrices, theta)
    value, gradient = (
        _rust.LmmDesign(**native_arguments(matrices))
        .with_response(matrices.y)
        .deviance_with_gradient(theta)
    )
    optimizer = LMMOptimizer(matrices, use_rust=True)
    assert_allclose(value, expected, rtol=0, atol=2e-5)
    assert_allclose(value, optimizer.objective(theta), rtol=0, atol=1e-10)
    scale = np.maximum(np.abs(theta), 1)
    assert_allclose(scale * gradient, scale * expected_gradient, rtol=3e-6, atol=5e-7)

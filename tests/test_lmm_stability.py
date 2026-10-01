"""Stable profiled objectives agree with groupwise Gaussian likelihoods."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import LMMOptimizer, _build_lambda
from numpy.testing import assert_allclose
from scipy import linalg

from tests.test_glmm_final_state import mode_problem
from tests.test_lmm_prepared_design import native_arguments


def dominant_random_effects(fixed, response_scale=1.0):
    matrices, _, _ = mode_problem("gaussian", "mode_only", n_obs=64, n_groups=4)
    row = np.arange(matrices.n_obs)
    groups = row % 4
    x = np.sin(row * 0.17)
    if fixed:
        # Keep the fixed column distinct from group intercepts even at large theta.
        means = np.bincount(groups, weights=matrices.weights * x) / np.bincount(
            groups, weights=matrices.weights
        )
        x -= means[groups]
        matrices = replace(matrices, X=x[:, None], n_fixed=1, fixed_names=["x"])
    fixed_part = 0.3 * x if fixed else 0.0
    response = response_scale * (1e6 * np.sin(groups) + fixed_part + 1e-3 * np.cos(row))
    return replace(matrices, y=response + matrices.offset), groups


def groupwise_likelihood(matrices, groups, theta, reml):
    """High-precision within/between-group decomposition, without normal matrices."""
    with localcontext() as context:
        context.prec = 60
        y = [Decimal.from_float(float(v)) for v in matrices.y - matrices.offset]
        weights = [Decimal.from_float(float(v)) for v in matrices.weights]
        x = (
            [Decimal.from_float(float(v)) for v in matrices.X[:, 0]]
            if matrices.n_fixed
            else [Decimal(0)] * matrices.n_obs
        )
        variance = Decimal.from_float(float(theta)) ** 2
        blocks = []
        information, rhs = Decimal(0), Decimal(0)
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            total = sum(weights[i] for i in indices)
            x_mean = sum(weights[i] * x[i] for i in indices) / total
            y_mean = sum(weights[i] * y[i] for i in indices) / total
            precision = 1 + variance * total
            information += sum(weights[i] * (x[i] - x_mean) ** 2 for i in indices)
            information += total * x_mean**2 / precision
            rhs += sum(weights[i] * (x[i] - x_mean) * (y[i] - y_mean) for i in indices)
            rhs += total * x_mean * y_mean / precision
            blocks.append((indices, total, x_mean, y_mean, precision))
        beta = rhs / information if matrices.n_fixed else Decimal(0)
        pwrss = Decimal(0)
        for indices, total, x_mean, y_mean, precision in blocks:
            mean = y_mean - x_mean * beta
            pwrss += sum(weights[i] * (y[i] - x[i] * beta - mean) ** 2 for i in indices)
            pwrss += total * mean**2 / precision
        df = matrices.n_obs - matrices.n_fixed if reml else matrices.n_obs
        deviance = df * (1 + math.log(2 * math.pi * float(pwrss) / df))
        deviance += math.fsum(math.log(float(block[-1])) for block in blocks)
        deviance -= math.fsum(math.log(float(weight)) for weight in weights)
        if reml and matrices.n_fixed:
            deviance += math.log(float(information))
        return deviance, math.sqrt(float(pwrss) / df)


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
    gradient_value, gradient = _rust.profiled_deviance_with_gradient(
        theta=theta, y=matrices.y, reml=reml, **native_arguments(matrices)
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
    gradient_value, gradient = _rust.profiled_deviance_with_gradient(
        theta=theta, y=matrices.y, **native_arguments(matrices)
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


def decimal_mode_likelihood(matrices, theta):
    """Small, high-precision penalized solve that also permits singular factors."""
    with localcontext() as context:
        context.prec = 60

        def decimal_array(values):
            return np.vectorize(lambda value: Decimal.from_float(float(value)))(values)

        z = decimal_array(matrices.Z.toarray())
        factor = decimal_array(_build_lambda(theta, matrices.random_structures).toarray())
        weights = decimal_array(matrices.weights)
        y = decimal_array(matrices.y - matrices.offset)
        design = z @ factor
        size = matrices.n_random
        identity = decimal_array(np.eye(size))
        precision = design.T @ (weights[:, None] * design) + identity
        # LDL decomposition uses only Decimal arithmetic, including the solve.
        lower = identity.copy()
        diagonal = []
        for i in range(size):
            diagonal.append(precision[i, i] - sum(lower[i, k] ** 2 * diagonal[k] for k in range(i)))
            for j in range(i + 1, size):
                lower[j, i] = (
                    precision[j, i] - sum(lower[j, k] * lower[i, k] * diagonal[k] for k in range(i))
                ) / diagonal[i]

        def solve(rhs):
            forward = []
            for i in range(size):
                forward.append(rhs[i] - sum(lower[i, k] * forward[k] for k in range(i)))
            result = [value / scale for value, scale in zip(forward, diagonal, strict=True)]
            for i in reversed(range(size)):
                result[i] -= sum(lower[k, i] * result[k] for k in range(i + 1, size))
            return np.array(result)

        spherical = solve(design.T @ (weights * y))
        residual = y - design @ spherical
        pwrss = sum(weights * residual**2) + sum(spherical**2)
        logdet = sum(math.log(float(value)) for value in diagonal)
        deviance = matrices.n_obs * (1 + math.log(2 * math.pi * float(pwrss) / matrices.n_obs))
        deviance += logdet - np.log(matrices.weights).sum()
        inverse = np.column_stack([solve(column) for column in identity])
        gradient = []
        for parameter in range(len(theta)):
            basis = np.zeros_like(theta)
            basis[parameter] = 1
            derivative = decimal_array(_build_lambda(basis, matrices.random_structures).toarray())
            d_design = z @ derivative
            d_precision = d_design.T @ (weights[:, None] * design)
            d_precision += design.T @ (weights[:, None] * d_design)
            d_logdet = np.trace(inverse @ d_precision)
            d_pwrss = -2 * sum(weights * residual * (d_design @ spherical))
            gradient.append(float(d_logdet + matrices.n_obs / pwrss * d_pwrss))
        return deviance, np.asarray(gradient)


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
    value, gradient = _rust.profiled_deviance_with_gradient(
        theta=theta, y=matrices.y, **native_arguments(matrices)
    )
    optimizer = LMMOptimizer(matrices, use_rust=True)
    assert_allclose(value, expected, rtol=0, atol=2e-5)
    assert_allclose(value, optimizer.objective(theta), rtol=0, atol=1e-10)
    scale = np.maximum(np.abs(theta), 1)
    assert_allclose(scale * gradient, scale * expected_gradient, rtol=3e-6, atol=5e-7)

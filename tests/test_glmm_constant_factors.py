"""Constant-weight Gaussian solves retain fresh factors and likelihood corrections."""

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_lmm_covariance_transforms import wide_problem
from tests.test_native_covariance_transforms import _args


@pytest.mark.parametrize("width", [2, 17, 32])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("maxiter", [1, 100])
def test_constant_factors_match_joint_system_across_parameters_and_offsets(
    width, fixed, singular, maxiter
):
    matrices, theta, factor = wide_problem(width, False, singular, fixed=fixed)
    arguments = _args(matrices, theta, "gaussian")
    problem = _rust.GlmmProblem(*arguments[:8], *arguments[9:])
    p, q = matrices.n_fixed, matrices.n_random
    for scale, change in [(1.0, 0.0), (0.0, 0.2), (1.7, -0.1), (1.0, 0.0)]:
        offset = matrices.offset + change * np.cos(np.arange(matrices.n_obs))
        lower = factor * scale
        design = np.column_stack((matrices.X, matrices.Z @ lower))
        information = design.T @ (matrices.weights[:, None] * design)
        information[p:, p:] += np.eye(q)
        coefficients = np.linalg.solve(
            information, design.T @ (matrices.weights * (matrices.y - offset))
        )
        residual = matrices.y - offset - design @ coefficients
        expected = (
            np.dot(matrices.weights * residual, residual)
            + coefficients[p:] @ coefficients[p:]
            + np.linalg.slogdet(information[p:, p:])[1]
        )
        actual = problem.evaluate(theta * scale, offset=offset, maxiter=maxiter, tol=1e-10)
        fresh = _rust.glmm_deviance(
            *arguments[:7],
            offset,
            theta * scale,
            *arguments[9:],
            1,
            maxiter=maxiter,
            tol=1e-10,
        )
        for value, reference in zip(actual, fresh, strict=True):
            assert_array_equal(value, reference)
        assert_allclose(actual[0], expected, rtol=2e-12, atol=2e-11)
        assert_allclose(actual[1], coefficients[:p], rtol=2e-11, atol=2e-11)
        assert_allclose(actual[2], lower @ coefficients[p:], rtol=2e-11, atol=2e-11)
        if maxiter == 100:
            assert actual[3]

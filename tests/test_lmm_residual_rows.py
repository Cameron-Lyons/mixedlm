"""Residual traversal agrees with fresh native designs and independent profiles."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import LMMOptimizer, _profiled_deviance_core
from numpy.testing import assert_allclose, assert_array_equal

from tests._lmm_oracles import (
    direct_profiled_likelihood,
    native_arguments,
    parameters,
    residual_problem,
)


@pytest.mark.parametrize("layout", ["intercept", "slope", "crossed", "mode_only"])
@pytest.mark.parametrize("size", [64, 1024])
@pytest.mark.parametrize("pattern", ["disjoint", "overlap", "empty_rows"])
@pytest.mark.parametrize("reml", [False, True])
def test_prepared_profiles_retain_residual_arithmetic(layout, size, pattern, reml):
    matrices = residual_problem(layout, size, pattern)
    theta = parameters(matrices)
    arguments = native_arguments(matrices)
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    for y in [matrices.y, matrices.y[::-1] + 0.2 * matrices.weights]:
        response = optimizer.with_response(y)
        actual = response._final_evaluation(theta)
        assert response.objective(theta) == actual.deviance
        # A design prepared for this response alone preserves the arithmetic exactly.
        fresh = _rust.LmmDesign(**arguments).with_response(y)
        assert_array_equal(fresh.deviance(theta, reml), actual.deviance)
        changed = replace(matrices, y=y)
        expected = (
            direct_profiled_likelihood(theta, changed, reml)
            if size == 64
            else vars(_profiled_deviance_core(theta, changed, reml))
        )
        for field, value in expected.items():
            assert_allclose(getattr(actual, field), value, rtol=2e-11, atol=2e-9)
        gradient_value, gradient = fresh.deviance_with_gradient(theta, reml)
        assert_allclose(gradient_value, actual.deviance, rtol=0, atol=2e-10)
        assert np.all(np.isfinite(gradient))

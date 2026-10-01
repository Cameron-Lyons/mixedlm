"""Changing weights reuse row layouts without changing fitted GLMM states."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.laplace import laplace_deviance, pirls
from mixedlm.families import Binomial, Poisson
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_lmm_covariance_transforms import wide_problem
from tests.test_native_covariance_transforms import _args
from tests.test_native_weighted_crossproducts import _noncanonical


def wide_count_problem(width, fixed, variance, family_name, noncanonical=False):
    matrices, theta, _ = wide_problem(width, False, variance == "singular", fixed)
    rng = np.random.default_rng(371)
    eta = (
        matrices.offset
        + matrices.X @ np.linspace(0.1, 0.3, matrices.n_fixed)
        + matrices.Z @ rng.normal(scale=0.1, size=matrices.n_random)
    )
    y = (
        rng.poisson(np.exp(eta))
        if family_name == "poisson"
        else rng.binomial(1, 1 / (1 + np.exp(-eta)))
    )
    matrices = replace(matrices, y=y.astype(float))
    if noncanonical:
        matrices = replace(matrices, Z=_noncanonical(matrices.Z.tocsc()))
    if variance == "zero":
        theta[:] = 0.0
    return matrices, theta


@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_wide_changing_weights_match_python(width, fixed, variance, family_name, noncanonical):
    matrices, theta = wide_count_problem(width, fixed, variance, family_name, noncanonical)
    family = Poisson() if family_name == "poisson" else Binomial()
    args = _args(matrices, theta, family_name)
    actual = _rust.pirls(*args, maxiter=100, tol=1e-9)
    expected = pirls(matrices, family, theta, maxiter=100, tol=1e-9)
    assert actual[3] and expected[3]
    for value, reference in zip(actual[:3], expected[:3], strict=True):
        assert_allclose(value, reference, rtol=2e-8, atol=2e-8)
    actual = _rust.laplace_deviance(*args, maxiter=100, tol=1e-9)
    expected = laplace_deviance(theta, matrices, family, pirls_maxiter=100, pirls_tol=1e-9)
    for value, reference in zip(actual, expected, strict=True):
        assert_allclose(value, reference, rtol=2e-8, atol=2e-8)
    problem = _rust.GlmmProblem(*args[:8], *args[9:])
    # Warm the same design with different weights before returning to this state.
    assert problem.evaluate(theta * 0.5, maxiter=100, tol=1e-9)[3]
    prepared = problem.evaluate(theta, maxiter=100, tol=1e-9)
    assert prepared[3]
    for value, reference in zip(prepared[:3], actual, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
def test_first_row_layout_can_be_shared_across_threads(family_name):
    matrices, theta = wide_count_problem(32, True, "regular", family_name)
    args = _args(matrices, theta, family_name)
    problem = _rust.GlmmProblem(*args[:8], *args[9:])
    cases = [(theta * scale, matrices.offset + scale / 10) for scale in [0, 0.7, 1.3, 2.0]]
    expected = [
        _rust.glmm_deviance(*args[:7], offset, current, *args[9:], 1, maxiter=100, tol=1e-9)
        for current, offset in cases
    ]
    barrier = Barrier(4)

    def evaluate(index):
        current, offset = cases[index]
        barrier.wait(timeout=20)
        return problem.evaluate(current, offset=offset, maxiter=100, tol=1e-9)

    indices = list(range(4)) * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, indices))
    for index, result in zip(indices, actual, strict=True):
        assert result[3]
        for value, reference in zip(result, expected[index], strict=True):
            assert_array_equal(value, reference)

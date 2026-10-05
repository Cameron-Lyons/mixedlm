"""Native final estimates reuse preparation and retain a stable residual scale."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.estimation.reml import LMMOptimizer, _LMMCrossproducts, _profiled_deviance_core
from numpy.testing import assert_allclose, assert_array_equal

from tests._glmm_oracles import mode_problem
from tests._lmm_oracles import direct_profiled_likelihood, matrices_fixture, parameters


@pytest.mark.parametrize(
    "kind", ["fixed", "no_fixed", "intercept", "correlated", "independent", "slope", "crossed"]
)
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("scale", [0, 0.7, 1.3])
def test_native_final_state_matches_independent_marginal_likelihood(kind, reml, scale):
    matrices = matrices_fixture("correlated" if kind == "independent" else kind)
    if kind == "independent":
        for structure in matrices.random_structures:
            structure.correlated = False
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    theta = parameters(matrices) * scale
    for y in (matrices.y, matrices.y[::-1] + 0.3 * matrices.weights):
        response = optimizer.with_response(y)
        expected = direct_profiled_likelihood(theta, replace(matrices, y=y), reml)
        serial = response._final_evaluation(theta)
        for field, value in expected.items():
            assert_allclose(getattr(serial, field), value, rtol=2e-12, atol=2e-11)
        python = _profiled_deviance_core(theta, replace(matrices, y=y), reml)
        assert_allclose([serial.ldL2, serial.ldRX2], [python.ldL2, python.ldRX2], atol=2e-12)
        assert serial.pwrss == serial.wrss + serial.ussq
        if matrices.n_random:
            assert "_crossproducts" not in response.__dict__
            assert response.objective(theta) == serial.deviance
        native = response._rust_cache.response.evaluate(theta, reml)
        for value, field in zip(native, vars(serial), strict=True):
            actual = np.asarray(value)
            if field == "fixed_information":
                actual = actual.reshape(matrices.n_fixed, matrices.n_fixed)
            assert_allclose(actual, getattr(serial, field), rtol=2e-12, atol=2e-11)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(response._final_evaluation, [theta] * 8))
        for result in results:
            for field in vars(serial):
                assert_array_equal(getattr(result, field), getattr(serial, field))


@pytest.mark.parametrize("reml", [False, True])
def test_native_final_state_matches_large_sparse_system(reml):
    matrices, _, _ = mode_problem("gaussian", "slope", n_obs=1024, n_groups=256)
    theta = parameters(matrices)
    expected = _profiled_deviance_core(theta, matrices, reml)
    actual = LMMOptimizer(matrices, REML=reml, use_rust=True)._final_evaluation(theta)
    for field in vars(expected):
        assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-11, atol=2e-10)


@pytest.mark.parametrize("reml", [False, True])
def test_final_scale_uses_conditional_residuals_when_random_effects_dominate(reml):
    n, groups = 64, 4
    matrices, _, _ = mode_problem("gaussian", "mode_only", n_obs=n, n_groups=groups)
    row = np.arange(n)
    group = row % groups
    y = 1e6 * np.sin(group) + 1e-3 * np.cos(row) + matrices.offset
    matrices = replace(matrices, y=y)
    theta = np.array([1e8])
    weight_sums = np.bincount(group, weights=matrices.weights)
    response_sums = np.bincount(group, weights=matrices.weights * (y - matrices.offset))
    random = theta[0] ** 2 * response_sums / (1 + theta[0] ** 2 * weight_sums)
    residual = y - matrices.offset - random[group]
    wrss = np.dot(matrices.weights * residual, residual)
    ussq = np.dot(random / theta[0], random / theta[0])
    pwrss = wrss + ussq
    actual = LMMOptimizer(matrices, REML=reml, use_rust=True)._final_evaluation(theta)
    assert np.isfinite(actual.deviance)
    assert actual.sigma > 0
    assert_allclose(actual.u, random, rtol=1e-14, atol=1e-9)
    assert_allclose(actual.pwrss, pwrss, rtol=2e-7, atol=1e-12)
    assert_allclose(actual.sigma, np.sqrt(pwrss / n), rtol=1e-7)
    assert actual.pwrss == actual.wrss + actual.ussq


@pytest.mark.parametrize("reml", [False, True])
def test_native_optimization_and_response_refit_do_not_build_python_products(reml):
    matrices = matrices_fixture("correlated")
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    with patch.object(_LMMCrossproducts, "from_matrices", side_effect=AssertionError("recomputed")):
        first = optimizer.optimize()
        repeated = optimizer.with_response(matrices.y[::-1])
        second = repeated.optimize(start=first.theta)
    assert first.converged and second.converged
    assert repeated._rust_cache.design is optimizer._rust_cache.design
    assert "_crossproducts" not in optimizer.__dict__
    assert "_crossproducts" not in repeated.__dict__


def test_explicit_python_crossproducts_remain_shared_across_native_refits():
    matrices = matrices_fixture()
    optimizer = LMMOptimizer(matrices, use_rust=True)
    products = optimizer._crossproducts
    repeated = optimizer.with_response(matrices.y[::-1])
    assert repeated._crossproducts.XtWX is products.XtWX
    assert repeated._crossproducts.ZtWZ is products.ZtWZ
    assert repeated._crossproducts.ZtWX is products.ZtWX
    expected = _profiled_deviance_core(
        parameters(matrices), repeated.matrices, crossproducts=repeated._crossproducts
    )
    actual = repeated._final_evaluation(parameters(matrices))
    assert_allclose(actual.deviance, expected.deviance, rtol=1e-13)


def test_returned_estimates_do_not_modify_prepared_response():
    matrices = matrices_fixture("correlated")
    optimizer = LMMOptimizer(matrices, use_rust=True)
    theta = parameters(matrices)
    expected = optimizer._final_evaluation(theta)
    actual = optimizer._final_evaluation(theta)
    actual.beta[:] = 0
    actual.u[:] = 0
    actual.fixed_information[:] = 0
    repeated = optimizer._final_evaluation(theta)
    for field in vars(expected):
        assert_array_equal(getattr(repeated, field), getattr(expected, field))


# A duplicated column makes the fixed-effect information exactly singular; whether
# SciPy warns about it, and with which message, depends on its version and rounding.
@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_fixed_only_rank_deficient_design_retains_python_fallback(reml, duplicate):
    matrices = matrices_fixture("fixed")
    x = np.column_stack((matrices.X[:, 0], matrices.X[:, 0]))
    if not duplicate:
        x[:, 1] = 0
    matrices = replace(matrices, X=x, n_fixed=2)
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    expected = _profiled_deviance_core(np.array([]), matrices, reml)
    if not np.isfinite(expected.deviance):
        with pytest.raises(RuntimeError, match="valid fit"):
            optimizer._final_evaluation(np.array([]))
    else:
        actual = optimizer._final_evaluation(np.array([]))
        for field in vars(expected):
            assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize(
    "position,value,label",
    [
        (0, np.nan, "deviance"),
        (1, [np.inf] * 3, "fixed effects"),
        (2, 0.0, "residual scale"),
        (3, [np.nan] * 14, "random effects"),
    ],
)
def test_native_final_estimates_pass_existing_fit_validation(position, value, label):
    matrices = matrices_fixture("correlated")
    optimizer = LMMOptimizer(matrices, use_rust=True)
    theta = parameters(matrices)
    native = list(optimizer._rust_cache.response.evaluate(theta))
    native[position] = value
    optimizer._rust_cache.response = SimpleNamespace(evaluate=lambda *args: native)
    with pytest.raises(RuntimeError, match=label):
        optimizer._final_evaluation(theta)

"""Prepared covariance gradients reuse owned design state across calls and threads."""

import gc
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.estimation.reml import _build_theta_bounds
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize

from tests.test_lmm_prepared_design import (
    matrices_fixture,
    native_arguments,
    observation_likelihood,
    parameters,
)
from tests.test_lmm_stability import dominant_random_effects, groupwise_likelihood
from tests.test_native_covariance_transforms import _problem


@pytest.mark.parametrize(
    "layout",
    ["fixed", "no_fixed", "intercept", "correlated", "diagonal", "mixed", "crossed_slopes"],
)
@pytest.mark.parametrize("variance", ["regular", "singular", "zero"])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_prepared_gradients_match_fresh_designs_and_independent_likelihood(
    layout, variance, overlap, reml
):
    matrices, theta, _ = _problem(layout, variance, True, overlap=overlap)
    arguments = native_arguments(matrices)
    design = _rust.LmmDesign(**arguments)
    for y in [matrices.y, matrices.y[::-1] + 0.2 * matrices.weights]:
        response = design.with_response(y)
        value, gradient = response.deviance_with_gradient(theta, reml)
        fresh = _rust.LmmDesign(**arguments).with_response(y)
        fresh_value, fresh_gradient = fresh.deviance_with_gradient(theta, reml)
        assert value == fresh_value == response.deviance(theta, reml)
        assert_array_equal(gradient, fresh_gradient)
        changed = replace(matrices, y=y)
        assert_allclose(value, observation_likelihood(changed, theta, reml), rtol=2e-12, atol=2e-11)
        expected = []
        for index in range(len(theta)):
            step = np.zeros_like(theta)
            step[index] = 1e-5
            expected.append(
                (
                    observation_likelihood(changed, theta + step, reml)
                    - observation_likelihood(changed, theta - step, reml)
                )
                / 2e-5
            )
        assert_allclose(gradient, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize(
    "fixed,reml,scale", list(product([False, True], [False, True], [0.0, 1e4, 1e8]))
)
def test_prepared_gradients_preserve_extreme_variance_profiles(fixed, reml, scale):
    matrices, groups = dominant_random_effects(fixed)
    theta = np.array([scale])
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    expected, _ = groupwise_likelihood(matrices, groups, scale, reml)
    assert_allclose(value, expected, rtol=0, atol=2e-5)
    if scale == 0:
        assert gradient[0] == 0
    else:
        lower = groupwise_likelihood(matrices, groups, scale * (1 - 1e-4), reml)[0]
        upper = groupwise_likelihood(matrices, groups, scale * (1 + 1e-4), reml)[0]
        assert_allclose(scale * gradient[0], (upper - lower) / 2e-4, rtol=3e-6, atol=5e-7)


def test_prepared_gradient_owns_inputs_and_returns_independent_arrays():
    matrices = matrices_fixture("correlated")
    arguments = native_arguments(matrices)
    design = _rust.LmmDesign(**arguments)
    y = matrices.y.copy()
    response = design.with_response(y)
    theta = parameters(matrices)
    value, gradient = response.deviance_with_gradient(theta)
    expected = gradient.copy()
    gradient[:] = np.nan
    for array in arguments.values():
        if isinstance(array, np.ndarray):
            array[:] = 0
    y[:] = 1000
    del design
    gc.collect()
    later_value, later_gradient = response.deviance_with_gradient(theta)
    assert value == later_value
    assert_array_equal(later_gradient, expected)
    assert not np.shares_memory(gradient, later_gradient)


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("kind", ["fixed", "correlated", "crossed"])
def test_shared_responses_keep_concurrent_gradients_independent(kind, reml):
    matrices = matrices_fixture(kind)
    design = _rust.LmmDesign(**native_arguments(matrices))
    ys = [matrices.y, matrices.y[::-1] + 0.3 * matrices.weights]
    responses = [design.with_response(y) for y in ys]
    theta = parameters(matrices)
    cases = [(response, theta * scale) for response in responses for scale in [0, 0.7, 1.3]]
    expected = [response.deviance_with_gradient(current, reml) for response, current in cases]

    def evaluate(index):
        response, current = cases[index % len(cases)]
        if index % 5 == 0:
            with pytest.raises(ValueError, match="theta"):
                response.deviance_with_gradient(np.append(current, np.nan), reml)
        return response.deviance_with_gradient(current, reml)

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, range(len(cases) * 4)))
    for index, (value, gradient) in enumerate(actual):
        expected_value, expected_gradient = expected[index % len(cases)]
        assert value == expected_value
        assert_array_equal(gradient, expected_gradient)


@pytest.mark.parametrize("theta", [[], [1.0, 2.0], [np.nan], [np.inf], [-np.inf]])
def test_prepared_gradients_reject_invalid_parameters(theta):
    matrices = matrices_fixture("intercept")
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    with pytest.raises(ValueError, match="theta"):
        response.deviance_with_gradient(np.asarray(theta))


@pytest.mark.parametrize("kind", ["fixed", "correlated"])
@pytest.mark.parametrize("reml", [False, True])
def test_prepared_gradients_preserve_factorization_failure_results(kind, reml):
    matrices = matrices_fixture(kind)
    arguments = native_arguments(matrices)
    arguments["x"][:] = 0
    response = _rust.LmmDesign(**arguments).with_response(matrices.y)
    value, gradient = response.deviance_with_gradient(parameters(matrices), reml)
    assert value == 1e10
    assert_array_equal(gradient, np.zeros_like(parameters(matrices)))


def test_prepared_gradient_rejects_invalid_reml_degrees_of_freedom():
    matrices = matrices_fixture("fixed")
    arguments = native_arguments(matrices)
    arguments["x"] = np.eye(matrices.n_obs)
    response = _rust.LmmDesign(**arguments).with_response(matrices.y)
    with pytest.raises(ValueError, match="REML"):
        response.deviance_with_gradient(np.array([]))


@pytest.mark.parametrize("change_layout", [False, True])
def test_prepared_gradient_detaches_and_snapshots_parameters(change_layout):
    # A separate interpreter keeps the long switch interval out of the runner.
    script = textwrap.dedent("""
        import json
        import sys
        import threading
        import numpy as np
        from scipy import sparse
        from mixedlm import _rust

        assert getattr(sys, "_is_gil_enabled", lambda: True)(), "This test requires the GIL"
        n = 3_000_000
        z = sparse.csc_matrix((np.ones(n), np.arange(n), [0, n]), shape=(n, 1))
        design = _rust.LmmDesign(
            np.empty((n, 0)), z.data, z.indices.astype(np.int64),
            z.indptr.astype(np.int64), z.shape, np.linspace(.5, 2, n),
            np.sin(np.arange(n)) / 5, [1], [1], [True],
        )
        response = design.with_response(np.random.default_rng(713).normal(size=n))
        theta = np.array([.8])
        expected_value, expected_gradient = response.deviance_with_gradient(theta)
        assert np.isfinite(expected_value) and np.all(np.isfinite(expected_gradient))
        started, finished = threading.Event(), threading.Event()
        outcome = []

        def evaluate():
            started.set()
            try:
                outcome.append(response.deviance_with_gradient(theta))
            finally:
                finished.set()

        sys.setswitchinterval(60)
        thread = threading.Thread(target=evaluate)
        thread.start()
        assert started.wait(10)
        progressed = not finished.is_set()
        theta[:] = 2
        thread.join(30)
        assert not thread.is_alive()
        value, gradient = outcome[0]
        assert value == expected_value
        np.testing.assert_array_equal(gradient, expected_gradient)
        print(json.dumps({'progressed': progressed}))
    """)
    if change_layout:
        script = script.replace("theta[:] = 2", "theta[:] = 2; theta.shape = ()")
    # Test GIL release explicitly, regardless of the parent's current GIL state.
    env = dict(
        os.environ,
        PYTHON_GIL="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    run = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=50
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["progressed"]


@pytest.mark.parametrize("kind", ["intercept", "correlated", "crossed"])
@pytest.mark.parametrize("reml", [False, True])
def test_prepared_gradient_drives_the_same_scipy_fit_as_fresh_designs(kind, reml):
    matrices = matrices_fixture(kind)
    arguments = native_arguments(matrices)
    response = _rust.LmmDesign(**arguments).with_response(matrices.y)
    theta = parameters(matrices)
    bounds = _build_theta_bounds(matrices.random_structures, len(theta))
    options = {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40}
    prepared = minimize(
        lambda current: response.deviance_with_gradient(current, reml),
        theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=options,
    )
    fresh = minimize(
        lambda current: _rust.LmmDesign(**arguments)
        .with_response(matrices.y)
        .deviance_with_gradient(current, reml),
        theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=options,
    )
    assert prepared.success and fresh.success
    assert prepared.nit == fresh.nit
    assert prepared.nfev == fresh.nfev
    assert_array_equal(prepared.x, fresh.x)
    assert_array_equal(prepared.jac, fresh.jac)
    assert prepared.fun == fresh.fun
    assert_allclose(prepared.fun, observation_likelihood(matrices, prepared.x, reml), rtol=2e-12)

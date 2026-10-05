"""Prepared linear likelihoods preserve independent threaded evaluations."""

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_lmm_prepared_design import (
    matrices_fixture,
    native_arguments,
    observation_likelihood,
    parameters,
)

pytestmark = pytest.mark.installed_wheel


@pytest.mark.parametrize(
    "kind", ["fixed", "no_fixed", "intercept", "correlated", "slope", "crossed"]
)
@pytest.mark.parametrize("reml", [False, True])
def test_shared_design_and_responses_match_independent_likelihoods_in_threads(kind, reml):
    matrices = matrices_fixture(kind)
    design = _rust.LmmDesign(**native_arguments(matrices))
    ys = [matrices.y, matrices.y[::-1] + 0.3 * matrices.weights]
    responses = [design.with_response(y) for y in ys]
    theta = parameters(matrices)
    cases = [(index, theta * scale) for index in range(2) for scale in [0, 0.7, 1.3]]
    expected = [responses[index].deviance(current, reml) for index, current in cases]
    for (index, current), value in zip(cases, expected, strict=True):
        reference = observation_likelihood(replace(matrices, y=ys[index]), current, reml)
        assert_allclose(value, reference, rtol=2e-13, atol=2e-12)

    def evaluate(case):
        index, current = cases[case]
        return responses[index].deviance(current, reml)

    order = list(range(len(cases))) * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, order))
    assert_array_equal(actual, [expected[index] for index in order])


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("method", ["deviance", "evaluate"])
def test_invalid_parameters_do_not_damage_shared_response(reml, method):
    matrices = matrices_fixture()
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    theta = parameters(matrices)
    likelihood = getattr(response, method)
    expected = likelihood(theta, reml)

    def evaluate(index):
        if index % 2:
            with pytest.raises(ValueError, match="theta must contain"):
                likelihood(np.array([np.nan]), reml)
        else:
            assert likelihood(theta, reml) == expected

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate, range(16)))
    assert likelihood(theta, reml) == expected


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("method,expected", [("deviance", 1e10), ("evaluate", None)])
def test_detached_singular_system_preserves_failure_value(reml, method, expected):
    matrices = matrices_fixture("fixed")
    matrices = replace(matrices, X=np.zeros_like(matrices.X))
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda _: getattr(response, method)(np.array([]), reml), range(16)))
    assert actual == [expected] * 16


@pytest.mark.parametrize("change_layout", [False, True])
@pytest.mark.parametrize("method", ["deviance", "evaluate"])
def test_evaluation_releases_interpreter_lock_and_snapshots_parameters(change_layout, method):
    # Isolate the long switch interval from the test runner. A large random-
    # intercept problem gives the waiting thread time to run during one solve.
    script = textwrap.dedent("""
        import json
        import sys
        import threading
        import numpy as np
        from scipy import sparse
        from mixedlm import _rust

        assert getattr(sys, "_is_gil_enabled", lambda: True)(), "This test requires the GIL"
        rng = np.random.default_rng(713)
        n = 3_000_000
        z = sparse.csc_matrix((np.ones(n), np.arange(n), np.array([0, n])), shape=(n, 1))
        design = _rust.LmmDesign(
            np.empty((n, 0)), z.data, z.indices.astype(np.int64),
            z.indptr.astype(np.int64), z.shape, np.linspace(.5, 2, n),
            np.sin(np.arange(n)) / 5, [1], [1], [True],
        )
        response = design.with_response(rng.normal(size=n))
        theta = np.array([.8])
        likelihood = getattr(response, sys.argv[1])
        expected = likelihood(theta)
        objective = expected if sys.argv[1] == 'deviance' else expected[0]
        assert np.isfinite(objective) and objective != 1e10
        started = threading.Event()
        finished = threading.Event()
        outcome = []

        def evaluate():
            started.set()
            try:
                outcome.append(likelihood(theta))
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
        assert outcome == [expected]
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
    result = subprocess.run(
        [sys.executable, "-c", script, method], env=env, text=True, capture_output=True, timeout=50
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["progressed"], result.stdout

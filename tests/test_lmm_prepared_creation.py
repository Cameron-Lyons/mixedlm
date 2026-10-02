"""Native design preprocessing uses independent owned snapshots in threads."""

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize(
    "kind", ["fixed", "no_fixed", "intercept", "correlated", "slope", "crossed"]
)
@pytest.mark.parametrize("reml", [False, True])
def test_concurrent_design_construction_matches_independent_likelihood(kind, reml):
    matrices = matrices_fixture(kind)
    arguments = native_arguments(matrices)
    theta = parameters(matrices)
    expected = observation_likelihood(matrices, theta, reml)

    def evaluate(_):
        design = _rust.LmmDesign(**arguments)
        return design.with_response(matrices.y).deviance(theta, reml)

    serial = evaluate(0)
    assert_allclose(serial, expected, rtol=2e-13, atol=2e-12)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, range(16)))
    assert_array_equal(actual, [serial] * 16)


@pytest.mark.parametrize("field", ["x", "weights", "offset"])
def test_detached_design_validation_errors_leave_other_constructions_usable(field):
    matrices = matrices_fixture("correlated")
    valid = native_arguments(matrices)
    invalid = native_arguments(matrices)
    invalid[field].flat[0] = np.nan
    theta = parameters(matrices)
    expected = _rust.LmmDesign(**valid).with_response(matrices.y).deviance(theta)

    def evaluate(index):
        if index % 2:
            with pytest.raises(ValueError, match="finite"):
                _rust.LmmDesign(**invalid)
        else:
            assert _rust.LmmDesign(**valid).with_response(matrices.y).deviance(theta) == expected

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate, range(16)))


@pytest.mark.parametrize("change_layout", [False, True])
def test_creation_releases_interpreter_lock_and_snapshots_all_arrays(change_layout):
    script = textwrap.dedent("""
        import json
        import sys
        import threading
        import numpy as np
        from scipy import sparse
        from mixedlm import _rust

        assert getattr(sys, "_is_gil_enabled", lambda: True)(), "This test requires the GIL"
        rng = np.random.default_rng(275)
        n, groups = 1_000_000, 16
        x = np.column_stack((np.ones(n), rng.normal(size=(n, 3))))
        rows = np.arange(n)
        z = sparse.coo_matrix((np.ones(n), (rows, rows % groups)), shape=(n, groups)).tocsc()
        arrays = [x, z.data, z.indices.astype(np.int64), z.indptr.astype(np.int64),
                  np.linspace(.5, 2, n), .2 * np.sin(rows)]
        y = x @ np.array([.3, -.2, .1, .05]) + rng.normal(size=n)
        theta = np.array([.8])

        def create():
            return _rust.LmmDesign(*arrays[:4], z.shape, *arrays[4:],
                                   [groups], [1], [True])

        expected = create().with_response(y).deviance(theta)
        assert np.isfinite(expected) and expected != 1e10
        started = threading.Event()
        finished = threading.Event()
        outcome = []

        def prepare():
            started.set()
            try:
                outcome.append(create())
            finally:
                finished.set()

        sys.setswitchinterval(60)
        thread = threading.Thread(target=prepare)
        thread.start()
        assert started.wait(10)
        progressed = not finished.is_set()
        for array in arrays:
            array.fill(0)
        thread.join(30)
        assert not thread.is_alive()
        assert len(outcome) == 1
        assert outcome[0].with_response(y).deviance(theta) == expected
        print(json.dumps({'progressed': progressed}))
    """)
    if change_layout:
        script = script.replace(
            "array.fill(0)",
            "array.fill(0); array.shape = (array.size,) if array.ndim == 2 else (array.size, 1)",
        )
    # Test GIL release explicitly, regardless of the parent's current GIL state.
    env = dict(
        os.environ,
        PYTHON_GIL="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=50
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["progressed"], result.stdout

"""Sparse native work releases the GIL while preserving owned input snapshots."""

import json
import os
import subprocess
import sys
from pathlib import Path

import mixedlm
import pytest

pytestmark = pytest.mark.installed_wheel

OPERATIONS = ["analysis", "factor", "cached_solve", "solve", "logdet"]

# The long switch interval makes progress proof deterministic: the main thread
# resumes before the worker finishes only if native work releases the GIL. It
# then rewrites and reshapes every input while that work is still running.
CHECK_OPERATIONS = """
import json
import sys
import threading
import traceback

import numpy as np
from mixedlm import _rust
from tests._sparse_systems import arrowhead_system, sparse_arguments

assert getattr(sys, '_is_gil_enabled', lambda: True)(), 'This test requires the GIL'
sys.setswitchinterval(60)


def check(operation):
    # Natural ordering fills the hub system densely, so a small system already
    # gives factorization and cached solves ample native work.
    natural = operation in ('factor', 'cached_solve')
    n = 1024 if natural else 60_000
    matrix, base_rhs, _, _ = arrowhead_system(n)
    data, indices, offsets = sparse_arguments(matrix, 'full')
    rhs = np.tile(base_rhs[:, :1], (1, 64))
    symbolic = _rust.SparseCholeskySymbolic(indices, offsets, n, ordering='natural') \\
        if natural else None
    numeric = symbolic.factor(data) if operation == 'cached_solve' else None

    def call():
        if operation == 'analysis':
            return _rust.SparseCholeskySymbolic(indices, offsets, n, ordering='amd')
        if operation == 'factor':
            return symbolic.factor(data)
        if operation == 'cached_solve':
            return numeric.solve(rhs)
        if operation == 'solve':
            return _rust.sparse_cholesky_solve(data, indices, offsets, matrix.shape, rhs)
        return _rust.sparse_cholesky_logdet(data, indices, offsets, matrix.shape)

    expected = call()
    started = threading.Event()
    finished = threading.Event()
    outcome = []
    errors = []

    def evaluate():
        started.set()
        try:
            outcome.append(call())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=evaluate)
    worker.start()
    assert started.wait(10)
    progressed = not finished.is_set()
    data[:] = 7
    indices[:] = 0
    offsets[:] = 0
    rhs[:] = 9
    data.shape = (data.size, 1)
    indices.shape = (indices.size, 1)
    offsets.shape = (offsets.size, 1)
    rhs.shape = (rhs.size, 1)
    worker.join(30)
    assert not worker.is_alive()
    assert not errors, errors
    assert len(outcome) == 1
    if operation == 'analysis':
        assert outcome[0].factor_nonzeros() == expected.factor_nonzeros()
        factor = outcome[0].factor(matrix.data)
        np.testing.assert_allclose(matrix @ factor.solve(base_rhs), base_rhs, atol=2e-12)
    elif operation == 'factor':
        np.testing.assert_array_equal(outcome[0].solve(base_rhs), expected.solve(base_rhs))
        assert outcome[0].logdet() == expected.logdet()
    else:
        np.testing.assert_array_equal(outcome[0], expected)
    return progressed


results = {}
for operation in sys.argv[1:]:
    try:
        results[operation] = {'progressed': check(operation)}
    except Exception:
        results[operation] = {'error': traceback.format_exc()}
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def thread_results():
    root = Path(__file__).resolve().parents[1]
    python = Path(mixedlm.__file__).resolve().parents[1]
    # Test GIL release explicitly, regardless of the parent's current GIL state.
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(python), str(root)]),
        PYTHON_GIL="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    run = subprocess.run(
        [sys.executable, "-c", CHECK_OPERATIONS, *OPERATIONS],
        env=env,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_sparse_work_releases_interpreter_lock_and_snapshots_inputs(thread_results, operation):
    result = thread_results[operation]
    assert "error" not in result, result["error"]
    assert result["progressed"]

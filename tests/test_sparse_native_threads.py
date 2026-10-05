"""Sparse native work releases the GIL while preserving owned input snapshots."""

import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("operation", ["analysis", "factor", "cached_solve", "solve", "logdet"])
@pytest.mark.parametrize("change_layout", [False, True])
def test_sparse_work_releases_interpreter_lock_and_snapshots_inputs(operation, change_layout):
    # The long switch interval makes progress proof deterministic: the parent
    # resumes only when native work explicitly releases the interpreter lock.
    script = textwrap.dedent("""
        import json
        import sys
        import threading
        import numpy as np
        from mixedlm import _rust
        from tests.test_sparse_ordering import arrowhead_system, sparse_arguments

        assert getattr(sys, '_is_gil_enabled', lambda: True)(), 'This test requires the GIL'
        operation = sys.argv[1]
        n = 2048 if operation in ('analysis', 'factor', 'cached_solve') else 60_000
        matrix, base_rhs, _, _ = arrowhead_system(n)
        data, indices, offsets = sparse_arguments(matrix, 'full')
        rhs = np.tile(base_rhs[:, :1], (1, 64))
        symbolic = _rust.SparseCholeskySymbolic(indices, offsets, n, ordering='natural') \
            if operation in ('factor', 'cached_solve') else None
        numeric = symbolic.factor(data) if operation == 'cached_solve' else None

        def call():
            if operation == 'analysis':
                return _rust.SparseCholeskySymbolic(indices, offsets, n, ordering='natural')
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

        sys.setswitchinterval(60)
        worker = threading.Thread(target=evaluate)
        worker.start()
        assert started.wait(10)
        progressed = not finished.is_set()
        data[:] = 7
        indices[:] = 0
        offsets[:] = 0
        rhs[:] = 9
        if sys.argv[2] == 'True':
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
        print(json.dumps({'progressed': progressed}))
    """)
    env = dict(os.environ, PYTHON_GIL="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", script, operation, str(change_layout)],
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["progressed"], result.stdout

"""Prepared native solves can share immutable inputs across Python threads."""

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import mixedlm
import pytest
from mixedlm.estimation.laplace import _native_glmm_args, _prepare_native_glmm
from numpy.testing import assert_array_equal

from tests.test_glmm_final_state import mode_problem

native = pytest.importorskip("mixedlm._rust")

pytestmark = pytest.mark.installed_wheel


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize(
    ("layout", "order"),
    [
        ("intercept", 1),
        ("intercept", 7),
        ("slope", 1),
        ("crossed", 1),
        ("mode_only", 1),
        ("mode_only", 7),
    ],
)
def test_shared_problem_matches_stateless_evaluations_in_threads(kind, layout, order):
    matrices, family, theta = mode_problem(kind, layout, n_obs=2048, n_groups=128)
    problem = _prepare_native_glmm(matrices, family)
    cases = [(theta * scale, matrices.offset + scale / 10) for scale in [0, 0.7, 1.3]]
    # A zero final variance changes the sparse design pattern shared by the problem.
    boundary = theta.copy()
    boundary[-1] = 0.0
    cases.append((boundary, matrices.offset - 0.05))
    expected = []
    for current, offset in cases:
        args = list(_native_glmm_args(current, matrices, family))
        args[7] = offset
        expected.append(native.glmm_deviance(*args, order, tol=1e-10))

    def evaluate(index):
        current, offset = cases[index]
        return problem.evaluate(current, order, offset=offset, tol=1e-10)

    indices = list(range(len(cases))) * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, indices))
    for index, result in zip(indices, actual, strict=True):
        for value, reference in zip(result, expected[index], strict=True):
            assert_array_equal(value, reference)


def test_detached_errors_do_not_damage_shared_problem():
    matrices, family, theta = mode_problem("poisson", "intercept", n_obs=2048, n_groups=64)
    overlapping = matrices.Z.tolil()
    overlapping[0, 1] = 0.1
    matrices = replace(matrices, Z=overlapping.tocsc())
    problem = _prepare_native_glmm(matrices, family)
    expected = native.glmm_deviance(*_native_glmm_args(theta, matrices, family), 1)

    def evaluate(order):
        if order == 7:
            with pytest.raises(ValueError, match="at most one nonzero"):
                problem.evaluate(theta, order)
        else:
            for value, reference in zip(problem.evaluate(theta, order), expected, strict=True):
                assert_array_equal(value, reference)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate, [7, 1] * 8))
    for value, reference in zip(problem.evaluate(theta), expected, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("entry", ["prepared", "stateless"])
@pytest.mark.parametrize("change_layout", [False, True])
def test_evaluation_releases_interpreter_lock_and_snapshots_parameters(entry, change_layout):
    # Isolate the long switch interval from the test runner and other tests.
    script = f"ENTRY = {entry!r}\n" + textwrap.dedent("""
        import json
        import sys
        import threading
        import numpy as np
        from mixedlm import _rust
        from mixedlm.estimation.laplace import _native_glmm_args, _prepare_native_glmm
        from tests.test_glmm_final_state import mode_problem

        assert getattr(sys, "_is_gil_enabled", lambda: True)(), "This test requires the GIL"
        matrices, family, theta = mode_problem('poisson', 'mode_only', n_obs=65536, n_groups=64)
        problem = _prepare_native_glmm(matrices, family)
        backing = np.zeros(2 * matrices.n_obs)
        backing[::2] = matrices.offset + 0.2
        offset = backing[::2]
        args = list(_native_glmm_args(theta, matrices, family))
        args[7:9] = offset, theta

        def solve():
            if ENTRY == "prepared":
                return problem.evaluate(theta, 31, offset=offset)
            return _rust.glmm_deviance(*args, 31)

        expected = solve()
        started = threading.Event()
        finished = threading.Event()
        outcome = []

        def evaluate():
            started.set()
            try:
                outcome.append(solve())
            finally:
                finished.set()

        sys.setswitchinterval(60)
        thread = threading.Thread(target=evaluate)
        thread.start()
        assert started.wait(10)
        progressed = not finished.is_set()
        theta[:] = 2
        backing[:] = 4
        thread.join(20)
        assert not thread.is_alive()
        assert len(outcome) == 1
        for actual, reference in zip(outcome[0], expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
        print(json.dumps({'progressed': progressed}))
    """)
    if change_layout:
        script = script.replace("theta[:] = 2", "theta[:] = 2; theta.shape = ()")
    root = Path(__file__).resolve().parents[1]
    python = Path(mixedlm.__file__).resolve().parents[1]
    # Test GIL release explicitly, regardless of the parent's current GIL state.
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(python), str(root)]),
        PYTHON_GIL="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["progressed"], result.stdout

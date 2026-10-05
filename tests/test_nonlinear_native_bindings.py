"""Native NLMM bindings validate shapes, snapshot their inputs and release the GIL."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from mixedlm import _rust


def micmen_arguments():
    x = np.tile([0.2, 0.5, 1.0, 2.0, 3.0, 5.0], 3)
    return {
        "theta": np.array([0.4]),
        "y": 2.8 * x / (0.9 + x),
        "x": x,
        "groups": np.repeat([4, -1, 9], 6),
        "model_name": "ssmicmen",
        "phi": np.array([2.0, 1.2]),
        "b": np.zeros((3, 1)),
        "random_params": [0],
        "sigma": 0.3,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"phi": np.array([2.0, 1.2, 0.5])}, r"phi has length 3, expected 2"),
        ({"random_params": [2]}, r"random_params must index the model's 2 parameters"),
        ({"b": np.zeros((4, 1))}, r"b has shape \(4, 1\), expected \(3, 1\)"),
        ({"b": np.zeros((3, 2))}, r"b has shape \(3, 2\), expected \(3, 1\)"),
        ({"theta": np.array([0.4, 0.1, 0.3])}, r"does not define a 1 x 1 covariance factor"),
        ({"x": np.ones(17)}, r"x has length 17, expected 18"),
        ({"groups": np.zeros(17, dtype=np.int64)}, r"groups has length 17, expected 18"),
    ],
)
def test_inconsistent_shapes_raise_value_errors(change, message):
    # These used to panic inside the solver; short groups silently dropped an observation.
    with pytest.raises(ValueError, match=message):
        _rust.nlmm_deviance_with_status(**{**micmen_arguments(), **change})


def test_objective_releases_interpreter_lock_and_snapshots_inputs():
    # A separate interpreter keeps the long switch interval out of the runner.
    script = textwrap.dedent("""
        import json
        import sys
        import threading
        import numpy as np
        from mixedlm import _rust

        assert getattr(sys, "_is_gil_enabled", lambda: True)(), "This test requires the GIL"
        n_groups, per = 4000, 10
        x = np.tile(np.linspace(0.0, 5.0, per), n_groups)
        groups = np.repeat(np.arange(n_groups), per)
        level = 10 + np.repeat(np.linspace(-1.0, 1.0, n_groups), per)
        y = level + (0.5 - level) * np.exp(-np.exp(-0.5) * x) + np.sin(7 * x) / 10
        weights = np.linspace(0.5, 2.0, y.size)
        theta, phi, b = np.array([1.0]), np.array([10.0, 0.5, -0.5]), np.zeros((n_groups, 1))

        def call():
            # A tolerance below rounding error runs every PNLS iteration.
            return _rust.nlmm_deviance_with_status(
                theta, y, x, groups, "ssasymp", phi, b, [0], 0.3, weights,
                maxiter=100, tol=1e-300,
            )

        expected = call()
        started, finished = threading.Event(), threading.Event()
        outcome = []

        def evaluate():
            started.set()
            try:
                outcome.append(call())
            finally:
                finished.set()

        sys.setswitchinterval(60)
        thread = threading.Thread(target=evaluate)
        thread.start()
        assert started.wait(10)
        progressed = not finished.is_set()
        for array in (theta, y, x, groups, phi, b, weights):
            array[...] = 7
            array.shape = (array.size, 1)
        thread.join(30)
        assert not thread.is_alive()
        assert len(outcome) == 1
        for actual, reference in zip(outcome[0], expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
        print(json.dumps({"progressed": progressed}))
    """)
    # Test GIL release explicitly, regardless of the parent's current GIL state.
    env = dict(
        os.environ,
        PYTHON_GIL="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    run = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=60
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["progressed"]

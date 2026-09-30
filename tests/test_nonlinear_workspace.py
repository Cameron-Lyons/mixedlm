from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.estimation import nlmm
from mixedlm.nlme.models import CustomModel


def linear_problem(random_params=(0,)):
    rng = np.random.default_rng(764)
    groups = np.repeat(np.arange(4), 6)
    x = np.column_stack([np.ones(len(groups)), np.tile(np.linspace(-1, 1, 6), 4)])
    q = len(random_params)
    y = x @ [1.3, -0.4] + rng.normal(0, 0.2, len(groups))
    order = rng.permutation(len(groups))
    model = CustomModel(lambda p, x: x @ p, lambda p, x: x.copy(), ["intercept", "slope"])
    factor = np.array([[0.12]]) if q == 1 else np.array([[0.12, 0], [0.03, 0.18]])
    return {
        "y": y[order],
        "x": x[order],
        "groups": np.array([-20, 0, 5, 800])[groups[order]],
        "model": model,
        "phi": np.array([1.3, -0.4]),
        "b": np.zeros((4, q)),
        "random_params": list(random_params),
        "sigma": 0.3,
        "weights": np.linspace(0.4, 2.0, len(y))[order],
    }, factor


def linear_oracle(data, factor):
    """Solve the penalized normal equations jointly, without PNLS iterations."""
    y, x, weights = data["y"], data["x"], data["weights"]
    _, labels = np.unique(data["groups"], return_inverse=True)
    q = factor.shape[0]
    z = np.zeros((len(y), 4 * q))
    for row, group in enumerate(labels):
        z[row, group * q : (group + 1) * q] = x[row, data["random_params"]]
    covariance = factor @ factor.T
    precision = np.linalg.inv(covariance + 1e-8 * np.eye(q))
    design = np.column_stack([x, z])
    normal = design.T @ (weights[:, None] * design)
    normal[2:, 2:] += np.kron(np.eye(4), precision)
    solution = np.linalg.solve(normal, design.T @ (weights * y))
    residuals = y - design @ solution
    penalty = solution[2:] @ np.kron(np.eye(4), precision) @ solution[2:]
    sigma_sq = (weights @ residuals**2 + penalty) / len(y)
    block_factor = np.kron(np.eye(4), factor)
    correction = np.linalg.slogdet(
        np.eye(4 * q) + block_factor.T @ (z.T @ (weights[:, None] * z)) @ block_factor
    )[1]
    deviance = len(y) * (1 + np.log(2 * np.pi * sigma_sq)) + correction
    return deviance, solution[:2], solution[2:].reshape(4, q), np.sqrt(sigma_sq)


@pytest.mark.parametrize("random_params", [(0,), (1, 0), (0, 0)])
@pytest.mark.parametrize("jobs", [1, 2, -1, 8])
def test_weighted_nonlinear_evaluation_matches_joint_linear_solution(random_params, jobs):
    data, factor = linear_problem(random_params)
    expected = linear_oracle(data, factor)
    inputs = {name: value.copy() for name, value in data.items() if isinstance(value, np.ndarray)}
    with patch.object(nlmm.os, "cpu_count", return_value=2):
        actual = nlmm.nlmm_deviance(factor[np.tril_indices(len(factor))], **data, n_jobs=jobs)
        pnls = nlmm.pnls_step(**data, Psi=factor @ factor.T, n_jobs=jobs)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-6)
    for value, reference in zip(pnls, actual[1:], strict=True):
        np.testing.assert_array_equal(value, reference)
    for name, value in inputs.items():
        np.testing.assert_array_equal(data[name], value)


@pytest.fixture
def pools(monkeypatch):
    created = []

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(nlmm, "ThreadPoolExecutor", RecordingExecutor)
    return created


def run_entry(entry, data, jobs):
    if entry == "pnls":
        return nlmm.pnls_step(**data, Psi=np.eye(1), n_jobs=jobs)
    if entry == "deviance":
        return nlmm.nlmm_deviance(np.array([0.4]), **data, n_jobs=jobs)
    data = data.copy()
    starts = {f"start_{name}": data.pop(name) for name in ("phi", "b", "sigma")}
    optimizer = nlmm.NLMMOptimizer(**data, n_jobs=jobs, use_rust=False)
    result = optimizer.optimize(**starts, maxiter=3)
    assert optimizer._workspace is None
    return result


@pytest.mark.parametrize("entry", ["pnls", "deviance", "fit"])
@pytest.mark.parametrize("jobs", [1, 2, -1, 8])
def test_worker_pool_and_group_preparation_are_scoped_to_call(entry, jobs, pools):
    data, _ = linear_problem()
    with (
        patch.object(nlmm.os, "cpu_count", return_value=2),
        patch.object(
            nlmm, "_grouped_observation_indices", wraps=nlmm._grouped_observation_indices
        ) as group,
    ):
        run_entry(entry, data, jobs)
    assert group.call_count == 1
    assert len(pools) == (1 if jobs in (2, -1) else 0)
    for pool in pools:
        assert pool._shutdown
        assert not any(thread.is_alive() for thread in pool._threads)


@pytest.mark.parametrize("entry", ["pnls", "deviance", "fit"])
def test_worker_pool_is_closed_after_model_failure(entry, pools):
    data, _ = linear_problem()

    def fail(params, x):
        raise ValueError("outside model domain")

    data["model"].predict = fail
    error = RuntimeError if entry == "fit" else ValueError
    with pytest.raises(error, match="outside model domain"):
        run_entry(entry, data, 2)
    assert len(pools) == 1
    assert pools[0]._shutdown
    assert not any(thread.is_alive() for thread in pools[0]._threads)


def test_interrupted_fit_closes_pool_and_releases_workspace(pools):
    data, _ = linear_problem()
    starts = {f"start_{name}": data.pop(name) for name in ("phi", "b", "sigma")}
    optimizer = nlmm.NLMMOptimizer(**data, n_jobs=2, use_rust=False)

    def interrupt(fun, x0, **kwargs):
        fun(x0)
        raise KeyboardInterrupt

    with patch.object(nlmm, "minimize", side_effect=interrupt), pytest.raises(KeyboardInterrupt):
        optimizer.optimize(**starts)
    assert optimizer._workspace is None
    assert len(pools) == 1 and pools[0]._shutdown
    assert not any(thread.is_alive() for thread in pools[0]._threads)
    assert np.isfinite(optimizer.objective(np.array([0.5])))
    assert len(pools) == 2 and pools[1]._shutdown


def test_failed_parallel_trial_drains_running_work_before_return():
    data, _ = linear_problem()
    started, release, finished = Event(), Event(), Event()
    failed, returned = Event(), Event()
    errors = []
    with nlmm._nlmm_workspace(
        data["y"], data["x"], data["groups"], data["weights"], 2
    ) as workspace:

        def evaluate(g, rows):
            if g == 0:
                assert started.wait(5)
                failed.set()
                raise ValueError("trial failure")
            if g == 1:
                started.set()
                assert release.wait(5)
                finished.set()
            return g

        def consume():
            try:
                list(workspace.map_groups(evaluate))
            except ValueError as exc:
                errors.append((str(exc), finished.is_set()))
            finally:
                returned.set()

        worker = Thread(target=consume)
        worker.start()
        try:
            assert failed.wait(5)
            assert not returned.wait(0.1)
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert errors == [("trial failure", True)]
        assert list(workspace.map_groups(lambda g, rows: g)) == [0, 1, 2, 3]


def test_repeated_fit_rebuilds_workspace_after_data_changes(pools):
    data, _ = linear_problem()
    starts = {f"start_{name}": data.pop(name) for name in ("phi", "b", "sigma")}
    optimizer = nlmm.NLMMOptimizer(**data, n_jobs=2, use_rust=False)
    optimizer.optimize(**starts, maxiter=3)
    optimizer.y = data["y"] + np.linspace(-0.1, 0.3, len(data["y"]))
    optimizer.weights = data["weights"] * np.linspace(1, 2, len(data["y"]))
    optimizer.groups = optimizer.groups[::-1].copy()
    actual = optimizer.optimize(**starts, maxiter=3)
    fresh_data = dict(data, y=optimizer.y, weights=optimizer.weights, groups=optimizer.groups)
    expected = nlmm.NLMMOptimizer(**fresh_data, n_jobs=1, use_rust=False).optimize(
        **starts, maxiter=3
    )
    for name in ("phi", "theta", "b", "sigma", "deviance", "converged", "n_iter"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert len(pools) == 2 and all(pool._shutdown for pool in pools)

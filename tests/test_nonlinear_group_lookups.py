from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.estimation import nlmm
from mixedlm.nlme.models import SSasymp


def problem(n_groups=3, per_group=10, random_params=(0,)):
    rng = np.random.default_rng(12)
    groups = np.repeat(np.arange(n_groups), per_group)
    x = np.tile(np.linspace(0, 6, per_group), n_groups)
    phi = np.array([10.0, 3.0, -1.0])
    model = SSasymp()
    y = model.predict(phi, x) + np.repeat(rng.normal(0, 0.2, n_groups), per_group)
    y += rng.normal(0, 0.1, len(y))
    order = rng.permutation(len(y))
    q = len(random_params)
    return {
        "x": x[order],
        "y": y[order],
        "groups": groups[order],
        "weights": np.linspace(0.5, 2.0, len(y))[order],
        "model": model,
        "phi": phi,
        "b": rng.normal(0, 0.05, (n_groups, q)),
        "Psi": np.eye(q),
        "sigma": 0.3,
        "random_params": list(random_params),
    }


LABELS = [
    np.array([1, 2, 3]),
    np.array([10, 20, 30]),
    np.array([-9, -5, -1]),
    np.array([np.iinfo(np.int64).min, 3, np.iinfo(np.int64).max]),
]


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("jobs", [1, 2, -1])
@pytest.mark.parametrize("random_params", [(0,), (1, 0)])
def test_pnls_uses_sorted_unique_group_labels(labels, jobs, random_params):
    data = problem(random_params=random_params)
    expected = nlmm.pnls_step(**data)
    data["groups"] = labels[data["groups"]]
    with patch.object(nlmm.os, "cpu_count", return_value=2):
        actual = nlmm.pnls_step(**data, n_jobs=jobs)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("random_params", [(0,), (1, 0)])
def test_deviance_uses_sorted_unique_group_labels(labels, jobs, random_params):
    data = problem(random_params=random_params)
    data.pop("Psi")
    q = len(random_params)
    theta = np.eye(q)[np.tril_indices(q)]
    expected = nlmm.nlmm_deviance(theta, **data)
    data["groups"] = labels[data["groups"]]
    actual = nlmm.nlmm_deviance(theta, **data, n_jobs=jobs)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("use_rust", [False, True])
def test_optimizer_is_invariant_to_group_labels(labels, use_rust):
    data = problem()
    starts = {"start_phi": data.pop("phi"), "start_b": data.pop("b")}
    data.pop("Psi")
    starts["start_sigma"] = data.pop("sigma")
    baseline = nlmm.NLMMOptimizer(**data, use_rust=use_rust)
    expected = baseline.optimize(**starts, maxiter=8)
    data["groups"] = labels[data["groups"]]
    optimizer = nlmm.NLMMOptimizer(**data, use_rust=use_rust)
    actual = optimizer.optimize(**starts, maxiter=8)
    for name in ("phi", "theta", "b", "sigma", "deviance"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.converged == expected.converged
    assert actual.n_iter == expected.n_iter


@pytest.mark.parametrize("jobs", [1, 2])
def test_custom_model_keeps_group_observation_order_and_copies_predictors(jobs):
    data = problem()
    data["x"] = np.column_stack([data["x"], data["x"] ** 2])
    original_x = data["x"].copy()
    labels = np.array([-50, 10, 300])

    def predict(params, x):
        x[:, 0] *= 0.5
        return params[0] + params[1] * x[:, 0] + params[2] * x[:, 1] + np.arange(len(x)) * 0.01

    def gradient(params, x):
        return np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])

    with (
        patch.object(data["model"], "predict", side_effect=predict),
        patch.object(data["model"], "gradient", side_effect=gradient),
    ):
        expected = nlmm.pnls_step(**data, n_jobs=jobs)
        data["groups"] = labels[data["groups"]]
        actual = nlmm.pnls_step(**data, n_jobs=jobs)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
    np.testing.assert_array_equal(data["x"], original_x)


@pytest.mark.parametrize("jobs", [1, 2])
def test_group_lookup_does_not_change_inputs(jobs):
    data = problem()
    arrays = {name: value.copy() for name, value in data.items() if isinstance(value, np.ndarray)}
    nlmm.pnls_step(**data, n_jobs=jobs)
    for name, original in arrays.items():
        np.testing.assert_array_equal(data[name], original)


def test_threaded_pnls_does_not_retain_a_full_row_mask_per_group(monkeypatch):
    import tracemalloc

    data = problem(n_groups=200, per_group=200)
    monkeypatch.setattr(nlmm, "_PNLS_MAX_ITER", 1)
    tracemalloc.start()
    try:
        nlmm.pnls_step(**data, n_jobs=2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 200 retained boolean masks alone need 8 MB, before residual/gradient arrays.
    assert peak < 6 * 1024**2

import numpy as np
import pytest
from mixedlm.estimation import nlmm
from mixedlm.nlme.models import NonlinearModel


class SquaredMean(NonlinearModel):
    """A nonlinear mean with an analytical symmetric PNLS solution."""

    def __init__(self, size=1):
        self.size = size

    @property
    def name(self):
        return "squared_mean"

    @property
    def param_names(self):
        return [f"a{i}" for i in range(self.size)]

    def predict(self, params, x):
        return params[x.astype(int)] ** 2

    def gradient(self, params, x):
        columns = x.astype(int)
        gradient = np.zeros((len(x), self.size))
        gradient[np.arange(len(x)), columns] = 2 * params[columns]
        return gradient


def symmetric_problem(target=4.0, size=1, per_group=1):
    direction = np.ones(size)
    covariance = np.eye(size) + 0.4 * np.ones((size, size)) / size
    return {
        "y": np.full(2 * per_group * size, target),
        "x": np.tile(np.arange(size), 2 * per_group),
        "groups": np.repeat(np.arange(2), per_group * size),
        "model": SquaredMean(size),
        "phi": np.zeros(size),
        "b": np.stack([-direction, direction]),
        "Psi": covariance,
        "sigma": 1.0,
        "random_params": list(range(size)),
        "weights": np.tile(np.repeat(np.linspace(0.5, 1.5, per_group), size), 2),
    }


def analytical_solution(data):
    size = len(data["phi"])
    per_group = len(data["y"]) // (2 * size)
    group_weight = data["weights"][: per_group * size : size].sum()
    precision = 1 / (1.4 + 1e-8)
    target = data["y"][0]
    radius_sq = max(target - precision / (2 * group_weight), 0.0)
    b = data["b"] * np.sqrt(radius_sq)
    sigma = np.sqrt((group_weight * (target - radius_sq) ** 2 + precision * radius_sq) / per_group)
    return b, sigma


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("per_group", [1, 3])
@pytest.mark.parametrize("target", [0.0, 4.0, 25.0])
def test_random_effects_converge_when_fixed_effects_are_stationary(jobs, size, per_group, target):
    data = symmetric_problem(target, size, per_group)
    initial_b = data["b"].copy()
    initial_phi = data["phi"].copy()
    expected_b, expected_sigma = analytical_solution(data)
    phi, b, sigma = nlmm.pnls_step(**data, n_jobs=jobs)
    np.testing.assert_allclose(phi, 0.0, atol=1e-8)
    np.testing.assert_allclose(b, expected_b, atol=5e-7, rtol=1e-7)
    np.testing.assert_allclose(sigma, expected_sigma, atol=5e-7, rtol=1e-7)
    np.testing.assert_array_equal(data["b"], initial_b)
    np.testing.assert_array_equal(data["phi"], initial_phi)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("target", [4.0, 25.0])
def test_deviance_uses_converged_random_effects(jobs, size, target):
    data = symmetric_problem(target, size, per_group=3)
    expected_b, expected_sigma = analytical_solution(data)
    covariance = data.pop("Psi")
    theta = np.linalg.cholesky(covariance)[np.tril_indices(size)]
    deviance, phi, b, sigma = nlmm.nlmm_deviance(theta, **data, n_jobs=jobs)
    group_weight = data["weights"][: 3 * size : size].sum()
    gradient_information = 4 * group_weight * expected_b[0, 0] ** 2
    correction = 2 * (
        np.log1p(1.4 * gradient_information) + (size - 1) * np.log1p(gradient_information)
    )
    expected_deviance = len(data["y"]) * (1 + np.log(2 * np.pi * expected_sigma**2)) + correction
    np.testing.assert_allclose(b, expected_b, atol=5e-7, rtol=1e-7)
    np.testing.assert_allclose(phi, 0.0, atol=1e-8)
    assert sigma == pytest.approx(expected_sigma, abs=5e-7, rel=1e-7)
    assert deviance == pytest.approx(expected_deviance, abs=5e-6, rel=1e-7)


@pytest.mark.parametrize("jobs", [1, 2])
def test_complete_optimizer_returns_stationary_random_effects(jobs):
    data = symmetric_problem(per_group=3)
    starts = {
        "start_phi": data.pop("phi"),
        "start_b": data.pop("b"),
        "start_sigma": data.pop("sigma"),
    }
    data.pop("Psi")
    optimizer = nlmm.NLMMOptimizer(**data, use_rust=False, n_jobs=jobs)
    fit = optimizer.optimize(start_theta=np.array([1.0]), **starts, maxiter=3)
    precision = 1 / (fit.theta[0] ** 2 + 1e-8)
    weight = data["weights"][:3].sum()
    radius = np.sqrt(max(4.0 - precision / (2 * weight), 0.0))
    np.testing.assert_allclose(fit.phi, 0.0, atol=1e-8)
    np.testing.assert_allclose(fit.b[:, 0], [-radius, radius], atol=5e-7, rtol=1e-7)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("limit", [1, 2])
def test_iteration_limit_still_bounds_work(monkeypatch, jobs, limit):
    data = symmetric_problem()
    precision = 1 / (1.4 + 1e-8)
    radius = 1.0
    weight = data["weights"][0]
    for _ in range(limit):
        radius = 2 * weight * radius * (4 + radius**2) / (4 * weight * radius**2 + precision)
    monkeypatch.setattr(nlmm, "_PNLS_MAX_ITER", limit)
    _, b, _ = nlmm.pnls_step(**data, n_jobs=jobs)
    np.testing.assert_allclose(b[:, 0], [-radius, radius], atol=1e-13, rtol=1e-13)

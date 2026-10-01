import numpy as np
import pytest
from mixedlm.estimation import nlmm
from mixedlm.nlme.models import CustomModel, SSasymp, SSlogis, SSmicmen
from scipy import linalg, optimize


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("n_groups", [2, 7])
@pytest.mark.parametrize("random_params", [(0,), (1, 0), (0, 0)])
def test_one_joint_update_matches_full_penalized_linear_system(jobs, n_groups, random_params):
    rng = np.random.default_rng(805)
    n = 6 * n_groups
    group = np.arange(n) % n_groups
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = x @ [1.2, -0.4] + rng.normal(0, 0.3, n)
    weights = np.geomspace(0.3, 3.0, n)
    phi = np.array([-2.0, 3.0])
    q = len(random_params)
    b = rng.normal(0, 0.4, (n_groups, q))
    factor = np.array([[0.8]]) if q == 1 else np.array([[0.8, 0.0], [-0.3, 0.5]])
    covariance = factor @ factor.T
    precision = np.linalg.inv(covariance + 1e-8 * np.eye(q))
    z = np.zeros((n, n_groups * q))
    for row, g in enumerate(group):
        z[row, g * q : (g + 1) * q] = x[row, random_params]
    design = np.column_stack([x, z])
    normal = design.T @ (weights[:, None] * design)
    normal[2:, 2:] += np.kron(np.eye(n_groups), precision)
    normal[:2, :2] += 1e-6 * np.eye(2)
    rhs = design.T @ (weights * y)
    rhs[:2] += 1e-6 * phi
    expected = np.linalg.solve(normal, rhs)
    residual = y - design @ expected
    pwrss = (
        weights @ residual**2 + expected[2:] @ np.kron(np.eye(n_groups), precision) @ expected[2:]
    )
    model = CustomModel(lambda p, x: x @ p, lambda p, x: x.copy(), ["a", "b"])
    actual = nlmm.pnls_step(
        y,
        x,
        10 * group - 50,
        model,
        phi,
        b,
        covariance,
        1.0,
        list(random_params),
        weights=weights,
        n_jobs=jobs,
        maxiter=1,
    )
    np.testing.assert_allclose(actual[0], expected[:2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[2:].reshape(n_groups, q), rtol=1e-12, atol=1e-12)
    assert actual[2] == pytest.approx(np.sqrt(pwrss / n), rel=1e-12, abs=1e-12)


MODELS = [
    (SSasymp, [5.0, 0.8, -0.4], [0.0, 0.5, 1.0, 2.0, 4.0, 7.0]),
    (SSmicmen, [3.0, 1.2], [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]),
    (SSlogis, [4.0, 1.0, 1.4], [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]),
]


@pytest.mark.parametrize("model_type,parameters,grid", MODELS)
@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("backend", ["serial", "threaded", "native"])
def test_nonlinear_solution_matches_independent_joint_least_squares(
    model_type, parameters, grid, q, backend
):
    rng = np.random.default_rng(806)
    model = model_type()
    phi = np.asarray(parameters)
    p = len(phi)
    group = np.repeat(np.arange(3), len(grid))
    x = np.tile(grid, 3)
    n = len(x)
    weights = np.geomspace(0.4, 2.0, n)
    random_params = list(range(q))
    effects = rng.normal(0, 0.1, (3, q))
    y = np.empty(n)
    for g in range(3):
        params = phi.copy()
        params[random_params] += effects[g]
        y[group == g] = model.predict(params, x[group == g])
    y += rng.normal(0, 0.025, n)
    factor = np.array([[0.4]]) if q == 1 else np.array([[0.4, 0.0], [-0.05, 0.2]])
    root = np.linalg.cholesky(factor @ factor.T + 1e-8 * np.eye(q))
    start_phi = phi + np.linspace(-0.1, 0.1, p)
    start_b = np.zeros((3, q))

    def residual(coefficients):
        fixed = coefficients[:p]
        random = coefficients[p:].reshape(3, q)
        data_residual = np.empty(n)
        for g in range(3):
            params = fixed.copy()
            params[random_params] += random[g]
            rows = group == g
            data_residual[rows] = np.sqrt(weights[rows]) * (
                y[rows] - model.predict(params, x[rows])
            )
        prior_residual = linalg.solve_triangular(root, random.T, lower=True).T.ravel()
        return np.r_[data_residual, prior_residual]

    reference = optimize.least_squares(
        residual,
        np.r_[start_phi, start_b.ravel()],
        jac="3-point",
        max_nfev=3000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    assert reference.success
    kwargs = dict(
        y=y,
        x=x,
        groups=group,
        model=model,
        phi=start_phi,
        b=start_b,
        random_params=random_params,
        sigma=0.3,
        weights=weights,
        pnls_maxiter=100,
        pnls_tol=1e-9,
    )
    theta = factor[np.tril_indices(q)]
    if backend == "native":
        actual = nlmm._nlmm_deviance_rust_with_status(theta, **kwargs)
    else:
        actual = nlmm.nlmm_deviance_with_status(
            theta, **kwargs, n_jobs=2 if backend == "threaded" else 1
        )
    deviance, fitted_phi, fitted_b, sigma, converged = actual
    assert converged
    np.testing.assert_allclose(fitted_phi, reference.x[:p], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(fitted_b, reference.x[p:].reshape(3, q), rtol=1e-7, atol=1e-7)
    variance = np.dot(reference.fun, reference.fun) / n
    assert sigma == pytest.approx(np.sqrt(variance), rel=1e-10, abs=1e-10)
    random_jacobian = reference.jac[:n, p:]
    block_factor = np.kron(np.eye(3), factor)
    system = np.eye(3 * q) + block_factor.T @ (random_jacobian.T @ random_jacobian) @ block_factor
    expected_deviance = n * (1 + np.log(2 * np.pi * variance)) + np.linalg.slogdet(system)[1]
    assert deviance == pytest.approx(expected_deviance, rel=1e-8, abs=1e-7)


@pytest.mark.parametrize("jobs", [1, 2])
def test_line_search_recovers_from_invalid_logarithm_trials_and_decreases_error(jobs):
    rejected = []

    def predict(params, x):
        if params[0] <= 0:
            rejected.append(params[0])
            raise ValueError("logarithm requires a positive level")
        return np.full(len(x), np.log(params[0]))

    model = CustomModel(predict, lambda p, x: np.full((len(x), 1), 1 / p[0]), ["level"])
    groups = np.repeat([0, 1], 6)
    y = np.repeat(np.log([0.1, 0.2]), 6)
    weights = np.geomspace(0.5, 2.0, len(y))
    phi, b = np.array([10.0]), np.zeros((2, 1))
    scores = []
    for limit in [1, 2, 3, 6, 30]:
        _, fitted_phi, fitted_b, sigma, converged = nlmm.nlmm_deviance_with_status(
            np.array([0.5]),
            y,
            np.ones(len(y)),
            groups,
            model,
            phi,
            b,
            [0],
            1.0,
            weights=weights,
            n_jobs=jobs,
            pnls_maxiter=limit,
            pnls_tol=1e-9,
        )
        assert np.all(fitted_phi[0] + fitted_b[:, 0] > 0)
        residual = y - np.log(fitted_phi[0] + fitted_b[groups, 0])
        score = weights @ residual**2 + np.sum(fitted_b**2) / (0.25 + 1e-8)
        assert sigma**2 * len(y) == pytest.approx(score, rel=1e-12, abs=1e-12)
        scores.append(score)
    assert converged
    assert rejected
    assert np.all(np.diff(scores) <= 1e-12)
    np.testing.assert_array_equal(phi, [10.0])
    np.testing.assert_array_equal(b, np.zeros((2, 1)))


@pytest.mark.parametrize("jobs", [1, 2])
def test_failed_line_search_retains_the_last_valid_point_without_convergence(jobs):
    model = CustomModel(
        lambda p, x: np.full(len(x), p[0] ** 2), lambda p, x: np.full((len(x), 1), -2 * p[0]), ["a"]
    )
    _, phi, b, sigma, converged = nlmm.nlmm_deviance_with_status(
        np.array([0.5]),
        np.full(4, 4.0),
        np.ones(4),
        np.repeat([0, 1], 2),
        model,
        np.array([1.0]),
        np.zeros((2, 1)),
        [0],
        1.0,
        n_jobs=jobs,
    )
    assert not converged
    np.testing.assert_array_equal(phi, [1.0])
    np.testing.assert_array_equal(b, np.zeros((2, 1)))
    assert sigma == 3.0


@pytest.mark.parametrize("jobs", [1, 2])
def test_tiny_accepted_step_cannot_hide_a_large_joint_proposal(jobs):
    def predict(p, x):
        if abs(p[0] - 1.0) > 1e-5:
            raise ValueError("outside valid parameter range")
        return np.full(len(x), p[0])

    model = CustomModel(predict, lambda p, x: np.ones((len(x), 1)), ["a"])
    _, phi, _, _, converged = nlmm.nlmm_deviance_with_status(
        np.array([0.5]),
        np.full(4, 2.0),
        np.ones(4),
        np.repeat([0, 1], 2),
        model,
        np.array([1.0]),
        np.zeros((2, 1)),
        [0],
        1.0,
        n_jobs=jobs,
        pnls_maxiter=1,
        pnls_tol=1e-4,
    )
    assert 0 < phi[0] - 1.0 < 1e-4
    assert not converged


@pytest.mark.parametrize("jobs", [1, 2])
def test_many_groups_require_only_small_joint_block_solves(monkeypatch, jobs):
    n_groups = 1000
    effects = np.linspace(-0.4, 0.4, n_groups)
    groups = np.repeat(np.arange(n_groups), 2)
    y = 1.0 + effects[groups]
    model = CustomModel(
        lambda p, x: np.full(len(x), p[0]), lambda p, x: np.ones((len(x), 1)), ["a"]
    )
    solve = linalg.solve

    def guarded_solve(matrix, *args, **kwargs):
        assert matrix.shape == (1, 1)
        return solve(matrix, *args, **kwargs)

    monkeypatch.setattr(linalg, "solve", guarded_solve)
    _, phi, b, _, converged = nlmm.nlmm_deviance_with_status(
        np.array([0.5]),
        y,
        np.ones(len(y)),
        groups,
        model,
        np.array([-10.0]),
        np.zeros((n_groups, 1)),
        [0],
        1.0,
        n_jobs=jobs,
        pnls_maxiter=3,
    )
    assert converged
    np.testing.assert_allclose(phi, [1.0], rtol=1e-12, atol=1e-12)
    expected = effects * (2 * (0.25 + 1e-8)) / (1 + 2 * (0.25 + 1e-8))
    np.testing.assert_allclose(b[:, 0], expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("jobs", [1, 2])
def test_wide_fixed_design_does_not_retain_one_normal_matrix_per_group(jobs):
    import tracemalloc

    rng = np.random.default_rng(807)
    groups = np.repeat(np.arange(256), 2)
    x = rng.normal(size=(len(groups), 64))
    y = x @ rng.normal(size=64) + rng.normal(size=len(groups))
    model = CustomModel(lambda p, x: x @ p, lambda p, x: x.copy(), [f"b{i}" for i in range(64)])
    tracemalloc.start()
    try:
        nlmm.pnls_step(
            y,
            x,
            groups,
            model,
            np.zeros(64),
            np.zeros((256, 1)),
            np.eye(1),
            1.0,
            [0],
            n_jobs=jobs,
            maxiter=1,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Retaining 256 group normal matrices would require 8 MiB alone.
    assert peak < 4 * 1024**2

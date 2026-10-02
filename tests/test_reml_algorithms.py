from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import cho_factor, cho_solve, expm
from scipy.optimize import minimize

try:
    from mixedlm._rust import augmented_ai_reml, mm_reml, riemannian_reml

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")


@pytest.fixture
def reml_data():
    rng = np.random.default_rng(42)
    n = 24
    n_groups = 6
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    z = np.eye(n_groups)[np.arange(n) % n_groups]
    y = x @ np.array([1.0, 0.5]) + rng.normal(scale=0.7, size=n)
    return y, x, z


def test_mm_reml_one_step_matches_dense_projection(reml_data):
    y, x, z = reml_data
    variance = 0.8
    sigma2 = 1.2

    estimated_variances, estimated_sigma2, iterations, _ = mm_reml(
        y,
        x,
        [z],
        [variance],
        sigma2,
        max_iter=1,
        tol=1e-12,
    )

    covariance = sigma2 * np.eye(len(y)) + variance * z @ z.T
    covariance_inverse = np.linalg.inv(covariance)
    weighted_x = covariance_inverse @ x
    projection = covariance_inverse - weighted_x @ np.linalg.solve(x.T @ weighted_x, weighted_x.T)
    projected_y = projection @ y
    expected_variance = variance * np.sqrt(
        np.linalg.norm(z.T @ projected_y) ** 2 / np.trace(z.T @ projection @ z)
    )
    expected_sigma2 = sigma2 * np.sqrt(np.linalg.norm(projected_y) ** 2 / np.trace(projection))

    assert iterations == 1
    assert_allclose(estimated_variances, [expected_variance], rtol=1e-11)
    assert_allclose(estimated_sigma2, expected_sigma2, rtol=1e-11)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
def test_reml_supports_no_fixed_effects(reml_function, reml_data):
    y, _, z = reml_data
    variances, sigma2, iterations, _ = reml_function(
        y, np.empty((len(y), 0)), [z], [1.0], 1.0, max_iter=1
    )

    assert iterations == 1
    assert np.all(np.isfinite(variances))
    assert np.isfinite(sigma2)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
def test_reml_rejects_mismatched_x_rows(reml_function, reml_data):
    y, x, z = reml_data
    with pytest.raises(ValueError, match="x must have 24 rows"):
        reml_function(y, x[:-1], [z], [1.0], 1.0)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
def test_reml_rejects_mismatched_z_rows(reml_function, reml_data):
    y, x, z = reml_data
    with pytest.raises(ValueError, match="Z block 0 must have 24 rows"):
        reml_function(y, x, [z[:-1]], [1.0], 1.0)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
def test_reml_rejects_mismatched_initial_variances(reml_function, reml_data):
    y, x, z = reml_data
    with pytest.raises(ValueError, match="one value per Z block"):
        reml_function(y, x, [z], [], 1.0)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
@pytest.mark.parametrize(
    ("initial_variance", "sigma2", "tol", "message"),
    [(-1.0, 1.0, 1e-6, "init_variances"), (1.0, 0.0, 1e-6, "init_sigma2"), (1.0, 1.0, 0.0, "tol")],
)
def test_reml_rejects_invalid_parameters(
    reml_function, reml_data, initial_variance, sigma2, tol, message
):
    y, x, z = reml_data
    with pytest.raises(ValueError, match=message):
        reml_function(y, x, [z], [initial_variance], sigma2, tol=tol)


@pytest.mark.parametrize(("field", "message"), [("y", "y"), ("x", "x"), ("z", "Z block 0")])
def test_mm_reml_rejects_nonfinite_data(field, message, reml_data):
    y, x, z = (value.copy() for value in reml_data)
    values = {"y": y, "x": x, "z": z}
    values[field].flat[0] = np.nan

    with pytest.raises(ValueError, match=message):
        mm_reml(y, x, [z], [1.0], 1.0)


@pytest.mark.parametrize("step_size", [0.0, -0.1, np.nan, np.inf])
def test_riemannian_reml_rejects_invalid_step_size(step_size, reml_data):
    y, x, z = reml_data
    with pytest.raises(ValueError, match="step_size"):
        riemannian_reml(y, x, [z], [1.0], 1.0, step_size=step_size)


@pytest.fixture
def identifiable_reml_data():
    rng = np.random.default_rng(334)
    n, groups = 48, 8
    z = np.eye(groups)[np.repeat(np.arange(groups), n // groups)]
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    y = (
        x @ np.array([1.0, 0.3])
        + z @ rng.normal(scale=1.1, size=groups)
        + rng.normal(scale=0.6, size=n)
    )
    return y, x, z


def dense_reml_projection(y, x, blocks, parameters):
    covariance = parameters[-1] * np.eye(len(y))
    for z, variance in zip(blocks, parameters[:-1], strict=True):
        covariance += variance * z @ z.T
    factor = cho_factor(covariance, lower=True)
    inverse = cho_solve(factor, np.eye(len(y)))
    if x.shape[1]:
        weighted_x = inverse @ x
        information = x.T @ weighted_x
        fixed_factor = cho_factor(information, lower=True)
        projection = inverse - weighted_x @ cho_solve(fixed_factor, weighted_x.T)
        fixed_logdet = 2 * np.log(np.diag(fixed_factor[0])).sum()
    else:
        projection = inverse
        fixed_logdet = 0.0
    objective = 2 * np.log(np.diag(factor[0])).sum() + fixed_logdet + y @ projection @ y
    return projection, objective


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
def test_reml_estimates_match_independent_likelihood_optimum(reml_function, identifiable_reml_data):
    y, x, z = identifiable_reml_data
    initial = np.array([0.8, 1.2])
    reference = minimize(
        lambda logs: dense_reml_projection(y, x, [z], np.exp(logs))[1],
        np.log(initial),
        method="BFGS",
        options={"gtol": 1e-7},
    )
    variances, sigma2, _, converged = reml_function(
        y, x, [z], initial[:1], initial[1], max_iter=2000, tol=1e-8
    )
    estimated = np.r_[variances, sigma2]
    assert converged
    assert_allclose(estimated, np.exp(reference.x), rtol=2e-5)
    assert_allclose(dense_reml_projection(y, x, [z], estimated)[1], reference.fun, atol=1e-9)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml, riemannian_reml])
def test_reml_updates_are_equivariant_to_response_units(reml_function, identifiable_reml_data):
    y, x, z = identifiable_reml_data
    initial = np.array([0.8, 1.2])
    scale = 100.0
    baseline = reml_function(y, x, [z], initial[:1], initial[1], max_iter=3, tol=1e-12)
    scaled = reml_function(
        scale * y, x, [z], scale**2 * initial[:1], scale**2 * initial[1], max_iter=3, tol=1e-12
    )
    assert_allclose(scaled[0], scale**2 * baseline[0], rtol=1e-10)
    assert_allclose(scaled[1], scale**2 * baseline[1], rtol=1e-10)


def test_ai_update_matches_dense_information_with_multiple_components():
    rng = np.random.default_rng(718)
    n = 32
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    blocks = [np.eye(8)[np.arange(n) % 8], np.eye(4)[np.arange(n) // 8]]
    y = x @ np.array([1.0, 0.4]) + sum(z @ rng.normal(size=z.shape[1]) for z in blocks)
    y += rng.normal(scale=0.7, size=n)
    initial = np.array([0.8, 0.6, 0.9])
    projection, objective = dense_reml_projection(y, x, blocks, initial)
    projected_y = projection @ y
    derivatives = [z @ z.T for z in blocks] + [np.eye(n)]
    actions = np.column_stack([derivative @ projected_y for derivative in derivatives])
    score = 0.5 * np.array(
        [
            projected_y @ derivative @ projected_y - np.trace(projection @ derivative)
            for derivative in derivatives
        ]
    )
    information = 0.5 * actions.T @ projection @ actions
    candidate = np.maximum(initial + np.linalg.solve(information, score), 1e-10)
    # This fixture takes a full step, allowing a direct independent AI comparison.
    assert dense_reml_projection(y, x, blocks, candidate)[1] < objective
    variances, sigma2, _, _ = augmented_ai_reml(
        y, x, blocks, initial[:-1], initial[-1], max_iter=1, tol=1e-12
    )
    assert_allclose(np.r_[variances, sigma2], candidate, rtol=1e-10, atol=1e-12)


def test_riemannian_step_matches_covariance_score_and_geodesic(identifiable_reml_data):
    y, x, z = identifiable_reml_data
    variance, sigma2, step = 0.8, 1.2, 0.01
    projection, _ = dense_reml_projection(y, x, [z], [variance, sigma2])
    projected_y = projection @ y
    score = 0.5 * (np.linalg.norm(z.T @ projected_y) ** 2 - np.trace(z.T @ projection @ z))
    expected_variance = variance * expm(np.array([[step * variance * score]]))[0, 0]
    expected_sigma2 = sigma2 * np.sqrt(np.linalg.norm(projected_y) ** 2 / np.trace(projection))
    variances, fitted_sigma2, _, _ = riemannian_reml(
        y, x, [z], [variance], sigma2, max_iter=1, step_size=step, tol=1e-12
    )
    assert_allclose(variances, [expected_variance], rtol=1e-11)
    assert_allclose(fitted_sigma2, expected_sigma2, rtol=1e-11)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml])
@pytest.mark.parametrize("initial_variance", [0.0, 1e-20])
def test_reml_recovers_positive_component_from_boundary_start(
    reml_function, initial_variance, identifiable_reml_data
):
    y, x, z = identifiable_reml_data
    variances, sigma2, _, converged = reml_function(
        y, x, [z], [initial_variance], 1.2, max_iter=2000, tol=1e-8
    )
    assert converged
    assert_allclose([variances[0], sigma2], [0.54933298, 0.33877381], rtol=2e-5)


@pytest.mark.parametrize("reml_function", [mm_reml, augmented_ai_reml])
def test_reml_boundary_optimum_matches_fixed_effect_residual_variance(reml_function):
    groups, observations_per_group = 6, 4
    z = np.eye(groups)[np.repeat(np.arange(groups), observations_per_group)]
    x = np.ones((len(z), 1))
    y = np.repeat(0.04 * np.arange(groups), observations_per_group)
    y += np.tile([-1.0, -0.5, 0.5, 1.0], groups)
    expected_sigma2 = np.sum((y - y.mean()) ** 2) / (len(y) - 1)
    variances, sigma2, _, converged = reml_function(y, x, [z], [0.8], 1.2, max_iter=2000, tol=1e-8)
    assert converged
    assert variances[0] <= 1e-9
    assert_allclose(sigma2, expected_sigma2, rtol=1e-8)


def test_ai_rejects_nonidentifiable_duplicate_components(identifiable_reml_data):
    y, x, z = identifiable_reml_data
    with pytest.raises(ValueError, match="AI matrix not positive definite"):
        augmented_ai_reml(y, x, [z, z], [0.4, 0.4], 1.2)


def test_riemannian_tiny_start_does_not_falsely_report_convergence(identifiable_reml_data):
    y, x, z = identifiable_reml_data
    _, _, iterations, converged = riemannian_reml(y, x, [z], [1e-20], 1.2, max_iter=100, tol=1e-8)
    assert iterations == 100
    assert not converged

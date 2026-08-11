from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
    expected_variance = variance**2 * np.linalg.norm(z.T @ projected_y) ** 2
    expected_variance /= np.trace(z.T @ projection @ z)
    expected_sigma2 = sigma2**2 * np.linalg.norm(projected_y) ** 2
    expected_sigma2 /= len(y) - x.shape[1]

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

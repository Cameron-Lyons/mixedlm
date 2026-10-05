"""Native PIRLS solves agree with the joint penalized Gaussian system."""

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose
from scipy import sparse


@pytest.mark.parametrize("p", [0, 1, 4, 17])
@pytest.mark.parametrize("q", [0, 1, 33, 129])
@pytest.mark.parametrize("scale", [0.0, 0.7])
def test_native_laplace_matches_joint_gaussian_system(p, q, scale):
    rng = np.random.default_rng(637 + 13 * p + q)
    n = max(80, 2 * (p + q))
    x = rng.normal(size=(n, p))
    if p:
        x[:, 0] = 1.0
    # Overlapping columns exercise nonzero off-diagonal Cholesky entries,
    # including solve sizes beyond the small dense-kernel boundaries.
    z_dense = rng.normal(scale=0.3, size=(n, q))
    z_dense[rng.random(z_dense.shape) < 0.3] = 0.0
    if q > 1:
        z_dense[:, -1] = 0.0  # Retain an unused random-effect level.
    z = sparse.csc_matrix(z_dense)
    weights = rng.uniform(0.2, 2.5, n)
    offset = rng.normal(scale=0.15, size=n)
    y = x @ rng.normal(scale=0.3, size=p) + offset
    y += z_dense @ rng.normal(scale=0.2, size=q) + rng.normal(scale=0.25, size=n)

    scaled_z = scale * z_dense
    design = np.column_stack((x, scaled_z))
    information = design.T @ (weights[:, None] * design)
    information[p:, p:] += np.eye(q)
    rhs = design.T @ (weights * (y - offset))
    solution = np.linalg.solve(information, rhs)
    expected_beta = solution[:p]
    spherical = solution[p:]
    expected_random = scale * spherical
    residual = y - offset - design @ solution
    expected_deviance = np.dot(weights * residual, residual) + np.dot(spherical, spherical)
    random_information = np.eye(q) + scaled_z.T @ (weights[:, None] * scaled_z)
    expected_laplace = expected_deviance + np.linalg.slogdet(random_information)[1]

    args = (
        y,
        x,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        weights,
        offset,
        np.array([scale]) if q else np.empty(0),
        [q] if q else [],
        [1] if q else [],
        [True] if q else [],
        "gaussian",
        "identity",
    )
    laplace, beta, random, converged = _rust.glmm_deviance(*args, 1)
    assert converged
    assert_allclose(beta, expected_beta, rtol=2e-11, atol=2e-12)
    assert_allclose(random, expected_random, rtol=2e-11, atol=2e-12)
    assert laplace == pytest.approx(expected_laplace, rel=2e-12, abs=2e-12)

"""Mode-only PIRLS updates agree with independent dense normal equations."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm.estimation.laplace import _native_glmm_args
from mixedlm.estimation.reml import _build_lambda
from numpy.testing import assert_allclose

from tests.test_glmm_final_state import mode_problem

native = pytest.importorskip("mixedlm._rust")

LAYOUTS = {
    "diagonal": ("mode_only", 96, 8),
    "dense": ("slope", 96, 8),
    "sparse": ("slope", 384, 64),
    "crossed": ("crossed", 512, 128),
    "empty": ("fixed_only", 48, 4),
}


def problem(kind, layout, zero_covariance):
    shape, n, groups = LAYOUTS[layout]
    matrices, family, theta = mode_problem(kind, shape, n_obs=n, n_groups=groups)
    matrices = replace(matrices, X=np.empty((n, 0)), n_fixed=0, fixed_names=[])
    if zero_covariance:
        theta[:] = 0
    covariance = _build_lambda(theta, matrices.random_structures).toarray()
    design = matrices.Z @ covariance
    return matrices, family, theta, covariance, design


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("zero_covariance", [False, True])
def test_first_mode_update_matches_dense_normal_equations(kind, layout, zero_covariance):
    matrices, family, theta, covariance, design = problem(kind, layout, zero_covariance)
    mean = family.clamp_mu(family.link.inverse(matrices.offset), eps=1e-10)
    weights = np.maximum(matrices.weights * family.weights(mean), 1e-10)
    working_response = family.link.deriv(mean) * (matrices.y - mean)
    precision = np.eye(matrices.n_random) + design.T @ (weights[:, None] * design)
    spherical = np.linalg.solve(precision, design.T @ (weights * working_response))
    expected_random = covariance @ spherical
    final_mean = family.clamp_mu(
        family.link.inverse(matrices.offset + matrices.Z @ expected_random), eps=1e-10
    )
    expected_deviance = (
        np.sum(family.deviance_resids(matrices.y, final_mean, matrices.weights))
        + spherical @ spherical
    )

    beta, random, deviance, converged = native.pirls(
        *_native_glmm_args(theta, matrices, family), maxiter=1, tol=1e-12
    )
    assert beta == []
    assert_allclose(random, expected_random, rtol=1e-11, atol=1e-12)
    assert_allclose(deviance, expected_deviance, rtol=1e-12, atol=1e-11)
    assert converged == (np.max(np.abs(spherical), initial=0.0) < 1e-12)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("zero_covariance", [False, True])
def test_gaussian_mode_and_likelihood_match_closed_form(layout, zero_covariance):
    matrices, family, theta, covariance, design = problem("gaussian", layout, zero_covariance)
    precision = np.eye(matrices.n_random) + design.T @ (matrices.weights[:, None] * design)
    spherical = np.linalg.solve(
        precision, design.T @ (matrices.weights * (matrices.y - matrices.offset))
    )
    residual = matrices.y - matrices.offset - design @ spherical
    sign, logdet = np.linalg.slogdet(precision)
    assert sign == 1
    expected = np.dot(matrices.weights, residual**2) + spherical @ spherical + logdet
    orders = [1, 7] if layout in {"diagonal", "empty"} else [1]
    for order in orders:
        deviance, beta, random, converged = native.glmm_deviance(
            *_native_glmm_args(theta, matrices, family), order, maxiter=100, tol=1e-12
        )
        assert converged
        assert beta == []
        assert_allclose(random, covariance @ spherical, rtol=1e-11, atol=1e-12)
        assert_allclose(deviance, expected, rtol=1e-12, atol=1e-11)


@pytest.mark.parametrize("n", [1, 3, 7, 15, 17, 31, 33, 129])
@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
def test_mode_update_with_short_and_odd_lengths_and_extreme_offsets(n, kind):
    matrices, family, theta = mode_problem(kind, "intercept", n_obs=n, n_groups=1)
    weights = np.geomspace(1e-14, 1e4, n)
    weights[::5] = 0
    matrices = replace(
        matrices,
        X=np.empty((n, 0)),
        n_fixed=0,
        fixed_names=[],
        offset=np.linspace(-30, 30, n),
        weights=weights,
    )
    mean = family.clamp_mu(family.link.inverse(matrices.offset), eps=1e-10)
    derivative = family.link.deriv(mean)
    variance = np.maximum(family.variance(mean), 1e-10)
    working_weights = np.maximum(weights / np.maximum(derivative**2 * variance, 1e-10), 1e-10)
    working_response = derivative * (matrices.y - mean)
    # One random intercept makes the penalized normal equation scalar.
    precision = 1 + theta[0] ** 2 * np.sum(working_weights)
    expected_random = theta[0] ** 2 * np.dot(working_weights, working_response) / precision
    beta, random, _, _ = native.pirls(
        *_native_glmm_args(theta, matrices, family), maxiter=1, tol=1e-12
    )
    assert beta == []
    assert_allclose(random, [expected_random], rtol=1e-11, atol=1e-12)

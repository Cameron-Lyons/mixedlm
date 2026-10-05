"""Likelihood corrections use the same final mode as the returned estimates."""

import numpy as np
import pytest
from mixedlm.estimation.laplace import _native_deviance_with_status
from mixedlm.estimation.reml import _build_lambda
from numpy.testing import assert_allclose

from tests._glmm_oracles import mode_problem


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("layout", ["intercept", "slope", "crossed", "fixed_only", "mode_only"])
@pytest.mark.parametrize("maxiter", [1, 100])
@pytest.mark.parametrize("zero_covariance", [False, True])
def test_laplace_correction_matches_independent_final_mode_information(
    kind, layout, maxiter, zero_covariance
):
    pytest.importorskip("mixedlm._rust")
    matrices, family, theta = mode_problem(kind, layout)
    if zero_covariance:
        theta[:] = 0
    deviance, beta, u, _ = _native_deviance_with_status(
        theta, matrices, family, 1, pirls_maxiter=maxiter, pirls_tol=1e-10
    )
    covariance = _build_lambda(theta, matrices.random_structures).toarray()
    # Every mode update lies in the row space of the covariance factor.
    spherical = np.linalg.pinv(covariance) @ u
    mean = family.link.inverse(matrices.X @ beta + matrices.Z @ u + matrices.offset)
    family.clamp_mu(mean, eps=1e-10, out=mean)
    conditional = np.sum(family.deviance_resids(matrices.y, mean, matrices.weights))
    weights = np.maximum(family.weights(mean) * matrices.weights, 1e-10)
    design = matrices.Z.toarray() @ covariance
    precision = np.eye(matrices.n_random) + design.T @ (weights[:, None] * design)
    sign, logdet = np.linalg.slogdet(precision)
    assert sign == 1
    assert_allclose(deviance, conditional + spherical @ spherical + logdet, rtol=1e-12, atol=1e-11)

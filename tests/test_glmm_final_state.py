"""Likelihood corrections use the same final mode as the returned estimates."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import families
from mixedlm.estimation.laplace import _native_deviance_with_status
from mixedlm.estimation.reml import _build_lambda, _count_theta
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose
from scipy import special


def mode_problem(kind, layout, *, n_obs=48, n_groups=4):
    rng = np.random.default_rng(3402)
    x = rng.uniform(-1, 1, n_obs)
    groups = np.arange(n_obs) % n_groups
    offset = 0.2 * np.cos(np.arange(n_obs))
    eta = 0.2 + 0.3 * x + 0.4 * np.sin(groups) + offset
    if kind == "binomial":
        trials = np.arange(n_obs) % 4 + 2
        y = rng.binomial(trials, special.expit(eta)) / trials
    elif kind == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = eta + rng.normal(scale=0.5, size=n_obs)
    data = pd.DataFrame(dict(y=y, x=x, g=groups, h=np.arange(n_obs) % 3))
    formulas = {
        "intercept": "y ~ x + (1 | g)",
        "slope": "y ~ x + (x | g)",
        "crossed": "y ~ x + (1 | g) + (1 | h)",
        "fixed_only": "y ~ x",
        "mode_only": "y ~ 0 + (1 | g)",
    }
    weights = np.geomspace(0.5, 2.0, n_obs)
    if kind == "binomial":
        weights *= trials
    matrices = build_model_matrices(
        parse_formula(formulas[layout]), data, weights=weights, offset=offset
    )
    family = {
        "gaussian": families.Gaussian,
        "binomial": families.Binomial,
        "poisson": families.Poisson,
    }[kind]()
    theta = np.full(_count_theta(matrices.random_structures), 0.4)
    return matrices, family, theta


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

"""nlmer against an exact marginal likelihood and the published lme4 Orange fit.

Orange (R datasets; Draper and Smith 1998) records the trunk circumference of
five trees at seven ages. In ``circumference ~ SSlogis(age, Asym, xmid, scal)``
with a random ``Asym`` per tree, the random parameter enters the mean linearly,
so the marginal likelihood is exactly Gaussian and the Laplace approximation is
exact. The oracle below evaluates that Gaussian likelihood directly.
"""

import numpy as np
import pandas as pd
import pytest
from mixedlm import nlme, nlmer
from numpy.testing import assert_allclose
from scipy import stats

AGES = np.array([118.0, 484.0, 664.0, 1004.0, 1231.0, 1372.0, 1582.0])
CIRCUMFERENCE = np.array(
    [
        [30, 58, 87, 115, 120, 142, 145],
        [33, 69, 111, 156, 172, 203, 203],
        [30, 51, 75, 108, 115, 139, 140],
        [32, 62, 112, 167, 179, 209, 214],
        [30, 49, 81, 125, 142, 174, 177],
    ],
    dtype=float,
)
# lme4's ?nlmer example: Asym, xmid, scal; Tree Asym SD; residual SD.
PUBLISHED_PHI = [192.053, 727.906, 348.073]
PUBLISHED_SD = [31.646, 7.843]


def exact_deviance(phi, tree_sd, residual_sd):
    """-2 log-likelihood of each tree's Gaussian marginal distribution."""
    asym, xmid, scal = phi
    shape = 1 / (1 + np.exp((xmid - AGES) / scal))
    covariance = residual_sd**2 * np.eye(len(AGES)) + tree_sd**2 * np.outer(shape, shape)
    marginal = stats.multivariate_normal(asym * shape, covariance)
    return -2 * sum(marginal.logpdf(tree) for tree in CIRCUMFERENCE)


@pytest.fixture(scope="module")
def orange_fit():
    data = pd.DataFrame(
        {
            "Tree": np.repeat([str(tree) for tree in range(1, 6)], len(AGES)),
            "age": np.tile(AGES, len(CIRCUMFERENCE)),
            "circumference": CIRCUMFERENCE.ravel(),
        }
    )
    return nlmer(
        nlme.SSlogis(),
        data,
        x_var="age",
        y_var="circumference",
        group_var="Tree",
        random_params=["Asym"],
        start={"Asym": 200.0, "xmid": 725.0, "scal": 350.0},
    )


def test_orange_deviance_is_the_exact_marginal_likelihood(orange_fit):
    tree_sd = orange_fit.theta[0] * orange_fit.sigma

    assert orange_fit.converged and orange_fit.pnls_converged
    assert orange_fit.deviance == pytest.approx(
        exact_deviance(orange_fit.phi, tree_sd, orange_fit.sigma), rel=1e-10
    )
    assert orange_fit.logLik().value == pytest.approx(-orange_fit.deviance / 2, rel=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="nlmer reports convergence at phi (191.06, 722.61, 344.20), 0.025 deviance "
    "units above the exact optimum that lme4 reaches",
)
def test_orange_fit_reaches_the_published_lme4_optimum(orange_fit):
    tree_sd = orange_fit.theta[0] * orange_fit.sigma

    assert orange_fit.deviance <= exact_deviance(PUBLISHED_PHI, *PUBLISHED_SD) + 1e-6
    assert_allclose(orange_fit.phi, PUBLISHED_PHI, rtol=1e-5)
    assert_allclose([tree_sd, orange_fit.sigma], PUBLISHED_SD, rtol=1e-4)

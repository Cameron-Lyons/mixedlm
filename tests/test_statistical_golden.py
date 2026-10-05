"""Canonical data checks against published lme4 results and observation-space GLS.

The LMM oracle builds the n-by-n marginal covariance directly and optimizes
SciPy's independent Gaussian likelihood. It does not call mixedlm's design,
covariance, likelihood, optimizer, or reporting implementations.
"""

from __future__ import annotations

import mixedlm as mlm
import numpy as np
import pytest
from mixedlm import families, pvalues
from numpy.testing import assert_allclose
from scipy import stats

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY
from tests._lmm_oracles import observation_space_reference

pytestmark = pytest.mark.installed_wheel

_CBPP_FLOAT_ATOL = 1e-6


@pytest.fixture(scope="class")
def model():
    return mlm.lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, REML=True)


@pytest.fixture(scope="class")
def sleepstudy_reference():
    return observation_space_reference(SLEEPSTUDY, slopes=True)


class TestSleepstudyGolden:
    def test_published_lme4_results(self, model):
        # https://lme4.github.io/lme4/reference/lmer.html
        # Respect published precision and optimizer variation for covariance.
        assert model.converged
        assert_allclose(model.beta, [251.40510, 10.46729], rtol=0, atol=5e-6)
        assert model.deviance == pytest.approx(1743.628, abs=5e-4)
        assert model.sigma == pytest.approx(25.592, abs=0.002)
        subject = model.VarCorr().groups["Subject"]
        assert subject.variance["(Intercept)"] == pytest.approx(612.10, abs=0.02)
        assert subject.variance["Days"] == pytest.approx(35.07, abs=0.01)

    def test_likelihood_and_variance_components(self, model, sleepstudy_reference):
        reference = sleepstudy_reference
        assert_allclose(model.beta, reference["beta"], rtol=0, atol=1e-10)
        assert_allclose(model.theta, reference["theta"], rtol=0, atol=2e-5)
        assert model.sigma == pytest.approx(reference["sigma"], abs=2e-5)
        assert model.deviance == pytest.approx(reference["deviance"], abs=2e-8)
        loglik = model.logLik()
        assert loglik.value == pytest.approx(-reference["deviance"] / 2, abs=1e-8)
        assert loglik.df == 6
        assert loglik.nobs == 180
        assert model.AIC() == pytest.approx(reference["deviance"] + 12, abs=2e-8)
        assert model.BIC() == pytest.approx(reference["deviance"] + 6 * np.log(180), abs=2e-8)
        lower = np.array([[reference["theta"][0], 0], reference["theta"][1:]])
        covariance = reference["sigma"] ** 2 * (lower @ lower.T)
        subject = model.VarCorr().groups["Subject"]
        assert subject.variance["(Intercept)"] == pytest.approx(covariance[0, 0], abs=0.002)
        assert subject.variance["Days"] == pytest.approx(covariance[1, 1], abs=0.0002)
        assert model.VarCorr().residual == pytest.approx(reference["sigma"] ** 2, abs=0.002)

    def test_vcov_residuals_fitted_and_pvalues(self, model, sleepstudy_reference):
        reference = sleepstudy_reference
        assert_allclose(model.vcov(), reference["vcov"], rtol=0, atol=2e-4)
        assert_allclose(model.fitted(), reference["fitted"], rtol=0, atol=2e-4)
        assert_allclose(model.residuals(), reference["residuals"], rtol=0, atol=2e-4)
        z = reference["beta"] / np.sqrt(np.diag(reference["vcov"]))
        expected_normal = 2 * stats.norm.sf(np.abs(z))
        # With identical time grids and a complete random intercept/slope
        # covariance, the balanced-study denominator df is groups minus one.
        expected_t = 2 * stats.t.sf(np.abs(z), df=17)
        for method, expected in (
            ("normal", expected_normal),
            ("Satterthwaite", expected_t),
            ("Kenward-Roger", expected_t),
        ):
            observed = pvalues(model, method=method)
            assert_allclose(list(observed.values()), expected, rtol=5e-4, atol=0)


def test_penicillin_crossed_random_effects_golden():
    data = mlm.load_penicillin()
    model = mlm.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data, REML=True)
    reference = observation_space_reference(data, slopes=False)
    # https://lme4.github.io/lme4/reference/Penicillin.html
    assert model.converged
    assert model.deviance == pytest.approx(330.8606, abs=5e-5)
    assert model.sigma == pytest.approx(0.5499, abs=5e-5)
    assert_allclose(model.beta, [data["diameter"].mean()], rtol=0, atol=1e-10)
    assert_allclose(model.theta, reference["theta"], rtol=0, atol=5e-5)
    assert model.sigma == pytest.approx(reference["sigma"], abs=2e-5)
    assert model.deviance == pytest.approx(reference["deviance"], abs=2e-8)
    assert_allclose(model.vcov(), reference["vcov"], rtol=0, atol=2e-5)
    assert_allclose(model.fitted(), reference["fitted"], rtol=0, atol=2e-5)
    assert_allclose(model.residuals(), reference["residuals"], rtol=0, atol=2e-5)
    varcorr = model.VarCorr()
    for group, theta in zip(("plate", "sample"), reference["theta"], strict=True):
        assert varcorr.groups[group].variance["(Intercept)"] == pytest.approx(
            (theta * reference["sigma"]) ** 2, abs=2e-4
        )
    assert varcorr.residual == pytest.approx(reference["sigma"] ** 2, abs=2e-5)


def test_cbpp_binomial_glmer_fast_approximation_golden() -> None:
    # Snapshot of the nAGQ=0 fit beyond lme4's published precision; its rounded
    # estimates are checked against lme4 in the next test. Full likelihood fits
    # have a separate independent integration oracle.
    model = mlm.glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), nAGQ=0)

    assert model.converged
    assert_allclose(
        model.beta,
        [-1.360471755378668, -0.976177470924035, -1.111076893294222, -1.559680515617786],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(model.theta, [0.641815070389878], rtol=0, atol=_CBPP_FLOAT_ATOL)
    assert model.sigma == pytest.approx(1.0, abs=0.0)
    assert model.deviance == pytest.approx(100.15188340645719, abs=_CBPP_FLOAT_ATOL)

    loglik = model.logLik()
    incidence, size = CBPP["incidence"], CBPP["size"]
    saturated_loglik = stats.binom.logpmf(incidence, size, incidence / size).sum()
    normalized_deviance = 100.15188340645719 - 2 * saturated_loglik
    assert loglik.value == pytest.approx(-0.5 * normalized_deviance, abs=_CBPP_FLOAT_ATOL)
    assert loglik.df == 5
    assert loglik.nobs == 56
    assert model.AIC() == pytest.approx(normalized_deviance + 10, abs=_CBPP_FLOAT_ATOL)
    assert model.BIC() == pytest.approx(normalized_deviance + 5 * np.log(56), abs=_CBPP_FLOAT_ATOL)
    assert_allclose(
        model.vcov(),
        [
            [0.051790208727, -0.024371090477, -0.024332790926, -0.024231789050],
            [-0.024371090477, 0.091983215914, 0.026523738201, 0.026293639936],
            [-0.024332790926, 0.026523738201, 0.104659886705, 0.025970642323],
            [-0.024231789050, 0.026293639936, 0.025970642323, 0.180162472641],
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(
        model.fitted()[:8],
        [
            0.309266031176,
            0.144336430227,
            0.128461611119,
            0.086019641495,
            0.155781652255,
            0.065001579149,
            0.057268396322,
            0.270616261989,
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(
        model.residuals()[:8],
        [
            -1.446018457708,
            0.960942734138,
            2.329516388142,
            -0.948399690844,
            -0.255661399684,
            -0.166471488371,
            -0.195723089899,
            0.952473483654,
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )

    herd = model.VarCorr().groups["herd"]
    assert herd.variance["(Intercept)"] == pytest.approx(0.41192658457956455, abs=_CBPP_FLOAT_ATOL)
    assert herd.stddev["(Intercept)"] == pytest.approx(0.641815070389878, abs=_CBPP_FLOAT_ATOL)


@pytest.mark.parametrize(
    "nAGQ,beta,scale",
    [
        (0, [-1.3605, -0.9762, -1.1111, -1.5597], 0.6418),
        (1, [-1.3983, -0.9919, -1.1282, -1.5797], 0.6421),
        (9, [-1.3992, -0.9914, -1.1278, -1.5795], 0.6475),
    ],
)
def test_original_cbpp_matches_published_lme4_estimates(nAGQ, beta, scale):
    # https://lme4.github.io/lme4/reference/glmer.html
    # Published rounded coefficients differ slightly with optimizer precision.
    model = mlm.glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), nAGQ=nAGQ)
    assert model.converged and model.pirls_converged
    assert model.ngrps()["herd"] == 15
    assert_allclose(model.beta, beta, rtol=0, atol=0.001)
    assert_allclose(model.theta, [scale], rtol=0, atol=0.0003)
    if nAGQ == 9:
        # This published higher-order criterion uses unit deviance residuals.
        assert model.deviance == pytest.approx(100.0100, abs=0.0001)

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
from scipy import linalg, optimize, stats

from tests._lmer_data import CBPP

_CBPP_FLOAT_ATOL = 1e-6


def observation_space_reference(data, *, slopes):
    y = data["Reaction" if slopes else "diameter"].to_numpy(dtype=float)
    n = len(y)
    if slopes:
        days = data["Days"].to_numpy(dtype=float)
        X = np.column_stack((np.ones(n), days))
        membership = data["Subject"].to_numpy()
        same = (membership[:, None] == membership[None, :]).astype(float)

        def covariance(theta):
            lower = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
            G = lower @ lower.T
            return np.eye(n) + same * (X @ G @ X.T)

        start = [1.0, 0.02, 0.2]
    else:
        X = np.ones((n, 1))
        grouping = [data[name].to_numpy() for name in ("plate", "sample")]
        same = [(values[:, None] == values[None, :]).astype(float) for values in grouping]

        def covariance(theta):
            return np.eye(n) + sum(
                value**2 * matrix for value, matrix in zip(theta, same, strict=True)
            )

        start = [1.5, 3.5]
    df = n - X.shape[1]

    def evaluate(theta):
        V = covariance(theta)
        factor = linalg.cho_factor(V, lower=True)
        inverse_X = linalg.cho_solve(factor, X)
        information = X.T @ inverse_X
        beta = np.linalg.solve(information, X.T @ linalg.cho_solve(factor, y))
        residuals = y - X @ beta
        inverse_residuals = linalg.cho_solve(factor, residuals)
        sigma_squared = float(residuals @ inverse_residuals / df)
        deviance = (
            2 * np.log(np.diag(factor[0])).sum()
            + np.linalg.slogdet(information)[1]
            + df * (1 + np.log(2 * np.pi * sigma_squared))
        )
        random_part = (V - np.eye(n)) @ inverse_residuals
        return {
            "deviance": float(deviance),
            "beta": beta,
            "sigma": np.sqrt(sigma_squared),
            "vcov": sigma_squared * np.linalg.inv(information),
            "fitted": X @ beta + random_part,
            "residuals": residuals - random_part,
        }

    optimum = optimize.minimize(
        lambda theta: evaluate(theta)["deviance"],
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 2000},
    )
    assert optimum.success, optimum.message
    reference = evaluate(optimum.x)
    reference["theta"] = optimum.x
    return reference


@pytest.fixture(scope="class")
def model():
    return mlm.lmer("Reaction ~ Days + (Days | Subject)", mlm.load_sleepstudy(), REML=True)


@pytest.fixture(scope="class")
def sleepstudy_reference():
    return observation_space_reference(mlm.load_sleepstudy(), slopes=True)


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


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply")
def test_cbpp_binomial_glmer_fast_approximation_golden() -> None:
    # Preserve the reference for the former joint PIRLS approximation.
    # Full likelihood fits have a separate independent integration oracle.
    data = CBPP.copy()
    model = mlm.glmer(
        "y ~ period + (1 | herd)",
        data,
        family=families.Binomial(),
        nAGQ=0,
        weights=data["size"].to_numpy(dtype=float),
    )

    assert model.converged
    assert_allclose(
        model.beta,
        [-1.861164732564236, -0.208553975355595, -0.078281733042295, -0.618291994021517],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(model.theta, [0.48750557774242], rtol=0, atol=_CBPP_FLOAT_ATOL)
    assert model.sigma == pytest.approx(1.0, abs=0.0)
    assert model.deviance == pytest.approx(74.03136198466316, abs=_CBPP_FLOAT_ATOL)

    loglik = model.logLik()
    saturated_loglik = stats.binom.logpmf(data["incidence"], data["size"], data["y"]).sum()
    normalized_deviance = 74.03136198466316 - 2 * saturated_loglik
    assert loglik.value == pytest.approx(-0.5 * normalized_deviance, abs=_CBPP_FLOAT_ATOL)
    assert loglik.df == 5
    assert loglik.nobs == 56
    assert model.AIC() == pytest.approx(normalized_deviance + 10, abs=_CBPP_FLOAT_ATOL)
    assert model.BIC() == pytest.approx(normalized_deviance + 5 * np.log(56), abs=_CBPP_FLOAT_ATOL)
    assert_allclose(
        model.vcov(),
        [
            [0.086506090158, -0.065213683451, -0.067123707411, -0.067040141073],
            [-0.065213683451, 0.160280508337, 0.065151552713, 0.064818269431],
            [-0.067123707411, 0.065151552713, 0.144882987863, 0.067106011166],
            [-0.067040141073, 0.064818269431, 0.067106011166, 0.189605383319],
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(
        model.fitted()[:8],
        [
            0.198309632363,
            0.167221722142,
            0.186157376781,
            0.117617807046,
            0.131593373719,
            0.109535223114,
            0.122902679701,
            0.075491973286,
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(
        model.residuals()[:8],
        [
            -0.541510359941,
            0.726873965828,
            1.773098171001,
            -1.118615177302,
            0.065853413303,
            -0.802052054265,
            0.272457372704,
            0.265837027814,
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )

    herd = model.VarCorr().groups["herd"]
    assert herd.variance["(Intercept)"] == pytest.approx(0.23766168832997042, abs=_CBPP_FLOAT_ATOL)
    assert herd.stddev["(Intercept)"] == pytest.approx(0.4875055777424197, abs=_CBPP_FLOAT_ATOL)


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
    model = mlm.glmer(
        "incidence / size ~ period + (1 | herd)",
        mlm.load_cbpp(),
        family=families.Binomial(),
        nAGQ=nAGQ,
    )
    assert model.converged and model.pirls_converged
    assert model.ngrps()["herd"] == 15
    assert_allclose(model.beta, beta, rtol=0, atol=0.001)
    assert_allclose(model.theta, [scale], rtol=0, atol=0.0003)
    if nAGQ == 9:
        # This published higher-order criterion uses unit deviance residuals.
        assert model.deviance == pytest.approx(100.0100, abs=0.0001)

"""Probability-distribution and integration oracles for GLMM likelihood reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, load_cbpp, model_selection
from scipy import integrate, special, stats


def _density(family, y, mu, weights, trials=None):
    """Use SciPy distributions independently of family deviance calculations."""
    if isinstance(family, families.Binomial):
        if trials is None and np.all((y == 0) | (y == 1)):
            return weights * stats.bernoulli.logpmf(y, mu)
        counts = weights if trials is None else trials
        return (weights / counts) * stats.binom.logpmf(np.rint(y * counts), counts, mu)
    if isinstance(family, families.Poisson):
        return weights * stats.poisson.logpmf(y, mu)
    if isinstance(family, families.NegativeBinomial):
        return weights * stats.nbinom.logpmf(y, family.theta, family.theta / (family.theta + mu))
    if isinstance(family, families.Gaussian):
        return stats.norm.logpdf(y, loc=mu, scale=1 / np.sqrt(weights))
    if isinstance(family, families.Gamma):
        return stats.gamma.logpdf(y, a=weights, scale=mu / weights)
    if isinstance(family, families.InverseGaussian):
        return stats.invgauss.logpdf(y, mu=mu / weights, scale=weights)
    raise AssertionError(f"Missing independent density for {family}")


_FAMILIES = [
    families.Binomial(),
    families.Poisson(),
    families.NegativeBinomial(theta=0.2),
    families.NegativeBinomial(theta=2.3),
    families.NegativeBinomial(theta=100),
    families.Gaussian(),
    families.Gamma(),
    families.GammaInverse(),
    families.InverseGaussian(),
    families.InverseGaussianCanonical(),
]


def _responses(family):
    if isinstance(family, families.Binomial):
        return np.array([0.0, 1.0, 0.0, 1.0])
    if isinstance(family, families.Gaussian):
        return np.array([-1.0, 0.4, 2.0, 3.0])
    if isinstance(family, (families.Poisson, families.NegativeBinomial)):
        return np.array([0.0, 1.0, 3.0, 8.0])
    return np.array([0.2, 0.8, 2.0, 5.0])


@pytest.mark.parametrize("family", _FAMILIES, ids=repr)
@pytest.mark.parametrize("weighted", [False, True])
def test_conditional_density_matches_scipy_and_deviance_difference(family, weighted):
    y = _responses(family)
    mu = np.array([0.1, 0.3, 0.5, 0.8]) if isinstance(family, families.Binomial) else y + 0.5
    weights = np.array([0.4, 1.3, 2.0, 0.8]) if weighted else np.ones(4)

    actual = family.log_likelihood(y, mu, weights)
    saturated = family.log_likelihood(y, y, weights)

    assert actual == pytest.approx(np.sum(_density(family, y, mu, weights)), abs=2e-12)
    assert saturated == pytest.approx(np.sum(_density(family, y, y, weights)), abs=2e-12)
    assert saturated - actual == pytest.approx(
        0.5 * np.sum(family.deviance_resids(y, mu, weights)), abs=2e-12
    )


@pytest.mark.parametrize("family", _FAMILIES, ids=repr)
@pytest.mark.parametrize("weighted", [False, True])
def test_real_model_reports_normalized_density_and_preserves_objective(family, weighted):
    frame = pd.DataFrame({"y": _responses(family)})
    weights = np.array([0.4, 1.3, 2.0, 0.8]) if weighted else np.ones(4)
    result = glmer("y ~ 1", frame, family=family, weights=weights)
    mu = result.fitted(na_expand=False)
    expected = float(np.sum(_density(family, result.matrices.y, mu, weights)))

    assert result.logLik().value == pytest.approx(expected, abs=2e-10)
    assert result.logLik().df == 1  # Dispersion and NB theta are fixed.
    assert result.get_deviance() == pytest.approx(-2 * expected, abs=4e-10)
    assert result.REMLcrit() == pytest.approx(-2 * expected, abs=4e-10)
    assert result.AIC() == pytest.approx(-2 * expected + 2, abs=4e-10)
    assert result.BIC() == pytest.approx(-2 * expected + np.log(4), abs=4e-10)
    assert result.extractAIC() == pytest.approx((1, -2 * expected + 2), abs=4e-10)
    assert result.deviance == pytest.approx(
        np.sum(family.deviance_resids(result.matrices.y, mu, weights)), abs=2e-10
    )
    assert result.as_function("deviance")(result.theta) == pytest.approx(result.deviance, abs=2e-10)


@pytest.mark.parametrize("explicit_trials", [False, True])
def test_grouped_binomial_prior_weights_preserve_trial_counts(explicit_trials):
    trials = np.array([2.0, 5.0, 10.0, 3.0])
    successes = np.array([0.0, 1.0, 6.0, 3.0])
    y, mu = successes / trials, np.array([0.1, 0.3, 0.5, 0.8])
    prior = np.array([0.4, 1.3, 2.0, 0.8]) if explicit_trials else np.ones(4)
    weights = trials * prior
    family = families.Binomial()
    trial_argument = trials if explicit_trials else None

    assert family.log_likelihood(y, mu, weights, trials=trial_argument) == pytest.approx(
        np.sum(prior * stats.binom.logpmf(successes, trials, mu)), abs=2e-12
    )
    assert family.log_likelihood(y, y, weights, trials=trial_argument) == pytest.approx(
        np.sum(prior * stats.binom.logpmf(successes, trials, y)), abs=2e-12
    )


def _integrated_binomial_loglik(result, successes, trials, groups, prior=None):
    """Integrate normalized binomial probabilities against a normal prior."""
    prior = np.ones(len(successes)) if prior is None else prior
    fixed = result.matrices.X @ result.beta + result.matrices.offset
    theta = float(result.theta[0])
    total = 0.0
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)

        def conditional_loglik(u, rows=rows):
            probabilities = special.expit(fixed[rows] + theta * u)
            return np.sum(
                prior[rows] * stats.binom.logpmf(successes[rows], trials[rows], probabilities)
            )

        reference = conditional_loglik(0.0)

        def integrand(u, reference=reference, conditional_loglik=conditional_loglik):
            return np.exp(conditional_loglik(u) - reference + stats.norm.logpdf(u))

        integral, error = integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-11)
        assert error < 1e-9 * integral
        total += np.log(integral) + reference
    return total


@pytest.fixture(scope="module", params=[0, 1, 9])
def cbpp_result(request):
    data = load_cbpp()
    return data, glmer("incidence / size ~ period + (1 | herd)", data, nAGQ=request.param)


def test_cbpp_normalized_reports_match_distribution_constant_and_official_laplace(cbpp_result):
    data, result = cbpp_result
    y = data.incidence.to_numpy() / data["size"].to_numpy()
    saturated = stats.binom.logpmf(data.incidence, data["size"], y).sum()
    assert saturated == pytest.approx(-41.97835376893558, abs=2e-12)
    expected = -0.5 * result.deviance + saturated

    assert result.logLik().value == pytest.approx(expected, abs=2e-12)
    assert result.get_deviance() == pytest.approx(-2 * expected, abs=2e-12)
    assert result.REMLcrit() == pytest.approx(-2 * expected, abs=2e-12)
    assert result.AIC() == pytest.approx(-2 * expected + 10, abs=2e-12)
    assert result.BIC() == pytest.approx(-2 * expected + 5 * np.log(56), abs=2e-12)
    summary = result.summary()
    assert "-2logL" in summary
    assert f"{expected:8.1f}" in summary
    # Published primary-source reference, not a snapshot of this implementation:
    # https://lme4.github.io/lme4/reference/glmer.html
    if result.nAGQ <= 1:
        official = {0: -92.0543, 1: -92.0266}[result.nAGQ]
        assert result.logLik().value == pytest.approx(official, abs=6e-4)
    else:
        oracle = _integrated_binomial_loglik(
            result, data.incidence.to_numpy(), data["size"].to_numpy(), data.herd.to_numpy()
        )
        assert result.logLik().value == pytest.approx(oracle, abs=4e-6)


def test_grouped_binomial_with_prior_weights_matches_direct_quadrature():
    data = load_cbpp()
    prior = np.linspace(0.8, 1.4, len(data))
    result = glmer("incidence / size ~ period + (1 | herd)", data, weights=prior, nAGQ=21)
    oracle = _integrated_binomial_loglik(
        result, data.incidence.to_numpy(), data["size"].to_numpy(), data.herd.to_numpy(), prior
    )
    assert result.logLik().value == pytest.approx(oracle, abs=2e-8)


def test_model_selection_propagates_normalized_real_model_reports():
    data = pd.DataFrame({"y": [0, 1, 3, 8, 0, 1, 2, 5], "x": np.arange(8) / 7})
    models = [glmer(formula, data, family=families.Poisson()) for formula in ("y ~ 1", "y ~ x")]
    selection = model_selection(*models, names=["intercept", "slope"])
    for index, model in enumerate(selection.models):
        expected = stats.poisson.logpmf(data.y, model.fitted(na_expand=False)).sum()
        df = model.matrices.n_fixed
        assert selection.loglik[index] == pytest.approx(expected, abs=2e-10)
        assert selection.aic[index] == pytest.approx(-2 * expected + 2 * df, abs=2e-10)
        assert selection.bic[index] == pytest.approx(-2 * expected + df * np.log(8), abs=2e-10)
        assert selection.aicc[index] == pytest.approx(
            -2 * expected + 2 * df + 2 * df * (df + 1) / (8 - df - 1), abs=2e-10
        )


def test_gaussian_marginal_density_matches_observation_space_covariance():
    rng = np.random.default_rng(117)
    group = np.repeat(np.arange(5), 6)
    x = rng.normal(size=len(group))
    weights = np.linspace(0.4, 2.5, len(group))
    y = 1.2 + 0.8 * x + np.repeat([-1.5, -0.3, 0.5, 1.8, -0.5], 6)
    y += rng.normal(scale=1 / np.sqrt(weights))
    data = pd.DataFrame({"y": y, "x": x, "g": group.astype(str)})
    result = glmer("y ~ x + (1 | g)", data, family=families.Gaussian(), weights=weights)

    covariance = np.diag(1 / weights) + result.theta[0] ** 2 * (group[:, None] == group)
    mean = result.beta[0] + result.beta[1] * x
    oracle = stats.multivariate_normal.logpdf(y, mean=mean, cov=covariance)
    assert result.theta[0] > 0.5
    assert result.logLik().value == pytest.approx(oracle, abs=2e-10)
    assert result.get_deviance() == pytest.approx(-2 * oracle, abs=4e-10)


@pytest.mark.parametrize(
    "family",
    [
        families.Gaussian(),
        families.Gamma(),
        families.GammaInverse(),
        families.InverseGaussian(),
        families.InverseGaussianCanonical(),
    ],
    ids=repr,
)
@pytest.mark.parametrize("nsim", [1, 4])
def test_continuous_simulation_uses_density_precision_weights(family, nsim):
    weights = np.array([0.4, 1.3, 2.0, 0.8])
    result = glmer("y ~ 1", pd.DataFrame({"y": _responses(family)}), family=family, weights=weights)
    rng = np.random.default_rng(173)
    oracle_rng = np.random.default_rng(173)
    mu = result.fitted(na_expand=False)
    if nsim > 1:
        mu = np.broadcast_to(mu[:, None], (len(mu), nsim))
        precision = weights[:, None]
    else:
        precision = weights
    if isinstance(family, families.Gaussian):
        expected = oracle_rng.normal(mu, 1 / np.sqrt(precision))
    elif isinstance(family, families.Gamma):
        expected = oracle_rng.gamma(precision, mu / precision)
    else:
        expected = oracle_rng.wald(mu, precision)
    np.testing.assert_array_equal(result.simulate(nsim=nsim, seed=rng, use_re=False), expected)
    if nsim == 1:
        from mixedlm.inference.bootstrap import _simulate_glmer

        np.testing.assert_array_equal(
            _simulate_glmer(result, rng=np.random.default_rng(173)), expected
        )


def test_custom_continuous_simulation_override_is_preserved():
    class CustomGaussian(families.Gaussian):
        def simulate(self, mu, rng=None):
            return np.full_like(mu, 123.0)

    result = glmer(
        "y ~ 1", pd.DataFrame({"y": [1, 2, 3]}), family=CustomGaussian(), weights=np.arange(1, 4)
    )
    np.testing.assert_array_equal(
        result.simulate(nsim=4, seed=173, use_re=False), np.full((3, 4), 123.0)
    )


class _CustomPoisson(families.CustomFamily):
    def __init__(self):
        super().__init__(link="log")

    def variance(self, mu):
        return mu

    def deviance_resids(self, y, mu, wt):
        return 2 * wt * (special.xlogy(y, y / mu) - y + mu)


@pytest.mark.parametrize(
    ("family", "error", "message"),
    [
        (_CustomPoisson(), NotImplementedError, "implement Family.log_likelihood"),
        (families.QuasiFamily(families.Poisson(), phi=2), ValueError, "quasi-likelihood"),
    ],
)
def test_unsupported_likelihoods_refuse_reports_but_keep_summary_usable(family, error, message):
    result = glmer("y ~ 1", pd.DataFrame({"y": [0, 1, 3, 8]}), family=family)
    for method in (result.logLik, result.AIC, result.BIC, result.extractAIC, result.get_deviance):
        with pytest.raises(error, match=message):
            method()
    with pytest.raises(error, match=message):
        model_selection(result, result)
    assert np.isfinite(result.deviance)
    assert "NA" in result.summary()
    assert message in result.summary()


def test_custom_density_hook_enables_normalized_reporting():
    class CustomDensity(_CustomPoisson):
        def log_likelihood(self, y, mu, wt, *, trials=None):
            return float(np.sum(wt * stats.poisson.logpmf(y, mu)))

    data = pd.DataFrame({"y": [0, 1, 3, 8]})
    result = glmer("y ~ 1", data, family=CustomDensity())
    oracle = stats.poisson.logpmf(data.y, result.fitted(na_expand=False)).sum()
    assert result.logLik().value == pytest.approx(oracle, abs=2e-10)


@pytest.mark.parametrize(
    ("family", "y", "weights"),
    [
        (families.Binomial(), [0.2, 0.8], [1, 1]),
        (families.Binomial(), [0.5, 0.5], [2.3, 2.3]),
        (families.Poisson(), [0.5, 2.5], [1, 1]),
        (families.NegativeBinomial(), [0.5, 2.5], [1, 1]),
    ],
)
def test_fractional_count_fits_do_not_invent_normalized_probabilities(family, y, weights):
    result = glmer("y ~ 1", pd.DataFrame({"y": y}), family=family, weights=np.array(weights))
    assert np.isfinite(result.deviance)
    with pytest.raises(ValueError, match="whole-number counts"):
        result.logLik()
    assert "NA" in result.summary()

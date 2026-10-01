"""Profile intervals agree with likelihood oracles, including asymmetric tails."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import GlmerControl, families, glmer
from mixedlm.estimation import joint_glmm, laplace
from mixedlm.inference import glmm_profile
from mixedlm.inference.profile import profile_glmer
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize, special, stats


def poisson_fit():
    return glmer("y ~ 1", pd.DataFrame({"y": [0.0, 0.0, 0.0, 1.0]}), family=families.Poisson())


@pytest.mark.parametrize("backend", ["native", "python"])
@pytest.mark.parametrize("level", [0.8, 0.95, 0.999])
@pytest.mark.parametrize("n_points", [3, 8, 15])
def test_poisson_profile_matches_analytic_likelihood(backend, level, n_points):
    with patch.object(laplace, "_HAS_RUST", backend == "native"):
        fitted = poisson_fit()
        before = fitted.beta.copy()
        profile = fitted.profile(n_points=n_points, level=level)["(Intercept)"]
    mle = np.log(0.25)
    cutoff = stats.chi2.isf(1 - level, 1)

    def ratio(value):
        return 2 * (4 * np.exp(value) - value - 1 + mle)

    expected = [
        optimize.brentq(lambda value: ratio(value) - cutoff, -20, mle),
        optimize.brentq(lambda value: ratio(value) - cutoff, mle, 5),
    ]
    assert_allclose([profile.ci_lower, profile.ci_upper], expected, atol=2e-7)
    assert_allclose(profile.mle, mle, atol=1e-7)
    assert_allclose(profile.zeta**2, ratio(profile.values), atol=2e-8)
    assert_allclose(profile.zeta[[0, -1]], [-np.sqrt(cutoff), np.sqrt(cutoff)], atol=2e-7)
    assert len(profile.values) == n_points
    assert np.all(np.diff(profile.values) > 0)
    assert profile.mle in profile.values
    assert_array_equal(fitted.beta, before)
    assert not np.allclose(expected, fitted.confint()["(Intercept)"])


def test_weighted_offset_poisson_profile_and_confint_agree():
    y = np.array([0.0, 1.0, 0.0, 2.0, 1.0])
    weights = np.array([0.5, 1.5, 2.0, 0.7, 1.0])
    offset = np.log([1.0, 2.0, 0.5, 3.0, 1.5])
    fitted = glmer(
        "y ~ 1", pd.DataFrame({"y": y}), weights=weights, offset=offset, family=families.Poisson()
    )
    profile = fitted.profile(n_points=7)["(Intercept)"]
    total = weights @ y
    exposure = weights @ np.exp(offset)
    mle = np.log(total / exposure)
    ratio = 2 * (exposure * np.exp(profile.values) - total * profile.values - total + total * mle)
    assert_allclose(profile.mle, mle, atol=2e-7)
    assert_allclose(profile.zeta**2, ratio, atol=1e-7)
    assert_allclose(
        fitted.confint(method="profile")["(Intercept)"], [profile.ci_lower, profile.ci_upper]
    )


def test_grouped_binomial_profile_uses_trial_counts_and_prior_weights():
    trials = np.array([5, 12, 6, 4, 10])
    successes = np.array([0, 2, 3, 1, 2])
    weights = np.array([0.5, 1, 2, 1, 0.7])
    fitted = glmer(
        "successes / trials ~ 1",
        pd.DataFrame(dict(successes=successes, trials=trials)),
        family=families.Binomial(),
        weights=weights,
    )
    profile = fitted.profile(n_points=9)["(Intercept)"]
    successes = weights @ successes
    trials = weights @ trials
    mle = special.logit(successes / trials)

    def nll(value):
        return trials * np.logaddexp(0, value) - successes * value

    assert_allclose(profile.mle, mle, atol=2e-7)
    assert_allclose(profile.zeta**2, 2 * (nll(profile.values) - nll(mle)), atol=2e-7)


def test_logistic_profile_reoptimizes_other_fixed_coefficients():
    rng = np.random.default_rng(840)
    x = rng.normal(loc=1, size=80)
    offset = np.linspace(-0.3, 0.4, 80)
    weights = np.linspace(0.7, 1.4, 80)
    y = rng.binomial(1, special.expit(-0.5 + 1.2 * x + offset))
    fitted = glmer(
        "y ~ x",
        pd.DataFrame(dict(y=y, x=x)),
        offset=offset,
        weights=weights,
        family=families.Binomial(),
    )
    profile = fitted.profile(which="x", n_points=7)["x"]

    def nll(beta):
        eta = beta[0] + beta[1] * x + offset
        return weights @ (np.logaddexp(0, eta) - y * eta)

    baseline = optimize.minimize(nll, fitted.beta, method="BFGS", tol=1e-9).fun
    independent = []
    for value in profile.values:
        nuisance = optimize.minimize_scalar(lambda intercept, v=value: nll([intercept, v]))
        independent.append(2 * (nuisance.fun - baseline))
    assert_allclose(profile.zeta**2, independent, atol=2e-6)
    fixed_nuisance = np.array(
        [2 * (nll([fitted.beta[0], value]) - baseline) for value in profile.values]
    )
    assert np.max(fixed_nuisance - profile.zeta**2) > 0.5


def random_intercept_fit(order=1):
    rng = np.random.default_rng(31)
    group = np.repeat(np.arange(8), 6)
    offset = 0.15 * np.sin(np.arange(len(group)))
    weights = np.linspace(0.8, 1.3, len(group))
    y = rng.poisson(np.exp(0.6 + np.repeat(np.linspace(-0.9, 0.9, 8), 6) + offset))
    fitted = glmer(
        "y ~ 1 + (1 | group)",
        pd.DataFrame(dict(y=y, group=group)),
        family=families.Poisson(),
        offset=offset,
        weights=weights,
        nAGQ=order,
        control=GlmerControl(tolPwrss=1e-10, pirls_maxiter=200),
    )
    return fitted, group


def test_random_covariance_is_reoptimized_against_independent_laplace_oracle():
    fitted, group = random_intercept_fit()
    profile = fitted.profile(n_points=7)["(Intercept)"]
    y, weights, offset = fitted.matrices.y, fitted.matrices.weights, fitted.matrices.offset

    def deviance(theta, beta):
        total = 0.0
        for index in range(8):
            keep = group == index
            yy, ww, oo = y[keep], weights[keep], offset[keep]
            mode = optimize.brentq(
                lambda u, w=ww, y=yy, o=oo: u - theta * (w @ (y - np.exp(beta + theta * u + o))),
                -30,
                30,
            )
            mu = np.exp(beta + theta * mode + oo)
            total += np.sum(2 * ww * (special.xlogy(yy, yy / mu) - yy + mu))
            total += mode**2 + np.log1p(theta**2 * (ww @ mu))
        return total

    baseline = optimize.minimize(
        lambda v: deviance(v[0], v[1]),
        np.r_[fitted.theta, fitted.beta],
        method="Nelder-Mead",
        bounds=[(0, 5), (None, None)],
        options={"xatol": 1e-10, "fatol": 1e-10},
    )
    assert baseline.success
    assert_allclose(profile.mle, baseline.x[1], atol=3e-6)
    ratios = []
    conditional = []
    for value in profile.values:
        nuisance = optimize.minimize_scalar(
            lambda theta, v=value: deviance(theta, v),
            bounds=(0, 5),
            method="bounded",
            options={"xatol": 1e-10},
        )
        ratios.append(nuisance.fun - baseline.fun)
        conditional.append(deviance(baseline.x[0], value) - baseline.fun)
    assert_allclose(profile.zeta**2, ratios, atol=3e-6)
    assert np.max(np.array(conditional) - profile.zeta**2) > 0.05


@pytest.mark.parametrize("order", [1, 7])
@pytest.mark.parametrize("backend", ["native", "python"])
def test_profiles_preserve_quadrature_controls_and_input_arrays(order, backend):
    with patch.object(laplace, "_HAS_RUST", backend == "native"):
        fitted, _ = random_intercept_fit(order)
        originals = [
            array.copy()
            for array in (fitted.beta, fitted.theta, fitted.matrices.X, fitted.matrices.offset)
        ]
        evaluate = joint_glmm.glmm_deviance_with_status
        calls = []

        def record(theta, matrices, family, **kwargs):
            calls.append(kwargs)
            return evaluate(theta, matrices, family, **kwargs)

        with patch.object(joint_glmm, "glmm_deviance_with_status", side_effect=record):
            profile = fitted.profile(n_points=3)["(Intercept)"]
    assert np.isfinite(profile.zeta).all()
    assert all(c == {"nAGQ": order, "pirls_maxiter": 200, "pirls_tol": 1e-10} for c in calls)
    for before, after in zip(
        originals,
        (fitted.beta, fitted.theta, fitted.matrices.X, fitted.matrices.offset),
        strict=True,
    ):
        assert_array_equal(before, after)


@pytest.mark.parametrize("value", [0, 1, 2, -1, 3.5, True, np.bool_(True), "5", None])
def test_invalid_grid_sizes_are_rejected_before_model_work(value):
    with pytest.raises(ValueError, match="n_points"):
        profile_glmer(object(), n_points=value)


@pytest.mark.parametrize("field", ["converged", "pirls_converged"])
def test_nonconverged_models_are_rejected(field):
    fitted = replace(poisson_fit(), **{field: False})
    with pytest.raises(ValueError, match="converged fitted model"):
        fitted.profile()


def test_solver_failures_do_not_return_wald_intervals():
    fitted = poisson_fit()
    with (
        patch.object(
            glmm_profile,
            "run_optimizer",
            return_value=SimpleNamespace(success=False, message="limit"),
        ),
        pytest.raises(RuntimeError, match="nuisance optimization failed: limit"),
    ):
        fitted.confint(method="profile")
    with (
        patch.object(joint_glmm, "glmm_deviance_with_status", return_value=(1, [], [], False)),
        pytest.raises(RuntimeError, match="converged inner PIRLS solve"),
    ):
        fitted.profile()


def test_empty_selections_need_no_likelihood_evaluations():
    fitted = poisson_fit()
    with patch.object(
        glmm_profile, "_GLMMProfileLikelihood", side_effect=AssertionError("unneeded")
    ):
        assert fitted.profile(which=[]) == {}
        assert fitted.profile(which="unknown") == {}


def test_adaptive_quadrature_profile_matches_independent_normal_integration():
    fitted, group = random_intercept_fit(order=11)
    profile = fitted.profile(n_points=5)["(Intercept)"]
    nodes, weights = np.polynomial.hermite.hermgauss(240)
    normal = np.sqrt(2) * nodes
    log_weights = np.log(weights) - np.log(np.pi) / 2
    y, prior, offset = fitted.matrices.y, fitted.matrices.weights, fitted.matrices.offset

    def deviance(theta, beta):
        total = 0.0
        for index in range(8):
            keep = group == index
            eta = beta + theta * normal[:, None] + offset[keep]
            conditional = np.sum(prior[keep] * (y[keep] * eta - np.exp(eta)), axis=1)
            total -= 2 * special.logsumexp(log_weights + conditional)
        return total

    baseline = optimize.minimize(
        lambda v: deviance(v[0], v[1]),
        np.r_[fitted.theta, fitted.beta],
        method="Nelder-Mead",
        bounds=[(0, 2), (None, None)],
        options={"xatol": 1e-10, "fatol": 1e-10},
    )
    assert baseline.success
    assert_allclose(profile.mle, baseline.x[1], atol=2e-5)
    for value, zeta in zip(profile.values, profile.zeta, strict=True):
        nuisance = optimize.minimize_scalar(
            lambda theta, v=value: deviance(theta, v),
            bounds=(0, 2),
            method="bounded",
            options={"xatol": 1e-10},
        )
        assert_allclose(zeta**2, nuisance.fun - baseline.fun, atol=3e-5)


def test_custom_family_profile_uses_python_likelihood():
    class CustomPoisson(families.Poisson):
        pass

    fitted = glmer("y ~ 1", pd.DataFrame({"y": [0.0, 0.0, 0.0, 1.0]}), family=CustomPoisson())
    with patch.object(
        laplace, "_native_deviance_with_status", side_effect=AssertionError("native")
    ):
        profile = fitted.profile(n_points=3)["(Intercept)"]
    assert_allclose([profile.ci_lower, profile.ci_upper], [-4.249964829, 0.095996331], atol=2e-7)


def test_insufficient_inner_iteration_limit_is_reported():
    fitted, _ = random_intercept_fit()
    with pytest.raises(RuntimeError, match="converged inner PIRLS solve"):
        replace(fitted, pirls_maxiter=1).profile()


def test_profile_reuses_evaluations_at_identical_parameter_values():
    fitted, _ = random_intercept_fit()
    likelihood = glmm_profile._GLMMProfileLikelihood(fitted, np.sqrt(np.diag(fitted.vcov())))
    minimum, optimum = likelihood.fit(likelihood.start)
    profile = glmm_profile._GLMMParameterProfile(likelihood, 1, optimum, minimum)
    with patch.object(likelihood, "fit", wraps=likelihood.fit) as evaluate:
        first = profile.deviance(profile.mle + 0.2)
        assert profile.deviance(profile.mle + 0.2) == first
        assert profile.deviance(profile.mle) == minimum
    assert evaluate.call_count == 1


def test_unbracketed_profile_raises_instead_of_substituting_wald():
    fitted = poisson_fit()
    with (
        patch.object(glmm_profile._GLMMParameterProfile, "deviance", return_value=fitted.deviance),
        pytest.raises(RuntimeError, match="Could not bracket"),
    ):
        fitted.confint(method="profile")


def test_confint_reuses_endpoint_fits_without_building_a_plot_grid():
    fitted, _ = random_intercept_fit()
    original = glmm_profile._GLMMProfileLikelihood.deviance
    evaluations = []

    def count(likelihood, parameters):
        evaluations.append(1)
        return original(likelihood, parameters)

    with patch.object(glmm_profile._GLMMProfileLikelihood, "deviance", count):
        intervals = fitted.confint(method="profile")["(Intercept)"]
        interval_evaluations = len(evaluations)
        evaluations.clear()
        profile = fitted.profile(n_points=20)["(Intercept)"]
    assert_array_equal(intervals, [profile.ci_lower, profile.ci_upper])
    assert interval_evaluations < len(evaluations)


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_coefficient_units_do_not_change_likelihood_intervals(scale):
    rng = np.random.default_rng(56)
    x = rng.normal(size=80)
    y = rng.poisson(np.exp(0.4 + 0.3 * x))
    ordinary = glmer("y ~ x", pd.DataFrame(dict(y=y, x=x)), family=families.Poisson())
    rescaled = glmer("y ~ x", pd.DataFrame(dict(y=y, x=x * scale)), family=families.Poisson())
    expected = ordinary.confint(parm="x", method="profile")["x"]
    actual = rescaled.confint(parm="x", method="profile")["x"]
    assert_allclose(np.array(actual) * scale, expected, rtol=2e-6, atol=2e-7)

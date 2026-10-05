"""Zero covariance gradients must not conceal better positive variances."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    glFormula,
    glmer,
    lFormula,
    lmer,
    mkGlmerDevfun,
    mkLmerDevfun,
    optimizeGlmer,
    optimizeLmer,
)
from mixedlm.estimation import laplace
from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.families import Poisson
from mixedlm.models.control import GlmerControl, LmerControl, glmerControl, lmerControl
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize, special, stats

from tests._lmm_oracles import linear_data


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("start", [0.0, 0.8, 2.0])
@pytest.mark.parametrize("reml", [False, True])
def test_linear_fit_recovers_analytic_balanced_variance(native, start, reml):
    data = linear_data()
    result = lmer(
        "y ~ 1 + (1 | g)",
        data,
        REML=reml,
        start=np.array([start]),
        control=LmerControl(optimizer="L-BFGS-B", use_rust=native),
    )
    # Orthogonal within/between group mean squares give the exact variance estimates.
    residual_variance = 4 / 3
    between_variance = 0.7**2 * (6 / 5 if reml else 1)
    expected_theta = np.sqrt((between_variance - residual_variance / 4) / residual_variance)
    assert result.converged
    assert_allclose(result.theta, [expected_theta], atol=2e-5)
    assert_allclose(result.sigma**2, residual_variance, atol=2e-5)
    assert not result.isSingular()


def test_linear_control_can_disable_restarts_and_modular_fit_uses_the_same_check():
    data = linear_data()
    with pytest.warns(UserWarning, match="singular"):
        disabled = lmer(
            "y ~ 1 + (1 | g)",
            data,
            REML=False,
            control=LmerControl(optimizer="L-BFGS-B", restart_edge=False),
        )
    assert disabled.theta[0] == 0
    devfun = mkLmerDevfun(lFormula("y ~ 1 + (1 | g)", data, REML=False))
    fixed = optimizeLmer(devfun)
    legacy = optimizeLmer(devfun, restart_edge=False)
    assert fixed.converged and legacy.converged
    assert_allclose(fixed.theta, [np.sqrt(0.1175)], atol=2e-5)
    assert_allclose(legacy.theta, disabled.theta)
    assert fixed.deviance < legacy.deviance - 0.3


@pytest.mark.parametrize("method", ["L-BFGS-B", "COBYQA"])
def test_genuine_zero_variance_is_retained(method):
    with pytest.warns(UserWarning, match="singular"):
        result = lmer(
            "y ~ 1 + (1 | g)",
            linear_data(0),
            REML=False,
            control=LmerControl(optimizer=method),
        )
    assert result.converged
    assert_allclose(result.theta, [0], atol=1e-7)
    assert_allclose(result.sigma, 1, atol=1e-7)
    assert_allclose(result.deviance, 24 * (1 + np.log(2 * np.pi)), atol=1e-9)


def poisson_deviance(theta, beta, data):
    total = 0.0
    for _, frame in data.groupby("g", sort=False):
        y = frame.y.to_numpy()
        mode = optimize.brentq(
            lambda u, y=y: u - theta * np.sum(y - np.exp(beta + theta * u)),
            -50,
            50,
        )
        mu = np.exp(beta + theta * mode)
        total += 2 * np.sum(special.xlogy(y, y / mu) - y + mu)
        total += mode**2 + np.log1p(theta**2 * len(y) * mu)
    return total


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("initialize", [False, True])
def test_generalized_zero_start_recovers_independent_laplace_optimum(native, initialize):
    data = pd.DataFrame({"y": np.repeat([1, 2, 3, 5, 8, 13], 4), "g": np.repeat(np.arange(6), 4)})
    with patch.object(laplace, "_HAS_RUST", native):
        result = glmer(
            "y ~ 1 + (1 | g)",
            data,
            family=Poisson(),
            start=np.zeros(1),
            control=GlmerControl(
                optimizer="L-BFGS-B",
                nAGQ0initStep=initialize,
                pirls_maxiter=200,
                tolPwrss=1e-10,
            ),
        )
    oracle = optimize.minimize(
        lambda x: poisson_deviance(x[0], x[1], data),
        [0.8, 1.5],
        method="Nelder-Mead",
        bounds=[(0, None), (None, None)],
        options={"xatol": 1e-9, "fatol": 1e-10},
    )
    assert result.converged and oracle.success
    assert_allclose(np.r_[result.theta, result.beta], oracle.x, atol=3e-5)
    assert_allclose(result.deviance, oracle.fun, atol=2e-7)


def test_generalized_modular_controls_retain_disabled_restart_choice():
    data = pd.DataFrame({"y": np.repeat([1, 2, 3, 5, 8, 13], 4), "g": np.repeat(np.arange(6), 4)})
    parsed = glFormula("y ~ 1 + (1 | g)", data, family=Poisson())
    fits = []
    for enabled in (False, True):
        control = GlmerControl(nAGQ0initStep=False, restart_edge=enabled)
        fitted = optimizeGlmer(mkGlmerDevfun(parsed, control=control), start=np.zeros(1))
        assert fitted.converged
        fits.append(fitted)
    assert fits[0].theta[0] == 0
    assert fits[1].theta[0] > 0.7
    assert fits[1].deviance < fits[0].deviance - 50


def test_generalized_profile_can_leave_a_true_zero_variance_fit():
    data = pd.DataFrame({"y": np.tile([1, 2, 3, 2], 6), "g": np.repeat(np.arange(6), 4)})
    with pytest.warns(UserWarning, match="singular"):
        result = glmer("y ~ 1 + (1 | g)", data, family=Poisson())
    assert result.theta[0] < 1e-6
    profile = result.profile("(Intercept)", level=0.9999, n_points=5)["(Intercept)"]
    minimum = poisson_deviance(0, np.log(2), data)
    fitted_variances = []
    for beta, zeta in zip(profile.values, profile.zeta, strict=True):
        nuisance = optimize.minimize_scalar(
            lambda theta, beta=beta: poisson_deviance(theta, beta, data),
            bounds=(0, 5),
            method="bounded",
            options={"xatol": 1e-10},
        )
        assert nuisance.success
        deviance = min(nuisance.fun, poisson_deviance(0, beta, data))
        assert_allclose(zeta**2, deviance - minimum, atol=2e-6)
        fitted_variances.append(nuisance.x)
    assert max(fitted_variances) > 0.2
    assert_allclose(profile.zeta[[0, -1]] ** 2, stats.chi2.isf(0.0001, 1), atol=2e-6)


def test_boundary_restart_counts_calls_preserves_options_and_respects_iteration_budget():
    calls = []

    def objective(x):
        calls.append(x.copy())
        return (x[0] ** 2 - 0.12) ** 2

    start = np.zeros(1)
    options = {"maxiter": 50, "gtol": 1e-9}
    result = run_optimizer(
        objective,
        start,
        "L-BFGS-B",
        [(0.0, None)],
        options=options,
        restart_edge=True,
    )
    assert result.success and "restart" in result.message
    assert_allclose(result.x, np.sqrt(0.12), atol=1e-6)
    assert result.nfev == len(calls)
    assert 0 < result.nit <= options["maxiter"]
    assert options == {"maxiter": 50, "gtol": 1e-9}
    assert_array_equal(start, [0])

    limited = run_optimizer(
        objective,
        start,
        "L-BFGS-B",
        [(0.0, None)],
        options={"maxiter": 1},
        restart_edge=True,
    )
    assert not limited.success
    assert limited.nit <= 1


def test_boundary_check_respects_function_evaluation_budget():
    calls = []

    def objective(x):
        calls.append(x.copy())
        return (x[0] ** 2 - 0.12) ** 2

    result = run_optimizer(
        objective,
        np.zeros(1),
        "L-BFGS-B",
        [(0.0, None)],
        options={"maxfun": 3},
        restart_edge=True,
    )
    assert not result.success
    assert "budget exhausted" in result.message.lower()
    assert result.nfev == len(calls) == 3


@pytest.mark.parametrize("method", ["L-BFGS-B", "SLSQP", "TNC", "COBYQA"])
def test_multiple_zero_variances_and_unbounded_nuisance_parameter(method):
    def objective(x):
        return (x[0] ** 2 - 0.12) ** 2 + (x[1] ** 2 - 0.64) ** 2 + (x[2] - 2) ** 2

    result = run_optimizer(
        objective,
        np.zeros(3),
        method,
        [(0.0, None), (0.0, None), (None, None)],
        options={"ftol": 1e-12} if method == "SLSQP" else None,
        restart_edge=True,
    )
    assert result.success
    assert_allclose(result.x, [np.sqrt(0.12), 0.8, 2], atol=1e-4)


def test_interior_fit_needs_no_extra_evaluations():
    calls = []

    def objective(x):
        calls.append(x.copy())
        return (x[0] - 2) ** 2

    results = []
    for enabled in (False, True):
        calls.clear()
        result = run_optimizer(
            objective,
            np.ones(1),
            "L-BFGS-B",
            [(0.0, None)],
            restart_edge=enabled,
        )
        results.append((result.x.copy(), result.fun, result.nit, len(calls)))
        assert result.nfev == len(calls)
    assert results[0] == results[1]


@pytest.mark.parametrize("factory", [LmerControl, GlmerControl, lmerControl, glmerControl])
def test_restart_control_is_boolean(factory):
    assert factory().restart_edge is True
    assert factory(restart_edge=False).restart_edge is False
    for invalid in [None, 0, 1, "yes", np.nan]:
        with pytest.raises(ValueError, match="restart_edge must be a boolean"):
            factory(restart_edge=invalid)

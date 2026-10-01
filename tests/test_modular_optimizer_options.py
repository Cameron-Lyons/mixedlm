"""Modular LMM fits honor solver options, tolerances, and restart budgets."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm import lFormula, lmer, lmerControl, mkLmerDevfun, optimizeLmer
from mixedlm.estimation import optimizers
from mixedlm.models.modular import LmerDevfun
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_variance_boundary_restarts import linear_data


def deviance_function(*, reml=False, **kwargs):
    parsed = lFormula("y ~ 1 + (1 | g)", linear_data(), REML=reml)
    return mkLmerDevfun(parsed, control=lmerControl(**kwargs))


def fit_with_calls(devfun, **kwargs):
    calls, forwarded = [], []
    original = optimizers.run_optimizer

    def capture(fun, *args, **options):
        forwarded.append({**options, "options": deepcopy(options["options"])})

        def counted(theta):
            calls.append(theta.copy())
            return fun(theta)

        return original(counted, *args, **options)

    with patch.object(optimizers, "run_optimizer", capture):
        result = optimizeLmer(devfun, **kwargs)
    return result, calls, forwarded


@pytest.mark.parametrize(
    "method,expected",
    [
        ("L-BFGS-B", {"maxiter": 7, "ftol": 3e-6, "gtol": 2e-4}),
        ("BFGS", {"maxiter": 7, "gtol": 2e-4}),
        ("Nelder-Mead", {"maxiter": 7, "fatol": 3e-6, "xatol": 4e-5}),
        ("Powell", {"maxiter": 7, "ftol": 3e-6, "xtol": 4e-5}),
        ("TNC", {"maxfun": 7, "ftol": 3e-6, "gtol": 2e-4, "xtol": 4e-5}),
        ("trust-constr", {"maxiter": 7, "gtol": 2e-4, "xtol": 4e-5}),
        ("SLSQP", {"maxiter": 7, "ftol": 3e-6}),
        ("COBYLA", {"maxiter": 7, "tol": 4e-5}),
        ("COBYQA", {"maxiter": 7}),
    ],
)
def test_tolerances_follow_the_requested_solver_and_call_limit(method, expected):
    devfun = deviance_function(
        optimizer="COBYQA", maxiter=99, ftol=3e-6, gtol=2e-4, xtol=4e-5, restart_edge=False
    )
    result, calls, forwarded = fit_with_calls(devfun, method=method, maxiter=7)
    assert forwarded[0]["method"] == method
    assert forwarded[0]["options"] == expected
    assert calls and np.isfinite(result.deviance)
    assert devfun.control.maxiter == 99
    assert devfun.control.optimizer == "COBYQA"


@pytest.mark.parametrize("method", ["L-BFGS-B", "BFGS", "TNC", "trust-constr"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("analytic", [False, True])
def test_gradient_tolerance_stops_the_actual_fit(method, override, native, analytic):
    devfun = deviance_function(
        use_rust=native,
        use_analytic_gradient=analytic,
        gtol=1e-12 if override else 1e20,
        optCtrl={"gtol": 1e20} if override else {},
        restart_edge=False,
    )
    start = np.ones(1)
    result = optimizeLmer(devfun, start=start, method=method)
    assert result.converged, result.message
    assert result.n_iter <= 1
    assert_array_equal(result.theta, start)
    assert result.deviance == devfun(start)


@pytest.mark.parametrize("limit_name", ["maxfev", "maxfun"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("restart", [False, True])
def test_cobyqa_evaluation_limit_including_legacy_alias(limit_name, native, restart):
    devfun = deviance_function(use_rust=native, optCtrl={limit_name: 1}, restart_edge=restart)
    result, calls, _ = fit_with_calls(devfun, method="COBYQA")
    assert not result.converged
    assert len(calls) == 1
    assert_array_equal(result.theta, calls[0])
    assert devfun.control.optCtrl == {limit_name: 1}


@pytest.mark.parametrize("options", [{"maxfun": 1}, {"maxiter": 1}, {"maxiter": 9, "maxfun": 1}])
@pytest.mark.parametrize("native", [False, True])
def test_tnc_limits_use_the_canonical_evaluation_option(options, native):
    devfun = deviance_function(use_rust=native, optCtrl=options, restart_edge=False)
    result, calls, forwarded = fit_with_calls(devfun, method="TNC", maxiter=200)
    assert not result.converged
    assert result.n_iter == 0
    # Numerical differentiation evaluates the starting point and its perturbation.
    assert len(calls) == 2
    assert forwarded[0]["options"]["maxfun"] == 1
    assert "maxiter" not in forwarded[0]["options"]
    assert devfun.control.optCtrl == options


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("analytic", [False, True])
def test_boundary_probes_share_the_control_evaluation_budget(native, analytic):
    devfun = deviance_function(
        use_rust=native, use_analytic_gradient=analytic, optCtrl={"maxfun": 3}
    )
    result, calls, _ = fit_with_calls(devfun, start=np.zeros(1))
    assert not result.converged
    assert "budget exhausted" in result.message.lower()
    assert len(calls) == 3
    assert 0 < result.theta[0] < 0.1
    assert result.deviance == devfun(result.theta)
    assert devfun.control.optCtrl == {"maxfun": 3}


@pytest.mark.parametrize("method", ["L-BFGS-B", "TNC", "COBYQA"])
def test_missing_control_preserves_the_legacy_options(method):
    devfun = replace(deviance_function(optCtrl={"maxiter": 1}), control=None)
    actual, _, forwarded = fit_with_calls(devfun, method=method, maxiter=4)
    expected = optimizers.run_optimizer(
        devfun,
        devfun.get_start(),
        method,
        devfun.get_bounds(),
        options={"maxiter": 4},
        restart_edge=True,
    )
    assert forwarded[0]["options"] == {"maxiter": 4}
    assert_array_equal(actual.theta, expected.x)
    assert actual.deviance == expected.fun
    assert actual.n_iter == expected.nit
    assert actual.converged == expected.success


def test_alias_normalization_and_repeated_fits_leave_the_control_unchanged():
    options = {"xatol": 2e-5, "fatol": 3e-6, "maxiter": 1}
    devfun = deviance_function(optCtrl=options, restart_edge=False)
    original = deepcopy(devfun.control)
    for method in ["Powell", "Nelder-Mead", "Powell"]:
        _, _, forwarded = fit_with_calls(devfun, method=method)
        expected = {"xtol": 2e-5, "ftol": 3e-6, "maxiter": 1} if method == "Powell" else options
        assert forwarded[0]["options"] == expected
        assert devfun.control == original
        assert devfun.control.optCtrl is options


def test_custom_deviance_callable_receives_control_options():
    class ShiftedDeviance(LmerDevfun):
        def __call__(self, theta):
            return super().__call__(theta) + 1e4

    original = deviance_function(use_analytic_gradient=True, optCtrl={"gtol": 1e20})
    devfun = ShiftedDeviance(original.parsed, original.optimizer, original.control)
    result = optimizeLmer(devfun)
    assert result.converged and result.n_iter == 0
    assert_array_equal(result.theta, devfun.get_start())
    assert result.deviance == devfun(result.theta)


@pytest.mark.parametrize(
    "options", [{"maxfun": 1, "maxfev": 2}, {"rhobeg": 0.2, "initial_tr_radius": 0.4}]
)
def test_conflicting_aliases_are_rejected_before_evaluating_the_objective(options):
    devfun = deviance_function(optCtrl=options)
    with (
        patch.object(devfun.optimizer, "objective", side_effect=AssertionError("unexpected call")),
        pytest.raises(ValueError),
    ):
        optimizeLmer(devfun, method="COBYQA")


@pytest.mark.parametrize("method", ["L-BFGS-B", "COBYQA", "TNC", "Powell"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_modular_fit_matches_public_fit_with_the_same_controls(method, native, reml):
    control = lmerControl(
        optimizer=method,
        maxiter=200,
        ftol=1e-12,
        gtol=1e-8,
        xtol=1e-8,
        use_rust=native,
        use_analytic_gradient=True,
    )
    start = np.ones(1)
    reference = lmer("y ~ 1 + (1 | g)", linear_data(), REML=reml, start=start, control=control)
    parsed = lFormula("y ~ 1 + (1 | g)", linear_data(), REML=reml)
    actual = optimizeLmer(
        mkLmerDevfun(parsed, control=control), start=start, method=method, maxiter=200
    )
    assert actual.converged and reference.converged
    assert_allclose(actual.theta, reference.theta, rtol=0, atol=1e-10)
    assert_allclose(actual.deviance, reference.deviance, rtol=0, atol=1e-11)

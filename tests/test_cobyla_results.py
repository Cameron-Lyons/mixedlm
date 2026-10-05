"""COBYLA returns usable fits when SciPy reports evaluations without iterations."""

import numpy as np
import pytest
from mixedlm import GlmerControl, glmer, lmer, lmerControl, load_sleepstudy
from mixedlm.estimation import optimizers
from mixedlm.estimation.optimizers import run_optimizer
from numpy.testing import assert_allclose

from tests.test_joint_glmm_optimization import independent_deviance, model_data
from tests.test_statistical_golden import observation_space_reference
from tests.test_variance_boundary_restarts import linear_data


@pytest.mark.parametrize("limit", [3, 4])
@pytest.mark.parametrize("tolerance", [0.1, 0.5, 1.0])
def test_cobyla_boundary_probes_respect_its_evaluation_limit(limit, tolerance):
    calls = []

    def objective(theta):
        calls.append(theta.copy())
        return theta[0] ** 2

    result = run_optimizer(
        objective,
        np.zeros(1),
        "COBYLA",
        [(0.0, None)],
        options={"maxiter": limit, "tol": tolerance},
        restart_edge=True,
    )
    assert result.nfev == len(calls) <= limit
    assert not result.success
    assert_allclose(result.x, [0.0], atol=0)


@pytest.mark.parametrize("limit", [6, 7, 9, 20])
def test_cobyla_restarts_receive_only_the_remaining_evaluations(monkeypatch, limit):
    calls, stages = [], []
    original = optimizers._run_optimizer_once

    def objective(theta):
        calls.append(theta.copy())
        return (theta[0] ** 2 - 0.12) ** 2

    def capture(fun, start, method, bounds, options, *args):
        stages.append((len(calls), dict(options)))
        return original(fun, start, method, bounds, options, *args)

    monkeypatch.setattr(optimizers, "_run_optimizer_once", capture)
    result = run_optimizer(
        objective,
        np.zeros(1),
        "COBYLA",
        [(0.0, None)],
        options={"maxiter": limit, "tol": 0.5},
        restart_edge=True,
    )
    assert result.nfev == len(calls) <= limit
    assert result.fun < 0.12**2
    assert result.x[0] > 0
    for consumed, options in stages:
        assert options["maxiter"] == limit - consumed
    if limit == 20:
        assert len(stages) >= 2
    elif limit <= 7:
        assert not result.success
        assert "budget exhausted" in result.message.lower()


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("limit", [5, 300])
def test_cobyla_normalizes_successful_and_budget_limited_results(restart, limit):
    calls = []

    def objective(theta):
        calls.append(theta.copy())
        return (theta[0] - 2.0) ** 2 + (theta[1] + 0.3) ** 2

    result = run_optimizer(
        objective,
        np.zeros(2),
        "COBYLA",
        [(0.0, None), (None, None)],
        options={"maxiter": limit, "tol": 1e-8},
        restart_edge=restart,
    )
    assert result.nit == result.nfev == len(calls)
    assert result.nit <= limit
    assert np.isfinite(result.fun)
    assert result.jac is None
    if limit == 5:
        assert not result.success
        assert result.nfev == limit
    else:
        assert result.success, result.message
        assert_allclose(result.x, [2.0, -0.3], atol=1e-6)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_public_lmm_cobyla_fit_matches_balanced_variance(native, reml):
    result = lmer(
        "y ~ 1 + (1 | g)",
        linear_data(),
        REML=reml,
        control=lmerControl(optimizer="COBYLA", use_rust=native, xtol=1e-8),
    )
    expected = np.sqrt((0.7**2 * (6 / 5 if reml else 1) - 1 / 3) / (4 / 3))
    assert result.converged
    assert result.n_iter > 0
    assert_allclose(result.theta, [expected], atol=2e-6)


def test_cobyla_returns_contiguous_parameters():
    # SciPy's COBYLA returns a strided view, which native evaluators reject.
    result = run_optimizer(
        lambda x: np.sum((x - [1.0, -2.0, 0.5]) ** 2),
        np.zeros(3),
        "COBYLA",
        [(0.0, None), (None, None), (None, None)],
    )
    assert result.x.flags.c_contiguous and result.x.dtype == np.float64
    assert_allclose(result.x, [1.0, -2.0, 0.5], atol=1e-4)


@pytest.fixture(scope="module")
def sleepstudy_reference():
    return observation_space_reference(load_sleepstudy(), slopes=True)


@pytest.mark.parametrize("native", [False, True])
def test_public_lmm_cobyla_fit_handles_correlated_slopes(native, sleepstudy_reference):
    result = lmer(
        "Reaction ~ Days + (Days | Subject)",
        load_sleepstudy(),
        control=lmerControl(optimizer="COBYLA", use_rust=native),
    )
    assert result.converged
    assert_allclose(result.deviance, sleepstudy_reference["deviance"], rtol=0, atol=1e-4)


@pytest.mark.parametrize("kind", ["poisson", "binomial"])
@pytest.mark.parametrize("initialize", [False, True])
def test_public_glmm_cobyla_fit_matches_independent_likelihood(kind, initialize):
    formula, data, family, weights, offset, groups = model_data(kind)
    result = glmer(
        formula,
        data,
        family=family,
        weights=weights,
        offset=offset,
        control=GlmerControl(optimizer="COBYLA", maxiter=2000, xtol=1e-8, nAGQ0initStep=initialize),
    )
    assert result.converged
    assert result.n_iter > 0
    objective = independent_deviance(result, groups, kind, 1)
    parameters = np.r_[result.theta, result.beta]
    assert_allclose(result.deviance, objective(parameters), rtol=0, atol=1e-8)
    # Check the fitted joint optimum against an independent scalar-mode oracle.
    from scipy.optimize import minimize

    reference = minimize(
        objective,
        parameters,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10},
    )
    assert reference.success
    assert_allclose(result.deviance, reference.fun, rtol=0, atol=1e-6)

"""Joint fits optimize the integrated likelihood, with an explicit fast mode."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    GlmerControl,
    families,
    glFormula,
    glmer,
    mkGlmerDevfun,
    mkGlmerMod,
    optimizeGlmer,
)
from mixedlm.estimation import laplace
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize, special


def model_data(kind):
    rng = np.random.default_rng(318)
    group = np.repeat(np.arange(8), 8)
    x = rng.normal(scale=0.4, size=len(group))
    offset = 0.15 * np.sin(np.arange(len(group)))
    weights = np.linspace(0.8, 1.3, len(group))
    eta = 0.4 + 0.5 * x + np.repeat(np.linspace(-1, 1, 8), 8) + offset
    if kind == "poisson":
        y = rng.poisson(np.exp(eta))
        data = pd.DataFrame(dict(y=y, x=x, g=group))
        formula = "y ~ x + (1 | g)"
        family = families.Poisson()
    else:
        trials = np.arange(len(group)) % 5 + 3
        successes = rng.binomial(trials, special.expit(eta))
        data = pd.DataFrame(dict(successes=successes, trials=trials, x=x, g=group))
        formula = "successes / trials ~ x + (1 | g)"
        family = families.Binomial()
    return formula, data, family, weights, offset, group


def independent_deviance(fitted, group, kind, order):
    y, weights, offset, x = (
        fitted.matrices.y,
        fitted.matrices.weights,
        fitted.matrices.offset,
        fitted.matrices.X,
    )
    if kind == "poisson":
        constant = 2 * np.sum(weights * (special.xlogy(y, y) - y))
    else:
        constant = 2 * np.sum(weights * (special.xlogy(y, y) + special.xlogy(1 - y, 1 - y)))
    nodes, node_weights = np.polynomial.hermite.hermgauss(240)
    nodes = np.sqrt(2) * nodes
    log_weights = np.log(node_weights) - np.log(np.pi) / 2

    def objective(parameters):
        theta, beta = parameters[0], parameters[1:]
        total = constant
        for index in np.unique(group):
            keep = group == index
            eta = x[keep] @ beta + offset[keep]
            yy, ww = y[keep], weights[keep]
            if order > 1:
                linear = eta + theta * nodes[:, None]
                cumulant = np.exp(linear) if kind == "poisson" else np.logaddexp(0, linear)
                log_likelihood = np.sum(ww * (yy * linear - cumulant), axis=1)
                total -= 2 * special.logsumexp(log_weights + log_likelihood)
            else:
                inverse = np.exp if kind == "poisson" else special.expit
                mode = optimize.brentq(
                    lambda u, w=ww, y=yy, e=eta, inv=inverse: u
                    - theta * (w @ (y - inv(e + theta * u))),
                    -30,
                    30,
                )
                linear = eta + theta * mode
                mu = inverse(linear)
                variance = mu if kind == "poisson" else mu * (1 - mu)
                cumulant = mu if kind == "poisson" else np.logaddexp(0, linear)
                total += 2 * np.sum(ww * (cumulant - yy * linear))
                total += mode**2 + np.log1p(theta**2 * (ww @ variance))
        return total

    return objective


@pytest.mark.parametrize("kind", ["poisson", "binomial"])
@pytest.mark.parametrize("order", [1, 15])
@pytest.mark.parametrize("native", [False, True])
def test_joint_fit_matches_independent_integrated_likelihood(kind, order, native):
    formula, data, family, weights, offset, group = model_data(kind)
    with patch.object(laplace, "_HAS_RUST", native):
        fitted = glmer(
            formula,
            data,
            family=family,
            weights=weights,
            offset=offset,
            nAGQ=order,
            control=GlmerControl(tolPwrss=1e-10, pirls_maxiter=200),
        )
        fast = glmer(formula, data, family=family, weights=weights, offset=offset, nAGQ=0)
    assert fitted.converged and fitted.joint_fit
    assert fast.converged and not fast.joint_fit
    objective = independent_deviance(fitted, group, kind, order)
    parameters = np.r_[fitted.theta, fitted.beta]
    reference = optimize.minimize(
        objective,
        parameters,
        method="Nelder-Mead",
        bounds=[(0, 3), (None, None), (None, None)],
        options={"xatol": 1e-9, "fatol": 1e-10},
    )
    assert reference.success
    assert_allclose(parameters, reference.x, atol=3e-5, rtol=3e-5)
    assert_allclose(fitted.deviance, reference.fun, atol=2e-5, rtol=1e-7)
    assert objective(np.r_[fast.theta, fast.beta]) - objective(parameters) > 1e-4
    assert np.max(np.abs(fitted.beta - fast.beta)) > 1e-3


@pytest.mark.parametrize("initialize", [False, True])
@pytest.mark.parametrize("order", [0, 1, 7])
def test_initialization_control_and_iteration_counts(initialize, order):
    formula, data, family, weights, offset, _ = model_data("poisson")
    run = laplace.run_optimizer
    dimensions, iterations = [], []

    def record(fun, start, **kwargs):
        dimensions.append(len(start))
        fitted = run(fun, start, **kwargs)
        iterations.append(fitted.nit)
        return fitted

    with patch.object(laplace, "run_optimizer", side_effect=record):
        fitted = glmer(
            formula,
            data,
            family=family,
            weights=weights,
            offset=offset,
            nAGQ=order,
            control=GlmerControl(nAGQ0initStep=initialize),
        )
    assert fitted.converged
    assert dimensions == ([1] if order == 0 else [1, 3] if initialize else [3])
    assert fitted.n_iter == sum(iterations)
    assert fitted.joint_fit == (order != 0)


@pytest.mark.parametrize("order", [0, 1, 7])
def test_modular_joint_and_fast_results_match_public_fits(order):
    formula, data, family, weights, offset, _ = model_data("poisson")
    control = GlmerControl(nAGQ0initStep=False)
    parsed = glFormula(formula, data, family=family, weights=weights, offset=offset)
    devfun = mkGlmerDevfun(parsed, control=control, nAGQ=order)
    opt = optimizeGlmer(devfun, method="COBYQA")
    result = mkGlmerMod(devfun, opt)
    expected = glmer(
        formula, data, family=family, weights=weights, offset=offset, control=control, nAGQ=order
    )
    assert_allclose(result.beta, expected.beta, atol=1e-10)
    assert_allclose(result.theta, expected.theta, atol=1e-10)
    assert result.deviance == expected.deviance
    assert result.joint_fit == expected.joint_fit == (order != 0)
    if order:
        assert len(devfun.get_start(joint=True)) == len(devfun.get_bounds(joint=True)) == 3
        assert_allclose(devfun(np.r_[opt.theta, opt.beta]), result.deviance, atol=1e-9)
    else:
        assert opt.beta is None
        assert devfun(opt.theta) == result.deviance


@pytest.mark.parametrize("order", [0, 1, 7])
def test_refits_and_reconstructed_objectives_keep_the_fitting_mode(order):
    formula, data, family, weights, offset, _ = model_data("poisson")
    fitted = glmer(formula, data, family=family, weights=weights, offset=offset, nAGQ=order)
    refitted = fitted.refit(method="COBYQA", nAGQ0initStep=False)
    assert refitted.nAGQ == order
    assert refitted.joint_fit == fitted.joint_fit == (order != 0)
    assert_allclose(refitted.beta, fitted.beta, atol=3e-5)
    assert_allclose(refitted.deviance, fitted.deviance, atol=2e-7)
    objective = fitted.as_function()
    assert_allclose(objective(fitted.theta), fitted.deviance, atol=1e-8)
    if order:
        assert_allclose(objective(np.r_[fitted.theta, fitted.beta]), fitted.deviance, atol=1e-8)
        shifted = np.r_[fitted.theta, fitted.beta + 0.1]
        assert objective(shifted) > fitted.deviance


@pytest.mark.parametrize("workflow", ["update", "bootstrap", "allfit", "drop1", "cv"])
def test_derived_fits_preserve_explicit_fast_mode(workflow):
    from mixedlm.inference.allfit import allfit_glmer
    from mixedlm.inference.bootstrap import bootstrap_glmer
    from mixedlm.inference.cross_validation import cross_validate
    from mixedlm.inference.drop1 import drop1_glmer

    formula, data, family, weights, offset, _ = model_data("poisson")
    fitted = glmer(formula, data, family=family, weights=weights, offset=offset, nAGQ=0)
    outcomes = []
    original = laplace.GLMMOptimizer.optimize

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        outcomes.append((self.nAGQ, result.joint_fit))
        return result

    with patch.object(laplace.GLMMOptimizer, "optimize", record):
        if workflow == "update":
            updated = fitted.update(data=data)
            assert_allclose(updated.beta, fitted.beta, atol=1e-8)
        elif workflow == "bootstrap":
            samples = bootstrap_glmer(fitted, n_boot=3, seed=25, n_jobs=1)
            assert samples.n_failed == 0
        elif workflow == "allfit":
            compared = allfit_glmer(fitted, data, optimizers=["COBYQA"], n_jobs=1)
            assert not compared.errors
        elif workflow == "drop1":
            compared = drop1_glmer(fitted, data, n_jobs=1)
            assert compared.terms == ["x"]
        else:
            cross_validate(fitted, data, cv=2, random_state=23, n_jobs=1)
    assert outcomes
    assert all(order == 0 and not joint for order, joint in outcomes)


@pytest.mark.parametrize("kind", ["poisson", "gaussian", "no_fixed"])
def test_exact_pirls_cases_avoid_redundant_joint_optimization(kind):
    formula, data, _, weights, offset, _ = model_data("poisson")
    family = families.Gaussian() if kind == "gaussian" else families.Poisson()
    if kind == "poisson":
        formula = "y ~ x"
    elif kind == "no_fixed":
        formula = "y ~ 0 + (1 | g)"
    run = laplace.run_optimizer
    with patch.object(laplace, "run_optimizer", wraps=run) as outer:
        fitted = glmer(formula, data, family=family, weights=weights, offset=offset)
    assert fitted.converged and fitted.joint_fit
    assert outer.call_count == 1
    fast = glmer(formula, data, family=family, weights=weights, offset=offset, nAGQ=0)
    assert_array_equal(fast.beta, fitted.beta)
    assert_array_equal(fast.theta, fitted.theta)
    assert fast.deviance == fitted.deviance


def test_joint_objective_preserves_model_arrays_and_checks_parameter_shape():
    formula, data, family, weights, offset, _ = model_data("poisson")
    parsed = glFormula(formula, data, family=family, weights=weights, offset=offset)
    before = parsed.matrices.offset.copy(), parsed.matrices.X.copy()
    objective = JointGLMMObjective(parsed.matrices, family)
    parameters = np.array([0.5, 0.4, 0.5])
    first = objective.evaluate(parameters)
    second = objective.evaluate(parameters)
    assert first[0] == second[0]
    assert_array_equal(first[1], parameters[1:])
    assert_array_equal(parsed.matrices.offset, before[0])
    assert_array_equal(parsed.matrices.X, before[1])
    for invalid in (np.ones(2), np.ones((3, 1)), [1, np.nan, 0], [1, 1j, 0]):
        with pytest.raises(ValueError, match="joint parameters"):
            objective(invalid)


def test_profile_refinement_warns_for_explicit_fast_approximation():
    formula, data, family, weights, offset, _ = model_data("poisson")
    fitted = glmer(formula, data, family=family, weights=weights, offset=offset, nAGQ=0)
    with pytest.warns(UserWarning, match="refined the joint optimum"):
        profile = fitted.profile(which="(Intercept)", n_points=3)["(Intercept)"]
    assert abs(profile.mle - fitted.beta[0]) > 1e-3


def test_cbpp_joint_fit_matches_independent_laplace_optimization():
    from tests._lmer_data import CBPP

    fitted = glmer(
        "y ~ period + (1 | herd)",
        CBPP,
        family=families.Binomial(),
        weights=CBPP["size"].to_numpy(dtype=float),
    )
    objective = independent_deviance(fitted, CBPP["herd"].to_numpy(), "binomial", 1)
    point = np.r_[fitted.theta, fitted.beta]
    reference = optimize.minimize(
        objective,
        point,
        method="Nelder-Mead",
        bounds=[(0, 5)] + [(None, None)] * len(fitted.beta),
        options={"xatol": 1e-9, "fatol": 1e-10},
    )
    assert fitted.converged and reference.success
    assert_allclose(point, reference.x, atol=5e-6)
    assert_allclose(fitted.deviance, reference.fun, atol=1e-8)


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_joint_fitting_is_stable_when_predictor_units_change(scale):
    formula, data, family, weights, offset, _ = model_data("poisson")
    original = glmer(formula, data, family=family, weights=weights, offset=offset)
    changed = data.assign(x=data.x * scale)
    rescaled = glmer(formula, changed, family=family, weights=weights, offset=offset)
    assert original.converged and rescaled.converged
    assert_allclose(rescaled.beta * [1, scale], original.beta, atol=2e-6)
    assert_allclose(rescaled.theta, original.theta, atol=2e-6)
    assert_allclose(rescaled.deviance, original.deviance, atol=2e-8)


@pytest.mark.parametrize("value", [0, 1, "yes", None, [], np.nan])
def test_initialization_flag_rejects_non_boolean_values(value):
    with pytest.raises(ValueError, match="nAGQ0initStep must be a boolean"):
        GlmerControl(nAGQ0initStep=value)

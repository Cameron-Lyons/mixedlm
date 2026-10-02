"""Compare constrained PIRLS modes with independently minimized likelihoods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer
from mixedlm.estimation.laplace import _compute_group_quadrature, _pirls_mean, _pirls_state
from mixedlm.families import Binomial, Gamma, Gaussian, IdentityLink, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import GlmerControl
from mixedlm.utils.quadrature import hermite_rule
from numpy.testing import assert_allclose
from scipy import integrate, optimize


def _inverse_gamma_data():
    # Low precision in the last group previously let a full PIRLS step cross
    # eta=0 and converge to a clamped mean of 1e10 with deviance over 1700.
    rng = np.random.default_rng(241)
    group = np.repeat(np.arange(8), 8)
    x = np.tile(np.linspace(-0.5, 0.5, 8), 8)
    eta = 1.5 + 0.4 * x + rng.normal(scale=0.25, size=8)[group]
    offset = 0.04 * np.sin(np.arange(len(group)))
    weights = np.linspace(2, 5, len(group))
    y = rng.gamma(weights, (1 / (eta + offset)) / weights)
    return pd.DataFrame({"y": y, "x": x, "group": group.astype(str)}), weights, offset


def _independent_inverse_gamma_mode(matrices, theta):
    # This fixture has one random intercept per group: its spherical design is
    # simply theta * Z, independent of the production covariance builder.
    design = np.column_stack([matrices.X, matrices.Z.toarray() * theta])
    p = matrices.n_fixed
    weights = matrices.weights
    y = matrices.y

    def objective(parameters):
        eta = design @ parameters + matrices.offset
        if np.any(eta <= 0):
            return np.inf
        return np.sum(2 * weights * (y * eta - 1 - np.log(y * eta))) + np.sum(parameters[p:] ** 2)

    def gradient(parameters):
        eta = design @ parameters + matrices.offset
        result = design.T @ (2 * weights * (y - 1 / eta))
        result[p:] += 2 * parameters[p:]
        return result

    start = np.r_[1.5, 0.0, np.zeros(matrices.n_random)]
    result = optimize.minimize(
        objective,
        start,
        jac=gradient,
        constraints=optimize.LinearConstraint(design, 1e-10 - matrices.offset, np.inf),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success, result.message
    return result


@pytest.mark.parametrize("order", [0, 1])
def test_public_inverse_gamma_fit_keeps_original_low_precision_data_inside_domain(order):
    data, weights, offset = _inverse_gamma_data()
    model = glmer(
        "y ~ x + (1 | group)",
        data,
        family=Gamma(link="inverse"),
        weights=weights,
        offset=offset,
        nAGQ=order,
        control=GlmerControl(tolPwrss=1e-10, pirls_maxiter=100),
    )

    assert model.converged and model.pirls_converged
    assert np.all(model.linear_predictor() > 0)
    assert np.all(model.fitted() < 10)
    assert model.deviance < 100
    if order == 0:
        reference = _independent_inverse_gamma_mode(model.matrices, model.theta[0])
        actual = np.r_[model.beta, model.u / model.theta[0]]
        assert_allclose(actual, reference.x, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("invalid_start", [False, True])
def test_inverse_gamma_mode_matches_constrained_optimizer(invalid_start):
    data, weights, offset = _inverse_gamma_data()
    matrices = build_model_matrices(
        parse_formula("y ~ x + (1 | group)"), data, weights=weights, offset=offset
    )
    theta = np.array([1.25535454])
    kwargs = {"beta_start": np.array([-1.0, 1.0])} if invalid_start else {}
    state = _pirls_state(matrices, Gamma(link="inverse"), theta, maxiter=100, tol=1e-10, **kwargs)
    reference = _independent_inverse_gamma_mode(matrices, theta[0])

    assert state.converged
    assert_allclose(np.r_[state.beta, state.spherical], reference.x, rtol=1e-6, atol=1e-7)
    assert state.deviance == pytest.approx(reference.fun, rel=1e-12)


def test_bounded_identity_link_recovers_feasible_weighted_mode_with_offsets():
    rng = np.random.default_rng(572)
    group = np.repeat(np.arange(5), 8)
    x = np.tile(np.linspace(-1, 1, 8), 5)
    offset = 0.04 * np.cos(np.arange(len(x)))
    weights = np.tile(np.arange(15.0, 23.0), 5)
    mean = 0.45 + 0.15 * x + np.repeat(np.linspace(-0.07, 0.07, 5), 8) + offset
    y = rng.binomial(weights.astype(int), mean) / weights
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
    matrices = build_model_matrices(
        parse_formula("y ~ x + (1 | group)"), data, weights=weights, offset=offset
    )
    theta = np.array([0.1])
    family = Binomial(link=IdentityLink())
    state = _pirls_state(
        matrices, family, theta, beta_start=np.array([2.0, 0.0]), maxiter=100, tol=1e-10
    )
    design = np.column_stack([matrices.X, matrices.Z.toarray() * theta[0]])
    p = matrices.n_fixed

    def objective(parameters):
        mu = design @ parameters + offset
        return np.sum(family.deviance_resids(y, mu, weights)) + np.sum(parameters[p:] ** 2)

    def gradient(parameters):
        mu = design @ parameters + offset
        result = design.T @ (2 * weights * (mu - y) / (mu * (1 - mu)))
        result[p:] += 2 * parameters[p:]
        return result

    reference = optimize.minimize(
        objective,
        np.r_[0.5, 0.0, np.zeros(matrices.n_random)],
        jac=gradient,
        constraints=optimize.LinearConstraint(design, 1e-8 - offset, 1 - 1e-8 - offset),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert reference.success
    assert state.converged
    eta = design @ np.r_[state.beta, state.spherical] + offset
    assert np.all((eta > 0) & (eta < 1))
    assert_allclose(np.r_[state.beta, state.spherical], reference.x, rtol=1e-6, atol=1e-7)
    assert state.deviance == pytest.approx(reference.fun, rel=1e-12)


def test_feasible_initialization_respects_negative_custom_mean_domain():
    class NegativeGaussian(Gaussian):
        mean_bounds = (None, 0.0)

        def valideta(self, eta):
            return eta < -0.1

    data = pd.DataFrame({"y": [-1.2, -0.8, -1.5, -0.5]})
    matrices = build_model_matrices(parse_formula("y ~ 1"), data)
    state = _pirls_state(matrices, NegativeGaussian(), np.empty(0), beta_start=np.array([2.0]))

    assert state.converged
    assert_allclose(state.beta, [-1.0], rtol=0, atol=1e-14)


def test_feasible_initialization_works_without_intercept():
    data = pd.DataFrame({"y": [1.2, 0.2, 0.1], "x": [1.0, 2.0, 3.0]})
    offset = np.array([-2.0, 0.0, 1.0])
    matrices = build_model_matrices(parse_formula("y ~ 0 + x"), data, offset=offset)
    state = _pirls_state(
        matrices, Gamma(link="inverse"), np.empty(0), beta_start=np.array([-1.0]), maxiter=100
    )

    assert state.converged
    assert state.beta[0] > 2
    assert np.all(matrices.X @ state.beta + offset > 0)
    gradient = matrices.X.T @ (data.y.to_numpy() - 1 / (matrices.X @ state.beta + offset))
    assert_allclose(gradient, 0.0, atol=1e-7)


def test_impossible_domain_does_not_report_clipped_convergence():
    data = pd.DataFrame({"y": [1.0, 2.0], "x": [-1.0, 1.0]})
    matrices = build_model_matrices(parse_formula("y ~ 0 + x"), data)
    state = _pirls_state(matrices, Gamma(link="inverse"), np.empty(0))

    assert not state.converged
    assert np.isinf(state.deviance)


def test_log_binomial_initialization_enforces_negative_predictors():
    data = pd.DataFrame({"y": [1 / 5, 3 / 7, 7 / 12]})
    weights = np.array([5.0, 7.0, 12.0])
    matrices = build_model_matrices(parse_formula("y ~ 1"), data, weights=weights)
    state = _pirls_state(
        matrices, Binomial(link="log"), np.empty(0), beta_start=np.array([2.0]), tol=1e-10
    )

    assert state.converged
    assert_allclose(state.beta, [np.log(11 / 24)], rtol=0, atol=1e-12)
    assert np.all(state.beta < 0)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "cauchit"])
def test_unrestricted_binomial_links_keep_numerically_saturated_predictors_valid(link):
    mean = _pirls_mean(Binomial(link=link), np.array([-1000.0, 1000.0]))

    assert mean is not None
    assert np.all((mean > 0) & (mean < 1))


def test_log_link_underflow_is_stabilized_and_sqrt_negative_predictors_are_rejected():
    mean = _pirls_mean(Poisson(), np.array([-1000.0]))
    assert mean is not None and mean[0] > 0
    assert _pirls_mean(Poisson(link="sqrt"), np.array([-1.0])) is None


def test_inverse_gamma_quadrature_matches_integration_over_valid_support():
    family = Gamma(link="inverse")
    y = np.array([1.0])
    prior_weights = np.array([0.05])
    eta_fixed = np.array([0.5])
    relative_scale = 2.0

    def integrand(spherical):
        eta = eta_fixed + relative_scale * spherical
        mean = 1 / eta
        log_likelihood = -0.5 * np.sum(family.deviance_resids(y, mean, prior_weights))
        return np.exp(log_likelihood - 0.5 * spherical**2) / np.sqrt(2 * np.pi)

    # eta>0 restricts the integration to spherical>-0.25. With small precision
    # weights, clamping negative eta previously added substantial invalid mass.
    integral, error = integrate.quad(integrand, -0.25, np.inf, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    nodes, weights = hermite_rule(801)
    actual = _compute_group_quadrature(
        0.0,
        0.5,
        relative_scale,
        np.ones(1),
        y,
        eta_fixed,
        prior_weights,
        nodes,
        weights,
        family,
    )
    assert actual == pytest.approx(np.log(integral), abs=1e-3)

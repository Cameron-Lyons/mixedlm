"""Check coefficient influence against independently solved deleted systems."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer
from mixedlm.datasets import load_cbpp
from mixedlm.diagnostics import influence
from mixedlm.estimation.reml import _build_lambda
from mixedlm.families import Binomial, Gamma
from mixedlm.models.control import GlmerControl, LmerControl
from numpy.testing import assert_allclose


def _deleted_coefficients(model, weights, response):
    """Solve each augmented penalized system, holding covariance fixed."""
    fixed = model.matrices.X
    random = (
        model.matrices.Z @ _build_lambda(model.theta, model.matrices.random_structures)
    ).toarray()
    design = np.column_stack([fixed, random])
    penalty = np.diag(np.r_[np.zeros(fixed.shape[1]), np.ones(random.shape[1])])

    def solve(keep):
        reduced = np.sqrt(weights[keep, None]) * design[keep]
        target = np.sqrt(weights[keep]) * response[keep]
        return np.linalg.solve(reduced.T @ reduced + penalty, reduced.T @ target)[: fixed.shape[1]]

    full = solve(np.ones(len(response), dtype=bool))
    deleted = np.array([solve(np.arange(len(response)) != row) for row in range(len(response))])
    return full, full - deleted


def test_weighted_random_slope_dfbeta_matches_exact_penalized_case_deletion():
    rng = np.random.default_rng(2137)
    groups = np.repeat(np.arange(9), 7)
    x = rng.normal(size=len(groups)) + np.repeat(np.linspace(-2, 2, 9), 7)
    effects = rng.multivariate_normal([0, 0], [[1.4, 0.2], [0.2, 0.5]], size=9)
    offset = 0.2 * np.sin(np.arange(len(groups)))
    weights = np.exp(np.linspace(-1, 1, len(groups)))
    y = 1 + 0.7 * x + effects[groups, 0] + effects[groups, 1] * x
    y += offset + rng.normal(scale=0.3, size=len(groups)) / np.sqrt(weights)
    data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
    model = lmer(
        "y ~ x + (x | group)",
        data,
        weights=weights,
        offset=offset,
        control=LmerControl(use_rust=False, em_init=False),
    )
    assert model.converged
    assert not model.isSingular()
    full, expected = _deleted_coefficients(model, weights, y - offset)

    assert_allclose(full, model.beta, rtol=0, atol=1e-12)
    diagnostics = influence(model)
    assert_allclose(diagnostics.dfbeta, expected, rtol=1e-9, atol=1e-12)
    assert_allclose(
        diagnostics.dfbetas, expected / np.sqrt(np.diag(model.vcov())), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("family_name", ["binomial", "gamma_inverse"])
def test_glmm_dfbeta_matches_deleted_final_working_system(family_name):
    if family_name == "binomial":
        data = load_cbpp()
        data["y"] = data["incidence"] / data["size"]
        weights = data["size"].to_numpy(dtype=float)
        offset = np.zeros(len(data))
        formula = "y ~ period + (1 | herd)"
        family = Binomial()
    else:
        rng = np.random.default_rng(241)
        group = np.repeat(np.arange(8), 8)
        x = np.tile(np.linspace(-0.5, 0.5, 8), 8)
        eta = 1.5 + 0.4 * x + rng.normal(scale=0.15, size=8)[group]
        offset = 0.04 * np.sin(np.arange(len(group)))
        weights = np.linspace(20, 40, len(group))
        mu = 1 / (eta + offset)
        y = rng.gamma(weights, mu / weights)
        data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
        formula = "y ~ x + (1 | group)"
        family = Gamma(link="inverse")

    model = glmer(
        formula,
        data,
        family=family,
        weights=weights,
        offset=offset,
        nAGQ=0,
        control=GlmerControl(tolPwrss=1e-10, pirls_maxiter=100),
    )
    assert model.converged
    assert not model.isSingular()
    eta = model.linear_predictor(na_expand=False)
    mu = model.fitted(type="response", na_expand=False)
    working_weights = weights * family.weights(mu)
    working_response = eta + (model.matrices.y - mu) * family.link.deriv(mu) - offset
    full, expected = _deleted_coefficients(model, working_weights, working_response)

    assert_allclose(full, model.beta, rtol=0, atol=1e-8)
    assert_allclose(influence(model).dfbeta, expected, rtol=1e-6, atol=2e-9)


def test_zero_variance_fit_matches_weighted_least_squares_case_deletion():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(6), 5)
    x = rng.normal(size=len(groups))
    weights = np.linspace(0.5, 2.0, len(groups))
    X = np.column_stack((np.ones(len(groups)), x))
    # Noise orthogonal to the fixed and group columns makes zero variance optimal.
    design = np.column_stack((X, groups[:, None] == np.arange(6))) * np.sqrt(weights)[:, None]
    noise = rng.normal(size=len(groups)) * np.sqrt(weights)
    basis = np.linalg.qr(design)[0]
    noise -= basis @ (basis.T @ noise)
    y = 1.0 + 0.5 * x + noise / np.sqrt(weights)
    data = pd.DataFrame({"y": y, "x": x, "g": groups})
    with pytest.warns(UserWarning, match="singular"):
        model = lmer("y ~ x + (1 | g)", data, weights=weights)
    assert_allclose(model.theta, [0.0], atol=0)

    def wls(keep):
        sqrt_w = np.sqrt(weights[keep])
        return np.linalg.lstsq(sqrt_w[:, None] * X[keep], sqrt_w * y[keep], rcond=None)[0]

    beta = wls(np.ones(len(y), dtype=bool))
    deltas = beta - np.array([wls(np.arange(len(y)) != row) for row in range(len(y))])
    information = X.T @ (weights[:, None] * X)
    n, p = X.shape
    s2 = weights @ (y - X @ beta) ** 2 / (n - p)
    leverage = weights * np.einsum("ij,ij->i", X @ np.linalg.inv(information), X)
    diagnostics = influence(model)

    assert_allclose(model.beta, beta, rtol=1e-10)
    assert_allclose(diagnostics.hat_values, leverage, rtol=1e-10)
    # Both measures delete each case from the fit. Deletion diagnostics hold variance
    # components fixed, so DFFITS scales by the full-fit sigma rather than the
    # leave-one-out s_(i) that lm's dffits() uses.
    assert_allclose(
        diagnostics.cooks_distance,
        np.einsum("ij,jk,ik->i", deltas, information, deltas) / (p * s2),
        rtol=1e-10,
    )
    assert_allclose(
        diagnostics.dffits,
        np.sqrt(weights) * np.einsum("ij,ij->i", X, deltas) / np.sqrt(s2 * leverage),
        rtol=1e-10,
    )

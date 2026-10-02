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

"""Exercise full scientific data without constructing observation-square matrices."""

import numpy as np
import pytest
from mixedlm import lFormula, lmer, load_insteval, mkLmerDevfun
from numpy.testing import assert_allclose

FORMULA = "y ~ service + (1 | s) + (1 | d)"


def test_complete_insteval_fit_satisfies_mixed_model_normal_equations():
    data = load_insteval()
    model = lmer(FORMULA, data)
    assert model.converged
    assert model.nobs() == 73421
    assert model.matrices.n_random == 4100
    assert np.all(model.theta > 0)

    X, Z, y = model.matrices.X, model.matrices.Z, model.matrices.y
    residuals = y - X @ model.beta - Z @ model.u
    # Independent first-order equations for fixed effects and conditional modes.
    # Scalar random-intercept priors make the precision explicit for every level.
    precision = np.concatenate(
        [
            np.full(structure.n_levels, 1 / theta**2)
            for theta, structure in zip(model.theta, model.matrices.random_structures, strict=True)
        ]
    )
    assert_allclose(X.T @ residuals, 0.0, atol=1e-7)
    assert_allclose(Z.T @ residuals, precision * model.u, rtol=1e-7, atol=1e-7)
    penalized_sum = residuals @ residuals + model.u @ (precision * model.u)
    assert_allclose(model.sigma**2, penalized_sum / (len(y) - X.shape[1]), rtol=1e-10)
    assert_allclose(model.fitted() + model.residuals(), y, atol=1e-12)

    # The normal equations hold at any theta; the variance estimates must also be
    # a minimum of the REML criterion, located here by central differences.
    devfun = mkLmerDevfun(lFormula(FORMULA, data))
    center = devfun(model.theta)
    assert center == pytest.approx(model.deviance, rel=1e-12)
    step = 1e-4
    for shift in np.eye(len(model.theta)) * step:
        upper, lower = devfun(model.theta + shift), devfun(model.theta - shift)
        gradient = (upper - lower) / (2 * step)
        curvature = (upper - 2 * center + lower) / step**2
        assert curvature > 0
        assert abs(gradient / curvature) < 1e-6

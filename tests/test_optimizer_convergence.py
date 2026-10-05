"""Optimizers must reach the optimum or report that they did not converge."""

import numpy as np
import pytest
from mixedlm import (
    checkConv,
    convergence_ok,
    families,
    glmer,
    glmerControl,
    lmer,
    lmerControl,
    load_cbpp,
    load_grouseticks,
    load_sleepstudy,
    set_cov_type,
)
from mixedlm.estimation.optimizers import NLOPT_OPTIMIZER_NAMES, available_optimizers, has_nlopt
from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose
from scipy import linalg, optimize

from tests.test_statistical_golden import observation_space_reference

INTERCEPTS = "Reaction ~ Days + (1 | Subject)"
SLOPES = "Reaction ~ Days + (Days | Subject)"
GROUSE = "TICKS ~ YEAR + HEIGHT + (1 | BROOD) + (1 | LOCATION)"

pytestmark = [
    pytest.mark.filterwarnings("ignore:Model failed to converge:UserWarning"),
    pytest.mark.filterwarnings("ignore:The 'bobyqa' optimizer name:DeprecationWarning"),
]


@pytest.fixture(autouse=True)
def seeded_nlopt():
    # PRAXIS draws random search directions from NLopt's global generator.
    if has_nlopt():
        import nlopt

        nlopt.srand(20240611)


@pytest.fixture(scope="module")
def sleepstudy():
    return load_sleepstudy()


@pytest.fixture(scope="module")
def slopes_reference(sleepstudy):
    # Optimum of the REML criterion from dense observation-space GLS.
    return observation_space_reference(sleepstudy, slopes=True)


@pytest.fixture(scope="module")
def equal_variance_reference(sleepstudy):
    """REML optimum of slopes with equal variances, by dense observation-space GLS.

    With two random coefficients, AR(1) and compound symmetry both reduce to
    this one-correlation model.
    """
    y = sleepstudy["Reaction"].to_numpy(dtype=float)
    X = np.column_stack((np.ones(len(y)), sleepstudy["Days"].to_numpy(dtype=float)))
    subject = sleepstudy["Subject"].to_numpy()
    same = subject[:, None] == subject[None, :]
    df = len(y) - X.shape[1]

    def deviance(params):
        scale, rho = params
        G = scale**2 * np.array([[1.0, rho], [rho, 1.0]])
        factor = linalg.cho_factor(np.eye(len(y)) + same * (X @ G @ X.T), lower=True)
        information = X.T @ linalg.cho_solve(factor, X)
        beta = np.linalg.solve(information, X.T @ linalg.cho_solve(factor, y))
        residuals = y - X @ beta
        sigma_squared = residuals @ linalg.cho_solve(factor, residuals) / df
        return (
            2 * np.log(np.diag(factor[0])).sum()
            + np.linalg.slogdet(information)[1]
            + df * (1 + np.log(2 * np.pi * sigma_squared))
        )

    optimum = optimize.minimize(
        deviance, [0.3, 0.5], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-10}
    )
    assert optimum.success, optimum.message
    return optimum.fun


def intercept_theta(data):
    """Balanced ANOVA estimate, which equals REML for a positive variance."""
    groups = data.groupby("Subject")
    centered = data["Reaction"] - groups["Reaction"].transform("mean")
    days = data["Days"] - groups["Days"].transform("mean")
    slope = (days @ centered) / (days @ days)
    within = np.sum((centered - slope * days) ** 2) / (len(data) - groups.ngroups - 1)
    means = groups["Reaction"].mean()
    size = len(data) // groups.ngroups
    between = size * np.sum((means - means.mean()) ** 2) / (groups.ngroups - 1)
    return np.sqrt((between - within) / size / within)


@pytest.mark.parametrize("name", available_optimizers())
def test_every_optimizer_reaches_the_optimum_or_reports_failure(name, sleepstudy, slopes_reference):
    slopes = lmer(SLOPES, sleepstudy, method=name)
    if slopes.converged:
        assert np.all(np.isfinite(slopes.theta))
        assert slopes.deviance == pytest.approx(slopes_reference["deviance"], abs=1e-3)

    # Only the square of a scalar theta is identified (BFGS ignores bounds), and
    # a theta error of 5e-3 raises this deviance by less than 1e-3.
    intercepts = lmer(INTERCEPTS, sleepstudy, method=name)
    if intercepts.converged:
        assert np.abs(intercepts.theta) == pytest.approx([intercept_theta(sleepstudy)], abs=5e-3)


@pytest.mark.parametrize("name", available_optimizers())
def test_every_optimizer_fits_a_structured_covariance_or_reports_failure(
    name, sleepstudy, equal_variance_reference
):
    # A relative function tolerance stopped nloptwrap_BOBYQA 0.11 above the optimum.
    # From a covariance scale in response units (about 24 here) NLopt's Nelder-Mead
    # and PRAXIS collapsed onto rho = 1 and reported convergence.
    fit = lmer(set_cov_type(SLOPES, "ar1"), sleepstudy, method=name)
    if fit.converged:
        assert np.all(np.isfinite(fit.theta))
        assert fit.deviance == pytest.approx(equal_variance_reference, abs=1e-3)


@pytest.mark.parametrize("structure", ["correlated", "independent", "cs", "ar1"])
def test_start_values_do_not_depend_on_response_units(structure, sleepstudy):
    # Theta is relative to the residual scale, so its start must be too.
    formula = SLOPES.replace("|", "||") if structure == "independent" else SLOPES
    if structure in ("cs", "ar1"):
        formula = set_cov_type(formula, structure)
    starts = [
        LMMOptimizer(
            build_model_matrices(
                parse_formula(formula) if isinstance(formula, str) else formula,
                sleepstudy.assign(Reaction=units * sleepstudy["Reaction"]),
            )
        ).get_start_theta()
        for units in (1e-4, 1.0, 1e4)
    ]
    assert_allclose(starts[0], starts[1], rtol=1e-10)
    assert_allclose(starts[2], starts[1], rtol=1e-10)


@pytest.fixture(scope="module")
def grouseticks_optimum():
    data = load_grouseticks()
    fit = glmer(GROUSE, data, family=families.Poisson())
    # A gradient-based optimizer independently confirms the optimum.
    control = glmerControl(optimizer="L-BFGS-B")
    gradient = glmer(GROUSE, data, family=families.Poisson(), control=control)
    assert gradient.deviance == pytest.approx(fit.deviance, abs=1e-6)
    return data, fit.deviance


@pytest.mark.skipif(not has_nlopt(), reason="nlopt missing")
@pytest.mark.parametrize("name", sorted(NLOPT_OPTIMIZER_NAMES))
def test_nlopt_generalized_fit_reaches_the_optimum_or_reports_failure(name, grouseticks_optimum):
    # A relative function tolerance stopped SBPLX 0.1 above this optimum.
    data, deviance = grouseticks_optimum
    fit = glmer(GROUSE, data, family=families.Poisson(), control=glmerControl(optimizer=name))
    if fit.converged:
        assert fit.deviance == pytest.approx(deviance, abs=1e-3)


def test_check_conv_reports_the_fitted_optimizer(sleepstudy, slopes_reference):
    default = lmer(SLOPES, sleepstudy)
    info = checkConv(default)
    assert info.converged and info.messages == []
    # The default fits with exact native gradients and records that method.
    assert info.optimizer == default.optimizer == "L-BFGS-B"
    assert info.iterations == default.n_iter > 0
    assert info.gradient_norm == default.gradient_norm < 1e-6 * len(sleepstudy)
    assert default.deviance == pytest.approx(slopes_reference["deviance"], abs=1e-8)
    assert convergence_ok(default)

    derivative_free = lmer(SLOPES, sleepstudy, control=lmerControl(optimizer="COBYQA"))
    info = checkConv(derivative_free)
    assert info.optimizer == "COBYQA" and info.messages == []
    # Derivative-free optimizers record no final gradient.
    assert info.gradient_norm is None

    gradient = lmer(SLOPES, sleepstudy, control=lmerControl(optimizer="L-BFGS-B"))
    assert gradient.deviance == pytest.approx(slopes_reference["deviance"], abs=1e-6)
    info = checkConv(gradient)
    assert info.optimizer == "L-BFGS-B" and info.messages == []
    assert info.gradient_norm == gradient.gradient_norm < 1e-3 * len(sleepstudy)


@pytest.mark.parametrize(
    "name",
    ["L-BFGS-B", "COBYQA", "Nelder-Mead"]
    + [
        pytest.param(name, marks=pytest.mark.skipif(not has_nlopt(), reason="nlopt missing"))
        for name in sorted(NLOPT_OPTIMIZER_NAMES)
    ],
)
def test_iteration_limit_is_reported_as_non_convergence(name, sleepstudy):
    fit = lmer(SLOPES, sleepstudy, method=name, control=lmerControl(maxiter=3))
    assert not fit.converged
    info = checkConv(fit)
    assert not info.converged
    assert info.optimizer == name
    assert info.messages[0] == f"Optimizer did not report convergence: {fit.message}"
    assert not convergence_ok(fit)


def test_check_conv_flags_a_premature_gradient_stop(sleepstudy, slopes_reference):
    # A loose relative reduction tolerance stops L-BFGS-B well before the optimum.
    loose = lmer(SLOPES, sleepstudy, control=lmerControl(optimizer="L-BFGS-B", ftol=1e-3))
    assert loose.converged
    assert loose.deviance > slopes_reference["deviance"] + 0.1
    info = checkConv(loose)
    assert any(message.startswith("Gradient norm per observation") for message in info.messages)


def test_generalized_fit_records_optimizer_metadata():
    data = load_cbpp()
    formula = "incidence / size ~ period + (1 | herd)"
    fit = glmer(formula, data, family=families.Binomial())
    info = checkConv(fit)
    assert info.optimizer == fit.optimizer == "COBYQA"
    assert info.messages == [] and convergence_ok(fit)

    limited = glmer(formula, data, family=families.Binomial(), control=glmerControl(maxiter=3))
    assert not limited.converged and limited.message
    assert checkConv(limited).messages[0].endswith(limited.message)
    assert not convergence_ok(limited)

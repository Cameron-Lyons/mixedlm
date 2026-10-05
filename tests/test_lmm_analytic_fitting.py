"""Analytic fitting shares evaluations and preserves solver and response isolation."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from mixedlm import lFormula, lmer, lmerControl, mkLmerDevfun, optimizeLmer
from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import LmerControl
from mixedlm.models.modular import LmerDevfun
from numpy.testing import assert_allclose, assert_array_equal

from tests._datasets import SLEEPSTUDY
from tests._lmm_oracles import linear_data, matrices_fixture, parameters


def sleep_matrices():
    n = len(SLEEPSTUDY)
    return build_model_matrices(
        parse_formula("Reaction ~ Days + (Days | Subject)"),
        SLEEPSTUDY,
        weights=np.geomspace(0.7, 1.8, n),
        offset=np.cos(np.arange(n)) * 0.3,
    )


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("start", [0.0, 0.8, 2.0])
def test_analytic_fit_recovers_balanced_variance_after_zero_gradient_start(reml, start):
    result = lmer(
        "y ~ 1 + (1 | g)",
        linear_data(),
        REML=reml,
        start=np.array([start]),
        control=lmerControl(optimizer="L-BFGS-B", use_analytic_gradient=True),
    )
    residual_variance = 4 / 3
    between_variance = 0.7**2 * (6 / 5 if reml else 1)
    expected = np.sqrt((between_variance - residual_variance / 4) / residual_variance)
    assert result.converged
    assert_allclose(result.theta, [expected], atol=2e-5)
    assert_allclose(result.sigma**2, residual_variance, atol=2e-5)


@pytest.mark.parametrize("start", [0.0, 0.8])
def test_analytic_fit_retains_genuine_zero_variance(start):
    with pytest.warns(UserWarning, match="singular"):
        result = lmer(
            "y ~ 1 + (1 | g)",
            linear_data(0),
            REML=False,
            start=np.array([start]),
            control=lmerControl(optimizer="L-BFGS-B", use_analytic_gradient=True),
        )
    assert result.converged
    assert_allclose(result.theta, [0], atol=1e-7)
    assert_allclose(result.sigma, 1, atol=1e-7)


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("method", ["L-BFGS-B", "BFGS", "TNC", "SLSQP", "trust-constr"])
def test_gradient_solvers_match_independent_python_fit(method, reml):
    matrices = sleep_matrices()
    start = parameters(matrices)
    reference = LMMOptimizer(matrices, REML=reml, use_rust=False).optimize(
        start=start, method="COBYQA", options={"final_tr_radius": 1e-9}
    )
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    options = {"gtol": 1e-8} if method not in {"SLSQP"} else {"ftol": 1e-10}
    if method == "BFGS":
        # The line search can reach objective roundoff near a 1e-6 gradient
        # with this uncentered response; retain the fit-accuracy checks below.
        options["gtol"] = 1e-5
    if method in {"L-BFGS-B", "TNC"}:
        options["ftol"] = 1e-12
    actual = optimizer.optimize(
        start=start, method=method, options=options, use_analytic_gradient=True
    )
    assert actual.converged, actual.message
    assert reference.converged
    assert_allclose(actual.deviance, reference.deviance, rtol=0, atol=2e-7)
    assert_allclose(actual.beta, reference.beta, rtol=1e-6, atol=1e-5)
    assert_allclose(actual.sigma, reference.sigma, rtol=1e-6)
    assert_allclose(actual.u, reference.u, rtol=2e-5, atol=2e-4)
    expected = optimizer._rust_cache.response.deviance_with_gradient(actual.theta, reml)
    assert actual.deviance == expected[0]
    assert_allclose(actual.gradient_norm, np.linalg.norm(expected[1]), atol=1e-12)
    if method == "BFGS":
        assert np.linalg.norm(expected[1], ord=np.inf) <= options["gtol"]


@pytest.mark.parametrize("reml", [False, True])
def test_value_and_gradient_cache_owns_parameters_and_returns_independent_derivatives(reml):
    optimizer = LMMOptimizer(matrices_fixture("correlated"), REML=reml, use_rust=True)
    response = optimizer._rust_cache.response
    calls = []

    class CountingResponse:
        def deviance_with_gradient(self, theta, mode):
            calls.append((theta.copy(), mode))
            return response.deviance_with_gradient(theta, mode)

    optimizer._rust_cache.response = CountingResponse()
    objective, gradient = optimizer._optimization_functions("L-BFGS-B", True)
    theta = parameters(optimizer.matrices)
    value, derivative = response.deviance_with_gradient(theta, reml)
    assert objective(theta) == value
    assert_array_equal(gradient(theta.copy()), derivative)
    assert len(calls) == 1
    changed = gradient(theta)
    changed[:] = np.nan
    assert_array_equal(gradient(theta), derivative)
    theta *= 1.3
    expected = response.deviance_with_gradient(theta, reml)
    assert_array_equal(gradient(theta), expected[1])
    assert objective(theta.copy()) == expected[0]
    assert len(calls) == 2
    with pytest.raises(ValueError, match="theta"):
        gradient(np.array([np.nan]))
    assert objective(theta) == expected[0]
    assert len(calls) == 3
    assert all(mode == reml for _, mode in calls)


@pytest.mark.parametrize(
    "kind,native,method,enabled",
    [
        ("correlated", False, "L-BFGS-B", True),
        ("cs", True, "L-BFGS-B", True),
        ("ar1", True, "L-BFGS-B", True),
        ("fixed", True, "L-BFGS-B", True),
        ("correlated", True, "COBYQA", True),
        ("correlated", True, "Powell", True),
        ("correlated", True, "L-BFGS-B", False),
    ],
)
def test_unsupported_or_disabled_gradient_keeps_scalar_evaluation(kind, native, method, enabled):
    optimizer = LMMOptimizer(matrices_fixture(kind), use_rust=native)
    objective, gradient = optimizer._optimization_functions(method, enabled)
    theta = parameters(optimizer.matrices)
    assert gradient is None
    assert objective(theta) == optimizer.objective(theta)


@pytest.mark.parametrize("reml", [False, True])
def test_response_refits_and_concurrent_fits_keep_caches_local(reml):
    matrices = sleep_matrices()
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=True)
    refit = optimizer.with_response(matrices.y[::-1] + matrices.offset)
    start = parameters(matrices)
    cases = [(fit, start * scale) for fit in [optimizer, refit] for scale in [0.7, 1.3]]

    def fit(case):
        prepared, theta = case
        return prepared.optimize(
            start=theta, options={"ftol": 1e-12, "gtol": 1e-8}, use_analytic_gradient=True
        )

    expected = [fit(case) for case in cases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(fit, cases * 2))
    for index, result in enumerate(actual):
        reference = expected[index % len(cases)]
        assert result.converged and reference.converged
        for field in vars(reference):
            assert_array_equal(getattr(result, field), getattr(reference, field))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("modular", [False, True])
def test_public_control_selects_analytic_or_numerical_fitting(monkeypatch, enabled, modular):
    from mixedlm.estimation import reml

    calls = []
    original = reml._LMMGradientObjective.__init__

    def capture(self, evaluate):
        calls.append(evaluate)
        original(self, evaluate)

    monkeypatch.setattr(reml._LMMGradientObjective, "__init__", capture)
    control = lmerControl(optimizer="L-BFGS-B", use_analytic_gradient=enabled)
    if modular:
        devfun = mkLmerDevfun(
            lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY), control=control
        )
        result = optimizeLmer(devfun)
    else:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=control)
    assert result.converged
    assert len(calls) == int(enabled)


def test_modular_explicit_gradient_setting_overrides_creation_control(monkeypatch):
    from mixedlm.estimation import reml

    def unexpected(*args):
        pytest.fail("Analytic gradient was explicitly disabled")

    monkeypatch.setattr(reml._LMMGradientObjective, "__init__", unexpected)
    devfun = mkLmerDevfun(
        lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY),
        control=LmerControl(use_analytic_gradient=True),
    )
    assert optimizeLmer(devfun, use_analytic_gradient=False).converged


@pytest.mark.parametrize("invalid", [None, "true", 0, 1, []])
def test_invalid_gradient_controls_are_rejected(invalid):
    for factory in [LmerControl, lmerControl]:
        with pytest.raises(ValueError, match="use_analytic_gradient must be a boolean"):
            factory(use_analytic_gradient=invalid)
    optimizer = LMMOptimizer(matrices_fixture("intercept"), use_rust=True)
    with pytest.raises(ValueError, match="use_analytic_gradient must be a boolean"):
        optimizer.optimize(use_analytic_gradient=invalid)


def test_analytic_fitting_reduces_objective_calls():
    matrices = sleep_matrices()
    optimizer = LMMOptimizer(matrices, use_rust=True)
    options = {"ftol": 1e-12, "gtol": 1e-8}
    numerical = optimizer.optimize(options=options, use_analytic_gradient=False)
    analytic = optimizer.optimize(options=options, use_analytic_gradient=True)
    assert numerical.converged and analytic.converged
    assert analytic.function_evals < numerical.function_evals
    assert_allclose(analytic.deviance, numerical.deviance, rtol=0, atol=1e-8)


def test_default_fit_retains_numerical_derivatives(monkeypatch):
    from mixedlm.estimation import reml

    def unexpected(*args):
        pytest.fail("Analytic fitting requires an explicit request")

    monkeypatch.setattr(reml._LMMGradientObjective, "__init__", unexpected)
    optimizer = LMMOptimizer(sleep_matrices(), use_rust=True)
    assert optimizer.optimize().converged
    assert not LmerControl().use_analytic_gradient
    assert not lmerControl().use_analytic_gradient


@pytest.mark.parametrize("analytic", [False, True])
def test_custom_modular_deviance_preserves_its_added_term(analytic):
    calls = []

    class PenalizedDeviance(LmerDevfun):
        def __call__(self, theta):
            calls.append(theta.copy())
            return super().__call__(theta) + float(np.dot(theta, theta))

    original = mkLmerDevfun(lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY))
    devfun = PenalizedDeviance(original.parsed, original.optimizer)
    result = optimizeLmer(devfun, use_analytic_gradient=analytic)
    assert result.converged and calls
    assert result.deviance == devfun(result.theta)

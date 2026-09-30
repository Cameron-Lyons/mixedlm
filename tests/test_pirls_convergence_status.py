from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import glFormula, glmer, glmerControl, mkGlmerDevfun, mkGlmerMod, optimizeGlmer
from mixedlm.estimation import laplace
from mixedlm.families import Binomial, Gaussian, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.modular import OptimizeResult


def model_data(case, random=True):
    responses = {
        "zero_poisson": ([0, 0, 0], Poisson()),
        "zero_binomial": ([0, 0, 0], Binomial()),
        "one_binomial": ([1, 1, 1], Binomial()),
        "poisson": ([1, 2, 3], Poisson()),
        "binomial": ([0, 1, 0], Binomial()),
        "gaussian": ([-0.2, 0.3, 1.0], Gaussian()),
    }
    values, family = responses[case]
    data = pd.DataFrame({"y": np.tile(values, 5), "g": np.repeat(np.arange(5), 3)})
    formula = "y ~ 1 + (1 | g)" if random else "y ~ 1"
    matrices = build_model_matrices(parse_formula(formula), data)
    theta = np.array([0.5]) if random else np.array([])
    return data, formula, matrices, family, theta


@pytest.mark.parametrize(
    "case", ["zero_poisson", "zero_binomial", "one_binomial", "poisson", "binomial", "gaussian"]
)
@pytest.mark.parametrize("order", [1, 7])
@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("random", [False, True])
def test_status_matches_inner_solution_and_preserves_legacy_results(case, order, backend, random):
    _, _, matrices, family, theta = model_data(case, random)
    if backend == "native":
        pytest.importorskip("mixedlm._rust")
    with patch.object(laplace, "_HAS_RUST", backend == "native"):
        actual = laplace.glmm_deviance_with_status(theta, matrices, family, nAGQ=order)
        legacy = laplace.adaptive_gh_deviance_fast(theta, matrices, family, nAGQ=order)
    assert len(actual) == 4
    assert len(legacy) == 3
    assert isinstance(actual[3], bool)
    assert actual[3] == (case in {"poisson", "binomial", "gaussian"})
    for value, expected in zip(actual[:3], legacy, strict=True):
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("case", ["zero_poisson", "zero_binomial", "one_binomial"])
@pytest.mark.parametrize("order", [1, 7])
@pytest.mark.parametrize("backend", ["python", "native"])
def test_public_fit_warns_about_failed_inner_solver(case, order, backend):
    data, formula, _, family, _ = model_data(case)
    if backend == "native":
        pytest.importorskip("mixedlm._rust")
    with (
        patch.object(laplace, "_HAS_RUST", backend == "native"),
        pytest.warns(UserWarning, match="inner PIRLS solver did not converge"),
    ):
        result = glmer(
            formula, data, family=family, nAGQ=order, control=glmerControl(check_singular=False)
        )
    assert not result.converged
    assert result.pirls_converged is False
    assert np.isfinite(result.deviance)
    summary = result.summary()
    assert "inner PIRLS solver did not converge" in summary
    assert "optimizer did not converge" not in summary


def test_convergence_warning_can_be_disabled_without_changing_status():
    data, formula, _, family, _ = model_data("zero_poisson")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = glmer(
            formula,
            data,
            family=family,
            control=glmerControl(check_conv=False, check_singular=False),
        )
    assert not caught
    assert not result.converged
    assert not result.pirls_converged


@pytest.mark.parametrize("order", [1, 7])
def test_refit_preserves_inner_status(order):
    data, formula, _, family, _ = model_data("poisson")
    original = glmer(
        formula, data, family=family, nAGQ=order, control=glmerControl(check_singular=False)
    )
    assert original.pirls_converged
    failed = original.refit(np.zeros(len(data)))
    assert not failed.converged
    assert not failed.pirls_converged
    recovered = failed.refit(data["y"].to_numpy())
    assert recovered.converged
    assert recovered.pirls_converged


@pytest.mark.parametrize("outer_success", [False, True])
@pytest.mark.parametrize("inner_success", [False, True])
def test_optimizer_combines_status_without_repeating_pirls(outer_success, inner_success):
    _, _, matrices, family, theta = model_data("poisson")
    state = replace(laplace._pirls_state(matrices, family, theta), converged=inner_success)
    outer = SimpleNamespace(x=theta, fun=123.0, success=outer_success, nit=2)
    with (
        patch.object(laplace, "_HAS_RUST", False),
        patch.object(laplace, "run_optimizer", return_value=outer),
        patch.object(laplace, "_pirls_state", return_value=state) as solve,
    ):
        result = laplace.GLMMOptimizer(matrices, family).optimize()
    assert solve.call_count == 1
    assert result.pirls_converged is inner_success
    assert result.converged == (outer_success and inner_success)


@pytest.mark.parametrize("case", ["zero_poisson", "zero_binomial", "one_binomial", "poisson"])
def test_modular_optimization_and_assembly_report_inner_status(case):
    data, formula, _, family, _ = model_data(case)
    devfun = mkGlmerDevfun(glFormula(formula, data, family=family))
    opt = optimizeGlmer(devfun)
    fitted = mkGlmerMod(devfun, opt)
    expected = case == "poisson"
    assert opt.pirls_converged is expected
    assert fitted.pirls_converged is expected
    if not expected:
        assert not opt.converged
        assert not fitted.converged
        assert "inner PIRLS" in opt.message


def test_modular_constructor_rechecks_manual_optimization_result():
    data, formula, _, family, theta = model_data("zero_poisson")
    devfun = mkGlmerDevfun(glFormula(formula, data, family=family))
    opt = OptimizeResult(theta=theta, deviance=123.0, converged=True, n_iter=1, message="success")
    result = mkGlmerMod(devfun, opt)
    assert not result.converged
    assert not result.pirls_converged
    assert result.deviance != opt.deviance

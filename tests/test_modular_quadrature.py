from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import glFormula, glmer, mkGlmerDevfun, mkGlmerMod, optimizeGlmer
from mixedlm.estimation import laplace
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from mixedlm.families import Binomial, Poisson
from mixedlm.models.modular import OptimizeResult


@pytest.fixture
def data():
    rng = np.random.default_rng(72)
    group = np.repeat(np.arange(8), 12)
    x = rng.uniform(0.5, 1.5, len(group))
    offset = np.linspace(-0.1, 0.2, len(group))
    eta = -0.2 + 0.3 * x + rng.normal(0, 0.4, 8)[group] + offset
    return pd.DataFrame(
        {
            "y": rng.poisson(np.exp(eta)),
            "binary": rng.binomial(1, 1 / (1 + np.exp(-eta))),
            "successes": rng.binomial(7, 1 / (1 + np.exp(-eta))),
            "trials": 7,
            "x": x,
            "g": group,
            "h": np.arange(len(group)) % 6,
            "offset": offset,
            "weight": np.linspace(0.8, 1.2, len(group)),
        }
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["poisson", "binomial", "grouped_binomial"])
@pytest.mark.parametrize("n_agq", [1, 5])
def test_modular_quadrature_matches_direct_fit_and_likelihood(data, native, kind, n_agq):
    if native:
        pytest.importorskip("mixedlm._rust")
    family = Poisson() if kind == "poisson" else Binomial()
    response = {"poisson": "y", "binomial": "binary", "grouped_binomial": "successes / trials"}[
        kind
    ]
    formula = f"{response} ~ x + (1 | g)"
    kwargs = dict(family=family, weights=data.weight.to_numpy(), offset=data.offset.to_numpy())
    with patch.object(laplace, "_HAS_RUST", native):
        parsed = glFormula(formula, data, **kwargs)
        devfun = mkGlmerDevfun(parsed, nAGQ=n_agq)
        opt = optimizeGlmer(devfun)
        result = mkGlmerMod(devfun, opt)
        direct = glmer(formula, data, nAGQ=n_agq, method="L-BFGS-B", **kwargs)
        expected = JointGLMMObjective(
            parsed.matrices, family, n_agq, pirls_tol=result.pirls_tol
        ).evaluate(np.r_[result.theta, result.beta])
    assert devfun.optimizer.nAGQ == opt.nAGQ == result.nAGQ == n_agq
    assert opt.converged == result.converged == direct.converged
    np.testing.assert_allclose(result.theta, direct.theta, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(result.beta, direct.beta, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(result.u, direct.u, rtol=1e-7, atol=1e-7)
    assert result.deviance == pytest.approx(direct.deviance, abs=1e-8)
    assert result.deviance == expected[0]
    np.testing.assert_array_equal(result.beta, expected[1])
    np.testing.assert_array_equal(result.u, expected[2])


@pytest.mark.parametrize("n_agq", [np.int32(3), np.int64(5)])
def test_numpy_integer_orders_and_custom_optimization_results(data, n_agq):
    devfun = mkGlmerDevfun(glFormula("y ~ x + (1 | g)", data, family=Poisson()), nAGQ=n_agq)
    theta = np.array([0.7])
    custom = OptimizeResult(theta, devfun(theta), False, 2, "custom limit")
    result = mkGlmerMod(devfun, custom)
    assert result.nAGQ == n_agq
    assert not result.converged
    assert result.deviance == custom.deviance
    explicit = mkGlmerMod(devfun, custom, nAGQ=int(n_agq))
    assert explicit.deviance == result.deviance


def test_result_retains_recorded_order_if_deviance_function_setting_changes(data):
    devfun = mkGlmerDevfun(glFormula("y ~ x + (1 | g)", data, family=Poisson()), nAGQ=5)
    opt = optimizeGlmer(devfun)
    devfun.optimizer.nAGQ = 1
    result = mkGlmerMod(devfun, opt)
    assert opt.nAGQ == result.nAGQ == 5
    assert result.deviance == pytest.approx(opt.deviance)


@pytest.mark.parametrize("recorded", [None, 1, 5])
def test_result_cannot_relabel_optimization_with_another_order(data, recorded):
    devfun = mkGlmerDevfun(glFormula("y ~ x + (1 | g)", data, family=Poisson()), nAGQ=5)
    opt = OptimizeResult(np.ones(1), 12.0, True, 0, "", nAGQ=recorded)
    expected = 5 if recorded is None else recorded
    with (
        patch.object(
            laplace, "adaptive_gh_deviance_fast", side_effect=AssertionError("must not evaluate")
        ),
        pytest.raises(ValueError, match="must match the setting used for optimization"),
    ):
        mkGlmerMod(devfun, opt, nAGQ=1 if expected == 5 else 5)


INVALID_ORDERS = [-1, True, False, np.bool_(True), 1.0, 2.5, np.nan, np.inf, "5", None]


@pytest.mark.parametrize("n_agq", INVALID_ORDERS)
@pytest.mark.parametrize("entry", ["optimizer", "python", "fast", "modular", "fit"])
def test_invalid_orders_fail_before_numerical_evaluation(data, n_agq, entry):
    parsed = glFormula("y ~ x + (1 | g)", data, family=Poisson())
    with (
        patch.object(laplace, "_pirls_state", side_effect=AssertionError("must not evaluate")),
        patch.object(
            laplace, "_rust_laplace_deviance", side_effect=AssertionError("must not evaluate")
        ),
        patch.object(
            laplace, "_rust_adaptive_gh_deviance", side_effect=AssertionError("must not evaluate")
        ),
        pytest.raises(ValueError, match="nAGQ must be a nonnegative integer"),
    ):
        if entry == "optimizer":
            laplace.GLMMOptimizer(parsed.matrices, parsed.family, nAGQ=n_agq)
        elif entry == "modular":
            mkGlmerDevfun(parsed, nAGQ=n_agq)
        elif entry == "fit":
            glmer("y ~ x + (1 | g)", data, family=Poisson(), nAGQ=n_agq)
        else:
            method = (
                laplace.adaptive_gh_deviance
                if entry == "python"
                else laplace.adaptive_gh_deviance_fast
            )
            method(np.ones(1), parsed.matrices, parsed.family, nAGQ=n_agq)


@pytest.mark.parametrize("n_agq", INVALID_ORDERS[:-1])
def test_invalid_result_order_is_rejected(data, n_agq):
    devfun = mkGlmerDevfun(glFormula("y ~ x + (1 | g)", data, family=Poisson()))
    opt = OptimizeResult(np.ones(1), 12.0, True, 0, "")
    with pytest.raises(ValueError, match="nAGQ must be a nonnegative integer"):
        mkGlmerMod(devfun, opt, nAGQ=n_agq)


@pytest.mark.parametrize("formula", ["y ~ x + (1 + x | g)", "y ~ x + (1 | g) + (1 | h)"])
@pytest.mark.parametrize("entry", ["optimizer", "python", "fast", "modular", "fit"])
def test_unsupported_quadrature_structure_does_not_silently_use_laplace(data, formula, entry):
    parsed = glFormula(formula, data, family=Poisson())
    with pytest.raises(ValueError, match="one random-effect term with one coefficient per group"):
        if entry == "optimizer":
            laplace.GLMMOptimizer(parsed.matrices, parsed.family, nAGQ=5)
        elif entry == "modular":
            mkGlmerDevfun(parsed, nAGQ=5)
        elif entry == "fit":
            glmer(formula, data, family=Poisson(), nAGQ=5)
        else:
            method = (
                laplace.adaptive_gh_deviance
                if entry == "python"
                else laplace.adaptive_gh_deviance_fast
            )
            method(np.ones(parsed.n_theta), parsed.matrices, parsed.family, nAGQ=5)
    assert np.isfinite(mkGlmerDevfun(parsed)(np.ones(parsed.n_theta)))


@pytest.mark.parametrize("formula", ["y ~ x", "y ~ x + (0 + x | g)"])
@pytest.mark.parametrize("native", [False, True])
def test_valid_zero_dimension_and_scalar_slope_quadrature(data, formula, native):
    if native:
        pytest.importorskip("mixedlm._rust")
    with patch.object(laplace, "_HAS_RUST", native):
        parsed = glFormula(formula, data, family=Poisson())
        devfun = mkGlmerDevfun(parsed, nAGQ=5)
        theta = np.ones(parsed.n_theta)
        value = devfun(theta)
        result = mkGlmerMod(devfun, OptimizeResult(theta, value, True, 0, ""))
    assert np.isfinite(value)
    assert result.nAGQ == 5
    assert result.deviance == value


def native_args(parsed):
    m = parsed.matrices
    z = m.Z.tocsc()
    return (
        m.y,
        m.X,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        m.weights,
        m.offset,
        np.ones(parsed.n_theta),
        [s.n_levels for s in m.random_structures],
        [s.n_terms for s in m.random_structures],
        [s.correlated for s in m.random_structures],
        "poisson",
        "log",
    )


@pytest.mark.parametrize(
    "formula,order",
    [
        ("y ~ x + (1 | g)", 0),
        ("y ~ x + (1 + x | g)", 5),
        ("y ~ x + (1 | g) + (1 | h)", 5),
    ],
)
def test_native_binding_rejects_unavailable_quadrature(data, formula, order):
    native = pytest.importorskip("mixedlm._rust")
    parsed = glFormula(formula, data, family=Poisson())
    with pytest.raises(ValueError, match="n_agq"):
        native.adaptive_gh_deviance(*native_args(parsed), order)


def test_existing_positional_arguments_keep_their_meaning(data):
    from mixedlm import glmerControl

    parsed = glFormula("y ~ x + (1 | g)", data, family=Poisson())
    devfun = mkGlmerDevfun(parsed, 0, glmerControl())
    assert devfun.optimizer.nAGQ == 1
    opt = OptimizeResult(np.ones(1), devfun(np.ones(1)), True, 0, "custom")
    result = mkGlmerMod(devfun, opt, 1)
    assert result.nAGQ == 1


@pytest.mark.parametrize("n_agq", INVALID_ORDERS)
@pytest.mark.parametrize("entry", ["direct", "modular"])
def test_changed_quadrature_setting_is_revalidated_before_optimization(data, n_agq, entry):
    devfun = mkGlmerDevfun(glFormula("y ~ x + (1 | g)", data, family=Poisson()))
    devfun.optimizer.nAGQ = n_agq
    with (
        patch.object(laplace, "run_optimizer", side_effect=AssertionError("must not optimize")),
        patch("scipy.optimize.minimize", side_effect=AssertionError("must not optimize")),
        pytest.raises(ValueError, match="nAGQ must be a nonnegative integer"),
    ):
        if entry == "direct":
            devfun.optimizer.optimize()
        else:
            optimizeGlmer(devfun)

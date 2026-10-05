"""The default LMM optimizer: exact-gradient L-BFGS-B with a COBYQA fallback."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer, lmerControl, load_dyestuff2, load_sleepstudy, set_cov_type
from mixedlm.estimation.reml import _HAS_RUST, LMMOptimizer
from mixedlm.models.control import GlmerControl

from tests.test_statistical_golden import observation_space_reference

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="native gradients unavailable")

SLOPES = "Reaction ~ Days + (Days | Subject)"


def fit(formula, data, optimizer, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model is singular")
        return lmer(formula, data, control=lmerControl(optimizer=optimizer, **kwargs))


def covariance(theta):
    lower = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
    return lower @ lower.T


def test_exact_gradient_fit_matches_independent_likelihood_optimum():
    sleepstudy = load_sleepstudy()
    reference = observation_space_reference(sleepstudy, slopes=True)
    result = fit(SLOPES, sleepstudy, "auto")

    assert result.optimizer == "L-BFGS-B" and result.converged
    assert result.deviance == pytest.approx(reference["deviance"], abs=1e-8)
    np.testing.assert_allclose(covariance(result.theta), covariance(reference["theta"]), rtol=1e-5)
    # Far fewer evaluations than the derivative-free fit of the same model.
    assert result.function_evals < fit(SLOPES, sleepstudy, "COBYQA").function_evals / 2


@pytest.mark.parametrize("reml", [False, True])
def test_weight_rescaling_preserves_the_default_fit(reml):
    sleepstudy = load_sleepstudy()
    weights = 1.0 + sleepstudy["Days"].to_numpy() / 9.0
    controls = lmerControl(optimizer="auto")
    result = lmer(SLOPES, sleepstudy, weights=weights, REML=reml, control=controls)
    scaled = lmer(SLOPES, sleepstudy, weights=25.0 * weights, REML=reml, control=controls)

    assert result.optimizer == scaled.optimizer == "L-BFGS-B"
    # COBYQA's final trust radius limits its own rescaling tests to 2e-5.
    np.testing.assert_allclose(scaled.theta * 5.0, result.theta, rtol=1e-6, atol=1e-7)
    assert scaled.deviance == pytest.approx(result.deviance, abs=1e-9)


def test_scalar_variance_at_zero_is_kept_after_boundary_probes():
    data = load_dyestuff2()
    result = fit("Yield ~ 1 + (1 | Batch)", data, "auto")
    reference = fit("Yield ~ 1 + (1 | Batch)", data, "COBYQA")

    assert result.optimizer == "L-BFGS-B" and result.converged
    assert result.theta[0] == reference.theta[0] == 0
    assert result.deviance == reference.deviance
    assert result.function_evals < reference.function_evals


def test_unprobed_variance_at_zero_is_refitted_without_gradients():
    data = load_dyestuff2()
    result = fit("Yield ~ 1 + (1 | Batch)", data, "auto", restart_edge=False)
    reference = fit("Yield ~ 1 + (1 | Batch)", data, "COBYQA", restart_edge=False)

    assert result.optimizer == "COBYQA" and result.converged
    assert result.deviance == reference.deviance
    assert "after L-BFGS-B" in result.message


def test_gradient_fit_stalled_at_zero_variance_is_restarted():
    # The gradient of a variance scale vanishes at zero, so L-BFGS-B stops
    # at a zero start; boundary probes find the positive variance optimum.
    matrices = fit("Reaction ~ Days + (1 | Subject)", load_sleepstudy(), "COBYQA").matrices
    optimizer = LMMOptimizer(matrices)
    reference = optimizer.optimize(method="COBYQA")
    result = optimizer.optimize(start=np.zeros(1), method="auto")

    assert result.optimizer == "L-BFGS-B" and result.converged
    assert "variance-boundary restart" in result.message
    assert result.deviance == pytest.approx(reference.deviance, abs=1e-9)


@pytest.mark.parametrize("limit", ["maxfev", "maxfun"])
def test_evaluation_limits_bound_each_stage(limit):
    optimizer = LMMOptimizer(fit(SLOPES, load_sleepstudy(), "COBYQA").matrices)
    result = optimizer.optimize(method="auto", options={limit: 8})

    assert result.optimizer == "COBYQA" and not result.converged
    assert "after L-BFGS-B: STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT" in result.message
    assert result.function_evals <= 2 * (8 + 1)


def singular_slopes(seed):
    rng = np.random.default_rng(seed)
    groups, size = 21, 8
    data = pd.DataFrame({"g": np.repeat(np.arange(groups), size)})
    data["x1"], data["x2"] = rng.normal(size=(2, len(data)))
    effects = rng.normal(size=(groups, 3)) * [0.6, 0.0, 0.4]
    data["y"] = (
        1
        + data.x1
        + effects[data.g, 0]
        + effects[data.g, 1] * data.x1
        + effects[data.g, 2] * data.x2
        + rng.normal(scale=0.8, size=len(data))
    )
    return fit("y ~ x1 + x2 + (x1 + x2 | g)", data, "COBYQA").matrices


@pytest.mark.parametrize("seed", [8, 27, 43])
def test_singular_correlated_slopes_are_never_worse_than_derivative_free_fits(seed):
    # Singular correlated slopes have several boundary optima, so variance
    # scales of correlated terms left near zero are always refitted. COBYQA
    # alone (the earlier default) reaches a higher optimum for some seeds.
    optimizer = LMMOptimizer(singular_slopes(seed))
    derivative_free = optimizer.optimize(method="COBYQA")
    result = optimizer.optimize(method="auto")

    assert result.converged
    assert result.deviance <= derivative_free.deviance + 1e-12 * abs(derivative_free.deviance)
    if result.optimizer == "COBYQA":
        assert result.deviance == derivative_free.deviance
        np.testing.assert_array_equal(result.theta, derivative_free.theta)


@pytest.mark.parametrize("kind", ["python", "ar1"])
def test_without_native_gradients_the_default_is_derivative_free(kind):
    sleepstudy = load_sleepstudy()
    formula = set_cov_type(SLOPES, "ar1") if kind == "ar1" else SLOPES
    result = fit(formula, sleepstudy, "auto", use_rust=kind != "python")
    reference = fit(formula, sleepstudy, "COBYQA", use_rust=kind != "python")

    assert result.optimizer == "COBYQA"
    assert result.deviance == reference.deviance
    np.testing.assert_array_equal(result.theta, reference.theta)


def test_auto_is_only_a_linear_model_policy():
    with pytest.raises(ValueError, match="Unknown optimizer 'auto'"):
        GlmerControl(optimizer="auto")


def test_native_evaluator_is_the_default_for_large_random_systems():
    rng = np.random.default_rng(3)
    data = pd.DataFrame({"a": rng.integers(0, 60, 600), "b": rng.integers(0, 40, 600)})
    data["y"] = rng.normal(size=60)[data.a] + rng.normal(size=40)[data.b] + rng.normal(size=600)
    result = fit("y ~ 1 + (1 | a) + (1 | b)", data, "auto")
    assert result.matrices.n_random == 100

    optimizer = LMMOptimizer(result.matrices, REML=result.REML)
    assert optimizer.use_rust
    deviance = result.as_function("deviance")
    assert deviance(result.theta) == pytest.approx(result.deviance, abs=1e-9)
    refit = result.refitML()
    assert refit.deviance == pytest.approx(fit_ml(data).deviance, abs=1e-6)


def fit_ml(data):
    return lmer("y ~ 1 + (1 | a) + (1 | b)", data, REML=False)


def test_modular_optimizer_accepts_the_auto_policy():
    from mixedlm.models.modular import lFormula, mkLmerDevfun, optimizeLmer

    sleepstudy = load_sleepstudy()
    devfun = mkLmerDevfun(lFormula("Reaction ~ Days + (Days | Subject)", sleepstudy))
    auto = optimizeLmer(devfun, method="auto")
    reference = optimizeLmer(devfun, method="COBYQA")

    assert auto.converged
    assert auto.optimizer == ("L-BFGS-B" if _HAS_RUST else "COBYQA")
    assert auto.deviance == pytest.approx(reference.deviance, rel=1e-9)

    # Custom objectives have no exact gradient, so "auto" falls back to COBYQA.
    class Shifted(type(devfun)):
        def __call__(self, theta):
            return super().__call__(theta) + 1.0

    shifted = Shifted(parsed=devfun.parsed, optimizer=devfun.optimizer, control=devfun.control)
    custom = optimizeLmer(shifted, method="auto")
    assert custom.optimizer == "COBYQA"
    assert custom.deviance == pytest.approx(reference.deviance + 1.0, rel=1e-9)


def test_refits_keep_the_auto_policy_and_record_the_kept_method():
    sleepstudy = load_sleepstudy()
    result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    refit = result.refitML()

    assert refit.optimizer == ("L-BFGS-B" if _HAS_RUST else "COBYQA")
    reference = lmer(
        "Reaction ~ Days + (Days | Subject)",
        sleepstudy,
        REML=False,
        control=lmerControl(optimizer="COBYQA"),
    )
    assert refit.deviance == pytest.approx(reference.deviance, rel=1e-9)

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer, lmer, lmerControl
from mixedlm.estimation import joint_glmm, laplace, reml
from mixedlm.estimation.optimizers import OptimizeResult
from mixedlm.families import Gaussian, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.bootstrap import bootstrap_glmer, bootstrap_lmer
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models import modular

FORMULA = "y ~ x + (1 | g)"
MODES = ["linear", "laplace_python", "laplace_native", "agq_python", "agq_native"]


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 4.0, 2.0, 3.0, 6.0, 1.0, 4.0, 5.0, 3.0, 4.0, 8.0],
            "x": [0.0, 1.0, 2.0] * 4,
            "g": np.repeat(np.arange(4), 3),
        }
    )


def make_optimizer(data, mode):
    matrices = build_model_matrices(parse_formula(FORMULA), data)
    if mode == "linear":
        return reml.LMMOptimizer(matrices, use_rust=False)
    return laplace.GLMMOptimizer(matrices, Poisson(), nAGQ=5 if mode.startswith("agq") else 1)


@contextmanager
def backend_patch(optimizer, mode, evaluation=None, error=None):
    kwargs = {"side_effect": error} if error is not None else {"return_value": evaluation}
    if mode == "linear":
        with patch.object(optimizer, "_evaluate_core", **kwargs):
            yield
        return
    if evaluation is not None and len(evaluation) == 3:
        kwargs = {"return_value": (*evaluation, True)}
    name = (
        "_native_deviance_with_status"
        if mode.endswith("native")
        else (
            "_adaptive_gh_deviance_with_status"
            if mode == "agq_python"
            else "_laplace_deviance_with_status"
        )
    )
    with patch.object(laplace, name, **kwargs):
        if mode.endswith("native"):
            with (
                patch.object(laplace, "_evaluate_native_problem", **kwargs),
                patch.object(joint_glmm, "_evaluate_native_problem", **kwargs),
            ):
                yield
        else:
            yield


def good_evaluation(optimizer, mode):
    if mode == "linear":
        return optimizer._evaluate_core(np.ones(optimizer.n_theta))
    return (12.0, np.ones(optimizer.matrices.n_fixed), np.zeros(optimizer.matrices.n_random))


def finished(fun, x0, **kwargs):
    # The optimizer's success and cached objective cannot establish fit validity.
    return OptimizeResult(x=np.asarray(x0), fun=999.0, success=True, nit=7, message="finished")


def final_fit(optimizer, mode):
    module = reml if mode == "linear" else laplace
    with patch.object(module, "run_optimizer", side_effect=finished):
        return optimizer.optimize(start=np.ones(optimizer.n_theta))


FAULTS = [
    ("deviance", np.nan),
    ("deviance", np.inf),
    ("deviance", -np.inf),
    ("deviance", 1 + 2j),
    ("deviance", [12.0]),
    ("beta", [np.nan, 1.0]),
    ("beta", [np.inf, 1.0]),
    ("beta", [1 + 1j, 1.0]),
    ("beta", [[1.0], [1.0]]),
    ("beta", [1.0]),
    ("u", [np.nan] * 4),
    ("u", [np.inf] * 4),
    ("u", [1 + 1j] * 4),
    ("u", np.zeros((4, 1))),
    ("u", np.zeros(3)),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("field,value", FAULTS)
def test_invalid_final_estimates_rejected_despite_optimizer_success(data, mode, field, value):
    optimizer = make_optimizer(data, mode)
    evaluation = good_evaluation(optimizer, mode)
    if mode == "linear":
        evaluation = replace(evaluation, **{field: np.asarray(value)})
    else:
        values = list(evaluation)
        values[{"deviance": 0, "beta": 1, "u": 2}[field]] = np.asarray(value)
        evaluation = tuple(values)
    label = {"deviance": "deviance", "beta": "fixed effects", "u": "random effects"}[field]
    with (
        patch.object(laplace, "_HAS_RUST", mode.endswith("native")),
        backend_patch(optimizer, mode, evaluation),
        pytest.raises(RuntimeError, match=f"did not produce a valid fit:.*{label}"),
    ):
        final_fit(optimizer, mode)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf, 1 + 1j, [1.0]])
def test_invalid_linear_residual_scale_is_rejected(data, scale):
    optimizer = make_optimizer(data, "linear")
    evaluation = replace(good_evaluation(optimizer, "linear"), sigma=np.asarray(scale))
    with (
        backend_patch(optimizer, "linear", evaluation),
        pytest.raises(RuntimeError, match="residual scale"),
    ):
        final_fit(optimizer, "linear")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "error", [ValueError, TypeError, FloatingPointError, OverflowError, np.linalg.LinAlgError]
)
def test_final_backend_failure_preserves_reason_and_cause(data, mode, error):
    optimizer = make_optimizer(data, mode)
    failure = error("final numerical failure")
    with (
        patch.object(laplace, "_HAS_RUST", mode.endswith("native")),
        backend_patch(optimizer, mode, error=failure),
        pytest.raises(RuntimeError, match="final numerical failure") as caught,
    ):
        final_fit(optimizer, mode)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "theta",
    [np.array([np.nan]), np.array([np.inf]), np.array([1 + 1j]), np.ones(2), np.ones((1, 1))],
)
@pytest.mark.parametrize("mode", ["linear", "laplace_native", "agq_native"])
def test_invalid_final_theta_is_rejected_before_backend(data, theta, mode):
    optimizer = make_optimizer(data, mode)
    module = reml if mode == "linear" else laplace
    with (
        patch.object(module, "run_optimizer", return_value=OptimizeResult(theta, 1.0, True, 0, "")),
        patch.object(laplace, "_HAS_RUST", True),
        backend_patch(optimizer, mode, error=AssertionError("backend must not be called")),
        pytest.raises(RuntimeError, match="variance parameters"),
    ):
        optimizer.optimize(start=np.ones(1))


@pytest.mark.parametrize("method", ["optimize", "final_evaluation", "modular"])
def test_failed_linear_factorization_does_not_fabricate_estimates(data, method):
    parsed = modular.lFormula(FORMULA, data)
    devfun = modular.mkLmerDevfun(parsed)
    optimizer = devfun.optimizer
    with (
        backend_patch(optimizer, "linear", None),
        pytest.raises(RuntimeError, match="factorization failed"),
    ):
        if method == "optimize":
            final_fit(optimizer, "linear")
        elif method == "final_evaluation":
            optimizer._final_evaluation(np.ones(1))
        else:
            modular.mkLmerMod(devfun, modular.OptimizeResult(np.ones(1), 1e10, True, 0, ""))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("deviance", [-1e101, 0.0, 1e10, 1e100])
def test_valid_fits_keep_status_and_use_recomputed_deviance(data, mode, success, deviance):
    optimizer = make_optimizer(data, mode)
    evaluation = good_evaluation(optimizer, mode)
    evaluation = (
        replace(evaluation, deviance=deviance) if mode == "linear" else (deviance, *evaluation[1:])
    )
    result = OptimizeResult(np.ones(1), np.nan, success, 7, "iteration limit", nfev=9)
    module = reml if mode == "linear" else laplace
    with (
        patch.object(laplace, "_HAS_RUST", mode.endswith("native")),
        backend_patch(optimizer, mode, evaluation),
        patch.object(
            module, "run_optimizer", side_effect=lambda fun, x0, **kwargs: replace(result, x=x0)
        ),
    ):
        fitted = optimizer.optimize(start=np.ones(1))
    assert fitted.converged is success
    assert fitted.deviance == deviance
    assert fitted.n_iter == (7 if mode == "linear" else 14)
    if mode == "linear":
        assert fitted.message == "iteration limit"
        assert fitted.function_evals == 9


@pytest.mark.parametrize("n_agq", [1, 5])
def test_modular_glmm_validates_final_estimates(data, n_agq):
    parsed = modular.glFormula(FORMULA, data, family=Poisson())
    devfun = modular.mkGlmerDevfun(parsed, nAGQ=n_agq)
    mode = "laplace_python" if n_agq == 1 else "agq_python"
    evaluation = (12.0, np.full(parsed.n_fixed, np.nan), np.zeros(parsed.n_random))
    with (
        patch.object(laplace, "_HAS_RUST", False),
        backend_patch(devfun.optimizer, mode, evaluation),
        pytest.raises(RuntimeError, match="fixed effects"),
    ):
        modular.mkGlmerMod(
            devfun, modular.OptimizeResult(np.ones(1), 12.0, True, 0, ""), nAGQ=n_agq
        )


@pytest.mark.parametrize("n_agq", [1, 5])
def test_modular_glmm_reports_final_quadrature_deviance(data, n_agq):
    parsed = modular.glFormula(FORMULA, data, family=Poisson())
    devfun = modular.mkGlmerDevfun(parsed, nAGQ=n_agq)
    theta = np.ones(1)
    expected = laplace.glmm_deviance_with_status(theta, parsed.matrices, parsed.family, nAGQ=n_agq)
    actual = modular.mkGlmerMod(
        devfun, modular.OptimizeResult(theta, -999.0, True, 0, ""), nAGQ=n_agq
    )
    assert actual.deviance == expected[0]
    np.testing.assert_array_equal(actual.beta, expected[1])
    np.testing.assert_array_equal(actual.u, expected[2])


@pytest.mark.parametrize("kind", ["linear", "generalized"])
def test_invalid_refits_raise_and_bootstrap_counts_failures(data, kind):
    result = lmer(FORMULA, data) if kind == "linear" else glmer(FORMULA, data, family=Poisson())
    target = (
        patch.object(reml.LMMOptimizer, "_evaluate_core", return_value=None)
        if kind == "linear"
        else patch.object(
            laplace,
            "glmm_deviance_with_status",
            return_value=(12.0, np.full(2, np.nan), np.zeros(4), True),
        )
    )
    with target:
        with pytest.raises(RuntimeError, match="did not produce a valid fit"):
            result.refit(result.matrices.y)
        bootstrap = bootstrap_lmer if kind == "linear" else bootstrap_glmer
        boot = bootstrap(result, n_boot=3, seed=42)
    assert boot.n_failed == 3
    assert np.isnan(boot.beta_samples).all()
    assert np.isnan(boot.theta_samples).all()


@pytest.mark.parametrize("native", [False, True])
def test_actual_zero_residual_linear_fit_is_rejected(data, native):
    if native:
        pytest.importorskip("mixedlm._rust")
    with (
        np.errstate(divide="ignore", invalid="ignore"),
        pytest.raises(RuntimeError, match="deviance|residual scale"),
    ):
        lmer(FORMULA, data.assign(y=0.0), control=lmerControl(use_rust=native), maxiter=2)


def test_custom_family_nonfinite_deviance_is_rejected(data):
    class NonfiniteGaussian(Gaussian):
        def deviance_resids(self, y, mu, weights):
            return np.full_like(y, np.nan)

    with pytest.raises(RuntimeError, match="deviance must contain finite real values"):
        glmer(FORMULA, data, family=NonfiniteGaussian(), maxiter=2)


@pytest.mark.parametrize("mode", ["linear", "laplace_python", "agq_python"])
@pytest.mark.parametrize("empty", ["fixed", "random", "both"])
def test_empty_estimate_vectors_are_valid_when_model_dimensions_match(data, mode, empty):
    from scipy import sparse

    optimizer = make_optimizer(data, mode)
    evaluation = good_evaluation(optimizer, mode)
    matrices = optimizer.matrices
    beta = evaluation.beta if mode == "linear" else evaluation[1]
    u = evaluation.u if mode == "linear" else evaluation[2]
    if empty in ("fixed", "both"):
        matrices = replace(matrices, X=np.empty((matrices.n_obs, 0)), n_fixed=0)
        beta = np.empty(0)
    if empty in ("random", "both"):
        matrices = replace(
            matrices, Z=sparse.csc_matrix((matrices.n_obs, 0)), n_random=0, random_structures=[]
        )
        u = np.empty(0)
        optimizer.n_theta = 0
    optimizer.matrices = matrices
    evaluation = (
        replace(evaluation, beta=beta, u=u) if mode == "linear" else (evaluation[0], beta, u)
    )
    with patch.object(laplace, "_HAS_RUST", False), backend_patch(optimizer, mode, evaluation):
        result = final_fit(optimizer, mode)
    assert result.converged
    assert result.beta.shape == (matrices.n_fixed,)
    assert result.u.shape == (matrices.n_random,)


def test_modular_linear_result_uses_validated_final_deviance(data):
    devfun = modular.mkLmerDevfun(modular.lFormula(FORMULA, data))
    theta = np.ones(1)
    expected = reml.profiled_deviance(theta, devfun.parsed.matrices)
    actual = modular.mkLmerMod(devfun, modular.OptimizeResult(theta, np.nan, False, 2, "limit"))
    assert actual.deviance == expected
    assert not actual.converged

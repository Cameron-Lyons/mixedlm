"""Fitting controls govern the inner solve through every likelihood route."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import _rust, glFormula, glmer, glmerControl, mkGlmerDevfun, mkGlmerMod
from mixedlm.estimation import laplace
from mixedlm.families import Binomial, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import GlmerControl
from mixedlm.models.modular import optimizeGlmer
from numpy.testing import assert_allclose, assert_array_equal


def fixture(family_name="poisson", random=True):
    rng = np.random.default_rng(191)
    n = 72
    x = rng.uniform(-1, 1, n)
    group = np.arange(n) % 6
    offset = 0.1 * np.cos(np.arange(n))
    eta = 0.7 + 0.8 * x + 0.2 * np.sin(group) + offset
    family = Poisson() if family_name == "poisson" else Binomial()
    y = (
        rng.poisson(np.exp(eta))
        if family_name == "poisson"
        else rng.binomial(1, 1 / (1 + np.exp(-eta)))
    )
    data = pd.DataFrame({"y": y.astype(float), "x": x, "g": group})
    formula = "y ~ x + (1 | g)" if random else "y ~ x"
    weights = np.linspace(0.5, 1.5, n)
    matrices = build_model_matrices(parse_formula(formula), data, weights=weights, offset=offset)
    theta = np.array([0.55]) if random else np.array([])
    return data, formula, matrices, family, theta


def stopped_optimizer(fun, start, **kwargs):
    # Fix the outer parameters so these tests measure the actual inner solve.
    return SimpleNamespace(x=np.asarray(start), fun=fun(start), success=True, nit=1)


@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
@pytest.mark.parametrize("random", [False, True])
@pytest.mark.parametrize("order", [1, 7])
@pytest.mark.parametrize("backend", ["python", "native"])
def test_likelihood_controls_change_inner_status_and_estimates(family_name, random, order, backend):
    _, _, matrices, family, theta = fixture(family_name, random)
    with patch.object(laplace, "_HAS_RUST", backend == "native"):
        limited = laplace.glmm_deviance_with_status(
            theta, matrices, family, order, pirls_maxiter=1, pirls_tol=1e-12
        )
        loose = laplace.glmm_deviance_with_status(
            theta, matrices, family, order, pirls_maxiter=1, pirls_tol=1e6
        )
        complete = laplace.glmm_deviance_with_status(
            theta, matrices, family, order, pirls_maxiter=100, pirls_tol=1e-12
        )
        legacy = laplace.adaptive_gh_deviance_fast(
            theta, matrices, family, order, pirls_maxiter=1, pirls_tol=1e-12
        )
        objective = laplace.GLMMOptimizer(
            matrices, family, nAGQ=order, pirls_maxiter=1, pirls_tol=1e-12
        ).objective(theta)
    assert limited[3] is False
    assert loose[3] is True
    assert complete[3] is True
    assert np.max(np.abs(limited[1] - complete[1])) > 1e-5
    for actual, expected in zip(limited[:3], legacy, strict=True):
        assert_array_equal(actual, expected)
    assert objective == limited[0]
    # Tolerance controls stopping, rather than changing an individual PIRLS update.
    for actual, expected in zip(limited[:3], loose[:3], strict=True):
        assert_array_equal(actual, expected)
    beta, random_effects, _, converged = laplace.pirls(
        matrices, family, theta, maxiter=100, tol=1e-12
    )
    assert converged
    assert_allclose(complete[1], beta, rtol=1e-9, atol=1e-9)
    assert_allclose(complete[2], random_effects, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("order", [1, 7])
def test_public_fit_honors_tolerance_and_retains_controls_for_refits(backend, order):
    data, formula, matrices, family, theta = fixture()
    control = glmerControl(tolPwrss=1e-12, pirls_maxiter=1, check_singular=False)
    with (
        patch.object(laplace, "_HAS_RUST", backend == "native"),
        patch.object(laplace, "run_optimizer", side_effect=stopped_optimizer),
    ):
        with pytest.warns(UserWarning, match="inner PIRLS solver did not converge"):
            fitted = glmer(
                formula,
                data,
                family=family,
                control=control,
                start=theta,
                nAGQ=order,
                weights=matrices.weights,
                offset=matrices.offset,
            )
        assert not fitted.converged
        assert not fitted.pirls_converged
        assert fitted.pirls_maxiter == 1
        assert fitted.pirls_tol == 1e-12
        # A mutable control object must not change an already fitted model's refit.
        control.pirls_maxiter = 100
        control.tolPwrss = 1e6
        repeated = fitted.refit()
        assert repeated.pirls_maxiter == 1
        assert repeated.pirls_tol == 1e-12
        assert not repeated.pirls_converged
        assert_array_equal(repeated.beta, fitted.beta)
        loose = fitted.refit(pirls_tol=1e6)
        assert loose.pirls_converged
        assert loose.pirls_maxiter == 1
        recovered = fitted.refit(pirls_maxiter=100)
        assert recovered.converged
        assert recovered.pirls_converged
        assert recovered.pirls_tol == 1e-12
        assert recovered.pirls_maxiter == 100
        # Exercise tolPwrss on a new public fit as well as on a refit.
        control.pirls_maxiter = 1
        loose_fit = glmer(
            formula,
            data,
            family=family,
            control=control,
            start=theta,
            nAGQ=order,
            weights=matrices.weights,
            offset=matrices.offset,
        )
        assert loose_fit.converged
        assert_array_equal(loose_fit.beta, fitted.beta)


@pytest.mark.parametrize("backend", ["python", "native"])
def test_modular_fitting_uses_inner_controls_in_objective_and_result(backend):
    data, formula, matrices, family, theta = fixture()
    parsed = glFormula(
        formula, data, family=family, weights=matrices.weights, offset=matrices.offset
    )
    control = glmerControl(tolPwrss=1e-12, pirls_maxiter=1)
    with (
        patch.object(laplace, "_HAS_RUST", backend == "native"),
        patch("scipy.optimize.minimize", side_effect=stopped_optimizer),
    ):
        devfun = mkGlmerDevfun(parsed, control=control)
        expected = laplace.glmm_deviance_with_status(
            theta, matrices, family, pirls_maxiter=1, pirls_tol=1e-12
        )
        assert devfun(theta) == expected[0]
        opt = optimizeGlmer(devfun, start=theta)
        assert not opt.pirls_converged
        assert not opt.converged
        assert opt.deviance == expected[0]
        result = mkGlmerMod(devfun, opt)
    assert not result.pirls_converged
    assert not result.converged
    assert result.pirls_maxiter == 1
    assert result.pirls_tol == 1e-12
    assert_array_equal(result.beta, expected[1])


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("order", [1, 7])
def test_direct_likelihood_defaults_preserve_backend_iteration_limits(backend, order):
    _, _, matrices, family, theta = fixture()
    with patch.object(laplace, "_HAS_RUST", backend == "native"):
        default = laplace.glmm_deviance_with_status(theta, matrices, family, order)
        explicit = laplace.glmm_deviance_with_status(
            theta,
            matrices,
            family,
            order,
            pirls_maxiter=100 if backend == "native" else 25,
            pirls_tol=1e-6,
        )
    for actual, expected in zip(default, explicit, strict=True):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("order", [1, 7])
def test_native_entry_points_forward_options_to_the_same_inner_solve(order):
    _, _, matrices, family, theta = fixture()
    args = laplace._native_glmm_args(theta, matrices, family)
    status = _rust.glmm_deviance(*args, order, maxiter=1, tol=1e-12)
    agq = _rust.adaptive_gh_deviance(*args, order, maxiter=1, tol=1e-12)
    beta, random, _, converged = _rust.pirls(*args, maxiter=1, tol=1e-12)
    assert not converged
    assert not status[3]
    assert_array_equal(beta, status[1])
    assert_array_equal(random, status[2])
    for actual, expected in zip(status[:3], agq, strict=True):
        assert_array_equal(actual, expected)
    if order == 1:
        actual = _rust.laplace_deviance(*args, maxiter=1, tol=1e-12)
        for left, right in zip(actual, status[:3], strict=True):
            assert_array_equal(left, right)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.nan, np.inf, "2", 1j, [1]])
def test_invalid_iteration_controls_are_rejected_before_fitting(value):
    _, _, matrices, family, theta = fixture()
    with pytest.raises(ValueError, match="pirls_maxiter must be a positive integer"):
        glmerControl(pirls_maxiter=value)
    with pytest.raises(ValueError, match="pirls_maxiter must be a positive integer"):
        laplace.GLMMOptimizer(matrices, family, pirls_maxiter=value)
    for native in (False, True):
        with (
            patch.object(laplace, "_HAS_RUST", native),
            pytest.raises(ValueError, match="pirls_maxiter must be a positive integer"),
        ):
            laplace.glmm_deviance_with_status(theta, matrices, family, pirls_maxiter=value)


@pytest.mark.parametrize("value", [0, -1, True, np.nan, np.inf, -np.inf, "1e-6", 1j, [1], 10**400])
def test_invalid_tolerances_are_rejected_before_fitting(value):
    _, _, matrices, family, theta = fixture()
    with pytest.raises(ValueError, match="tolPwrss must be positive and finite"):
        GlmerControl(tolPwrss=value)
    with pytest.raises(ValueError, match="pirls_tol must be positive and finite"):
        laplace.GLMMOptimizer(matrices, family, pirls_tol=value)
    for native in (False, True):
        with (
            patch.object(laplace, "_HAS_RUST", native),
            pytest.raises(ValueError, match="pirls_tol must be positive and finite"),
        ):
            laplace.glmm_deviance_with_status(theta, matrices, family, pirls_tol=value)


@pytest.mark.parametrize(
    "function", ["pirls", "laplace_deviance", "adaptive_gh_deviance", "glmm_deviance"]
)
@pytest.mark.parametrize("options", [{"maxiter": 0}, {"tol": 0}, {"tol": np.nan}, {"tol": np.inf}])
def test_native_entry_points_reject_invalid_controls(function, options):
    _, _, matrices, family, theta = fixture()
    args = laplace._native_glmm_args(theta, matrices, family)
    if function in {"adaptive_gh_deviance", "glmm_deviance"}:
        args += (7,)
    with pytest.raises(ValueError, match="must be"):
        getattr(_rust, function)(*args, **options)


def test_numpy_scalar_controls_and_omitted_limit_are_supported():
    control = glmerControl(tolPwrss=np.float64(1e-8), pirls_maxiter=np.int64(30))
    assert control.pirls_maxiter == 30
    assert control.tolPwrss == 1e-8
    assert glmerControl().pirls_maxiter is None


@pytest.mark.parametrize("order", [1, 7])
@pytest.mark.parametrize("fallback", ["starts", "custom_family"])
def test_controls_survive_native_to_python_fallback(order, fallback):
    _, _, matrices, family, theta = fixture()
    kwargs = {}
    if fallback == "starts":
        kwargs = {"beta_start": np.zeros(matrices.n_fixed), "u_start": np.zeros(matrices.n_random)}
    else:

        class CustomPoisson(Poisson):
            pass

        family = CustomPoisson()
    expected = laplace.pirls(matrices, family, theta, maxiter=1, tol=1e-12, **kwargs)
    with patch.object(laplace, "_HAS_RUST", True):
        actual = laplace.glmm_deviance_with_status(
            theta, matrices, family, order, pirls_maxiter=1, pirls_tol=1e-12, **kwargs
        )
    assert actual[3] is False
    assert_array_equal(actual[1], expected[0])
    assert_array_equal(actual[2], expected[1])


def fitted_with_controls(tol=1e-12):
    data, formula, matrices, family, theta = fixture()
    result = glmer(
        formula,
        data,
        family=family,
        start=theta,
        weights=matrices.weights,
        offset=matrices.offset,
        control=glmerControl(tolPwrss=tol, pirls_maxiter=1, check_conv=False, check_singular=False),
    )
    return data, result


@pytest.mark.parametrize("backend", ["python", "native"])
def test_objective_reconstruction_and_model_update_retain_inner_controls(backend):
    with (
        patch.object(laplace, "_HAS_RUST", backend == "native"),
        patch.object(laplace, "run_optimizer", side_effect=stopped_optimizer),
    ):
        data, result = fitted_with_controls()
        assert result.as_function()(result.theta) == result.deviance
        with pytest.warns(UserWarning, match="inner PIRLS solver"):
            updated = result.update(data=data, start=result.theta)
        assert updated.pirls_maxiter == 1
        assert updated.pirls_tol == 1e-12
        assert not updated.pirls_converged
        assert_array_equal(updated.beta, result.beta)
        overridden = result.update(
            data=data,
            start=result.theta,
            control=glmerControl(tolPwrss=1e6, pirls_maxiter=1, check_singular=False),
        )
        assert overridden.pirls_converged
        assert overridden.pirls_tol == 1e6


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cross_validation_inherits_controls_and_accepts_explicit_override(backend, n_jobs):
    from mixedlm.inference.cross_validation import cross_validate

    with (
        patch.object(laplace, "_HAS_RUST", backend == "native"),
        patch.object(laplace, "run_optimizer", side_effect=stopped_optimizer),
    ):
        data, result = fitted_with_controls()
        with pytest.warns(UserWarning, match="inner PIRLS solver"):
            limited = cross_validate(result, data, cv=2, random_state=23, n_jobs=n_jobs)
        assert not limited.all_converged
        recovered = cross_validate(
            result,
            data,
            cv=2,
            random_state=23,
            n_jobs=n_jobs,
            fit_kwargs={"control": glmerControl(tolPwrss=1e6, pirls_maxiter=1)},
        )
        assert recovered.all_converged


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bootstrap_uses_fitted_controls_in_serial_and_worker_paths(n_jobs):
    from concurrent.futures import ThreadPoolExecutor

    from mixedlm.inference import bootstrap

    original_optimizer = laplace.GLMMOptimizer
    settings = []

    def record_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        settings.append((optimizer.pirls_maxiter, optimizer.pirls_tol))
        return optimizer

    with patch.object(laplace, "run_optimizer", side_effect=stopped_optimizer):
        _, result = fitted_with_controls(1e6)
        with (
            patch.object(laplace, "GLMMOptimizer", side_effect=record_optimizer),
            patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            samples = bootstrap.bootstrap_glmer(result, n_boot=3, seed=25, n_jobs=n_jobs)
    assert samples.n_failed == 0
    assert settings == [(1, 1e6)] * 3
    assert np.isfinite(samples.beta_samples).all()


@pytest.mark.parametrize("workflow", ["allfit", "drop1"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_model_comparisons_retain_inner_controls(workflow, n_jobs):
    from concurrent.futures import ThreadPoolExecutor

    from mixedlm.inference import allfit, drop1
    from mixedlm.models.glmer import GlmerMod

    settings = []
    original_fit = GlmerMod.fit

    def record_fit(self, *args, **kwargs):
        settings.append((self.control.pirls_maxiter, self.control.tolPwrss))
        return original_fit(self, *args, **kwargs)

    with patch.object(laplace, "run_optimizer", side_effect=stopped_optimizer):
        data, result = fitted_with_controls(1e6)
        with (
            patch.object(GlmerMod, "fit", record_fit),
            patch.object(allfit, "ProcessPoolExecutor", ThreadPoolExecutor),
            patch.object(drop1, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            if workflow == "allfit":
                comparison = allfit.allfit_glmer(
                    result, data, optimizers=["L-BFGS-B"], n_jobs=n_jobs
                )
                assert not comparison.errors
                fitted = comparison.fits["L-BFGS-B"]
                assert fitted.pirls_maxiter == 1
                assert fitted.pirls_tol == 1e6
            else:
                comparison = drop1.drop1_glmer(result, data, n_jobs=n_jobs)
                assert comparison.terms == ["x"]
    assert settings == [(1, 1e6)]

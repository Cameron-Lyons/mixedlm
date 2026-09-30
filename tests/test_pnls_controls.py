from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import _rust, nlmer
from mixedlm.estimation import nlmm
from mixedlm.nlme.models import SSmicmen


def problem():
    x = np.tile([0.2, 0.5, 1.0, 2.0, 3.0, 5.0], 3)
    groups = np.repeat([-10, 4, 80], 6)
    y = np.repeat([2.8, 3.1, 3.4], 6) * x / (0.9 + x)
    y += 0.04 * np.sin(np.arange(len(x)))
    return dict(
        y=y,
        x=x,
        groups=groups,
        model=SSmicmen(),
        phi=np.array([2.0, 1.2]),
        b=np.array([[-0.1], [0.0], [0.1]]),
        random_params=[0],
        sigma=0.3,
        weights=np.linspace(0.5, 2.0, len(x)),
    )


def first_update_oracle(data, theta):
    """Independent weighted Michaelis-Menten formulas for one PNLS update."""
    x, y, weights = data["x"], data["y"], data["weights"]
    phi, b = data["phi"], data["b"][:, 0]
    group = np.repeat(np.arange(3), 6)
    vm = phi[0] + b[group]
    fraction = x / (phi[1] + x)
    gradient = np.column_stack([fraction, -vm * x / (phi[1] + x) ** 2])
    residual = y - vm * fraction
    update = np.linalg.solve(
        gradient.T @ (weights[:, None] * gradient) + 1e-6 * np.eye(2),
        gradient.T @ (weights * residual),
    )
    phi = phi + 0.5 * update
    precision = 1 / (theta[0] ** 2 + 1e-8)
    b = np.empty(3)
    correction = 0.0
    for g in range(3):
        rows = group == g
        z = x[rows] / (phi[1] + x[rows])
        info = weights[rows] @ z**2
        b[g] = (weights[rows] * z) @ (y[rows] - phi[0] * z) / (info + precision)
        correction += np.log1p(theta[0] ** 2 * info)
    residual = y - (phi[0] + b[group]) * x / (phi[1] + x)
    variance = (weights @ residual**2 + precision * (b @ b)) / len(y)
    deviance = len(y) * (1 + np.log(2 * np.pi * variance)) + correction
    return deviance, phi, b[:, None], np.sqrt(variance)


def evaluate(backend, data, theta, **controls):
    if backend == "native":
        return nlmm._nlmm_deviance_rust_with_status(theta, **data, **controls)
    return nlmm.nlmm_deviance_with_status(theta, **data, n_jobs=int(backend), **controls)


@pytest.mark.parametrize("backend", ["1", "2", "native"])
def test_iteration_limit_and_tolerance_match_independent_update(backend):
    data = problem()
    theta = np.array([0.4])
    expected = first_update_oracle(data, theta)
    limited = evaluate(backend, data, theta, pnls_maxiter=1, pnls_tol=1e-12)
    loose = evaluate(backend, data, theta, pnls_maxiter=1, pnls_tol=1e6)
    complete = evaluate(backend, data, theta, pnls_maxiter=1000, pnls_tol=1e-10)
    assert limited[4] is False and loose[4] is True and complete[4] is True
    for actual, reference in zip(limited[:4], expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    for actual, reference in zip(limited[:4], loose[:4], strict=True):
        np.testing.assert_array_equal(actual, reference)
    assert np.linalg.norm(complete[1] - limited[1]) > 1e-3


@pytest.mark.parametrize("backend", ["1", "2", "native"])
def test_legacy_deviance_returns_the_same_values_without_status(backend):
    data = problem()
    theta = np.array([0.4])
    expected = evaluate(backend, data, theta, pnls_maxiter=1, pnls_tol=1e-12)
    if backend == "native":
        actual = nlmm._nlmm_deviance_rust(theta, **data, pnls_maxiter=1, pnls_tol=1e-12)
    else:
        actual = nlmm.nlmm_deviance(
            theta, **data, n_jobs=int(backend), pnls_maxiter=1, pnls_tol=1e-12
        )
    assert len(actual) == 4
    for value, reference in zip(actual, expected[:4], strict=True):
        np.testing.assert_array_equal(value, reference)


def stopped_optimizer(fun, start, **kwargs):
    fun(start)
    return SimpleNamespace(x=np.asarray(start), success=True, nit=0)


@pytest.mark.parametrize("backend", ["python", "native"])
def test_public_fit_refit_and_update_retain_independent_inner_controls(backend):
    data = problem()
    offsets = np.linspace(-0.2, 0.2, len(data["y"]))
    frame = pd.DataFrame(dict(y=data["y"] + offsets, x=data["x"], g=data["groups"]))
    with (
        patch.object(nlmm, "_HAS_RUST", backend == "native"),
        patch.object(nlmm, "minimize", side_effect=stopped_optimizer),
    ):
        with pytest.warns(UserWarning, match="inner PNLS solver did not converge"):
            fit = nlmer(
                data["model"],
                frame,
                x_var="x",
                y_var="y",
                group_var="g",
                random_params=[0],
                start={"Vm": 2.0, "K": 1.2},
                weights=data["weights"],
                offset=offsets,
                pnls_maxiter=1,
                pnls_tol=1e-12,
            )
        assert not fit.converged and not fit.pnls_converged
        assert fit.pnls_maxiter == 1 and fit.pnls_tol == 1e-12
        assert "inner PNLS convergence: no" in fit.summary()
        repeated = fit.refit()
        assert not repeated.converged and not repeated.pnls_converged
        assert repeated.pnls_maxiter == 1 and repeated.pnls_tol == 1e-12
        loose = fit.refit(pnls_tol=1e6)
        assert loose.converged and loose.pnls_converged
        assert loose.pnls_maxiter == 1
        complete = fit.refit(pnls_maxiter=3000)
        assert complete.converged and complete.pnls_converged
        assert complete.pnls_maxiter == 3000 and complete.pnls_tol == 1e-12
        with pytest.warns(UserWarning, match="inner PNLS solver did not converge"):
            updated = fit.update(start={"Vm": 2.0, "K": 1.2})
        assert not updated.pnls_converged
        assert updated.pnls_maxiter == 1 and updated.pnls_tol == 1e-12
        recovered = fit.update(start={"Vm": 2.0, "K": 1.2}, pnls_maxiter=3000)
        assert recovered.pnls_converged and recovered.converged
    np.testing.assert_array_equal(complete.weights(), data["weights"])
    np.testing.assert_array_equal(complete.offset(), offsets)


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("outer_converged", [False, True])
@pytest.mark.parametrize("inner_converged", [False, True])
def test_outer_success_requires_inner_convergence_and_final_status_is_cached(
    backend, outer_converged, inner_converged
):
    data = problem()
    starts = {f"start_{name}": data.pop(name) for name in ("phi", "b", "sigma")}
    optimizer = nlmm.NLMMOptimizer(
        **data,
        use_rust=backend == "native",
        pnls_maxiter=1,
        pnls_tol=1e6 if inner_converged else 1e-12,
    )
    target = "_nlmm_deviance_rust_with_status" if backend == "native" else "_nlmm_deviance"

    def stop(fun, start, **kwargs):
        fun(start)
        return SimpleNamespace(x=np.asarray(start), success=outer_converged, nit=0)

    with (
        patch.object(nlmm, "minimize", side_effect=stop),
        patch.object(nlmm, target, wraps=getattr(nlmm, target)) as evaluator,
    ):
        result = optimizer.optimize(**starts)
    assert evaluator.call_count == 1
    assert result.pnls_converged is inner_converged
    assert result.converged is (outer_converged and inner_converged)


INVALID_CONTROLS = [
    ("pnls_maxiter", value) for value in [0, -1, 1.5, True, np.bool_(True), "2", 1j]
] + [
    ("pnls_tol", value)
    for value in [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False), "0.1", 1j, 10**1000]
]


@pytest.mark.parametrize("name,value", INVALID_CONTROLS)
@pytest.mark.parametrize("entry", ["optimizer", "pnls", "deviance", "status", "native"])
def test_invalid_controls_fail_before_evaluation(entry, name, value):
    data = problem()
    controls = {name: value}
    with (
        patch.object(data["model"], "predict", side_effect=AssertionError("model evaluated")),
        pytest.raises(ValueError, match=name),
    ):
        if entry == "optimizer":
            for key in ("phi", "b", "sigma"):
                data.pop(key)
            nlmm.NLMMOptimizer(**data, **controls)
        elif entry == "pnls":
            key = "maxiter" if name == "pnls_maxiter" else "tol"
            nlmm.pnls_step(**data, Psi=np.eye(1), **{key: value})
        elif entry == "native":
            nlmm._nlmm_deviance_rust_with_status(np.array([0.4]), **data, **controls)
        else:
            function = nlmm.nlmm_deviance if entry == "deviance" else nlmm.nlmm_deviance_with_status
            function(np.array([0.4]), **data, **controls)


@pytest.mark.parametrize("tol", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("entry", ["deviance", "status", "pnls"])
def test_native_bindings_reject_invalid_tolerances(entry, tol):
    data = problem()
    args = (data["y"], data["x"], data["groups"], "ssmicmen", data["phi"], data["b"])
    theta = np.array([0.4])
    with pytest.raises(ValueError, match="pnls_tol"):
        if entry == "pnls":
            _rust.pnls_step(*args, theta, data["sigma"], [0], data["weights"], tol=tol)
        else:
            function = (
                _rust.nlmm_deviance if entry == "deviance" else _rust.nlmm_deviance_with_status
            )
            function(theta, *args, [0], data["sigma"], data["weights"], tol=tol)


@pytest.mark.parametrize("entry", ["deviance", "status", "pnls"])
def test_native_bindings_reject_zero_iteration_limit(entry):
    data = problem()
    args = (data["y"], data["x"], data["groups"], "ssmicmen", data["phi"], data["b"])
    theta = np.array([0.4])
    with pytest.raises(ValueError, match="pnls_maxiter"):
        if entry == "pnls":
            _rust.pnls_step(*args, theta, data["sigma"], [0], data["weights"], maxiter=0)
        else:
            function = (
                _rust.nlmm_deviance if entry == "deviance" else _rust.nlmm_deviance_with_status
            )
            function(theta, *args, [0], data["sigma"], data["weights"], maxiter=0)


@pytest.mark.parametrize("backend", ["python", "native"])
def test_real_outer_optimum_at_default_limit_does_not_claim_inner_convergence(backend):
    from mixedlm.nlme.models import SSasymp

    rng = np.random.default_rng(12)
    x = np.tile(np.linspace(0, 5, 10), 5)
    groups = np.repeat(np.arange(5), 10)
    model = SSasymp()
    phi = np.array([10.0, 0.5, -0.5])
    y = np.concatenate([model.predict(phi + [b, 0, 0], x[:10]) for b in rng.normal(0, 1, 5)])
    y += rng.normal(0, 0.3, len(x))
    optimizer = nlmm.NLMMOptimizer(y, x, groups, model, [0], use_rust=backend == "native")
    result = optimizer.optimize(start_phi=phi)
    assert not result.converged and not result.pnls_converged
    assert result.n_iter > 0 and np.isfinite(result.deviance)
    # Verify incompleteness using an additional update at the final covariance.
    advanced_phi, advanced_b, _ = nlmm.pnls_step(
        y,
        x,
        groups,
        model,
        result.phi,
        result.b,
        np.array([[result.theta[0] ** 2]]),
        result.sigma,
        [0],
        maxiter=1,
    )
    remaining = max(
        np.max(np.abs(advanced_phi - result.phi)), np.max(np.abs(advanced_b - result.b))
    )
    assert remaining > 1e-4


@pytest.mark.parametrize("backend", ["1", "2", "native"])
def test_numpy_scalar_controls_and_default_limit_match_explicit_controls(backend):
    data = problem()
    theta = np.array([0.4])
    implicit = evaluate(backend, data, theta)
    explicit = evaluate(backend, data, theta, pnls_maxiter=np.int64(50), pnls_tol=np.float64(1e-6))
    for actual, reference in zip(implicit, explicit, strict=True):
        np.testing.assert_array_equal(actual, reference)

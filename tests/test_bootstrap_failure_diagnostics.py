"""Failure details remain useful across model types and worker counts."""

import pickle
from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm import BootstrapFailure, bootMer
from mixedlm.inference import bootstrap
from mixedlm.models.nlmer import NlmerResult

from tests.test_bootstrap_workers import ImmediateExecutor
from tests.test_model_random_streams import make_result
from tests.test_nonlinear_simulation_streams import make_result as make_nonlinear_result


@pytest.fixture(params=["lmer", "glmer", "nlmer"])
def model(request):
    kind = request.param
    result = (
        make_nonlinear_result()
        if kind == "nlmer"
        else make_result("lmm" if kind == "lmer" else "poisson")
    )
    return kind, result


def valid_fit(result):
    fixed = result.phi if isinstance(result, NlmerResult) else result.beta
    return SimpleNamespace(
        beta=fixed.copy(),
        phi=fixed.copy(),
        theta=result.theta.copy(),
        sigma=0.7,
        converged=True,
        pirls_converged=True,
        pnls_converged=True,
    )


@contextmanager
def model_outputs(kind, result, *, simulations=None, refits=None):
    with ExitStack() as stack:
        if simulations is not None:
            target, name = (
                (NlmerResult, "simulate")
                if kind == "nlmer"
                else (bootstrap, f"_simulate_{kind}_components")
            )
            stack.enter_context(patch.object(target, name, side_effect=simulations))
        target, name = (
            (NlmerResult, "refit") if kind == "nlmer" else (bootstrap, f"_refit_{kind}_response")
        )
        refit = stack.enter_context(
            patch.object(target, name, side_effect=refits, return_value=valid_fit(result))
        )
        yield refit


def samples(result):
    fixed = (
        result.phi_samples
        if isinstance(result, bootstrap.NlmerBootstrapResult)
        else result.beta_samples
    )
    return [
        array for array in (fixed, result.theta_samples, result.sigma_samples) if array is not None
    ]


@pytest.mark.parametrize("jobs", [1, 2])
def test_mixed_failures_identify_rows_stages_and_messages(model, jobs):
    kind, result = model
    response = result.y if kind == "nlmer" else result.matrices.y
    nonconverged, invalid = valid_fit(result), valid_fit(result)
    flag = {"lmer": "converged", "glmer": "pirls_converged", "nlmer": "pnls_converged"}[kind]
    setattr(nonconverged, flag, False)
    nonconverged.message = "Iteration limit reached"
    invalid.theta[:] = np.nan
    completion_order = []

    def newest_first(pending, **kwargs):
        # Force out-of-order delivery independently of OS scheduling.
        future = max(pending, key=lambda item: item._result.index)
        completion_order.append(future._result.index)
        return {future}, pending - {future}

    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        patch.object(bootstrap, "wait", side_effect=newest_first),
        model_outputs(
            kind,
            result,
            simulations=[
                response,
                RuntimeError("draw failed"),
                response,
                response,
                response,
                response,
            ],
            refits=[
                valid_fit(result),
                np.linalg.LinAlgError("singular refit"),
                nonconverged,
                invalid,
                valid_fit(result),
            ],
        ),
    ):
        actual = bootMer(result, nsim=6, seed=123, n_jobs=jobs)
    if jobs == 2:
        assert completion_order != sorted(completion_order)
    assert actual.n_failed == len(actual.failures) == 4
    assert [(f.index, f.stage, f.exception_type) for f in actual.failures] == [
        (1, "simulation", "RuntimeError"),
        (2, "refit", "LinAlgError"),
        (3, "convergence", "ValueError"),
        (4, "validation", "ValueError"),
    ]
    assert [f.message for f in actual.failures[:2]] == ["draw failed", "singular refit"]
    assert flag in actual.failures[2].message
    assert "Iteration limit reached" in actual.failures[2].message
    assert "theta" in actual.failures[3].message
    for array in samples(actual):
        assert np.isnan(array[1:5]).all()
        assert np.isfinite(array[[0, 5]]).all()
    summary = actual.summary()
    assert "(4 failed)" in summary
    for stage in ("simulation", "refit", "convergence", "validation"):
        assert f"  {stage}: 1" in summary
    assert "draw failed" not in summary
    restored = pickle.loads(pickle.dumps(actual))
    assert restored.failures == actual.failures
    assert restored.summary() == summary


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize(
    "invalid", [None, 1.0, [1.0], [complex(1, 2)], [np.nan], ["bad"], [object()]]
)
def test_invalid_simulations_are_not_refitted_and_later_samples_survive(model, jobs, invalid):
    kind, result = model
    response = result.y if kind == "nlmer" else result.matrices.y
    if isinstance(invalid, list) and invalid != [1.0]:
        invalid = np.full(response.shape, invalid[0])
    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        model_outputs(kind, result, simulations=[invalid, response]) as refit,
    ):
        actual = bootMer(result, nsim=2, seed=2, n_jobs=jobs)
    assert refit.call_count == 1
    assert actual.n_failed == 1
    assert actual.failures[0].index == 0
    assert actual.failures[0].stage == "simulation"
    assert actual.failures[0].exception_type == "ValueError"
    assert "Simulated response" in actual.failures[0].message
    for array in samples(actual):
        assert np.isnan(array[0]).all()
        assert np.isfinite(array[1]).all()


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("field", ["fixed", "theta", "sigma", "converged"])
def test_missing_refit_attributes_are_classified(model, jobs, field):
    kind, result = model
    field = ("phi" if kind == "nlmer" else "beta") if field == "fixed" else field
    fitted = valid_fit(result)
    delattr(fitted, field)
    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        model_outputs(kind, result, refits=[fitted]),
    ):
        actual = bootMer(result, nsim=1, seed=1, n_jobs=jobs)
    if kind == "glmer" and field == "sigma":
        assert actual.n_failed == 0
        assert actual.failures == ()
        return
    (failure,) = actual.failures
    assert failure.stage == ("convergence" if field == "converged" else "validation")
    assert failure.exception_type == "AttributeError"
    assert field in failure.message


@pytest.mark.parametrize("jobs", [1, 2])
def test_success_and_legacy_results_have_empty_diagnostics(model, jobs):
    kind, result = model
    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        model_outputs(kind, result),
    ):
        actual = bootMer(result, nsim=3, seed=3, n_jobs=jobs)
    assert actual.n_failed == 0
    assert actual.failures == ()
    assert "Failed samples by stage" not in actual.summary()
    legacy = type(actual)(
        **{f.name: getattr(actual, f.name) for f in fields(actual) if f.name != "failures"}
    )
    # Old pickles have no per-instance failures attribute either.
    del legacy.failures
    assert legacy.failures == ()
    assert legacy.summary() == actual.summary()


class UnpicklableError(RuntimeError):
    def __reduce__(self):
        raise TypeError("Exception objects must not cross process boundaries")


class ExplodingRefitResult(NlmerResult):
    def refit(self, response):
        raise UnpicklableError("custom refit failed")


class ExplodingSimulationResult(NlmerResult):
    def simulate(self, **kwargs):
        raise UnpicklableError("custom simulation failed")


@pytest.mark.parametrize(
    "result_type,stage",
    [(ExplodingRefitResult, "refit"), (ExplodingSimulationResult, "simulation")],
)
def test_worker_processes_preserve_details_without_serializing_exceptions(result_type, stage):
    result = make_nonlinear_result()
    result = result_type(**{f.name: getattr(result, f.name) for f in fields(result)})
    parallel = bootMer(result, nsim=3, seed=42, n_jobs=2)
    serial = bootMer(result, nsim=3, seed=42)
    assert parallel.failures == serial.failures
    assert parallel.n_failed == 3
    assert [failure.index for failure in parallel.failures] == [0, 1, 2]
    assert all(failure.stage == stage for failure in parallel.failures)
    assert all(failure.exception_type == "UnpicklableError" for failure in parallel.failures)
    assert all(failure.message == f"custom {stage} failed" for failure in parallel.failures)
    assert pickle.loads(pickle.dumps(parallel)).failures == parallel.failures


class UnprintableError(Exception):
    def __str__(self):
        raise RuntimeError("broken formatting")


def test_unprintable_exceptions_still_leave_diagnostics(model):
    kind, result = model
    with model_outputs(kind, result, refits=[UnprintableError()]):
        actual = bootMer(result, nsim=1, seed=42)
    (failure,) = actual.failures
    assert failure.exception_type == "UnprintableError"
    assert failure.message == "Exception message could not be formatted"


@pytest.mark.parametrize("stage", ["simulation", "refit"])
def test_interruptions_propagate(model, stage):
    kind, result = model
    outputs = {"simulations" if stage == "simulation" else "refits": [KeyboardInterrupt()]}
    with model_outputs(kind, result, **outputs), pytest.raises(KeyboardInterrupt):
        bootMer(result, nsim=1, seed=42)


def test_failure_records_are_public_and_immutable():
    assert BootstrapFailure is bootstrap.BootstrapFailure
    failure = BootstrapFailure(4, "refit", "RuntimeError", "failed")
    with pytest.raises(FrozenInstanceError):
        failure.index = 5


@pytest.mark.parametrize("component", ["fixed", "theta"])
def test_shape_errors_identify_the_component_and_expected_dimensions(model, component):
    kind, result = model
    fitted = valid_fit(result)
    name = ("phi" if kind == "nlmer" else "beta") if component == "fixed" else "theta"
    expected = getattr(fitted, name).shape
    setattr(fitted, name, np.ones((1, 9)))
    with model_outputs(kind, result, refits=[fitted]):
        actual = bootMer(result, nsim=1, seed=42)
    (failure,) = actual.failures
    assert failure.stage == "validation"
    assert failure.message == f"{name} must have shape {expected}, got (1, 9)"


@pytest.mark.parametrize("model", ["glmer", "nlmer"], indirect=True)
def test_worker_preparation_errors_are_reported(model):
    kind, result = model
    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        patch.object(bootstrap, "deepcopy", side_effect=RuntimeError("copy failed")),
    ):
        actual = bootMer(result, nsim=2, seed=42, n_jobs=2)
    expected_stage = "simulation" if kind == "glmer" else "refit"
    assert actual.failures == tuple(
        BootstrapFailure(index, expected_stage, "RuntimeError", "copy failed") for index in range(2)
    )
    assert actual.n_failed == 2

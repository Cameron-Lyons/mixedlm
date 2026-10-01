"""Parallel nonlinear refits preserve the sequential simulation stream."""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import bootMer, nlmer
from mixedlm.inference import bootstrap
from mixedlm.models.nlmer import NlmerResult
from mixedlm.nlme.models import SSasymp
from numpy.testing import assert_array_equal

from tests.test_bootstrap_workers import ImmediateExecutor, PendingExecutor
from tests.test_nonlinear_simulation_streams import legacy_draws, make_result


def run(result, entry, count=3, seed=42, jobs=2):
    if entry == "bootMer":
        return bootMer(result, nsim=count, seed=seed, n_jobs=jobs)
    if entry == "confint":
        return result.confint(n_boot=count, seed=seed, n_jobs=jobs)
    return bootstrap.bootstrap_nlmer(result, n_boot=count, seed=seed, n_jobs=jobs)


def summarize_response(result, response, *, index=0):
    return bootstrap._BootstrapOutcome(
        index,
        np.array([np.mean(response), np.std(response), response[0]]),
        np.full_like(result.theta, np.var(response)),
        float(np.std(response)),
    )


def assert_samples_equal(first, second):
    assert first.n_failed == second.n_failed
    for name in ("phi_samples", "theta_samples", "sigma_samples"):
        assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("entry", ["direct", "bootMer", "confint"])
@pytest.mark.parametrize("jobs", [0, -2, True, np.bool_(False), 1.0, 2.5, "2", None])
def test_invalid_workers_fail_before_simulation_or_stream_consumption(entry, jobs):
    result = make_result()
    rng = np.random.default_rng(12)
    before = pickle.dumps(rng.bit_generator.state)
    with (
        patch.object(result, "simulate") as simulate,
        patch.object(result, "refit") as refit,
        patch.object(bootstrap, "ProcessPoolExecutor") as pool,
        pytest.raises((TypeError, ValueError), match="n_jobs"),
    ):
        run(result, entry, seed=rng, jobs=jobs)
    simulate.assert_not_called()
    refit.assert_not_called()
    pool.assert_not_called()
    assert pickle.dumps(rng.bit_generator.state) == before


@pytest.mark.parametrize("entry", ["direct", "bootMer", "confint"])
@pytest.mark.parametrize("jobs", [2, -1, np.int64(20)])
def test_public_entries_forward_and_cap_workers(entry, jobs):
    result = make_result()
    with (
        patch.object(bootstrap.os, "cpu_count", return_value=6),
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=summarize_response),
    ):
        actual = run(result, entry, count=3, jobs=jobs)
        serial = run(result, entry, count=3, jobs=1)
    pool = ImmediateExecutor.instances[-1]
    assert pool.workers == (2 if jobs == 2 else 3)
    assert pool.submitted == pool.consumed == 3
    assert pool.closed
    if entry == "confint":
        assert actual == serial
    else:
        assert_samples_equal(actual, serial)


@pytest.mark.parametrize("random_params", [(), (0,), (1, 0), (2, 0, 1)])
@pytest.mark.parametrize("jobs", [1, 2])
def test_seeded_bootstrap_preserves_the_legacy_draw_sequence(random_params, jobs):
    result = make_result(random_params)
    expected = legacy_draws(result, 7, 22)
    observed = []

    def record(result, response, *, index=0):
        observed.append(response.copy())
        return summarize_response(result, response, index=index)

    before = pickle.dumps(np.random.get_state())
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=record),
    ):
        actual = bootstrap.bootstrap_nlmer(result, n_boot=7, seed=22, n_jobs=jobs)
    assert actual.n_failed == 0
    assert_array_equal(np.column_stack(observed), expected)
    assert pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("factory", [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize("entry", ["direct", "bootMer", "confint"])
def test_streams_continue_across_calls_and_worker_counts(factory, entry):
    result = make_result()
    rng = factory(43)
    reference = factory(43)
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=summarize_response),
    ):
        first = run(result, entry, seed=rng, jobs=1)
        expected_first = run(result, entry, seed=reference, jobs=2)
        second = run(result, entry, seed=rng, jobs=2)
        expected_second = run(result, entry, seed=reference, jobs=1)
    if entry == "confint":
        assert first == expected_first
        assert second == expected_second
        assert first != second
    else:
        assert_samples_equal(first, expected_first)
        assert_samples_equal(second, expected_second)
        assert not np.array_equal(first.phi_samples, second.phi_samples)
    assert pickle.dumps(rng) == pickle.dumps(reference)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize(
    "field,value",
    [
        ("converged", False),
        ("pnls_converged", False),
        ("phi", np.array([np.nan, 2, 3])),
        ("phi", np.array([[1, 2, 3]])),
        ("theta", np.array([np.inf])),
        ("sigma", 0),
        ("sigma", 2j),
        ("exception", RuntimeError("fit failed")),
        ("missing", SimpleNamespace(phi=np.ones(3))),
    ],
)
def test_failed_refits_leave_whole_rows_missing(jobs, field, value):
    result = make_result()
    invalid = value if field in {"exception", "missing"} else replace(result, **{field: value})
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(NlmerResult, "refit", side_effect=[invalid, result]),
    ):
        actual = bootstrap.bootstrap_nlmer(result, n_boot=2, seed=42, n_jobs=jobs)
    assert actual.n_failed == 1
    for name in ("phi_samples", "theta_samples", "sigma_samples"):
        assert np.isnan(getattr(actual, name)[0]).all()
    assert_array_equal(actual.phi_samples[1], result.phi)
    assert_array_equal(actual.theta_samples[1], result.theta)
    assert actual.sigma_samples[1] == result.sigma
    assert np.isnan(list(actual.ci().values())).all()


@pytest.mark.parametrize("jobs", [1, 2])
def test_simulation_failure_does_not_skip_later_refits(jobs):
    result = make_result()
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(result, "simulate", side_effect=[ValueError("draw failed"), result.y]),
        patch.object(NlmerResult, "refit", return_value=result) as refit,
    ):
        actual = bootstrap.bootstrap_nlmer(result, n_boot=2, seed=42, n_jobs=jobs)
    refit.assert_called_once()
    assert actual.n_failed == 1
    assert np.isnan(actual.phi_samples[0]).all()
    assert_array_equal(actual.phi_samples[1], result.phi)


def test_parallel_queue_does_not_simulate_the_entire_bootstrap_ahead():
    result = make_result()
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", PendingExecutor),
        patch.object(result, "simulate", wraps=result.simulate) as simulate,
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=summarize_response),
    ):
        responses = bootstrap._nlmer_bootstrap_responses(result, 1000, np.random.RandomState(2))
        stream = bootstrap._parallel_bootstrap_tasks(
            bootstrap._nlmer_bootstrap_worker, (result,), responses, 2
        )
        assert next(stream).index == 0
        assert simulate.call_count == 4
        stream.close()
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed and pool.submitted == 4
    assert all(future.cancelled() for future in pool.tasks[1:])


def test_generator_owns_responses_before_asynchronous_serialization():
    result = make_result()
    reused = np.zeros_like(result.y)

    def simulate(**kwargs):
        reused[:] += 1
        return reused

    with patch.object(result, "simulate", side_effect=simulate):
        actual = list(bootstrap._nlmer_bootstrap_responses(result, 3, np.random.RandomState(2)))
    for index, response in actual:
        assert_array_equal(response, np.full_like(reused, index + 1))
        assert not np.shares_memory(response, reused)


def test_interrupted_simulation_cancels_queued_tasks():
    result = make_result()
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", PendingExecutor),
        patch.object(result, "simulate", side_effect=[result.y, KeyboardInterrupt]),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=summarize_response),
        pytest.raises(KeyboardInterrupt),
    ):
        bootstrap.bootstrap_nlmer(result, n_boot=1000, seed=42, n_jobs=2)
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed and pool.submitted == 1


def test_public_interrupt_closes_pool_with_retained_traceback():
    result = make_result()
    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(result, "simulate", wraps=result.simulate) as simulate,
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=summarize_response),
        patch("builtins.print", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt) as error,
    ):
        bootstrap.bootstrap_nlmer(result, n_boot=1000, seed=42, n_jobs=2, verbose=True)
    assert error.traceback is not None
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed and pool.submitted <= 104
    assert simulate.call_count <= 104


class PythonAsymptotic(SSasymp):
    """Importable custom model using the Python estimator in spawned workers."""


def fitted_model(custom=False):
    rng = np.random.default_rng(17)
    n = 40
    x = np.tile(np.linspace(0, 10, 10), 4)
    weights = np.linspace(1.0, 4.0, n)
    offsets = np.linspace(-2.0, 2.0, n)
    y = 10.0 + (3.0 - 10.0) * np.exp(-np.exp(-1.0) * x)
    y += np.repeat(rng.normal(0, 0.5, 4), 10) + offsets + rng.normal(0, 0.2, n)
    data = pd.DataFrame({"x": x, "y": y, "subject": np.repeat(list("abcd"), 10)})
    return nlmer(
        PythonAsymptotic() if custom else SSasymp(),
        data,
        x_var="x",
        y_var="y",
        group_var="subject",
        weights=weights,
        offset=offsets,
        random_params=["Asym"],
        pnls_maxiter=2000,
    )


@pytest.mark.parametrize("custom", [False, True])
def test_spawn_refits_match_serial_with_weights_offsets_and_inner_controls(custom):
    result = fitted_model(custom)
    assert result.converged and result.pnls_converged
    # The original frame is not needed by workers and may contain unpicklable data.
    result._data["unused"] = [lambda: None] * len(result.y)
    before = pickle.dumps(np.random.get_state())
    serial = bootstrap.bootstrap_nlmer(result, n_boot=4, seed=2026)
    context = multiprocessing.get_context("spawn")
    with patch.object(
        bootstrap, "ProcessPoolExecutor", partial(ProcessPoolExecutor, mp_context=context)
    ):
        parallel = bootstrap.bootstrap_nlmer(result, n_boot=4, seed=2026, n_jobs=2)
    assert_samples_equal(serial, parallel)
    assert serial.n_failed == 0
    assert pickle.dumps(np.random.get_state()) == before


def test_worker_custom_model_state_is_independent_between_refits():
    result = make_result()
    result.model.refit_count = 0

    def refit(self, response):
        self.model.refit_count += 1
        return replace(self, phi=np.full(3, self.model.refit_count, dtype=float))

    with patch.object(NlmerResult, "refit", refit):
        first = bootstrap._nlmer_bootstrap_worker((0, result.y, result))
        second = bootstrap._nlmer_bootstrap_worker((1, result.y, result))
    assert_array_equal(first.fixed, np.ones(3))
    assert_array_equal(second.fixed, first.fixed)
    assert result.model.refit_count == 0

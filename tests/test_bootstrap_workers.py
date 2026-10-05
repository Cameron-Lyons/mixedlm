import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.families import Poisson
from mixedlm.inference import bootstrap

from tests._bootstrap_helpers import ImmediateExecutor, PendingExecutor, fake_refit, make_result


def echo_task(args):
    index, seed, data = args
    return index, seed + data


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
@pytest.mark.parametrize("entry", ["direct", "wrapper"])
@pytest.mark.parametrize("jobs", [0, -2, True, np.bool_(False), 1.0, 2.5, "2", None])
def test_invalid_workers_fail_before_consuming_random_stream(kind, entry, jobs):
    result = make_result(kind)
    rng = np.random.default_rng(13)
    state = pickle.dumps(rng.bit_generator.state)
    fn = bootstrap.bootstrap_lmer if kind == "lmm" else bootstrap.bootstrap_glmer
    with (
        patch.object(bootstrap, "process_pool") as pool,
        pytest.raises((TypeError, ValueError), match="n_jobs"),
    ):
        if entry == "wrapper":
            bootstrap.bootMer(result, nsim=2, seed=rng, n_jobs=jobs)
        else:
            fn(result, n_boot=2, seed=rng, n_jobs=jobs)
    pool.assert_not_called()
    assert pickle.dumps(rng.bit_generator.state) == state


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_single_worker_runs_serially_without_a_pool(kind):
    result = make_result(kind)
    fn = bootstrap.bootstrap_lmer if kind == "lmm" else bootstrap.bootstrap_glmer
    with (
        patch.object(bootstrap, "process_pool") as pool,
        patch.object(bootstrap, "_refit_lmer_response", side_effect=fake_refit),
        patch.object(bootstrap, "_refit_glmer_response", side_effect=fake_refit),
    ):
        actual = fn(result, n_boot=1, seed=42, n_jobs=8)
        serial = fn(result, n_boot=1, seed=42, n_jobs=1)
    pool.assert_not_called()
    np.testing.assert_array_equal(actual.beta_samples, serial.beta_samples)


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
@pytest.mark.parametrize("count,jobs", [(3, 8), (11, 2), (1003, 3)])
def test_parallel_submission_is_bounded_and_sends_only_index_and_seed(kind, count, jobs):
    result = make_result(kind)
    fn = bootstrap.bootstrap_lmer if kind == "lmm" else bootstrap.bootstrap_glmer
    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        patch.object(bootstrap, "_refit_lmer_response", side_effect=fake_refit),
        patch.object(bootstrap, "_refit_glmer_response", side_effect=fake_refit),
    ):
        actual = fn(result, n_boot=count, seed=42, n_jobs=jobs)
        serial = fn(result, n_boot=count, seed=42, n_jobs=1)
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed
    assert pool.workers == min(count, jobs)
    assert pool.submitted == pool.consumed == count
    assert pool.peak_pending <= 2 * pool.workers
    assert all(
        len(args) == 1 and len(args[0]) == 2 and all(type(v) is int for v in args[0])
        for args in pool.tasks
    )
    np.testing.assert_array_equal(actual.beta_samples, serial.beta_samples)
    np.testing.assert_array_equal(actual.theta_samples, serial.theta_samples)
    assert actual.n_failed == serial.n_failed == 0


@pytest.mark.parametrize("failure", [False, True])
def test_early_exit_cancels_queued_work_and_closes_the_pool(failure):
    with (
        patch.object(bootstrap, "process_pool", PendingExecutor),
        patch.object(PendingExecutor, "failure", failure),
    ):
        stream = bootstrap._parallel_bootstrap_samples(echo_task, (100,), np.arange(500), 2)
        if failure:
            with pytest.raises(RuntimeError, match="pool task failed"):
                next(stream)
        else:
            assert next(stream) == (0, 100)
            stream.close()
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed and pool.submitted == 4
    assert all(future.cancelled() for future in pool.tasks[1:])


class SerializationProbe:
    count = 0

    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        type(self).count += 1
        return type(self), (self.value,)


def probe_task(args):
    index, seed, probe = args
    return index, int(seed + probe.value)


def test_worker_processes_receive_shared_data_once_each():
    SerializationProbe.count = 0
    actual = list(
        bootstrap._parallel_bootstrap_samples(
            probe_task, (SerializationProbe(17),), np.arange(17), 2
        )
    )
    assert sorted(actual) == [(i, i + 17) for i in range(17)]
    assert SerializationProbe.count == 2


def barrier_task(args):
    index, seed, value, barrier = args
    barrier.wait(timeout=10)
    return index, seed + value


def test_simultaneous_pools_keep_their_worker_data_separate():
    barrier = Barrier(4)

    def run(value):
        return sorted(
            bootstrap._parallel_bootstrap_samples(barrier_task, (value, barrier), np.arange(8), 2)
        )

    with (
        patch.object(bootstrap, "process_pool", ThreadPoolExecutor),
        ThreadPoolExecutor(max_workers=2) as callers,
    ):
        first = callers.submit(run, 100)
        second = callers.submit(run, 1000)
        assert first.result(timeout=20) == [(i, i + 100) for i in range(8)]
        assert second.result(timeout=20) == [(i, i + 1000) for i in range(8)]


class CountingPoisson(Poisson):
    count = 0

    def simulate(self, mu, rng=None):
        self.count += 1
        return super().simulate(mu, rng=rng) + self.count


def test_each_parallel_replicate_receives_a_fresh_custom_family():
    family = CountingPoisson()
    result = replace(make_result("poisson"), family=family)
    data = tuple(bootstrap._prepare_glmer_worker_data(result).values())
    with (
        patch.object(bootstrap, "process_pool", ThreadPoolExecutor),
        patch.object(bootstrap, "_refit_glmer_response", side_effect=fake_refit),
    ):
        actual = sorted(
            bootstrap._parallel_bootstrap_samples(
                bootstrap._glmer_bootstrap_worker, data, np.array([42, 42]), 1
            ),
            key=lambda row: row.index,
        )
    np.testing.assert_array_equal(actual[0].fixed, actual[1].fixed)
    np.testing.assert_array_equal(actual[0].theta, actual[1].theta)
    assert family.count == 0


def test_failed_submission_cancels_previously_queued_tasks():
    class FailedSubmitExecutor(PendingExecutor):
        def submit(self, fn, *args):
            if self.submitted == 2:
                raise RuntimeError("submission failed")
            return super().submit(fn, *args)

    with (
        patch.object(bootstrap, "process_pool", FailedSubmitExecutor),
        pytest.raises(RuntimeError, match="submission failed"),
    ):
        list(bootstrap._parallel_bootstrap_samples(echo_task, (100,), np.arange(500), 2))
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed and pool.submitted == 2
    assert pool.tasks[1].cancelled()


def test_interrupted_wait_cancels_queued_tasks():
    with (
        patch.object(bootstrap, "process_pool", PendingExecutor),
        patch.object(bootstrap, "wait", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        list(bootstrap._parallel_bootstrap_samples(echo_task, (100,), np.arange(500), 2))
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed and pool.submitted == 4
    assert all(future.cancelled() for future in pool.tasks[1:])


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_public_interrupt_closes_the_pool_even_when_the_traceback_is_retained(kind):
    result = make_result(kind)
    fn = bootstrap.bootstrap_lmer if kind == "lmm" else bootstrap.bootstrap_glmer
    with (
        patch.object(bootstrap, "process_pool", ImmediateExecutor),
        patch.object(bootstrap, "_refit_lmer_response", side_effect=fake_refit),
        patch.object(bootstrap, "_refit_glmer_response", side_effect=fake_refit),
        patch("builtins.print", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt) as error,
    ):
        fn(result, n_boot=1000, seed=42, n_jobs=2, verbose=True)
    assert error.traceback is not None
    pool = ImmediateExecutor.instances[-1]
    assert pool.closed
    assert pool.submitted <= 104

"""Simulation preparation is local to one call and does not change draw order."""

from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.estimation.nlmm import _grouped_observation_indices
from mixedlm.inference import bootstrap
from mixedlm.models.nlmer import NlmerResult, _NlmerSimulation
from numpy.testing import assert_array_equal

from tests.test_bootstrap_workers import ImmediateExecutor
from tests.test_nonlinear_bootstrap_workers import summarize_response
from tests.test_nonlinear_simulation_streams import legacy_draws, make_result


@pytest.mark.parametrize(
    "codes", [[], [0, 0, 0], [4, 1, 0, 1, 4, 0], [-2, 2, 6, 2, 0], [0, 1.5, 2]]
)
@pytest.mark.parametrize("count", [0, 1, 3, 6])
def test_group_rows_match_masks_including_empty_levels(codes, count):
    groups = np.asarray(codes)
    actual = _grouped_observation_indices(groups, n_groups=count)
    assert len(actual) == count
    for code, rows in enumerate(actual):
        assert_array_equal(rows, np.flatnonzero(groups == code))


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("random_params", [(), (0,), (1, 0)])
def test_bootstrap_prepares_groups_and_covariance_once(jobs, random_params):
    result = make_result(random_params)
    expected = legacy_draws(result, 5, 22)
    observed = []

    def record(result, response):
        observed.append(response.copy())
        return summarize_response(result, response)

    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=record),
        patch.object(_NlmerSimulation, "prepare", wraps=_NlmerSimulation.prepare) as prepare,
        patch(
            "mixedlm.models.nlmer._grouped_observation_indices",
            wraps=_grouped_observation_indices,
        ) as rows,
        patch.object(np.linalg, "svd", wraps=np.linalg.svd) as svd,
    ):
        actual = bootstrap.bootstrap_nlmer(result, n_boot=5, seed=22, n_jobs=jobs)
    assert prepare.call_count == rows.call_count == 1
    assert svd.call_count == bool(random_params)
    assert actual.n_failed == 0
    assert_array_equal(np.column_stack(observed), expected)


@pytest.mark.parametrize("jobs", [1, 2])
def test_bootstrap_refreshes_preparation_after_result_changes(jobs):
    result = make_result()
    observed = []

    def record(result, response):
        observed.append(response.copy())
        return summarize_response(result, response)

    with (
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=record),
    ):
        bootstrap.bootstrap_nlmer(result, n_boot=3, seed=72, n_jobs=jobs)
        observed.clear()
        result.phi[0] += 4
        result.theta *= 2
        result.sigma *= 0.5
        result._offset += 1.5
        result._weights *= 3
        result.groups = np.roll(result.groups, 2)
        result.x *= 1.2
        bootstrap.bootstrap_nlmer(result, n_boot=3, seed=72, n_jobs=jobs)
    assert_array_equal(np.column_stack(observed), legacy_draws(result, 3, 72))


@pytest.mark.parametrize("failure", ["setup", "prediction"])
def test_failed_preparation_or_draw_does_not_poison_later_responses(failure):
    result = make_result()
    original = np.linalg.svd if failure == "setup" else result.model.predict
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == (1 if failure == "setup" else 3):
            raise ValueError("failed once")
        return original(*args, **kwargs)

    target = np.linalg if failure == "setup" else result.model
    name = "svd" if failure == "setup" else "predict"
    expected, rng = [], np.random.RandomState(23)
    with patch.object(target, name, side_effect=fail_once):
        for _ in range(3):
            try:
                expected.append(result.simulate(seed=rng))
            except ValueError:
                expected.append(None)
    expected_state = rng.get_state()
    calls = 0
    rng = np.random.RandomState(23)
    with patch.object(target, name, side_effect=fail_once):
        actual = list(bootstrap._nlmer_bootstrap_responses(result, 3, rng))
    assert actual[0][1] is expected[0] is None
    for (_, response), reference in zip(actual[1:], expected[1:], strict=True):
        assert_array_equal(response, reference)
    assert_array_equal(rng.get_state()[1], expected_state[1])
    assert rng.get_state()[2:] == expected_state[2:]


def test_permanent_preparation_failure_leaves_all_samples_missing():
    result = make_result()
    result.theta[:] = np.nan
    rng = np.random.RandomState(3)
    before = rng.get_state()
    with patch.object(result, "refit") as refit:
        actual = bootstrap.bootstrap_nlmer(result, n_boot=3, seed=rng)
    assert actual.n_failed == 3
    assert np.isnan(actual.phi_samples).all()
    assert np.isnan(actual.theta_samples).all()
    assert np.isnan(actual.sigma_samples).all()
    refit.assert_not_called()
    assert_array_equal(rng.get_state()[1], before[1])
    assert rng.get_state()[2:] == before[2:]


class CustomSimulationResult(NlmerResult):
    def simulate(self, nsim=1, seed=None, use_re=True, re_form=None):
        return seed.standard_normal(len(self.y)) + np.arange(len(self.y))


@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_custom_simulation_overrides_are_used(override, jobs):
    result = make_result()
    if override == "subclass":
        result = CustomSimulationResult(**vars(result))
    calls = []

    def custom_simulate(self, **kwargs):
        calls.append(kwargs)
        return CustomSimulationResult.simulate(self, **kwargs)

    if override == "instance":
        patched = patch.object(
            result, "simulate", side_effect=lambda **kw: custom_simulate(result, **kw)
        )
    else:
        target = CustomSimulationResult if override == "subclass" else NlmerResult
        # Keep the subclass implementation callable while replacing the dispatch method.
        implementation = CustomSimulationResult.simulate

        def custom_simulate(self, **kwargs):
            calls.append(kwargs)
            return implementation(self, **kwargs)

        patched = patch.object(target, "simulate", custom_simulate)
    observed = []

    def record(result, response):
        observed.append(response.copy())
        return summarize_response(result, response)

    with (
        patched,
        patch.object(
            _NlmerSimulation, "prepare", side_effect=AssertionError("override bypassed")
        ) as prepare,
        patch.object(bootstrap, "ProcessPoolExecutor", ImmediateExecutor),
        patch.object(bootstrap, "_nlmer_bootstrap_refit", side_effect=record),
    ):
        actual = bootstrap.bootstrap_nlmer(result, n_boot=3, seed=24, n_jobs=jobs)
    assert actual.n_failed == 0
    assert len(calls) == 3
    prepare.assert_not_called()
    rng = np.random.RandomState(24)
    for response in observed:
        assert_array_equal(response, rng.standard_normal(len(result.y)) + np.arange(len(result.y)))


def test_fixed_only_bootstrap_still_evaluates_custom_predictions_each_draw():
    result = make_result(())
    original = result.model.predict
    with patch.object(result.model, "predict", wraps=original) as predict:
        responses = list(bootstrap._nlmer_bootstrap_responses(result, 3, np.random.RandomState(2)))
    assert predict.call_count == 3 * len(result.group_levels)
    assert all(response is not None for _, response in responses)


@pytest.mark.parametrize("inplace", [False, True])
def test_prepared_draws_preserve_multivariate_predictor_order_and_ownership(inplace):
    result = make_result()
    result.x = np.column_stack([result.x, result.x**2])
    original_x = result.x.copy()

    def predict(params, x):
        if inplace:
            x[:, 0] *= 2
        return params[0] + params[1] * x[:, 0] + params[2] * x[:, 1] + np.arange(len(x))

    with patch.object(result.model, "predict", side_effect=predict):
        expected = legacy_draws(result, 4, 25)
        actual = list(bootstrap._nlmer_bootstrap_responses(result, 4, np.random.RandomState(25)))
    assert_array_equal(np.column_stack([response for _, response in actual]), expected)
    assert_array_equal(result.x, original_x)

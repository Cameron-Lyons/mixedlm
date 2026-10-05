from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.inference.bootstrap import bootMer, bootstrap_nlmer

from tests._nlmm_models import legacy_draws, make_result


@pytest.mark.parametrize("random_params", [(), (0,), (1, 0), (2, 0, 1)])
@pytest.mark.parametrize("count", [1, 7])
@pytest.mark.parametrize("seed", [0, 47])
@pytest.mark.parametrize("mode", ["random", "fixed", "NA", "~0"])
def test_integer_seeds_preserve_legacy_draws(random_params, count, seed, mode):
    result = make_result(random_params)
    actual = result.simulate(nsim=count, seed=seed, use_re=mode != "fixed", re_form=mode)
    expected = legacy_draws(result, count, seed, include_re=mode == "random")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("factory", [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize("use_re", [True, False])
def test_stream_continues_across_calls_and_batch_sizes(factory, use_re):
    result = make_result()
    rng = factory(72)
    first = result.simulate(seed=rng, use_re=use_re)
    rest = result.simulate(nsim=6, seed=rng, use_re=use_re)
    together = result.simulate(nsim=7, seed=factory(72), use_re=use_re)
    np.testing.assert_array_equal(np.column_stack([first, rest]), together)


@pytest.mark.parametrize("seed", [None, 72])
@pytest.mark.parametrize("entry", ["simulate", "bootstrap", "bootMer", "confint"])
def test_calls_preserve_global_random_state(seed, entry):
    result = make_result()
    state = np.random.get_state()
    try:
        np.random.seed(901)
        np.random.standard_normal()  # Include a cached Gaussian in the state.
        before = np.random.get_state()
        with patch.object(result, "refit", return_value=result):
            if entry == "simulate":
                result.simulate(nsim=3, seed=seed)
            elif entry == "bootstrap":
                bootstrap_nlmer(result, n_boot=3, seed=seed)
            elif entry == "bootMer":
                bootMer(result, nsim=3, seed=seed)
            else:
                result.confint(n_boot=3, seed=seed)
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("entry", ["simulate", "bootstrap", "confint"])
def test_failures_preserve_global_random_state(entry):
    result = make_result()
    before = np.random.get_state()
    with patch.object(result.model, "predict", side_effect=RuntimeError("prediction failed")):
        if entry == "bootstrap":
            assert bootstrap_nlmer(result, n_boot=2, seed=37).n_failed == 2
        else:
            # The shared bootstrap path can either propagate or record draw failures.
            try:
                if entry == "simulate":
                    result.simulate(nsim=3, seed=37)
                else:
                    result.confint(n_boot=2, seed=37)
            except RuntimeError:
                pass
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("entry", ["bootstrap", "confint"])
def test_bootstrap_draws_preserve_legacy_sequence(entry):
    result = make_result()
    expected = legacy_draws(result, 5, 22)
    observed = []

    def refit(response):
        observed.append(response.copy())
        return replace(result, phi=result.phi + np.mean(response))

    with patch.object(result, "refit", side_effect=refit):
        if entry == "bootstrap":
            boot = bootstrap_nlmer(result, n_boot=5, seed=22)
            assert boot.n_failed == 0
        else:
            result.confint(n_boot=5, seed=22)
    np.testing.assert_array_equal(np.column_stack(observed), expected)


@pytest.mark.parametrize("count", [-1, 1.0, 1.5, True, np.bool_(False), "2", None])
def test_invalid_counts_fail_before_model_evaluation(count):
    result = make_result()
    expected = ValueError if count == -1 else TypeError
    with (
        patch.object(result.model, "predict") as predict,
        pytest.raises(expected, match="nsim must be a nonnegative integer"),
    ):
        result.simulate(nsim=count, seed=72)
    predict.assert_not_called()


@pytest.mark.parametrize("count", [0, np.int32(0), np.int64(0)])
def test_zero_draws_do_not_evaluate_model_or_advance_stream(count):
    result = make_result()
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    with patch.object(result.model, "predict") as predict:
        actual = result.simulate(nsim=count, seed=rng)
    assert actual.shape == (len(result.y), 0)
    assert actual.dtype == np.float64
    assert rng.bit_generator.state == state
    predict.assert_not_called()


@pytest.mark.parametrize("count", [np.int32(1), np.int64(3)])
def test_numpy_integer_draw_counts(count):
    result = make_result()
    np.testing.assert_array_equal(
        result.simulate(nsim=count, seed=17), legacy_draws(result, count, 17)
    )


@pytest.mark.parametrize("use_re", [True, False])
def test_setup_is_refreshed_after_result_changes(use_re):
    result = make_result()
    result.simulate(nsim=3, seed=72, use_re=use_re)
    result.phi[0] += 4.0
    result.theta *= 2.0
    result.sigma *= 0.5
    result._offset += 1.5
    result._weights *= 3.0
    result.groups = np.roll(result.groups, 2)
    result.x *= 1.2
    np.testing.assert_array_equal(
        result.simulate(nsim=3, seed=72, use_re=use_re),
        legacy_draws(result, 3, 72, include_re=use_re),
    )


@pytest.mark.parametrize("groups,per_group", [(0, 0), (3, 0)])
@pytest.mark.parametrize("count", [1, 4])
def test_empty_observations(groups, per_group, count):
    result = make_result(n_groups=groups, per_group=per_group)
    expected = (0,) if count == 1 else (0, count)
    assert result.simulate(nsim=count, seed=7).shape == expected


@pytest.mark.parametrize("inplace", [False, True])
def test_custom_model_keeps_group_row_order_and_two_dimensional_predictors(inplace):
    result = make_result()
    result.x = np.column_stack([result.x, result.x**2])
    original_x = result.x.copy()

    def predict(params, x):
        if inplace:
            x[:, 0] *= 2.0
        return params[0] + params[1] * x[:, 0] + params[2] * x[:, 1] + np.arange(len(x))

    with patch.object(result.model, "predict", side_effect=predict):
        np.testing.assert_array_equal(result.simulate(nsim=4, seed=25), legacy_draws(result, 4, 25))
    np.testing.assert_array_equal(result.x, original_x)

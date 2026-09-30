from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import bootCI, bootMer, nlme, nlmer
from mixedlm.inference.bootstrap import bootstrap_nlmer
from mixedlm.models.nlmer import NlmerResult


@pytest.fixture
def result():
    return NlmerResult(
        model=nlme.SSasymp(),
        group_var="subject",
        phi=np.array([200.0, 180.0, -3.0]),
        theta=np.array([2.0]),
        sigma=2.0,
        b=np.zeros((2, 1)),
        random_params=[0],
        deviance=0.0,
        converged=True,
        n_iter=1,
        x=np.tile(np.linspace(0, 10, 5), 2),
        y=np.zeros(10),
        groups=np.repeat([0, 1], 5),
        group_levels=["a", "b"],
    )


def run_bootstrap(result, entry, count):
    if entry == "confint":
        return result.confint(n_boot=count, seed=42)
    if entry == "bootMer":
        return bootMer(result, nsim=count, seed=42)
    return bootstrap_nlmer(result, n_boot=count, seed=42)


@pytest.mark.parametrize("entry", ["bootstrap_nlmer", "bootMer", "confint"])
@pytest.mark.parametrize("count", [True, np.bool_(False), 1.5, "2", None, -1, 0])
def test_invalid_counts_fail_before_simulation(result, entry, count):
    expected = ValueError if type(count) is int else TypeError
    with (
        patch.object(result, "simulate") as simulate,
        patch.object(result, "refit") as refit,
        pytest.raises(expected, match="positive integer"),
    ):
        run_bootstrap(result, entry, count)
    simulate.assert_not_called()
    refit.assert_not_called()


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_interval_level_fails_before_simulation(result, level):
    with (
        patch.object(result, "simulate") as simulate,
        patch.object(result, "refit") as refit,
        pytest.raises(ValueError, match="level"),
    ):
        result.confint(n_boot=2, level=level)
    simulate.assert_not_called()
    refit.assert_not_called()


@pytest.mark.parametrize("count", [1, np.int32(2), np.int64(3)])
def test_positive_integer_counts(result, count):
    with patch.object(result, "refit", return_value=result):
        boot = bootstrap_nlmer(result, n_boot=count, seed=42)
    assert boot.n_failed == 0
    assert boot.phi_samples.shape == (count, 3)


@pytest.mark.parametrize("entry", ["bootstrap_nlmer", "bootMer", "confint"])
def test_failed_refits_are_excluded_from_intervals(result, entry):
    first = replace(result, phi=np.array([10.0, 20.0, -4.0]))
    last = replace(result, phi=np.array([30.0, 60.0, -2.0]))
    with (
        patch.object(result, "simulate", return_value=result.y),
        patch.object(result, "refit", side_effect=[first, RuntimeError("refit failed"), last]),
    ):
        actual = run_bootstrap(result, entry, 3)
    if entry != "confint":
        assert actual.n_failed == 1
        assert np.isnan(actual.phi_samples[1]).all()
        assert np.isnan(actual.theta_samples[1]).all()
        assert np.isnan(actual.sigma_samples[1])
        actual = actual.ci()
    for column, name in enumerate(result.model.param_names):
        expected = np.percentile([first.phi[column], last.phi[column]], [2.5, 97.5])
        np.testing.assert_allclose(actual[name], expected)


@pytest.mark.parametrize("entry", ["bootstrap_nlmer", "confint"])
def test_simulation_failures_do_not_abort_later_replicates(result, entry):
    with (
        patch.object(result, "simulate", side_effect=[ValueError("draw failed"), result.y]),
        patch.object(result, "refit", return_value=result) as refit,
    ):
        actual = run_bootstrap(result, entry, 2)
    refit.assert_called_once()
    if entry != "confint":
        assert actual.n_failed == 1
        actual = actual.ci()
    assert actual == {
        name: (value, value)
        for name, value in zip(result.model.param_names, result.phi, strict=True)
    }


@pytest.mark.parametrize("entry", ["bootstrap_nlmer", "confint"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phi", np.array([1.0, np.nan, 3.0])),
        ("phi", np.array([1.0, np.inf, 3.0])),
        ("phi", np.array([1.0])),
        ("phi", np.array([[1.0, 2.0, 3.0]])),
        ("theta", np.array([np.inf])),
        ("theta", np.array([np.nan])),
        ("theta", np.array([1.0, 2.0])),
        ("sigma", np.nan),
        ("sigma", np.inf),
        ("sigma", np.array([2.0])),
        ("sigma", "invalid"),
    ],
)
def test_invalid_refits_leave_entire_sample_missing(result, entry, field, value):
    invalid = replace(result, **{field: value})
    with (
        patch.object(result, "simulate", return_value=result.y),
        patch.object(result, "refit", side_effect=[invalid, result]),
    ):
        actual = run_bootstrap(result, entry, 2)
    if entry != "confint":
        assert actual.n_failed == 1
        assert np.isnan(actual.phi_samples[0]).all()
        assert np.isnan(actual.theta_samples[0]).all()
        assert np.isnan(actual.sigma_samples[0])
        table = bootCI(actual, component="all")
        assert (table["n.success"] == 1).all()
        actual = actual.ci()
    assert actual == {
        name: (value, value)
        for name, value in zip(result.model.param_names, result.phi, strict=True)
    }


@pytest.mark.parametrize("entry", ["bootstrap_nlmer", "bootMer", "confint"])
def test_all_failed_samples_produce_missing_intervals(result, entry):
    with (
        patch.object(result, "simulate", return_value=result.y),
        patch.object(result, "refit", side_effect=RuntimeError("refit failed")),
    ):
        actual = run_bootstrap(result, entry, 3)
    if entry != "confint":
        assert actual.n_failed == 3
        actual = actual.ci()
    assert list(actual) == result.model.param_names
    assert np.isnan(list(actual.values())).all()


def test_confint_keeps_requested_order_and_ignores_unknown_names(result):
    with patch.object(result, "refit", return_value=result):
        actual = result.confint(parm=["lrc", "missing", "Asym"], n_boot=2, seed=42)
    assert list(actual) == ["lrc", "Asym"]
    assert actual == {"lrc": (-3.0, -3.0), "Asym": (200.0, 200.0)}


def test_confint_selects_single_parameter(result):
    with patch.object(result, "refit", return_value=result):
        actual = result.confint(parm="R0", n_boot=2, seed=42)
    assert actual == {"R0": (180.0, 180.0)}


def test_refit_missing_component_leaves_entire_sample_missing(result):
    missing = SimpleNamespace(phi=result.phi, theta=result.theta)
    with patch.object(result, "refit", side_effect=[missing, result]):
        boot = bootstrap_nlmer(result, n_boot=2, seed=42)
    assert boot.n_failed == 1
    assert np.isnan(boot.phi_samples[0]).all()
    assert np.isnan(boot.theta_samples[0]).all()
    assert np.isnan(boot.sigma_samples[0])


def test_seeded_weighted_offset_intervals_match_bootstrap_result():
    rng = np.random.default_rng(17)
    n = 40
    x = np.tile(np.linspace(0, 10, 10), 4)
    weights = np.linspace(1.0, 4.0, n)
    offsets = np.linspace(-2.0, 2.0, n)
    y = 10.0 + (3.0 - 10.0) * np.exp(-np.exp(-1.0) * x)
    y += np.repeat(rng.normal(0, 0.5, 4), 10) + offsets + rng.normal(0, 0.2, n)
    data = pd.DataFrame({"x": x, "y": y, "subject": np.repeat(list("abcd"), 10)})
    fit = nlmer(
        nlme.SSasymp(),
        data,
        x_var="x",
        y_var="y",
        group_var="subject",
        weights=weights,
        offset=offsets,
        random_params=["Asym"],
    )
    boot = bootstrap_nlmer(fit, n_boot=4, seed=2026)
    assert boot.n_failed == 0
    intervals = fit.confint(n_boot=4, seed=2026)
    assert intervals == boot.ci()
    np.testing.assert_array_equal(fit.weights(), weights)
    np.testing.assert_array_equal(fit.offset(), offsets)

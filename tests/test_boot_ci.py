from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import bootCI
from mixedlm.inference.bootstrap import BootstrapResult, NlmerBootstrapResult


@pytest.fixture
def bootstrap_result() -> BootstrapResult:
    return BootstrapResult(
        n_boot=5,
        beta_samples=np.array(
            [
                [0.8, 1.7],
                [0.9, 1.9],
                [1.1, 2.2],
                [1.3, 2.4],
                [np.nan, 2.1],
            ]
        ),
        theta_samples=np.array([[0.3], [0.4], [0.5], [0.6], [np.inf]]),
        sigma_samples=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        fixed_names=["intercept", "slope"],
        original_beta=np.array([1.0, 2.0]),
        original_theta=np.array([0.45]),
        original_sigma=1.0,
        n_failed=1,
    )


def test_boot_ci_percentile_matches_finite_quantiles(bootstrap_result: BootstrapResult) -> None:
    result = bootCI(bootstrap_result, parameters="intercept")

    expected = np.quantile([0.8, 0.9, 1.1, 1.3], [0.025, 0.975])
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "conf.low"] == pytest.approx(expected[0])
    assert result.loc[0, "conf.high"] == pytest.approx(expected[1])
    assert result.loc[0, "n.success"] == 4


def test_boot_ci_basic_reflects_percentile_interval(bootstrap_result: BootstrapResult) -> None:
    percentile = bootCI(bootstrap_result, method="percentile", parameters="slope")
    basic = bootCI(bootstrap_result, method="basic", parameters="slope")

    estimate = bootstrap_result.original_beta[1]
    assert basic.loc[0, "conf.low"] == pytest.approx(2 * estimate - percentile.loc[0, "conf.high"])
    assert basic.loc[0, "conf.high"] == pytest.approx(2 * estimate - percentile.loc[0, "conf.low"])


def test_boot_ci_normal_is_bias_corrected(bootstrap_result: BootstrapResult) -> None:
    result = bootCI(bootstrap_result, method="normal", parameters="intercept")
    samples = np.array([0.8, 0.9, 1.1, 1.3])
    bias = samples.mean() - 1.0
    center = 1.0 - bias

    assert result.loc[0, "bias"] == pytest.approx(bias)
    assert (result.loc[0, "conf.low"] + result.loc[0, "conf.high"]) / 2 == pytest.approx(center)
    assert result.loc[0, "std.error"] == pytest.approx(samples.std(ddof=1))


def test_boot_ci_all_components_and_methods(bootstrap_result: BootstrapResult) -> None:
    result = bootCI(
        bootstrap_result,
        component="all",
        method=["perc", "basic", "norm"],
    )

    assert set(result["parameter"]) == {"intercept", "slope", "theta[1]", "sigma"}
    assert set(result["component"]) == {"fixed", "theta", "sigma"}
    assert set(result["method"]) == {"percentile", "basic", "normal"}
    assert len(result) == 12
    assert result.attrs["n_boot"] == 5
    assert result.attrs["n_failed"] == 1


def test_boot_ci_filters_named_parameters_in_requested_component(
    bootstrap_result: BootstrapResult,
) -> None:
    result = bootCI(
        bootstrap_result,
        component="all",
        parameters=["sigma", "theta[1]"],
    )

    assert result["parameter"].tolist() == ["theta[1]", "sigma"]


def test_boot_ci_supports_nonlinear_results() -> None:
    result = NlmerBootstrapResult(
        n_boot=3,
        phi_samples=np.array([[1.0, 2.0], [1.1, 2.2], [0.9, 1.8]]),
        theta_samples=np.array([[0.3], [0.4], [0.5]]),
        sigma_samples=np.array([0.8, 1.0, 1.2]),
        param_names=["Asym", "rate"],
        original_phi=np.array([1.0, 2.0]),
        original_theta=np.array([0.4]),
        original_sigma=1.0,
        n_failed=0,
    )

    intervals = bootCI(result, component="all")

    assert set(intervals["parameter"]) == {"Asym", "rate", "theta[1]", "sigma"}


def test_boot_ci_skips_nonfinite_samples(bootstrap_result: BootstrapResult) -> None:
    result = bootCI(bootstrap_result, component="theta")

    assert result.loc[0, "n.success"] == 4
    assert np.isfinite(result.loc[0, "conf.low"])


def test_boot_ci_returns_undefined_statistics_when_all_replicates_fail() -> None:
    result = BootstrapResult(
        n_boot=3,
        beta_samples=np.full((3, 1), np.nan),
        theta_samples=np.full((3, 1), np.nan),
        sigma_samples=None,
        fixed_names=["x"],
        original_beta=np.array([1.0]),
        original_theta=np.array([0.5]),
        original_sigma=None,
        n_failed=3,
    )

    interval = bootCI(result, method="normal")

    assert interval.loc[0, "n.success"] == 0
    assert np.isnan(interval.loc[0, "std.error"])
    assert np.isnan(interval.loc[0, "conf.low"])
    assert np.isnan(interval.loc[0, "conf.high"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"level": 1.0}, "between 0 and 1"),
        ({"method": "bca"}, "Unknown method"),
        ({"method": []}, "at least one"),
        ({"method": 1}, "interval name"),
        ({"component": "random"}, "Unknown component"),
        ({"component": "sigma"}, "not available"),
        ({"parameters": []}, "cannot be empty"),
        ({"parameters": [1]}, "non-empty names"),
        ({"parameters": "missing"}, "Unknown parameter"),
    ],
)
def test_boot_ci_validates_requests(kwargs, message: str) -> None:
    result = BootstrapResult(
        n_boot=2,
        beta_samples=np.array([[1.0], [1.1]]),
        theta_samples=np.array([[0.5], [0.6]]),
        sigma_samples=None,
        fixed_names=["x"],
        original_beta=np.array([1.0]),
        original_theta=np.array([0.5]),
        original_sigma=None,
        n_failed=0,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        bootCI(result, **kwargs)

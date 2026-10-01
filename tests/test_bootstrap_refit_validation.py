from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import bootCI, glmer, lmer
from mixedlm.families import Binomial
from mixedlm.inference import bootstrap


@pytest.fixture(scope="module", params=["lmer", "glmer"])
def fitted(request):
    rng = np.random.default_rng(809)
    groups = np.repeat(np.arange(6), 12)
    x = rng.normal(size=len(groups))
    effect = rng.normal(0, 0.4, 6)[groups]
    eta = 0.2 + 0.5 * x + effect
    y = (
        eta + rng.normal(0, 0.3, len(x))
        if request.param == "lmer"
        else rng.binomial(1, 1 / (1 + np.exp(-eta)))
    )
    data = pd.DataFrame({"x": x, "y": y, "group": groups})
    result = (
        lmer("y ~ x + (1|group)", data)
        if request.param == "lmer"
        else glmer("y ~ x + (1|group)", data, family=Binomial())
    )
    return request.param, result


def valid_refit(result):
    return SimpleNamespace(
        beta=result.beta.copy(),
        theta=result.theta.copy(),
        sigma=0.5,
        converged=True,
        pirls_converged=True,
    )


INVALID = [
    ("converged", False),
    ("converged", "yes"),
    ("converged", None),
    ("beta", np.array([np.nan, 2.0])),
    ("beta", np.array([np.inf, 2.0])),
    ("beta", np.array([1.0])),
    ("beta", np.array([[1.0, 2.0]])),
    ("beta", np.array([1.0 + 1j, 2.0])),
    ("theta", np.array([np.nan])),
    ("theta", np.array([np.inf])),
    ("theta", np.array([1.0, 2.0])),
    ("theta", np.array([[1.0]])),
    ("theta", np.array([1.0 + 1j])),
]


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_refit_is_counted_and_leaves_the_whole_sample_missing(fitted, jobs, field, value):
    kind, result = fitted
    refit = valid_refit(result)
    setattr(refit, field, value)
    with (
        patch.object(bootstrap, f"_refit_{kind}_response", return_value=refit),
        patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
    ):
        actual = getattr(bootstrap, f"bootstrap_{kind}")(result, n_boot=1, seed=7, n_jobs=jobs)
    assert actual.n_failed == 1
    assert np.isnan(actual.beta_samples).all()
    assert np.isnan(actual.theta_samples).all()
    if actual.sigma_samples is not None:
        assert np.isnan(actual.sigma_samples).all()
    table = bootCI(actual, component="all", method=["percentile", "basic", "normal"])
    assert (table["n.success"] == 0).all()
    assert table[["conf.low", "conf.high"]].isna().all().all()


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("fitted", ["lmer"], indirect=True)
@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf, None, [0.5], 0.5 + 1j])
def test_invalid_residual_scale_cannot_leave_finite_coefficient_samples(fitted, jobs, sigma):
    kind, result = fitted
    refit = valid_refit(result)
    refit.sigma = sigma
    with (
        patch.object(bootstrap, "_refit_lmer_response", return_value=refit),
        patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
    ):
        actual = bootstrap.bootstrap_lmer(result, n_boot=1, seed=7, n_jobs=jobs)
    assert actual.n_failed == 1
    assert np.isnan(actual.beta_samples).all()
    assert np.isnan(actual.theta_samples).all()
    assert np.isnan(actual.sigma_samples).all()


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("fitted", ["glmer"], indirect=True)
def test_inner_glmm_nonconvergence_is_rejected_even_if_outer_status_is_true(fitted, jobs):
    kind, result = fitted
    refit = valid_refit(result)
    refit.pirls_converged = False
    with (
        patch.object(bootstrap, "_refit_glmer_response", return_value=refit),
        patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
    ):
        actual = bootstrap.bootstrap_glmer(result, n_boot=1, seed=7, n_jobs=jobs)
    assert actual.n_failed == 1
    assert np.isnan(actual.beta_samples).all()
    assert np.isnan(actual.theta_samples).all()


@pytest.mark.parametrize("jobs", [1, 2])
def test_later_valid_refits_survive_an_earlier_failure(fitted, jobs):
    kind, result = fitted
    first = valid_refit(result)
    last = valid_refit(result)
    first.beta = np.array([1.0, 3.0])
    last.beta = np.array([5.0, 7.0])
    with (
        patch.object(
            bootstrap,
            f"_refit_{kind}_response",
            side_effect=[first, RuntimeError("refit failed"), last],
        ),
        patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
    ):
        actual = getattr(bootstrap, f"bootstrap_{kind}")(result, n_boot=3, seed=7, n_jobs=jobs)
    assert actual.n_failed == 1
    good = np.isfinite(actual.beta_samples).all(axis=1)
    assert good.sum() == 2
    np.testing.assert_allclose(np.sort(actual.beta_samples[good], axis=0), [[1.0, 3.0], [5.0, 7.0]])
    assert np.isnan(actual.theta_samples[~good]).all()
    for index, name in enumerate(result.matrices.fixed_names):
        np.testing.assert_allclose(
            actual.ci()[name], np.quantile([first.beta[index], last.beta[index]], [0.025, 0.975])
        )


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("method", ["percentile", "basic", "normal"])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_intervals_require_two_valid_samples_per_component(nonlinear, method, count):
    samples = np.array([2.0, 4.0, np.nan, np.inf])
    samples[count:2] = np.nan
    common = dict(
        n_boot=4,
        theta_samples=samples[:, None].copy(),
        sigma_samples=samples.copy(),
        original_theta=np.array([3.0]),
        original_sigma=3.0,
        n_failed=4 - count,
    )
    if nonlinear:
        result = bootstrap.NlmerBootstrapResult(
            phi_samples=np.column_stack([samples, [1.0, 2.0, 3.0, 4.0]]),
            param_names=["a", "b"],
            original_phi=np.array([3.0, 2.5]),
            **common,
        )
    else:
        result = bootstrap.BootstrapResult(
            beta_samples=np.column_stack([samples, [1.0, 2.0, 3.0, 4.0]]),
            fixed_names=["a", "b"],
            original_beta=np.array([3.0, 2.5]),
            **common,
        )
    ci = result.ci(method=method)
    table = bootCI(result, component="all", method=method)
    sparse = table[table.parameter != "b"]
    assert (sparse["n.success"] == count).all()
    assert np.isfinite(ci["b"]).all()
    if count < 2:
        assert np.isnan(ci["a"]).all()
        assert sparse[["std.error", "conf.low", "conf.high"]].isna().all().all()
        if count == 1:
            np.testing.assert_array_equal(sparse["mean"], 2.0)
    else:
        np.testing.assert_allclose(
            ci["a"], sparse.iloc[0][["conf.low", "conf.high"]].to_numpy(dtype=float)
        )
        assert np.isfinite(sparse[["std.error", "conf.low", "conf.high"]]).all().all()


@pytest.mark.parametrize("fitted", ["glmer"], indirect=True)
def test_real_unfinished_glmm_refits_are_excluded_in_serial_and_worker_processes(fitted):
    from dataclasses import replace

    _, result = fitted
    limited = replace(result, pirls_maxiter=1, pirls_tol=1e-12)
    serial = bootstrap.bootstrap_glmer(limited, n_boot=3, seed=14, n_jobs=1)
    parallel = bootstrap.bootstrap_glmer(limited, n_boot=3, seed=14, n_jobs=2)
    assert serial.n_failed == parallel.n_failed == 3
    assert serial.failures == parallel.failures
    assert all(f.stage == "convergence" for f in serial.failures)
    assert all("pirls_converged" in f.message for f in serial.failures)
    for actual in (serial, parallel):
        assert np.isnan(actual.beta_samples).all()
        assert np.isnan(actual.theta_samples).all()
        assert np.isnan(list(actual.ci().values())).all()


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("missing", ["converged", "beta", "theta"])
def test_missing_refit_components_are_counted_as_failed_samples(fitted, jobs, missing):
    kind, result = fitted
    refit = valid_refit(result)
    delattr(refit, missing)
    with (
        patch.object(bootstrap, f"_refit_{kind}_response", return_value=refit),
        patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
    ):
        actual = getattr(bootstrap, f"bootstrap_{kind}")(result, n_boot=1, seed=7, n_jobs=jobs)
    assert actual.n_failed == 1
    assert np.isnan(actual.beta_samples).all()
    assert np.isnan(actual.theta_samples).all()


@pytest.mark.parametrize("jobs", [1, 2])
def test_numpy_boolean_success_and_zero_variance_components_remain_valid(fitted, jobs):
    kind, result = fitted
    refit = valid_refit(result)
    refit.converged = refit.pirls_converged = np.bool_(True)
    refit.theta[:] = 0.0
    with (
        patch.object(bootstrap, f"_refit_{kind}_response", return_value=refit),
        patch.object(bootstrap, "ProcessPoolExecutor", ThreadPoolExecutor),
    ):
        actual = getattr(bootstrap, f"bootstrap_{kind}")(result, n_boot=2, seed=7, n_jobs=jobs)
    assert actual.n_failed == 0
    assert np.isfinite(actual.beta_samples).all()
    assert (actual.theta_samples == 0.0).all()
    for method in ["percentile", "basic", "normal"]:
        table = bootCI(actual, component="theta", method=method)
        assert np.isfinite(table[["conf.low", "conf.high"]]).all().all()
        assert (table["conf.low"] == table["conf.high"]).all()

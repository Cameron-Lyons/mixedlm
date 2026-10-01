from __future__ import annotations

import pickle
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm.families import Binomial, Gamma, Gaussian, InverseGaussian, NegativeBinomial, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.inference import bootstrap as bootstrap_module
from mixedlm.inference.bootstrap import bootMer, bootstrap_glmer, bootstrap_lmer
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult

KINDS = [
    "lmm",
    "binomial",
    "grouped_binomial",
    "poisson",
    "gaussian",
    "gamma",
    "inverse_gaussian",
    "negative_binomial",
]
PATHS = ["single", "fixed_batch", "native_batch", "python_batch"]


def make_result(kind="lmm", n_groups=4, n_per_group=10):
    x = np.tile(np.linspace(-0.5, 0.5, n_per_group), n_groups)
    data = pd.DataFrame(
        {"y": np.ones(len(x)), "x": x, "group": np.repeat(np.arange(n_groups), n_per_group)}
    )
    formula = parse_formula("y ~ x + (1 | group)")
    matrices = build_model_matrices(formula, data)
    matrices = replace(
        matrices, weights=np.linspace(0.5, 2.0, len(x)), offset=np.linspace(0.1, 0.3, len(x))
    )
    if kind == "grouped_binomial":
        matrices = replace(matrices, trials=np.full(len(x), 9.0))
    common = dict(
        formula=formula,
        matrices=matrices,
        beta=np.array([0.2, 0.3]),
        theta=np.array([0.3]),
        u=np.zeros(n_groups),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    if kind == "lmm":
        return LmerResult(**common, sigma=0.7, REML=False)
    families = dict(
        binomial=Binomial,
        grouped_binomial=Binomial,
        poisson=Poisson,
        gaussian=Gaussian,
        gamma=Gamma,
        inverse_gaussian=InverseGaussian,
        negative_binomial=NegativeBinomial,
    )
    return GlmerResult(**common, family=families[kind](), nAGQ=1)


def simulate_path(result, path, seed):
    if path == "native_batch":
        pytest.importorskip("mixedlm._rust")
    context = (
        patch.dict("sys.modules", {"mixedlm._rust": None})
        if path == "python_batch"
        else nullcontext()
    )
    with context:
        return result.simulate(
            nsim=1 if path == "single" else 4, seed=seed, use_re=path != "fixed_batch"
        )


def assert_global_state(state):
    for before, after in zip(state, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seed", [None, 41])
def test_simulation_preserves_global_state(kind, path, seed):
    result = make_result(kind)
    np.random.seed(12345)
    np.random.standard_normal(3)  # Include the cached Gaussian draw in state comparisons.
    state = np.random.get_state()
    actual = simulate_path(result, path, seed)
    assert_global_state(state)
    assert actual.shape == ((40,) if path == "single" else (40, 4))
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("kind", ["lmm", "poisson", "grouped_binomial", "inverse_gaussian"])
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("factory", [np.random.RandomState, np.random.default_rng])
def test_streams_continue_reproducibly_across_calls(kind, path, factory):
    result = make_result(kind)
    rng = factory(42)
    state = np.random.get_state()
    first = simulate_path(result, path, rng)
    second = simulate_path(result, path, rng)
    assert_global_state(state)
    reference = factory(42)
    np.testing.assert_array_equal(first, simulate_path(result, path, reference))
    np.testing.assert_array_equal(second, simulate_path(result, path, reference))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, np.nan, "3", None])
def test_invalid_simulation_count_does_not_consume_stream(kind, value):
    result = make_result(kind)
    rng = np.random.RandomState(12)
    before = pickle.dumps(rng.get_state())
    with pytest.raises(ValueError, match="nsim must be a positive integer"):
        result.simulate(nsim=value, seed=rng)
    assert pickle.dumps(rng.get_state()) == before


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_numpy_integer_count_and_seed_are_supported(kind):
    result = make_result(kind)
    np.testing.assert_array_equal(
        result.simulate(nsim=np.int64(3), seed=np.int64(42)), result.simulate(nsim=3, seed=42)
    )


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_simulation_failure_preserves_global_state(kind):
    result = make_result(kind)
    state = np.random.get_state()
    with (
        patch.object(result, "_simulate_once", side_effect=RuntimeError("simulation failed")),
        pytest.raises(RuntimeError, match="simulation failed"),
    ):
        result.simulate(seed=42)
    assert_global_state(state)


@pytest.mark.parametrize("path", PATHS)
def test_custom_family_receives_the_supplied_stream(path):
    class RecordingGaussian(Gaussian):
        def simulate(self, mu, rng=None):
            self.received = rng
            return super().simulate(mu, rng=rng)

    result = make_result("gaussian")
    family = RecordingGaussian()
    result = replace(result, family=family)
    rng = np.random.default_rng(123)
    simulate_path(result, path, rng)
    assert family.received is rng


def fake_refit(matrices, response, theta, *args):
    return SimpleNamespace(
        beta=np.array([np.mean(response), np.std(response)]),
        theta=np.array([np.var(response)]),
        sigma=float(np.std(response)),
        converged=True,
        pirls_converged=True,
    )


def run_bootstrap(result, seed, n_jobs=1):
    method = bootstrap_lmer if isinstance(result, LmerResult) else bootstrap_glmer
    return method(result, n_boot=3, seed=seed, n_jobs=n_jobs)


@pytest.mark.parametrize("kind", ["lmm", "poisson", "grouped_binomial"])
@pytest.mark.parametrize("seed_kind", ["none", "integer", "randomstate", "generator"])
def test_bootstrap_preserves_global_state_and_accepts_streams(kind, seed_kind):
    seeds = {
        "none": None,
        "integer": 42,
        "randomstate": np.random.RandomState(42),
        "generator": np.random.default_rng(42),
    }
    result = make_result(kind)
    state = np.random.get_state()
    with (
        patch.object(bootstrap_module, "_refit_lmer_response", side_effect=fake_refit),
        patch.object(bootstrap_module, "_refit_glmer_response", side_effect=fake_refit),
    ):
        boot = run_bootstrap(result, seeds[seed_kind])
    assert_global_state(state)
    assert boot.n_failed == 0
    assert np.isfinite(boot.beta_samples).all()


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
@pytest.mark.parametrize("factory", [np.random.RandomState, np.random.default_rng])
def test_bootstrap_stream_continuation_and_wrapper(kind, factory):
    result = make_result(kind)
    rng = factory(42)
    reference = factory(42)
    with (
        patch.object(bootstrap_module, "_refit_lmer_response", side_effect=fake_refit),
        patch.object(bootstrap_module, "_refit_glmer_response", side_effect=fake_refit),
    ):
        first = run_bootstrap(result, rng)
        second = run_bootstrap(result, rng)
        np.testing.assert_array_equal(
            first.beta_samples, run_bootstrap(result, reference).beta_samples
        )
        np.testing.assert_array_equal(
            second.beta_samples, bootMer(result, nsim=3, seed=reference).beta_samples
        )
    assert not np.array_equal(first.beta_samples, second.beta_samples)


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, np.nan, "3", None])
def test_invalid_bootstrap_count_does_not_consume_stream(kind, value):
    result = make_result(kind)
    rng = np.random.default_rng(12)
    before = pickle.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        bootMer(result, nsim=value, seed=rng)
    assert pickle.dumps(rng.bit_generator.state) == before


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_bootstrap_failures_preserve_global_state(kind):
    result = make_result(kind)
    state = np.random.get_state()
    with (
        patch.object(
            bootstrap_module, "_refit_lmer_response", side_effect=RuntimeError("failed refit")
        ),
        patch.object(
            bootstrap_module, "_refit_glmer_response", side_effect=RuntimeError("failed refit")
        ),
    ):
        boot = run_bootstrap(result, 42)
    assert_global_state(state)
    assert boot.n_failed == 3
    assert np.isnan(boot.beta_samples).all()


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_actual_parallel_refits_match_serial_and_preserve_global_state(kind):
    result = make_result(kind, n_groups=8)
    state = np.random.get_state()
    serial = run_bootstrap(result, 42)
    parallel = run_bootstrap(result, 42, n_jobs=2)
    assert_global_state(state)
    assert serial.n_failed == parallel.n_failed == 0
    np.testing.assert_array_equal(serial.beta_samples, parallel.beta_samples)
    np.testing.assert_array_equal(serial.theta_samples, parallel.theta_samples)
    if kind == "lmm":
        np.testing.assert_array_equal(serial.sigma_samples, parallel.sigma_samples)


@pytest.mark.parametrize("kind", ["lmm", "poisson"])
def test_bootstrap_workers_preserve_their_process_random_state(kind):
    result = make_result(kind, n_groups=8)
    if kind == "lmm":
        worker = bootstrap_module._lmer_bootstrap_worker
        payload = bootstrap_module._prepare_lmer_worker_data(result)
    else:
        worker = bootstrap_module._glmer_bootstrap_worker
        payload = bootstrap_module._prepare_glmer_worker_data(result)
    state = np.random.get_state()
    output = worker((0, 42, *payload.values()))
    assert_global_state(state)
    assert output[1] is not None

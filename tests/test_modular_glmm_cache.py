"""Modular joint likelihoods reuse preparation while honoring current controls."""

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.estimation import laplace
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from numpy.testing import assert_array_equal

from tests.test_glmm_serialization import CustomPoisson, make_object


def fresh_value(devfun, parameters):
    optimizer = devfun.optimizer
    return JointGLMMObjective(
        optimizer.matrices,
        optimizer.family,
        optimizer.nAGQ,
        pirls_maxiter=optimizer.pirls_maxiter,
        pirls_tol=optimizer.pirls_tol,
    )(parameters)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("order", [0, 1, 7])
def test_repeated_joint_calls_prepare_once_and_match_fresh_solves(native, kind, order, monkeypatch):
    if native:
        pytest.importorskip("mixedlm._rust")
    monkeypatch.setattr(laplace, "_HAS_RUST", native)
    devfun, parameters = make_object("modular_joint", kind, order)
    with patch.object(
        devfun.optimizer, "joint_objective", wraps=devfun.optimizer.joint_objective
    ) as prepare:
        assert devfun._joint_cache is None
        # Covariance-only calls do not pay for joint preparation.
        assert np.isfinite(devfun(parameters[: devfun.parsed.n_theta]))
        assert prepare.call_count == 0
        for scale in [1.0, 0.0, 1.8, 1.0]:
            current = parameters * scale
            assert devfun(current) == fresh_value(devfun, current)
        assert prepare.call_count == 1


@pytest.mark.parametrize("native", [False, True])
def test_changed_settings_refresh_preparation_and_restore_original_values(native, monkeypatch):
    if native:
        pytest.importorskip("mixedlm._rust")
    monkeypatch.setattr(laplace, "_HAS_RUST", native)
    devfun, parameters = make_object("modular_joint")
    changes = [
        (1, 100, 1e-10),
        (7, 100, 1e-10),
        (7, 1, 1e-10),
        (7, 1, 1e6),
        (0, None, 1e-6),
        (1, 100, 1e-10),
    ]
    with patch.object(
        devfun.optimizer, "joint_objective", wraps=devfun.optimizer.joint_objective
    ) as prepare:
        for count, settings in enumerate(changes, 1):
            devfun.optimizer.nAGQ, devfun.optimizer.pirls_maxiter, devfun.optimizer.pirls_tol = (
                settings
            )
            expected = fresh_value(devfun, parameters)
            assert devfun(parameters) == expected
            assert devfun(parameters) == expected
            assert prepare.call_count == count


@pytest.mark.parametrize(
    "name,valid,invalid",
    [
        ("nAGQ", 1, True),
        ("nAGQ", 1, 1.0),
        ("nAGQ", 1, -1),
        ("pirls_maxiter", 1, True),
        ("pirls_maxiter", 1, 1.0),
        ("pirls_maxiter", 1, 0),
        ("pirls_tol", 1.0, True),
        ("pirls_tol", 1.0, np.nan),
        ("pirls_tol", 1.0, -1.0),
    ],
)
def test_invalid_changed_controls_do_not_reuse_or_damage_cached_objective(name, valid, invalid):
    devfun, parameters = make_object("modular_joint")
    setattr(devfun.optimizer, name, valid)
    expected = devfun(parameters)
    cached = devfun._joint_cache
    setattr(devfun.optimizer, name, invalid)
    with pytest.raises(ValueError):
        devfun(parameters)
    setattr(devfun.optimizer, name, valid)
    assert devfun(parameters) == expected
    assert devfun._joint_cache is cached


def test_changed_order_still_rejects_unsupported_structures():
    devfun, parameters = make_object("modular_joint", layout="crossed")
    assert np.isfinite(devfun(parameters))
    devfun.optimizer.nAGQ = 7
    with pytest.raises(ValueError, match="one random-effect term"):
        devfun(parameters)


def test_replacing_optimizer_refreshes_joint_objective():
    devfun, parameters = make_object("modular_joint")
    assert np.isfinite(devfun(parameters))
    cached = devfun._joint_cache
    replacement, _ = make_object("modular_joint", kind="binomial")
    devfun.optimizer = replacement.optimizer
    devfun.parsed = replacement.parsed
    assert devfun(parameters) == fresh_value(devfun, parameters)
    assert devfun._joint_cache is not cached
    assert devfun._joint_cache[0] is replacement.optimizer


@pytest.mark.parametrize("kind", ["native", "custom"])
@pytest.mark.parametrize("method", ["deepcopy", "pickle"])
def test_warm_cache_survives_copying_with_shared_input_aliases(kind, method):
    family = CustomPoisson() if kind == "custom" else None
    devfun, parameters = make_object("modular_joint", order=7, family=family)
    expected = devfun(parameters)
    restored = copy.deepcopy(devfun) if method == "deepcopy" else pickle.loads(pickle.dumps(devfun))
    assert restored._joint_cache[0] is restored.optimizer
    assert restored._joint_cache[1].matrices is restored.optimizer.matrices
    assert restored._joint_cache[1].family is restored.optimizer.family
    assert restored._joint_cache is not devfun._joint_cache
    with patch.object(
        restored.optimizer, "joint_objective", side_effect=AssertionError("cache lost")
    ):
        assert restored(parameters) == expected


def test_legacy_deviance_state_without_cache_remains_callable():
    devfun, parameters = make_object("modular_joint")
    expected = devfun(parameters)
    del devfun._joint_cache
    restored = pickle.loads(pickle.dumps(devfun))
    assert restored._joint_cache is None
    assert restored(parameters) == expected


def test_concurrent_calls_have_independent_modes_and_offsets():
    devfun, parameters = make_object("modular_joint", order=7)
    values = [parameters * scale for scale in [1.0, 0.0, 1.8, 0.3]] * 3
    expected = [fresh_value(devfun, value) for value in values]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(devfun, values))
    assert_array_equal(actual, expected)


def test_failed_parameter_evaluation_does_not_damage_cache():
    devfun, parameters = make_object("modular_joint", order=7)
    expected = devfun(parameters)
    with pytest.raises(ValueError, match="joint parameters"):
        devfun(np.full_like(parameters, np.nan))
    assert devfun(parameters) == expected

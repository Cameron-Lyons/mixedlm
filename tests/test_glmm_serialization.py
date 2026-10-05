"""GLMM callables retain their inputs and controls across copies and processes."""

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest
from mixedlm.estimation import laplace
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from numpy.testing import assert_allclose, assert_array_equal

from tests._glmm_oracles import CustomPoisson, make_glmm_objective


def evaluate(obj, parameters):
    if isinstance(obj, laplace.GLMMOptimizer):
        return (obj.objective(parameters), *obj._final_evaluation_with_status(parameters)[1:])
    if isinstance(obj, JointGLMMObjective):
        return obj.evaluate(parameters)
    return (obj(parameters),)


def owner(obj):
    return getattr(obj, "optimizer", obj)


def round_trip(obj, method):
    if method == "copy":
        return copy.copy(obj)
    if method == "deepcopy":
        return copy.deepcopy(obj)
    return pickle.loads(pickle.dumps(obj, protocol=method))


def assert_same(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("target", ["optimizer", "joint", "modular", "modular_joint"])
@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("order", [0, 1, 7])
@pytest.mark.parametrize("method", ["deepcopy", 4, 5])
def test_glmm_copy_and_pickle_preserve_evaluations_and_controls(target, kind, order, method):
    obj, parameters = make_glmm_objective(target, kind, order)
    obj.user_metadata = {"labels": ["fit"]}
    expected = evaluate(obj, parameters)
    original_native = owner(obj)._native_problem
    restored = round_trip(obj, method)

    assert restored is not obj
    assert restored.user_metadata == obj.user_metadata
    assert restored.user_metadata is not obj.user_metadata
    assert owner(restored).matrices is not owner(obj).matrices
    assert owner(restored).nAGQ == order
    assert owner(restored).pirls_maxiter == 100
    assert owner(restored).pirls_tol == 1e-10
    if original_native is not None:
        assert owner(restored)._native_problem is not original_native
        assert owner(restored)._native_problem is not None
    if target.startswith("modular"):
        assert restored.parsed.matrices is restored.optimizer.matrices
        assert restored.parsed.family is restored.optimizer.family
        assert restored.control == obj.control
        assert restored.optimizer.nAGQ0initStep is False
    elif target == "joint":
        assert restored.mode_matrices.y is restored.matrices.y
        assert restored.mode_matrices.Z is restored.matrices.Z
        assert restored.bounds == obj.bounds
    assert_same(evaluate(restored, parameters), expected)
    assert_same(evaluate(obj, parameters), expected)
    assert owner(obj)._native_problem is original_native


@pytest.mark.parametrize("target", ["optimizer", "joint"])
@pytest.mark.parametrize("layout", ["slope", "crossed", "fixed_only", "mode_only"])
@pytest.mark.parametrize("maxiter", [1, 100])
def test_round_trip_preserves_other_layouts_and_limited_solves(target, layout, maxiter):
    obj, parameters = make_glmm_objective(target, layout=layout, maxiter=maxiter)
    restored = round_trip(obj, 5)
    assert_same(evaluate(restored, parameters), evaluate(obj, parameters))
    assert owner(restored).pirls_maxiter == maxiter


@pytest.mark.parametrize("target", ["optimizer", "joint"])
def test_shallow_copy_keeps_shared_python_inputs(target):
    obj, parameters = make_glmm_objective(target)
    restored = copy.copy(obj)
    assert restored is not obj
    assert restored.matrices is obj.matrices
    assert restored.family is obj.family
    assert_same(evaluate(restored, parameters), evaluate(obj, parameters))


@pytest.mark.parametrize("target", ["optimizer", "joint", "modular_joint"])
@pytest.mark.parametrize("method", ["deepcopy", 5])
def test_custom_family_keeps_python_route_after_round_trip(target, method):
    obj, parameters = make_glmm_objective(target, order=7, family=CustomPoisson())
    restored = round_trip(obj, method)
    assert type(owner(restored).family) is CustomPoisson
    assert owner(restored)._native_problem is None
    assert_same(evaluate(restored, parameters), evaluate(obj, parameters))


@pytest.mark.parametrize("target", ["optimizer", "joint", "modular_joint"])
@pytest.mark.parametrize("native_on_restore", [False, True])
def test_restore_uses_available_backend(target, native_on_restore, monkeypatch):
    pytest.importorskip("mixedlm._rust")
    monkeypatch.setattr(laplace, "_HAS_RUST", not native_on_restore)
    obj, parameters = make_glmm_objective(target, order=7)
    expected = evaluate(obj, parameters)
    serialized = pickle.dumps(obj)
    monkeypatch.setattr(laplace, "_HAS_RUST", native_on_restore)
    restored = pickle.loads(serialized)
    assert (owner(restored)._native_problem is not None) == native_on_restore
    for value, reference in zip(evaluate(restored, parameters), expected, strict=True):
        assert_allclose(value, reference, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("target", ["optimizer", "joint"])
def test_legacy_state_without_native_cache_can_be_restored(target):
    obj, parameters = make_glmm_objective(target)
    expected = evaluate(obj, parameters)
    del obj._native_problem
    restored = round_trip(obj, 5)
    assert_same(evaluate(restored, parameters), expected)


def test_restored_optimizer_can_complete_a_fit():
    obj, theta = make_glmm_objective("optimizer", order=7)
    original = obj.optimize(start=theta)
    restored = round_trip(obj, 5).optimize(start=theta)
    assert original.converged and restored.converged
    for name in ["theta", "beta", "u", "deviance", "n_iter", "pirls_converged", "joint_fit"]:
        assert_array_equal(getattr(restored, name), getattr(original, name))


def test_objectives_and_modular_devfun_work_in_spawned_processes(monkeypatch):
    # Spawn requires actual serialization; fork alone can conceal this failure.
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"]:
        monkeypatch.setenv(name, "1")
    jobs = []
    for target in ["optimizer", "joint", "modular", "modular_joint"]:
        obj, parameters = make_glmm_objective(target, order=7)
        function = obj.objective if target == "optimizer" else obj
        jobs.append((function, parameters, function(parameters)))
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(function, parameters) for function, parameters, _ in jobs]
        for future, (_, _, expected) in zip(futures, jobs, strict=True):
            assert_array_equal(future.result(timeout=60), expected)

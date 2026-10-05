"""Prepared likelihoods preserve fresh solves, input ownership, and fallback routes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm.estimation import laplace
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from mixedlm.families import Poisson
from numpy.testing import assert_array_equal

from tests._glmm_oracles import mode_problem, native_glmm_arguments

native = pytest.importorskip("mixedlm._rust")


def prepare(args):
    return native.GlmmProblem(**{key: value for key, value in args.items() if key != "theta"})


def assert_state_equal(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("layout", ["intercept", "slope", "crossed", "fixed_only", "mode_only"])
@pytest.mark.parametrize("maxiter", [1, 100])
def test_reused_problem_matches_fresh_solves_including_zero_covariance(kind, layout, maxiter):
    args = native_glmm_arguments(kind, layout)
    problem = prepare(args)
    for scale in [1.0, 0.0, 1.7, 1.0]:
        current = dict(args, theta=args["theta"] * scale)
        options = dict(maxiter=maxiter, tol=1e-10)
        expected = native.glmm_deviance(**current, n_agq=1, **options)
        assert_state_equal(problem.evaluate(current["theta"], **options), expected)


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("layout", ["slope", "crossed"])
def test_reused_sparse_problem_matches_fresh_solves_across_design_patterns(kind, layout):
    # 128 groups select the sparse random-effect system. A zero final variance
    # removes design columns: slopes keep a smaller sparse pattern, while the
    # crossed model becomes diagonal before returning to the original pattern.
    matrices, family, theta = mode_problem(kind, layout, n_obs=1024, n_groups=128)
    problem = laplace._prepare_native_glmm(matrices, family)
    boundary = theta.copy()
    boundary[-1] = 0.0
    for current in [theta, boundary, theta, 1.6 * theta, boundary, 0.7 * theta]:
        expected = native.glmm_deviance(*laplace._native_glmm_args(current, matrices, family), 1)
        assert_state_equal(problem.evaluate(current), expected)


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
@pytest.mark.parametrize("layout", ["intercept", "fixed_only", "mode_only"])
@pytest.mark.parametrize("order", [1, 7])
def test_offset_overrides_use_current_start_and_do_not_change_prepared_offset(kind, layout, order):
    args = native_glmm_arguments(kind, layout)
    problem = prepare(args)
    for change in [0.2, -0.3, 0.0]:
        backing = np.zeros(len(args["offset"]) * 2)
        backing[::2] = args["offset"] + change
        offset = backing[::2]
        expected = native.glmm_deviance(**dict(args, offset=offset), n_agq=order)
        assert_state_equal(problem.evaluate(args["theta"], order, offset=offset), expected)
    assert_state_equal(
        problem.evaluate(args["theta"], order), native.glmm_deviance(**args, n_agq=order)
    )


@pytest.mark.parametrize(
    "field", ["y", "x", "z_data", "z_indices", "z_indptr", "weights", "offset"]
)
def test_prepared_problem_owns_numpy_inputs(field):
    args = native_glmm_arguments("poisson")
    expected = native.glmm_deviance(**args, n_agq=1)
    problem = prepare(args)
    args[field][...] = 0
    assert_state_equal(problem.evaluate(args["theta"]), expected)


def test_prepared_problem_owns_structure_lists():
    args = native_glmm_arguments("poisson", "slope")
    expected = native.glmm_deviance(**args, n_agq=1)
    problem = prepare(args)
    for field in ["n_levels", "n_terms", "correlated"]:
        args[field].clear()
    assert_state_equal(problem.evaluate(args["theta"]), expected)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"theta": []}, "theta must have length"),
        ({"theta": [0.3, 0.4]}, "theta must have length"),
        ({"offset": []}, "offset must have length"),
        ({"offset": np.zeros(49)}, "offset must have length"),
        ({"maxiter": 0}, "maxiter must be"),
        ({"tol": 0.0}, "tol must be"),
        ({"tol": np.nan}, "tol must be"),
        ({"tol": np.inf}, "tol must be"),
        ({"n_agq": 0}, "n_agq must be"),
    ],
)
def test_failed_evaluation_does_not_damage_reusable_problem(options, message):
    args = native_glmm_arguments("poisson")
    problem = prepare(args)
    with pytest.raises(ValueError, match=message):
        problem.evaluate(**dict({"theta": args["theta"]}, **options))
    assert_state_equal(problem.evaluate(args["theta"]), native.glmm_deviance(**args, n_agq=1))


@pytest.mark.parametrize("layout", ["slope", "crossed"])
def test_prepared_problem_rejects_unsupported_quadrature(layout):
    args = native_glmm_arguments("poisson", layout)
    with pytest.raises(ValueError, match="one random-effect term with one coefficient"):
        prepare(args).evaluate(args["theta"], 7)


def test_shared_problem_has_independent_concurrent_evaluations():
    args = native_glmm_arguments("poisson")
    problem = prepare(args)
    cases = [
        dict(args, theta=np.array([scale], dtype=float), offset=args["offset"] + scale)
        for scale in [0, 0.3, 1]
    ]
    expected = [native.glmm_deviance(**case, n_agq=7) for case in cases]

    def evaluate(index):
        case = cases[index]
        return problem.evaluate(case["theta"], 7, offset=case["offset"])

    indices = list(range(3)) * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, indices))
    for index, result in zip(indices, actual, strict=True):
        assert_state_equal(result, expected[index])


@pytest.mark.parametrize("order", [0, 1, 7])
@pytest.mark.parametrize("layout", ["intercept", "fixed_only"])
def test_optimizer_and_joint_objective_preserve_dispatch_and_controls(order, layout):
    matrices, family, theta = mode_problem("poisson", layout)
    optimizer = laplace.GLMMOptimizer(
        matrices, family, nAGQ=order, pirls_maxiter=1, pirls_tol=1e-10
    )
    expected = laplace.glmm_deviance_with_status(
        theta, matrices, family, order, pirls_maxiter=1, pirls_tol=1e-10
    )
    assert optimizer.objective(theta) == expected[0]
    objective = JointGLMMObjective(matrices, family, order, pirls_maxiter=1, pirls_tol=1e-10)
    for beta in [np.full(matrices.n_fixed, 0.2), np.full(matrices.n_fixed, -0.1)]:
        current = replace(objective.mode_matrices, offset=matrices.offset + matrices.X @ beta)
        expected = laplace.glmm_deviance_with_status(
            theta, current, family, max(1, order), pirls_maxiter=1, pirls_tol=1e-10
        )
        actual = objective.evaluate(np.r_[theta, beta])
        assert_state_equal(
            (actual[0], actual[2], actual[3]), (expected[0], expected[2], expected[3])
        )
        assert_array_equal(actual[1], beta)


@pytest.mark.parametrize("fallback", ["backend", "family", "link", "cs", "ar1"])
def test_custom_and_unsupported_models_keep_python_evaluation(fallback):
    from mixedlm.families.base import LogLink

    matrices, family, theta = mode_problem("poisson", "slope")
    if fallback == "family":

        class CustomPoisson(Poisson):
            pass

        family = CustomPoisson()
    elif fallback == "link":

        class CustomLog(LogLink):
            pass

        family.link = CustomLog()
    elif fallback in {"cs", "ar1"}:
        matrices.random_structures[0].cov_type = fallback
        theta = theta[:2]
    with patch.object(laplace, "_HAS_RUST", fallback != "backend"):
        optimizer = laplace.GLMMOptimizer(matrices, family)
        objective = optimizer.joint_objective()
        assert optimizer._native_problem is None
        assert objective._native_problem is None
        expected = laplace.glmm_deviance_with_status(theta, matrices, family)
        assert optimizer.objective(theta) == expected[0]
        beta = np.array([0.2, -0.1])
        current = replace(objective.mode_matrices, offset=matrices.offset + matrices.X @ beta)
        expected = laplace.glmm_deviance_with_status(theta, current, family)
        actual = objective.evaluate(np.r_[theta, beta])
        assert_state_equal(
            (actual[0], actual[2], actual[3]), (expected[0], expected[2], expected[3])
        )

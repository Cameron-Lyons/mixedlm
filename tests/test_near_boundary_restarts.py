"""Tiny positive variance scales need likelihood checks before accepting convergence."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import lmer, lmerControl
from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.estimation.reml import LMMOptimizer, _profiled_deviance_core
from numpy.testing import assert_allclose, assert_array_equal

from tests._glmm_oracles import mode_problem
from tests._lmm_oracles import linear_data, parameters


@pytest.mark.parametrize("method", ["L-BFGS-B", "TNC", "SLSQP"])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("start", [2e-6, 5e-5])
def test_flat_positive_scale_recovers_known_interior_minimum(method, analytic, start):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return 1e4 + (x[0] ** 2 - 0.12) ** 2

    options = {"ftol": 1e-12}
    if method != "SLSQP":
        options["gtol"] = 1e-4
    initial = np.array([start])
    result = run_optimizer(
        objective,
        initial,
        method,
        [(0.0, None)],
        options=options,
        jac=(lambda x: 4 * x * (x * x - 0.12)) if analytic else None,
        restart_edge=True,
    )
    assert result.success, result.message
    assert_allclose(result.x, np.sqrt(0.12), atol=3e-4)
    assert_allclose(result.fun, 1e4, rtol=0, atol=5e-8)
    assert result.nfev == len(calls)
    assert_array_equal(initial, [start])


def test_disabling_restarts_retains_the_near_zero_solution():
    result = run_optimizer(
        lambda x: 1e4 + (x[0] ** 2 - 0.12) ** 2,
        np.array([5e-5]),
        "L-BFGS-B",
        [(0.0, None)],
        restart_edge=False,
    )
    assert result.success
    assert_array_equal(result.x, [5e-5])
    assert result.fun > 1e4 + 0.01


@pytest.mark.parametrize("minimum", [0.0, 5e-5, 5e-4])
def test_true_zero_and_small_positive_minima_are_not_rounded_or_restarted(minimum):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return (x[0] - minimum) ** 2

    result = run_optimizer(
        objective,
        np.array([minimum]),
        "L-BFGS-B",
        [(0.0, None)],
        restart_edge=True,
    )
    assert result.success
    assert_array_equal(result.x, [minimum])
    assert result.fun == 0
    assert result.nit == 0
    assert result.nfev == len(calls)
    if minimum == 5e-4:
        # An interior solution has no extra probe calls.
        assert all(abs(point[0] - minimum) < 1e-7 for point in calls)


@pytest.mark.parametrize("options", [{"maxfun": 3}, {"maxiter": 0}, {"maxiter": 1}])
def test_near_zero_checks_keep_improvements_within_the_remaining_budget(options):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return 1e4 + (x[0] ** 2 - 0.12) ** 2

    result = run_optimizer(
        objective,
        np.array([5e-5]),
        "L-BFGS-B",
        [(0.0, None)],
        options=options,
        restart_edge=True,
    )
    assert not result.success
    assert result.x[0] > 5e-5
    assert result.fun < 1e4 + (5e-5**2 - 0.12) ** 2
    assert result.nfev == len(calls)
    assert result.nfev <= options.get("maxfun", np.inf)
    assert result.nit <= options.get("maxiter", np.inf)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("start", [2e-6, 5e-5])
def test_public_fit_recovers_balanced_variance_from_small_positive_start(native, reml, start):
    result = lmer(
        "y ~ 1 + (1 | g)",
        linear_data(),
        REML=reml,
        start=np.array([start]),
        control=lmerControl(optimizer="L-BFGS-B", use_rust=native, gtol=1e-3, ftol=1e-12),
    )
    residual_variance = 4 / 3
    between_variance = 0.7**2 * (6 / 5 if reml else 1)
    expected = np.sqrt((between_variance - residual_variance / 4) / residual_variance)
    assert result.converged
    assert_allclose(result.theta, [expected], atol=1e-4)
    assert_allclose(result.sigma**2, residual_variance, atol=1e-4)


@pytest.mark.parametrize("groups,response_index", [(64, 0), (256, 1)])
def test_weighted_coupled_fit_leaves_premature_near_zero_variance(groups, response_index):
    matrices, _, _ = mode_problem("gaussian", "slope", n_obs=8192, n_groups=groups)
    columns = np.roll(np.arange(matrices.n_random), 2)
    matrices = replace(matrices, Z=(matrices.Z + 0.15 * matrices.Z[:, columns]).tocsc())
    if response_index:
        matrices = replace(matrices, y=matrices.y[::-1] + 0.2 * matrices.weights)
    optimizer = LMMOptimizer(matrices, REML=True, use_rust=True)
    options = {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40}
    result = optimizer.optimize(start=parameters(matrices), options=options)
    reference = optimizer.optimize(
        start=parameters(matrices), options=options, use_analytic_gradient=True
    )
    assert result.converged and reference.converged
    assert result.theta[-1] > 0.07
    assert_allclose(result.deviance, reference.deviance, rtol=0, atol=2e-7)
    assert_allclose(result.theta, reference.theta, rtol=0, atol=2e-5)
    expected = _profiled_deviance_core(result.theta, matrices, True)
    assert_allclose(result.deviance, expected.deviance, rtol=0, atol=2e-9)
    assert_allclose(result.beta, expected.beta, rtol=0, atol=2e-10)
    assert_allclose(result.sigma, expected.sigma, rtol=0, atol=2e-10)
    assert_allclose(result.u, expected.u, rtol=0, atol=2e-10)

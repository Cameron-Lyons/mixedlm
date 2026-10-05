from __future__ import annotations

import warnings
from importlib.util import find_spec

import numpy as np
import pytest
from mixedlm.estimation.optimizers import (
    NLOPT_OPTIMIZER_NAMES,
    NelderMead,
    OptimizeResult,
    available_optimizers,
    golden,
    has_bobyqa,
    has_cobyqa,
    has_nlopt,
    nlminbwrap,
    run_optimizer,
)
from numpy.testing import assert_allclose, assert_array_equal


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def quadratic(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2


def sphere(x):
    return np.sum(x**2)


class TestAvailableOptimizers:
    def test_lists_scipy_methods_and_installed_nlopt_wrappers(self):
        opts = available_optimizers()
        assert opts == sorted(set(opts))
        assert {"L-BFGS-B", "Nelder-Mead", "COBYQA", "bobyqa"} <= set(opts)
        installed = NLOPT_OPTIMIZER_NAMES if has_nlopt() else set()
        assert NLOPT_OPTIMIZER_NAMES & set(opts) == installed


class TestHasOptionalDeps:
    def test_scipy_backed_optimizers_are_always_available(self):
        assert has_bobyqa() is True
        assert has_cobyqa() is True

    def test_has_nlopt_reports_whether_nlopt_is_importable(self):
        assert has_nlopt() is (find_spec("nlopt") is not None)


class TestOptimizeResult:
    def test_dataclass_fields(self):
        result = OptimizeResult(
            x=np.array([1.0, 2.0]),
            fun=0.5,
            success=True,
            nit=10,
            message="Converged",
        )
        assert_allclose(result.x, [1.0, 2.0])
        assert result.fun == 0.5
        assert result.success is True
        assert result.nit == 10
        assert result.message == "Converged"


class TestNelderMead:
    def test_simple_quadratic(self):
        nm = NelderMead(quadratic, np.array([0.0, 0.0]))
        result = nm.optimize()
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)
        assert result.fun < 1e-8

    def test_sphere_function(self):
        nm = NelderMead(sphere, np.array([1.0, 1.0, 1.0]))
        result = nm.optimize()
        assert result.success
        assert_allclose(result.x, [0.0, 0.0, 0.0], atol=1e-4)

    def test_rosenbrock(self):
        nm = NelderMead(rosenbrock, np.array([0.0, 0.0]), maxiter=2000)
        result = nm.optimize()
        assert_allclose(result.x, [1.0, 1.0], atol=0.1)

    def test_with_history_tracking(self):
        nm = NelderMead(quadratic, np.array([0.0, 0.0]), track_history=True)
        _result = nm.optimize()
        assert nm.state is not None
        assert len(nm.state.x_history) > 0
        assert len(nm.state.f_history) > 0
        assert nm.state.converged

    def test_custom_parameters(self):
        nm = NelderMead(
            quadratic,
            np.array([0.0, 0.0]),
            alpha=1.5,
            gamma=2.5,
            rho=0.4,
            sigma=0.4,
        )
        result = nm.optimize()
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)

    def test_convergence_tolerance(self):
        nm = NelderMead(quadratic, np.array([0.0, 0.0]), ftol=1e-10, xtol=1e-10)
        result = nm.optimize()
        assert result.fun < 1e-10


class TestGolden:
    def test_simple_quadratic(self):
        def f(x):
            return (x - 2) ** 2

        result = golden(f, (0, 5))
        assert result.success
        assert result.x[0] == pytest.approx(2.0, abs=1e-6)

    def test_different_interval(self):
        def f(x):
            return (x + 3) ** 2

        result = golden(f, (-10, 10))
        assert result.success
        assert result.x[0] == pytest.approx(-3.0, abs=1e-6)

    def test_minimum_at_boundary(self):
        def f(x):
            return x

        result = golden(f, (0, 10))
        assert result.x[0] == pytest.approx(0.0, abs=1e-6)

    def test_convergence(self):
        def f(x):
            return (x - 3.5) ** 2

        result = golden(f, (2, 5), tol=1e-10)
        assert result.x[0] == pytest.approx(3.5, abs=1e-6)


class TestNlminbwrap:
    def test_simple_optimization(self):
        result = nlminbwrap(quadratic, np.array([0.0, 0.0]))
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)

    def test_with_bounds(self):
        def f(x):
            return (x[0] - 5) ** 2 + (x[1] - 5) ** 2

        result = nlminbwrap(f, np.array([0.0, 0.0]), bounds=[(0, 3), (0, 3)])
        assert_allclose(result.x, [3.0, 3.0], atol=1e-6)

    def test_iteration_limit_stops_without_success(self):
        result = nlminbwrap(rosenbrock, np.array([0.0, 0.0]), maxiter=5, ftol=1e-10)
        assert not result.success
        assert result.nit == 5


class TestRunOptimizer:
    @pytest.mark.parametrize("analytic", [False, True])
    def test_trust_constr_reports_objective_gradient_and_adapts_callback(self, analytic):
        points = []

        def derivative(x):
            return 2 * (x - np.array([2.0, 3.0]))

        result = run_optimizer(
            quadratic,
            np.array([0.5, 0.5]),
            "trust-constr",
            [(0.0, 1.0), (0.0, 1.0)],
            jac=derivative if analytic else None,
            callback=lambda x: points.append(x.copy()),
            options={"gtol": 1e-8},
        )
        assert result.success
        assert_allclose(result.x, [1.0, 1.0], atol=1e-3)
        assert_allclose(result.jac, derivative(result.x), atol=1e-6)
        assert points
        assert_array_equal(points[-1], result.x)

    def test_lbfgsb(self):
        bounds = [(None, None), (None, None)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "L-BFGS-B", bounds)
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)

    def test_nelder_mead(self):
        bounds = [(None, None), (None, None)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "Nelder-Mead", bounds)
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)

    def test_powell(self):
        bounds = [(None, None), (None, None)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "Powell", bounds)
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)

    def test_nlminb(self):
        bounds = [(None, None), (None, None)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "nlminb", bounds)
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)

    def test_unknown_optimizer_raises(self):
        bounds = [(None, None), (None, None)]
        with pytest.raises(ValueError, match="Unknown optimizer"):
            run_optimizer(quadratic, np.array([0.0, 0.0]), "unknown_opt", bounds)

    def test_options_reach_the_optimizer(self):
        bounds = [(None, None), (None, None)]
        options = {"maxiter": 5}
        result = run_optimizer(
            rosenbrock, np.array([0.0, 0.0]), "L-BFGS-B", bounds, options=options
        )
        assert not result.success
        assert result.nit == 5

    def test_with_bounds(self):
        bounds = [(0.0, 1.5), (0.0, 2.5)]
        result = run_optimizer(quadratic, np.array([0.5, 0.5]), "L-BFGS-B", bounds)
        assert_allclose(result.x, [1.5, 2.5], atol=1e-6)

    def test_bfgs_omits_unsupported_bounds(self):
        bounds = [(0.0, 1.5), (0.0, 2.5)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_optimizer(quadratic, np.array([0.5, 0.5]), "BFGS", bounds)

        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)
        assert not any("cannot handle bounds" in str(item.message) for item in caught)

    def test_tnc_translates_maxiter_to_maxfun(self):
        bounds = [(None, None), (None, None)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_optimizer(
                quadratic,
                np.array([0.0, 0.0]),
                "TNC",
                bounds,
                options={"maxiter": 100},
            )

        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)
        assert not any("Unknown solver options" in str(item.message) for item in caught)

    def test_powell_translates_legacy_tolerance_names(self):
        bounds = [(None, None), (None, None)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_optimizer(
                quadratic,
                np.array([0.0, 0.0]),
                "Powell",
                bounds,
                options={"xatol": 1e-8, "fatol": 1e-8},
            )

        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-4)
        assert not any("Unknown solver options" in str(item.message) for item in caught)


class TestCobyqa:
    def test_simple_optimization(self):
        bounds = [(-5, 5), (-5, 5)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "COBYQA", bounds)
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-2)

    def test_legacy_name_and_options(self):
        bounds = [(-5, 5), (-5, 5)]
        options = {"rhobeg": 0.25, "rhoend": 1e-6, "maxfun": 1000}
        with pytest.deprecated_call(match="use 'COBYQA'"):
            result = run_optimizer(
                quadratic,
                np.array([0.0, 0.0]),
                "bobyqa",
                bounds,
                options=options,
            )
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-2)

    def test_rejects_unsupported_global_search(self):
        bounds = [(-5, 5), (-5, 5)]
        with pytest.raises(ValueError, match="does not support 'seek_global_minimum'"):
            run_optimizer(
                quadratic,
                np.array([0.0, 0.0]),
                "COBYQA",
                bounds,
                options={"seek_global_minimum": True},
            )


NLOPT_NAMES = sorted(NLOPT_OPTIMIZER_NAMES)


@pytest.mark.skipif(not has_nlopt(), reason="nlopt not installed")
class TestNlopt:
    @pytest.mark.parametrize("name", NLOPT_NAMES)
    def test_unbounded_parameters_reach_a_distant_optimum(self, name):
        # Finite 1e30 stand-ins for infinite bounds sent NLopt's first steps
        # to 5e29 and left most algorithms at false optima.
        def objective(x):
            return (x[0] - 0.7) ** 2 + (x[1] - 40.0) ** 2 + 0.5 * (x[0] - 0.7) * (x[1] - 40.0)

        result = run_optimizer(objective, np.array([1.0, 0.0]), name, [(0.0, None), (None, None)])
        assert result.success, result.message
        assert_allclose(result.x, [0.7, 40.0], atol=1e-4)
        assert result.nfev == result.nit > 0

    # A 0.5 step wider than these boxes made BOBYQA fail with an invalid
    # argument and left NEWUOA at a bound and NELDERMEAD at the start.
    @pytest.mark.parametrize(
        ("name", "bounds"),
        [(name, [(-0.2, 0.2), (-0.2, 0.2)]) for name in NLOPT_NAMES]
        # NLopt's PRAXIS has no native bounds and stalls when started on one.
        + [(name, [(0.0, 0.3), (-0.2, 0.0)]) for name in NLOPT_NAMES if name != "nloptwrap_PRAXIS"],
    )
    def test_narrow_bounds_reach_the_interior_optimum(self, name, bounds):
        def objective(x):
            return (x[0] - 0.15) ** 2 + (x[1] + 0.1) ** 2

        result = run_optimizer(objective, np.zeros(2), name, bounds)
        assert result.success, result.message
        assert_allclose(result.x, [0.15, -0.1], atol=1e-4)

    @pytest.mark.parametrize("name", NLOPT_NAMES)
    def test_large_objective_values_do_not_loosen_the_stopping_rule(self, name):
        # Like deviances in the thousands, the offset made a relative function
        # tolerance stop COBYLA, NELDERMEAD and SBPLX up to 6e-3 from the optimum.
        target = np.array([0.7, -1.3, 2.0])

        def objective(x):
            return 1e4 + np.sum((x - target) ** 2) + 0.3 * (x[0] - 0.7) * (x[1] + 1.3)

        bounds = [(0.0, None), (None, None), (None, None)]
        result = run_optimizer(objective, np.zeros(3), name, bounds)
        assert result.success, result.message
        assert_allclose(result.x, target, atol=1e-3)

    @pytest.mark.parametrize("name", NLOPT_NAMES)
    def test_evaluation_limit_is_not_convergence(self, name):
        calls = []

        def objective(x):
            calls.append(x.copy())
            return rosenbrock(x)

        result = run_optimizer(
            objective, np.array([-1.2, 1.0]), name, [(None, None)] * 2, options={"maxiter": 5}
        )
        assert not result.success
        assert result.message == "Max evaluations reached"
        assert result.nfev == len(calls) == 5
        assert result.fun == min(rosenbrock(x) for x in calls)

    # NLopt's COBYLA never returns on infinite values, so a regression in the
    # shared finite penalty would hang it; the other algorithms cover it.
    @pytest.mark.parametrize("name", [name for name in NLOPT_NAMES if name != "nloptwrap_COBYLA"])
    def test_infeasible_region_does_not_end_the_search(self, name):
        target = np.array([1.0, 0.3, -0.5])

        def objective(x):
            return np.inf if x[1] < -1 else np.sum((x - target) ** 2)

        bounds = [(0.0, None), (None, None), (None, None)]
        result = run_optimizer(objective, np.array([0.5, 0.0, 0.0]), name, bounds)
        assert result.success, result.message
        assert_allclose(result.x, target, atol=1e-4)

    def test_algorithm_failure_returns_the_unconverged_start(self):
        # NLopt rejects NEWUOA for one parameter instead of optimizing.
        result = run_optimizer(sphere, np.array([0.5]), "nloptwrap_NEWUOA", [(0.0, None)])
        assert not result.success
        assert result.message.startswith("NLopt LN_NEWUOA_BOUND failed")
        assert_array_equal(result.x, [0.5])

    @pytest.mark.parametrize("step", [0.5, 0.125])
    def test_initial_step_sets_the_first_trial_point(self, step):
        calls = []

        def objective(x):
            calls.append(x.copy())
            return quadratic(x)

        options = {} if step == 0.5 else {"initial_step": step}
        run_optimizer(objective, np.zeros(2), "nloptwrap_BOBYQA", [(None, None)] * 2, options)
        assert_array_equal(calls[1], [step, 0.0])

    def test_initial_step_is_limited_to_a_quarter_of_finite_bounds(self):
        calls = []

        def objective(x):
            calls.append(x.copy())
            return quadratic(x)

        bounds = [(-0.2, 0.2), (0.0, None)]
        options = {"initial_step": [0.5, 0.25]}
        run_optimizer(objective, np.array([0.0, 1.0]), "nloptwrap_BOBYQA", bounds, options)
        assert_allclose(calls[1], [0.1, 1.0])
        assert_allclose(calls[2], [0.0, 1.25])

from __future__ import annotations

import numpy as np
import pytest
from mixedlm.estimation.optimizers import (
    NelderMead,
    OptimizeResult,
    available_optimizers,
    golden,
    has_bobyqa,
    has_nlopt,
    nlminbwrap,
    run_optimizer,
)
from numpy.testing import assert_allclose


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def quadratic(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2


def sphere(x):
    return np.sum(x**2)


class TestAvailableOptimizers:
    def test_returns_list(self):
        opts = available_optimizers()
        assert isinstance(opts, list)
        assert len(opts) > 0

    def test_contains_scipy_optimizers(self):
        opts = available_optimizers()
        assert "L-BFGS-B" in opts
        assert "Nelder-Mead" in opts


class TestHasOptionalDeps:
    def test_has_bobyqa_returns_bool(self):
        result = has_bobyqa()
        assert isinstance(result, bool)

    def test_has_nlopt_returns_bool(self):
        result = has_nlopt()
        assert isinstance(result, bool)


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
        assert result.x[0] <= 3.0 + 1e-6
        assert result.x[1] <= 3.0 + 1e-6

    def test_with_options(self):
        result = nlminbwrap(
            quadratic,
            np.array([0.0, 0.0]),
            maxiter=500,
            ftol=1e-10,
        )
        assert result.success


class TestRunOptimizer:
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

    def test_with_options(self):
        bounds = [(None, None), (None, None)]
        options = {"maxiter": 100}
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "L-BFGS-B", bounds, options=options)
        assert result.success

    def test_with_bounds(self):
        bounds = [(0.0, 1.5), (0.0, 2.5)]
        result = run_optimizer(quadratic, np.array([0.5, 0.5]), "L-BFGS-B", bounds)
        assert result.x[0] <= 1.5 + 1e-6
        assert result.x[1] <= 2.5 + 1e-6


@pytest.mark.skipif(not has_bobyqa(), reason="pybobyqa not installed")
class TestBobyqa:
    def test_simple_optimization(self):
        bounds = [(-5, 5), (-5, 5)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "bobyqa", bounds)
        assert result.success
        assert_allclose(result.x, [2.0, 3.0], atol=1e-2)


@pytest.mark.skipif(not has_nlopt(), reason="nlopt not installed")
class TestNlopt:
    def test_bobyqa(self):
        bounds = [(-5, 5), (-5, 5)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "nloptwrap_BOBYQA", bounds)
        assert_allclose(result.x, [2.0, 3.0], atol=0.1)

    def test_neldermead(self):
        bounds = [(-5, 5), (-5, 5)]
        result = run_optimizer(quadratic, np.array([0.0, 0.0]), "nloptwrap_NELDERMEAD", bounds)
        assert_allclose(result.x, [2.0, 3.0], atol=0.1)

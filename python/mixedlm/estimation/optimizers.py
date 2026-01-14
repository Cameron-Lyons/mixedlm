from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

try:
    import pybobyqa

    _HAS_BOBYQA = True
except ImportError:
    _HAS_BOBYQA = False

try:
    import nlopt

    _HAS_NLOPT = True
except ImportError:
    _HAS_NLOPT = False


SCIPY_OPTIMIZERS = {
    "L-BFGS-B",
    "BFGS",
    "Nelder-Mead",
    "Powell",
    "trust-constr",
    "SLSQP",
    "TNC",
    "COBYLA",
}

NLOPT_OPTIMIZERS = {
    "nloptwrap_BOBYQA": nlopt.LN_BOBYQA if _HAS_NLOPT else None,
    "nloptwrap_NEWUOA": nlopt.LN_NEWUOA_BOUND if _HAS_NLOPT else None,
    "nloptwrap_PRAXIS": nlopt.LN_PRAXIS if _HAS_NLOPT else None,
    "nloptwrap_SBPLX": nlopt.LN_SBPLX if _HAS_NLOPT else None,
    "nloptwrap_COBYLA": nlopt.LN_COBYLA if _HAS_NLOPT else None,
    "nloptwrap_NELDERMEAD": nlopt.LN_NELDERMEAD if _HAS_NLOPT else None,
}

NLOPT_OPTIMIZER_NAMES = set(NLOPT_OPTIMIZERS.keys())

EXTERNAL_OPTIMIZERS = {"bobyqa"} | NLOPT_OPTIMIZER_NAMES

ALL_OPTIMIZERS = SCIPY_OPTIMIZERS | EXTERNAL_OPTIMIZERS


def has_bobyqa() -> bool:
    return _HAS_BOBYQA


def has_nlopt() -> bool:
    return _HAS_NLOPT


def available_optimizers() -> list[str]:
    opts = list(SCIPY_OPTIMIZERS)
    if _HAS_BOBYQA:
        opts.append("bobyqa")
    if _HAS_NLOPT:
        opts.extend(NLOPT_OPTIMIZER_NAMES)
    return sorted(opts)


@dataclass
class OptimizeResult:
    x: NDArray[np.floating]
    fun: float
    success: bool
    nit: int
    message: str


def _convert_bounds_to_arrays(
    bounds: list[tuple[float | None, float | None]],
    n: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    lower = np.full(n, -np.inf, dtype=np.float64)
    upper = np.full(n, np.inf, dtype=np.float64)

    for i, (lb, ub) in enumerate(bounds):
        if lb is not None:
            lower[i] = lb
        if ub is not None:
            upper[i] = ub

    return lower, upper


def _optimize_bobyqa(
    fun: Callable[[NDArray[np.floating]], float],
    x0: NDArray[np.floating],
    bounds: list[tuple[float | None, float | None]],
    options: dict[str, Any],
) -> OptimizeResult:
    if not _HAS_BOBYQA:
        raise ImportError(
            "pybobyqa is required for the 'bobyqa' optimizer. "
            "Install it with: pip install Py-BOBYQA"
        )

    lower, upper = _convert_bounds_to_arrays(bounds, len(x0))

    maxfun = options.get("maxiter", 1000) * (len(x0) + 1)
    rhobeg = options.get("rhobeg", 0.5)
    rhoend = options.get("rhoend", 1e-4)

    seek_global_minimum = options.get("seek_global_minimum", False)

    has_finite_bounds = np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    scaling_within_bounds = options.get("scaling_within_bounds", has_finite_bounds)

    result = pybobyqa.solve(
        fun,
        x0,
        bounds=(lower, upper),
        maxfun=maxfun,
        rhobeg=rhobeg,
        rhoend=rhoend,
        seek_global_minimum=seek_global_minimum,
        scaling_within_bounds=scaling_within_bounds,
    )

    success = result.flag in (result.EXIT_SUCCESS, result.EXIT_SLOW_WARNING)

    return OptimizeResult(
        x=result.x,
        fun=result.f,
        success=success,
        nit=result.nf,
        message=result.msg,
    )


def _optimize_nlopt(
    fun: Callable[[NDArray[np.floating]], float],
    x0: NDArray[np.floating],
    bounds: list[tuple[float | None, float | None]],
    options: dict[str, Any],
    algorithm: int,
) -> OptimizeResult:
    if not _HAS_NLOPT:
        raise ImportError(
            "nlopt is required for nloptwrap optimizers. Install it with: pip install nlopt"
        )

    n = len(x0)
    lower, upper = _convert_bounds_to_arrays(bounds, n)

    opt = nlopt.opt(algorithm, n)

    lower_clean = np.where(np.isfinite(lower), lower, -1e30)
    upper_clean = np.where(np.isfinite(upper), upper, 1e30)

    opt.set_lower_bounds(lower_clean.tolist())
    opt.set_upper_bounds(upper_clean.tolist())

    neval = [0]

    def nlopt_objective(x: list[float], grad: list[float]) -> float:
        neval[0] += 1
        return fun(np.array(x))

    opt.set_min_objective(nlopt_objective)

    maxeval = options.get("maxiter", 1000)
    opt.set_maxeval(maxeval)

    ftol_rel = options.get("ftol", 1e-8)
    xtol_rel = options.get("xtol", 1e-8)
    opt.set_ftol_rel(ftol_rel)
    opt.set_xtol_rel(xtol_rel)

    if "ftol_abs" in options:
        opt.set_ftol_abs(options["ftol_abs"])
    if "xtol_abs" in options:
        opt.set_xtol_abs(options["xtol_abs"])

    try:
        x_opt = opt.optimize(x0.tolist())
        f_opt = opt.last_optimum_value()
        result_code = opt.last_optimize_result()

        success = result_code > 0

        messages = {
            nlopt.SUCCESS: "Optimization succeeded",
            nlopt.STOPVAL_REACHED: "Stopval reached",
            nlopt.FTOL_REACHED: "Ftol reached",
            nlopt.XTOL_REACHED: "Xtol reached",
            nlopt.MAXEVAL_REACHED: "Max evaluations reached",
            nlopt.MAXTIME_REACHED: "Max time reached",
            nlopt.FAILURE: "Generic failure",
            nlopt.INVALID_ARGS: "Invalid arguments",
            nlopt.OUT_OF_MEMORY: "Out of memory",
            nlopt.ROUNDOFF_LIMITED: "Roundoff limited",
            nlopt.FORCED_STOP: "Forced stop",
        }
        message = messages.get(result_code, f"Unknown result code: {result_code}")

        return OptimizeResult(
            x=np.array(x_opt),
            fun=f_opt,
            success=success,
            nit=neval[0],
            message=message,
        )
    except Exception as e:
        return OptimizeResult(
            x=x0,
            fun=float("inf"),
            success=False,
            nit=neval[0],
            message=str(e),
        )


def _optimize_scipy(
    fun: Callable[[NDArray[np.floating]], float],
    x0: NDArray[np.floating],
    method: str,
    bounds: list[tuple[float | None, float | None]],
    options: dict[str, Any],
    callback: Callable[[NDArray[np.floating]], None] | None = None,
) -> OptimizeResult:
    result = minimize(
        fun,
        x0,
        method=method,
        bounds=bounds,
        options=options,
        callback=callback,
    )

    return OptimizeResult(
        x=result.x,
        fun=result.fun,
        success=result.success,
        nit=result.nit,
        message=result.message if hasattr(result, "message") else "",
    )


@dataclass
class NelderMeadState:
    """State of the Nelder-Mead algorithm with history tracking.

    This class tracks the optimization progress, including all function
    evaluations and simplex states.

    Attributes
    ----------
    x_history : list
        History of best x values at each iteration.
    f_history : list
        History of best function values at each iteration.
    simplex_history : list
        History of simplex vertices at each iteration.
    n_iter : int
        Number of iterations completed.
    converged : bool
        Whether the algorithm has converged.
    """

    x_history: list[NDArray[np.floating]]
    f_history: list[float]
    simplex_history: list[NDArray[np.floating]]
    n_iter: int
    converged: bool


class NelderMead:
    """Nelder-Mead optimizer with history tracking.

    This class implements the Nelder-Mead simplex algorithm with the
    ability to track optimization history for diagnostics and visualization.

    Parameters
    ----------
    fun : callable
        Objective function to minimize.
    x0 : ndarray
        Initial parameter values.
    alpha : float, default 1.0
        Reflection coefficient.
    gamma : float, default 2.0
        Expansion coefficient.
    rho : float, default 0.5
        Contraction coefficient.
    sigma : float, default 0.5
        Shrink coefficient.
    maxiter : int, default 1000
        Maximum number of iterations.
    ftol : float, default 1e-8
        Tolerance for function value convergence.
    xtol : float, default 1e-8
        Tolerance for parameter convergence.
    track_history : bool, default False
        Whether to track optimization history.

    Examples
    --------
    >>> def rosenbrock(x):
    ...     return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>> nm = NelderMead(rosenbrock, np.array([0.0, 0.0]), track_history=True)
    >>> result = nm.optimize()
    >>> result.x
    array([1., 1.])
    """

    def __init__(
        self,
        fun: Callable[[NDArray[np.floating]], float],
        x0: NDArray[np.floating],
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        maxiter: int = 1000,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        track_history: bool = False,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.n = len(self.x0)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.maxiter = maxiter
        self.ftol = ftol
        self.xtol = xtol
        self.track_history = track_history

        self.state: NelderMeadState | None = None

    def optimize(self) -> OptimizeResult:
        """Run the Nelder-Mead optimization.

        Returns
        -------
        OptimizeResult
            Optimization result with x, fun, success, nit, message.
        """
        n = self.n

        simplex = np.zeros((n + 1, n), dtype=np.float64)
        simplex[0] = self.x0
        for i in range(n):
            simplex[i + 1] = self.x0.copy()
            if self.x0[i] != 0:
                simplex[i + 1, i] = self.x0[i] * 1.05
            else:
                simplex[i + 1, i] = 0.00025

        f_vals = np.array([self.fun(simplex[i]) for i in range(n + 1)])

        x_history: list[NDArray[np.floating]] = []
        f_history: list[float] = []
        simplex_history: list[NDArray[np.floating]] = []

        for iteration in range(self.maxiter):
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]

            if self.track_history:
                x_history.append(simplex[0].copy())
                f_history.append(float(f_vals[0]))
                simplex_history.append(simplex.copy())

            if (
                np.max(np.abs(f_vals[1:] - f_vals[0])) < self.ftol
                and np.max(np.abs(simplex[1:] - simplex[0])) < self.xtol
            ):
                self.state = NelderMeadState(
                    x_history=x_history,
                    f_history=f_history,
                    simplex_history=simplex_history,
                    n_iter=iteration,
                    converged=True,
                )
                return OptimizeResult(
                    x=simplex[0],
                    fun=f_vals[0],
                    success=True,
                    nit=iteration,
                    message="Converged",
                )

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + self.alpha * (centroid - simplex[-1])
            f_r = self.fun(x_r)

            if f_vals[0] <= f_r < f_vals[-2]:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            elif f_r < f_vals[0]:
                x_e = centroid + self.gamma * (x_r - centroid)
                f_e = self.fun(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            else:
                if f_r < f_vals[-1]:
                    x_c = centroid + self.rho * (x_r - centroid)
                else:
                    x_c = centroid + self.rho * (simplex[-1] - centroid)
                f_c = self.fun(x_c)
                if f_c < min(f_r, f_vals[-1]):
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                        f_vals[i] = self.fun(simplex[i])

        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        self.state = NelderMeadState(
            x_history=x_history,
            f_history=f_history,
            simplex_history=simplex_history,
            n_iter=self.maxiter,
            converged=False,
        )

        return OptimizeResult(
            x=simplex[0],
            fun=f_vals[0],
            success=False,
            nit=self.maxiter,
            message="Maximum iterations reached",
        )


def golden(
    fun: Callable[[float], float],
    interval: tuple[float, float],
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> OptimizeResult:
    """Golden section search for 1D optimization.

    Finds the minimum of a unimodal function within a given interval
    using the golden section search algorithm.

    Parameters
    ----------
    fun : callable
        Objective function that takes a single float and returns a float.
    interval : tuple of float
        The (lower, upper) bounds of the search interval.
    tol : float, default 1e-8
        Tolerance for convergence.
    maxiter : int, default 1000
        Maximum number of iterations.

    Returns
    -------
    OptimizeResult
        Optimization result with x as a 1-element array.

    Examples
    --------
    >>> def f(x):
    ...     return (x - 2)**2
    >>> result = golden(f, (0, 5))
    >>> result.x[0]  # Should be approximately 2
    2.0
    """
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    a, b = interval
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = fun(c)
    fd = fun(d)

    for iteration in range(maxiter):
        if abs(b - a) < tol:
            x_opt = (a + b) / 2
            return OptimizeResult(
                x=np.array([x_opt]),
                fun=fun(x_opt),
                success=True,
                nit=iteration,
                message="Converged",
            )

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = fun(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = fun(d)

    x_opt = (a + b) / 2
    return OptimizeResult(
        x=np.array([x_opt]),
        fun=fun(x_opt),
        success=False,
        nit=maxiter,
        message="Maximum iterations reached",
    )


def nlminbwrap(
    fun: Callable[[NDArray[np.floating]], float],
    x0: NDArray[np.floating],
    bounds: list[tuple[float | None, float | None]] | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
) -> OptimizeResult:
    """Wrapper for L-BFGS-B optimization mimicking R's nlminb interface.

    This provides an interface similar to R's nlminb function, which is
    commonly used in lme4 for optimization.

    Parameters
    ----------
    fun : callable
        Objective function to minimize.
    x0 : ndarray
        Initial parameter values.
    bounds : list of tuple, optional
        Bounds for each parameter as (lower, upper) pairs.
        None means unbounded.
    maxiter : int, default 1000
        Maximum number of iterations.
    ftol : float, default 1e-8
        Tolerance for function value convergence.
    gtol : float, default 1e-5
        Tolerance for gradient convergence.

    Returns
    -------
    OptimizeResult
        Optimization result with x, fun, success, nit, message.

    Examples
    --------
    >>> def f(x):
    ...     return (x[0] - 1)**2 + (x[1] - 2)**2
    >>> result = nlminbwrap(f, np.array([0.0, 0.0]))
    >>> result.x
    array([1., 2.])
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n = len(x0)

    if bounds is None:
        bounds = [(None, None)] * n

    options = {
        "maxiter": maxiter,
        "ftol": ftol,
        "gtol": gtol,
    }

    return _optimize_scipy(fun, x0, "L-BFGS-B", bounds, options)


def run_optimizer(
    fun: Callable[[NDArray[np.floating]], float],
    x0: NDArray[np.floating],
    method: str,
    bounds: list[tuple[float | None, float | None]],
    options: dict[str, Any] | None = None,
    callback: Callable[[NDArray[np.floating]], None] | None = None,
) -> OptimizeResult:
    options = options or {}

    method_lower = method.lower()

    if method_lower == "bobyqa":
        return _optimize_bobyqa(fun, x0, bounds, options)
    elif method_lower == "nlminb":
        return nlminbwrap(
            fun,
            x0,
            bounds,
            maxiter=options.get("maxiter", 1000),
            ftol=options.get("ftol", 1e-8),
            gtol=options.get("gtol", 1e-5),
        )
    elif method in NLOPT_OPTIMIZER_NAMES:
        algorithm = NLOPT_OPTIMIZERS[method]
        if algorithm is None:
            raise ImportError(
                f"nlopt is required for '{method}'. Install it with: pip install nlopt"
            )
        return _optimize_nlopt(fun, x0, bounds, options, algorithm)
    elif method in SCIPY_OPTIMIZERS:
        return _optimize_scipy(fun, x0, method, bounds, options, callback)
    else:
        raise ValueError(
            f"Unknown optimizer '{method}'. Valid options: {', '.join(sorted(ALL_OPTIMIZERS))}"
        )

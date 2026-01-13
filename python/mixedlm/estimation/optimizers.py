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
            "nlopt is required for nloptwrap optimizers. "
            "Install it with: pip install nlopt"
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
            f"Unknown optimizer '{method}'. "
            f"Valid options: {', '.join(sorted(ALL_OPTIMIZERS))}"
        )

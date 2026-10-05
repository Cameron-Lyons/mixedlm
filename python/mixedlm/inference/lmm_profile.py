"""Maximum-likelihood profiles with covariance and residual-scale optimization."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, stats

from mixedlm._parallel import process_pool, resolve_n_jobs
from mixedlm.estimation.reml import (
    _build_theta_bounds,
    _count_theta,
    _LMMCrossproducts,
    _profiled_deviance_core,
)
from mixedlm.estimation.validation import validate_finite_real
from mixedlm.inference.profile_types import Profile2DResult, ProfileResult
from mixedlm.utils.names import _check_unique_coefficient_names
from mixedlm.utils.validation import _validate_confidence_level

if TYPE_CHECKING:
    from mixedlm.estimation.reml import _DevianceCoreResult
    from mixedlm.matrices.design import ModelMatrices
    from mixedlm.models.lmer import LmerResult


@dataclass
class _LMMProfileFit:
    theta: NDArray[np.floating]
    evaluation: _DevianceCoreResult

    @property
    def deviance(self) -> float:
        return float(self.evaluation.deviance)


class _LMMProfileLikelihood:
    """Reuse the weighted design products across constrained covariance fits."""

    def __init__(self, matrices: ModelMatrices) -> None:
        self.matrices = replace(matrices, frame=None)
        self.products = _LMMCrossproducts.from_matrices(matrices)
        self.n_theta = _count_theta(matrices.random_structures)
        self.bounds = _build_theta_bounds(matrices.random_structures, self.n_theta)

    def fit(
        self,
        start: NDArray[np.floating],
        indices: tuple[int, ...] = (),
        values: tuple[float, ...] = (),
    ) -> _LMMProfileFit:
        validate_finite_real("variance parameters", start, (self.n_theta,))
        keep = np.asarray([i for i in range(self.matrices.n_fixed) if i not in indices], dtype=int)
        fixed = np.asarray(indices, dtype=int)
        held = np.asarray(values, dtype=float)
        validate_finite_real("profile coefficients", held, (len(indices),))
        adjusted_y = self.products.y_adj - self.matrices.X[:, fixed] @ held
        matrices = replace(
            self.matrices,
            y=adjusted_y,
            offset=np.zeros(self.matrices.n_obs),
            X=self.matrices.X[:, keep],
            n_fixed=len(keep),
            fixed_names=[self.matrices.fixed_names[i] for i in keep],
        )
        products = replace(
            self.products,
            y_adj=adjusted_y,
            XtWX=self.products.XtWX[np.ix_(keep, keep)],
            XtWy=self.products.XtWy[keep] - self.products.XtWX[np.ix_(keep, fixed)] @ held,
            ZtWX=self.products.ZtWX[:, keep],
            ZtWy=self.products.ZtWy - self.products.ZtWX[:, fixed] @ held,
        )
        scale = np.maximum(np.abs(start), 1.0)
        last_point = None
        last_evaluation = None

        def evaluate(scaled: NDArray[np.floating]) -> _DevianceCoreResult:
            nonlocal last_point, last_evaluation
            if last_evaluation is not None and np.array_equal(scaled, last_point):
                return last_evaluation
            theta = scaled * scale
            evaluation = _profiled_deviance_core(
                theta, matrices, REML=False, crossproducts=products
            )
            if evaluation is None:
                raise RuntimeError("LMM profile covariance factorization failed")
            if (
                not np.isfinite(evaluation.deviance)
                or not np.isfinite(evaluation.sigma)
                or evaluation.sigma <= 0
            ):
                raise RuntimeError(
                    "LMM likelihood profiling requires a finite deviance "
                    "and positive residual scale"
                )
            validate_finite_real("profile fixed effects", evaluation.beta, (len(keep),))
            validate_finite_real("profile random effects", evaluation.u, (matrices.n_random,))
            last_point, last_evaluation = scaled.copy(), evaluation
            return evaluation

        if not self.n_theta:
            return _LMMProfileFit(start.copy(), evaluate(start))
        bounds = [
            (None if lo is None else lo / step, None if hi is None else hi / step)
            for (lo, hi), step in zip(self.bounds, scale, strict=True)
        ]

        # Variance is quadratic in its scale: a zero scale has zero gradient
        # even when increasing it improves the constrained likelihood.
        def at_boundary(theta: NDArray[np.floating]) -> bool:
            return any(
                lo == 0 and value <= 1e-6 for value, (lo, _) in zip(theta, self.bounds, strict=True)
            )

        methods = ["COBYQA"] if at_boundary(start) else ["L-BFGS-B", "COBYQA"]
        solver_start = start / scale
        failures = []
        for method in methods:
            # Finite differences can also stall near an interior optimum.
            # Retry the same likelihood from the valid start if needed.
            fitted = optimize.minimize(
                lambda point: evaluate(point).deviance,
                solver_start,
                method=method,
                jac="3-point" if method == "L-BFGS-B" else None,
                bounds=bounds,
                options=(
                    {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6}
                    if method == "L-BFGS-B"
                    else {"maxiter": 2000, "final_tr_radius": 1e-8}
                ),
            )
            if fitted.success:
                if method == "L-BFGS-B" and at_boundary(fitted.x * scale):
                    solver_start = fitted.x
                    continue
                return _LMMProfileFit(fitted.x * scale, evaluate(fitted.x))
            failures.append(f"{method}: {fitted.message}")
        raise RuntimeError("LMM profile nuisance optimization failed: " + "; ".join(failures))


def _check_deviance(deviance: float, minimum: float) -> None:
    tolerance = max(1e-6, 128 * np.finfo(float).eps * abs(minimum))
    if deviance < minimum - tolerance:
        raise RuntimeError(
            "LMM profile found a lower deviance than the ML optimum; "
            "the likelihood optimization is unreliable for this model"
        )


class _LMMParameterProfile:
    def __init__(
        self, likelihood: _LMMProfileLikelihood, index: int, optimum: _LMMProfileFit
    ) -> None:
        self.likelihood = likelihood
        self.index = index
        self.minimum = optimum.deviance
        self.mle = float(optimum.evaluation.beta[index])
        self.cache = {self.mle: (self.minimum, optimum.theta)}

    def deviance(self, value: float) -> float:
        if value not in self.cache:
            nearest = min(self.cache, key=lambda point: abs(point - value))
            fitted = self.likelihood.fit(self.cache[nearest][1], (self.index,), (value,))
            _check_deviance(fitted.deviance, self.minimum)
            self.cache[value] = fitted.deviance, fitted.theta
        return self.cache[value][0]

    def endpoint(self, direction: int, step: float, cutoff: float) -> float:
        def difference(value: float) -> float:
            return self.deviance(value) - self.minimum - cutoff

        for _ in range(30):
            outer = self.mle + direction * step
            if difference(outer) >= 0:
                lower, upper = sorted((self.mle, outer))
                return float(
                    optimize.brentq(difference, lower, upper, xtol=step * 1e-8, rtol=1e-12)
                )
            step *= 2
        raise RuntimeError("Could not bracket the LMM likelihood-ratio confidence interval")


def _grid(lower: float, upper: float, center: float, n_points: int) -> NDArray[np.float64]:
    # Equal numbers of plotted points on each side retain an exact middle for odd grids.
    n_lower = n_points // 2
    return np.r_[
        np.linspace(lower, center, n_lower + 1)[:-1], np.linspace(center, upper, n_points - n_lower)
    ]


def _validate_options(n_points: int, n_jobs: int) -> int:
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, Integral) or n_points < 3:
        raise ValueError("n_points must be an integer of at least 3")
    return resolve_n_jobs(n_jobs)


def _reference(
    result: LmerResult,
) -> tuple[_LMMProfileLikelihood, _LMMProfileFit, NDArray[np.floating]]:
    if not result.converged:
        raise ValueError("LMM likelihood profiling requires a converged fitted model")
    likelihood = _LMMProfileLikelihood(result.matrices)
    optimum = likelihood.fit(result.theta)
    information = optimum.evaluation.fixed_information
    covariance = linalg.cho_solve(
        (linalg.cholesky(information, lower=True), True), np.eye(len(result.beta))
    )
    standard_errors = optimum.evaluation.sigma * np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(standard_errors) & (standard_errors > 0)):
        raise ValueError("LMM likelihood profiling requires finite, positive standard errors")
    if np.max(np.abs(optimum.evaluation.beta - result.beta) / standard_errors) > 1e-3:
        warnings.warn(
            "Likelihood profiling uses an ML refit; "
            "profile centers may differ from the fitted coefficients.",
            UserWarning,
            stacklevel=4,
        )
    return likelihood, optimum, standard_errors


def _profile_one(task: tuple[Any, ...]) -> tuple[str, ProfileResult]:
    name, index, likelihood, optimum, se, n_points, level = task
    profile = _LMMParameterProfile(likelihood, index, optimum)
    cutoff = float(stats.chi2.isf(1 - level, 1))
    lower = profile.endpoint(-1, se, cutoff)
    upper = profile.endpoint(1, se, cutoff)
    values = _grid(lower, upper, profile.mle, n_points)
    zeta = np.empty(n_points)
    for i in np.argsort(np.abs(values - profile.mle)):
        value = float(values[i])
        zeta[i] = np.sign(value - profile.mle) * np.sqrt(
            max(0.0, profile.deviance(value) - profile.minimum)
        )
    return name, ProfileResult(name, values, zeta, profile.mle, lower, upper, level)


def _run_tasks(worker: Callable[..., Any], tasks: list[Any], n_jobs: int) -> list[Any]:
    workers = min(n_jobs, len(tasks))
    if workers <= 1:
        return [worker(task) for task in tasks]
    try:
        executor = process_pool(workers)
    except (NotImplementedError, OSError) as error:
        warnings.warn(
            "Process-based parallel profiling is unavailable; "
            f"falling back to serial execution ({error}).",
            RuntimeWarning,
            stacklevel=3,
        )
        return [worker(task) for task in tasks]
    with executor:
        return list(executor.map(worker, tasks))


def likelihood_profiles(
    result: LmerResult,
    which: str | list[str] | None,
    n_points: int,
    level: float,
    n_jobs: int,
) -> dict[str, ProfileResult]:
    level = _validate_confidence_level(level)
    jobs = _validate_options(n_points, n_jobs)
    names = result.matrices.fixed_names
    requested = names if which is None else [which] if isinstance(which, str) else which
    _check_unique_coefficient_names(
        names,
        requested,
        alternative="Rename colliding formula variables before requesting named profiles.",
    )
    selected = list(dict.fromkeys(name for name in requested if name in names))
    if not selected:
        return {}
    likelihood, optimum, standard_errors = _reference(result)
    tasks = [
        (
            name,
            names.index(name),
            likelihood,
            optimum,
            float(standard_errors[names.index(name)]),
            n_points,
            level,
        )
        for name in selected
    ]
    return dict(_run_tasks(_profile_one, tasks, jobs))


def _surface_row(task: tuple[Any, ...]) -> NDArray[np.floating]:
    likelihood, optimum, indices, first, second_values, second_mle = task
    cache: dict[float, NDArray[np.floating]] = {}
    row = np.empty(len(second_values))
    for j in np.argsort(np.abs(second_values - second_mle)):
        second = float(second_values[j])
        if first == optimum.evaluation.beta[indices[0]] and second == second_mle:
            row[j] = 0.0
            cache[second] = optimum.theta
            continue
        nearest = min(cache, key=lambda point: abs(point - second)) if cache else None
        start = optimum.theta if nearest is None else cache[nearest]
        fitted = likelihood.fit(start, indices, (first, second))
        _check_deviance(fitted.deviance, optimum.deviance)
        cache[second] = fitted.theta
        row[j] = np.sqrt(max(0.0, fitted.deviance - optimum.deviance))
    return row


def likelihood_surface(
    result: LmerResult,
    param1: str,
    param2: str,
    n_points: int,
    level: float,
    n_jobs: int,
) -> Profile2DResult:
    level = _validate_confidence_level(level)
    jobs = _validate_options(n_points, n_jobs)
    names = result.matrices.fixed_names
    _check_unique_coefficient_names(
        names,
        [param1, param2],
        alternative="Rename colliding formula variables before requesting named profiles.",
    )
    if param1 == param2:
        raise ValueError("The two profile parameters must be distinct")
    for name in (param1, param2):
        if name not in names:
            raise ValueError(f"Parameter '{name}' not found in fixed effects")
    likelihood, optimum, standard_errors = _reference(result)
    indices = (names.index(param1), names.index(param2))
    grids = []
    cutoff = float(stats.chi2.isf(1 - level, 2))
    for index in indices:
        profile = _LMMParameterProfile(likelihood, index, optimum)
        lower = profile.endpoint(-1, float(standard_errors[index]), cutoff)
        upper = profile.endpoint(1, float(standard_errors[index]), cutoff)
        grids.append(_grid(lower, upper, profile.mle, n_points))
    first_mle, second_mle = (float(optimum.evaluation.beta[index]) for index in indices)
    tasks = [
        (likelihood, optimum, indices, float(first), grids[1], second_mle) for first in grids[0]
    ]
    zeta = np.stack(_run_tasks(_surface_row, tasks, jobs))
    return Profile2DResult(
        param1,
        param2,
        grids[0],
        grids[1],
        zeta,
        first_mle,
        second_mle,
        level,
        profile_covariance=True,
    )

"""Likelihood-ratio profiles with joint GLMM nuisance optimization."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from mixedlm.estimation.joint_glmm import JointGLMMObjective
from mixedlm.estimation.optimizers import run_optimizer
from mixedlm.inference.profile_types import ProfileResult
from mixedlm.utils.names import _check_unique_coefficient_names
from mixedlm.utils.validation import _validate_confidence_level

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult


class _GLMMProfileLikelihood:
    """Hold the design fixed and optimize beta and theta outside the random-mode solve."""

    def __init__(self, result: GlmerResult, standard_errors: NDArray[np.floating]) -> None:
        self.result = result
        self.n_theta = len(result.theta)
        self.scale = np.r_[np.maximum(np.abs(result.theta), 1.0), standard_errors]
        self.start = np.r_[result.theta, result.beta] / self.scale
        self.objective = JointGLMMObjective(
            result.matrices,
            result.family,
            result.nAGQ,
            pirls_maxiter=result.pirls_maxiter,
            pirls_tol=result.pirls_tol,
        )
        bounds = self.objective.bounds
        self.bounds = [
            (
                None if lower is None else lower / scale,
                None if upper is None else upper / scale,
            )
            for (lower, upper), scale in zip(bounds, self.scale, strict=True)
        ]

    def deviance(self, scaled: NDArray[np.floating]) -> float:
        deviance, _, _, converged = self.objective.evaluate(scaled * self.scale)
        if not converged or not np.isfinite(deviance):
            raise RuntimeError(
                "GLMM likelihood profiling requires a finite, converged inner PIRLS solve; "
                "check the model or refit with a larger pirls_maxiter"
            )
        return float(deviance)

    def fit(
        self, start: NDArray[np.floating], fixed: tuple[int, float] | None = None
    ) -> tuple[float, NDArray[np.floating]]:
        free = np.ones(len(start), dtype=bool)
        template = start.copy()
        if fixed is not None:
            index, value = fixed
            free[index] = False
            template[index] = value / self.scale[index]

        def unpack(values: NDArray[np.floating]) -> NDArray[np.floating]:
            parameters = template.copy()
            parameters[free] = values
            return parameters

        def objective(values: NDArray[np.floating]) -> float:
            return self.deviance(unpack(values))

        if not np.any(free):
            return self.deviance(template), template
        bounds = [bound for bound, keep in zip(self.bounds, free, strict=True) if keep]
        failures = []
        # A start at the optimum can stall both optimizers without an improvement;
        # then the best point either evaluated, including the start, is the fit.
        best_value, best = objective(template[free]), template[free]
        for method in ("L-BFGS-B", "COBYQA"):
            # The converged mode solve can leave enough objective roundoff to
            # stall a finite-difference line search near a constrained optimum.
            # Retry the same bounded likelihood from the valid starting point.
            fitted = run_optimizer(
                objective,
                template[free],
                method=method,
                jac="3-point" if method == "L-BFGS-B" else None,
                bounds=bounds,
                options=(
                    {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6}
                    if method == "L-BFGS-B"
                    else {"maxiter": 2000, "final_tr_radius": 1e-8}
                ),
                restart_edge=True,
            )
            if fitted.success:
                parameters = unpack(fitted.x)
                return self.deviance(parameters), parameters
            failures.append(f"{method}: {fitted.message}")
            point = getattr(fitted, "x", None)
            value = getattr(fitted, "fun", np.nan)
            if point is not None and np.all(np.isfinite(point)) and value < best_value:
                best_value, best = float(value), np.asarray(point, dtype=np.float64)
        if np.isfinite(best_value) and self._stationary(objective, best, bounds):
            parameters = unpack(best)
            return self.deviance(parameters), parameters
        raise RuntimeError("GLMM profile nuisance optimization failed: " + "; ".join(failures))

    @staticmethod
    def _stationary(
        objective: Callable[[NDArray[np.floating]], float],
        point: NDArray[np.floating],
        bounds: list[tuple[float | None, float | None]],
    ) -> bool:
        """Whether the bounded deviance gradient vanishes at ``point``.

        Parameters are scaled by their standard errors, so the deviance has
        roughly unit curvature, and a gradient below 1e-3 leaves the deviance
        within about 1e-6 of its constrained minimum. A parameter at a bound is
        also probed inward, where the deviance can fall with no first-order slope.
        """
        step = 1e-5
        for index, (lower, upper) in enumerate(bounds):
            values = []
            for direction in (-1.0, 1.0):
                trial = point.copy()
                trial[index] += direction * step
                inside = (lower is None or trial[index] >= lower) and (
                    upper is None or trial[index] <= upper
                )
                try:
                    values.append(objective(trial) if inside else None)
                except RuntimeError:
                    return False
            center = objective(point)
            below, above = values
            if below is not None and above is not None:
                gradient = (above - below) / (2 * step)
            elif above is not None:
                # At a lower bound only an increase is allowed.
                gradient = min(0.0, (above - center) / step)
            elif below is not None:
                gradient = max(0.0, (center - below) / step)
            else:
                continue
            if abs(gradient) > 1e-3:
                return False
        # A variance parameter at its bound has no first-order slope by symmetry,
        # so probe inward, as restart_edge does, before accepting the boundary.
        center = objective(point)
        for index, (lower, upper) in enumerate(bounds):
            for bound, direction in ((lower, 1.0), (upper, -1.0)):
                if bound is None or abs(point[index] - bound) > 1e-8:
                    continue
                for size in (1e-3, 1e-2, 1e-1, 0.3):
                    trial = point.copy()
                    trial[index] += direction * size
                    try:
                        if objective(trial) < center - 1e-6:
                            return False
                    except RuntimeError:
                        break
        return True


class _GLMMParameterProfile:
    def __init__(
        self,
        likelihood: _GLMMProfileLikelihood,
        index: int,
        optimum: NDArray[np.floating],
        minimum: float,
    ) -> None:
        self.likelihood = likelihood
        self.index = index
        self.minimum = minimum
        self.mle = float(optimum[index] * likelihood.scale[index])
        self.cache = {self.mle: (minimum, optimum)}

    def deviance(self, value: float) -> float:
        if value not in self.cache:
            nearest = min(self.cache, key=lambda point: abs(point - value))
            deviance, parameters = self.likelihood.fit(
                self.cache[nearest][1], fixed=(self.index, value)
            )
            tolerance = max(1e-6, 128 * np.finfo(float).eps * abs(self.minimum))
            if deviance < self.minimum - tolerance:
                raise RuntimeError(
                    "GLMM profile found a lower deviance than the joint optimum; "
                    "the likelihood optimization is unreliable for this model"
                )
            self.cache[value] = deviance, parameters
        return self.cache[value][0]

    def zeta(self, value: float) -> float:
        return float(
            np.sign(value - self.mle) * np.sqrt(max(0.0, self.deviance(value) - self.minimum))
        )

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
        raise RuntimeError("Could not bracket the GLMM likelihood-ratio confidence interval")


def likelihood_profiles(
    result: GlmerResult,
    which: str | list[str] | None,
    n_points: int,
    level: float,
) -> dict[str, ProfileResult]:
    level = _validate_confidence_level(level)
    if isinstance(n_points, bool) or not isinstance(n_points, Integral) or n_points < 3:
        raise ValueError("n_points must be an integer of at least 3")
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
    if not result.converged or not result.pirls_converged:
        raise ValueError("GLMM likelihood profiling requires a converged fitted model")
    standard_errors = np.sqrt(np.diag(result.vcov()))
    if not np.all(np.isfinite(standard_errors) & (standard_errors > 0)):
        raise ValueError("GLMM likelihood profiling requires finite, positive standard errors")
    likelihood = _GLMMProfileLikelihood(result, standard_errors)
    minimum, optimum = likelihood.fit(likelihood.start)
    if (
        np.max(np.abs(optimum[likelihood.n_theta :] - likelihood.start[likelihood.n_theta :]))
        > 1e-3
    ):
        warnings.warn(
            "Likelihood profiling refined the joint optimum; profile centers may differ "
            "from the fitted coefficients.",
            UserWarning,
            stacklevel=3,
        )
    cutoff = float(stats.norm.isf((1 - level) / 2) ** 2)
    profiles = {}
    for name in selected:
        index = names.index(name)
        profile = _GLMMParameterProfile(likelihood, likelihood.n_theta + index, optimum, minimum)
        lower = profile.endpoint(-1, float(standard_errors[index]), cutoff)
        upper = profile.endpoint(1, float(standard_errors[index]), cutoff)
        values = np.linspace(lower, upper, n_points)
        center = int(np.clip(np.argmin(np.abs(values - profile.mle)), 1, n_points - 2))
        values[center] = profile.mle
        zeta = np.empty(n_points)
        # Continue outward from the optimum, reusing the nearest constrained fit.
        for j in np.argsort(np.abs(values - profile.mle)):
            zeta[j] = profile.zeta(float(values[j]))
        profiles[name] = ProfileResult(name, values, zeta, profile.mle, lower, upper, level)
    return profiles

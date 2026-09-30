"""Likelihood-ratio profiles with joint GLMM nuisance optimization."""

from __future__ import annotations

import warnings
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from mixedlm.estimation.laplace import glmm_deviance_with_status
from mixedlm.estimation.reml import _build_theta_bounds
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
        bounds = _build_theta_bounds(result.matrices.random_structures, self.n_theta)
        bounds += [(None, None)] * len(result.beta)
        self.bounds = [
            (
                None if lower is None else lower / scale,
                None if upper is None else upper / scale,
            )
            for (lower, upper), scale in zip(bounds, self.scale, strict=True)
        ]
        # Fixed coefficients enter through the offset. PIRLS then optimizes only
        # random effects, so its joint beta/mode approximation cannot substitute
        # for minimization of the integrated likelihood over nuisance beta.
        self.matrices = replace(
            result.matrices,
            X=np.empty((result.matrices.n_obs, 0), dtype=np.float64),
            n_fixed=0,
            fixed_names=[],
        )

    def deviance(self, scaled: NDArray[np.floating]) -> float:
        parameters = scaled * self.scale
        matrices = replace(
            self.matrices,
            offset=self.result.matrices.offset
            + self.result.matrices.X @ parameters[self.n_theta :],
        )
        deviance, _, _, converged = glmm_deviance_with_status(
            parameters[: self.n_theta],
            matrices,
            self.result.family,
            nAGQ=self.result.nAGQ,
            pirls_maxiter=self.result.pirls_maxiter,
            pirls_tol=self.result.pirls_tol,
        )
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
        fitted = optimize.minimize(
            objective,
            template[free],
            method="L-BFGS-B",
            jac="3-point",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6},
        )
        if not fitted.success:
            raise RuntimeError(f"GLMM profile nuisance optimization failed: {fitted.message}")
        parameters = unpack(fitted.x)
        return self.deviance(parameters), parameters


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

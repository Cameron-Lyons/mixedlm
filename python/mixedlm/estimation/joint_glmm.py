"""GLMM likelihood evaluation with fixed coefficients outside the mode solve."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.laplace import (
    _evaluate_native_problem,
    _prepare_native_glmm,
    _validate_quadrature,
    glmm_deviance_with_status,
)
from mixedlm.estimation.pirls_control import validate_pirls_controls
from mixedlm.estimation.reml import _build_theta_bounds, _count_theta
from mixedlm.estimation.validation import validate_finite_real
from mixedlm.families.base import Family
from mixedlm.matrices.design import ModelMatrices


class JointGLMMObjective:
    """Evaluate covariance parameters followed by beta for a fixed model.

    Treat the model arrays and family as immutable for this object's lifetime.
    Each evaluation uses an independent mode solve with its own combined offset.
    """

    def __init__(
        self,
        matrices: ModelMatrices,
        family: Family,
        nAGQ: int = 1,
        *,
        pirls_maxiter: int | None = None,
        pirls_tol: float = 1e-6,
    ) -> None:
        _validate_quadrature(nAGQ, matrices)
        validate_pirls_controls(pirls_maxiter, pirls_tol)
        self.matrices = matrices
        self.family = family
        self.nAGQ = nAGQ
        self.pirls_maxiter = pirls_maxiter
        self.pirls_tol = pirls_tol
        self.n_theta = _count_theta(matrices.random_structures)
        self.n_parameters = self.n_theta + matrices.n_fixed
        self.mode_matrices = replace(
            matrices,
            X=np.empty((matrices.n_obs, 0), dtype=np.float64),
            n_fixed=0,
            fixed_names=[],
        )
        self._native_problem = _prepare_native_glmm(self.mode_matrices, family)
        self.bounds = _build_theta_bounds(matrices.random_structures, self.n_theta)
        self.bounds += [(None, None)] * matrices.n_fixed

    def __getstate__(self) -> dict[str, Any]:
        # Rebuild the mode-only native problem in the receiving process.
        state = self.__dict__.copy()
        state.pop("_native_problem", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._native_problem = _prepare_native_glmm(self.mode_matrices, self.family)

    def evaluate(
        self, parameters: NDArray[np.floating]
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
        parameters = np.asarray(parameters)
        validate_finite_real("joint parameters", parameters, (self.n_parameters,))
        theta, beta = parameters[: self.n_theta], parameters[self.n_theta :]
        offset = self.matrices.offset + self.matrices.X @ beta
        if self._native_problem is not None and (self.nAGQ <= 1 or self.matrices.n_random):
            deviance, _, u, converged = _evaluate_native_problem(
                self._native_problem,
                theta,
                self.nAGQ,
                offset=offset,
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )
        else:
            matrices = replace(self.mode_matrices, offset=offset)
            deviance, _, u, converged = glmm_deviance_with_status(
                theta,
                matrices,
                self.family,
                nAGQ=max(1, self.nAGQ),
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )
        return deviance, beta.copy(), u, converged

    def __call__(self, parameters: NDArray[np.floating]) -> float:
        deviance, _, _, converged = self.evaluate(parameters)
        return float(deviance) if converged and np.isfinite(deviance) else np.inf

    def parameter_scale(self, theta: NDArray[np.floating]) -> NDArray[np.floating]:
        # Weighted column RMS makes coefficient steps insensitive to predictor units.
        weights = self.matrices.weights / np.max(self.matrices.weights)
        weights = weights / np.sum(weights)
        maxima = np.max(np.abs(self.matrices.X), axis=0, initial=0.0)
        normalized = np.divide(
            self.matrices.X, maxima, out=np.zeros_like(self.matrices.X), where=maxima > 0
        )
        column_scale = maxima * np.sqrt(np.sum(weights[:, None] * normalized**2, axis=0))
        beta_scale = np.divide(
            1.0,
            np.maximum(column_scale, np.finfo(float).tiny),
            out=np.ones_like(column_scale),
            where=column_scale > 0,
        )
        return np.r_[np.maximum(1.0, np.abs(theta)), beta_scale]

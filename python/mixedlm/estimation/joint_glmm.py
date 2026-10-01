"""GLMM likelihood evaluation with fixed coefficients outside the mode solve."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from mixedlm.estimation.laplace import _validate_quadrature, glmm_deviance_with_status
from mixedlm.estimation.pirls_control import validate_pirls_controls
from mixedlm.estimation.reml import _build_theta_bounds, _count_theta
from mixedlm.estimation.validation import validate_finite_real
from mixedlm.families.base import Family
from mixedlm.matrices.design import ModelMatrices


class JointGLMMObjective:
    """Evaluate parameters ordered as covariance parameters followed by beta."""

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
        self.bounds = _build_theta_bounds(matrices.random_structures, self.n_theta)
        self.bounds += [(None, None)] * matrices.n_fixed

    def evaluate(
        self, parameters: NDArray[np.floating]
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating], bool]:
        parameters = np.asarray(parameters)
        validate_finite_real("joint parameters", parameters, (self.n_parameters,))
        theta, beta = parameters[: self.n_theta], parameters[self.n_theta :]
        matrices = replace(self.mode_matrices, offset=self.matrices.offset + self.matrices.X @ beta)
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

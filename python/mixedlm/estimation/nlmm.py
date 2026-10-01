from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import minimize

from mixedlm.estimation.pnls_control import validate_pnls_controls
from mixedlm.nlme.models import (
    NonlinearModel,
    SSasymp,
    SSbiexp,
    SSfpl,
    SSgompertz,
    SSlogis,
    SSmicmen,
)

try:
    from mixedlm._rust import nlmm_deviance_with_status as _rust_nlmm_deviance_with_status

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


_RUST_MODEL_IMPLEMENTATIONS: dict[
    type[NonlinearModel],
    tuple[str, Callable[..., NDArray[np.floating]], Callable[..., NDArray[np.floating]]],
] = {
    model_type: (name, model_type.predict, model_type.gradient)
    for model_type, name in (
        (SSasymp, "ssasymp"),
        (SSlogis, "sslogis"),
        (SSmicmen, "ssmicmen"),
        (SSfpl, "ssfpl"),
        (SSgompertz, "ssgompertz"),
        (SSbiexp, "ssbiexp"),
    )
}
_PSI_REGULARIZATION = 1e-8
_PNLS_REGULARIZATION = 1e-6
_PNLS_MAX_ITER = 50
_PNLS_TOL = 1e-6
_MIN_VARIANCE = np.finfo(np.float64).tiny
_INVALID_OBJECTIVE = 1e100


def _get_rust_model_name(model: NonlinearModel) -> str | None:
    """Select native formulas only for their original Python implementations."""
    implementation = _RUST_MODEL_IMPLEMENTATIONS.get(type(model))
    if implementation is None:
        return None
    name, predict, gradient = implementation
    # Built-in instances can also have their methods replaced without subclassing.
    if (
        getattr(model.predict, "__func__", None) is not predict
        or getattr(model.gradient, "__func__", None) is not gradient
    ):
        return None
    return name


@dataclass
class NLMMOptimizationResult:
    phi: NDArray[np.floating]
    theta: NDArray[np.floating]
    sigma: float
    b: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int
    pnls_converged: bool = True


def _as_prior_weights(
    weights: NDArray[np.floating] | None,
    n: int,
) -> NDArray[np.float64]:
    if weights is None:
        return np.ones(n, dtype=np.float64)

    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if len(weights_array) != n:
        raise ValueError(f"weights has length {len(weights_array)}, expected {n}")
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("weights must contain only finite values")
    if np.any(weights_array <= 0.0):
        raise ValueError("weights must be strictly positive")
    return np.ascontiguousarray(weights_array)


def _weighted_standard_deviation(
    values: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    weight_sum = float(np.sum(weights))
    mean = float(np.dot(weights, values) / weight_sum)
    variance = float(np.dot(weights, (values - mean) ** 2) / weight_sum)
    return np.sqrt(max(variance, _MIN_VARIANCE))


def _build_psi_factor(
    theta: NDArray[np.floating],
    n_random: int,
) -> NDArray[np.floating]:
    if len(theta) == 0:
        return np.eye(n_random, dtype=np.float64)

    n_theta = len(theta)
    q = int((-1 + np.sqrt(1 + 8 * n_theta)) / 2)

    if q * (q + 1) // 2 != n_theta:
        q = int(np.sqrt(n_theta))
        return theta.reshape(q, q) if q * q == n_theta else np.diag(theta[:n_random])
    else:
        L = np.zeros((q, q), dtype=np.float64)
        row_indices, col_indices = np.tril_indices(q)
        L[row_indices, col_indices] = theta

    return L


def _build_psi_matrix(
    theta: NDArray[np.floating],
    n_random: int,
) -> NDArray[np.floating]:
    factor = _build_psi_factor(theta, n_random)
    return factor @ factor.T


def _grouped_observation_indices(groups: NDArray[np.integer]) -> list[NDArray[np.intp]]:
    """Return rows in observation order for each sorted group label."""
    order = np.argsort(groups, kind="stable")
    if len(order) == 0:
        return []
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    return list(np.split(order, boundaries))


_GroupResult = TypeVar("_GroupResult")


@dataclass
class _NLMMWorkspace:
    y: NDArray[np.floating]
    x: NDArray[np.floating]
    group_rows: list[NDArray[np.intp]]
    weights: NDArray[np.float64]
    sqrt_weights: NDArray[np.float64]
    executor: ThreadPoolExecutor | None

    def map_groups(
        self,
        function: Callable[[int, NDArray[np.intp]], _GroupResult],
        *,
        parallel: bool = True,
    ) -> Iterator[_GroupResult]:
        indices = range(len(self.group_rows))
        if not parallel or self.executor is None:
            yield from map(function, indices, self.group_rows)
            return
        futures = []
        try:
            for g, rows in enumerate(self.group_rows):
                futures.append(self.executor.submit(function, g, rows))
            for future in futures:
                yield future.result()
        except BaseException:
            # A failed trial must finish using the model before the next trial.
            for future in futures:
                future.cancel()
            wait(futures)
            raise


@contextmanager
def _nlmm_workspace(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    weights: NDArray[np.floating] | None,
    n_jobs: int,
) -> Iterator[_NLMMWorkspace]:
    """Reuse immutable preparation and scope worker lifetime to one call or fit."""
    prior_weights = _as_prior_weights(weights, len(y))
    group_rows = _grouped_observation_indices(groups)
    workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    use_parallel = workers > 1 and len(group_rows) >= workers
    with ThreadPoolExecutor(max_workers=workers) if use_parallel else nullcontext() as executor:
        yield _NLMMWorkspace(y, x, group_rows, prior_weights, np.sqrt(prior_weights), executor)


def _compute_group_resid_grad(
    g: int,
    rows: NDArray,
    x: NDArray,
    y: NDArray,
    phi: NDArray,
    b: NDArray,
    random_params: list[int],
    model: NonlinearModel,
) -> tuple[int, NDArray, NDArray, NDArray]:
    x_g = x[rows]
    y_g = y[rows]

    params_g = phi.copy()
    np.add.at(params_g, random_params, b[g, :])

    pred_g = model.predict(params_g, x_g)
    grad_g = model.gradient(params_g, x_g)

    return (g, rows, y_g - pred_g, grad_g)


def _update_group_random_effects(
    g: int,
    rows: NDArray,
    x: NDArray,
    y: NDArray,
    phi: NDArray,
    b: NDArray,
    random_params: list[int],
    model: NonlinearModel,
    Psi_inv: NDArray,
    weights: NDArray[np.floating],
) -> tuple[int, NDArray]:
    x_g = x[rows]
    y_g = y[rows]
    weights_g = weights[rows]

    params_g = phi.copy()
    np.add.at(params_g, random_params, b[g, :])

    pred_g = model.predict(params_g, x_g)
    grad_g = model.gradient(params_g, x_g)

    Z_g = grad_g[:, random_params]
    resid_g = y_g - pred_g + Z_g @ b[g, :]

    ZtZ = Z_g.T @ (weights_g[:, None] * Z_g)
    Ztr = Z_g.T @ (weights_g * resid_g)

    C = ZtZ + Psi_inv
    try:
        b_g = linalg.solve(C, Ztr, assume_a="pos")
    except linalg.LinAlgError:
        b_g = linalg.lstsq(C, Ztr)[0]

    return (g, b_g)


def _compute_group_rss(
    g: int,
    rows: NDArray,
    x: NDArray,
    y: NDArray,
    phi: NDArray,
    b: NDArray,
    random_params: list[int],
    model: NonlinearModel,
    weights: NDArray[np.floating],
) -> float:
    x_g = x[rows]
    y_g = y[rows]
    weights_g = weights[rows]

    params_g = phi.copy()
    np.add.at(params_g, random_params, b[g, :])

    pred_g = model.predict(params_g, x_g)
    residuals_g = y_g - pred_g
    return float(np.dot(weights_g, residuals_g**2))


def pnls_step(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    Psi: NDArray[np.floating],
    sigma: float,
    random_params: list[int],
    n_jobs: int = 1,
    weights: NDArray[np.floating] | None = None,
    *,
    maxiter: int | None = None,
    tol: float = _PNLS_TOL,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Update parameters with random-effect rows in sorted group-label order."""
    validate_pnls_controls(maxiter, tol)
    with _nlmm_workspace(y, x, groups, weights, n_jobs) as workspace:
        phi_new, b_new, sigma_sq, _ = _pnls_step(
            workspace,
            model,
            phi,
            b,
            Psi,
            random_params,
            _PNLS_MAX_ITER if maxiter is None else maxiter,
            tol,
        )
    return phi_new, b_new, np.sqrt(sigma_sq)


def _pnls_step(
    workspace: _NLMMWorkspace,
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    Psi: NDArray[np.floating],
    random_params: list[int],
    maxiter: int,
    tol: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, bool]:
    """Return updated parameters and the profiled residual variance."""
    y, x = workspace.y, workspace.x
    n = len(y)
    prior_weights = workspace.weights
    sqrt_weights = workspace.sqrt_weights
    n_phi = len(phi)
    n_random = len(random_params)

    Psi_reg = Psi + _PSI_REGULARIZATION * np.eye(n_random)
    try:
        Psi_chol = linalg.cholesky(Psi_reg, lower=True)
        Psi_inv = linalg.cho_solve((Psi_chol, True), np.eye(n_random))
    except linalg.LinAlgError:
        Psi_inv = linalg.pinv(Psi_reg)
        Psi_chol = None

    phi_new = phi.copy()
    b_new = b.copy()
    reg_phi = _PNLS_REGULARIZATION * np.eye(n_phi)
    resid_total = np.empty(n, dtype=np.float64)
    grad_total = np.empty((n, n_phi), dtype=np.float64)

    def residual_gradient(g: int, rows: NDArray[np.intp]) -> tuple[int, NDArray, NDArray, NDArray]:
        return _compute_group_resid_grad(g, rows, x, y, phi, b, random_params, model)

    def update_random_effects(g: int, rows: NDArray[np.intp]) -> tuple[int, NDArray]:
        return _update_group_random_effects(
            g, rows, x, y, phi_new, b, random_params, model, Psi_inv, prior_weights
        )

    converged = False
    for _iteration in range(maxiter):
        # After the first iteration b and b_new share storage.
        b_previous = b.copy()
        for _, rows, resid_g, grad_g in workspace.map_groups(residual_gradient):
            resid_total[rows] = resid_g
            grad_total[rows, :] = grad_g

        resid_total *= sqrt_weights
        grad_total *= sqrt_weights[:, None]
        GtG = grad_total.T @ grad_total
        Gtr = grad_total.T @ resid_total

        try:
            delta_phi = linalg.solve(GtG + reg_phi, Gtr, assume_a="pos")
        except linalg.LinAlgError:
            delta_phi = linalg.lstsq(GtG, Gtr)[0]

        phi_new = phi + 0.5 * delta_phi
        for g, b_g in workspace.map_groups(update_random_effects, parallel=Psi_chol is not None):
            b_new[g, :] = b_g

        max_delta = max(
            float(np.max(np.abs(phi_new - phi))),
            float(np.max(np.abs(b_new - b_previous))),
        )
        phi = phi_new
        b = b_new
        if max_delta < tol:
            converged = True
            break

    def group_rss(g: int, rows: NDArray[np.intp]) -> float:
        return _compute_group_rss(
            g, rows, x, y, phi_new, b_new, random_params, model, prior_weights
        )

    rss = sum(workspace.map_groups(group_rss))
    penalty = float(np.einsum("gi,ij,gj->", b_new, Psi_inv, b_new, optimize=True))
    sigma_sq = max((rss + penalty) / n, _MIN_VARIANCE)
    return phi_new, b_new, sigma_sq, converged


def nlmm_deviance(
    theta: NDArray[np.floating],
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    random_params: list[int],
    sigma: float,
    n_jobs: int = 1,
    weights: NDArray[np.floating] | None = None,
    *,
    pnls_maxiter: int | None = None,
    pnls_tol: float = _PNLS_TOL,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float]:
    """Evaluate deviance with random-effect rows in sorted group-label order."""
    return nlmm_deviance_with_status(
        theta,
        y,
        x,
        groups,
        model,
        phi,
        b,
        random_params,
        sigma,
        n_jobs=n_jobs,
        weights=weights,
        pnls_maxiter=pnls_maxiter,
        pnls_tol=pnls_tol,
    )[:4]


def nlmm_deviance_with_status(
    theta: NDArray[np.floating],
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    random_params: list[int],
    sigma: float,
    n_jobs: int = 1,
    weights: NDArray[np.floating] | None = None,
    *,
    pnls_maxiter: int | None = None,
    pnls_tol: float = _PNLS_TOL,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float, bool]:
    """Return the Python likelihood evaluation and its inner PNLS convergence flag."""
    validate_pnls_controls(pnls_maxiter, pnls_tol)
    with _nlmm_workspace(y, x, groups, weights, n_jobs) as workspace:
        return _nlmm_deviance(
            theta,
            workspace,
            model,
            phi,
            b,
            random_params,
            _PNLS_MAX_ITER if pnls_maxiter is None else pnls_maxiter,
            pnls_tol,
        )


def _nlmm_deviance(
    theta: NDArray[np.floating],
    workspace: _NLMMWorkspace,
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    random_params: list[int],
    pnls_maxiter: int,
    pnls_tol: float,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float, bool]:
    n = len(workspace.y)
    n_random = len(random_params)
    Psi_factor = _build_psi_factor(theta, n_random)
    Psi = Psi_factor @ Psi_factor.T
    phi_new, b_new, sigma_sq, converged = _pnls_step(
        workspace, model, phi, b, Psi, random_params, pnls_maxiter, pnls_tol
    )

    laplace_correction = 0.0
    identity = np.eye(n_random, dtype=np.float64)
    for g, rows in enumerate(workspace.group_rows):
        params_g = phi_new.copy()
        np.add.at(params_g, random_params, b_new[g, :])
        grad_g = model.gradient(params_g, workspace.x[rows])
        Z_g = grad_g[:, random_params]
        weights_g = workspace.weights[rows]
        # Stable form of log|Psi| + log|Z'WZ + Psi^-1| for Psi = L L'.
        ZtWZ = Z_g.T @ (weights_g[:, None] * Z_g)
        system = identity + Psi_factor.T @ ZtWZ @ Psi_factor
        sign, logdet = np.linalg.slogdet(system)
        if sign <= 0 or not np.isfinite(logdet):
            return _INVALID_OBJECTIVE, phi_new, b_new, np.sqrt(sigma_sq), False
        laplace_correction += logdet

    deviance = n * (1.0 + np.log(2.0 * np.pi * sigma_sq)) + laplace_correction

    return deviance, phi_new, b_new, np.sqrt(sigma_sq), converged


def _nlmm_deviance_rust(
    theta: NDArray[np.floating],
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    random_params: list[int],
    sigma: float,
    weights: NDArray[np.floating],
    *,
    pnls_maxiter: int | None = None,
    pnls_tol: float = _PNLS_TOL,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float]:
    return _nlmm_deviance_rust_with_status(
        theta,
        y,
        x,
        groups,
        model,
        phi,
        b,
        random_params,
        sigma,
        weights,
        pnls_maxiter=pnls_maxiter,
        pnls_tol=pnls_tol,
    )[:4]


def _nlmm_deviance_rust_with_status(
    theta: NDArray[np.floating],
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    random_params: list[int],
    sigma: float,
    weights: NDArray[np.floating],
    *,
    pnls_maxiter: int | None = None,
    pnls_tol: float = _PNLS_TOL,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float, bool]:
    validate_pnls_controls(pnls_maxiter, pnls_tol)
    model_name = _get_rust_model_name(model)
    if model_name is None:
        raise ValueError("Native nonlinear evaluation requires an unmodified built-in model")
    dev, phi_out, b_out, sigma_out, converged = _rust_nlmm_deviance_with_status(
        np.ascontiguousarray(theta, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(groups, dtype=np.int64),
        model_name,
        np.ascontiguousarray(phi, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        list(random_params),
        float(sigma),
        np.ascontiguousarray(weights, dtype=np.float64),
        maxiter=_PNLS_MAX_ITER if pnls_maxiter is None else pnls_maxiter,
        tol=pnls_tol,
    )
    return dev, np.array(phi_out), np.array(b_out), sigma_out, converged


class NLMMOptimizer:
    def __init__(
        self,
        y: NDArray[np.floating],
        x: NDArray[np.floating],
        groups: NDArray[np.integer],
        model: NonlinearModel,
        random_params: list[int],
        verbose: int = 0,
        use_rust: bool = True,
        n_jobs: int = 1,
        weights: NDArray[np.floating] | None = None,
        *,
        pnls_maxiter: int | None = None,
        pnls_tol: float = _PNLS_TOL,
    ) -> None:
        validate_pnls_controls(pnls_maxiter, pnls_tol)
        self.pnls_maxiter = _PNLS_MAX_ITER if pnls_maxiter is None else int(pnls_maxiter)
        self.pnls_tol = float(pnls_tol)
        self.y = y
        self.x = x
        self.groups = groups
        self.model = model
        self.random_params = random_params
        self.verbose = verbose
        self.use_rust = use_rust and _HAS_RUST and _get_rust_model_name(model) is not None
        self.n_jobs = n_jobs
        self.weights = _as_prior_weights(weights, len(y)).copy()

        self.n_groups = len(np.unique(groups))
        self.n_random = len(random_params)
        self.n_theta = self.n_random * (self.n_random + 1) // 2

        self._start_phi: NDArray[np.floating] | None = None
        self._start_b = np.zeros((self.n_groups, self.n_random), dtype=np.float64)
        self._start_sigma = _weighted_standard_deviation(self.y, self.weights)
        self._workspace: _NLMMWorkspace | None = None
        self._last_theta: NDArray[np.floating] | None = None
        self._last_failure: str | None = None
        self._last_evaluation: (
            tuple[
                float,
                NDArray[np.floating],
                NDArray[np.floating],
                float,
                bool,
            ]
            | None
        ) = None

    def get_start_theta(self) -> NDArray[np.floating]:
        theta = np.zeros(self.n_theta, dtype=np.float64)
        idx = 0
        for i in range(self.n_random):
            for j in range(i + 1):
                if i == j:
                    theta[idx] = 1.0
                idx += 1
        return theta

    def get_start_phi(self) -> NDArray[np.floating]:
        return self.model.get_start(self.x, self.y)

    def _evaluate(
        self,
        theta: NDArray[np.floating],
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float, bool]:
        if (
            self._last_theta is not None
            and np.array_equal(theta, self._last_theta)
            and self._last_evaluation is not None
        ):
            return self._last_evaluation

        if self._start_phi is None:
            self._start_phi = self.get_start_phi()

        self._last_failure = None
        try:
            if not np.all(np.isfinite(theta)):
                raise ValueError("variance parameters must be finite")
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                if self.use_rust:
                    evaluation = _nlmm_deviance_rust_with_status(
                        theta,
                        self.y,
                        self.x,
                        self.groups,
                        self.model,
                        self._start_phi,
                        self._start_b,
                        self.random_params,
                        self._start_sigma,
                        self.weights,
                        pnls_maxiter=self.pnls_maxiter,
                        pnls_tol=self.pnls_tol,
                    )
                elif self._workspace is not None:
                    evaluation = _nlmm_deviance(
                        theta,
                        self._workspace,
                        self.model,
                        self._start_phi,
                        self._start_b,
                        self.random_params,
                        self.pnls_maxiter,
                        self.pnls_tol,
                    )
                else:
                    evaluation = nlmm_deviance_with_status(
                        theta,
                        self.y,
                        self.x,
                        self.groups,
                        self.model,
                        self._start_phi,
                        self._start_b,
                        self.random_params,
                        self._start_sigma,
                        n_jobs=self.n_jobs,
                        weights=self.weights,
                        pnls_maxiter=self.pnls_maxiter,
                        pnls_tol=self.pnls_tol,
                    )

            deviance, phi, b, sigma, pnls_converged = evaluation
            if not isinstance(pnls_converged, (bool, np.bool_)):
                raise ValueError("PNLS convergence status must be a boolean")
            for name, values, shape in (
                ("deviance", deviance, ()),
                ("fixed parameters", phi, (self.model.n_params,)),
                ("random effects", b, (self.n_groups, self.n_random)),
                ("residual scale", sigma, ()),
            ):
                array = np.asarray(values)
                if array.shape != shape:
                    raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
                if not np.isrealobj(array) or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must contain finite real values")
            if deviance == _INVALID_OBJECTIVE:
                raise ValueError("deviance evaluation returned the failure penalty")
            if sigma <= 0:
                raise ValueError("residual scale must be strictly positive")
        except (
            FloatingPointError,
            OverflowError,
            TypeError,
            ValueError,
            linalg.LinAlgError,
        ) as exc:
            self._last_failure = f"{type(exc).__name__}: {exc}"
            evaluation = (
                _INVALID_OBJECTIVE,
                self._start_phi.copy(),
                self._start_b.copy(),
                self._start_sigma,
                False,
            )

        self._last_theta = theta.copy()
        self._last_evaluation = evaluation
        return evaluation

    def objective(self, theta: NDArray[np.floating]) -> float:
        deviance = float(self._evaluate(theta)[0])
        return deviance if np.isfinite(deviance) else _INVALID_OBJECTIVE

    def optimize(
        self,
        start_theta: NDArray[np.floating] | None = None,
        start_phi: NDArray[np.floating] | None = None,
        start_b: NDArray[np.floating] | None = None,
        start_sigma: float | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
    ) -> NLMMOptimizationResult:
        if start_theta is None:
            start_theta = self.get_start_theta()

        self._start_phi = start_phi.copy() if start_phi is not None else self.get_start_phi()
        if start_b is None:
            self._start_b = np.zeros((self.n_groups, self.n_random), dtype=np.float64)
        else:
            start_b_array = np.asarray(start_b, dtype=np.float64)
            expected_shape = (self.n_groups, self.n_random)
            if start_b_array.shape != expected_shape:
                raise ValueError(
                    f"start_b has shape {start_b_array.shape}, expected {expected_shape}"
                )
            self._start_b = start_b_array.copy()

        sigma = (
            _weighted_standard_deviation(self.y, self.weights)
            if start_sigma is None
            else float(start_sigma)
        )
        self._start_sigma = max(sigma, np.sqrt(_MIN_VARIANCE))
        self._last_theta = None
        self._last_evaluation = None
        self._last_failure = None

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * self.n_theta
        idx = 0
        for i in range(self.n_random):
            bounds[idx + i] = (1e-6, None)
            idx += i + 1

        callback: Callable[[NDArray[np.floating]], None] | None = None
        if self.verbose > 0:

            def callback(x: NDArray[np.floating]) -> None:
                dev = self.objective(x)
                print(f"theta = {x}, deviance = {dev:.6f}")

        context = (
            nullcontext()
            if self.use_rust
            else _nlmm_workspace(self.y, self.x, self.groups, self.weights, self.n_jobs)
        )
        with context as workspace:
            self._workspace = workspace
            try:
                result = minimize(
                    self.objective,
                    start_theta,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": maxiter},
                    callback=callback,
                )
                deviance, phi, b, sigma, pnls_converged = self._evaluate(result.x)
            finally:
                self._workspace = None

        if self._last_failure is not None:
            raise RuntimeError(
                f"Nonlinear optimization did not produce a valid fit: {self._last_failure}"
            )

        return NLMMOptimizationResult(
            phi=phi,
            theta=result.x,
            sigma=sigma,
            b=b,
            deviance=deviance,
            converged=bool(result.success and pnls_converged and np.isfinite(deviance)),
            pnls_converged=bool(pnls_converged),
            n_iter=result.nit,
        )

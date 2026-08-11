from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import minimize

from mixedlm.nlme.models import NonlinearModel

try:
    from mixedlm._rust import nlmm_deviance as _rust_nlmm_deviance

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


_SUPPORTED_RUST_MODELS = {"ssasymp", "sslogis", "ssmicmen", "ssfpl", "ssgompertz", "ssbiexp"}
_PSI_REGULARIZATION = 1e-8
_PNLS_REGULARIZATION = 1e-6
_PNLS_MAX_ITER = 50
_PNLS_TOL = 1e-6
_MIN_VARIANCE = np.finfo(np.float64).tiny
_INVALID_OBJECTIVE = 1e100


@dataclass
class NLMMOptimizationResult:
    phi: NDArray[np.floating]
    theta: NDArray[np.floating]
    sigma: float
    b: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int


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


def _compute_group_resid_grad(
    g: int,
    groups: NDArray,
    x: NDArray,
    y: NDArray,
    phi: NDArray,
    b: NDArray,
    random_params: list[int],
    model: NonlinearModel,
) -> tuple[int, NDArray, NDArray, NDArray]:
    mask = groups == g
    x_g = x[mask]
    y_g = y[mask]

    params_g = phi.copy()
    np.add.at(params_g, random_params, b[g, :])

    pred_g = model.predict(params_g, x_g)
    grad_g = model.gradient(params_g, x_g)

    return (g, mask, y_g - pred_g, grad_g)


def _update_group_random_effects(
    g: int,
    groups: NDArray,
    x: NDArray,
    y: NDArray,
    phi: NDArray,
    b: NDArray,
    random_params: list[int],
    model: NonlinearModel,
    Psi_chol: NDArray,
) -> tuple[int, NDArray]:
    mask = groups == g
    x_g = x[mask]
    y_g = y[mask]

    params_g = phi.copy()
    np.add.at(params_g, random_params, b[g, :])

    pred_g = model.predict(params_g, x_g)
    grad_g = model.gradient(params_g, x_g)

    Z_g = grad_g[:, random_params]
    resid_g = y_g - pred_g + Z_g @ b[g, :]

    ZtZ = Z_g.T @ Z_g
    Ztr = Z_g.T @ resid_g

    n_random = len(random_params)
    Psi_inv = linalg.cho_solve((Psi_chol, True), np.eye(n_random))
    C = ZtZ + Psi_inv
    try:
        b_g = linalg.solve(C, Ztr, assume_a="pos")
    except linalg.LinAlgError:
        b_g = linalg.lstsq(C, Ztr)[0]

    return (g, b_g)


def _compute_group_rss(
    g: int,
    groups: NDArray,
    x: NDArray,
    y: NDArray,
    phi: NDArray,
    b: NDArray,
    random_params: list[int],
    model: NonlinearModel,
) -> float:
    mask = groups == g
    x_g = x[mask]
    y_g = y[mask]

    params_g = phi.copy()
    np.add.at(params_g, random_params, b[g, :])

    pred_g = model.predict(params_g, x_g)
    return float(np.sum((y_g - pred_g) ** 2))


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
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    _ = sigma
    n = len(y)
    n_groups = len(np.unique(groups))
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

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    use_parallel = n_jobs > 1 and n_groups >= n_jobs

    reg_phi = _PNLS_REGULARIZATION * np.eye(n_phi)

    for _iteration in range(_PNLS_MAX_ITER):
        resid_total = np.zeros(n, dtype=np.float64)
        grad_total = np.zeros((n, n_phi), dtype=np.float64)

        if use_parallel:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                residual_futures = [
                    executor.submit(
                        _compute_group_resid_grad, g, groups, x, y, phi, b, random_params, model
                    )
                    for g in range(n_groups)
                ]
                for residual_future in residual_futures:
                    g, mask, resid_g, grad_g = residual_future.result()
                    resid_total[mask] = resid_g
                    grad_total[mask, :] = grad_g
        else:
            for g in range(n_groups):
                mask = groups == g
                x_g = x[mask]
                y_g = y[mask]

                params_g = phi.copy()
                np.add.at(params_g, random_params, b[g, :])

                pred_g = model.predict(params_g, x_g)
                grad_g = model.gradient(params_g, x_g)

                resid_total[mask] = y_g - pred_g
                grad_total[mask, :] = grad_g

        GtG = grad_total.T @ grad_total
        Gtr = grad_total.T @ resid_total

        try:
            delta_phi = linalg.solve(GtG + reg_phi, Gtr, assume_a="pos")
        except linalg.LinAlgError:
            delta_phi = linalg.lstsq(GtG, Gtr)[0]

        phi_new = phi + 0.5 * delta_phi

        if use_parallel and Psi_chol is not None:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                update_futures = [
                    executor.submit(
                        _update_group_random_effects,
                        g,
                        groups,
                        x,
                        y,
                        phi_new,
                        b,
                        random_params,
                        model,
                        Psi_chol,
                    )
                    for g in range(n_groups)
                ]
                for update_future in update_futures:
                    result = update_future.result()
                    b_new[result[0], :] = result[1]
        else:
            for g in range(n_groups):
                mask = groups == g
                x_g = x[mask]
                y_g = y[mask]

                params_g = phi_new.copy()
                np.add.at(params_g, random_params, b[g, :])

                pred_g = model.predict(params_g, x_g)
                grad_g = model.gradient(params_g, x_g)

                Z_g = grad_g[:, random_params]
                resid_g = y_g - pred_g + Z_g @ b[g, :]

                ZtZ = Z_g.T @ Z_g
                Ztr = Z_g.T @ resid_g

                C = ZtZ + Psi_inv
                try:
                    b_new[g, :] = linalg.solve(C, Ztr, assume_a="pos")
                except linalg.LinAlgError:
                    b_new[g, :] = linalg.lstsq(C, Ztr)[0]

        max_delta = max(
            float(np.max(np.abs(phi_new - phi))),
            float(np.max(np.abs(b_new - b))),
        )

        phi = phi_new
        b = b_new
        if max_delta < _PNLS_TOL:
            break

    rss: float
    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            rss_futures = [
                executor.submit(
                    _compute_group_rss,
                    g,
                    groups,
                    x,
                    y,
                    phi_new,
                    b_new,
                    random_params,
                    model,
                )
                for g in range(n_groups)
            ]
            rss = float(sum(future.result() for future in rss_futures))
    else:
        rss = sum(
            _compute_group_rss(g, groups, x, y, phi_new, b_new, random_params, model)
            for g in range(n_groups)
        )

    penalty = float(np.einsum("gi,ij,gj->", b_new, Psi_inv, b_new, optimize=True))
    sigma_new = np.sqrt(max((rss + penalty) / n, _MIN_VARIANCE))

    return phi_new, b_new, sigma_new


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
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float]:
    n = len(y)
    n_groups = len(np.unique(groups))
    n_random = len(random_params)

    Psi_factor = _build_psi_factor(theta, n_random)
    Psi = Psi_factor @ Psi_factor.T

    phi_new, b_new, _sigma_new = pnls_step(
        y, x, groups, model, phi, b, Psi, sigma, random_params, n_jobs=n_jobs
    )

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    use_parallel = n_jobs > 1 and n_groups >= n_jobs

    rss: float
    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            rss_futures = [
                executor.submit(
                    _compute_group_rss,
                    g,
                    groups,
                    x,
                    y,
                    phi_new,
                    b_new,
                    random_params,
                    model,
                )
                for g in range(n_groups)
            ]
            rss = float(sum(future.result() for future in rss_futures))
    else:
        rss = sum(
            _compute_group_rss(g, groups, x, y, phi_new, b_new, random_params, model)
            for g in range(n_groups)
        )

    Psi_reg = Psi + _PSI_REGULARIZATION * np.eye(n_random)
    try:
        Psi_chol = linalg.cholesky(Psi_reg, lower=True)
        Psi_inv = linalg.cho_solve((Psi_chol, True), np.eye(n_random))
    except linalg.LinAlgError:
        Psi_inv = linalg.pinv(Psi_reg)

    penalty = float(np.einsum("gi,ij,gj->", b_new, Psi_inv, b_new, optimize=True))
    pwrss = rss + penalty
    sigma_sq = max(pwrss / n, _MIN_VARIANCE)

    laplace_correction = 0.0
    identity = np.eye(n_random, dtype=np.float64)
    for g in range(n_groups):
        mask = groups == g
        params_g = phi_new.copy()
        np.add.at(params_g, random_params, b_new[g, :])
        grad_g = model.gradient(params_g, x[mask])
        Z_g = grad_g[:, random_params]
        # Stable form of log|Psi| + log|Z'Z + Psi^-1| for Psi = L L'.
        system = identity + Psi_factor.T @ (Z_g.T @ Z_g) @ Psi_factor
        sign, logdet = np.linalg.slogdet(system)
        if sign <= 0 or not np.isfinite(logdet):
            return _INVALID_OBJECTIVE, phi_new, b_new, np.sqrt(sigma_sq)
        laplace_correction += logdet

    deviance = n * (1.0 + np.log(2.0 * np.pi * sigma_sq)) + laplace_correction

    return deviance, phi_new, b_new, np.sqrt(sigma_sq)


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
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float]:
    dev, phi_out, b_out, sigma_out = _rust_nlmm_deviance(
        theta.astype(np.float64),
        y.astype(np.float64),
        x.astype(np.float64),
        groups.astype(np.int64),
        model.name.lower(),
        phi.astype(np.float64),
        b.astype(np.float64),
        list(random_params),
        float(sigma),
    )
    return dev, np.array(phi_out), np.array(b_out), sigma_out


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
    ) -> None:
        self.y = y
        self.x = x
        self.groups = groups
        self.model = model
        self.random_params = random_params
        self.verbose = verbose
        self.use_rust = use_rust and _HAS_RUST and model.name.lower() in _SUPPORTED_RUST_MODELS
        self.n_jobs = n_jobs
        self.weights = weights if weights is not None else np.ones(len(y), dtype=np.float64)

        self.n_groups = len(np.unique(groups))
        self.n_random = len(random_params)
        self.n_theta = self.n_random * (self.n_random + 1) // 2

        self._start_phi: NDArray[np.floating] | None = None
        self._start_b = np.zeros((self.n_groups, self.n_random), dtype=np.float64)
        self._start_sigma = max(float(np.std(self.y)), np.sqrt(_MIN_VARIANCE))
        self._last_theta: NDArray[np.floating] | None = None
        self._last_evaluation: (
            tuple[
                float,
                NDArray[np.floating],
                NDArray[np.floating],
                float,
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
    ) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float]:
        if (
            self._last_theta is not None
            and np.array_equal(theta, self._last_theta)
            and self._last_evaluation is not None
        ):
            return self._last_evaluation

        if self._start_phi is None:
            self._start_phi = self.get_start_phi()

        invalid_evaluation: tuple[
            float,
            NDArray[np.floating],
            NDArray[np.floating],
            float,
        ] = (
            _INVALID_OBJECTIVE,
            self._start_phi.copy(),
            self._start_b.copy(),
            self._start_sigma,
        )

        if not np.all(np.isfinite(theta)):
            evaluation = invalid_evaluation
        else:
            try:
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    if self.use_rust:
                        evaluation = _nlmm_deviance_rust(
                            theta,
                            self.y,
                            self.x,
                            self.groups,
                            self.model,
                            self._start_phi,
                            self._start_b,
                            self.random_params,
                            self._start_sigma,
                        )
                    else:
                        evaluation = nlmm_deviance(
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
                        )
            except (FloatingPointError, OverflowError, ValueError, linalg.LinAlgError):
                evaluation = invalid_evaluation

        if not np.isfinite(evaluation[0]):
            evaluation = invalid_evaluation

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

        sigma = float(np.std(self.y)) if start_sigma is None else float(start_sigma)
        self._start_sigma = max(sigma, np.sqrt(_MIN_VARIANCE))
        self._last_theta = None
        self._last_evaluation = None

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

        result = minimize(
            self.objective,
            start_theta,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter},
            callback=callback,
        )

        deviance, phi, b, sigma = self._evaluate(result.x)

        return NLMMOptimizationResult(
            phi=phi,
            theta=result.x,
            sigma=sigma,
            b=b,
            deviance=deviance,
            converged=bool(result.success and np.isfinite(deviance)),
            n_iter=result.nit,
        )

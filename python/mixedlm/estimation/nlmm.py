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
    from mixedlm._rust import pnls_step as _rust_pnls_step

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


_SUPPORTED_RUST_MODELS = {"ssasymp", "sslogis", "ssmicmen", "ssfpl", "ssgompertz", "ssbiexp"}


@dataclass
class NLMMOptimizationResult:
    phi: NDArray[np.floating]
    theta: NDArray[np.floating]
    sigma: float
    b: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int


def _build_psi_matrix(
    theta: NDArray[np.floating],
    n_random: int,
) -> NDArray[np.floating]:
    if len(theta) == 0:
        return np.eye(n_random, dtype=np.float64)

    n_theta = len(theta)
    q = int((-1 + np.sqrt(1 + 8 * n_theta)) / 2)

    if q * (q + 1) // 2 != n_theta:
        q = int(np.sqrt(n_theta))
        L = theta.reshape(q, q) if q * q == n_theta else np.diag(theta[:n_random])
    else:
        L = np.zeros((q, q), dtype=np.float64)
        row_indices, col_indices = np.tril_indices(q)
        L[row_indices, col_indices] = theta

    return L @ L.T


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
    sigma: float,
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
    C = ZtZ / sigma**2 + Psi_inv
    try:
        b_g = linalg.solve(C, Ztr / sigma**2, assume_a="pos")
    except linalg.LinAlgError:
        b_g = linalg.lstsq(C, Ztr / sigma**2)[0]

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
    n = len(y)
    n_groups = len(np.unique(groups))
    n_phi = len(phi)
    n_random = len(random_params)

    Psi_reg = Psi + 1e-8 * np.eye(n_random)
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

    reg_phi = 1e-6 * np.eye(n_phi)

    for _iteration in range(10):
        resid_total = np.zeros(n, dtype=np.float64)
        grad_total = np.zeros((n, n_phi), dtype=np.float64)

        if use_parallel:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(
                        _compute_group_resid_grad, g, groups, x, y, phi, b, random_params, model
                    )
                    for g in range(n_groups)
                ]
                for future in futures:
                    g, mask, resid_g, grad_g = future.result()
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
                futures = [
                    executor.submit(
                        _update_group_random_effects,  # type: ignore[arg-type]
                        g,
                        groups,
                        x,
                        y,
                        phi_new,
                        b,
                        random_params,
                        model,
                        Psi_chol,
                        sigma,
                    )
                    for g in range(n_groups)
                ]
                for future in futures:
                    result = future.result()
                    b_new[result[0], :] = result[1]
        else:
            sigma2 = sigma**2
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

                C = ZtZ / sigma2 + Psi_inv
                try:
                    b_new[g, :] = linalg.solve(C, Ztr / sigma2, assume_a="pos")
                except linalg.LinAlgError:
                    b_new[g, :] = linalg.lstsq(C, Ztr / sigma2)[0]

        if np.max(np.abs(phi_new - phi)) < 1e-6:
            break

        phi = phi_new
        b = b_new

    rss: float
    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _compute_group_rss,  # type: ignore[arg-type]
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
            rss = float(sum(future.result() for future in futures))  # type: ignore[misc]
    else:
        rss = 0.0
        for g in range(n_groups):
            mask = groups == g
            x_g = x[mask]
            y_g = y[mask]

            params_g = phi_new.copy()
            np.add.at(params_g, random_params, b_new[g, :])

            pred_g = model.predict(params_g, x_g)
            rss += float(np.sum((y_g - pred_g) ** 2))

    sigma_new = np.sqrt(rss / n)

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

    Psi = _build_psi_matrix(theta, n_random)

    phi_new, b_new, sigma_new = pnls_step(
        y, x, groups, model, phi, b, Psi, sigma, random_params, n_jobs=n_jobs
    )

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    use_parallel = n_jobs > 1 and n_groups >= n_jobs

    rss: float
    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _compute_group_rss,  # type: ignore[arg-type]
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
            rss = float(sum(future.result() for future in futures))  # type: ignore[misc]
    else:
        rss = 0.0
        for g in range(n_groups):
            mask = groups == g
            x_g = x[mask]
            y_g = y[mask]

            params_g = phi_new.copy()
            np.add.at(params_g, random_params, b_new[g, :])

            pred_g = model.predict(params_g, x_g)
            rss += float(np.sum((y_g - pred_g) ** 2))

    deviance = n * np.log(2 * np.pi * sigma_new**2) + rss / sigma_new**2

    Psi_reg = Psi + 1e-8 * np.eye(n_random)
    try:
        for g in range(n_groups):
            deviance += b_new[g, :] @ linalg.solve(Psi_reg, b_new[g, :], assume_a="pos")
    except linalg.LinAlgError:
        Psi_inv = linalg.pinv(Psi_reg)
        for g in range(n_groups):
            deviance += b_new[g, :] @ Psi_inv @ b_new[g, :]

    sign, logdet = np.linalg.slogdet(Psi)
    if sign > 0:
        deviance += n_groups * logdet

    return deviance, phi_new, b_new, sigma_new


def _pnls_step_rust(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    groups: NDArray[np.integer],
    model: NonlinearModel,
    phi: NDArray[np.floating],
    b: NDArray[np.floating],
    theta: NDArray[np.floating],
    sigma: float,
    random_params: list[int],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    phi_out, b_out, sigma_out = _rust_pnls_step(
        y.astype(np.float64),
        x.astype(np.float64),
        groups.astype(np.int64),
        model.name.lower(),
        phi.astype(np.float64),
        b.astype(np.float64),
        theta.astype(np.float64),
        float(sigma),
        list(random_params),
    )
    return np.array(phi_out), np.array(b_out), sigma_out


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


def pnls_step_fast(
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
    if _HAS_RUST and model.name.lower() in _SUPPORTED_RUST_MODELS:
        n_random = len(random_params)
        n_theta = n_random * (n_random + 1) // 2
        theta = np.zeros(n_theta, dtype=np.float64)
        idx = 0
        for i in range(n_random):
            for j in range(i + 1):
                if i == j:
                    theta[idx] = 1.0
                idx += 1
        return _pnls_step_rust(y, x, groups, model, phi, b, theta, sigma, random_params)
    return pnls_step(y, x, groups, model, phi, b, Psi, sigma, random_params, n_jobs=n_jobs)


def nlmm_deviance_fast(
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
    if _HAS_RUST and model.name.lower() in _SUPPORTED_RUST_MODELS:
        return _nlmm_deviance_rust(theta, y, x, groups, model, phi, b, random_params, sigma)
    return nlmm_deviance(theta, y, x, groups, model, phi, b, random_params, sigma, n_jobs=n_jobs)


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
    ) -> None:
        self.y = y
        self.x = x
        self.groups = groups
        self.model = model
        self.random_params = random_params
        self.verbose = verbose
        self.use_rust = use_rust and _HAS_RUST and model.name.lower() in _SUPPORTED_RUST_MODELS
        self.n_jobs = n_jobs

        self.n_groups = len(np.unique(groups))
        self.n_random = len(random_params)
        self.n_theta = self.n_random * (self.n_random + 1) // 2

        self._phi_cache: NDArray[np.floating] | None = None
        self._b_cache: NDArray[np.floating] | None = None
        self._sigma_cache: float = 1.0

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

    def objective(self, theta: NDArray[np.floating]) -> float:
        if self._phi_cache is None:
            self._phi_cache = self.get_start_phi()
        if self._b_cache is None:
            self._b_cache = np.zeros((self.n_groups, self.n_random), dtype=np.float64)

        if self.use_rust:
            dev, phi, b, sigma = _nlmm_deviance_rust(
                theta,
                self.y,
                self.x,
                self.groups,
                self.model,
                self._phi_cache,
                self._b_cache,
                self.random_params,
                self._sigma_cache,
            )
        else:
            dev, phi, b, sigma = nlmm_deviance(
                theta,
                self.y,
                self.x,
                self.groups,
                self.model,
                self._phi_cache,
                self._b_cache,
                self.random_params,
                self._sigma_cache,
                n_jobs=self.n_jobs,
            )

        self._phi_cache = phi
        self._b_cache = b
        self._sigma_cache = sigma

        return dev

    def optimize(
        self,
        start_theta: NDArray[np.floating] | None = None,
        start_phi: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
    ) -> NLMMOptimizationResult:
        if start_theta is None:
            start_theta = self.get_start_theta()

        if start_phi is not None:
            self._phi_cache = start_phi
        else:
            self._phi_cache = self.get_start_phi()

        self._b_cache = np.zeros((self.n_groups, self.n_random), dtype=np.float64)
        self._sigma_cache = np.std(self.y)

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

        return NLMMOptimizationResult(
            phi=self._phi_cache,
            theta=result.x,
            sigma=self._sigma_cache,
            b=self._b_cache,
            deviance=result.fun,
            converged=result.success,
            n_iter=result.nit,
        )

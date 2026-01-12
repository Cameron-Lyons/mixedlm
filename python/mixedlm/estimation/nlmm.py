from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import minimize

from mixedlm.nlme.models import NonlinearModel


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
        idx = 0
        for i in range(q):
            for j in range(i + 1):
                L[i, j] = theta[idx]
                idx += 1

    return L @ L.T


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
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    n = len(y)
    n_groups = len(np.unique(groups))
    n_phi = len(phi)
    n_random = len(random_params)

    Psi_inv = linalg.inv(Psi + 1e-8 * np.eye(n_random))

    phi_new = phi.copy()
    b_new = b.copy()

    for _iteration in range(10):
        resid_total = np.zeros(n, dtype=np.float64)
        grad_total = np.zeros((n, n_phi), dtype=np.float64)

        for g in range(n_groups):
            mask = groups == g
            x_g = x[mask]
            y_g = y[mask]
            np.sum(mask)

            params_g = phi.copy()
            for j, p_idx in enumerate(random_params):
                params_g[p_idx] += b[g, j]

            pred_g = model.predict(params_g, x_g)
            grad_g = model.gradient(params_g, x_g)

            resid_total[mask] = y_g - pred_g
            grad_total[mask, :] = grad_g

        GtG = grad_total.T @ grad_total
        Gtr = grad_total.T @ resid_total

        try:
            delta_phi = linalg.solve(GtG + 1e-6 * np.eye(n_phi), Gtr, assume_a="pos")
        except linalg.LinAlgError:
            delta_phi = linalg.lstsq(GtG, Gtr)[0]

        phi_new = phi + 0.5 * delta_phi

        for g in range(n_groups):
            mask = groups == g
            x_g = x[mask]
            y_g = y[mask]

            params_g = phi_new.copy()
            for j, p_idx in enumerate(random_params):
                params_g[p_idx] += b[g, j]

            pred_g = model.predict(params_g, x_g)
            grad_g = model.gradient(params_g, x_g)

            Z_g = grad_g[:, random_params]
            resid_g = y_g - pred_g + Z_g @ b[g, :]

            ZtZ = Z_g.T @ Z_g
            Ztr = Z_g.T @ resid_g

            C = ZtZ / sigma**2 + Psi_inv
            try:
                b_new[g, :] = linalg.solve(C, Ztr / sigma**2, assume_a="pos")
            except linalg.LinAlgError:
                b_new[g, :] = linalg.lstsq(C, Ztr / sigma**2)[0]

        if np.max(np.abs(phi_new - phi)) < 1e-6:
            break

        phi = phi_new
        b = b_new

    rss = 0.0
    for g in range(n_groups):
        mask = groups == g
        x_g = x[mask]
        y_g = y[mask]

        params_g = phi_new.copy()
        for j, p_idx in enumerate(random_params):
            params_g[p_idx] += b_new[g, j]

        pred_g = model.predict(params_g, x_g)
        rss += np.sum((y_g - pred_g) ** 2)

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
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], float]:
    n = len(y)
    n_groups = len(np.unique(groups))
    n_random = len(random_params)

    Psi = _build_psi_matrix(theta, n_random)

    phi_new, b_new, sigma_new = pnls_step(y, x, groups, model, phi, b, Psi, sigma, random_params)

    rss = 0.0
    for g in range(n_groups):
        mask = groups == g
        x_g = x[mask]
        y_g = y[mask]

        params_g = phi_new.copy()
        for j, p_idx in enumerate(random_params):
            params_g[p_idx] += b_new[g, j]

        pred_g = model.predict(params_g, x_g)
        rss += np.sum((y_g - pred_g) ** 2)

    deviance = n * np.log(2 * np.pi * sigma_new**2) + rss / sigma_new**2

    Psi_inv = linalg.inv(Psi + 1e-8 * np.eye(n_random))
    for g in range(n_groups):
        deviance += b_new[g, :] @ Psi_inv @ b_new[g, :]

    sign, logdet = np.linalg.slogdet(Psi)
    if sign > 0:
        deviance += n_groups * logdet

    return deviance, phi_new, b_new, sigma_new


class NLMMOptimizer:
    def __init__(
        self,
        y: NDArray[np.floating],
        x: NDArray[np.floating],
        groups: NDArray[np.integer],
        model: NonlinearModel,
        random_params: list[int],
        verbose: int = 0,
    ) -> None:
        self.y = y
        self.x = x
        self.groups = groups
        self.model = model
        self.random_params = random_params
        self.verbose = verbose

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

        bounds: list[tuple[float | None, float | None]] = []
        idx = 0
        for i in range(self.n_random):
            for j in range(i + 1):
                if i == j:
                    bounds.append((1e-6, None))
                else:
                    bounds.append((None, None))
                idx += 1

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

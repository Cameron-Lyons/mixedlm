from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, xlogy

from mixedlm.families.base import Family, Link
from mixedlm.families.likelihood import likelihood_inputs


class Gamma(Family):
    mean_bounds = (0.0, None)

    def __init__(self, link: str | Link | None = None) -> None:
        super().__init__(
            link,
            default_link="log",
            allowed_links=("log", "inverse", "identity"),
        )

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        mu = self.clamp_mu(mu)
        return mu**2

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = self.clamp_mu(mu, eps=eps)
        y = np.maximum(y, eps)

        ratio = y / mu
        return 2 * wt * (ratio - 1 - np.log(ratio))

    def log_likelihood(
        self,
        y: NDArray[np.floating],
        mu: NDArray[np.floating],
        wt: NDArray[np.floating],
        *,
        trials: NDArray[np.floating] | None = None,
    ) -> float:
        y, mu, wt = likelihood_inputs(y, mu, wt)
        if np.any(y <= 0) or np.any(mu <= 0):
            raise ValueError("Gamma likelihood responses and means must be positive")
        # Unit dispersion with precision weights gives shape=wt, scale=mu/wt.
        return float(
            np.sum(xlogy(wt - 1, y) - wt * y / mu - gammaln(wt) - wt * (np.log(mu) - np.log(wt)))
        )

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        mu = np.minimum(self.clamp_mu(mu, eps=1e-6), 1e10)
        return rng.gamma(1.0, mu)


class GammaInverse(Gamma):
    def __init__(self) -> None:
        super().__init__(link="inverse")

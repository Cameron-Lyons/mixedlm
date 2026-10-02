from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, xlogy

from mixedlm.families.base import Family, Link
from mixedlm.families.likelihood import likelihood_inputs, whole_counts


class Poisson(Family):
    mean_bounds = (0.0, None)

    def __init__(self, link: str | Link | None = None) -> None:
        super().__init__(link, default_link="log", allowed_links=("log", "identity", "sqrt"))

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        mu = self.clamp_mu(mu)

        term = xlogy(y, y / mu)
        return 2 * wt * (term - (y - mu))

    def log_likelihood(
        self,
        y: NDArray[np.floating],
        mu: NDArray[np.floating],
        wt: NDArray[np.floating],
        *,
        trials: NDArray[np.floating] | None = None,
    ) -> float:
        y, mu, wt = likelihood_inputs(y, mu, wt)
        y = whole_counts(y, "Poisson")
        if np.any(mu < 0):
            raise ValueError("Poisson likelihood means must be nonnegative")
        return float(np.sum(wt * (xlogy(y, mu) - mu - gammaln(y + 1))))

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        mu = np.minimum(self.clamp_mu(mu, eps=1e-6), 1e15)
        return rng.poisson(mu).astype(np.float64)

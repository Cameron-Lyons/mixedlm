from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from mixedlm.families.base import Family, LogitLink


class Binomial(Family):
    mean_bounds = (0.0, 1.0)

    def __init__(self) -> None:
        self.link = LogitLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu * (1 - mu)

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        mu = self.clamp_mu(mu)

        term1 = xlogy(y, y / mu)
        term2 = xlogy(1 - y, (1 - y) / (1 - mu))

        return 2 * wt * (term1 + term2)

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        mu = self.clamp_mu(mu, eps=1e-6)
        return rng.binomial(1, mu).astype(np.float64)

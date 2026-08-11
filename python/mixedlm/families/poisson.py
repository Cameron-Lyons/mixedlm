from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from mixedlm.families.base import Family, LogLink


class Poisson(Family):
    mean_bounds = (0.0, None)

    def __init__(self) -> None:
        self.link = LogLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        mu = self.clamp_mu(mu)

        term = xlogy(y, y / mu)
        return 2 * wt * (term - (y - mu))

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        mu = np.minimum(self.clamp_mu(mu, eps=1e-6), 1e15)
        return rng.poisson(mu).astype(np.float64)

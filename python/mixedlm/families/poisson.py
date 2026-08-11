from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from mixedlm.families.base import Family, LogLink


class Poisson(Family):
    mu_lower_bound = 0.0

    def __init__(self) -> None:
        self.link = LogLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)

        term = xlogy(y, y / mu)
        return 2 * wt * (term - (y - mu))

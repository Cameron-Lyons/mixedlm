from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from mixedlm.families.base import Family, LogitLink


class Binomial(Family):
    def __init__(self) -> None:
        self.link = LogitLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu * (1 - mu)

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.clip(mu, eps, 1 - eps)

        term1 = xlogy(y, y / mu)
        term2 = xlogy(1 - y, (1 - y) / (1 - mu))

        return 2 * wt * (term1 + term2)

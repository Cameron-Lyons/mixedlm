from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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

        term1 = np.where(y > 0, y * np.log(y / mu), 0)
        term2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0)

        return 2 * wt * (term1 + term2)

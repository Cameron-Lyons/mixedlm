from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, LogLink


class Poisson(Family):
    def __init__(self) -> None:
        self.link = LogLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)

        term = np.where(y > 0, y * np.log(y / mu), 0)
        return 2 * wt * (term - (y - mu))

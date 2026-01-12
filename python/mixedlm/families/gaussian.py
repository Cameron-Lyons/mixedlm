from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, IdentityLink


class Gaussian(Family):
    def __init__(self) -> None:
        self.link = IdentityLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones_like(mu)

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return wt * (y - mu) ** 2

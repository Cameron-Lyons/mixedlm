from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, Link


class Gamma(Family):
    mu_lower_bound = 0.0

    def __init__(self, link: str | Link | None = None) -> None:
        super().__init__(
            link,
            default_link="log",
            allowed_links=("log", "inverse", "identity"),
        )

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)
        return mu**2

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)
        y = np.maximum(y, eps)

        return 2 * wt * (-((y - mu) / mu) + np.log(y / mu))


class GammaInverse(Gamma):
    def __init__(self) -> None:
        super().__init__(link="inverse")

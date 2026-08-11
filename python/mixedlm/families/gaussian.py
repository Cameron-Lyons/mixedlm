from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, Link


class Gaussian(Family):
    def __init__(self, link: str | Link | None = None) -> None:
        super().__init__(link, default_link="identity", allowed_links=("identity", "log"))

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones_like(mu)

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return wt * (y - mu) ** 2

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        return rng.normal(mu, 1.0)

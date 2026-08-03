from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, Link
from mixedlm.families.base import InverseSquaredLink as _InverseSquaredLink

InverseSquaredLink = _InverseSquaredLink


class InverseGaussian(Family):
    mu_lower_bound = 0.0

    def __init__(self, link: str | Link | None = None) -> None:
        super().__init__(
            link,
            default_link="log",
            allowed_links=("log", "inverse", "identity", "1/mu^2"),
        )

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)
        return mu**3

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)
        y = np.maximum(y, eps)

        return wt * ((y - mu) ** 2) / (mu**2 * y)


class InverseGaussianCanonical(InverseGaussian):
    def __init__(self) -> None:
        super().__init__(link="1/mu^2")

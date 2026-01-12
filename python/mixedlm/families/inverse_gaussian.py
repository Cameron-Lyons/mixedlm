from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, Link, LogLink


class InverseSquaredLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (mu**2)

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        eta = np.maximum(eta, 1e-10)
        return 1.0 / np.sqrt(eta)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return -2.0 / (mu**3)


class InverseGaussian(Family):
    def __init__(self, link: Link | None = None) -> None:
        self.link = link if link is not None else LogLink()

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
        super().__init__(link=InverseSquaredLink())

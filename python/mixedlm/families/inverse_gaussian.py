from __future__ import annotations

from typing import Any

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
    mean_bounds = (0.0, None)

    def __init__(self, link: Link | None = None) -> None:
        self.link = link if link is not None else LogLink()

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        mu = self.clamp_mu(mu)
        return mu**3

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = self.clamp_mu(mu, eps=eps)
        y = np.maximum(y, eps)

        return wt * ((y - mu) ** 2) / (mu**2 * y)

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        mu = np.minimum(self.clamp_mu(mu, eps=1e-6), 1e10)
        return rng.wald(mu, 1.0)


class InverseGaussianCanonical(InverseGaussian):
    def __init__(self) -> None:
        super().__init__(link=InverseSquaredLink())

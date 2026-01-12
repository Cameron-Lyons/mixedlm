from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from mixedlm.families.base import Family, LogLink


class NegativeBinomial(Family):
    def __init__(self, theta: float = 1.0) -> None:
        self.link = LogLink()
        self.theta = theta

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu + mu**2 / self.theta

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        eps = 1e-10
        mu = np.maximum(mu, eps)
        theta = self.theta

        term1 = np.where(y > 0, y * np.log(np.maximum(y, eps) / mu), 0)
        term2 = (y + theta) * np.log((y + theta) / (mu + theta))

        return 2 * wt * (term1 - term2)

    def log_likelihood(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> float:
        theta = self.theta
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (mu + theta))
            + y * np.log(mu / (mu + theta))
        )
        return float(np.sum(wt * ll))

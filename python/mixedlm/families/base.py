from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Link(ABC):
    @abstractmethod
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        pass


class IdentityLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return eta

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones_like(mu)


class LogLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.log(mu)

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.exp(eta)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / mu


class LogitLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.log(mu / (1 - mu))

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (1.0 + np.exp(-eta))

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (mu * (1 - mu))


class ProbitLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        from scipy import stats

        return stats.norm.ppf(mu)

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        from scipy import stats

        return stats.norm.cdf(eta)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        from scipy import stats

        return 1.0 / stats.norm.pdf(stats.norm.ppf(mu))


class CloglogLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.log(-np.log(1 - mu))

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1 - np.exp(-np.exp(eta))

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / ((1 - mu) * (-np.log(1 - mu)))


class InverseLink(Link):
    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / mu

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / eta

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return -1.0 / (mu**2)


class Family(ABC):
    link: Link

    @abstractmethod
    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        pass

    def weights(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (self.link.deriv(mu) ** 2 * self.variance(mu))

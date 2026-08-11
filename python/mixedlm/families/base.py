from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


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
        return expit(eta)

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
    mu_lower_bound: float | None = None
    mu_upper_bound: float | None = None

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

    def clip_mu(self, mu: NDArray[np.floating], eps: float = 1e-10) -> NDArray[np.floating]:
        lower = None if self.mu_lower_bound is None else self.mu_lower_bound + eps
        upper = None if self.mu_upper_bound is None else self.mu_upper_bound - eps
        if lower is not None or upper is not None:
            np.clip(mu, lower, upper, out=mu)
        return mu

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
    mean_bounds: tuple[float | None, float | None] = (None, None)

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

    def clamp_mu(
        self,
        mu: NDArray[np.floating],
        eps: float = 1e-10,
        *,
        out: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Clamp response means to this family's valid domain.

        Custom families can set ``mean_bounds`` or override this method when
        their response mean has a more specialized domain.
        """
        lower, upper = self.mean_bounds
        if lower is None and upper is None:
            if out is None or out is mu:
                return mu
            np.copyto(out, mu)
            return out

        lower_bound = None if lower is None else lower + eps
        upper_bound = None if upper is None else upper - eps
        return np.clip(mu, lower_bound, upper_bound, out=out)

    def simulate(self, mu: NDArray[np.floating], rng: Any | None = None) -> NDArray[np.floating]:
        """Draw responses at the supplied means.

        The default provides a small Gaussian perturbation for custom families.
        Built-in families override it with their corresponding distribution.
        """
        rng = np.random if rng is None else rng
        return mu + rng.standard_normal(mu.shape) * 0.1

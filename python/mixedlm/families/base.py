from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

_LINK_EPS = 1e-10


class Link(ABC):
    name: str
    mu_lower_bound: float | None = None
    mu_upper_bound: float | None = None

    def __call__(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.link(mu)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

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
    name = "identity"

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return eta

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones_like(mu)


class LogLink(Link):
    name = "log"
    mu_lower_bound = 0.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.log(mu)

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.exp(eta)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / mu


class LogitLink(Link):
    name = "logit"
    mu_lower_bound = 0.0
    mu_upper_bound = 1.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.log(mu / (1 - mu))

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return expit(eta)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (mu * (1 - mu))


class ProbitLink(Link):
    name = "probit"
    mu_lower_bound = 0.0
    mu_upper_bound = 1.0

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
    name = "cloglog"
    mu_lower_bound = 0.0
    mu_upper_bound = 1.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.log(-np.log(1 - mu))

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        exp_eta = np.exp(np.minimum(eta, np.log(np.finfo(np.float64).max)))
        return -np.expm1(-exp_eta)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / ((1 - mu) * (-np.log(1 - mu)))


class InverseLink(Link):
    name = "inverse"
    mu_lower_bound = 0.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / mu

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / np.maximum(eta, _LINK_EPS)

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return -1.0 / (mu**2)


class SqrtLink(Link):
    name = "sqrt"
    mu_lower_bound = 0.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.sqrt(mu)

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return eta**2

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 0.5 / np.sqrt(mu)


class CauchitLink(Link):
    name = "cauchit"
    mu_lower_bound = 0.0
    mu_upper_bound = 1.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.tan(np.pi * (mu - 0.5))

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.arctan(eta) / np.pi + 0.5

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        angle = np.pi * (mu - 0.5)
        return np.pi / np.cos(angle) ** 2


class InverseSquaredLink(Link):
    name = "1/mu^2"
    mu_lower_bound = 0.0

    def link(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (mu**2)

    def inverse(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / np.sqrt(np.maximum(eta, _LINK_EPS))

    def deriv(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return -2.0 / (mu**3)


_LINK_TYPES: dict[str, type[Link]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "cloglog": CloglogLink,
    "cauchit": CauchitLink,
    "inverse": InverseLink,
    "sqrt": SqrtLink,
    "1/mu^2": InverseSquaredLink,
}

_LINK_ALIASES = {
    "complementary log-log": "cloglog",
    "complementary_log_log": "cloglog",
    "comploglog": "cloglog",
    "inverse squared": "1/mu^2",
    "inverse-squared": "1/mu^2",
    "inverse_squared": "1/mu^2",
    "1/mu2": "1/mu^2",
}


def _canonical_link_name(name: str) -> str:
    normalized = name.strip().lower()
    return _LINK_ALIASES.get(normalized, normalized)


def resolve_link(
    link: str | Link | None,
    *,
    default: str,
    allowed: tuple[str, ...] | None = None,
) -> Link:
    """Create a link from a name while optionally enforcing a supported set."""
    if isinstance(link, Link):
        return link
    if link is not None and not isinstance(link, str):
        raise TypeError("link must be a string, Link instance, or None")

    name = _canonical_link_name(default if link is None else link)
    if name not in _LINK_TYPES:
        choices = ", ".join(sorted(_LINK_TYPES))
        raise ValueError(f"Unknown link '{link}'. Available links: {choices}")

    if allowed is not None:
        allowed_names = tuple(_canonical_link_name(item) for item in allowed)
        if name not in allowed_names:
            choices = ", ".join(allowed)
            raise ValueError(f"Link '{name}' is not supported here. Choose from: {choices}")

    return _LINK_TYPES[name]()


class Family(ABC):
    link: Link
    mu_lower_bound: float | None = None
    mu_upper_bound: float | None = None

    def __init__(
        self,
        link: str | Link | None = None,
        *,
        default_link: str = "identity",
        allowed_links: tuple[str, ...] | None = None,
    ) -> None:
        self.link = resolve_link(link, default=default_link, allowed=allowed_links)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(link={self.link.name})"

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

    def linkinv(self, eta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the inverse link to a linear predictor."""
        return self.link.inverse(eta)

    def deviance_residuals(
        self,
        y: NDArray[np.floating],
        mu: NDArray[np.floating],
        wt: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Return deviance residual contributions with optional unit weights."""
        weights = np.ones_like(y, dtype=np.float64) if wt is None else wt
        return self.deviance_resids(y, mu, weights)

    def _mean_bounds(self) -> tuple[float | None, float | None]:
        lower_bounds = (self.mu_lower_bound, self.link.mu_lower_bound)
        upper_bounds = (self.mu_upper_bound, self.link.mu_upper_bound)
        defined_lowers = [bound for bound in lower_bounds if bound is not None]
        defined_uppers = [bound for bound in upper_bounds if bound is not None]
        lower = max(defined_lowers) if defined_lowers else None
        upper = min(defined_uppers) if defined_uppers else None
        return lower, upper

    def clip_mu(self, mu: NDArray[np.floating], eps: float = 1e-10) -> NDArray[np.floating]:
        lower, upper = self._mean_bounds()
        lower = None if lower is None else lower + eps
        upper = None if upper is None else upper - eps
        if lower is not None or upper is not None:
            np.clip(mu, lower, upper, out=mu)
        return mu

    def initialize_mu(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Choose finite starting means inside the family and link domains."""
        mu = np.asarray(y, dtype=np.float64).copy()
        lower, upper = self._mean_bounds()

        if lower == 0.0 and upper == 1.0:
            mu = (mu + 0.5) / 2.0
        elif lower == 0.0:
            np.maximum(mu, 0.1, out=mu)

        return self.clip_mu(mu, eps=1e-7)

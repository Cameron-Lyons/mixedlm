from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, xlog1py, xlogy

from mixedlm.families.base import Family, Link
from mixedlm.families.likelihood import likelihood_inputs, whole_counts


class Binomial(Family):
    mean_bounds = (0.0, 1.0)

    def __init__(self, link: str | Link | None = None) -> None:
        super().__init__(
            link,
            default_link="logit",
            allowed_links=("logit", "probit", "cloglog", "cauchit", "log"),
        )

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return mu * (1 - mu)

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        mu = self.clamp_mu(mu)

        term1 = xlogy(y, y / mu)
        term2 = xlogy(1 - y, (1 - y) / (1 - mu))

        return 2 * wt * (term1 + term2)

    def log_likelihood(
        self,
        y: NDArray[np.floating],
        mu: NDArray[np.floating],
        wt: NDArray[np.floating],
        *,
        trials: NDArray[np.floating] | None = None,
    ) -> float:
        y, mu, wt = likelihood_inputs(y, mu, wt)
        if np.any((y < 0) | (y > 1)) or np.any((mu < 0) | (mu > 1)):
            raise ValueError("Binomial likelihood responses and means must be between zero and one")

        if trials is None and np.all((y == 0) | (y == 1)):
            # Bernoulli prior weights may be fractional power weights.
            return float(np.sum(wt * (xlogy(y, mu) + xlog1py(1 - y, -mu))))

        n = wt if trials is None else np.asarray(trials, dtype=np.float64)
        if n.shape != y.shape or not np.all(np.isfinite(n)) or np.any(n <= 0):
            raise ValueError("Binomial likelihood trials must be finite positive matching counts")
        n = whole_counts(n, "binomial trial-count")
        if np.any(n == 0):
            raise ValueError("Binomial likelihood trial counts must be positive whole numbers")
        successes = whole_counts(n * y, "binomial success-count")
        failures = n - successes
        log_density = (
            gammaln(n + 1)
            - gammaln(successes + 1)
            - gammaln(failures + 1)
            + xlogy(successes, mu)
            + xlog1py(failures, -mu)
        )
        return float(np.sum((wt / n) * log_density))

    def simulate(
        self,
        mu: NDArray[np.floating],
        rng: Any | None = None,
        *,
        weights: NDArray[np.floating] | None = None,
        trials: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        # Prior weights of binary responses are likelihood powers, not trials.
        rng = np.random if rng is None else rng
        mu = self.clamp_mu(mu, eps=1e-6)
        n = 1 if trials is None else np.asarray(trials).astype(np.int64)
        return rng.binomial(n, mu).astype(np.float64)

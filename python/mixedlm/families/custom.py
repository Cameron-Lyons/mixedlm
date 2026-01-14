"""Utilities for creating custom exponential family distributions.

This module provides helper classes and validation functions for users who want
to create their own custom families for use with glmer().

Example
-------
Creating a Tweedie family with power parameter p:

>>> from mixedlm.families.custom import CustomFamily, validate_family
>>> from mixedlm.families.base import LogLink
>>>
>>> class TweedieFamily(CustomFamily):
...     def __init__(self, p: float = 1.5):
...         self.p = p
...         self.link = LogLink()
...
...     def variance(self, mu):
...         return mu ** self.p
...
...     def deviance_resids(self, y, mu, wt):
...         # Tweedie deviance for 1 < p < 2
...         a = y * mu**(1-self.p) / (1 - self.p) if self.p != 1 else y * np.log(mu)
...         b = mu**(2-self.p) / (2 - self.p) if self.p != 2 else np.log(mu)
...         return 2 * wt * (y * np.log(y/mu) - (y - mu))  # simplified
>>>
>>> validate_family(TweedieFamily(p=1.5))
True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mixedlm.families.base import Family, Link


class CustomFamily(Family):
    """Base class for creating custom exponential family distributions.

    Subclass this to create your own family. You must implement:
    - __init__: Set self.link to an appropriate Link object
    - variance(mu): Return the variance function V(mu)
    - deviance_resids(y, mu, wt): Return the weighted deviance residuals

    The weights() method is inherited and computed from the link and variance.

    Example
    -------
    >>> class QuasiPoissonFamily(CustomFamily):
    ...     def __init__(self, phi: float = 1.0):
    ...         from mixedlm.families.base import LogLink
    ...         self.phi = phi
    ...         self.link = LogLink()
    ...
    ...     def variance(self, mu):
    ...         return self.phi * mu
    ...
    ...     def deviance_resids(self, y, mu, wt):
    ...         y_safe = np.maximum(y, 1e-10)
    ...         mu_safe = np.maximum(mu, 1e-10)
    ...         return 2 * wt * (y * np.log(y_safe / mu_safe) - (y - mu))
    """

    link: Link

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError("Subclass must implement variance()")

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        raise NotImplementedError("Subclass must implement deviance_resids()")


def validate_family(family: Family, test_mu: NDArray[np.floating] | None = None) -> bool:
    """Validate that a custom family has required methods and returns valid values.

    This function tests that a family object:
    1. Has a link attribute that is a Link object
    2. Has variance(), deviance_resids(), and weights() methods
    3. Returns arrays of the correct shape
    4. Returns finite values for typical inputs

    Parameters
    ----------
    family : Family
        The family object to validate.
    test_mu : NDArray, optional
        Test values for mu. If None, uses [0.1, 0.5, 0.9] for bounded families
        or [0.5, 1.0, 5.0] for unbounded families.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If validation fails, with a message explaining the problem.

    Example
    -------
    >>> from mixedlm.families import Binomial
    >>> validate_family(Binomial())
    True
    """
    if not hasattr(family, "link"):
        raise ValueError("Family must have a 'link' attribute")

    if not isinstance(family.link, Link):
        raise ValueError("Family.link must be a Link instance")

    required_methods = ["variance", "deviance_resids", "weights"]
    for method in required_methods:
        if not hasattr(family, method) or not callable(getattr(family, method)):
            raise ValueError(f"Family must have a callable '{method}' method")

    if test_mu is None:
        from mixedlm.families.base import CloglogLink, LogitLink, ProbitLink

        if isinstance(family.link, (LogitLink, ProbitLink, CloglogLink)):
            test_mu = np.array([0.1, 0.5, 0.9])
        else:
            test_mu = np.array([0.5, 1.0, 5.0])

    test_y = test_mu * 1.1
    test_wt = np.ones_like(test_mu)

    try:
        var = family.variance(test_mu)
        if var.shape != test_mu.shape:
            raise ValueError(f"variance() returned wrong shape: {var.shape} vs {test_mu.shape}")
        if not np.all(np.isfinite(var)):
            raise ValueError("variance() returned non-finite values")
        if np.any(var <= 0):
            raise ValueError("variance() must return positive values")
    except Exception as e:
        raise ValueError(f"variance() failed: {e}") from e

    try:
        dev = family.deviance_resids(test_y, test_mu, test_wt)
        if dev.shape != test_mu.shape:
            raise ValueError(
                f"deviance_resids() returned wrong shape: {dev.shape} vs {test_mu.shape}"
            )
        if not np.all(np.isfinite(dev)):
            raise ValueError("deviance_resids() returned non-finite values")
    except Exception as e:
        raise ValueError(f"deviance_resids() failed: {e}") from e

    try:
        wts = family.weights(test_mu)
        if wts.shape != test_mu.shape:
            raise ValueError(f"weights() returned wrong shape: {wts.shape} vs {test_mu.shape}")
        if not np.all(np.isfinite(wts)):
            raise ValueError("weights() returned non-finite values")
        if np.any(wts <= 0):
            raise ValueError("weights() must return positive values")
    except Exception as e:
        raise ValueError(f"weights() failed: {e}") from e

    try:
        eta = family.link.link(test_mu)
        mu_back = family.link.inverse(eta)
        if not np.allclose(test_mu, mu_back, rtol=1e-6):
            raise ValueError("link.inverse(link.link(mu)) != mu")
        deriv = family.link.deriv(test_mu)
        if not np.all(np.isfinite(deriv)):
            raise ValueError("link.deriv() returned non-finite values")
    except Exception as e:
        raise ValueError(f"Link validation failed: {e}") from e

    return True


class QuasiFamily(CustomFamily):
    """A quasi-likelihood family that wraps an existing family with overdispersion.

    The quasi-family multiplies the variance function by a dispersion parameter,
    which is useful for handling overdispersion in count or binary data.

    Parameters
    ----------
    base_family : Family
        The base family to wrap (e.g., Poisson, Binomial).
    phi : float, default 1.0
        The dispersion parameter. Values > 1 indicate overdispersion.

    Example
    -------
    >>> from mixedlm.families import Poisson
    >>> from mixedlm.families.custom import QuasiFamily
    >>> quasi_poisson = QuasiFamily(Poisson(), phi=2.0)
    """

    def __init__(self, base_family: Family, phi: float = 1.0):
        self.base_family = base_family
        self.phi = phi
        self.link = base_family.link

    def variance(self, mu: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.phi * self.base_family.variance(mu)

    def deviance_resids(
        self, y: NDArray[np.floating], mu: NDArray[np.floating], wt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return self.base_family.deviance_resids(y, mu, wt) / self.phi

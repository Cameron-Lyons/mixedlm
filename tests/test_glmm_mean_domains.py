from __future__ import annotations

import mixedlm.estimation.laplace as laplace
import numpy as np
import pytest
from mixedlm.families import Gamma, InverseGaussian, NegativeBinomial, Poisson


class DerivedPoisson(Poisson):
    pass


@pytest.mark.parametrize(
    "family",
    [Gamma(), InverseGaussian(), NegativeBinomial(), DerivedPoisson()],
)
def test_unsupported_native_families_fall_back_to_python(monkeypatch, family) -> None:
    expected = (1.25, np.array([2.0]), np.array([3.0]))

    monkeypatch.setattr(laplace, "_HAS_RUST", True)
    monkeypatch.setattr(laplace, "laplace_deviance", lambda *args: expected)

    def fail_native(*args):
        raise AssertionError("unsupported family reached the native backend")

    monkeypatch.setattr(laplace, "_laplace_deviance_rust", fail_native)

    actual = laplace.laplace_deviance_fast(np.array([0.5]), object(), family)

    assert actual is expected

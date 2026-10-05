from __future__ import annotations

import mixedlm.estimation.laplace as laplace
import numpy as np
import pandas as pd
import pytest
from mixedlm.families import Binomial, Gamma, InverseGaussian, NegativeBinomial, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_array_equal


class DerivedPoisson(Poisson):
    pass


def matrices_for(family):
    rng = np.random.default_rng(29)
    group = np.repeat(np.arange(6), 5)
    x = np.linspace(-1.0, 1.0, group.size)
    mean = np.exp(0.4 + 0.3 * x + rng.normal(scale=0.3, size=6)[group])
    if isinstance(family, Binomial):
        y = rng.binomial(1, mean / (1 + mean))
    else:
        y = 1 + rng.poisson(mean)
    data = pd.DataFrame({"y": y.astype(float), "x": x, "g": group})
    return build_model_matrices(parse_formula("y ~ x + (1 | g)"), data)


@pytest.mark.parametrize(
    "family",
    [
        Gamma(),
        InverseGaussian(),
        NegativeBinomial(),
        DerivedPoisson(),
        Binomial(link="probit"),
        Poisson(link="sqrt"),
    ],
)
def test_unsupported_native_families_use_python_likelihood(family) -> None:
    pytest.importorskip("mixedlm._rust")
    matrices = matrices_for(family)
    theta = np.array([0.6])
    # The exact native family types are eligible for this design; these are not.
    assert laplace._prepare_native_glmm(matrices, Poisson()) is not None
    assert laplace._prepare_native_glmm(matrices, family) is None

    expected = laplace.laplace_deviance(theta, matrices, family)
    actual = laplace.glmm_deviance_with_status(theta, matrices, family)
    for value, reference in zip(actual[:3], expected, strict=True):
        assert_array_equal(value, reference)
    assert laplace.GLMMOptimizer(matrices, family).objective(theta) == expected[0]

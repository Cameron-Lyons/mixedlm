"""Eliminating a shared structure creates covariance between disjoint designs."""

import numpy as np
import pytest
from mixedlm import _rust
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
from numpy.testing import assert_allclose
from scipy import sparse

from tests.test_lmm_gradient_contractions import observation_gradient
from tests.test_lmm_prepared_design import native_arguments
from tests.test_reml_profiled_deviance import _direct_profiled_likelihood


def shared_structure_problem(widths, correlated):
    rng = np.random.default_rng(509)
    n = 64
    designs, structures, parameters = [], [], []
    for index, width in enumerate(widths):
        design = rng.normal(scale=0.4, size=(n, 2 * width))
        if index:
            # Siblings have exactly disjoint support but both couple to the
            # first structure. Their original crossproduct vanishes exactly.
            design[np.arange(n) % 2 != index - 1] = 0
        designs.append(design)
        structures.append(
            RandomEffectStructure(
                f"group{index}",
                [f"term{j}" for j in range(width)],
                2,
                width,
                correlated,
                {"a": 0, "b": 1},
            )
        )
        lower = np.diag(np.linspace(0.5, 0.9, width))
        if correlated:
            lower[np.tril_indices(width, -1)] = 0.05
        parameters.extend(lower[np.tril_indices(width)] if correlated else lower.diagonal())
    z = np.column_stack(designs)
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    matrices = ModelMatrices(
        y=rng.normal(size=n),
        X=x,
        Z=sparse.csc_matrix(z),
        fixed_names=["Intercept", "x"],
        random_structures=structures,
        n_obs=n,
        n_fixed=2,
        n_random=z.shape[1],
        weights=np.geomspace(0.4, 2.0, n),
        offset=np.linspace(-0.2, 0.3, n),
    )
    assert np.all(designs[1].T @ (matrices.weights[:, None] * designs[2]) == 0)
    assert np.any(designs[0].T @ (matrices.weights[:, None] * designs[1]) != 0)
    assert np.any(designs[0].T @ (matrices.weights[:, None] * designs[2]) != 0)
    return matrices, np.asarray(parameters)


@pytest.mark.parametrize("widths", [(1, 1, 1), (2, 3, 2), (16, 3, 2)])
@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_three_structures_preserve_fill_in_likelihood_estimates_and_gradient(
    widths, correlated, reml
):
    matrices, theta = shared_structure_problem(widths, correlated)
    response = _rust.LmmDesign(**native_arguments(matrices)).with_response(matrices.y)
    expected = _direct_profiled_likelihood(theta, matrices, reml)
    actual = response.evaluate(theta, reml)
    for value, field in zip(actual[:4], ["deviance", "beta", "sigma", "u"], strict=True):
        assert_allclose(value, expected[field], rtol=2e-11, atol=2e-10)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert_allclose(value, expected["deviance"], rtol=2e-12, atol=2e-10)
    assert_allclose(gradient, observation_gradient(matrices, theta, reml), rtol=2e-10, atol=2e-9)

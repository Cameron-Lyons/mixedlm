"""Coupled random effects are eliminated in an internal order chosen for fill.

Nested and crossed structures couple their levels. The native design eliminates
structures with the most random-effect columns first, then the most levels, and
factors the result sparsely or with a dense Schur complement, keeping the
caller's order for theta, gradients and random effects.
"""

import numpy as np
import pandas as pd
import pytest
from mixedlm import _rust, lmer
from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import LmerControl
from numpy.testing import assert_allclose

from tests._lmm_oracles import native_arguments, observation_likelihood


def matrices(formula, data):
    return build_model_matrices(parse_formula(formula), data)


def nested_data(sizes, per=1, seed=5):
    """Factors f0, f1, ... nested within each other, coarsest first."""
    rng = np.random.default_rng(seed)
    cells = int(np.prod(sizes))
    cell = np.repeat(np.arange(cells), per)
    x = rng.normal(size=cell.size)
    y = 0.5 * x + rng.normal(size=cell.size)
    columns = {"x": x}
    stride = cells
    for depth, size in enumerate(sizes):
        stride //= size
        level = cell // stride
        columns[f"f{depth}"] = level % size
        y += rng.normal(scale=1.0 / (depth + 1), size=cells // stride)[level]
    columns["y"] = y
    names = [f"f{depth}" for depth in range(len(sizes))]
    nested = f"y ~ x + (1 | {'/'.join(names)})"
    sorted_terms = " + ".join(
        f"(1 | {':'.join(names[: depth + 1])})" for depth in reversed(range(len(sizes)))
    )
    return pd.DataFrame(columns), nested, f"y ~ x + {sorted_terms}"


def random_crossed_data(n_small, n_large, n, seed=7):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, n_small, n)
    large = rng.integers(0, n_large, n)
    x, z = rng.normal(size=(2, n))
    slopes = rng.normal(scale=0.3, size=(2, n_large))
    y = (
        x
        + rng.normal(size=n_small)[small]
        + rng.normal(size=n_large)[large]
        + slopes[0, large] * x
        + slopes[1, large] * z
        + rng.normal(size=n)
    )
    return pd.DataFrame(dict(y=y, x=x, z=z, small=small, large=large))


def central_differences(matrices, theta, reml, step=1e-5):
    gradient = []
    for index in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[index] = step
        gradient.append(
            (
                observation_likelihood(matrices, theta + shift, reml)
                - observation_likelihood(matrices, theta - shift, reml)
            )
            / (2 * step)
        )
    return np.array(gradient)


@pytest.mark.parametrize("sizes", [(80, 6), (8, 10, 5)])
@pytest.mark.parametrize("reml", [False, True])
def test_nested_formula_order_matches_sorted_terms_and_dense_likelihood(sizes, reml):
    data, formula, sorted_formula = nested_data(sizes)
    nested, ordered = matrices(formula, data), matrices(sorted_formula, data)
    depth = len(sizes)
    # Formula order lists the coarsest factor, with the fewest levels, first.
    levels = [structure.n_levels for structure in nested.random_structures]
    assert (
        levels == sorted(levels) and levels == [s.n_levels for s in ordered.random_structures][::-1]
    )
    design = _rust.LmmDesign(**native_arguments(nested))
    ordered_design = _rust.LmmDesign(**native_arguments(ordered))
    assert design.elimination_order == list(reversed(range(depth)))
    assert ordered_design.elimination_order == list(range(depth))
    # Nesting adds no fill, so no level block is factored densely.
    assert design.engine == ordered_design.engine == "sparse"
    assert design.dense_dimension == 0

    theta = np.linspace(0.6, 1.2, depth)
    response = design.with_response(nested.y)
    ordered_response = ordered_design.with_response(ordered.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    ordered_value, ordered_gradient = ordered_response.deviance_with_gradient(
        theta[::-1].copy(), reml
    )
    assert_allclose(value, ordered_value, rtol=1e-13)
    assert_allclose(gradient, ordered_gradient[::-1], rtol=1e-11)
    assert response.deviance(theta, reml) == value

    estimates = response.evaluate(theta, reml)
    ordered_estimates = ordered_response.evaluate(theta[::-1].copy(), reml)
    assert_allclose(estimates[1], ordered_estimates[1], rtol=1e-11)
    blocks = np.cumsum([0, *levels])
    ordered_blocks = np.cumsum([0, *levels[::-1]])
    for index in range(depth):
        mirrored = depth - 1 - index
        assert_allclose(
            estimates[3][blocks[index] : blocks[index + 1]],
            ordered_estimates[3][ordered_blocks[mirrored] : ordered_blocks[mirrored + 1]],
            rtol=1e-10,
            atol=1e-12,
        )

    # Independent oracles: the dense marginal likelihood and the Python engine.
    assert_allclose(value, observation_likelihood(nested, theta, reml), rtol=1e-10)
    assert_allclose(gradient, central_differences(nested, theta, reml), rtol=1e-6, atol=1e-6)
    core = LMMOptimizer(nested, REML=reml, use_rust=False)._evaluate_core(theta)
    assert_allclose(value, core.deviance, rtol=1e-10)
    assert_allclose(estimates[1], core.beta, rtol=1e-9)
    assert_allclose(estimates[3], core.u, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "n_small,n_large,slopes,theta",
    [
        (30, 40, "x", [0.8, 1.1, -0.2, 0.4]),
        # Fewer slope levels, but still more columns than the intercepts.
        (40, 30, "x + z", [0.8, 1.1, -0.2, 0.4, 0.3, 0.1, 0.5]),
    ],
)
@pytest.mark.parametrize("reml", [False, True])
def test_random_crossings_factor_the_smaller_structure_densely(
    n_small, n_large, slopes, theta, reml
):
    data = random_crossed_data(n_small, n_large, n=300)
    crossed = matrices(f"y ~ x + (1 | small) + ({slopes} | large)", data)
    design = _rust.LmmDesign(**native_arguments(crossed))
    # The slope structure has more columns, so it is eliminated first and the
    # random crossings fill only the smaller structure's Schur complement.
    assert design.elimination_order == [1, 0]
    assert design.engine == "blocked"
    assert design.dense_dimension == n_small < crossed.n_random

    theta = np.array(theta)
    response = design.with_response(crossed.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert_allclose(value, observation_likelihood(crossed, theta, reml), rtol=1e-10)
    assert_allclose(gradient, central_differences(crossed, theta, reml), rtol=1e-6, atol=1e-6)
    core = LMMOptimizer(crossed, REML=reml, use_rust=False)._evaluate_core(theta)
    estimates = response.evaluate(theta, reml)
    assert_allclose(estimates[0], core.deviance, rtol=1e-10)
    assert_allclose(estimates[3], core.u, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("reml", [False, True])
def test_nesting_within_a_wider_parent_factors_sparsely(reml):
    rng = np.random.default_rng(3)
    cell = np.repeat(np.arange(40 * 2), 6)
    school = cell // 2
    x1, x2, x3 = rng.normal(size=(3, cell.size))
    y = x1 + rng.normal(size=40)[school] * (1 + x1 + x2) + rng.normal(size=80)[cell]
    data = pd.DataFrame(
        dict(y=y + rng.normal(size=cell.size), x1=x1, x2=x2, x3=x3, school=school, cls=cell % 2)
    )
    nested = matrices("y ~ x1 + (x1 + x2 + x3 | school) + (1 | school:cls)", data)
    design = _rust.LmmDesign(**native_arguments(nested))
    # The parent's slopes give it more columns, so it is eliminated first and
    # the sparse factorization orders the nested levels to avoid fill.
    assert [s.n_levels * s.n_terms for s in nested.random_structures] == [160, 80]
    assert design.elimination_order == [0, 1]
    assert design.engine == "sparse"
    assert design.dense_dimension == 0

    theta = np.array([0.8, 0.1, 0.5, 0.1, 0.1, 0.4, 0.05, 0.1, 0.1, 0.3, 0.6])
    response = design.with_response(nested.y)
    value, gradient = response.deviance_with_gradient(theta, reml)
    assert_allclose(value, observation_likelihood(nested, theta, reml), rtol=1e-10)
    assert_allclose(gradient, central_differences(nested, theta, reml), rtol=1e-6, atol=1e-6)
    core = LMMOptimizer(nested, REML=reml, use_rust=False)._evaluate_core(theta)
    assert_allclose(value, core.deviance, rtol=1e-10)
    assert_allclose(response.evaluate(theta, reml)[3], core.u, rtol=1e-8, atol=1e-10)


def regular_crossed_data():
    # The fixture of large_crossed_sparse_data: few distinct crossings per level.
    rng = np.random.default_rng(123)
    rows = np.arange(5_000)
    x = rng.normal(size=rows.size)
    y = x + rng.normal(size=500)[rows % 500] + rng.normal(size=400)[(rows * 11) % 400]
    return pd.DataFrame(dict(y=y, x=x, g1=rows % 500, g2=(rows * 11) % 400))


@pytest.mark.parametrize(
    "layout,engine,dense",
    [("nested", "sparse", 0), ("regular", "sparse", 0), ("random", "blocked", 800)],
)
def test_large_coupled_designs_never_factor_a_square_dense_system(layout, engine, dense):
    if layout == "nested":
        data, formula, _ = nested_data((300, 10))
    elif layout == "regular":
        data, formula = regular_crossed_data(), "y ~ x + (1 | g2) + (1 | g1)"
    else:
        data = random_crossed_data(n_small=800, n_large=1000, n=9000)
        formula = "y ~ x + (1 | small) + (1 | large)"
    coupled = matrices(formula, data)
    design = _rust.LmmDesign(**native_arguments(coupled))
    assert coupled.n_random >= 900
    assert design.engine == engine
    # Only random crossings keep a dense block, sized by the smaller factor.
    assert design.dense_dimension == dense
    theta = np.array([0.9, 0.6])
    value = design.with_response(coupled.y).deviance(theta, True)
    python = LMMOptimizer(coupled, use_rust=False).objective(theta)
    assert_allclose(value, python, rtol=1e-10)


def test_nested_fit_keeps_formula_order_for_theta_and_random_effects():
    data, formula, _ = nested_data((40, 5), per=4, seed=11)
    native = lmer(formula, data)
    python = lmer(formula, data, control=LmerControl(use_rust=False))
    assert [s.grouping_factor for s in native.matrices.random_structures] == ["f0", "f0:f1"]
    assert_allclose(native.deviance, python.deviance, rtol=1e-8)
    assert_allclose(native.theta, python.theta, rtol=1e-4)
    # The coarse factor was simulated with twice the scale of the nested one.
    assert native.theta[0] > native.theta[1]
    native_effects, python_effects = native.ranef(), python.ranef()
    assert list(native_effects) == list(python_effects)
    for group, terms in python_effects.items():
        for term, values in terms.items():
            assert_allclose(native_effects[group][term], values, rtol=1e-4, atol=1e-6)

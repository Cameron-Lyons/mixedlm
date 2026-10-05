from __future__ import annotations

import numpy as np
import pytest
from mixedlm._rust import simulate_re_batch

pytestmark = pytest.mark.installed_wheel


class TestRandomEffectSimulationValidation:
    @pytest.mark.parametrize(
        ("n_levels", "n_terms", "correlated"),
        [
            ([2, 3], [1], [False, False]),
            ([2], [1, 1], [False]),
            ([2], [1], [False, True]),
        ],
    )
    def test_rejects_mismatched_structure_lengths(self, n_levels, n_terms, correlated):
        with pytest.raises(ValueError, match="must have the same length"):
            simulate_re_batch(
                np.array([1.0]),
                1.0,
                n_levels,
                n_terms,
                correlated,
                1,
                seed=42,
            )

    @pytest.mark.parametrize("theta", [np.array([]), np.array([1.0, 2.0])])
    def test_rejects_incorrect_theta_length(self, theta):
        with pytest.raises(ValueError, match="theta must contain exactly 1 value"):
            simulate_re_batch(theta, 1.0, [2], [1], [False], 1, seed=42)

    @pytest.mark.parametrize("sigma", [-1.0, np.nan, np.inf])
    def test_rejects_invalid_sigma(self, sigma):
        with pytest.raises(ValueError, match="sigma must be finite and non-negative"):
            simulate_re_batch(np.array([1.0]), sigma, [2], [1], [False], 1, seed=42)

    def test_rejects_nonfinite_theta(self):
        with pytest.raises(ValueError, match=r"theta\[0\] must be finite"):
            simulate_re_batch(np.array([np.nan]), 1.0, [2], [1], [False], 1, seed=42)

    @pytest.mark.parametrize(
        ("n_levels", "n_terms", "message"),
        [
            ([0], [1], r"n_levels\[0\] must be positive"),
            ([2], [0], r"n_terms\[0\] must be positive"),
        ],
    )
    def test_rejects_empty_structure_dimensions(self, n_levels, n_terms, message):
        with pytest.raises(ValueError, match=message):
            simulate_re_batch(np.array([]), 1.0, n_levels, n_terms, [False], 1, seed=42)

    def test_empty_batch_preserves_random_effect_dimension(self):
        result = simulate_re_batch(np.array([1.0]), 1.0, [5], [1], [False], 0, seed=42)

        assert np.asarray(result).shape == (0, 5)

    def test_empty_structure_preserves_simulation_dimension(self):
        result = simulate_re_batch(np.array([]), 1.0, [], [], [], 3, seed=42)

        assert np.asarray(result).shape == (3, 0)

    def test_multiple_structures_have_expected_shape_and_are_reproducible(self):
        arguments = (np.array([1.0, 0.3, 0.8]), 1.5, [4, 3], [1, 2], [False, False], 20)

        first = simulate_re_batch(*arguments, seed=42)
        second = simulate_re_batch(*arguments, seed=42)

        assert np.asarray(first).shape == (20, 10)
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("width", [1, 2, 8, 64])
@pytest.mark.parametrize("correlated", [False, True])
def test_seeded_draws_match_independent_covariance_transform(width, correlated):
    # A unit diagonal sampler exposes the same independent normal stream.
    # Transform it with NumPy rather than reusing the production factor builder.
    levels = [3, 2]
    widths = [width, 2]
    factors = [np.diag(np.linspace(0.3, 1.2, size)) for size in widths]
    factors[0][0, 0] = 0.0
    if correlated:
        for factor in factors:
            factor[np.tril_indices(len(factor), -1)] = -0.15
    theta = np.concatenate(
        [
            factor[np.tril_indices(len(factor))] if correlated else factor.diagonal()
            for factor in factors
        ]
    )
    standard = np.asarray(
        simulate_re_batch(np.ones(sum(widths)), 1.0, levels, widths, [False, False], 5, seed=73)
    )
    expected = np.empty_like(standard)
    start = 0
    for n_levels, factor in zip(levels, factors, strict=True):
        stop = start + n_levels * len(factor)
        draws = standard[:, start:stop].reshape(5, n_levels, len(factor))
        expected[:, start:stop] = (draws @ (1.3 * factor).T).reshape(5, -1)
        start = stop

    actual = simulate_re_batch(theta, 1.3, levels, widths, [correlated] * 2, 5, seed=73)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_concurrent_batches_retain_seeded_draws():
    from concurrent.futures import ThreadPoolExecutor

    def draw(seed):
        return simulate_re_batch(np.linspace(0.1, 0.8, 32), 1.2, [128], [32], [False], 8, seed=seed)

    seeds = [3, 91, 3, 72, 91]
    expected = [draw(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(draw, seeds))
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("stream", [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize("width", [1, 8, 64])
def test_python_sampler_preserves_mixed_structure_covariance_and_random_stream(stream, width):
    from mixedlm.matrices.design import RandomEffectStructure
    from mixedlm.utils.simulation import simulate_random_effects

    factors = [np.diag(np.linspace(0.0, 0.9, width)), np.array([[0.8, 0.0], [-0.2, 0.5]])]
    structures = [
        RandomEffectStructure("group", [f"term{i}" for i in range(width)], 3, width, False, {}),
        RandomEffectStructure("other", ["intercept", "slope"], 4, 2, True, {}),
    ]
    theta = np.r_[factors[0].diagonal(), [0.8, -0.2, 0.5]]
    reference_stream = stream(74)
    reference = (
        np.concatenate(
            [
                (reference_stream.standard_normal((s.n_levels, s.n_terms)) @ factor.T).ravel()
                for s, factor in zip(structures, factors, strict=True)
            ]
        )
        * 1.3
    )
    actual_stream = stream(74)
    actual = simulate_random_effects(theta, structures, 1.3, rng=actual_stream)
    np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(
        actual_stream.standard_normal(5), reference_stream.standard_normal(5)
    )


@pytest.mark.parametrize("stream", [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize(
    "theta", [np.array([]), np.array([0.4]), np.array([0.4, 0.7, 0.9]), np.array([[0.4, 0.7]])]
)
def test_python_sampler_rejects_invalid_theta_before_consuming_random_stream(stream, theta):
    from mixedlm.matrices.design import RandomEffectStructure
    from mixedlm.utils.simulation import simulate_random_effects

    structures = [RandomEffectStructure("group", ["intercept", "slope"], 3, 2, False, {})]
    actual_stream = stream(94)
    reference_stream = stream(94)

    with pytest.raises(ValueError, match="one-dimensional array of exactly 2 values"):
        simulate_random_effects(theta, structures, rng=actual_stream)

    np.testing.assert_array_equal(
        actual_stream.standard_normal(5), reference_stream.standard_normal(5)
    )

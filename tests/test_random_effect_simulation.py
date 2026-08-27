from __future__ import annotations

import numpy as np
import pytest

try:
    from mixedlm._rust import simulate_re_batch

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")


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

    @pytest.mark.parametrize("cov_type", ["cs", "ar1"])
    def test_structured_covariance_matches_requested_correlation(self, cov_type):
        relative_scale = 1.2
        sigma = 1.5
        rho = 0.35
        n_terms = 3

        samples = np.asarray(
            simulate_re_batch(
                np.array([relative_scale, rho]),
                sigma,
                [1],
                [n_terms],
                [True],
                50_000,
                seed=42,
                cov_types=[cov_type],
            )
        )

        if cov_type == "cs":
            correlation = np.full((n_terms, n_terms), rho)
            np.fill_diagonal(correlation, 1.0)
        else:
            indices = np.arange(n_terms)
            correlation = rho ** np.abs(indices[:, None] - indices[None, :])
        expected = relative_scale**2 * sigma**2 * correlation
        np.testing.assert_allclose(np.cov(samples, rowvar=False), expected, rtol=0.03, atol=0.03)

    def test_rejects_unknown_covariance_type(self):
        with pytest.raises(ValueError, match=r"cov_types\[0\].*'us', 'cs', or 'ar1'"):
            simulate_re_batch(
                np.array([1.0]),
                1.0,
                [2],
                [1],
                [False],
                1,
                cov_types=["unknown"],
            )

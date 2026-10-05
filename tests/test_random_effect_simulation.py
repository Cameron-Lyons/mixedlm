from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

try:
    from mixedlm._rust import compute_zu, simulate_re_batch

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = [
    pytest.mark.installed_wheel,
    pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available"),
]


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


@pytest.mark.parametrize(
    "data,indices,indptr,shape,u,n_obs,message",
    [
        ([1.0], [0], [0, 1], (2, 1), [1.0], 1, "n_obs must equal"),
        ([1.0], [0], [0, 1], (2, 1), [], 2, "u must contain exactly"),
        ([1.0], [0], [0, 1], (2, 1), [1.0, 2.0], 2, "u must contain exactly"),
        ([1.0], [-1], [0, 1], (2, 1), [1.0], 2, "indices"),
        ([1.0], [0], [0, -1], (2, 1), [1.0], 2, "indptr"),
        ([1.0], [2], [0, 1], (2, 1), [1.0], 2, "row index"),
        ([1.0], [0], [0], (2, 1), [1.0], 2, "indptr"),
        ([1.0], [0], [0, 2], (2, 1), [1.0], 2, "indptr"),
        ([1.0], [0], [1, 1], (2, 1), [1.0], 2, "indptr"),
        ([1.0], [], [0, 0], (2, 1), [1.0], 2, "data has length"),
        ([1.0], [0], [0, 1, 0, 1], (2, 3), [1.0] * 3, 2, "invalid range"),
    ],
)
def test_sparse_random_product_rejects_invalid_buffers(
    data, indices, indptr, shape, u, n_obs, message
):
    with pytest.raises(ValueError, match=message):
        compute_zu(u, data, indices, indptr, shape, n_obs)


def test_sparse_random_product_handles_unsorted_duplicates_and_empty_rows():
    data = np.array([2.0, 3.0, -0.5, 1.5, 4.0])
    indices = np.array([3, 1, 3, 0, 1], dtype=np.int64)
    indptr = np.array([0, 3, 3, 5], dtype=np.int64)
    design = sparse.csc_matrix((data, indices, indptr), shape=(5, 3))
    coefficients = np.array([0.8, -0.3, 1.2])
    actual = compute_zu(coefficients, data, indices, indptr, design.shape, 5)
    np.testing.assert_allclose(actual, design @ coefficients, rtol=0, atol=1e-15)


def test_sparse_random_product_multiplies_duplicates_before_their_sum_can_overflow():
    data = np.array([1e308, 1e308])
    indices = np.array([0, 0], dtype=np.int64)
    indptr = np.array([0, 2], dtype=np.int64)
    coefficients = np.array([1e-308])
    design = sparse.csc_matrix((data, indices, indptr), shape=(1, 1))

    actual = compute_zu(coefficients, data, indices, indptr, design.shape, 1)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, [2.0], rtol=2e-16)
    np.testing.assert_array_equal(actual, design @ coefficients)


@pytest.mark.parametrize(
    "data,indices,indptr,coefficients,expected",
    [
        ([1e16, 1.0, -1e16], [0, 0, 0], [0, 3], [0.1], [0.125, 0.0]),
        ([1e16, -1e16, 1.0], [0, 0, 0], [0, 3], [0.1], [0.1, 0.0]),
        # Unsorted row entries must retain each row's original accumulation order.
        ([1e16, 2.0, 1.0, -1e16, 3.0], [0, 1, 0, 0, 1], [0, 5], [0.1], [0.125, 0.5]),
        # Cancellation continues in CSC column order, including empty columns.
        ([1e16, 1.0, -1e16, 1.0], [0, 0, 0, 0], [0, 3, 3, 4], [0.1, 0, 0.5], [0.625, 0]),
    ],
)
def test_sparse_random_product_preserves_duplicate_and_column_accumulation_order(
    data, indices, indptr, coefficients, expected
):
    data = np.array(data, dtype=np.float64)
    indices = np.array(indices, dtype=np.int64)
    indptr = np.array(indptr, dtype=np.int64)
    coefficients = np.array(coefficients, dtype=np.float64)
    original = [array.copy() for array in (data, indices, indptr, coefficients)]
    design = sparse.csc_matrix((data, indices, indptr), shape=(2, len(coefficients)))

    actual = compute_zu(coefficients, data, indices, indptr, design.shape, 2)

    # SciPy's sparse product may fuse multiplication and addition on ARM,
    # changing these cancellation-sensitive results. Build each row separately
    # from COO entries and force the product to round before adding it instead.
    entries = design.tocoo()
    reference = np.zeros(design.shape[0])
    for row in range(design.shape[0]):
        total = 0.0
        for entry_row, column, value in zip(entries.row, entries.col, entries.data, strict=True):
            if entry_row == row:
                product = float(value) * float(coefficients[column])
                total += product
        reference[row] = total

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, reference)
    for array, saved in zip((data, indices, indptr, coefficients), original, strict=True):
        np.testing.assert_array_equal(array, saved)


@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 3)])
def test_sparse_random_product_handles_empty_dimensions(shape):
    design = sparse.csc_matrix(shape)
    actual = compute_zu(
        np.zeros(shape[1]),
        design.data,
        design.indices.astype(np.int64),
        design.indptr.astype(np.int64),
        shape,
        shape[0],
    )
    np.testing.assert_array_equal(actual, np.zeros(shape[0]))

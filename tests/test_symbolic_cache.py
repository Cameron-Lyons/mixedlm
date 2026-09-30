import numpy as np
import pytest
from mixedlm._rust import SparseCholeskySymbolic
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


class TestSymbolicCholeskyCache:
    def test_basic_solve(self):
        n = 5
        A = sparse.diags([4.0] * n, format="csc")
        A = A + sparse.diags([1.0] * (n - 1), 1, format="csc")
        A = A + sparse.diags([1.0] * (n - 1), -1, format="csc")
        A = A.tocsc()

        symbolic = SparseCholeskySymbolic(
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            n,
        )

        numeric = symbolic.factor(A.data)

        b = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        x = numeric.solve(b)

        for j in range(b.shape[1]):
            residual = A @ x[:, j] - b[:, j]
            assert_allclose(residual, 0, atol=1e-10)

    def test_logdet(self):
        n = 4
        A = sparse.eye(n, format="csc") * 2.0
        A = A.tocsc()

        symbolic = SparseCholeskySymbolic(
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            n,
        )
        numeric = symbolic.factor(A.data)

        logdet = numeric.logdet()
        expected = n * np.log(2.0)
        assert_allclose(logdet, expected, rtol=1e-10)

    def test_cache_reuse_with_different_values(self):
        n = 10
        A1 = sparse.diags([4.0] * n, format="csc")
        A1 = A1 + sparse.diags([1.0] * (n - 1), 1, format="csc")
        A1 = A1 + sparse.diags([1.0] * (n - 1), -1, format="csc")
        A1 = A1.tocsc()

        symbolic = SparseCholeskySymbolic(
            A1.indices.astype(np.int64),
            A1.indptr.astype(np.int64),
            n,
        )

        numeric1 = symbolic.factor(A1.data)
        b = np.ones((n, 1))
        x1 = numeric1.solve(b)

        A2_data = A1.data.copy()
        A2_data[A2_data == 4.0] = 5.0
        numeric2 = symbolic.factor(A2_data)
        x2 = numeric2.solve(b)

        assert not np.allclose(x1, x2)

        A2 = A1.copy()
        A2.data[:] = A2_data
        residual2 = A2 @ x2[:, 0] - b[:, 0]
        assert_allclose(residual2, 0, atol=1e-10)

    def test_versus_scipy(self):
        rng = np.random.default_rng(42)
        n = 20
        A = sparse.random(n, n, density=0.3, format="csc", random_state=rng)
        A = A @ A.T + sparse.eye(n, format="csc") * 5.0
        A = A.tocsc()
        A.sort_indices()

        symbolic = SparseCholeskySymbolic(
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            n,
        )
        numeric = symbolic.factor(A.data)

        b = rng.standard_normal((n, 1))
        x_cached = numeric.solve(b)

        x_scipy = sparse_linalg.spsolve(A, b.ravel())

        assert_allclose(x_cached.ravel(), x_scipy, rtol=1e-8)

    @pytest.mark.parametrize("pattern", ["tridiagonal", "random"])
    @pytest.mark.parametrize("n_rhs", [1, 3])
    def test_repeated_factorizations_preserve_numeric_factors(self, pattern, n_rhs):
        rng = np.random.default_rng(123)
        n = 40
        if pattern == "tridiagonal":
            A_base = sparse.diags(
                (-np.ones(n - 1), np.full(n, 4.0), -np.ones(n - 1)),
                offsets=(-1, 0, 1),
                format="csc",
            )
        else:
            A_base = sparse.random(n, n, density=0.1, format="csc", random_state=rng)
            A_base = A_base @ A_base.T + sparse.eye(n, format="csc") * 10.0
        A_base = A_base.tocsc()
        A_base.sort_indices()

        symbolic = SparseCholeskySymbolic(
            A_base.indices.astype(np.int64),
            A_base.indptr.astype(np.int64),
            n,
        )

        b = rng.standard_normal((n, n_rhs))
        factors = []
        for i, scale in enumerate((0.25, 1.0, 4.0, 0.5, 2.0, 0.75, 3.0, 1.25)):
            A_scaled = A_base.copy()
            A_scaled.data *= scale
            # Vary diagonal entries independently while preserving the sparse pattern.
            A_scaled.setdiag(A_scaled.diagonal() + (i + 1) * np.linspace(0.1, 1.0, n))
            np.testing.assert_array_equal(A_scaled.indices, A_base.indices)
            np.testing.assert_array_equal(A_scaled.indptr, A_base.indptr)
            factors.append((symbolic.factor(A_scaled.data), A_scaled))

        # Earlier numeric factors must remain usable after reusing the symbolic cache.
        for numeric, matrix in reversed(factors):
            solution = numeric.solve(b)
            assert_allclose(matrix @ solution, b, rtol=1e-11, atol=1e-11)
            sign, expected_logdet = np.linalg.slogdet(matrix.toarray())
            assert sign == 1
            assert_allclose(numeric.logdet(), expected_logdet, rtol=1e-11)

    def test_singular_matrix_raises(self):
        n = 3
        A = sparse.csc_matrix(np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1]]))

        symbolic = SparseCholeskySymbolic(
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            n,
        )

        with pytest.raises(ValueError, match="positive definite"):
            symbolic.factor(A.data)

    def test_n_property(self):
        n = 7
        A = sparse.eye(n, format="csc")

        symbolic = SparseCholeskySymbolic(
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            n,
        )

        assert symbolic.n() == n

"""Measure public sparse Cholesky ordering on independently verified hub systems.

Run from an installed checkout:
    python benchmarks/benchmark_sparse_ordering.py --sizes 128 512 1024 --repeats 5

Times include symbolic analysis, numeric factorization, and a five-column solve.
Validation and input construction are outside the timed region. No speed cutoff
is asserted: factor storage and numerical oracles provide stable regressions.
"""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from mixedlm import SparseCholeskySymbolic
from scipy import sparse


def arrowhead_system(size: int):
    diagonal = np.linspace(1.2, 3.7, size)
    coupling = 0.09 * np.cos(np.arange(1, size) + 0.3)
    schur = 0.8
    diagonal[0] = schur + np.sum(coupling**2 / diagonal[1:])
    leaves = np.arange(1, size)
    rows = np.concatenate((np.arange(size), leaves, np.zeros(size - 1, dtype=int)))
    columns = np.concatenate((np.arange(size), np.zeros(size - 1, dtype=int), leaves))
    matrix = sparse.csc_matrix(
        (np.concatenate((diagonal, coupling, coupling)), (rows, columns)), shape=(size, size)
    )
    rhs = np.random.default_rng(440).normal(size=(size, 5))
    expected = np.empty_like(rhs)
    expected[0] = (rhs[0] - coupling @ (rhs[1:] / diagonal[1:, None])) / schur
    expected[1:] = (rhs[1:] - coupling[:, None] * expected[0]) / diagonal[1:, None]
    logdet = np.log(diagonal[1:]).sum() + np.log(schur)
    return matrix, rhs, expected, logdet


def benchmark(size: int, repeats: int) -> dict:
    matrix, rhs, expected, logdet = arrowhead_system(size)
    indices = matrix.indices.astype(np.int64)
    offsets = matrix.indptr.astype(np.int64)
    output = {"size": size, "input_nonzeros": matrix.nnz}
    for ordering in ("natural", "amd"):
        durations = []
        for _ in range(repeats + 1):
            start = perf_counter()
            symbolic = SparseCholeskySymbolic(indices, offsets, size, ordering=ordering)
            numeric = symbolic.factor(matrix.data)
            result = numeric.solve(rhs)
            durations.append(perf_counter() - start)
            np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(matrix @ result, rhs, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(numeric.logdet(), logdet, rtol=2e-12, atol=2e-12)
        output[ordering] = {
            "median_seconds": median(durations[1:]),
            "factor_nonzeros": symbolic.factor_nonzeros(),
        }
    output["speedup"] = output["natural"]["median_seconds"] / output["amd"]["median_seconds"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1:
        parser.error("sizes must be at least two and repeats must be positive")
    print(json.dumps([benchmark(size, args.repeats) for size in args.sizes], indent=2))


if __name__ == "__main__":
    main()

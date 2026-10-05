"""Sparse systems with closed-form oracles and raw CSC storage variants for native tests."""

import numpy as np
from scipy import sparse


def arrowhead_system(size, hub=0):
    """Build a sparse SPD system with a closed-form Schur complement oracle."""
    leaves = np.delete(np.arange(size), hub)
    diagonal = np.linspace(1.2, 3.7, size)
    coupling = 0.09 * np.cos(leaves + 0.3)
    schur = 0.8
    diagonal[hub] = schur + np.sum(coupling**2 / diagonal[leaves])
    rows = np.concatenate((np.arange(size), leaves, np.full(size - 1, hub)))
    columns = np.concatenate((np.arange(size), np.full(size - 1, hub), leaves))
    values = np.concatenate((diagonal, coupling, coupling))
    matrix = sparse.csc_matrix((values, (rows, columns)), shape=(size, size))
    rng = np.random.default_rng(440 + hub)
    rhs = rng.normal(size=(size, 7))
    solution = np.empty_like(rhs)
    solution[hub] = (rhs[hub] - coupling @ (rhs[leaves] / diagonal[leaves, None])) / schur
    solution[leaves] = (rhs[leaves] - coupling[:, None] * solution[hub]) / diagonal[leaves, None]
    logdet = np.log(diagonal[leaves]).sum() + np.log(schur)
    return matrix, rhs, solution, logdet


def sparse_arguments(matrix, storage):
    """Return owned CSC data, row indices and column offsets for native calls."""
    matrix = sparse.tril(matrix, format="csc") if storage == "lower" else matrix
    if storage != "noncanonical":
        return matrix.data.copy(), matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64)
    # Valid raw CSC can contain duplicates and descending row indices. The
    # symbolic mapping must retain the numeric sources after canonicalization.
    values, rows, offsets = [], [], [0]
    for column in range(matrix.shape[1]):
        for entry in range(matrix.indptr[column + 1] - 1, matrix.indptr[column] - 1, -1):
            rows.extend([matrix.indices[entry]] * 2)
            values.extend([0.25 * matrix.data[entry], 0.75 * matrix.data[entry]])
        offsets.append(len(values))
    return np.asarray(values), np.asarray(rows, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def noncanonical_csc(z):
    """Split each value between duplicates, reverse rows, and store explicit zeros."""
    values, rows, offsets = [], [], [0]
    for column in range(z.shape[1]):
        for index in range(z.indptr[column + 1] - 1, z.indptr[column] - 1, -1):
            values.extend([0.25 * z.data[index], 0.75 * z.data[index]])
            rows.extend([z.indices[index], z.indices[index]])
        if z.shape[0]:
            values.append(0.0)
            rows.append(column % z.shape[0])
        offsets.append(len(values))
    return sparse.csc_matrix((values, rows, offsets), shape=z.shape)

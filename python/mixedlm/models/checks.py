from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

if TYPE_CHECKING:
    from mixedlm.matrices.design import ModelMatrices
    from mixedlm.models.control import GlmerControl, LmerControl


class ModelCheckError(Exception):
    pass


def _handle_check(
    action: str,
    message: str,
    category: type[Warning] = UserWarning,
) -> None:
    if action == "stop":
        raise ModelCheckError(message)
    elif action == "warning":
        warnings.warn(message, category, stacklevel=4)
    elif action == "message":
        print(message)


def check_nobs_vs_rankZ(
    matrices: ModelMatrices,
    action: str,
    *,
    allow_equal: bool = False,
) -> None:
    if action == "ignore":
        return

    n_obs = matrices.y.shape[0]
    Z = matrices.Z
    if Z is None:
        return

    # rank(Z) cannot exceed the number of observations, so generalized models
    # that allow equality can never fail this check. Linear models also need
    # their residual variance identified and therefore reject full row rank.
    if allow_equal or not _has_full_row_rank(Z):
        return

    _handle_check(
        action,
        f"Number of observations ({n_obs}) <= rank(Z) ({n_obs}). "
        "Random-effects parameters and residual variance may be unidentifiable.",
    )


def _has_full_row_rank(
    matrix: NDArray[np.floating] | sparse.spmatrix,
) -> bool:
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols < n_rows:
        return False

    if not sparse.issparse(matrix):
        return bool(np.linalg.matrix_rank(matrix) == n_rows)

    sparse_matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    if sparse.csgraph.structural_rank(sparse_matrix) < n_rows:
        return False

    if n_rows == 1:
        return bool(sparse_matrix.nnz)

    largest = float(
        sparse_linalg.svds(
            sparse_matrix,
            k=1,
            which="LM",
            return_singular_vectors=False,
            random_state=0,
        )[0]
    )
    if largest == 0.0:
        return False

    smallest = float(
        sparse_linalg.svds(
            sparse_matrix,
            k=1,
            which="SM",
            return_singular_vectors=False,
            random_state=0,
        )[0]
    )
    tolerance = largest * max(sparse_matrix.shape) * np.finfo(np.float64).eps
    return smallest > tolerance


def check_nobs_vs_nlev(
    matrices: ModelMatrices,
    action: str,
) -> None:
    if action == "ignore":
        return

    n_obs = matrices.y.shape[0]

    for struct in matrices.random_structures:
        n_levels = struct.n_levels
        obs_per_level = n_obs / n_levels

        if obs_per_level < 2:
            _handle_check(
                action,
                f"Grouping factor '{struct.grouping_factor}' has only "
                f"{obs_per_level:.1f} observations per level on average. "
                "Consider using fewer random effect levels.",
            )


def check_nlev_gtreq_5(
    matrices: ModelMatrices,
    action: str,
) -> None:
    if action == "ignore":
        return

    for struct in matrices.random_structures:
        n_levels = struct.n_levels
        if n_levels < 5:
            _handle_check(
                action,
                f"Grouping factor '{struct.grouping_factor}' has only "
                f"{n_levels} levels. Mixed models typically need at least "
                "5-6 levels for reliable variance component estimation.",
            )


def check_nlev_gtr_1(
    matrices: ModelMatrices,
    action: str,
) -> None:
    if action == "ignore":
        return

    for struct in matrices.random_structures:
        n_levels = struct.n_levels
        if n_levels <= 1:
            _handle_check(
                action,
                f"Grouping factor '{struct.grouping_factor}' has only "
                f"{n_levels} level(s). Random effects require at least "
                "2 levels.",
            )


def kept_columns(X: NDArray[np.floating], tol: float = 1e-7) -> NDArray[np.intp]:
    """Columns that R's LINPACK QR (``dqrdc2``, used by ``lm()`` and lme4) keeps.

    A column is aliased when its norm after removing the kept columns before it
    falls below ``tol`` times its original norm. R moves such columns to the end
    and drops them, so an earlier column always wins over a later one.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    norms = np.linalg.norm(X, axis=0)
    limits = tol * np.where(norms > 0, norms, 1.0)
    if 0 < p <= n:
        # Without aliasing, an unpivoted QR gives the same residual norms.
        diagonal = np.abs(np.diag(linalg.qr(X, mode="r", check_finite=False)[0]))
        if np.all(diagonal >= limits):
            return np.arange(p)
    basis = np.empty((n, min(n, p)))
    kept: list[int] = []
    for j in range(p):
        if len(kept) == n:
            break
        q = basis[:, : len(kept)]
        residual = X[:, j] - q @ (q.T @ X[:, j])
        residual -= q @ (q.T @ residual)
        size = float(np.linalg.norm(residual))
        if size < limits[j]:
            continue
        basis[:, len(kept)] = residual / size
        kept.append(j)
    return np.asarray(kept, dtype=np.intp)


_LME4_RANKX_ACTIONS = {
    "silent.drop.cols": "ignore+drop.cols",
    "warn+drop.cols": "warning+drop.cols",
    "stop.deficient": "stop",
}


def check_rankX(
    matrices: ModelMatrices,
    action: str,
) -> tuple[NDArray[np.floating], list[int] | None]:
    X = matrices.X
    p = X.shape[1]

    if p == 0:
        return X, None

    kept = [int(i) for i in kept_columns(X)]
    rank = len(kept)

    if rank == p:
        return X, None

    action = _LME4_RANKX_ACTIONS.get(action, action)
    drop_cols = action.endswith("+drop.cols")
    base_action = action.replace("+drop.cols", "") if drop_cols else action

    if base_action == "ignore" and not drop_cols:
        return X, None

    message = f"Fixed-effects design matrix is rank deficient (rank {rank} < {p} columns)."

    if drop_cols:
        dropped = sorted(set(range(p)) - set(kept))

        if dropped:
            message += f" Dropping columns: {dropped}"
            names = getattr(matrices, "fixed_names", None)
            if names is not None and len(names) == p:
                message += f" ({', '.join(names[i] for i in dropped)})"
            if base_action != "ignore":
                _handle_check(base_action, message)
            X_new = X[:, kept]
            return X_new, dropped
    else:
        _handle_check(base_action, message)

    return X, None


def check_scaleX(
    matrices: ModelMatrices,
    action: str,
) -> None:
    if action == "ignore":
        return

    X = matrices.X

    if X.shape[1] <= 1:
        return

    col_ranges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col_range = np.ptp(col)
        if col_range > 0:
            col_ranges.append(col_range)

    if len(col_ranges) < 2:
        return

    col_ranges_arr = np.array(col_ranges)
    log_ratio = np.log10(np.max(col_ranges_arr) / np.min(col_ranges_arr))

    if log_ratio > 3:
        _handle_check(
            action,
            f"Fixed-effects columns have very different scales "
            f"(ratio: 10^{log_ratio:.1f}). Consider centering/scaling "
            "predictors for numerical stability.",
        )


def run_model_checks(
    matrices: ModelMatrices,
    control: LmerControl | GlmerControl,
) -> tuple[ModelMatrices, list[int] | None]:
    from mixedlm.models.control import GlmerControl

    check_nlev_gtr_1(matrices, control.check_nlev_gtr_1)
    check_nlev_gtreq_5(matrices, control.check_nlev_gtreq_5)
    check_nobs_vs_nlev(matrices, control.check_nobs_vs_nlev)
    check_nobs_vs_rankZ(
        matrices,
        control.check_nobs_vs_rankZ,
        allow_equal=isinstance(control, GlmerControl),
    )
    check_scaleX(matrices, control.check_scaleX)

    X_new, dropped_cols = check_rankX(matrices, control.check_rankX)

    if dropped_cols is not None:
        from dataclasses import replace

        dropped = set(dropped_cols)
        fixed_names = [
            name for index, name in enumerate(matrices.fixed_names) if index not in dropped
        ]
        source_names = matrices.fixed_source_names
        source_indices = matrices.fixed_column_indices
        if source_names is None or source_indices is None:
            source_names = tuple(matrices.fixed_names)
            source_indices = tuple(range(len(matrices.fixed_names)))
        matrices = replace(
            matrices,
            X=X_new,
            fixed_names=fixed_names,
            n_fixed=X_new.shape[1],
            fixed_source_names=source_names,
            fixed_column_indices=tuple(
                source for index, source in enumerate(source_indices) if index not in dropped
            ),
        )

    return matrices, dropped_cols

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mixedlm.matrices.design import MixedModelMatrices
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
    matrices: MixedModelMatrices,
    action: str,
) -> None:
    if action == "ignore":
        return

    n_obs = matrices.y.shape[0]
    Z = matrices.Z
    if Z is None:
        return

    if hasattr(Z, "toarray"):
        Z_dense = Z.toarray()
    else:
        Z_dense = Z

    rank_Z = np.linalg.matrix_rank(Z_dense)

    if n_obs < rank_Z:
        _handle_check(
            action,
            f"Number of observations ({n_obs}) < rank(Z) ({rank_Z}). "
            "Model may be overparameterized.",
        )


def check_nobs_vs_nlev(
    matrices: MixedModelMatrices,
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
    matrices: MixedModelMatrices,
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
    matrices: MixedModelMatrices,
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


def check_rankX(
    matrices: MixedModelMatrices,
    action: str,
) -> tuple[NDArray[np.floating], list[int] | None]:
    X = matrices.X
    n, p = X.shape

    rank = np.linalg.matrix_rank(X)

    if rank == p:
        return X, None

    drop_cols = action.endswith("+drop.cols")
    base_action = action.replace("+drop.cols", "") if drop_cols else action

    if base_action == "ignore" and not drop_cols:
        return X, None

    message = f"Fixed-effects design matrix is rank deficient (rank {rank} < {p} columns)."

    if drop_cols:
        _, R = np.linalg.qr(X)
        diag_R = np.abs(np.diag(R))
        tol = max(n, p) * np.finfo(X.dtype).eps * np.max(diag_R)
        keep_cols = diag_R > tol
        dropped = [i for i, keep in enumerate(keep_cols) if not keep]

        if dropped:
            message += f" Dropping columns: {dropped}"
            if base_action != "ignore":
                _handle_check(base_action, message)
            X_new = X[:, keep_cols]
            return X_new, dropped
    else:
        _handle_check(base_action, message)

    return X, None


def check_scaleX(
    matrices: MixedModelMatrices,
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

    col_ranges = np.array(col_ranges)
    log_ratio = np.log10(np.max(col_ranges) / np.min(col_ranges))

    if log_ratio > 3:
        _handle_check(
            action,
            f"Fixed-effects columns have very different scales "
            f"(ratio: 10^{log_ratio:.1f}). Consider centering/scaling "
            "predictors for numerical stability.",
        )


def run_model_checks(
    matrices: MixedModelMatrices,
    control: LmerControl | GlmerControl,
) -> tuple[MixedModelMatrices, list[int] | None]:
    check_nlev_gtr_1(matrices, control.check_nlev_gtr_1)
    check_nlev_gtreq_5(matrices, control.check_nlev_gtreq_5)
    check_nobs_vs_nlev(matrices, control.check_nobs_vs_nlev)
    check_nobs_vs_rankZ(matrices, control.check_nobs_vs_rankZ)
    check_scaleX(matrices, control.check_scaleX)

    X_new, dropped_cols = check_rankX(matrices, control.check_rankX)

    if dropped_cols is not None:
        from dataclasses import replace

        matrices = replace(matrices, X=X_new)

    return matrices, dropped_cols

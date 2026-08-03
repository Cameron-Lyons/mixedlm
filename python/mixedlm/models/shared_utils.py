from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.utils.dataframe import dataframe_length, get_column_numpy, get_columns


def is_singular_theta(theta: NDArray[np.floating], structures: list[Any], tol: float) -> bool:
    """Return whether any random-effect covariance block is rank deficient."""
    theta_idx = 0

    for struct in structures:
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type in ("cs", "ar1"):
            sigma_rel = theta[theta_idx]
            theta_idx += 2 if q > 1 else 1
            if abs(sigma_rel) < tol:
                return True
        elif struct.correlated:
            n_theta = q * (q + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta

            diag_idx = 0
            for i in range(q):
                if abs(theta_block[diag_idx]) < tol:
                    return True
                diag_idx += i + 2
        else:
            theta_block = theta[theta_idx : theta_idx + q]
            theta_idx += q
            if np.any(np.abs(theta_block) < tol):
                return True

    return False


def resolve_optional_vector(
    data: Any,
    value: NDArray[np.floating] | str | None,
    name: str,
) -> NDArray[np.floating] | None:
    """Resolve optional vector-like inputs from array or data column name."""
    if value is None:
        return None

    if isinstance(value, str):
        if value not in get_columns(data):
            raise ValueError(f"{name} column '{value}' not found in data")
        return get_column_numpy(data, value, dtype=np.float64)

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.shape[0] != dataframe_length(data):
        raise ValueError(
            f"{name} length {arr.shape[0]} does not match data length {dataframe_length(data)}"
        )
    return arr

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.utils.dataframe import dataframe_length, get_column_numpy, get_columns


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

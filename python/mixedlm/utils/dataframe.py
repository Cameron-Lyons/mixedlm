from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@runtime_checkable
class DataFrameLike(Protocol):
    @property
    def columns(self) -> list[str]: ...
    def __len__(self) -> int: ...


def get_column_numpy(data: Any, name: str, dtype: type | None = None) -> NDArray:
    """Extract a column as a numpy array.

    Works with pandas DataFrames, polars DataFrames, and polars LazyFrames.

    Parameters
    ----------
    data : DataFrame
        pandas or polars DataFrame.
    name : str
        Column name.
    dtype : type, optional
        If provided, convert to this dtype. If None, preserve original dtype.
    """
    data = ensure_dataframe(data, columns=[name])

    if _is_polars(data):
        arr = data.get_column(name).to_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    col = data[name]
    if dtype is not None:
        try:
            return col.to_numpy(dtype=dtype)
        except (ValueError, TypeError):
            return col.to_numpy()
    return col.to_numpy()


def get_column_values(data: Any, name: str) -> NDArray:
    """Extract column values preserving dtype (for categorical handling)."""
    data = ensure_dataframe(data, columns=[name])
    if _is_polars(data):
        col = data.get_column(name)
        return col.to_numpy()
    return data[name].values


def get_row_value(data: Any, col_name: str, row_idx: int) -> Any:
    """Get a single value from a column at a specific row index."""
    data = ensure_dataframe(data, columns=[col_name])
    if _is_polars(data):
        return data.get_column(col_name)[row_idx]
    return data[col_name].iloc[row_idx]


def get_unique_sorted(data: Any, col_name: str) -> list:
    """Get sorted unique values from a column, excluding nulls."""
    data = ensure_dataframe(data, columns=[col_name])
    if _is_polars(data):
        col = data.get_column(col_name)
        unique_vals = col.drop_nulls().unique().sort().to_list()
        return unique_vals
    col = data[col_name]
    return sorted(col.dropna().unique().tolist())


def get_categories(data: Any, col_name: str) -> list:
    """Get category levels for a categorical column, or sorted unique values."""
    data = ensure_dataframe(data, columns=[col_name])
    if _is_polars(data):
        col = data.get_column(col_name)
        dtype_name = str(col.dtype)
        if "Categorical" in dtype_name or "Enum" in dtype_name:
            cats = col.cat.get_categories().to_list()
            return cats if cats else get_unique_sorted(data, col_name)
        return get_unique_sorted(data, col_name)

    col = data[col_name]
    if hasattr(col.dtype, "name") and col.dtype.name == "category":
        return col.cat.categories.tolist()
    return sorted(col.dropna().unique().tolist())


def is_categorical_or_string(data: Any, col_name: str) -> bool:
    """Check if a column is categorical or string type."""
    data = ensure_dataframe(data, columns=[col_name])
    if _is_polars(data):
        import polars as pl

        col = data.get_column(col_name)
        return col.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum)

    col = data[col_name]

    if col.dtype.name == "category":
        return True

    if col.dtype == object:
        return True

    dtype_str = str(col.dtype)
    return "string" in dtype_str.lower() or "str" in dtype_str.lower()


def dataframe_length(data: Any) -> int:
    """Get the number of rows in a DataFrame."""
    data = ensure_dataframe(data)
    if _is_polars(data):
        return data.height
    return len(data)


def get_columns(data: Any) -> list[str]:
    """Get column names from a DataFrame."""
    if _is_polars_lazy(data):
        return data.collect_schema().names()
    if _is_polars(data):
        return data.columns
    return list(data.columns)


def select_columns(data: Any, columns: list[str]) -> Any:
    """Select specific columns from a DataFrame."""
    data = ensure_dataframe(data, columns=columns)
    if _is_polars(data):
        return data.select(columns)
    return data[columns].copy()


def copy_dataframe(data: Any) -> Any:
    """Return an independent copy while preserving the dataframe backend."""
    data = ensure_dataframe(data)
    if _is_polars(data):
        return data.clone()
    return data.copy()


def concat_columns_as_string(data: Any, columns: list[str], separator: str = "/") -> NDArray:
    """Concatenate multiple columns into a single string column."""
    import numpy as np

    if not columns:
        return np.full(dataframe_length(data), "", dtype=object)

    data = ensure_dataframe(data, columns=columns)
    if _is_polars(data):
        import polars as pl

        expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in columns], separator=separator)
        result = data.select(expr.alias("_combined")).get_column("_combined")
        return result.to_numpy()

    values = data[list(columns)].to_numpy(dtype=object)
    string_values = np.frompyfunc(str, 1, 1)(values)
    combine = np.frompyfunc(lambda left, right: left + separator + right, 2, 1)
    combined = string_values[:, 0]
    for column_index in range(1, string_values.shape[1]):
        combined = combine(combined, string_values[:, column_index])
    return np.asarray(combined, dtype=object)


def factorize_levels(values: NDArray) -> tuple[NDArray, dict[Any, int]]:
    """Factor grouping values into sorted integer levels in one pass."""
    import numpy as np
    import pandas as pd

    codes, levels = pd.factorize(values, sort=True)
    indices = np.asarray(codes, dtype=np.int64)
    level_values = levels.tolist()
    return indices, {level: index for index, level in enumerate(level_values)}


def _is_polars(data: Any) -> bool:
    """Check if data is a polars DataFrame or LazyFrame."""
    return type(data).__module__.partition(".")[0] == "polars"


def _is_polars_lazy(data: Any) -> bool:
    """Check if data is a polars LazyFrame without importing polars."""
    return _is_polars(data) and type(data).__name__ == "LazyFrame"


def _is_pandas(data: Any) -> bool:
    """Check if data is a pandas DataFrame."""
    return type(data).__module__.partition(".")[0] == "pandas"


def ensure_dataframe(data: Any, columns: Collection[str] | None = None) -> Any:
    """Validate and materialize a supported DataFrame type.

    Polars LazyFrames are collected once. When ``columns`` is provided, the
    projection is applied before collection so Polars can avoid evaluating and
    loading columns that are not needed by the model.
    """
    if _is_pandas(data):
        return data
    if _is_polars_lazy(data):
        if columns is not None:
            data = data.select(list(columns))
        return data.collect()
    if _is_polars(data):
        return data
    raise TypeError(
        f"Expected pandas or polars DataFrame, got {type(data).__name__}. "
        "Install pandas or polars and pass a DataFrame."
    )

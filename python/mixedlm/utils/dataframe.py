from __future__ import annotations

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
    if _is_polars(data):
        col = data.get_column(name)
        return col.to_numpy()
    return data[name].values


def get_row_value(data: Any, col_name: str, row_idx: int) -> Any:
    """Get a single value from a column at a specific row index."""
    if _is_polars(data):
        return data.get_column(col_name)[row_idx]
    return data[col_name].iloc[row_idx]


def get_unique_sorted(data: Any, col_name: str) -> list:
    """Get sorted unique values from a column, excluding nulls."""
    if _is_polars(data):
        col = data.get_column(col_name)
        unique_vals = col.drop_nulls().unique().sort().to_list()
        return unique_vals
    col = data[col_name]
    return sorted(col.dropna().unique().tolist())


def get_categories(data: Any, col_name: str) -> list:
    """Get category levels for a categorical column, or sorted unique values."""
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
    if _is_polars(data):
        import polars as pl

        col = data.get_column(col_name)
        return col.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum)

    col = data[col_name]

    # Check for categorical
    if col.dtype.name == "category":
        return True

    # Check for object dtype (strings in older pandas)
    if col.dtype == object:
        return True

    # Check for pandas 2.x StringDtype
    dtype_str = str(col.dtype)
    return "string" in dtype_str.lower() or "str" in dtype_str.lower()


def dataframe_length(data: Any) -> int:
    """Get the number of rows in a DataFrame."""
    if _is_polars(data):
        return data.height
    return len(data)


def get_columns(data: Any) -> list[str]:
    """Get column names from a DataFrame."""
    if _is_polars(data):
        return data.columns
    return list(data.columns)


def select_columns(data: Any, columns: list[str]) -> Any:
    """Select specific columns from a DataFrame."""
    if _is_polars(data):
        return data.select(columns)
    return data[columns].copy()


def concat_columns_as_string(data: Any, columns: list[str], separator: str = "/") -> NDArray:
    """Concatenate multiple columns into a single string column."""
    if _is_polars(data):
        import polars as pl

        expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in columns], separator=separator)
        result = data.select(expr.alias("_combined")).get_column("_combined")
        return result.to_numpy()

    combined = data[list(columns)].apply(lambda row: separator.join(str(x) for x in row), axis=1)
    dtype_str = str(combined.dtype)
    if "string" in dtype_str.lower():
        combined = combined.astype(object)
    return combined.values


def _is_polars(data: Any) -> bool:
    """Check if data is a polars DataFrame or LazyFrame."""
    type_name = type(data).__module__
    return "polars" in type_name


def _is_pandas(data: Any) -> bool:
    """Check if data is a pandas DataFrame."""
    type_name = type(data).__module__
    return "pandas" in type_name


def ensure_dataframe(data: Any) -> Any:
    """Validate that data is a supported DataFrame type.

    Returns the data unchanged if valid, raises TypeError otherwise.
    """
    if _is_pandas(data) or _is_polars(data):
        return data
    raise TypeError(
        f"Expected pandas or polars DataFrame, got {type(data).__name__}. "
        "Install pandas or polars and pass a DataFrame."
    )

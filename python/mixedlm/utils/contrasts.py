from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


class ContrastType(Enum):
    TREATMENT = "treatment"
    SUM = "sum"
    HELMERT = "helmert"
    POLY = "poly"
    DEVIATION = "deviation"


def contr_treatment(n: int, base: int = 0) -> NDArray[np.floating]:
    """Treatment (dummy) contrasts.

    Compares each level to a reference level (default: first level).
    This is the default in R and creates n-1 dummy variables.

    Parameters
    ----------
    n : int
        Number of levels.
    base : int, default 0
        Index of the reference level (0-indexed).

    Returns
    -------
    NDArray
        Contrast matrix of shape (n, n-1).
    """
    if n < 2:
        return np.ones((n, 1), dtype=np.float64)

    contrasts = np.eye(n, dtype=np.float64)
    contrasts = np.delete(contrasts, base, axis=1)
    return contrasts


def contr_sum(n: int) -> NDArray[np.floating]:
    """Sum (deviation) contrasts.

    Compares each level to the grand mean. The coefficients for each
    factor sum to zero. The last level is the reference.

    Parameters
    ----------
    n : int
        Number of levels.

    Returns
    -------
    NDArray
        Contrast matrix of shape (n, n-1).
    """
    if n < 2:
        return np.ones((n, 1), dtype=np.float64)

    contrasts = np.eye(n, n - 1, dtype=np.float64)
    contrasts[-1, :] = -1
    return contrasts


def contr_helmert(n: int) -> NDArray[np.floating]:
    """Helmert contrasts.

    Compares each level to the mean of subsequent levels.
    Useful for ordered factors where you want to test if
    each level differs from the average of later levels.

    Parameters
    ----------
    n : int
        Number of levels.

    Returns
    -------
    NDArray
        Contrast matrix of shape (n, n-1).
    """
    if n < 2:
        return np.ones((n, 1), dtype=np.float64)

    contrasts = np.zeros((n, n - 1), dtype=np.float64)
    for j in range(n - 1):
        contrasts[: j + 1, j] = -1
        contrasts[j + 1, j] = j + 1

    for j in range(n - 1):
        contrasts[:, j] = contrasts[:, j] / np.sqrt(np.sum(contrasts[:, j] ** 2))

    return contrasts


def contr_poly(n: int) -> NDArray[np.floating]:
    """Polynomial contrasts for ordered factors.

    Creates orthogonal polynomial contrasts (linear, quadratic, cubic, etc.)
    for ordered categorical variables.

    Parameters
    ----------
    n : int
        Number of levels.

    Returns
    -------
    NDArray
        Contrast matrix of shape (n, n-1).
    """
    if n < 2:
        return np.ones((n, 1), dtype=np.float64)

    x = np.arange(1, n + 1, dtype=np.float64)
    x = (x - x.mean()) / x.std()

    contrasts = np.zeros((n, n - 1), dtype=np.float64)

    for degree in range(1, n):
        col = x**degree

        for prev_degree in range(degree):
            proj = np.dot(col, contrasts[:, prev_degree])
            col = col - proj * contrasts[:, prev_degree]

        norm = np.sqrt(np.sum(col**2))
        if norm > 1e-10:
            contrasts[:, degree - 1] = col / norm

    return contrasts


def contr_deviation(n: int) -> NDArray[np.floating]:
    """Deviation contrasts (effect coding).

    Similar to sum contrasts but with a different parameterization.
    Each level is compared to the overall mean, with the last level
    being the reference (coefficient = -1 for all contrasts).

    Parameters
    ----------
    n : int
        Number of levels.

    Returns
    -------
    NDArray
        Contrast matrix of shape (n, n-1).
    """
    return contr_sum(n)


def get_contrast_matrix(
    n_levels: int,
    contrast_type: str | ContrastType | NDArray[np.floating] | None = None,
    base: int = 0,
) -> NDArray[np.floating]:
    """Get contrast matrix for a categorical variable.

    Parameters
    ----------
    n_levels : int
        Number of levels in the categorical variable.
    contrast_type : str, ContrastType, or NDArray, optional
        Type of contrasts to use. Can be:
        - "treatment" (default): Treatment/dummy contrasts
        - "sum": Sum/deviation contrasts
        - "helmert": Helmert contrasts
        - "poly": Polynomial contrasts for ordered factors
        - A custom contrast matrix (NDArray of shape (n_levels, n_levels-1))
    base : int, default 0
        Reference level for treatment contrasts (0-indexed).

    Returns
    -------
    NDArray
        Contrast matrix of shape (n_levels, n_levels-1).
    """
    if contrast_type is None:
        contrast_type = "treatment"

    if isinstance(contrast_type, np.ndarray):
        if contrast_type.shape[0] != n_levels:
            raise ValueError(
                f"Custom contrast matrix has {contrast_type.shape[0]} rows, "
                f"expected {n_levels} for number of levels."
            )
        return contrast_type.astype(np.float64)

    if isinstance(contrast_type, ContrastType):
        contrast_type = contrast_type.value

    contrast_type = contrast_type.lower()

    if contrast_type == "treatment":
        return contr_treatment(n_levels, base=base)
    elif contrast_type == "sum":
        return contr_sum(n_levels)
    elif contrast_type == "helmert":
        return contr_helmert(n_levels)
    elif contrast_type == "poly":
        return contr_poly(n_levels)
    elif contrast_type == "deviation":
        return contr_deviation(n_levels)
    else:
        raise ValueError(
            f"Unknown contrast type: '{contrast_type}'. "
            "Use 'treatment', 'sum', 'helmert', 'poly', or provide a custom matrix."
        )


def apply_contrasts(
    col: pd.Series,  # type: ignore[type-arg]
    name: str,
    contrast_matrix: NDArray[np.floating],
    categories: list,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    """Apply contrast matrix to a categorical column.

    Parameters
    ----------
    col : Series
        Categorical column from the data.
    name : str
        Variable name for creating column names.
    contrast_matrix : NDArray
        Contrast matrix of shape (n_levels, n_cols).
    categories : list
        List of category levels in order.

    Returns
    -------
    tuple
        (list of encoded columns, list of column names)
    """
    n = len(col)
    n_contrasts = contrast_matrix.shape[1]

    level_to_idx = {cat: i for i, cat in enumerate(categories)}

    columns: list[NDArray[np.floating]] = []
    names: list[str] = []

    for j in range(n_contrasts):
        encoded = np.zeros(n, dtype=np.float64)
        for i in range(n):
            val = col.iloc[i]
            if val in level_to_idx:
                idx = level_to_idx[val]
                encoded[i] = contrast_matrix[idx, j]
            else:
                encoded[i] = np.nan

        columns.append(encoded)
        names.append(f"{name}.{j + 1}")

    return columns, names


ContrastsSpec = dict[str, str | ContrastType | NDArray[np.floating]]


def apply_contrasts_array(
    col_values: NDArray,
    name: str,
    contrast_matrix: NDArray[np.floating],
    categories: list,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    """Apply contrast matrix to a categorical column represented as a numpy array.

    Parameters
    ----------
    col_values : NDArray
        Values from the categorical column.
    name : str
        Variable name for creating column names.
    contrast_matrix : NDArray
        Contrast matrix of shape (n_levels, n_cols).
    categories : list
        List of category levels in order.

    Returns
    -------
    tuple
        (list of encoded columns, list of column names)
    """
    n = len(col_values)
    n_contrasts = contrast_matrix.shape[1]

    level_to_idx = {cat: i for i, cat in enumerate(categories)}

    columns: list[NDArray[np.floating]] = []
    names: list[str] = []

    for j in range(n_contrasts):
        encoded = np.zeros(n, dtype=np.float64)
        for i in range(n):
            val = col_values[i]
            if val in level_to_idx:
                idx = level_to_idx[val]
                encoded[i] = contrast_matrix[idx, j]
            else:
                encoded[i] = np.nan

        columns.append(encoded)
        names.append(f"{name}.{j + 1}")

    return columns, names

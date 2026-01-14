from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.utils.dataframe import (
    _is_polars,
    dataframe_length,
    get_columns,
)

if TYPE_CHECKING:
    from mixedlm.formula.terms import Formula


class NAAction(Enum):
    OMIT = "omit"
    EXCLUDE = "exclude"
    FAIL = "fail"

    @classmethod
    def from_string(cls, value: str | None) -> NAAction:
        if value is None:
            return cls.OMIT
        value_lower = value.lower().replace("na.", "").replace("na_", "")
        if value_lower in ("omit", "na.omit", "na_omit"):
            return cls.OMIT
        elif value_lower in ("exclude", "na.exclude", "na_exclude"):
            return cls.EXCLUDE
        elif value_lower in ("fail", "na.fail", "na_fail"):
            return cls.FAIL
        else:
            raise ValueError(f"Unknown na_action: '{value}'. Use 'omit', 'exclude', or 'fail'.")


@dataclass
class NAInfo:
    omitted_indices: NDArray[np.intp]
    n_original: int
    action: NAAction

    @property
    def n_omitted(self) -> int:
        return len(self.omitted_indices)

    @property
    def n_complete(self) -> int:
        return self.n_original - self.n_omitted

    def expand_to_original(
        self, values: NDArray[np.floating], fill_value: float = np.nan
    ) -> NDArray[np.floating]:
        if self.n_omitted == 0:
            return values

        result = np.full(self.n_original, fill_value, dtype=np.float64)
        mask = np.ones(self.n_original, dtype=bool)
        mask[self.omitted_indices] = False
        result[mask] = values
        return result


def get_model_variables(formula: Formula) -> list[str]:
    from mixedlm.formula.terms import InteractionTerm, VariableTerm

    variables = [formula.response]

    for term in formula.fixed.terms:
        if isinstance(term, VariableTerm):
            variables.append(term.name)
        elif isinstance(term, InteractionTerm):
            variables.extend(term.variables)

    for rterm in formula.random:
        if isinstance(rterm.grouping, str):
            variables.append(rterm.grouping)
        else:
            variables.extend(rterm.grouping_factors)

        for term in rterm.expr:
            if isinstance(term, VariableTerm):
                variables.append(term.name)
            elif isinstance(term, InteractionTerm):
                variables.extend(term.variables)

    return list(dict.fromkeys(variables))


def _get_na_mask_polars(data: Any, variables: list[str]) -> NDArray[np.bool_]:
    """Get NA mask for polars DataFrame."""
    import polars as pl

    available_cols = get_columns(data)
    available_vars = [v for v in variables if v in available_cols]

    if not available_vars:
        return np.zeros(dataframe_length(data), dtype=bool)

    null_expr = pl.any_horizontal(*[pl.col(v).is_null() for v in available_vars])
    mask_series = data.select(null_expr.alias("_na_mask")).get_column("_na_mask")
    return mask_series.to_numpy()


def _get_na_mask_pandas(data: Any, variables: list[str]) -> NDArray[np.bool_]:
    """Get NA mask for pandas DataFrame."""
    available_cols = list(data.columns)
    available_vars = [v for v in variables if v in available_cols]

    if not available_vars:
        return np.zeros(len(data), dtype=bool)

    subset = data[available_vars]
    return subset.isna().any(axis=1).values


def _filter_rows_polars(data: Any, keep_mask: NDArray[np.bool_]) -> Any:
    """Filter rows in polars DataFrame."""
    import polars as pl

    return data.filter(pl.Series(keep_mask))


def _filter_rows_pandas(data: Any, keep_mask: NDArray[np.bool_]) -> Any:
    """Filter rows in pandas DataFrame."""
    return data[keep_mask].reset_index(drop=True)


def _check_column_has_na_polars(data: Any, var: str) -> bool:
    """Check if a column has NA values in polars."""
    return data.get_column(var).null_count() > 0


def _check_column_has_na_pandas(data: Any, var: str) -> bool:
    """Check if a column has NA values in pandas."""
    return data[var].isna().any()


def handle_na(
    data: Any,
    formula: Formula,
    na_action: NAAction | str | None = None,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
) -> tuple[Any, NAInfo, NDArray[np.floating] | None, NDArray[np.floating] | None]:
    if isinstance(na_action, str) or na_action is None:
        na_action = NAAction.from_string(na_action)

    variables = get_model_variables(formula)
    available_cols = get_columns(data)
    available_vars = [v for v in variables if v in available_cols]

    is_polars = _is_polars(data)

    if is_polars:
        na_mask = _get_na_mask_polars(data, available_vars)
    else:
        na_mask = _get_na_mask_pandas(data, available_vars)

    if weights is not None:
        na_mask = na_mask | np.isnan(weights)
    if offset is not None:
        na_mask = na_mask | np.isnan(offset)

    omitted_indices = np.where(na_mask)[0]
    n_original = dataframe_length(data)

    if len(omitted_indices) > 0:
        if na_action == NAAction.FAIL:
            na_vars = []
            for var in available_vars:
                if is_polars:
                    has_na = _check_column_has_na_polars(data, var)
                else:
                    has_na = _check_column_has_na_pandas(data, var)
                if has_na:
                    na_vars.append(var)
            raise ValueError(
                f"Missing values in data. Variables with NA: {na_vars}. "
                "Use na_action='omit' or 'exclude' to handle missing values."
            )

        keep_mask = ~na_mask
        if is_polars:
            clean_data = _filter_rows_polars(data, keep_mask)
        else:
            clean_data = _filter_rows_pandas(data, keep_mask)

        if weights is not None:
            weights = weights[keep_mask]
        if offset is not None:
            offset = offset[keep_mask]
    else:
        clean_data = data

    na_info = NAInfo(
        omitted_indices=omitted_indices,
        n_original=n_original,
        action=na_action,
    )

    return clean_data, na_info, weights, offset

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

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


def handle_na(
    data: pd.DataFrame,
    formula: Formula,
    na_action: NAAction | str | None = None,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
) -> tuple[pd.DataFrame, NAInfo, NDArray[np.floating] | None, NDArray[np.floating] | None]:
    if isinstance(na_action, str) or na_action is None:
        na_action = NAAction.from_string(na_action)

    variables = get_model_variables(formula)
    available_vars = [v for v in variables if v in data.columns]

    subset = data[available_vars]
    na_mask = subset.isna().any(axis=1)

    if weights is not None:
        na_mask = na_mask | np.isnan(weights)
    if offset is not None:
        na_mask = na_mask | np.isnan(offset)

    omitted_indices = np.where(na_mask)[0]
    n_original = len(data)

    if len(omitted_indices) > 0:
        if na_action == NAAction.FAIL:
            na_vars = []
            for var in available_vars:
                if data[var].isna().any():
                    na_vars.append(var)
            raise ValueError(
                f"Missing values in data. Variables with NA: {na_vars}. "
                "Use na_action='omit' or 'exclude' to handle missing values."
            )

        clean_data = data[~na_mask].reset_index(drop=True)

        if weights is not None:
            weights = weights[~na_mask]
        if offset is not None:
            offset = offset[~na_mask]
    else:
        clean_data = data

    na_info = NAInfo(
        omitted_indices=omitted_indices,
        n_original=n_original,
        action=na_action,
    )

    return clean_data, na_info, weights, offset

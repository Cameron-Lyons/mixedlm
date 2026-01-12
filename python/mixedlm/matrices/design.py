from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.formula.terms import (
    Formula,
    InteractionTerm,
    InterceptTerm,
    RandomTerm,
    VariableTerm,
)


@dataclass
class RandomEffectStructure:
    grouping_factor: str
    term_names: list[str]
    n_levels: int
    n_terms: int
    correlated: bool
    level_map: dict[str, int]


@dataclass
class ModelMatrices:
    y: NDArray[np.floating]
    X: NDArray[np.floating]
    Z: sparse.csc_matrix
    fixed_names: list[str]
    random_structures: list[RandomEffectStructure]
    n_obs: int
    n_fixed: int
    n_random: int
    weights: NDArray[np.floating]
    offset: NDArray[np.floating]

    @cached_property
    def Zt(self) -> sparse.csc_matrix:
        return self.Z.T.tocsc()


def build_model_matrices(
    formula: Formula,
    data: pd.DataFrame,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
) -> ModelMatrices:
    y = _build_response(formula, data)
    X, fixed_names = build_fixed_matrix(formula, data)
    Z, random_structures = build_random_matrix(formula, data)

    n = len(y)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    if offset is None:
        offset = np.zeros(n, dtype=np.float64)

    return ModelMatrices(
        y=y,
        X=X,
        Z=Z,
        fixed_names=fixed_names,
        random_structures=random_structures,
        n_obs=n,
        n_fixed=X.shape[1],
        n_random=Z.shape[1],
        weights=weights,
        offset=offset,
    )


def _build_response(formula: Formula, data: pd.DataFrame) -> NDArray[np.floating]:
    return data[formula.response].to_numpy(dtype=np.float64)


def build_fixed_matrix(
    formula: Formula, data: pd.DataFrame
) -> tuple[NDArray[np.floating], list[str]]:
    n = len(data)
    columns: list[NDArray[np.floating]] = []
    names: list[str] = []

    if formula.fixed.has_intercept:
        columns.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")

    for term in formula.fixed.terms:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            col, col_names = _encode_variable(term.name, data)
            columns.extend(col)
            names.extend(col_names)
        elif isinstance(term, InteractionTerm):
            col, col_names = _encode_interaction(term.variables, data)
            columns.extend(col)
            names.extend(col_names)

    if not columns:
        columns.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")

    X = np.column_stack(columns)
    return X, names


def _encode_variable(name: str, data: pd.DataFrame) -> tuple[list[NDArray[np.floating]], list[str]]:
    col = data[name]

    if col.dtype == object or col.dtype.name == "category":
        return _encode_categorical(name, col)
    else:
        return [col.to_numpy(dtype=np.float64)], [name]


def _encode_categorical(
    name: str,
    col: pd.Series,  # type: ignore[type-arg]
) -> tuple[list[NDArray[np.floating]], list[str]]:
    if col.dtype.name == "category":
        categories = col.cat.categories.tolist()
    else:
        categories = sorted(col.dropna().unique().tolist())

    if len(categories) < 2:
        return [np.ones(len(col), dtype=np.float64)], [f"{name}"]

    columns: list[NDArray[np.floating]] = []
    names: list[str] = []

    for cat in categories[1:]:
        dummy = (col == cat).astype(np.float64)
        columns.append(dummy)
        names.append(f"{name}{cat}")

    return columns, names


def _encode_interaction(
    variables: tuple[str, ...], data: pd.DataFrame
) -> tuple[list[NDArray[np.floating]], list[str]]:
    encoded_vars: list[tuple[list[NDArray[np.floating]], list[str]]] = []
    for var in variables:
        cols, nms = _encode_variable(var, data)
        encoded_vars.append((cols, nms))

    result_cols: list[NDArray[np.floating]] = []
    result_names: list[str] = []

    def _product(
        idx: int,
        current_col: NDArray[np.floating],
        current_name: str,
    ) -> None:
        if idx >= len(encoded_vars):
            result_cols.append(current_col)
            result_names.append(current_name)
            return

        cols, nms = encoded_vars[idx]
        for col, nm in zip(cols, nms, strict=False):
            new_col = current_col * col
            new_name = f"{current_name}:{nm}" if current_name else nm
            _product(idx + 1, new_col, new_name)

    _product(0, np.ones(len(data), dtype=np.float64), "")
    return result_cols, result_names


def build_random_matrix(
    formula: Formula, data: pd.DataFrame
) -> tuple[sparse.csc_matrix, list[RandomEffectStructure]]:
    n = len(data)
    Z_blocks: list[sparse.csc_matrix] = []
    structures: list[RandomEffectStructure] = []

    for rterm in formula.random:
        Z_block, structure = _build_random_block(rterm, data, n)
        Z_blocks.append(Z_block)
        structures.append(structure)

    if not Z_blocks:
        return sparse.csc_matrix((n, 0), dtype=np.float64), []

    Z = sparse.hstack(Z_blocks, format="csc")
    return Z, structures


def _build_random_block(
    rterm: RandomTerm, data: pd.DataFrame, n: int
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    if rterm.is_nested:
        return _build_nested_random_block(rterm, data, n)

    grouping_factor = rterm.grouping
    assert isinstance(grouping_factor, str)

    group_col = data[grouping_factor]
    levels = sorted(group_col.dropna().unique().tolist())
    level_map = {lv: i for i, lv in enumerate(levels)}
    n_levels = len(levels)

    term_cols: list[NDArray[np.floating]] = []
    term_names: list[str] = []

    if rterm.has_intercept:
        term_cols.append(np.ones(n, dtype=np.float64))
        term_names.append("(Intercept)")

    for term in rterm.expr:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            cols, nms = _encode_variable(term.name, data)
            term_cols.extend(cols)
            term_names.extend(nms)
        elif isinstance(term, InteractionTerm):
            cols, nms = _encode_interaction(term.variables, data)
            term_cols.extend(cols)
            term_names.extend(nms)

    n_terms = len(term_cols)
    n_random_cols = n_levels * n_terms

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    for i in range(n):
        group_val = group_col.iloc[i]
        if group_val not in level_map:
            continue
        level_idx = level_map[group_val]

        for j, term_col in enumerate(term_cols):
            col_idx = level_idx * n_terms + j
            val = term_col[i]
            if val != 0:
                row_indices.append(i)
                col_indices.append(col_idx)
                values.append(val)

    Z_block = sparse.csc_matrix(
        (values, (row_indices, col_indices)),
        shape=(n, n_random_cols),
        dtype=np.float64,
    )

    structure = RandomEffectStructure(
        grouping_factor=grouping_factor,
        term_names=term_names,
        n_levels=n_levels,
        n_terms=n_terms,
        correlated=rterm.correlated,
        level_map=level_map,
    )

    return Z_block, structure


def _build_nested_random_block(
    rterm: RandomTerm, data: pd.DataFrame, n: int
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    grouping_factors = rterm.grouping_factors
    combined_group = data[list(grouping_factors)].apply(
        lambda row: "/".join(str(x) for x in row), axis=1
    )

    levels = sorted(combined_group.dropna().unique().tolist())
    level_map = {lv: i for i, lv in enumerate(levels)}
    n_levels = len(levels)

    term_cols: list[NDArray[np.floating]] = []
    term_names: list[str] = []

    if rterm.has_intercept:
        term_cols.append(np.ones(n, dtype=np.float64))
        term_names.append("(Intercept)")

    for term in rterm.expr:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            cols, nms = _encode_variable(term.name, data)
            term_cols.extend(cols)
            term_names.extend(nms)
        elif isinstance(term, InteractionTerm):
            cols, nms = _encode_interaction(term.variables, data)
            term_cols.extend(cols)
            term_names.extend(nms)

    n_terms = len(term_cols)
    n_random_cols = n_levels * n_terms

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    for i in range(n):
        group_val = combined_group.iloc[i]
        if group_val not in level_map:
            continue
        level_idx = level_map[group_val]

        for j, term_col in enumerate(term_cols):
            col_idx = level_idx * n_terms + j
            val = term_col[i]
            if val != 0:
                row_indices.append(i)
                col_indices.append(col_idx)
                values.append(val)

    Z_block = sparse.csc_matrix(
        (values, (row_indices, col_indices)),
        shape=(n, n_random_cols),
        dtype=np.float64,
    )

    structure = RandomEffectStructure(
        grouping_factor="/".join(grouping_factors),
        term_names=term_names,
        n_levels=n_levels,
        n_terms=n_terms,
        correlated=rterm.correlated,
        level_map=level_map,
    )

    return Z_block, structure

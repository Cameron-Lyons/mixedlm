from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from mixedlm.utils.dataframe import (
    concat_columns_as_string,
    dataframe_length,
    ensure_dataframe,
    get_categories,
    get_column_numpy,
    get_columns,
    get_unique_sorted,
    is_categorical_or_string,
    select_columns,
)

if TYPE_CHECKING:
    from mixedlm.utils.na_action import NAInfo

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
    cov_type: str = "us"


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
    frame: Any | None = field(default=None)
    na_info: NAInfo | None = field(default=None)

    @cached_property
    def Zt(self) -> sparse.csc_matrix:
        return self.Z.T.tocsc()


def _get_formula_variables(formula: Formula) -> set[str]:
    return formula.all_variables


def build_model_matrices(
    formula: Formula,
    data: Any,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    na_action: str | None = None,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> ModelMatrices:
    from mixedlm.utils.na_action import handle_na

    data = ensure_dataframe(data)

    if na_action is not None:
        clean_data, na_info, weights, offset = handle_na(data, formula, na_action, weights, offset)
    else:
        clean_data = data
        na_info = None

    y = _build_response(formula, clean_data)
    X, fixed_names = build_fixed_matrix(formula, clean_data, contrasts=contrasts)
    Z, random_structures = build_random_matrix(formula, clean_data, contrasts=contrasts)

    frame_vars = _get_formula_variables(formula)
    available_cols = get_columns(clean_data)
    available_vars = [v for v in frame_vars if v in available_cols]
    model_frame = select_columns(clean_data, available_vars)

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
        frame=model_frame,
        na_info=na_info,
    )


def _build_response(formula: Formula, data: Any) -> NDArray[np.floating]:
    arr = get_column_numpy(data, formula.response, dtype=np.float64)
    return arr


def build_fixed_matrix(
    formula: Formula,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[NDArray[np.floating], list[str]]:
    n = dataframe_length(data)
    columns: list[NDArray[np.floating]] = []
    names: list[str] = []

    if formula.fixed.has_intercept:
        columns.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")

    for term in formula.fixed.terms:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            col, col_names = _encode_variable(term.name, data, contrasts)
            columns.extend(col)
            names.extend(col_names)
        elif isinstance(term, InteractionTerm):
            col, col_names = _encode_interaction(term.variables, data, contrasts)
            columns.extend(col)
            names.extend(col_names)

    if not columns:
        columns.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")

    X = np.column_stack(columns)
    return X, names


def _encode_variable(
    name: str,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    if is_categorical_or_string(data, name):
        return _encode_categorical(name, data, contrasts)
    else:
        arr = get_column_numpy(data, name, dtype=np.float64)
        return [arr], [name]


def _encode_categorical(
    name: str,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    from mixedlm.utils.contrasts import apply_contrasts_array, get_contrast_matrix

    categories = get_categories(data, name)
    n_levels = len(categories)
    n = dataframe_length(data)

    if n_levels < 2:
        return [np.ones(n, dtype=np.float64)], [f"{name}"]

    contrast_spec = None
    if contrasts is not None and name in contrasts:
        contrast_spec = contrasts[name]

    contrast_matrix = get_contrast_matrix(n_levels, contrast_spec)

    col_values = get_column_numpy(data, name)  # Keep original dtype for categorical
    return apply_contrasts_array(col_values, name, contrast_matrix, categories)


def _encode_interaction(
    variables: tuple[str, ...],
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    encoded_vars: list[tuple[list[NDArray[np.floating]], list[str]]] = []
    for var in variables:
        cols, nms = _encode_variable(var, data, contrasts)
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

    n = dataframe_length(data)
    _product(0, np.ones(n, dtype=np.float64), "")
    return result_cols, result_names


def build_random_matrix(
    formula: Formula,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[sparse.csc_matrix, list[RandomEffectStructure]]:
    n = dataframe_length(data)
    Z_blocks: list[sparse.csc_matrix] = []
    structures: list[RandomEffectStructure] = []

    for rterm in formula.random:
        Z_block, structure = _build_random_block(rterm, data, n, contrasts)
        Z_blocks.append(Z_block)
        structures.append(structure)

    if not Z_blocks:
        return sparse.csc_matrix((n, 0), dtype=np.float64), []

    Z = sparse.hstack(Z_blocks, format="csc")
    return Z, structures


def _build_random_block(
    rterm: RandomTerm,
    data: Any,
    n: int,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    if rterm.is_nested:
        return _build_nested_random_block(rterm, data, n, contrasts)

    grouping_factor = rterm.grouping
    assert isinstance(grouping_factor, str)

    levels = get_unique_sorted(data, grouping_factor)
    level_map = {lv: i for i, lv in enumerate(levels)}
    n_levels = len(levels)

    group_values = get_column_numpy(data, grouping_factor)  # Keep original dtype for grouping

    term_cols: list[NDArray[np.floating]] = []
    term_names: list[str] = []

    if rterm.has_intercept:
        term_cols.append(np.ones(n, dtype=np.float64))
        term_names.append("(Intercept)")

    for term in rterm.expr:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            cols, nms = _encode_variable(term.name, data, contrasts)
            term_cols.extend(cols)
            term_names.extend(nms)
        elif isinstance(term, InteractionTerm):
            cols, nms = _encode_interaction(term.variables, data, contrasts)
            term_cols.extend(cols)
            term_names.extend(nms)

    n_terms = len(term_cols)
    n_random_cols = n_levels * n_terms

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    for i in range(n):
        group_val = group_values[i]
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
        cov_type=rterm.cov_type,
    )

    return Z_block, structure


def _build_nested_random_block(
    rterm: RandomTerm,
    data: Any,
    n: int,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    grouping_factors = rterm.grouping_factors
    combined_group = concat_columns_as_string(data, list(grouping_factors), separator="/")

    if np.issubdtype(combined_group.dtype, np.floating):
        valid_mask = ~np.isnan(combined_group.astype(float, casting="safe"))
    else:
        valid_mask = np.ones(len(combined_group), dtype=bool)
    unique_combined = np.unique(combined_group[valid_mask])
    levels = sorted([str(x) for x in unique_combined if x is not None and str(x) != "nan"])
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
            cols, nms = _encode_variable(term.name, data, contrasts)
            term_cols.extend(cols)
            term_names.extend(nms)
        elif isinstance(term, InteractionTerm):
            cols, nms = _encode_interaction(term.variables, data, contrasts)
            term_cols.extend(cols)
            term_names.extend(nms)

    n_terms = len(term_cols)
    n_random_cols = n_levels * n_terms

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    for i in range(n):
        group_val = str(combined_group[i])
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
        cov_type=rterm.cov_type,
    )

    return Z_block, structure

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
    factorize_levels,
    get_categories,
    get_column_numpy,
    get_columns,
    is_categorical_or_string,
    select_columns,
)

if TYPE_CHECKING:
    from mixedlm.utils.na_action import NAInfo

from mixedlm.formula.terms import (
    Formula,
    InteractionTerm,
    InterceptTerm,
    PowerTerm,
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
    level_indices: NDArray[np.int32] | None = field(default=None, repr=False, compare=False)


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
    category_levels: dict[str, list[Any]] = field(default_factory=dict)
    contrasts: dict[str, str | NDArray[np.floating]] | None = field(default=None)

    @cached_property
    def Zt(self) -> sparse.csc_matrix:
        return self.Z.T.tocsc()


def validate_prior_weights(
    weights: NDArray[np.floating],
    n: int,
    *,
    allow_missing: bool = False,
) -> NDArray[np.float64]:
    """Return validated, contiguous prior weights."""
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if len(weights_array) != n:
        raise ValueError(f"weights has length {len(weights_array)}, expected {n}")
    finite = ~np.isinf(weights_array) if allow_missing else np.isfinite(weights_array)
    if not np.all(finite):
        raise ValueError("weights must contain only finite values")
    if np.any(weights_array <= 0.0):
        raise ValueError("weights must be strictly positive")
    return np.ascontiguousarray(weights_array)


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

    data = ensure_dataframe(data, columns=sorted(_get_formula_variables(formula)))
    if weights is not None:
        weights = validate_prior_weights(
            weights,
            dataframe_length(data),
            allow_missing=True,
        )

    if na_action is not None:
        clean_data, na_info, weights, offset = handle_na(data, formula, na_action, weights, offset)
    else:
        clean_data = data
        na_info = None

    frame_vars = _get_formula_variables(formula)
    available_cols = get_columns(clean_data)
    encoded_variables = formula.fixed_variables | formula.random_variables
    category_levels = {
        name: get_categories(clean_data, name)
        for name in encoded_variables
        if name in available_cols and is_categorical_or_string(clean_data, name)
    }
    stored_contrasts = (
        None
        if contrasts is None
        else {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in contrasts.items()
        }
    )

    y = _build_response(formula, clean_data)
    X, fixed_names = build_fixed_matrix(
        formula,
        clean_data,
        contrasts=stored_contrasts,
        category_levels=category_levels,
    )
    Z, random_structures = build_random_matrix(
        formula,
        clean_data,
        contrasts=stored_contrasts,
        category_levels=category_levels,
    )

    available_vars = [v for v in frame_vars if v in available_cols]
    model_frame = select_columns(clean_data, available_vars)

    n = len(y)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = validate_prior_weights(weights, n)
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
        category_levels=category_levels,
        contrasts=stored_contrasts,
    )


def _build_response(formula: Formula, data: Any) -> NDArray[np.floating]:
    arr = get_column_numpy(data, formula.response, dtype=np.float64)
    return arr


def build_fixed_matrix(
    formula: Formula,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
) -> tuple[NDArray[np.floating], list[str]]:
    n = dataframe_length(data)
    columns: list[NDArray[np.floating]] = []
    names: list[str] = []

    if formula.fixed.has_intercept:
        columns.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")

    term_columns, term_names = _encode_terms(
        formula.fixed.terms,
        data,
        has_intercept=formula.fixed.has_intercept,
        contrasts=contrasts,
        category_levels=category_levels,
    )
    columns.extend(term_columns)
    names.extend(term_names)

    if not columns:
        columns.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")

    X = np.column_stack(columns)
    return X, names


def _encode_variable(
    name: str,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    *,
    full_rank: bool = False,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    if name in (category_levels or {}) or is_categorical_or_string(data, name):
        return _encode_categorical(
            name,
            data,
            contrasts,
            category_levels,
            full_rank=full_rank,
        )
    else:
        arr = get_column_numpy(data, name, dtype=np.float64)
        return [arr], [name]


def _encode_power(
    term: PowerTerm,
    data: Any,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    arr = get_column_numpy(data, term.name, dtype=np.float64)
    return [np.power(arr, term.exponent)], [f"I({term.name}**{term.exponent})"]


def _encode_categorical(
    name: str,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    *,
    full_rank: bool = False,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    from mixedlm.utils.contrasts import apply_contrasts_array, get_contrast_matrix

    categories = (category_levels or {}).get(name)
    if categories is None:
        categories = get_categories(data, name)
    n_levels = len(categories)
    n = dataframe_length(data)

    if n_levels < 2:
        return [np.ones(n, dtype=np.float64)], [f"{name}"]

    contrast_matrix: NDArray[np.floating]
    if full_rank:
        contrast_matrix = np.eye(n_levels, dtype=np.float64)
    else:
        contrast_spec = None
        if contrasts is not None and name in contrasts:
            contrast_spec = contrasts[name]
        contrast_matrix = get_contrast_matrix(n_levels, contrast_spec)

    col_values = get_column_numpy(data, name)
    columns, names = apply_contrasts_array(col_values, name, contrast_matrix, categories)
    if full_rank:
        names = [f"{name}.{category}" for category in categories]
    return columns, names


def _encode_terms(
    terms: tuple[InterceptTerm | VariableTerm | PowerTerm | InteractionTerm, ...],
    data: Any,
    has_intercept: bool,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    """Encode model terms, using full indicators for the first factor without an intercept."""
    columns: list[NDArray[np.floating]] = []
    names: list[str] = []
    use_full_rank_factor = not has_intercept

    for term in terms:
        if isinstance(term, InterceptTerm):
            continue
        if isinstance(term, VariableTerm):
            is_categorical = term.name in (category_levels or {}) or is_categorical_or_string(
                data, term.name
            )
            full_rank = use_full_rank_factor and is_categorical
            term_columns, term_names = _encode_variable(
                term.name,
                data,
                contrasts,
                category_levels,
                full_rank=full_rank,
            )
            if full_rank:
                use_full_rank_factor = False
        elif isinstance(term, PowerTerm):
            term_columns, term_names = _encode_power(term, data)
        else:
            term_columns, term_names = _encode_interaction(
                term.variables,
                data,
                contrasts,
                category_levels,
            )

        columns.extend(term_columns)
        names.extend(term_names)

    return columns, names


def _encode_interaction(
    variables: tuple[str | PowerTerm, ...],
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    encoded_vars: list[tuple[list[NDArray[np.floating]], list[str]]] = []
    for var in variables:
        if isinstance(var, PowerTerm):
            cols, nms = _encode_power(var, data)
        else:
            cols, nms = _encode_variable(var, data, contrasts, category_levels)
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


def _build_sparse_Z_block(
    level_indices: NDArray[np.int64],
    term_cols: list[NDArray[np.floating]],
    n: int,
    n_levels: int,
    n_terms: int,
) -> sparse.csc_matrix:
    """Build sparse Z block using vectorized operations."""
    n_random_cols = n_levels * n_terms

    valid_mask = level_indices >= 0

    all_rows: list[NDArray[np.int64]] = []
    all_cols: list[NDArray[np.int64]] = []
    all_vals: list[NDArray[np.floating]] = []

    for j, term_col in enumerate(term_cols):
        nonzero_mask = (term_col != 0) & valid_mask
        row_idx = np.where(nonzero_mask)[0]
        if len(row_idx) > 0:
            col_idx = level_indices[row_idx] * n_terms + j
            all_rows.append(row_idx.astype(np.int64))
            all_cols.append(col_idx.astype(np.int64))
            all_vals.append(term_col[row_idx])

    if all_rows:
        row_indices = np.concatenate(all_rows)
        col_indices = np.concatenate(all_cols)
        values = np.concatenate(all_vals)
    else:
        row_indices = np.array([], dtype=np.int64)
        col_indices = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)

    return sparse.csc_matrix(
        (values, (row_indices, col_indices)),
        shape=(n, n_random_cols),
        dtype=np.float64,
    )


def build_random_matrix(
    formula: Formula,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
) -> tuple[sparse.csc_matrix, list[RandomEffectStructure]]:
    n = dataframe_length(data)
    Z_blocks: list[sparse.csc_matrix] = []
    structures: list[RandomEffectStructure] = []

    for rterm in formula.random:
        Z_block, structure = _build_random_block(
            rterm,
            data,
            n,
            contrasts,
            category_levels,
        )
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
    category_levels: dict[str, list[Any]] | None = None,
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    if rterm.is_nested:
        return _build_nested_random_block(rterm, data, n, contrasts, category_levels)

    grouping_factor = rterm.grouping
    assert isinstance(grouping_factor, str)

    group_values = get_column_numpy(data, grouping_factor)
    level_indices, level_map = factorize_levels(group_values)
    n_levels = len(level_map)

    term_cols: list[NDArray[np.floating]] = (
        [np.ones(n, dtype=np.float64)] if rterm.has_intercept else []
    )
    term_names = ["(Intercept)"] if rterm.has_intercept else []
    encoded_cols, encoded_names = _encode_terms(
        rterm.expr,
        data,
        has_intercept=rterm.has_intercept,
        contrasts=contrasts,
        category_levels=category_levels,
    )
    term_cols.extend(encoded_cols)
    term_names.extend(encoded_names)

    n_terms = len(term_cols)

    Z_block = _build_sparse_Z_block(level_indices, term_cols, n, n_levels, n_terms)

    structure = RandomEffectStructure(
        grouping_factor=grouping_factor,
        term_names=term_names,
        n_levels=n_levels,
        n_terms=n_terms,
        correlated=rterm.correlated,
        level_map=level_map,
        cov_type=rterm.cov_type,
        level_indices=level_indices,
    )

    return Z_block, structure


def _build_nested_random_block(
    rterm: RandomTerm,
    data: Any,
    n: int,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    grouping_factors = rterm.grouping_factors
    combined_group = concat_columns_as_string(data, list(grouping_factors), separator="/")

    level_indices, level_map = factorize_levels(combined_group)
    n_levels = len(level_map)

    term_cols: list[NDArray[np.floating]] = (
        [np.ones(n, dtype=np.float64)] if rterm.has_intercept else []
    )
    term_names = ["(Intercept)"] if rterm.has_intercept else []
    encoded_cols, encoded_names = _encode_terms(
        rterm.expr,
        data,
        has_intercept=rterm.has_intercept,
        contrasts=contrasts,
        category_levels=category_levels,
    )
    term_cols.extend(encoded_cols)
    term_names.extend(encoded_names)

    n_terms = len(term_cols)

    Z_block = _build_sparse_Z_block(level_indices, term_cols, n, n_levels, n_terms)

    structure = RandomEffectStructure(
        grouping_factor="/".join(grouping_factors),
        term_names=term_names,
        n_levels=n_levels,
        n_terms=n_terms,
        correlated=rterm.correlated,
        level_map=level_map,
        cov_type=rterm.cov_type,
        level_indices=level_indices,
    )

    return Z_block, structure

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from numbers import Real
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
    get_observed_categories,
    is_categorical_or_string,
    select_columns,
)

if TYPE_CHECKING:
    from mixedlm.utils.na_action import NAInfo

from mixedlm.formula.terms import (
    Factor,
    Formula,
    InteractionTerm,
    InterceptTerm,
    ParsedTerm,
    PowerTerm,
    RandomTerm,
    TransformTerm,
    VariableTerm,
    _format_grouping,
    format_term,
    term_factors,
)
from mixedlm.formula.transforms import (
    STORED_PREFIX,
    TransformState,
    encode_numeric_transform,
    factor_values,
    formula_offset,
    learn_transform_state,
    response_values,
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
    # Columns of each formula term, in column order; None for fits saved before it was kept.
    term_sizes: tuple[int, ...] | None = field(default=None, repr=False, compare=False)


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
    trials: NDArray[np.floating] | None = field(default=None)
    category_levels: dict[str, list[Any]] = field(default_factory=dict)
    contrasts: dict[str, str | NDArray[np.floating]] | None = field(default=None)
    # Retain column identity when rank checks remove formula columns whose
    # display names may coincide with other variables or encoded factors.
    fixed_source_names: tuple[str, ...] | None = field(default=None, repr=False)
    fixed_column_indices: tuple[int, ...] | None = field(default=None, repr=False)
    response_levels: tuple[Any, Any] | None = field(default=None)
    fixed_term_labels: tuple[str, ...] | None = field(default=None, repr=False)

    # Numeric binomial proportions use the supplied weights as trial counts.
    # Retain that interpretation when a refit happens to draw only 0s and 1s.
    binomial_trial_weights: bool = False
    # Parameters that stateful formula transforms such as scale(x), poly(x, 2)
    # and factor(x) learned from the fitting data, keyed by term label.
    transform_state: dict[str, Any] | None = field(default=None, repr=False)
    # How categorical factors were coded: "R" follows model.matrix(), with
    # indicators in interactions that lack a margin and the first factor of an
    # interaction varying fastest. Fits saved before then keep "legacy" coding.
    factor_coding: str = "legacy"
    # The offset supplied by the caller, excluding formula offset() terms.
    # None when the formula has no offset terms, so it equals ``offset``.
    supplied_offset: NDArray[np.floating] | None = field(default=None, repr=False)

    @property
    def response_trials(self) -> NDArray[np.floating] | None:
        return self.weights if self.binomial_trial_weights else self.trials

    @property
    def argument_offset(self) -> NDArray[np.floating]:
        """The offset to pass when refitting the formula, which adds its own terms."""
        return self.offset if self.supplied_offset is None else self.supplied_offset

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
    grouped_binomial: bool = False,
    transform_state: Mapping[str, Any] | None = None,
    factor_coding: str = "R",
    category_levels: Mapping[str, Sequence[Any]] | None = None,
) -> ModelMatrices:
    """Build the design matrices of ``formula`` for ``data``.

    ``transform_state`` reuses parameters that stateful transforms learned in an
    earlier fit, such as when a model is refitted to its own stored rows.
    ``factor_coding`` is "R", as in ``model.matrix()``, or "legacy" for refits of
    models saved by mixedlm 1.3 and earlier, which keep their own columns.
    ``category_levels`` fixes the levels of the named categorical variables, such
    as levels learned from a larger data set that ``data`` is part of.
    """
    if factor_coding not in ("R", "legacy"):
        raise ValueError(f"factor_coding must be 'R' or 'legacy', got {factor_coding!r}")
    from mixedlm.utils.na_action import handle_na

    data = ensure_dataframe(data, columns=sorted(_get_formula_variables(formula)))
    if weights is not None:
        weights = validate_prior_weights(
            weights,
            dataframe_length(data),
            allow_missing=True,
        )

    learned: dict[str, Any] = dict(transform_state or {})
    learn_transform_state(formula, data, learned)
    transform_state = learned
    if na_action is not None:
        clean_data, na_info, weights, offset = handle_na(data, formula, na_action, weights, offset)
    else:
        clean_data = data
        na_info = None

    frame_vars = _get_formula_variables(formula)
    available_cols = get_columns(clean_data)
    encoded_variables = formula.fixed_variables | formula.random_variables
    fixed_levels = category_levels or {}
    levels: dict[str, list[Any]] = {
        name: list(fixed_levels[name])
        if name in fixed_levels
        else get_observed_categories(clean_data, name)
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

    response_levels = (
        _binomial_factor_levels(clean_data, formula.response)
        if grouped_binomial and formula.plain_response
        else None
    )
    y, trials = _build_response(
        formula, clean_data, grouped_binomial=grouped_binomial, response_levels=response_levels
    )
    fixed_term_labels: list[str] = []
    X, fixed_names = build_fixed_matrix(
        formula,
        clean_data,
        contrasts=stored_contrasts,
        category_levels=levels,
        term_labels=fixed_term_labels,
        transform_state=transform_state,
        coding=factor_coding,
    )
    Z, random_structures = build_random_matrix(
        formula,
        clean_data,
        contrasts=stored_contrasts,
        category_levels=levels,
        transform_state=transform_state,
        coding=factor_coding,
    )

    available_vars = [v for v in frame_vars if v in available_cols]
    # Keep a refit's stored responses so that updates of the model reuse them.
    available_vars += [c for c in available_cols if c.startswith(STORED_PREFIX)]
    model_frame = select_columns(clean_data, available_vars)

    n = len(y)
    if trials is not None:
        weights = trials.copy() if weights is None else np.asarray(weights) * trials
    elif weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = validate_prior_weights(weights, n)
    if offset is None:
        offset = np.zeros(n, dtype=np.float64)
    supplied_offset = None
    term_offset = formula_offset(formula, clean_data)
    if term_offset is not None:
        supplied_offset = np.asarray(offset, dtype=np.float64)
        offset = supplied_offset + term_offset

    return ModelMatrices(
        y=y,
        X=X,
        Z=Z,
        fixed_names=fixed_names,
        fixed_term_labels=tuple(fixed_term_labels),
        random_structures=random_structures,
        n_obs=n,
        n_fixed=X.shape[1],
        n_random=Z.shape[1],
        weights=weights,
        offset=offset,
        frame=model_frame,
        na_info=na_info,
        trials=trials,
        binomial_trial_weights=grouped_binomial
        and trials is None
        and bool(np.any((y > 0) & (y < 1))),
        response_levels=response_levels,
        category_levels=levels,
        contrasts=stored_contrasts,
        transform_state=transform_state,
        supplied_offset=supplied_offset,
        factor_coding=factor_coding,
    )


def _build_response(
    formula: Formula,
    data: Any,
    *,
    grouped_binomial: bool,
    response_levels: tuple[Any, Any] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    denominator = formula.response_denominator
    if formula.response_failures is not None:
        if not grouped_binomial:
            raise ValueError("cbind(successes, failures) responses require a binomial GLMM")
        successes = _response_column(formula, data)
        failures_label = formula.failures_label
        assert failures_label is not None
        failures = response_values(formula.response_failures, failures_label, data)
        trials = successes + failures
        return _normalize_grouped_binomial_response(successes, trials), trials
    if formula.response_expr is not None and denominator is None:
        return _response_column(formula, data), None
    if denominator is None:
        if grouped_binomial:
            levels = response_levels or _binomial_factor_levels(data, formula.response)
            if levels is not None and _response_is_factor(data, formula.response):
                return _encode_binomial_factor(
                    get_column_numpy(data, formula.response), levels
                ), None
        try:
            successes = np.asarray(
                get_column_numpy(data, formula.response, dtype=np.float64), dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Non-numeric responses are only supported for binomial GLMMs") from exc
        return successes, None

    if not grouped_binomial:
        raise ValueError(
            "The 'successes / trials' response syntax is only supported for binomial GLMMs"
        )

    successes = get_column_numpy(data, formula.response, dtype=np.float64)
    trials = get_column_numpy(data, denominator, dtype=np.float64)
    return _normalize_grouped_binomial_response(successes, trials), trials


def _response_column(formula: Formula, data: Any) -> NDArray[np.float64]:
    if formula.response_expr is None:
        return np.asarray(
            get_column_numpy(data, formula.response, dtype=np.float64), dtype=np.float64
        )
    return response_values(formula.response_expr, formula.response, data)


def _numeric_response(values: Any) -> bool:
    """Recognize numeric responses without treating numeric-looking labels as numbers."""
    dtype_name = str(getattr(values, "dtype", ""))
    if dtype_name == "category" or "Categorical" in dtype_name or "Enum" in dtype_name:
        return False
    array = np.asarray(values)
    return array.dtype.kind in "biuf" or (
        array.dtype.kind == "O" and all(isinstance(value, Real | np.bool_) for value in array)
    )


def _response_is_factor(data: Any, name: str) -> bool:
    if not is_categorical_or_string(data, name):
        return False
    dtype_name = str(data[name].dtype)
    if dtype_name == "category" or "Categorical" in dtype_name or "Enum" in dtype_name:
        return True
    return not _numeric_response(get_column_numpy(data, name))


def _binomial_factor_levels(data: Any, name: str) -> tuple[Any, Any] | None:
    """Use declared factor order, or sorted labels for an untyped string response."""
    if not _response_is_factor(data, name):
        return None
    levels = get_categories(data, name)
    if type(data).__module__.startswith("polars") and "Categorical" in str(data[name].dtype):
        # Polars categoricals may share a process-wide pool with unrelated
        # columns. Only Enum categories declare unobserved response levels.
        observed = set(get_column_numpy(data, name))
        levels = [level for level in levels if level in observed]
    if len(levels) > 2:
        # As lme4's model frame does, drop declared levels that no row takes.
        levels = get_observed_categories(data, name)
    if len(levels) != 2:
        raise ValueError(
            f"Binomial factor responses require exactly two levels; '{name}' has {len(levels)}"
        )
    return levels[0], levels[1]


def _encode_binomial_factor(
    values: Any, levels: tuple[Any, Any], *, allow_missing: bool = False
) -> NDArray[np.float64]:
    import pandas as pd

    values = np.asarray(values)
    codes = pd.Categorical(values, categories=list(levels)).codes
    unknown = codes < 0
    if allow_missing:
        unknown &= ~pd.isna(values)
    if np.any(unknown):
        raise ValueError(f"Missing or unknown binomial response levels; expected {levels!r}")
    encoded = codes.astype(np.float64)
    encoded[codes < 0] = np.nan
    return encoded


def _restore_binomial_factor(data: Any, name: str, levels: tuple[Any, Any] | None) -> Any:
    """Retain a fitted response's two levels in subsets and formula-based refits."""
    if levels is None:
        return data
    data = ensure_dataframe(data)
    if name not in get_columns(data) or not _response_is_factor(data, name):
        return data
    values = get_column_numpy(data, name)
    _encode_binomial_factor(values, levels, allow_missing=True)
    if type(data).__module__.startswith("pandas"):
        import pandas as pd

        restored = data.copy()
        restored[name] = pd.Categorical(values, categories=list(levels))
        return restored
    import polars as pl

    return data.with_columns(pl.col(name).cast(pl.Enum(list(levels))))


def _normalize_grouped_binomial_response(
    successes: NDArray[np.floating],
    trials: NDArray[np.floating],
) -> NDArray[np.floating]:
    if not np.all(np.isfinite(successes)) or not np.all(np.isfinite(trials)):
        raise ValueError("Grouped binomial successes and trials must be finite")
    if np.any(trials <= 0):
        raise ValueError("Grouped binomial trials must be greater than zero")
    if np.any(successes < 0) or np.any(successes > trials):
        raise ValueError("Grouped binomial successes must be between zero and trials")
    if np.any(successes != np.floor(successes)) or np.any(trials != np.floor(trials)):
        raise ValueError("Grouped binomial successes and trials must be whole numbers")

    return successes / trials


def build_fixed_matrix(
    formula: Formula,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    *,
    term_labels: list[str] | None = None,
    transform_state: TransformState | None = None,
    coding: str = "R",
) -> tuple[NDArray[np.floating], list[str]]:
    """Encode fixed effects and optionally collect the source term for each column.

    ``transform_state`` holds the parameters of stateful formula transforms. Pass
    the fitted model's state to encode new data; missing entries are learned from
    ``data`` and recorded.
    """
    n = dataframe_length(data)
    factors = list(
        _term_factors(
            formula.fixed.terms,
            data,
            has_intercept=formula.fixed.has_intercept,
            contrasts=contrasts,
            category_levels=category_levels,
            transform_state=transform_state,
            coding=coding,
        )
    )
    first_fastest = coding == "R"
    names = ["(Intercept)"] if formula.fixed.has_intercept else []
    if term_labels is not None:
        term_labels.extend(names)
    for term, encoded in factors:
        start = len(names)
        names.extend(_factor_names(encoded, first_fastest))
        if term_labels is not None:
            term_labels.extend([format_term(term)] * (len(names) - start))

    if not names:
        return np.empty((n, 0), dtype=np.float64), names

    # Allocate the final matrix once; retain base encodings but stream interaction
    # products instead of keeping a second matrix's worth of product columns.
    X = np.empty((n, len(names)), dtype=np.float64)
    column_index = 0
    if formula.fixed.has_intercept:
        X[:, 0] = 1.0
        column_index = 1
    for _, encoded in factors:
        positions = _column_positions(encoded, first_fastest)
        for position, column in zip(positions, _factor_columns(encoded, n), strict=True):
            X[:, column_index + position] = column
        column_index += len(positions)
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


def _encode_factor_transform(
    term: TransformTerm,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None,
    transform_state: TransformState | None,
    *,
    full_rank: bool,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    """Encode factor(x), C(x, contrast) or relevel(x, ref) with fitted levels."""
    import pandas as pd

    from mixedlm.utils.contrasts import apply_contrasts_array, get_contrast_matrix

    label = term.label
    values, levels = factor_values(term, data, transform_state)
    observed = pd.Index(values).dropna().unique()
    unknown = observed[~observed.isin(levels)]
    if len(unknown):
        listed = ", ".join(repr(value) for value in unknown)
        raise ValueError(f"New level(s) {listed} in factor '{label}'.")
    n = dataframe_length(data)
    if len(levels) < 2:
        return [np.ones(n, dtype=np.float64)], [label]
    contrast_matrix: NDArray[np.floating]
    if full_rank:
        contrast_matrix = np.eye(len(levels), dtype=np.float64)
    else:
        spec = (contrasts or {}).get(label, term.option("contr") if term.func == "C" else None)
        if term.func == "C" and spec == "helmert" and label not in (contrasts or {}):
            # In formula syntax, C(x, helmert) is R's unscaled contr.helmert.
            spec = "contr.helmert"
        contrast_matrix = get_contrast_matrix(len(levels), spec)
    columns, names = apply_contrasts_array(values, label, contrast_matrix, levels)
    if full_rank:
        names = [f"{label}.{level}" for level in levels]
    return columns, names


_EncodedFactor = tuple[list[NDArray[np.floating]], list[str]]


def _term_factors(
    terms: tuple[ParsedTerm, ...],
    data: Any,
    has_intercept: bool,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    transform_state: TransformState | None = None,
    coding: str = "R",
) -> Iterator[
    tuple[VariableTerm | PowerTerm | TransformTerm | InteractionTerm, list[_EncodedFactor]]
]:
    """Encode each base factor once, keeping full and reduced rank encodings distinct."""
    cache: dict[tuple[Factor, bool], _EncodedFactor] = {}

    def encode(variable: Factor, full_rank: bool = False) -> _EncodedFactor:
        key = (variable, full_rank)
        if key not in cache:
            if isinstance(variable, PowerTerm):
                cache[key] = _encode_power(variable, data)
            elif isinstance(variable, TransformTerm) and variable.is_categorical:
                cache[key] = _encode_factor_transform(
                    variable, data, contrasts, transform_state, full_rank=full_rank
                )
            elif isinstance(variable, TransformTerm):
                columns, names = encode_numeric_transform(variable, data, transform_state)
                cache[key] = (list[NDArray[np.floating]](columns), names)
            else:
                cache[key] = _encode_variable(
                    variable, data, contrasts, category_levels, full_rank=full_rank
                )
        return cache[key]

    def categorical(variable: Factor) -> bool:
        if isinstance(variable, TransformTerm):
            return variable.is_categorical
        if isinstance(variable, PowerTerm):
            return False
        return variable in (category_levels or {}) or is_categorical_or_string(data, variable)

    codings = (
        _legacy_indicator_coding(terms, has_intercept, categorical)
        if coding == "legacy"
        else _indicator_coding(terms, has_intercept, categorical)
    )
    for term, full_rank in zip(terms, codings, strict=True):
        if isinstance(term, InterceptTerm):
            continue
        variables = term_factors(term)
        yield term, [encode(v, full) for v, full in zip(variables, full_rank, strict=True)]


def _legacy_indicator_coding(
    terms: tuple[ParsedTerm, ...],
    has_intercept: bool,
    categorical: Callable[[Factor], bool],
) -> list[tuple[bool, ...]]:
    """The coding of fits saved by version 1.3 and earlier, kept for their predictions.

    Interactions always used contrasts. Without an intercept, the first categorical
    main effect in formula order used indicators.
    """
    codings: list[tuple[bool, ...]] = []
    pending = not has_intercept
    for term in terms:
        factors = term_factors(term)
        main_effect = isinstance(term, VariableTerm | TransformTerm)
        full = pending and main_effect and categorical(factors[0])
        pending = pending and not full
        codings.append((full,) if main_effect else (False,) * len(factors))
    return codings


def _indicator_coding(
    terms: tuple[ParsedTerm, ...],
    has_intercept: bool,
    categorical: Callable[[Factor], bool],
) -> list[tuple[bool, ...]]:
    """Choose indicator (full-rank) or contrast coding for each factor, as R does.

    A categorical factor in a term is coded by contrasts when the term without it
    is contained in a term that precedes it in order of degree; otherwise every
    level gets an indicator. Without an intercept, the first factor of the first
    term that has one also uses indicators. See Chambers and Hastie (1992),
    Statistical Models in S, p. 38.
    """
    factor_sets = [frozenset(term_factors(term)) for term in terms]
    order = sorted(range(len(terms)), key=lambda index: (len(factor_sets[index]), index))
    first_factor: tuple[int, Factor] | None = None
    if not has_intercept:
        first_factor = next(
            (
                (index, variable)
                for index in order
                for variable in term_factors(terms[index])
                if categorical(variable)
            ),
            None,
        )
    codings: list[tuple[bool, ...]] = [()] * len(terms)
    for rank, index in enumerate(order):
        preceding = [factor_sets[other] for other in order[:rank]]
        coding = []
        for variable in term_factors(terms[index]):
            rest = factor_sets[index] - {variable}
            coding.append(
                categorical(variable)
                and (
                    (index, variable) == first_factor
                    or (bool(rest) and not any(rest <= other for other in preceding))
                )
            )
        codings[index] = tuple(coding)
    return codings


def _join_names(parts: Iterable[str]) -> str:
    name = ""
    for part in parts:
        name = f"{name}:{part}" if name else part
    return name


def _factor_names(factors: list[_EncodedFactor], first_fastest: bool = True) -> Iterator[str]:
    """Name an interaction's columns; R's model.matrix() varies the first factor fastest."""
    if first_fastest:
        for names in product(*(factor[1] for factor in factors[::-1])):
            yield _join_names(names[::-1])
    else:
        for names in product(*(factor[1] for factor in factors)):
            yield _join_names(names)


def _column_positions(factors: list[_EncodedFactor], first_fastest: bool) -> list[int]:
    """Positions, in name order, of the columns that :func:`_factor_columns` yields.

    Columns are formed with the last factor varying fastest, so products keep
    their left-to-right arithmetic; R's order varies the first factor fastest.
    """
    sizes = [len(factor[1]) for factor in factors]
    count = int(np.prod(sizes)) if sizes else 1
    if not first_fastest or len(sizes) < 2:
        return list(range(count))
    # Generation order is row-major over the factors; R's order is column-major.
    return list(np.ravel_multi_index(np.unravel_index(np.arange(count), sizes), sizes, order="F"))


def _factor_columns(factors: list[_EncodedFactor], n: int) -> Iterator[NDArray[np.floating]]:
    """Form an interaction's columns, varying the last factor fastest."""
    if not factors:
        yield np.ones(n, dtype=np.float64)
        return
    if len(factors) == 1:
        yield from factors[0][0]
        return

    # Depth-first traversal reuses common multiplication prefixes while retaining
    # only one partial product per factor, without a recursive closure.
    iterators = [iter(factors[0][0])]
    prefixes: list[NDArray[np.floating]] = []
    while iterators:
        try:
            column = next(iterators[-1])
        except StopIteration:
            iterators.pop()
            if prefixes:
                prefixes.pop()
            continue
        if prefixes:
            column = prefixes[-1] * column
        if len(iterators) == len(factors):
            yield column
        else:
            prefixes.append(column)
            iterators.append(iter(factors[len(iterators)][0]))


def _encode_terms(
    terms: tuple[ParsedTerm, ...],
    data: Any,
    has_intercept: bool,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    transform_state: TransformState | None = None,
    coding: str = "R",
    sizes: list[int] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    """Materialize columns for sparse random-effect construction.

    ``sizes`` collects each term's number of columns.
    """
    columns: list[NDArray[np.floating]] = []
    names: list[str] = []
    n = dataframe_length(data)
    first_fastest = coding == "R"
    for _, factors in _term_factors(
        terms, data, has_intercept, contrasts, category_levels, transform_state, coding
    ):
        start = len(names)
        term_columns = list(_factor_columns(factors, n))
        positions = _column_positions(factors, first_fastest)
        columns.extend(term_columns[index] for index in np.argsort(positions, kind="stable"))
        names.extend(_factor_names(factors, first_fastest))
        if sizes is not None:
            sizes.append(len(names) - start)
    return columns, names


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

    if n_terms == 1:
        # Each row has at most one entry. Its group code is already the CSR
        # column index, so no coordinate arrays or intermediate sort are needed.
        values = term_cols[0]
        present = valid_mask & (values != 0)
        index_dtype = np.int32 if max(n, n_levels) <= np.iinfo(np.int32).max else np.int64
        if present.all():
            indptr = np.arange(n + 1, dtype=index_dtype)
            indices = level_indices
        else:
            indptr = np.empty(n + 1, dtype=index_dtype)
            indptr[0] = 0
            np.cumsum(present, out=indptr[1:])
            indices = level_indices[present]
            values = values[present]
        return sparse.csr_matrix(
            (values, indices, indptr), shape=(n, n_levels), dtype=np.float64
        ).tocsc()

    all_rows: list[NDArray[np.int64]] = []
    all_cols: list[NDArray[np.int64]] = []
    all_vals: list[NDArray[np.floating]] = []

    for j, term_col in enumerate(term_cols):
        nonzero_mask = (term_col != 0) & valid_mask
        row_idx = np.where(nonzero_mask)[0]
        if len(row_idx) > 0:
            col_idx = level_indices[row_idx] * n_terms + j
            all_rows.append(row_idx)
            all_cols.append(col_idx)
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


def _random_term_columns(
    rterm: RandomTerm,
    data: Any,
    n: int,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    transform_state: TransformState | None = None,
    coding: str = "R",
    sizes: list[int] | None = None,
) -> tuple[list[NDArray[np.floating]], list[str]]:
    """Encode within-group random terms consistently for fitting and prediction."""
    columns: list[NDArray[np.floating]] = (
        [np.ones(n, dtype=np.float64)] if rterm.has_intercept else []
    )
    names = ["(Intercept)"] if rterm.has_intercept else []
    if sizes is not None and rterm.has_intercept:
        sizes.append(1)
    encoded_columns, encoded_names = _encode_terms(
        rterm.expr,
        data,
        has_intercept=rterm.has_intercept,
        contrasts=contrasts,
        category_levels=category_levels,
        transform_state=transform_state,
        coding=coding,
        sizes=sizes,
    )
    columns.extend(encoded_columns)
    names.extend(encoded_names)
    return columns, names


def build_random_matrix(
    formula: Formula,
    data: Any,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    *,
    transform_state: TransformState | None = None,
    coding: str = "R",
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
            transform_state,
            coding,
        )
        Z_blocks.append(Z_block)
        structures.append(structure)

    if not Z_blocks:
        return sparse.csc_matrix((n, 0), dtype=np.float64), []

    if len(Z_blocks) == 1:
        return Z_blocks[0], structures
    return sparse.hstack(Z_blocks, format="csc"), structures


def _build_random_block(
    rterm: RandomTerm,
    data: Any,
    n: int,
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    category_levels: dict[str, list[Any]] | None = None,
    transform_state: TransformState | None = None,
    coding: str = "R",
) -> tuple[sparse.csc_matrix, RandomEffectStructure]:
    if isinstance(rterm.grouping, str):
        group_values = get_column_numpy(data, rterm.grouping)
    else:
        group_values = concat_columns_as_string(data, list(rterm.grouping), escape=True)
    level_indices, level_map = factorize_levels(group_values)
    n_levels = len(level_map)

    sizes: list[int] = []
    term_cols, term_names = _random_term_columns(
        rterm, data, n, contrasts, category_levels, transform_state, coding, sizes
    )

    n_terms = len(term_cols)

    Z_block = _build_sparse_Z_block(level_indices, term_cols, n, n_levels, n_terms)

    structure = RandomEffectStructure(
        grouping_factor=_format_grouping(rterm.grouping),
        term_names=term_names,
        n_levels=n_levels,
        n_terms=n_terms,
        correlated=rterm.correlated,
        level_map=level_map,
        cov_type=rterm.cov_type,
        level_indices=level_indices,
        term_sizes=tuple(sizes),
    )

    return Z_block, structure

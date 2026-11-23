"""Evaluate computed formula variables such as ``log(x)`` and ``poly(x, 2)``.

Stateful transforms learn their parameters from the fitting data: ``scale()``
its center and scale, orthogonal ``poly()`` its recurrence coefficients, and
factor coding its levels. The parameters are recorded in a state mapping keyed
by the term label, and predictions for new data reuse them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mixedlm.formula.terms import (
    Binary,
    Call,
    Expr,
    Formula,
    Literal,
    Num,
    TransformTerm,
    Unary,
    Var,
    expr_variables,
    format_expr,
)
from mixedlm.utils.dataframe import (
    dataframe_length,
    get_categories,
    get_column_numpy,
    get_columns,
    is_categorical_or_string,
)

TransformState = MutableMapping[str, Any]

_ELEMENTWISE: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "exp": np.exp,
    "expm1": np.expm1,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "floor": np.floor,
    "ceiling": np.ceil,
    "trunc": np.trunc,
    "sign": np.sign,
}

_BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "**": np.power,
}


def _numeric_column(data: Any, name: str, context: str) -> NDArray[np.float64]:
    if is_categorical_or_string(data, name):
        raise ValueError(f"{context} requires a numeric variable, but '{name}' is categorical")
    values = get_column_numpy(data, name, dtype=np.float64)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        # Nullable integer columns export missing values as pd.NA.
        import pandas as pd

        numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
        return numeric.to_numpy(dtype=np.float64, na_value=np.nan)


# Refits store a computed response, which cannot in general be inverted, under
# its label behind this prefix, so that it never collides with a data column.
STORED_PREFIX = "\x00"


def stored_column(label: str) -> str:
    """The frame column that holds a refit's values of a computed response."""
    return STORED_PREFIX + label


def _stored_columns(data: Any) -> dict[str, str]:
    return {
        column[len(STORED_PREFIX) :]: column
        for column in get_columns(data)
        if column.startswith(STORED_PREFIX)
    }


def evaluate_numeric(
    expr: Expr,
    data: Any,
    context: str | None = None,
    *,
    stored: Mapping[str, str] | None = None,
) -> NDArray[np.float64]:
    """Evaluate an arithmetic expression to one float64 value per row.

    ``stored`` maps subexpression labels to the columns that hold their values.
    """
    context = context or format_expr(expr)
    n = dataframe_length(data)

    def visit(node: Expr) -> Any:
        if stored and not isinstance(node, Var | Num | Literal):
            column = stored.get(format_expr(node))
            if column is not None:
                return _numeric_column(data, column, context)
        if isinstance(node, Var):
            return _numeric_column(data, node.name, context)
        if isinstance(node, Num):
            return np.float64(node.value)
        if isinstance(node, Literal):
            raise ValueError(f"{context} cannot use the constant {format_expr(node)} as data")
        if isinstance(node, Unary):
            return np.negative(visit(node.operand))
        if isinstance(node, Binary):
            return _BINARY[node.op](visit(node.left), visit(node.right))
        if isinstance(node, Call):
            value = visit(node.args[0])
            if node.func == "I":
                return value
            if node.func == "log":
                base = dict(node.options).get("base")
                if isinstance(base, Num):
                    return np.log(value) / np.log(base.value)
            function = _ELEMENTWISE.get(node.func)
            if function is None:
                raise ValueError(f"{node.func}() cannot be nested inside {context}")
            return function(value)
        raise TypeError(f"Unsupported formula expression {node!r}")

    with np.errstate(all="ignore"):
        values = visit(expr)
    return np.broadcast_to(np.asarray(values, dtype=np.float64), (n,)).copy()


def _missing_inputs(expr: Expr, data: Any) -> NDArray[np.bool_]:
    missing = np.zeros(dataframe_length(data), dtype=bool)
    for name in expr_variables(expr):
        values = get_column_numpy(data, name)
        try:
            missing |= np.isnan(np.asarray(values, dtype=np.float64))
        except (TypeError, ValueError):
            import pandas as pd

            missing |= np.asarray(pd.isna(values), dtype=bool)
    return missing


def _require_finite(values: NDArray[np.float64], expr: Expr, data: Any, label: str) -> None:
    """Reject values that a transform made non-finite from observed inputs."""
    finite: NDArray[np.bool_] = np.isfinite(values)
    if finite.ndim > 1:
        finite = np.all(finite, axis=1)
    if finite.all():
        return
    invalid = ~finite & ~_missing_inputs(expr, data)
    if invalid.any():
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"Formula term '{label}' is not finite for {int(invalid.sum())} observation(s), "
            f"first at position {first} among the rows evaluated after removing missing "
            "values. Check the transform's domain, such as log() of values <= 0."
        )


def evaluate_expression(
    expr: Expr, data: Any, label: str, *, stored: Mapping[str, str] | None = None
) -> NDArray[np.float64]:
    """Evaluate an offset or response expression, rejecting domain errors."""
    values = evaluate_numeric(expr, data, label, stored=stored)
    _require_finite(values, expr, data, label)
    return values


def response_values(expr: Expr, label: str, data: Any) -> NDArray[np.float64]:
    """Evaluate a computed response, preferring a refit's stored values.

    Refits store replacement values of a computed response in a reserved column
    (see :func:`stored_column`). A later response that transforms it, such as
    exp(log(y)) from update("exp(.) ~ ."), reads the stored log(y) values too.
    Data columns are always computed from, as in R, whatever their names.
    """
    stored = _stored_columns(data)
    if label in stored:
        return _numeric_column(data, stored[label], label)
    return evaluate_expression(expr, data, label, stored=stored)


def learn_transform_state(formula: Formula, data: Any, state: TransformState) -> None:
    """Learn scale() and orthogonal poly() parameters before missing rows are dropped.

    As R's model.frame() does, these use every observed value of the transform's
    own inputs, including rows dropped for missing values in other variables.
    Factor levels are learned from the fitting rows, as lme4 drops unused levels.
    """
    from mixedlm.formula.terms import term_factors

    terms = [*formula.fixed.terms, *(term for rterm in formula.random for term in rterm.expr)]
    for term in terms:
        for factor in term_factors(term):
            if not isinstance(factor, TransformTerm) or factor.label in state:
                continue
            stateful = factor.kind == "scale" or (
                factor.kind == "poly" and not factor.option("raw")
            )
            if not stateful:
                continue
            values = evaluate_numeric(factor.data_expr, data, factor.label)
            observed = values[np.isfinite(values)]
            if factor.kind == "scale":
                state[factor.label] = _scale_parameters(factor, observed)
            else:
                degree = int(factor.option("degree"))
                state[factor.label] = _poly_parameters(observed, degree, factor.label)


def formula_offset(formula: Formula, data: Any) -> NDArray[np.float64] | None:
    """Sum the formula's offset() terms, or return None when it has none."""
    if not formula.offsets:
        return None
    total = np.zeros(dataframe_length(data), dtype=np.float64)
    for offset in formula.offsets:
        total += evaluate_expression(offset, data, f"offset({format_expr(offset)})")
    return total


def factor_transform_levels(formula: Formula, state: TransformState | None) -> dict[str, list[Any]]:
    """Fitted levels of variables coded directly by factor(), C() or relevel().

    Reference grids for such variables use these levels rather than numeric
    summaries such as the mean, which would be unseen levels.
    """
    from mixedlm.formula.terms import term_factors

    levels: dict[str, list[Any]] = {}
    for term in formula.fixed.terms:
        for factor in term_factors(term):
            if (
                isinstance(factor, TransformTerm)
                and factor.is_categorical
                and isinstance(factor.data_expr, Var)
                and state is not None
                and factor.label in state
            ):
                levels.setdefault(factor.data_expr.name, list(state[factor.label]))
    return levels


def encode_numeric_transform(
    term: TransformTerm,
    data: Any,
    state: TransformState | None,
) -> tuple[list[NDArray[np.float64]], list[str]]:
    """Encode a numeric transform as columns, learning parameters when needed."""
    label = term.label
    if term.kind in ("elementwise", "identity"):
        columns = evaluate_numeric(term.expr, data, label)[:, None]
    else:
        values = evaluate_numeric(term.data_expr, data, label)
        if term.kind == "scale":
            center, scale = _cached(state, label, lambda: _scale_parameters(term, values))
            columns = ((values - center) / scale)[:, None]
        elif term.kind == "poly":
            degree = int(term.option("degree"))
            if term.option("raw"):
                columns = np.column_stack([values**power for power in range(1, degree + 1)])
            else:
                alpha, norm2 = _cached(
                    state, label, lambda: _poly_parameters(values, degree, label)
                )
                columns = _poly_basis(values, alpha, norm2)
        else:
            raise TypeError(f"{label} is not a numeric transform")
    _require_finite(columns, term.data_expr, data, label)
    return [np.ascontiguousarray(column) for column in columns.T], list(term.column_names)


def factor_values(
    term: TransformTerm,
    data: Any,
    state: TransformState | None,
) -> tuple[NDArray[Any], list[Any]]:
    """Return a factor transform's values and its fitted level order."""
    label = term.label
    expr = term.data_expr
    if isinstance(expr, Var):
        values = np.asarray(get_column_numpy(data, expr.name))
    else:
        values = evaluate_numeric(expr, data, label)

    def fitted_levels() -> list[Any]:
        import pandas as pd

        if isinstance(expr, Var):
            levels = list(get_categories(data, expr.name))
        else:
            levels = sorted(np.unique(values[np.isfinite(values)]).tolist())
        if term.func == "relevel":
            levels = _relevel(levels, term.option("ref"), label)
        # lme4's model frame drops levels no fitted row takes, after relevel(), so
        # an unused reference gives way to the first observed level.
        observed = set(pd.Index(values).dropna().unique().tolist())
        return [level for level in levels if level in observed]

    levels = _cached(state, label, fitted_levels)
    return values, list(levels)


def _relevel(levels: list[Any], ref: Any, label: str) -> list[Any]:
    """Move the reference level first, matching by value and then by its text."""
    matches = [i for i, level in enumerate(levels) if level == ref and type(level) is not bool]
    matches = matches or [i for i, level in enumerate(levels) if str(level) == str(ref)]
    if not matches:
        raise ValueError(f"{label}: reference level {ref!r} is not one of {levels!r}")
    index = matches[0]
    return [levels[index], *levels[:index], *levels[index + 1 :]]


def _cached(state: TransformState | None, label: str, compute: Callable[[], Any]) -> Any:
    if state is None:
        return compute()
    if label not in state:
        state[label] = compute()
    return state[label]


def _scale_parameters(term: TransformTerm, values: NDArray[np.float64]) -> tuple[float, float]:
    """Match R's scale(): the mean, and the root mean square with n - 1 divisor."""
    observed = values[np.isfinite(values)]
    center = float(np.mean(observed)) if term.option("center") and len(observed) else 0.0
    if not term.option("scale"):
        return center, 1.0
    if len(observed) < 2:
        raise ValueError(f"{term.label} needs at least two observations to estimate a scale")
    scale = float(np.sqrt(np.sum((observed - center) ** 2) / (len(observed) - 1)))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(f"{term.label} has zero scale in the fitting data")
    return center, scale


def _poly_parameters(
    values: NDArray[np.float64], degree: int, label: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return R's orthogonal polynomial coefficients from a QR factorization."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} cannot fit orthogonal polynomials to missing values")
    if degree >= len(np.unique(values)):
        raise ValueError(f"{label}: 'degree' must be less than the number of unique points")
    mean = float(np.mean(values))
    centered = values - mean
    vandermonde = np.vander(centered, degree + 1, increasing=True)
    q, r = np.linalg.qr(vandermonde)
    raw = q * np.diag(r)
    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum(centered[:, None] * raw**2, axis=0) / norm2 + mean)[:degree]
    return tuple(alpha.tolist()), (1.0, *norm2.tolist())


def _poly_basis(
    values: NDArray[np.float64], alpha: tuple[float, ...], norm2: tuple[float, ...]
) -> NDArray[np.float64]:
    """Evaluate orthogonal polynomials by R's three-term recurrence."""
    degree = len(alpha)
    basis = np.ones((len(values), degree + 1), dtype=np.float64)
    basis[:, 1] = values - alpha[0]
    for i in range(1, degree):
        basis[:, i + 1] = (values - alpha[i]) * basis[:, i] - (norm2[i + 1] / norm2[i]) * basis[
            :, i - 1
        ]
    basis /= np.sqrt(np.asarray(norm2[1:]))
    return basis[:, 1:]

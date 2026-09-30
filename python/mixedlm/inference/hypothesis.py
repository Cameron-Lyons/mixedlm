from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, stats

if TYPE_CHECKING:
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

ConstraintRow: TypeAlias = Mapping[str, float]
HypothesisSpec: TypeAlias = (
    ArrayLike | ConstraintRow | Sequence[ConstraintRow] | Mapping[str, ConstraintRow] | pd.DataFrame
)


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.ma.is_masked(value):
        raise TypeError(f"{name} must not contain masked values.")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric.") from None
    if np.iscomplexobj(array) or (
        array.dtype.kind == "O" and any(np.iscomplexobj(item) for item in array.flat)
    ):
        raise TypeError(f"{name} must contain real numeric values.")
    if array.dtype.kind == "O" and any(np.ma.is_masked(item) for item in array.flat):
        raise TypeError(f"{name} must not contain masked values.")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric.") from None


def _normalize_rows(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Use binary scales with headroom for dot products and retain cancellation."""
    exponents = np.frexp(np.max(np.abs(matrix), axis=1))[1]
    normalized = np.ldexp(matrix, -exponents[:, None])
    extra = np.frexp(np.sum(np.abs(normalized), axis=1))[1] + 1
    return np.ldexp(normalized, -extra[:, None]), exponents + extra


def _constraint_rank(constraints: NDArray[np.float64]) -> int:
    mantissa, exponent = np.frexp(constraints)
    active = mantissa != 0
    row_power = np.max(np.where(active, exponent, np.iinfo(np.int32).min), axis=1)
    relative = exponent - row_power[:, None]
    column_power = np.max(np.where(active, relative, np.iinfo(np.int32).min), axis=0)
    column_power = np.where(np.any(active, axis=0), column_power, 0)
    # Apply both scales together so neither row nor column units erase the other.
    scaled = np.ldexp(mantissa, relative - column_power[None, :])
    return int(np.linalg.matrix_rank(scaled))


def _scaled_products(
    matrix: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.bool_]]:
    mantissa, exponent = np.frexp(matrix)
    weight_mantissa, weight_exponent = np.frexp(weights)
    products = mantissa * weight_mantissa
    powers = exponent + weight_exponent
    row_power = np.max(np.where(products != 0, powers, np.iinfo(np.int32).min), axis=1)
    row_power = np.where(np.any(products != 0, axis=1), row_power, 0)
    scaled = np.ldexp(products, powers - row_power[:, None])
    lost = np.any((products != 0) & (scaled == 0), axis=1)
    return scaled, row_power, lost


def _scaled_estimates(
    constraints: NDArray[np.float64], beta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    if len(beta) == 1:
        coefficient, coefficient_exponent = np.frexp(constraints[:, 0])
        value, value_exponent = np.frexp(beta[0])
        return coefficient * value, coefficient_exponent + value_exponent
    products, exponents, lost = _scaled_products(constraints, beta)
    estimates = products.sum(axis=1)
    total = np.abs(products).sum(axis=1)
    cancellation = (total > 0) & (np.abs(estimates) <= 8 * np.finfo(float).eps * total)
    for row in np.flatnonzero(lost | cancellation):
        # Only severely cancelling or out-of-range products need exact accumulation.
        value = sum(
            (
                Fraction(float(c)) * Fraction(float(b))
                for c, b in zip(constraints[row], beta, strict=True)
            ),
            Fraction(),
        )
        if not value:
            estimates[row], exponents[row] = 0.0, 0
            continue
        exponent = value.numerator.bit_length() - value.denominator.bit_length()
        normalized = value / (1 << exponent) if exponent >= 0 else value * (1 << -exponent)
        estimates[row], exponents[row] = float(normalized), exponent
    return estimates, exponents


def _scaled_sum(
    left: NDArray[np.float64],
    left_exponent: NDArray[np.int32] | int,
    right: NDArray[np.float64],
    right_exponent: NDArray[np.int32] | int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Add binary-scaled values before restoring potentially extreme units."""
    left_mantissa, left_shift = np.frexp(left)
    right_mantissa, right_shift = np.frexp(right)
    left_power = left_shift + left_exponent
    right_power = right_shift + right_exponent
    left_power = np.where(left_mantissa == 0, right_power, left_power)
    right_power = np.where(right_mantissa == 0, left_power, right_power)
    exponent = np.maximum(left_power, right_power)
    value = np.ldexp(left_mantissa, left_power - exponent) + np.ldexp(
        right_mantissa, right_power - exponent
    )
    return value, exponent


def _scaled_ratio(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    exponent: NDArray[np.int32],
) -> NDArray[np.float64]:
    top, top_exponent = np.frexp(numerator)
    bottom, bottom_exponent = np.frexp(denominator)
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(top / bottom, top_exponent - bottom_exponent + exponent)


def _symmetric_part(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    # The usual average preserves subnormals, but can overflow large finite entries.
    with np.errstate(over="ignore"):
        result = matrix + matrix.T
        result *= 0.5
    overflow = np.isinf(result)
    if np.any(overflow):
        result[overflow] = matrix[overflow] * 0.5 + matrix.T[overflow] * 0.5
    return result


def _scaled_hypothesis_covariance(
    constraints: NDArray[np.float64], covariance: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Equilibrate coefficient units, then scale each covariance projection row."""
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0) or np.any(covariance[diagonal == 0] != 0):
        raise ValueError("model covariance must have valid non-negative variances.")
    if len(diagonal) == 1:
        # A single selected coefficient needs only scalar variance scaling.
        value, value_exponent = np.frexp(diagonal[0])
        coefficient, exponent = np.frexp(constraints[:, 0])
        projected = coefficient[:, None] * coefficient[None, :]
        projected *= np.ldexp(value, value_exponent % 2)
        return projected, exponent + value_exponent // 2
    sd = np.sqrt(diagonal)
    divisor = np.where(sd == 0, 1.0, sd)
    correlation = np.empty_like(covariance)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(covariance, divisor[:, None], out=correlation)
        correlation /= divisor[None, :]
    if not np.all(np.isfinite(correlation)):
        raise ValueError("model covariance cannot be standardized to finite correlations.")

    weighted, row_power, _ = _scaled_products(constraints, sd)
    normalized, extra = _normalize_rows(weighted)
    projected = normalized @ correlation @ normalized.T
    return _symmetric_part(projected), row_power + extra


def _wald_statistic(
    covariance: NDArray[np.float64],
    std_error: NDArray[np.float64],
    row_statistic: NDArray[np.float64],
    divisor: int,
) -> float:
    """Whiten standardized restrictions, avoiding an inverse and quadratic cancellation."""
    n_rows = len(row_statistic)
    if n_rows == 1:
        with np.errstate(over="ignore", under="ignore"):
            return float(np.square(row_statistic[0]))
    with np.errstate(over="ignore", invalid="ignore"):
        correlation = covariance / std_error[:, None] / std_error[None, :]
    if not np.all(np.isfinite(correlation)):
        raise ValueError("hypothesis covariance must be positive definite.")
    np.fill_diagonal(correlation, 1.0)
    if np.linalg.matrix_rank(correlation) < n_rows:
        raise ValueError("hypothesis is not estimable from the fitted coefficient covariance.")
    try:
        factor = linalg.cholesky(correlation, lower=True)
    except linalg.LinAlgError:
        raise ValueError("hypothesis covariance must be positive definite.") from None
    if np.any(np.isinf(row_statistic)):
        return np.inf
    exponent = np.frexp(np.max(np.abs(row_statistic)))[1]
    scaled = np.ldexp(row_statistic, -exponent)
    scaled = linalg.solve_triangular(factor, scaled, lower=True)
    root = linalg.norm(scaled) / np.sqrt(divisor)
    with np.errstate(over="ignore", under="ignore"):
        return float(np.square(np.ldexp(root, exponent)))


def _readonly_float_array(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LinearHypothesisResult:
    """Wald test results for one or more linear fixed-effect hypotheses.

    The joint null hypothesis is ``C @ beta = rhs``. Row-level estimates,
    standard errors, confidence intervals, and tests are available through
    :attr:`table`; the scalar ``statistic`` and ``p_value`` describe the joint
    test across all rows of ``C``.
    """

    coefficient_names: tuple[str, ...]
    labels: tuple[str, ...]
    constraints: NDArray[np.float64]
    rhs: NDArray[np.float64]
    estimate: NDArray[np.float64]
    difference: NDArray[np.float64]
    std_error: NDArray[np.float64]
    conf_low: NDArray[np.float64]
    conf_high: NDArray[np.float64]
    row_statistic: NDArray[np.float64]
    row_p_value: NDArray[np.float64]
    covariance: NDArray[np.float64]
    statistic: float
    numerator_df: int
    denominator_df: float | None
    p_value: float
    test: str
    level: float

    def __post_init__(self) -> None:
        for field_name in (
            "constraints",
            "rhs",
            "estimate",
            "difference",
            "std_error",
            "conf_low",
            "conf_high",
            "row_statistic",
            "row_p_value",
            "covariance",
        ):
            object.__setattr__(self, field_name, _readonly_float_array(getattr(self, field_name)))

    @property
    def table(self) -> pd.DataFrame:
        """Return one row per hypothesis with estimates and uncertainty."""
        statistic_name = "t_value" if self.test == "F" else "z_value"
        return pd.DataFrame(
            {
                "hypothesis": self.labels,
                "estimate": self.estimate,
                "null_value": self.rhs,
                "difference": self.difference,
                "std_error": self.std_error,
                statistic_name: self.row_statistic,
                "p_value": self.row_p_value,
                "conf_low": self.conf_low,
                "conf_high": self.conf_high,
            }
        )

    def summary(self) -> str:
        """Return a readable row-level table followed by the joint test."""
        lines = ["Linear hypothesis test", "", self.table.to_string(index=False)]
        lines.append("")
        if self.test == "F":
            assert self.denominator_df is not None
            lines.append(
                f"Joint test: F({self.numerator_df}, {self.denominator_df:.2f}) = "
                f"{self.statistic:.6g}, p = {self.p_value:.6g}"
            )
        else:
            lines.append(
                f"Joint test: Chisq({self.numerator_df}) = {self.statistic:.6g}, "
                f"p = {self.p_value:.6g}"
            )
        lines.append(f"Confidence level: {self.level:.1%}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"LinearHypothesisResult(n_hypotheses={len(self.labels)}, "
            f"test={self.test!r}, p_value={self.p_value:.6g})"
        )


def _format_constraint(row: ConstraintRow) -> str:
    pieces: list[str] = []
    for name, raw_weight in row.items():
        weight = float(raw_weight)
        if weight == 0:
            continue
        magnitude = abs(weight)
        term = name if magnitude == 1 else f"{magnitude:g}*{name}"
        if not pieces:
            pieces.append(term if weight > 0 else f"-{term}")
        else:
            pieces.append(f"{'+' if weight > 0 else '-'} {term}")
    return " ".join(pieces) if pieces else "0"


def _rows_to_matrix(
    rows: Sequence[ConstraintRow],
    coefficient_names: tuple[str, ...],
) -> NDArray[np.float64]:
    name_to_index = {name: index for index, name in enumerate(coefficient_names)}
    matrix = np.zeros((len(rows), len(coefficient_names)), dtype=np.float64)

    for row_index, row in enumerate(rows):
        if any(not isinstance(name, str) for name in row):
            raise TypeError("Constraint coefficient names must be strings.")
        unknown = sorted(set(row) - set(name_to_index))
        if unknown:
            raise ValueError(
                f"Unknown coefficient name(s): {unknown}. "
                f"Available coefficients: {list(coefficient_names)}"
            )
        for name, weight in row.items():
            numeric = _real_array(weight, f"Constraint weight for {name!r}")
            try:
                matrix[row_index, name_to_index[name]] = float(numeric)
            except (TypeError, ValueError):
                raise TypeError(f"Constraint weight for {name!r} must be numeric.") from None

    return matrix


def _resolve_labels(
    defaults: Sequence[str],
    labels: Sequence[str] | None,
    n_rows: int,
) -> tuple[str, ...]:
    if labels is None:
        resolved = tuple(str(label) for label in defaults)
    else:
        if isinstance(labels, str):
            raise TypeError("labels must be a sequence of strings, not a single string.")
        resolved = tuple(str(label) for label in labels)

    if len(resolved) != n_rows:
        raise ValueError(f"labels has length {len(resolved)}, expected {n_rows}.")
    if any(not label for label in resolved):
        raise ValueError("labels must not contain empty strings.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("labels must be unique.")
    return resolved


def _coerce_constraints(
    hypothesis: HypothesisSpec,
    coefficient_names: tuple[str, ...],
    labels: Sequence[str] | None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    default_labels: Sequence[str]

    if isinstance(hypothesis, pd.DataFrame):
        if hypothesis.empty:
            raise ValueError("hypothesis DataFrame must contain at least one row.")
        if not hypothesis.columns.is_unique:
            raise ValueError("hypothesis DataFrame columns must be unique.")
        if any(not isinstance(name, str) for name in hypothesis.columns):
            raise TypeError("hypothesis DataFrame column names must be strings.")
        unknown = sorted(set(hypothesis.columns) - set(coefficient_names))
        if unknown:
            raise ValueError(
                f"Unknown coefficient name(s): {unknown}. "
                f"Available coefficients: {list(coefficient_names)}"
            )
        matrix = np.zeros((len(hypothesis), len(coefficient_names)), dtype=np.float64)
        name_to_index = {name: index for index, name in enumerate(coefficient_names)}
        for name in hypothesis.columns:
            matrix[:, name_to_index[str(name)]] = _real_array(
                hypothesis[name].to_numpy(), f"Constraint column {name!r}"
            )
        default_labels = [str(value) for value in hypothesis.index]

    elif isinstance(hypothesis, Mapping):
        if not hypothesis:
            raise ValueError("hypothesis mapping must not be empty.")
        nested = [isinstance(value, Mapping) for value in hypothesis.values()]
        if all(nested):
            nested_rows = [cast(ConstraintRow, value) for value in hypothesis.values()]
            matrix = _rows_to_matrix(nested_rows, coefficient_names)
            default_labels = [str(name) for name in hypothesis]
        elif any(nested):
            raise TypeError("hypothesis mapping cannot mix numeric weights and constraint rows.")
        else:
            row = cast(ConstraintRow, hypothesis)
            matrix = _rows_to_matrix([row], coefficient_names)
            default_labels = [_format_constraint(row)]

    else:
        if isinstance(hypothesis, str):
            raise TypeError("hypothesis must be a matrix or named coefficient weights.")

        materialized: Any = hypothesis
        if not isinstance(hypothesis, np.ndarray):
            try:
                materialized = list(cast(Any, hypothesis))
            except TypeError:
                materialized = hypothesis

        if (
            isinstance(materialized, list)
            and materialized
            and all(isinstance(row, Mapping) for row in materialized)
        ):
            rows = [cast(ConstraintRow, row) for row in materialized]
            matrix = _rows_to_matrix(rows, coefficient_names)
            default_labels = [_format_constraint(row) for row in rows]
        else:
            matrix = _real_array(materialized, "hypothesis")
            if matrix.ndim == 1:
                matrix = matrix[np.newaxis, :]
            elif matrix.ndim != 2:
                raise ValueError("hypothesis matrix must be one- or two-dimensional.")
            if matrix.shape[0] == 0:
                raise ValueError("hypothesis matrix must contain at least one row.")
            default_labels = [f"H{index + 1}" for index in range(matrix.shape[0])]

    if matrix.shape[1] != len(coefficient_names):
        raise ValueError(
            f"hypothesis matrix has {matrix.shape[1]} columns, "
            f"expected {len(coefficient_names)} for {list(coefficient_names)}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hypothesis constraints must contain only finite values.")
    if np.any(np.all(matrix == 0, axis=1)):
        raise ValueError("hypothesis constraints must not contain an all-zero row.")

    n_rows = matrix.shape[0]
    if n_rows > len(coefficient_names):
        raise ValueError("hypothesis rows must be linearly independent.")

    return matrix, _resolve_labels(default_labels, labels, n_rows)


def _coerce_rhs(rhs: float | ArrayLike, n_rows: int) -> NDArray[np.float64]:
    values = _real_array(rhs, "rhs")

    if values.ndim == 0:
        result = np.full(n_rows, float(values), dtype=np.float64)
    elif values.ndim == 1 and len(values) == n_rows:
        result = values.copy()
    elif values.ndim == 1:
        raise ValueError(f"rhs has length {len(values)}, expected {n_rows}.")
    else:
        raise ValueError("rhs must be a scalar or one-dimensional array.")

    if not np.all(np.isfinite(result)):
        raise ValueError("rhs must contain only finite values.")
    return result


def _normalise_test(test: str, is_lmm: bool) -> str:
    if not isinstance(test, str):
        raise TypeError("test must be a string.")
    normalised = test.strip().lower().replace("_", "").replace("-", "")
    if normalised == "auto":
        return "F" if is_lmm else "Chisq"
    if normalised == "f":
        return "F"
    if normalised in {"chisq", "chi2", "chisquare"}:
        return "Chisq"
    raise ValueError("test must be 'auto', 'F', or 'Chisq'.")


def linear_hypothesis(
    model: LmerResult | GlmerResult,
    hypothesis: HypothesisSpec,
    rhs: float | ArrayLike = 0.0,
    *,
    labels: Sequence[str] | None = None,
    test: str = "auto",
    denominator_df: float | None = None,
    level: float = 0.95,
) -> LinearHypothesisResult:
    """Test one or more linear restrictions on fixed-effect coefficients.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        Fitted linear or generalized linear mixed model.
    hypothesis : array-like, mapping, sequence of mappings, nested mapping, or DataFrame
        Constraint matrix ``C`` or named coefficient weights. A flat mapping
        defines one row, for example ``{"x": 1, "z": -1}``. A nested mapping
        defines labeled rows. A DataFrame uses coefficient names as columns and
        its index as labels. Numeric matrices must follow the fitted coefficient
        order in ``model.matrices.fixed_names``.
    rhs : float or array-like, default 0
        Null value or one null value per constraint row.
    labels : sequence of str, optional
        Replacement row labels.
    test : {"auto", "F", "Chisq"}, default "auto"
        Joint Wald test. ``auto`` selects an F test for LMMs and a chi-square
        test for GLMMs.
    denominator_df : float, optional
        Denominator degrees of freedom for an F test. Defaults to the model's
        residual degrees of freedom. Invalid for chi-square tests.
    level : float, default 0.95
        Confidence level for row-level intervals.

    Returns
    -------
    LinearHypothesisResult
        Row-level estimates and the joint Wald test of ``C @ beta = rhs``.

    Notes
    -----
    Restrictions and coefficient uncertainty are scaled internally to handle
    different units. Returned estimates, intervals, and covariance retain the
    original restriction units. Values outside float64's range can display as
    zero or infinity even when the test statistics remain finite. Constraints
    and null values must be real and must not contain masked elements.

    Examples
    --------
    Test whether two fixed-effect slopes are equal:

    >>> result = linear_hypothesis(model, {"x": 1, "z": -1})
    >>> result.p_value

    Test two restrictions jointly against non-zero null values:

    >>> result = linear_hypothesis(
    ...     model,
    ...     {"equal slopes": {"x": 1, "z": -1}, "sum": {"x": 1, "z": 1}},
    ...     rhs=[0, 2],
    ... )
    """
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    if isinstance(model, LmerResult):
        is_lmm = True
    elif isinstance(model, GlmerResult):
        is_lmm = False
    else:
        raise TypeError("model must be a fitted LmerResult or GlmerResult.")

    try:
        level = float(level)
    except (TypeError, ValueError):
        raise TypeError("level must be numeric.") from None
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1.")

    test_name = _normalise_test(test, is_lmm)
    if test_name == "F":
        if denominator_df is None:
            denominator_df = float(model.df_residual())
        else:
            try:
                denominator_df = float(denominator_df)
            except (TypeError, ValueError):
                raise TypeError("denominator_df must be numeric.") from None
        if not np.isfinite(denominator_df) or denominator_df <= 0:
            raise ValueError("denominator_df must be a positive finite number.")
    elif denominator_df is not None:
        raise ValueError("denominator_df is only valid for an F test.")

    coefficient_names = tuple(model.matrices.fixed_names)
    constraints, resolved_labels = _coerce_constraints(hypothesis, coefficient_names, labels)
    rhs_values = _coerce_rhs(rhs, constraints.shape[0])
    n_hypotheses = constraints.shape[0]
    active = np.flatnonzero(np.any(constraints != 0, axis=0))
    working_constraints = constraints[:, active]
    if n_hypotheses > 1 and _constraint_rank(working_constraints) < n_hypotheses:
        raise ValueError("hypothesis rows must be linearly independent.")

    beta = np.asarray(model.beta, dtype=np.float64)
    beta_covariance = np.asarray(model.vcov(), dtype=np.float64)
    p = len(coefficient_names)
    if beta.shape != (p,):
        raise ValueError(f"model beta has shape {beta.shape}, expected ({p},).")
    if beta_covariance.shape != (p, p):
        raise ValueError(
            f"model covariance has shape {beta_covariance.shape}, expected ({p}, {p})."
        )
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(beta_covariance)):
        raise ValueError("model coefficients and covariance must be finite.")

    selected_covariance = (
        beta_covariance if len(active) == p else beta_covariance[np.ix_(active, active)]
    )
    covariance, covariance_exponent = _scaled_hypothesis_covariance(
        working_constraints, _symmetric_part(selected_covariance)
    )
    variances = np.diag(covariance)
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        raise ValueError("hypothesis has non-positive or non-finite variance.")
    scaled_se = np.sqrt(variances)
    scaled_estimate, estimate_exponent = _scaled_estimates(working_constraints, beta[active])
    scaled_difference, difference_exponent = _scaled_sum(
        scaled_estimate, estimate_exponent, -rhs_values, 0
    )
    row_statistic = _scaled_ratio(
        scaled_difference, scaled_se, difference_exponent - covariance_exponent
    )

    if test_name == "F":
        critical = stats.t.isf((1 - level) / 2, denominator_df)
        row_p_value = 2 * stats.t.sf(np.abs(row_statistic), denominator_df)
    else:
        critical = stats.norm.isf((1 - level) / 2)
        row_p_value = 2 * stats.norm.sf(np.abs(row_statistic))

    critical_mantissa, critical_exponent = np.frexp(critical)
    se_mantissa, se_exponent = np.frexp(scaled_se)
    margin = critical_mantissa * se_mantissa
    margin_exponent = critical_exponent + se_exponent + covariance_exponent
    lower, lower_exponent = _scaled_sum(
        scaled_estimate, estimate_exponent, -margin, margin_exponent
    )
    upper, upper_exponent = _scaled_sum(scaled_estimate, estimate_exponent, margin, margin_exponent)
    with np.errstate(over="ignore", under="ignore"):
        estimate = np.ldexp(scaled_estimate, estimate_exponent)
        difference = np.ldexp(scaled_difference, difference_exponent)
        std_error = np.ldexp(scaled_se, covariance_exponent)
        conf_low = np.ldexp(lower, lower_exponent)
        conf_high = np.ldexp(upper, upper_exponent)
        contrast_covariance = np.ldexp(
            covariance, covariance_exponent[:, None] + covariance_exponent[None, :]
        )

    statistic = _wald_statistic(
        covariance, scaled_se, row_statistic, n_hypotheses if test_name == "F" else 1
    )
    if test_name == "F":
        assert denominator_df is not None
        p_value = float(stats.f.sf(statistic, n_hypotheses, denominator_df))
    else:
        p_value = float(stats.chi2.sf(statistic, n_hypotheses))

    return LinearHypothesisResult(
        coefficient_names=coefficient_names,
        labels=resolved_labels,
        constraints=constraints,
        rhs=rhs_values,
        estimate=estimate,
        difference=difference,
        std_error=std_error,
        conf_low=conf_low,
        conf_high=conf_high,
        row_statistic=row_statistic,
        row_p_value=row_p_value,
        covariance=contrast_covariance,
        statistic=statistic,
        numerator_df=n_hypotheses,
        denominator_df=denominator_df,
        p_value=p_value,
        test=test_name,
        level=level,
    )

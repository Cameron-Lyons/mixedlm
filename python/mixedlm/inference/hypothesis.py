from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
            try:
                matrix[row_index, name_to_index[name]] = float(weight)
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
            try:
                matrix[:, name_to_index[str(name)]] = hypothesis[name].to_numpy(dtype=np.float64)
            except (TypeError, ValueError):
                raise TypeError(f"Constraint column {name!r} must be numeric.") from None
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
            try:
                matrix = np.asarray(materialized, dtype=np.float64)
            except (TypeError, ValueError):
                raise TypeError("hypothesis must be a numeric matrix or named weights.") from None
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
    if np.linalg.matrix_rank(matrix) < n_rows:
        raise ValueError("hypothesis rows must be linearly independent.")

    return matrix, _resolve_labels(default_labels, labels, n_rows)


def _coerce_rhs(rhs: float | ArrayLike, n_rows: int) -> NDArray[np.float64]:
    try:
        values = np.asarray(rhs, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError("rhs must be numeric.") from None

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

    coefficient_names = tuple(model.matrices.fixed_names)
    constraints, resolved_labels = _coerce_constraints(hypothesis, coefficient_names, labels)
    rhs_values = _coerce_rhs(rhs, constraints.shape[0])

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

    beta_covariance = 0.5 * (beta_covariance + beta_covariance.T)
    estimate = constraints @ beta
    difference = estimate - rhs_values
    contrast_covariance = constraints @ beta_covariance @ constraints.T
    contrast_covariance = 0.5 * (contrast_covariance + contrast_covariance.T)

    n_hypotheses = constraints.shape[0]
    if np.linalg.matrix_rank(contrast_covariance) < n_hypotheses:
        raise ValueError("hypothesis is not estimable from the fitted coefficient covariance.")

    variances = np.diag(contrast_covariance)
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        raise ValueError("hypothesis has non-positive or non-finite variance.")
    std_error = np.sqrt(variances)

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
        critical = stats.t.ppf(0.5 + level / 2, denominator_df)
        row_statistic = difference / std_error
        row_p_value = 2 * stats.t.sf(np.abs(row_statistic), denominator_df)
    else:
        if denominator_df is not None:
            raise ValueError("denominator_df is only valid for an F test.")
        critical = stats.norm.ppf(0.5 + level / 2)
        row_statistic = difference / std_error
        row_p_value = 2 * stats.norm.sf(np.abs(row_statistic))

    conf_low = estimate - critical * std_error
    conf_high = estimate + critical * std_error

    try:
        factor = linalg.cholesky(contrast_covariance, lower=True)
        solved = linalg.cho_solve((factor, True), difference)
    except linalg.LinAlgError:
        solved = linalg.solve(contrast_covariance, difference, assume_a="sym")
    wald = max(float(difference @ solved), 0.0)

    if test_name == "F":
        assert denominator_df is not None
        statistic = wald / n_hypotheses
        p_value = float(stats.f.sf(statistic, n_hypotheses, denominator_df))
    else:
        statistic = wald
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

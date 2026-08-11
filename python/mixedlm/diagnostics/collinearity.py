from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CollinearityResult:
    """Weighted fixed-effect collinearity diagnostics."""

    terms: tuple[str, ...]
    df: NDArray[np.int64]
    gvif: NDArray[np.float64]
    adjusted_gvif: NDArray[np.float64]
    vif: NDArray[np.float64]
    tolerance: NDArray[np.float64]
    severity: tuple[str, ...]
    condition_indices: NDArray[np.float64]
    condition_number: float
    rank: int
    n_columns: int
    weighting: str

    def problematic(self, threshold: float = 5.0) -> tuple[str, ...]:
        """Return terms at or above a VIF-equivalent threshold."""
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be finite and positive")
        return tuple(
            term for term, value in zip(self.terms, self.vif, strict=True) if value >= threshold
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return term-level diagnostics as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "term": self.terms,
                "df": self.df,
                "GVIF": self.gvif,
                "GVIF^(1/(2*df))": self.adjusted_gvif,
                "VIF": self.vif,
                "tolerance": self.tolerance,
                "severity": self.severity,
            }
        )

    def __len__(self) -> int:
        return len(self.terms)

    def __str__(self) -> str:
        lines = [
            f"Collinearity diagnostics ({self.weighting})",
            f"Condition number: {self.condition_number:.3f} (rank {self.rank}/{self.n_columns})",
            "",
            f"{'Term':<24} {'df':>4} {'GVIF':>10} {'VIF':>10} {'Tolerance':>11} {'Severity':>10}",
            "-" * 75,
        ]
        for i, term in enumerate(self.terms):
            lines.append(
                f"{term[:24]:<24} {self.df[i]:>4d} {self.gvif[i]:>10.3f} "
                f"{self.vif[i]:>10.3f} {self.tolerance[i]:>11.3f} "
                f"{self.severity[i]:>10}"
            )
        return "\n".join(lines)


def check_collinearity(
    model: Any,
    moderate: float = 5.0,
    high: float = 10.0,
) -> CollinearityResult:
    """Diagnose fixed-effect multicollinearity in a fitted mixed model.

    The calculation centers and scales ordinary fixed-effect columns using
    model-appropriate weights. Nonlinear parameter gradients are scaled without
    centering so constant information directions are retained. All VIF and
    generalized VIF values then come from one eigendecomposition. Multi-column
    terms are reported with raw GVIF, the conventional
    ``GVIF**(1 / (2 * df))`` adjustment, and a VIF-equivalent
    ``GVIF**(1 / df)`` value on the same threshold scale as one-column terms.

    Linear models use prior weights, generalized models use prior times working
    weights, and nonlinear models use the local parameter-gradient design.

    Parameters
    ----------
    model : LmerResult, GlmerResult, or NlmerResult
        Fitted mixed model.
    moderate : float, default 5
        Lower VIF-equivalent threshold for moderate collinearity.
    high : float, default 10
        Lower threshold for high collinearity.

    Returns
    -------
    CollinearityResult
        Term-level VIF diagnostics and global condition indices.
    """
    if not np.isfinite(moderate) or not np.isfinite(high):
        raise ValueError("Collinearity thresholds must be finite")
    if moderate <= 1.0 or high <= moderate:
        raise ValueError("Thresholds must satisfy 1 < moderate < high")

    design, names, weights, weighting, center = _design_and_weights(model)
    standardized, retained_names = _weighted_standardize(design, names, weights, center=center)
    n_columns = standardized.shape[1]

    if n_columns == 0:
        empty_float = np.array([], dtype=np.float64)
        return CollinearityResult(
            terms=(),
            df=np.array([], dtype=np.int64),
            gvif=empty_float,
            adjusted_gvif=empty_float.copy(),
            vif=empty_float.copy(),
            tolerance=empty_float.copy(),
            severity=(),
            condition_indices=empty_float.copy(),
            condition_number=1.0,
            rank=0,
            n_columns=0,
            weighting=weighting,
        )

    correlation = standardized.T @ standardized
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    largest = float(eigenvalues[-1])
    tolerance = np.finfo(np.float64).eps * max(design.shape) * largest
    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(positive))
    condition_indices = np.full(n_columns, np.inf, dtype=np.float64)
    condition_indices[positive] = np.sqrt(largest / eigenvalues[positive])

    term_indices = _group_term_indices(retained_names)
    if rank < n_columns:
        gvif = np.full(len(term_indices), np.inf, dtype=np.float64)
    else:
        inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
        gvif = _generalized_vif(correlation, inverse, term_indices)

    term_df = np.fromiter(
        (len(indices) for indices in term_indices.values()),
        dtype=np.int64,
        count=len(term_indices),
    )
    adjusted = np.power(gvif, 1.0 / (2.0 * term_df))
    vif = adjusted**2
    term_tolerance = np.divide(
        1.0,
        vif,
        out=np.zeros_like(vif),
        where=np.isfinite(vif) & (vif > 0.0),
    )
    severity = tuple(
        "high" if value >= high else "moderate" if value >= moderate else "low" for value in vif
    )

    return CollinearityResult(
        terms=tuple(term_indices),
        df=term_df,
        gvif=gvif,
        adjusted_gvif=adjusted,
        vif=vif,
        tolerance=term_tolerance,
        severity=severity,
        condition_indices=condition_indices,
        condition_number=float(np.max(condition_indices)),
        rank=rank,
        n_columns=n_columns,
        weighting=weighting,
    )


def _design_and_weights(
    model: Any,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64], str, bool]:
    if _model_flag(model, "isLMM"):
        design = np.asarray(model.matrices.X, dtype=np.float64)
        names = list(model.matrices.fixed_names)
        weights = np.asarray(model.matrices.weights, dtype=np.float64).reshape(-1)
        weighting = "prior weights"
        center = True
    elif _model_flag(model, "isGLMM"):
        design = np.asarray(model.matrices.X, dtype=np.float64)
        names = list(model.matrices.fixed_names)
        prior = np.asarray(model.matrices.weights, dtype=np.float64).reshape(-1)
        eta = np.asarray(model.linear_predictor(), dtype=np.float64).reshape(-1)
        mu = np.asarray(model.family.link.inverse(eta), dtype=np.float64).reshape(-1)
        working = np.asarray(model.family.weights(mu), dtype=np.float64).reshape(-1)
        weights = prior * working
        weighting = "prior × working weights"
        center = True
    elif _model_flag(model, "isNLMM"):
        design = np.asarray(model.model.gradient(model.phi, model.x), dtype=np.float64)
        names = [str(name) for name in model.model.param_names]
        nlmm_prior = getattr(model, "_weights", None)
        weights = (
            np.ones(design.shape[0], dtype=np.float64)
            if nlmm_prior is None
            else np.asarray(nlmm_prior, dtype=np.float64).reshape(-1)
        )
        weighting = "local gradient with prior weights"
        center = False
    else:
        raise TypeError("Expected a fitted linear, generalized, or nonlinear mixed model")

    _validate_design(design, names, weights)
    return design, names, weights, weighting, center


def _validate_design(
    design: NDArray[np.float64],
    names: list[str],
    weights: NDArray[np.float64],
) -> None:
    if design.ndim != 2:
        raise ValueError("Fixed-effect design must be a two-dimensional matrix")
    if design.shape[1] != len(names):
        raise ValueError(
            f"Fixed-effect design has {design.shape[1]} columns but {len(names)} names"
        )
    if design.shape[0] != len(weights):
        raise ValueError(f"weights has length {len(weights)}, expected {design.shape[0]}")
    if np.any(~np.isfinite(design)):
        raise ValueError("Fixed-effect design must contain only finite values")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Diagnostic weights must be finite and positive")


def _weighted_standardize(
    design: NDArray[np.float64],
    names: list[str],
    weights: NDArray[np.float64],
    center: bool,
) -> tuple[NDArray[np.float64], list[str]]:
    if center:
        weight_sum = float(np.sum(weights))
        means = np.sum(weights[:, None] * design, axis=0) / weight_sum
        transformed = design - means
    else:
        transformed = design
    sum_squares = np.sum(weights[:, None] * transformed**2, axis=0)
    reference_squares = np.sum(weights[:, None] * design**2, axis=0)
    column_tolerance = (
        np.finfo(np.float64).eps
        * max(design.shape)
        * np.maximum(reference_squares, np.finfo(np.float64).tiny)
    )
    varying = sum_squares > column_tolerance
    retained = [index for index, keep in enumerate(varying) if keep]
    if not retained:
        return np.empty((design.shape[0], 0), dtype=np.float64), []

    transformed = transformed[:, retained]
    sum_squares = sum_squares[retained]
    normalized = transformed * np.sqrt(weights[:, None] / sum_squares)
    return normalized, [names[index] for index in retained]


def _group_term_indices(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        term = _term_name(name)
        groups.setdefault(term, []).append(index)
    return groups


def _term_name(column_name: str) -> str:
    parts = column_name.split(":")
    return ":".join(re.sub(r"\[[^]]*\]", "", part) for part in parts)


def _generalized_vif(
    correlation: NDArray[np.float64],
    inverse: NDArray[np.float64],
    term_indices: dict[str, list[int]],
) -> NDArray[np.float64]:
    result = np.empty(len(term_indices), dtype=np.float64)
    max_log = np.log(np.finfo(np.float64).max)
    for position, indices in enumerate(term_indices.values()):
        block = np.ix_(indices, indices)
        sign_correlation, logdet_correlation = np.linalg.slogdet(correlation[block])
        sign_inverse, logdet_inverse = np.linalg.slogdet(inverse[block])
        log_gvif = logdet_correlation + logdet_inverse
        if sign_correlation <= 0 or sign_inverse <= 0 or log_gvif >= max_log:
            value = np.inf
        else:
            value = float(np.exp(log_gvif))
        result[position] = max(value, 1.0)
    return result


def _model_flag(model: Any, name: str) -> bool:
    flag = getattr(model, name, None)
    return bool(flag()) if callable(flag) else False

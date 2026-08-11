from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray


@dataclass
class RanefResult:
    values: dict[str, dict[str, NDArray[np.floating]]]
    condVar: dict[str, dict[str, NDArray[np.floating]]] | None = None

    def __getitem__(self, key: str) -> dict[str, NDArray[np.floating]]:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def keys(self):
        return self.values.keys()

    def items(self):
        return self.values.items()


@dataclass
class PredictResult:
    """Result of prediction with optional intervals.

    Attributes
    ----------
    fit : NDArray
        Predicted values.
    se_fit : NDArray or None
        Standard errors of predictions (if requested).
    lower : NDArray or None
        Lower bound of interval (if requested).
    upper : NDArray or None
        Upper bound of interval (if requested).
    interval : str
        Type of interval: "none", "confidence", or "prediction".
    level : float
        Confidence level used for intervals.
    """

    fit: NDArray[np.floating]
    se_fit: NDArray[np.floating] | None = None
    lower: NDArray[np.floating] | None = None
    upper: NDArray[np.floating] | None = None
    interval: str = "none"
    level: float = 0.95

    def __array__(self, dtype=None, copy: bool | None = None) -> NDArray:
        """Return predictions as an array while honoring NumPy conversion options."""
        if copy is None:
            return np.asarray(self.fit, dtype=dtype)
        return np.array(self.fit, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        return len(self.fit)

    def __iter__(self):
        return iter(self.fit)

    def __getitem__(self, idx):
        value = self.fit[idx]
        return float(value) if np.ndim(value) == 0 else value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.fit.shape

    @property
    def ndim(self) -> int:
        return self.fit.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.fit.dtype

    @property
    def size(self) -> int:
        return self.fit.size


class LogLik(float):
    """Numeric log-likelihood value with model-selection metadata."""

    df: int
    nobs: int
    REML: bool

    def __new__(
        cls,
        value: float,
        df: int,
        nobs: int,
        REML: bool = False,
    ) -> LogLik:
        instance = super().__new__(cls, value)
        instance.df = df
        instance.nobs = nobs
        instance.REML = REML
        return instance

    @property
    def value(self) -> float:
        """Return the plain numeric value for compatibility."""
        return float(self)

    def __reduce_ex__(
        self,
        _protocol: SupportsIndex,
    ) -> tuple[type[LogLik], tuple[float, int, int, bool]]:
        return type(self), (self.value, self.df, self.nobs, self.REML)

    def __str__(self) -> str:
        reml_str = " (REML)" if self.REML else ""
        return f"'log Lik.' {self.value:.4f} (df={self.df}){reml_str}"

    def __repr__(self) -> str:
        return f"LogLik(value={self.value:.4f}, df={self.df}, nobs={self.nobs}, REML={self.REML})"


@dataclass
class VarCorrGroup:
    name: str
    term_names: list[str]
    variance: dict[str, float]
    stddev: dict[str, float]
    cov: NDArray[np.floating]
    corr: NDArray[np.floating] | None


@dataclass
class ModelTerms:
    response: str
    fixed_terms: list[str]
    random_terms: dict[str, list[str]]
    fixed_variables: set[str]
    random_variables: set[str]
    grouping_factors: set[str]
    has_intercept: bool

    def __str__(self) -> str:
        lines = ["Model terms:"]
        lines.append(f"  Response: {self.response}")
        lines.append(f"  Fixed effects: {', '.join(self.fixed_terms)}")
        for group, terms in self.random_terms.items():
            lines.append(f"  Random effects ({group}): {', '.join(terms)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_fixed = len(self.fixed_terms)
        n_groups = len(self.random_terms)
        return f"ModelTerms({n_fixed} fixed, {n_groups} random groups)"


@dataclass
class RePCAGroup:
    name: str
    n_terms: int
    sdev: NDArray[np.floating]
    proportion: NDArray[np.floating]
    cumulative: NDArray[np.floating]

    def __str__(self) -> str:
        lines = [f"Random effect PCA: {self.name}"]
        lines.append(f"{'Component':<12} {'Std.Dev':>10} {'Prop.Var':>10} {'Cumulative':>10}")
        for i in range(self.n_terms):
            pc_name = f"PC{i + 1}"
            sdev = self.sdev[i]
            prop = self.proportion[i]
            cumul = self.cumulative[i]
            lines.append(f"{pc_name:<12} {sdev:>10.4f} {prop:>10.4f} {cumul:>10.4f}")
        return "\n".join(lines)


@dataclass
class RePCA:
    groups: dict[str, RePCAGroup]

    def __str__(self) -> str:
        lines = []
        for group in self.groups.values():
            lines.append(str(group))
            lines.append("")
        return "\n".join(lines).rstrip()

    def __repr__(self) -> str:
        return f"RePCA({len(self.groups)} groups)"

    def __getitem__(self, key: str) -> RePCAGroup:
        return self.groups[key]

    def is_singular(self, tol: float = 1e-4) -> dict[str, bool]:
        return {name: bool(np.any(group.sdev < tol)) for name, group in self.groups.items()}

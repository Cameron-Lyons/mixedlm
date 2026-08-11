from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import (
    ModelMatrices,
    RandomEffectStructure,
    _normalize_grouped_binomial_response,
)
from mixedlm.models.lmer_types import ModelTerms, RanefResult
from mixedlm.utils.dataframe import (
    concat_columns_as_string,
    copy_dataframe,
    get_column_numpy,
    get_columns,
)

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.inference.profile_types import ProfileResult


class MerResultMixin:
    formula: Formula
    matrices: ModelMatrices
    beta: NDArray[np.floating]
    theta: NDArray[np.floating]
    _IS_GLMM: ClassVar[bool] = False
    _IS_LMM: ClassVar[bool] = False
    _IS_NLMM: ClassVar[bool] = False

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        raise NotImplementedError

    def fixef(self) -> dict[str, float]:
        raise NotImplementedError

    def _compute_condVar(
        self, include_cov: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]]:
        raise NotImplementedError

    def _fixef_dict(self, beta: NDArray[np.floating]) -> dict[str, float]:
        return dict(zip(self.matrices.fixed_names, beta, strict=False))

    def _ranef_values_from_u(
        self, u: NDArray[np.floating]
    ) -> dict[str, dict[str, NDArray[np.floating]]]:
        result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            u_block = u[u_idx : u_idx + n_u].reshape(n_levels, n_terms)
            u_idx += n_u

            term_ranefs: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_ranefs[term_name] = u_block[:, j]

            group_ranefs = result.setdefault(struct.grouping_factor, {})
            for term_name, values in term_ranefs.items():
                if term_name in group_ranefs:
                    group_ranefs[term_name] = group_ranefs[term_name] + values
                else:
                    group_ranefs[term_name] = values

        return result

    def _ranef_with_optional_condvar(
        self,
        u: NDArray[np.floating],
        condVar: bool,
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        values = self._ranef_values_from_u(u)
        if not condVar:
            return values
        cond_var = self._compute_condVar()
        return RanefResult(values=values, condVar=cond_var)

    def _model_matrix(
        self, type: str = "fixed"
    ) -> NDArray[np.floating] | sparse.csc_matrix | tuple[NDArray[np.floating], sparse.csc_matrix]:
        if type in ("fixed", "X"):
            return self.matrices.X
        if type in ("random", "Z"):
            return self.matrices.Z
        if type == "both":
            return (self.matrices.X, self.matrices.Z)
        raise ValueError(f"Unknown type '{type}'. Use 'fixed', 'random', 'X', 'Z', or 'both'.")

    def _prediction_fixed_matrix(self, newdata: pd.DataFrame) -> NDArray[np.floating]:
        """Build a fixed-effects matrix using the fitted encoding schema."""
        import pandas as pd

        from mixedlm.matrices.design import build_fixed_matrix
        from mixedlm.utils.dataframe import ensure_dataframe, get_column_numpy, get_columns

        data = ensure_dataframe(newdata)
        columns = set(get_columns(data))
        missing = sorted(self.formula.fixed_variables - columns)
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(f"New data is missing fixed-effect variable(s): {names}.")

        for name, fitted_levels in self.matrices.category_levels.items():
            if name not in self.formula.fixed_variables or name not in columns:
                continue
            unknown_levels: list[object] = []
            for value in get_column_numpy(data, name):
                if pd.isna(value) or value in fitted_levels or value in unknown_levels:
                    continue
                unknown_levels.append(value)
            if unknown_levels:
                levels = ", ".join(repr(value) for value in unknown_levels)
                raise ValueError(f"New level(s) {levels} in fixed-effect factor '{name}'.")

        X, fixed_names = build_fixed_matrix(
            self.formula,
            data,
            contrasts=self.matrices.contrasts,
            category_levels=self.matrices.category_levels,
        )
        column_indices = {name: index for index, name in enumerate(fixed_names)}
        try:
            fitted_indices = [column_indices[name] for name in self.matrices.fixed_names]
        except KeyError as exc:
            raise ValueError(
                f"New data is missing fitted fixed-effect column '{exc.args[0]}'."
            ) from None
        return X[:, fitted_indices]

    def _model_frame(self) -> Any:
        import pandas as pd

        if self.matrices.frame is not None:
            return copy_dataframe(self.matrices.frame)
        return pd.DataFrame({"y": self.matrices.y})

    def _weights_array(self, copy: bool = True) -> NDArray[np.floating]:
        return self.matrices.weights.copy() if copy else self.matrices.weights

    def _offset_array(self, copy: bool = True) -> NDArray[np.floating]:
        return self.matrices.offset.copy() if copy else self.matrices.offset

    def _should_expand_na(self) -> bool:
        from mixedlm.utils.na_action import NAAction

        return (
            self.matrices.na_info is not None
            and self.matrices.na_info.action == NAAction.EXCLUDE
            and self.matrices.na_info.n_omitted > 0
        )

    def _build_model_terms(self, formula: Formula) -> ModelTerms:
        from mixedlm.formula.terms import InteractionTerm, PowerTerm, VariableTerm

        fixed_terms = list(self.matrices.fixed_names)

        random_terms: dict[str, list[str]] = {}
        for struct in self.matrices.random_structures:
            group_terms = random_terms.setdefault(struct.grouping_factor, [])
            for term_name in struct.term_names:
                if term_name not in group_terms:
                    group_terms.append(term_name)

        fixed_variables: set[str] = set()
        for term in formula.fixed.terms:
            if isinstance(term, VariableTerm | PowerTerm):
                fixed_variables.add(term.name)
            elif isinstance(term, InteractionTerm):
                fixed_variables.update(term.source_variables)

        random_variables: set[str] = set()
        for rterm in formula.random:
            for term in rterm.expr:
                if isinstance(term, VariableTerm | PowerTerm):
                    random_variables.add(term.name)
                elif isinstance(term, InteractionTerm):
                    random_variables.update(term.source_variables)

        grouping_factors = {struct.grouping_factor for struct in self.matrices.random_structures}

        return ModelTerms(
            response=formula.response,
            fixed_terms=fixed_terms,
            random_terms=random_terms,
            fixed_variables=fixed_variables,
            random_variables=random_variables,
            grouping_factors=grouping_factors,
            has_intercept=formula.fixed.has_intercept,
        )

    def _condvar_from_cov(
        self, cond_cov: NDArray[np.floating]
    ) -> dict[str, dict[str, NDArray[np.floating]]]:
        term_indices: dict[str, dict[str, list[NDArray[np.integer]]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms
            for j, term_name in enumerate(struct.term_names):
                indices = u_idx + np.arange(n_levels) * n_terms + j
                group_indices = term_indices.setdefault(struct.grouping_factor, {})
                group_indices.setdefault(term_name, []).append(indices)
            u_idx += n_u

        result: dict[str, dict[str, NDArray[np.floating]]] = {}
        for group_name, group_terms in term_indices.items():
            term_vars: dict[str, NDArray[np.floating]] = {}
            for term_name, index_parts in group_terms.items():
                if len(index_parts) == 1:
                    single_indices = np.asarray(index_parts[0], dtype=np.intp)
                    term_vars[term_name] = cond_cov[single_indices, single_indices]
                    continue

                stacked_indices = np.vstack(index_parts)
                variances = np.empty(stacked_indices.shape[1], dtype=np.float64)
                for level_index, level_indices in enumerate(stacked_indices.T):
                    variances[level_index] = np.sum(cond_cov[np.ix_(level_indices, level_indices)])
                term_vars[term_name] = variances
            result[group_name] = term_vars

        return result

    def _iter_random_cov_blocks(
        self, scale: float = 1.0
    ) -> Iterator[tuple[RandomEffectStructure, NDArray[np.floating]]]:
        from mixedlm.utils.variance import getL

        level_factors = cast(
            list[NDArray[np.floating]],
            getL(self.theta, self.matrices.random_structures, as_blocks=True),
        )

        for struct, level_factor in zip(
            self.matrices.random_structures, level_factors, strict=True
        ):
            cov = level_factor @ level_factor.T

            yield struct, cov * scale

    def _is_singular_covariance(self, tol: float = 1e-4) -> bool:
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite, non-negative number")

        threshold = tol**2
        for _struct, cov in self._iter_random_cov_blocks():
            if float(np.min(np.linalg.eigvalsh(cov), initial=np.inf)) < threshold:
                return True
        return False

    def _theta_lower_bounds(self) -> NDArray[np.floating]:
        from mixedlm.estimation.reml import _build_theta_bounds

        bounds = _build_theta_bounds(self.matrices.random_structures, len(self.theta))
        return np.asarray(
            [lower if lower is not None else -np.inf for lower, _upper in bounds],
            dtype=np.float64,
        )

    def _random_effect_prediction_contrib(
        self,
        newdata: Any,
        allow_new_levels: bool,
        u: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        import pandas as pd

        n = len(newdata)
        contrib = np.zeros(n, dtype=np.float64)
        u_idx = 0
        columns = set(get_columns(newdata))

        for struct in self.matrices.random_structures:
            group_col = struct.grouping_factor
            n_terms = struct.n_terms
            n_levels = struct.n_levels
            n_u = n_levels * n_terms

            if group_col in columns:
                group_values = get_column_numpy(newdata, group_col)
            else:
                nested_cols = group_col.split("/")
                if len(nested_cols) < 2 or not all(col in columns for col in nested_cols):
                    u_idx += n_u
                    continue
                group_values = concat_columns_as_string(newdata, nested_cols)

            string_level_map = {str(level): idx for level, idx in struct.level_map.items()}
            group_series = pd.Series(group_values, copy=False)
            mapped = group_series.map(struct.level_map)
            if mapped.isna().any():
                mapped = mapped.fillna(group_series.astype(str).map(string_level_map))
            mapped_levels = mapped.fillna(-1).to_numpy(dtype=np.int64)
            known_mask = mapped_levels >= 0

            if not allow_new_levels and not np.all(known_mask):
                unknown_level = str(group_values[np.flatnonzero(~known_mask)[0]])
                raise ValueError(
                    f"New level '{unknown_level}' in grouping factor '{group_col}'. "
                    "Set allow_new_levels=True to predict with random effects = 0."
                )

            if not np.any(known_mask):
                u_idx += n_u
                continue

            level_idx = mapped_levels[known_mask]
            u_block = u[u_idx : u_idx + n_u].reshape(n_levels, n_terms)
            u_idx += n_u

            block_contrib = np.zeros(level_idx.shape[0], dtype=np.float64)
            for i, term_name in enumerate(struct.term_names):
                if term_name == "(Intercept)":
                    term_values = np.ones(n, dtype=np.float64)
                elif term_name in columns:
                    term_values = get_column_numpy(newdata, term_name, dtype=np.float64)
                else:
                    continue

                block_contrib += u_block[level_idx, i] * term_values[known_mask]

            contrib[known_mask] += block_contrib

        return contrib

    def _coerce_new_response(self, newresp: NDArray[np.floating] | None) -> NDArray[np.floating]:
        if newresp is None:
            return self.matrices.y

        arr = np.asarray(newresp, dtype=np.float64)
        if len(arr) != self.matrices.n_obs:
            raise ValueError(f"newresp has length {len(arr)}, expected {self.matrices.n_obs}")
        if self.matrices.trials is not None:
            return _normalize_grouped_binomial_response(arr, self.matrices.trials)
        return arr

    def _clone_matrices_with_response_base(self, y: NDArray[np.floating]) -> ModelMatrices:
        return ModelMatrices(
            y=y,
            X=self.matrices.X,
            Z=self.matrices.Z,
            fixed_names=self.matrices.fixed_names,
            random_structures=self.matrices.random_structures,
            n_obs=self.matrices.n_obs,
            n_fixed=self.matrices.n_fixed,
            n_random=self.matrices.n_random,
            weights=self.matrices.weights,
            offset=self.matrices.offset,
            frame=self.matrices.frame,
            na_info=self.matrices.na_info,
            trials=self.matrices.trials,
            category_levels=self.matrices.category_levels,
            contrasts=self.matrices.contrasts,
        )

    def coef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        ranef_result = self.ranef()
        ranefs = ranef_result.values if isinstance(ranef_result, RanefResult) else ranef_result
        fixefs = self.fixef()
        result: dict[str, dict[str, NDArray[np.floating]]] = {}

        random_only_terms: list[str] = []
        for terms in ranefs.values():
            for term_name in terms:
                if term_name not in fixefs and term_name not in random_only_terms:
                    random_only_terms.append(term_name)

        n_levels_by_group = {
            struct.grouping_factor: struct.n_levels for struct in self.matrices.random_structures
        }
        for group, terms in ranefs.items():
            n_levels = n_levels_by_group[group]
            group_coef: dict[str, NDArray[np.floating]] = {
                term_name: np.zeros(n_levels, dtype=np.float64) for term_name in random_only_terms
            }
            group_coef.update(
                {
                    term_name: np.full(n_levels, value, dtype=np.float64)
                    for term_name, value in fixefs.items()
                }
            )
            for term_name, ranef_vals in terms.items():
                group_coef[term_name] += ranef_vals
            result[group] = group_coef

        return result

    def nobs(self) -> int:
        return self.matrices.n_obs

    def ngrps(self) -> dict[str, int]:
        return {
            struct.grouping_factor: struct.n_levels for struct in self.matrices.random_structures
        }

    def df_residual(self) -> int:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        return n - p

    def isREML(self) -> bool:
        return bool(getattr(self, "REML", False))

    def isGLMM(self) -> bool:
        return self._IS_GLMM

    def isLMM(self) -> bool:
        return self._IS_LMM

    def isNLMM(self) -> bool:
        return self._IS_NLMM

    def isSingular(self, tol: float = 1e-4) -> bool:
        raise NotImplementedError

    def is_singular(self, tol: float = 1e-4) -> bool:
        """Return whether any variance component is near its boundary."""
        return self.isSingular(tol=tol)

    def profile(
        self,
        which: str | list[str] | None = None,
        n_points: int = 20,
        level: float = 0.95,
        n_jobs: int = 1,
    ) -> dict[str, ProfileResult]:
        """Compute fixed-effect likelihood profiles for this fitted model."""
        from mixedlm.inference.profile import profile_glmer, profile_lmer
        from mixedlm.models.glmer import GlmerResult
        from mixedlm.models.lmer import LmerResult

        if isinstance(self, LmerResult):
            return profile_lmer(
                self,
                which=which,
                n_points=n_points,
                level=level,
                n_jobs=n_jobs,
            )
        if isinstance(self, GlmerResult):
            if n_jobs != 1:
                raise ValueError("Parallel profiling is currently supported only for LmerResult")
            return profile_glmer(self, which=which, n_points=n_points, level=level)
        raise TypeError(f"Profiling is not supported for {type(self).__name__}")

    def _npar_count(self, include_sigma: bool = False) -> int:
        n = len(self.beta) + len(self.theta)
        if include_sigma:
            n += 1
        return n

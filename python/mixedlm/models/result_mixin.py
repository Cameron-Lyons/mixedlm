from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
from mixedlm.models.lmer_types import ModelTerms, RanefResult

if TYPE_CHECKING:
    import pandas as pd


class MerResultMixin:
    matrices: ModelMatrices
    beta: NDArray[np.floating]
    theta: NDArray[np.floating]
    _IS_GLMM: bool = False
    _IS_LMM: bool = False
    _IS_NLMM: bool = False

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        raise NotImplementedError

    def fixef(self) -> dict[str, float]:
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

            result[struct.grouping_factor] = term_ranefs

        return result

    def _ranef_with_optional_condvar(
        self,
        u: NDArray[np.floating],
        condVar: bool,
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        values = self._ranef_values_from_u(u)
        if not condVar:
            return values
        cond_var = self._compute_condVar()  # type: ignore[attr-defined]
        return RanefResult(values=values, condVar=cond_var)

    def _model_matrix(
        self, type: str = "fixed"
    ) -> (
        NDArray[np.floating]
        | sparse.csc_matrix
        | tuple[NDArray[np.floating], sparse.csc_matrix]
    ):
        if type in ("fixed", "X"):
            return self.matrices.X
        if type in ("random", "Z"):
            return self.matrices.Z
        if type == "both":
            return (self.matrices.X, self.matrices.Z)
        raise ValueError(f"Unknown type '{type}'. Use 'fixed', 'random', 'X', 'Z', or 'both'.")

    def _model_frame(self) -> pd.DataFrame:
        import pandas as pd

        if self.matrices.frame is not None:
            return self.matrices.frame.copy()
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
        from mixedlm.formula.terms import InteractionTerm, VariableTerm

        fixed_terms = list(self.matrices.fixed_names)

        random_terms: dict[str, list[str]] = {}
        for struct in self.matrices.random_structures:
            random_terms[struct.grouping_factor] = list(struct.term_names)

        fixed_variables: set[str] = set()
        for term in formula.fixed.terms:
            if isinstance(term, VariableTerm):
                fixed_variables.add(term.name)
            elif isinstance(term, InteractionTerm):
                fixed_variables.update(term.variables)

        random_variables: set[str] = set()
        for rterm in formula.random:
            for term in rterm.expr:
                if isinstance(term, VariableTerm):
                    random_variables.add(term.name)
                elif isinstance(term, InteractionTerm):
                    random_variables.update(term.variables)

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
        result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            block_cov = cond_cov[u_idx : u_idx + n_u, u_idx : u_idx + n_u]
            block_diag = np.diag(block_cov).reshape(n_levels, n_terms)

            term_vars: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_vars[term_name] = block_diag[:, j]

            result[struct.grouping_factor] = term_vars
            u_idx += n_u

        return result

    def _iter_random_cov_blocks(
        self, scale: float = 1.0
    ) -> Iterator[tuple[RandomEffectStructure, NDArray[np.floating]]]:
        theta_idx = 0

        for struct in self.matrices.random_structures:
            q = struct.n_terms

            if struct.correlated:
                n_theta = q * (q + 1) // 2
                theta_block = self.theta[theta_idx : theta_idx + n_theta]
                theta_idx += n_theta

                L_block = np.zeros((q, q), dtype=np.float64)
                idx = 0
                for i in range(q):
                    row_size = i + 1
                    L_block[i, :row_size] = theta_block[idx : idx + row_size]
                    idx += row_size

                cov = L_block @ L_block.T
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q
                cov = np.diag(theta_block**2)

            yield struct, cov * scale

    def _random_effect_prediction_contrib(
        self,
        newdata: pd.DataFrame,
        allow_new_levels: bool,
        u: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        n = len(newdata)
        contrib = np.zeros(n, dtype=np.float64)
        u_idx = 0

        for struct in self.matrices.random_structures:
            group_col = struct.grouping_factor
            n_terms = struct.n_terms
            n_levels = struct.n_levels
            n_u = n_levels * n_terms

            if group_col not in newdata.columns:
                u_idx += n_u
                continue

            group_values = newdata[group_col].astype(str)
            mapped_levels = group_values.map(struct.level_map)
            known_mask = mapped_levels.notna().to_numpy()

            if not allow_new_levels and not np.all(known_mask):
                unknown_level = str(group_values[~known_mask].iloc[0])
                raise ValueError(
                    f"New level '{unknown_level}' in grouping factor '{group_col}'. "
                    "Set allow_new_levels=True to predict with random effects = 0."
                )

            if not np.any(known_mask):
                u_idx += n_u
                continue

            level_idx = mapped_levels[known_mask].astype(int).to_numpy()
            u_block = u[u_idx : u_idx + n_u].reshape(n_levels, n_terms)
            u_idx += n_u

            block_contrib = np.zeros(level_idx.shape[0], dtype=np.float64)
            for i, term_name in enumerate(struct.term_names):
                if term_name == "(Intercept)":
                    term_values = np.ones(n, dtype=np.float64)
                elif term_name in newdata.columns:
                    term_values = np.asarray(newdata[term_name], dtype=np.float64)
                else:
                    continue

                block_contrib += u_block[level_idx, i] * term_values[known_mask]

            contrib[known_mask] += block_contrib

        return contrib

    def _coerce_new_response(
        self, newresp: NDArray[np.floating] | None
    ) -> NDArray[np.floating]:
        if newresp is None:
            return self.matrices.y

        arr = np.asarray(newresp, dtype=np.float64)
        if len(arr) != self.matrices.n_obs:
            raise ValueError(f"newresp has length {len(arr)}, expected {self.matrices.n_obs}")
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
        )

    def coef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        ranefs = self.ranef()
        fixefs = self.fixef()
        result: dict[str, dict[str, NDArray[np.floating]]] = {}

        for group, terms in ranefs.items():
            group_coef: dict[str, NDArray[np.floating]] = {}
            for term_name, ranef_vals in terms.items():
                if term_name in fixefs:
                    group_coef[term_name] = ranef_vals + fixefs[term_name]
                else:
                    group_coef[term_name] = ranef_vals
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

    def _npar_count(self, include_sigma: bool = False) -> int:
        n = len(self.beta) + len(self.theta)
        if include_sigma:
            n += 1
        return n

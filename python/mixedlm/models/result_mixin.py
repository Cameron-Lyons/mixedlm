from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import (
    ModelMatrices,
    RandomEffectStructure,
    _encode_binomial_factor,
    _normalize_grouped_binomial_response,
    _numeric_response,
    _random_term_columns,
    _restore_binomial_factor,
    build_random_matrix,
)
from mixedlm.models.lmer_types import ModelTerms, RanefResult, RePCA, RePCAGroup, VarCorrGroup
from mixedlm.models.shared_utils import sparse_covariance_factor_diagonal
from mixedlm.utils.dataframe import (
    concat_columns_as_string,
    copy_dataframe,
    dataframe_length,
    ensure_dataframe,
    get_column_numpy,
    get_columns,
)
from mixedlm.utils.random import RandomSeed, native_seed, random_stream, validate_simulation_count
from mixedlm.utils.simulation import simulate_random_effects, simulation_parameters
from mixedlm.utils.validation import _validate_confidence_level

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from mixedlm.inference.bootstrap import BootstrapResult
    from mixedlm.inference.profile_types import ProfileResult
    from mixedlm.inference.reporting import MixedModelResult
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult
    from mixedlm.models.lmer_types import LogLik

_Result = TypeVar("_Result", bound="MerResultMixin")


class _ResultBase:
    """Accessors shared by every fitted mixed-model result."""

    _IS_GLMM: ClassVar[bool] = False
    _IS_LMM: ClassVar[bool] = False
    _IS_NLMM: ClassVar[bool] = False

    if TYPE_CHECKING:

        def logLik(self) -> LogLik: ...

        def isSingular(self, tol: float = 1e-4) -> bool: ...

        def summary(self) -> str: ...

    def isGLMM(self) -> bool:
        return self._IS_GLMM

    def isLMM(self) -> bool:
        return self._IS_LMM

    def isNLMM(self) -> bool:
        return self._IS_NLMM

    def is_singular(self, tol: float = 1e-4) -> bool:
        """Return whether any variance component is near its boundary."""
        return self.isSingular(tol=tol)

    def AIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + 2 * ll.df

    def BIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + ll.df * np.log(ll.nobs)

    def extractAIC(self) -> tuple[float, float]:
        """Extract AIC with effective degrees of freedom.

        Returns
        -------
        tuple of (float, float)
            ``(edf, AIC)``, matching the interface of R's ``extractAIC``.
        """
        return (float(self.logLik().df), float(self.AIC()))

    def tidy(
        self,
        effects: str | Sequence[str] = "fixed",
        *,
        conf_int: bool = False,
        conf_level: float = 0.95,
        ddf_method: str | None = "Satterthwaite",
    ) -> pd.DataFrame:
        """Return model components in an analysis-ready table."""
        from mixedlm.inference.reporting import tidy

        return tidy(
            cast("MixedModelResult", self),
            effects=effects,
            conf_int=conf_int,
            conf_level=conf_level,
            ddf_method=ddf_method,
        )

    def glance(self) -> pd.DataFrame:
        """Return one row of model-level fit statistics."""
        from mixedlm.inference.reporting import glance

        return glance(cast("MixedModelResult", self))

    def __str__(self) -> str:
        return self.summary()


class MerResultMixin(_ResultBase):
    """Methods shared by fitted linear and generalized linear mixed models.

    Subclasses provide the fitted state and a few model-specific hooks:
    ``sigma``, ``_hat_values``, ``_compute_condVar``, ``_compute_RX``,
    ``_compute_RZX``, ``_devcomp_cmp``, ``_simulate_from_eta``,
    ``_refit_from_matrices``, ``profile`` and ``_bootstrap``.
    """

    formula: Formula
    matrices: ModelMatrices
    beta: NDArray[np.floating]
    theta: NDArray[np.floating]
    # Conditional modes b = Lambda u of the random effects (not the spherical u).
    u: NDArray[np.floating]
    deviance: float
    # Whether a residual scale sigma is estimated and counted as a parameter.
    _HAS_SIGMA: ClassVar[bool] = False

    if TYPE_CHECKING:

        @property
        def sigma(self) -> float: ...

        @property
        def _hat_values(self) -> NDArray[np.float64]: ...

        def _compute_condVar(
            self, include_cov: bool = False
        ) -> dict[str, dict[str, NDArray[np.floating]]]: ...

        def _compute_RX(self) -> NDArray[np.floating]: ...

        def _compute_RZX(self) -> NDArray[np.floating]: ...

        def _devcomp_cmp(self) -> dict[str, float]: ...

        def _simulate_from_eta(
            self, eta: NDArray[np.floating], rng: Any
        ) -> NDArray[np.floating]: ...

        def _refit_from_matrices(
            self: _Result, matrices: ModelMatrices, **kwargs: Any
        ) -> _Result: ...

        def profile(
            self,
            which: str | list[str] | None = None,
            n_points: int = 20,
            level: float = 0.95,
            n_jobs: int = 1,
        ) -> dict[str, ProfileResult]: ...

        def _bootstrap(self, n_boot: int, seed: RandomSeed) -> BootstrapResult: ...

        def vcov(self) -> NDArray[np.floating]: ...

    def fixef(self) -> dict[str, float]:
        return self._fixef_dict(self.beta)

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        return self._ranef_with_optional_condvar(self.u, condVar)

    def _fixef_dict(self, beta: NDArray[np.floating]) -> dict[str, float]:
        from mixedlm.utils.names import _check_unique_coefficient_names

        _check_unique_coefficient_names(
            self.matrices.fixed_names,
            alternative="Use tidy() or beta to inspect coefficients in fitted column order.",
        )
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

    def model_matrix(
        self, type: str = "fixed"
    ) -> NDArray[np.floating] | sparse.csc_matrix | tuple[NDArray[np.floating], sparse.csc_matrix]:
        """Get the model design matrix.

        Parameters
        ----------
        type : str, default "fixed"
            Which design matrix to return:
            - "fixed" or "X": Fixed effects design matrix
            - "random" or "Z": Random effects design matrix (sparse)
            - "both": Tuple of (X, Z)

        Returns
        -------
        NDArray or sparse.csc_matrix or tuple
            The requested design matrix. X is dense, Z is sparse.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> X = result.model_matrix("fixed")
        >>> Z = result.model_matrix("random")
        >>> X, Z = result.model_matrix("both")
        """
        if type in ("fixed", "X"):
            return self.matrices.X
        if type in ("random", "Z"):
            return self.matrices.Z
        if type == "both":
            return (self.matrices.X, self.matrices.Z)
        raise ValueError(f"Unknown type '{type}'. Use 'fixed', 'random', 'X', 'Z', or 'both'.")

    def _prepare_prediction_data(
        self,
        newdata: Any,
        *,
        include_re: bool,
        extra_columns: Sequence[str] = (),
    ) -> Any:
        """Collect a lazy prediction query once, projecting to relevant columns."""
        from mixedlm.utils.dataframe import _is_polars_lazy, ensure_dataframe

        if not _is_polars_lazy(newdata):
            return ensure_dataframe(newdata)

        variables = self.formula.fixed_variables | set(extra_columns)
        if include_re:
            variables |= self.formula.random_variables | self.formula.grouping_factors
            # Preserve direct nested-group keys and encoded slope columns accepted
            # by the existing random-effect prediction path.
            for structure in self.matrices.random_structures:
                variables.add(structure.grouping_factor)
                variables.update(name for name in structure.term_names if name != "(Intercept)")
        columns = [name for name in get_columns(newdata) if name in variables]
        if columns:
            return ensure_dataframe(newdata, columns=columns)

        import pandas as pd
        import polars as pl

        # An intercept-only grid still needs its row count. Polars select([])
        # loses that count; an index-only pandas frame retains it without a column.
        count = pl.len() if hasattr(pl, "len") else pl.count()
        n_rows = int(newdata.select(count).collect().item())
        return pd.DataFrame(index=pd.RangeIndex(n_rows))

    def _validated_prediction_data(self, newdata: Any, variables: set[str], kind: str) -> Any:
        """Validate required columns and reuse the fitted predictor categories."""
        import pandas as pd

        data = ensure_dataframe(newdata)
        columns = set(get_columns(data))
        missing = sorted(variables - columns)
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(f"New data is missing {kind} variable(s): {names}.")
        if kind == "random-effect":
            missing_groups = sorted(
                name
                for name in self.formula.grouping_factors
                if pd.isna(get_column_numpy(data, name)).any()
            )
            if missing_groups:
                names = ", ".join(repr(name) for name in missing_groups)
                raise ValueError(
                    f"New data contains missing values in grouping factor(s): {names}."
                )

        for name, fitted_levels in self.matrices.category_levels.items():
            if name not in variables:
                continue
            values = pd.Index(get_column_numpy(data, name)).dropna().unique()
            unknown_levels = values[~values.isin(fitted_levels)]
            if len(unknown_levels):
                levels = ", ".join(repr(value) for value in unknown_levels)
                raise ValueError(f"New level(s) {levels} in {kind} factor '{name}'.")
        return data

    def _prediction_fixed_matrix(
        self,
        newdata: pd.DataFrame,
        *,
        contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    ) -> NDArray[np.floating]:
        """Build a fixed-effects matrix using the fitted encoding schema."""
        from mixedlm.matrices.design import build_fixed_matrix

        data = self._validated_prediction_data(
            newdata, self.formula.fixed_variables, "fixed-effect"
        )
        X, fixed_names = build_fixed_matrix(
            self.formula,
            data,
            contrasts=self.matrices.contrasts if contrasts is None else contrasts,
            category_levels=self.matrices.category_levels,
        )
        return self._align_prediction_columns(X, fixed_names)

    def _align_prediction_columns(
        self, X: NDArray[np.floating], fixed_names: list[str]
    ) -> NDArray[np.floating]:
        """Align by fitted positions when available, without copying an aligned matrix."""
        fitted_names = self.matrices.fixed_names
        source_indices = self.matrices.fixed_column_indices
        fitted_indices: Sequence[int]
        if source_indices is not None and self.matrices.fixed_source_names == tuple(fixed_names):
            if (
                any(index < 0 or index >= len(fixed_names) for index in source_indices)
                or [fixed_names[index] for index in source_indices] != fitted_names
            ):
                raise ValueError(
                    "Fitted fixed-effect column positions do not match the model schema."
                )
            fitted_indices = source_indices
        elif source_indices is None and fixed_names == fitted_names:
            return X
        else:
            column_indices = {}
            ambiguous = set()
            for index, name in enumerate(fixed_names):
                if name in column_indices:
                    ambiguous.add(name)
                column_indices[name] = index
            try:
                fitted_indices = [column_indices[name] for name in fitted_names]
            except KeyError as exc:
                raise ValueError(
                    f"New data is missing fitted fixed-effect column '{exc.args[0]}'."
                ) from None
            requested = set()
            for name in fitted_names:
                if name in ambiguous or name in requested:
                    raise ValueError(
                        f"Cannot align ambiguous fixed-effect column name '{name}' with the fitted "
                        "model. Refit the model or use the fitted contrast schema."
                    )
                requested.add(name)
        if tuple(fitted_indices) == tuple(range(X.shape[1])):
            return X
        return X[:, fitted_indices]

    def _prediction_offset(
        self,
        newdata: pd.DataFrame,
        offset: ArrayLike | str | None,
    ) -> NDArray[np.floating]:
        return self._prediction_vector(newdata, offset, name="offset", default=0.0)

    def _prediction_vector(
        self,
        newdata: pd.DataFrame,
        value: ArrayLike | str | None,
        *,
        name: str,
        default: float,
    ) -> NDArray[np.float64]:
        """Resolve and validate a scalar, array, or named column for prediction rows."""
        from mixedlm.models.shared_utils import resolve_prediction_vector

        return resolve_prediction_vector(newdata, value, name=name, default=default)

    def model_frame(self) -> Any:
        """Get the model frame.

        Returns the data frame containing only the variables used
        in the model formula, after any NA handling.

        Returns
        -------
        DataFrame
            Data frame with the response variable, fixed effect variables,
            and grouping factors. The fitted input's backend is preserved.

        Examples
        --------
        >>> result = lmer("y ~ x + (1 | group)", data)
        >>> mf = result.model_frame()
        >>> print(mf.columns.tolist())  # ['y', 'x', 'group']
        """
        import pandas as pd

        if self.matrices.frame is not None:
            return copy_dataframe(self.matrices.frame)
        return pd.DataFrame({"y": self.matrices.y})

    def weights(self, copy: bool = True) -> NDArray[np.floating]:
        """Get the prior weights used in model fitting.

        If no weights were specified, returns an array of ones. In a linear
        mixed model the residual variance of observation ``i`` is
        ``sigma**2 / weights[i]``.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy of the weights array.
            If False, return the original array (faster but should not be modified).

        Returns
        -------
        NDArray
            Array of weights with length equal to number of observations.
        """
        return self.matrices.weights.copy() if copy else self.matrices.weights

    def offset(self, copy: bool = True) -> NDArray[np.floating]:
        """Get the offset used in model fitting.

        If no offset was specified, returns an array of zeros.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy of the offset array.
            If False, return the original array (faster but should not be modified).

        Returns
        -------
        NDArray
            Array of offsets with length equal to number of observations.
        """
        return self.matrices.offset.copy() if copy else self.matrices.offset

    def _should_expand_na(self) -> bool:
        from mixedlm.utils.na_action import NAAction

        return (
            self.matrices.na_info is not None
            and self.matrices.na_info.action == NAAction.EXCLUDE
            and self.matrices.na_info.n_omitted > 0
        )

    def terms(self) -> ModelTerms:
        """Get information about the model terms.

        Returns
        -------
        ModelTerms
            Object containing:
            - response: Name of the response variable
            - fixed_terms: List of fixed effect term names
            - random_terms: Dict mapping grouping factors to their term names
            - fixed_variables: Set of variables in fixed effects
            - random_variables: Set of variables in random effects
            - grouping_factors: Set of grouping factor names
            - has_intercept: Whether the model has an intercept

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> t = result.terms()
        >>> print(t.response)  # 'y'
        >>> print(t.fixed_terms)  # ['(Intercept)', 'x']
        >>> print(t.random_terms)  # {'group': ['(Intercept)', 'x']}
        """
        from mixedlm.formula.terms import InteractionTerm, PowerTerm, VariableTerm

        formula = self.formula
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

    def _varcorr_groups(self, scale: float) -> dict[str, VarCorrGroup]:
        """Report every covariance block under a unique, stable name."""
        from mixedlm.utils.variance import _covariance_block_names, cov2sdcor

        names = _covariance_block_names(self.matrices.random_structures)
        groups: dict[str, VarCorrGroup] = {}
        for name, (struct, cov) in zip(
            names, self._iter_random_cov_blocks(scale=scale), strict=True
        ):
            variances = np.diag(cov)
            if struct.correlated or struct.cov_type in ("cs", "ar1"):
                stddevs, corr = cov2sdcor(cov)
            else:
                stddevs = np.sqrt(variances)
                corr = None
            terms = list(struct.term_names)
            groups[name] = VarCorrGroup(
                name=name,
                term_names=terms,
                variance=dict(zip(terms, variances, strict=True)),
                stddev=dict(zip(terms, stddevs, strict=True)),
                cov=cov,
                corr=corr,
                grouping_factor=struct.grouping_factor,
            )
        return groups

    def rePCA(self) -> RePCA:
        """Perform PCA on the random effects covariance matrix.

        This function computes principal component analysis on the covariance
        matrix of each random effect grouping factor. It's useful for diagnosing
        overparameterization in the random effects structure. Block spectra are
        combined without constructing a larger covariance matrix.

        Returns
        -------
        RePCA
            Object containing PCA results for each random effect group,
            including standard deviations, proportion of variance, and
            cumulative proportion for each principal component.

        Notes
        -----
        If any principal component has very small standard deviation (< 1e-4),
        this suggests the random effects structure may be overparameterized
        (singular or near-singular). Use the `is_singular()` method on the
        result to check for this condition.

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> pca = result.rePCA()
        >>> print(pca)
        >>> pca.is_singular()  # Check if any components are near-zero
        """
        spectra: dict[str, list[NDArray[np.floating]]] = {}
        for struct, cov in self._iter_random_cov_blocks(scale=self.sigma**2):
            spectra.setdefault(struct.grouping_factor, []).append(linalg.eigvalsh(cov))

        groups: dict[str, RePCAGroup] = {}
        for name, blocks in spectra.items():
            eigenvalues = np.maximum(np.sort(np.concatenate(blocks))[::-1], 0.0)
            total_var = np.sum(eigenvalues)
            proportion = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
            groups[name] = RePCAGroup(
                name=name,
                n_terms=len(eigenvalues),
                sdev=np.sqrt(eigenvalues),
                proportion=proportion,
                cumulative=np.cumsum(proportion),
            )
        return RePCA(groups=groups)

    def isSingular(self, tol: float = 1e-4) -> bool:
        """Return whether any random-effect covariance block is near singular."""
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

        data = self._validated_prediction_data(
            newdata,
            self.formula.random_variables | self.formula.grouping_factors,
            "random-effect",
        )
        n = dataframe_length(data)
        contrib = np.zeros(n, dtype=np.float64)
        u_idx = 0
        for rterm, struct in zip(self.formula.random, self.matrices.random_structures, strict=True):
            group_col = struct.grouping_factor
            n_terms = struct.n_terms
            n_levels = struct.n_levels
            n_u = n_levels * n_terms

            if rterm.is_nested:
                group_values = concat_columns_as_string(
                    data, list(rterm.grouping_factors), escape=True
                )
            else:
                assert isinstance(rterm.grouping, str)
                group_values = get_column_numpy(data, rterm.grouping)

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

            term_columns, term_names = _random_term_columns(
                rterm,
                data,
                n,
                self.matrices.contrasts,
                self.matrices.category_levels,
            )
            if term_names != struct.term_names:
                raise ValueError(
                    f"New data produced incompatible random-effect columns for '{group_col}'"
                )

            block_contrib = np.zeros(level_idx.shape[0], dtype=np.float64)
            for i, term_values in enumerate(term_columns):
                block_contrib += u_block[level_idx, i] * term_values[known_mask]

            contrib[known_mask] += block_contrib

        return contrib

    def _prediction_random_matrix(
        self,
        newdata: Any,
        allow_new_levels: bool,
        *,
        scale: float = 1.0,
    ) -> tuple[sparse.csr_matrix, NDArray[np.floating]]:
        """Align new-data random-effect columns to the fitted coefficient order."""
        fitted_structures = self.matrices.random_structures
        data = self._validated_prediction_data(
            newdata,
            self.formula.random_variables | self.formula.grouping_factors,
            "random-effect",
        )
        Z, new_structures = build_random_matrix(
            self.formula,
            data,
            contrasts=self.matrices.contrasts,
            category_levels=self.matrices.category_levels,
        )
        n_pred = Z.shape[0]
        if len(new_structures) != len(fitted_structures):
            raise ValueError("New data produced an incompatible random-effects structure")

        from mixedlm.utils.variance import getL

        level_factors = cast(
            list[NDArray[np.floating]],
            getL(self.theta, fitted_structures, sigma=scale, as_blocks=True),
        )

        known_rows: list[NDArray[np.integer]] = []
        known_cols: list[NDArray[np.integer]] = []
        known_values: list[NDArray[np.floating]] = []
        prior_var = np.zeros(n_pred, dtype=np.float64)
        fitted_offset = 0
        new_offset = 0

        for fitted, new, level_factor in zip(
            fitted_structures, new_structures, level_factors, strict=True
        ):
            if fitted.grouping_factor != new.grouping_factor:
                raise ValueError("New data produced an incompatible random-effects structure")

            # The fitted contrast schema reproduces columns in their original order.
            # Names can coincide (e.g. an encoded factor and a literal column), so
            # their positions must retain identity instead of matching by a dict.
            if new.term_names != fitted.term_names:
                raise ValueError(
                    f"New data produced incompatible random-effect columns for "
                    f"'{fitted.grouping_factor}'"
                )
            mapped_terms = np.arange(fitted.n_terms, dtype=np.int64)

            fitted_string_levels = {
                str(fitted_level): index for fitted_level, index in fitted.level_map.items()
            }
            new_levels: list[Any | None] = [None] * new.n_levels
            for stored_level, index in new.level_map.items():
                new_levels[index] = stored_level

            mapped_levels = np.full(new.n_levels, -1, dtype=np.int64)
            unknown_levels: list[Any] = []
            for index, candidate_level in enumerate(new_levels):
                if candidate_level in fitted.level_map:
                    mapped_levels[index] = fitted.level_map[candidate_level]
                elif str(candidate_level) in fitted_string_levels:
                    mapped_levels[index] = fitted_string_levels[str(candidate_level)]
                else:
                    unknown_levels.append(candidate_level)

            if unknown_levels and not allow_new_levels:
                raise ValueError(
                    f"New level '{unknown_levels[0]}' in grouping factor "
                    f"'{fitted.grouping_factor}'. Set allow_new_levels=True to predict "
                    "with random effects = 0."
                )

            new_width = new.n_levels * new.n_terms
            block = Z[:, new_offset : new_offset + new_width].tocoo()
            entry_levels = block.col // new.n_terms
            entry_terms = block.col % new.n_terms
            target_levels = mapped_levels[entry_levels]
            target_terms = mapped_terms[entry_terms]
            known = target_levels >= 0

            if np.any(known):
                known_rows.append(block.row[known])
                known_cols.append(
                    fitted_offset + target_levels[known] * fitted.n_terms + target_terms[known]
                )
                known_values.append(block.data[known])

            unknown = ~known
            if np.any(unknown):
                unknown_rows, compact_rows = np.unique(block.row[unknown], return_inverse=True)
                unknown_design = sparse.csr_matrix(
                    (block.data[unknown], (compact_rows, target_terms[unknown])),
                    shape=(len(unknown_rows), fitted.n_terms),
                )
                prior_var[unknown_rows] += sparse_covariance_factor_diagonal(
                    unknown_design, level_factor
                )

            fitted_offset += fitted.n_levels * fitted.n_terms
            new_offset += new_width

        if known_values:
            rows = np.concatenate(known_rows)
            cols = np.concatenate(known_cols)
            values = np.concatenate(known_values)
        else:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float64)

        aligned = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(n_pred, self.matrices.n_random),
            dtype=np.float64,
        )
        return aligned, prior_var

    def _coerce_new_response(self, newresp: ArrayLike | None) -> NDArray[np.floating]:
        if newresp is None:
            return self.matrices.y

        response = np.asarray(newresp)
        if self.matrices.response_levels is not None and not _numeric_response(newresp):
            arr = _encode_binomial_factor(response, self.matrices.response_levels)
        else:
            arr = np.asarray(newresp, dtype=np.float64)
        if len(arr) != self.matrices.n_obs:
            raise ValueError(f"newresp has length {len(arr)}, expected {self.matrices.n_obs}")
        if self.matrices.trials is not None:
            return _normalize_grouped_binomial_response(arr, self.matrices.trials)
        return arr

    def _clone_matrices_with_response_base(self, y: NDArray[np.floating]) -> ModelMatrices:
        frame = self.matrices.frame
        levels = self.matrices.response_levels
        if frame is not None:
            # Keep the stored frame aligned with refitted responses
            # so subsequent updates and cross-validation reuse the current data.
            values = y if self.matrices.trials is None else np.rint(y * self.matrices.trials)
            binary_factor = levels is not None and bool(np.all((y == 0) | (y == 1)))
            if binary_factor:
                values = np.asarray(levels, dtype=object)[y.astype(np.intp)]
            if type(frame).__module__.startswith("pandas"):
                import pandas as pd

                frame = frame.copy()
                frame[self.formula.response] = (
                    pd.Categorical(values, categories=list(levels))
                    if binary_factor and levels is not None
                    else values
                )
            else:
                import polars as pl

                frame = frame.with_columns(pl.Series(self.formula.response, values.tolist()))
            frame = _restore_binomial_factor(frame, self.formula.response, levels)
        return replace(self.matrices, y=y, frame=frame)

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

    def get_sigma(self) -> float:
        return self.sigma

    def npar(self) -> int:
        """Get the number of estimated parameters.

        Counts the fixed effects (beta), the covariance parameters (theta)
        and, for models with a residual scale, sigma.
        """
        return len(self.beta) + len(self.theta) + int(self._HAS_SIGMA)

    def hatvalues(self) -> NDArray[np.floating]:
        """Return leverage values (the diagonal of the mixed-model hat matrix).

        The hat matrix maps the response to fitted values through both fixed
        and random effects; GLMMs use the final PIRLS working weights. Values
        lie in [0, 1), and values close to 1 mark observations that largely
        determine their own fitted value.
        """
        return self._hat_values.copy()

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        ``D_i = r_i**2 / (p * sigma**2) * h_i / (1 - h_i)**2`` with leverage
        ``h_i``, ``p`` fixed-effect parameters and ``r_i`` the residual on the
        fitted working scale: square-root prior weights times the response
        residual for LMMs, and the Pearson residual with ``sigma = 1`` for
        GLMMs. Models without fixed effects return NaN. A common rule of thumb
        flags observations with ``D_i > 4/n`` or ``D_i > 1``.
        """
        if self.matrices.n_fixed == 0:
            # Skip the influence projection; the normalization has no coefficients.
            return np.full(self.matrices.n_obs, np.nan)

        from mixedlm.diagnostics.influence import influence

        return influence(cast("LmerResult | GlmerResult", self)).cooks_distance

    def dotplot(
        self,
        group: str | None = None,
        term: str | None = None,
        condVar: bool = True,
        order: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Create caterpillar plots of random effects.

        Each panel shows one random-effect term of a grouping factor, with
        95% intervals from the conditional variances, ordered by magnitude.

        Parameters
        ----------
        group : str, optional
            Grouping factor to plot. Defaults to the first one.
        term : str, optional
            Random-effect term to plot. Defaults to one panel per term.
        condVar : bool, default True
            Whether to show intervals from the conditional variances.
        order : bool, default True
            Whether to order levels by random-effect value.
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure; requires matplotlib.

        Examples
        --------
        >>> fig = result.dotplot()  # One panel per random-effect term
        >>> fig = result.dotplot(term="(Intercept)")  # Only intercepts
        """
        from mixedlm.diagnostics.plots import _ranef_dotplot

        return _ranef_dotplot(
            cast("LmerResult | GlmerResult", self),
            group=group,
            term=term,
            condVar=condVar,
            order=order,
            figsize=figsize,
        )

    def qqmath(
        self,
        group: str | None = None,
        term: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Create normal QQ plots of random effects.

        Points along the reference line indicate normally distributed random
        effects.

        Parameters
        ----------
        group : str, optional
            Grouping factor to plot. Defaults to the first one.
        term : str, optional
            Random-effect term to plot. Defaults to one panel per term.
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure; requires matplotlib.

        Examples
        --------
        >>> fig = result.qqmath()  # QQ plots for all random effects
        >>> fig = result.qqmath(term="(Intercept)")  # Only intercepts
        """
        from mixedlm.diagnostics.plots import _ranef_qqmath

        return _ranef_qqmath(
            cast("LmerResult | GlmerResult", self), group=group, term=term, figsize=figsize
        )

    def plot(
        self,
        which: list[int] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Create residual diagnostic plots, like R's plot() for merMod objects.

        Parameters
        ----------
        which : list of int, optional
            Plots to include. Default is [1, 2, 3, 4].
            1 = Residuals vs Fitted values
            2 = Normal Q-Q plot of residuals
            3 = Scale-Location plot (sqrt of standardized residuals vs fitted)
            4 = Residuals by Group (boxplot, only if random effects exist)
        figsize : tuple, optional
            Figure size (width, height) in inches. Defaults to a size based on
            the number of plots.

        Returns
        -------
        Figure
            Matplotlib figure; requires matplotlib.

        Examples
        --------
        >>> fig = result.plot()  # All 4 diagnostic plots
        >>> fig = result.plot(which=[1, 2])  # Only residuals vs fitted and Q-Q

        See Also
        --------
        qqmath : QQ plots of random effects.
        """
        from mixedlm.diagnostics.plots import plot_diagnostics

        return plot_diagnostics(
            cast("LmerResult | GlmerResult", self), which=which, figsize=figsize
        )

    def getME(self, name: str) -> Any:
        """Extract model components by name, like lme4's getME().

        Parameters
        ----------
        name : str
            Name of the component to extract:
            - "X", "Z", "Zt" : Design matrices (Z is sparse, n x q)
            - "y" : Response vector
            - "beta", "theta" : Fixed effects and relative covariance parameters
            - "Lambda", "Lambdat" : Relative covariance factor (sparse, q x q)
              and its transpose
            - "u" : Spherical random effects, with ``b = Lambda @ u``
            - "b" : Conditional modes of the random effects
            - "n"/"n_obs", "p"/"n_fixed", "q"/"n_random" : Dimensions
            - "lower" : Lower bounds for theta
            - "weights", "offset" : Prior weights and offset
            - "deviance" : Fitting criterion
            - "fixef_names" : Fixed-effect coefficient names
            - "flist", "cnms" : Grouping factors and their term names
            - "Gp" : Group pointers (cumulative random-effect columns)
            - "RX", "RZX" : Fixed-effect Cholesky factor and cross term
            - "Lind" : Index map from theta to Lambda entries
            - "devcomp" : Deviance components and dimensions
            Linear mixed models also provide "sigma" and "REML";
            generalized models provide "family" and "nAGQ".

        Returns
        -------
        The requested component. Arrays are copies unless they are the
        model's design matrices or response.

        Raises
        ------
        ValueError
            If an unknown component name is requested.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> X = result.getME("X")
        >>> Lambda = result.getME("Lambda")
        >>> np.allclose(Lambda @ result.getME("u"), result.getME("b"))
        True
        """
        components = self._getme_components()
        try:
            component = components[name]
        except KeyError:
            raise ValueError(
                f"Unknown component name: '{name}'. Valid names are: {list(components)}"
            ) from None
        return component()

    def _getme_components(self) -> dict[str, Callable[[], Any]]:
        """Map getME() names to accessors; subclasses add model-specific names."""
        matrices = self.matrices
        structures = matrices.random_structures

        def lambda_matrix() -> sparse.csc_matrix:
            return _build_lambda(self.theta, structures)

        return {
            "X": lambda: matrices.X,
            "Z": lambda: matrices.Z,
            "Zt": lambda: matrices.Zt,
            "y": lambda: matrices.y,
            "beta": self.beta.copy,
            "theta": self.theta.copy,
            "Lambda": lambda_matrix,
            "Lambdat": lambda: lambda_matrix().T,
            "u": self._spherical_u,
            "b": self.u.copy,
            "n": self.nobs,
            "n_obs": self.nobs,
            "p": lambda: matrices.n_fixed,
            "n_fixed": lambda: matrices.n_fixed,
            "q": lambda: matrices.n_random,
            "n_random": lambda: matrices.n_random,
            "lower": self._theta_lower_bounds,
            "weights": self.weights,
            "offset": self.offset,
            "deviance": lambda: self.deviance,
            "fixef_names": lambda: list(matrices.fixed_names),
            "flist": lambda: [s.grouping_factor for s in structures],
            "cnms": lambda: {s.grouping_factor: s.term_names for s in structures},
            "Gp": lambda: np.cumsum([0] + [s.n_levels * s.n_terms for s in structures]),
            "RX": self._compute_RX,
            "RZX": self._compute_RZX,
            "Lind": self._build_Lind,
            "devcomp": self._get_devcomp,
        }

    def _spherical_u(self) -> NDArray[np.float64]:
        """Return the spherical random effects u, where b = Lambda @ u.

        Penalized least squares leaves u without a component in the null space
        of Lambda, so the pseudo-inverse of each level's covariance factor
        recovers it, including on singular fits. For nearly singular factors,
        rounding in b is amplified by the factor's conditioning.
        """
        from mixedlm.utils.variance import getL

        structures = self.matrices.random_structures
        level_factors = cast(
            list[NDArray[np.floating]], getL(self.theta, structures, as_blocks=True)
        )
        u = np.empty(self.matrices.n_random, dtype=np.float64)
        start = 0
        for struct, level_factor in zip(structures, level_factors, strict=True):
            stop = start + struct.n_levels * struct.n_terms
            b = self.u[start:stop].reshape(struct.n_levels, struct.n_terms)
            u[start:stop] = (b @ np.linalg.pinv(level_factor).T).ravel()
            start = stop
        return u

    def _build_Lind(self) -> NDArray[np.int64]:
        """Build Lind, the index mapping from theta to Lambda entries."""
        indices = []
        theta_idx = 0

        for struct in self.matrices.random_structures:
            n_terms = struct.n_terms

            if struct.correlated:
                n_theta = n_terms * (n_terms + 1) // 2
                template_indices = list(range(theta_idx, theta_idx + n_theta))
            else:
                n_theta = n_terms
                template_indices = list(range(theta_idx, theta_idx + n_terms))

            for _ in range(struct.n_levels):
                indices.extend(template_indices)

            theta_idx += n_theta

        return np.array(indices, dtype=np.int64)

    def _get_devcomp(self) -> dict[str, Any]:
        """Return lme4-style deviance components ``cmp`` and dimensions ``dims``.

        Components a model does not define, such as ``REML`` for an ML fit or
        ``drsum`` for a linear model, are NaN.
        """
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        q = self.matrices.n_random
        dims = {
            "n": n,
            "p": p,
            "q": q,
            "nmp": n - p,
            "nth": len(self.theta),
            "REML": int(self.isREML()),
            "useSc": int(self._HAS_SIGMA),
            "nAGQ": int(getattr(self, "nAGQ", 1)),
            "q0": q,
            "q1": 0,
            "qrx": p,
            "ngrps": len(self.ngrps()),
        }
        return {"cmp": self._devcomp_cmp(), "dims": dims}

    def _update_formula(self, new_formula: str) -> str:
        """Process formula update syntax with '.' placeholders."""
        original = str(self.formula)

        if "." not in new_formula:
            return new_formula

        lhs, rhs = original.split("~", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        if "~" in new_formula:
            new_lhs, new_rhs = new_formula.split("~", 1)
            new_lhs = new_lhs.strip()
            new_rhs = new_rhs.strip()

            if new_lhs == ".":
                new_lhs = lhs

            if new_rhs.startswith(". +"):
                new_rhs = rhs + " +" + new_rhs[3:]
            elif new_rhs.startswith(". -"):
                terms_to_remove = new_rhs[3:].strip().split("+")
                terms_to_remove = [t.strip() for t in terms_to_remove]
                rhs_terms = [t.strip() for t in rhs.split("+")]
                rhs_terms = [t for t in rhs_terms if t not in terms_to_remove]
                new_rhs = " + ".join(rhs_terms)
            elif new_rhs == ".":
                new_rhs = rhs

            return f"{new_lhs} ~ {new_rhs}"
        else:
            return new_formula

    def get_formula(
        self,
        random_only: bool = False,
        fixed_only: bool = False,
    ) -> Formula | str:
        """Get the model formula.

        Returns the Formula object, or a string when random_only or
        fixed_only is set.

        Parameters
        ----------
        random_only : bool, default False
            If True, return only the random effects part as a string.
        fixed_only : bool, default False
            If True, return only the fixed effects part as a string.

        Examples
        --------
        >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
        >>> result.get_formula().response
        'Reaction'
        >>> result.get_formula(fixed_only=True)
        'Reaction ~ Days'
        >>> result.get_formula(random_only=True)
        '(Days | Subject)'
        """
        from mixedlm.formula.parser import getFixedFormulaStr, getRandomFormulaStr

        if random_only and fixed_only:
            raise ValueError("Cannot specify both random_only and fixed_only")

        if random_only:
            return getRandomFormulaStr(str(self.formula))
        elif fixed_only:
            return getFixedFormulaStr(str(self.formula))
        else:
            return self.formula

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "Wald",
        n_boot: int = 1000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for fixed effects.

        Parameters
        ----------
        parm : str or list of str, optional
            Coefficient names. Defaults to every fixed effect.
        level : float, default 0.95
            Confidence level.
        method : str, default "Wald"
            "Wald" (normal quantiles of vcov), "profile" (likelihood
            profiles) or "boot" (parametric bootstrap percentiles).
        n_boot : int, default 1000
            Bootstrap replicates for method="boot".
        seed : int, optional
            Random seed for method="boot".

        Returns
        -------
        dict
            Coefficient names mapped to (lower, upper) bounds.
        """
        from scipy import stats

        from mixedlm.utils.names import _check_unique_coefficient_names

        level = _validate_confidence_level(level)
        if parm is None:
            parm = self.matrices.fixed_names
        elif isinstance(parm, str):
            parm = [parm]

        _check_unique_coefficient_names(
            self.matrices.fixed_names,
            None if method == "boot" else parm,
            alternative="Use tidy(conf_int=True) for intervals in coefficient order.",
        )

        if method == "Wald":
            vcov = self.vcov()
            z_crit = stats.norm.isf((1 - level) / 2)

            result: dict[str, tuple[float, float]] = {}
            for p in parm:
                if p not in self.matrices.fixed_names:
                    continue
                idx = self.matrices.fixed_names.index(p)
                se = np.sqrt(vcov[idx, idx])
                lower = self.beta[idx] - z_crit * se
                upper = self.beta[idx] + z_crit * se
                result[p] = (float(lower), float(upper))
            return result

        if method == "profile":
            # Endpoints use root finding; confidence intervals need no interior plot grid.
            profiles = self.profile(which=parm, level=level, n_points=3)
            return {p: (profiles[p].ci_lower, profiles[p].ci_upper) for p in parm if p in profiles}

        if method == "boot":
            return self._bootstrap(n_boot=n_boot, seed=seed).ci(level=level)

        raise ValueError(f"Unknown method: {method}. Use 'Wald', 'profile', or 'boot'.")

    def simulate(
        self,
        nsim: int = 1,
        seed: RandomSeed = None,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        """Simulate responses using an isolated or caller-provided random stream.

        ``seed`` accepts an integer, ``RandomState``, ``Generator``, or ``None``.
        Integer seeds preserve the existing draw sequence for the selected backend.
        Reusing a stream continues it across calls without changing NumPy's global state.
        New random effects are drawn unless ``use_re`` is False or ``re_form`` is
        "NA" or "~0". Returns shape (n,) for ``nsim=1``, else (n, nsim).
        """
        validate_simulation_count(nsim)
        rng = random_stream(seed)

        if nsim == 1:
            return self._simulate_once(use_re, re_form, rng)

        if not (use_re and self.matrices.n_random > 0 and re_form not in ("~0", "NA")):
            fixed = self.matrices.X @ self.beta + self.matrices.offset
            eta = np.broadcast_to(fixed[:, None], (self.matrices.n_obs, nsim))
            return self._simulate_from_eta(eta, rng)

        try:
            from mixedlm._rust import simulate_re_batch
        except ImportError:
            return np.column_stack([self._simulate_once(use_re, re_form, rng) for _ in range(nsim)])
        return self._simulate_batch_rust(nsim, native_seed(seed, rng), simulate_re_batch, rng)

    def _simulate_batch_rust(
        self,
        nsim: int,
        seed: int | None,
        simulate_re_batch: Any,
        rng: Any | None = None,
    ) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        structures = self.matrices.random_structures
        theta, correlated = simulation_parameters(self.theta, structures)
        u_batch = simulate_re_batch(
            theta,
            self.sigma,
            [structure.n_levels for structure in structures],
            [structure.n_terms for structure in structures],
            correlated,
            nsim,
            seed,
        )
        eta = np.asarray(self.matrices.Z @ u_batch.T, dtype=np.float64)
        eta += (self.matrices.X @ self.beta + self.matrices.offset)[:, None]
        return self._simulate_from_eta(eta, rng)

    def _simulate_once(
        self,
        use_re: bool = True,
        re_form: str | None = None,
        rng: Any | None = None,
    ) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        eta = self.matrices.X @ self.beta + self.matrices.offset

        if use_re and self.matrices.n_random > 0 and re_form not in ("~0", "NA"):
            u_new = simulate_random_effects(
                self.theta, self.matrices.random_structures, self.sigma, rng=rng
            )
            eta += self.matrices.Z @ u_new

        return self._simulate_from_eta(eta, rng)

    def refit(self: _Result, newresp: ArrayLike | None = None, **kwargs: Any) -> _Result:
        """Refit the model with a new response vector.

        The formula, design matrices, weights and offset are reused, which
        suits simulation studies, bootstrap and permutation tests.

        Parameters
        ----------
        newresp : array-like, optional
            New response values with the original length. If None, refits with
            the original response. Grouped binomial models take success counts
            and reuse the original trial counts. Two-level factor models accept
            the fitted labels or encoded numeric 0/1 responses.
        **kwargs
            Additional arguments passed to the optimizer (start, method,
            maxiter). GLMM inner controls pirls_maxiter and pirls_tol default to
            the original fit's settings.

        Returns
        -------
        New fitted model result with the updated response.

        Examples
        --------
        >>> y_sim = result.simulate()
        >>> result_sim = result.refit(newresp=y_sim)
        """
        y_new = self._coerce_new_response(newresp)
        matrices = self._clone_matrices_with_response_base(y_new)
        return self._refit_from_matrices(matrices, **kwargs)

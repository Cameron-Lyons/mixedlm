from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.control import LmerControl

from mixedlm.estimation.reml import LMMOptimizer, _build_lambda, _count_theta
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices


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

    def __array__(self) -> NDArray[np.floating]:
        return self.fit

    def __len__(self) -> int:
        return len(self.fit)

    def __getitem__(self, idx: int) -> float:
        return float(self.fit[idx])


@dataclass
class LogLik:
    value: float
    df: int
    nobs: int
    REML: bool = False

    def __float__(self) -> float:
        return float(self.value)

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
        return {name: np.any(group.sdev < tol) for name, group in self.groups.items()}


@dataclass
class VarCorr:
    groups: dict[str, VarCorrGroup]
    residual: float

    def __str__(self) -> str:
        lines = ["Random effects:"]
        lines.append(f" {'Groups':<11} {'Name':<12} {'Variance':>10} {'Std.Dev.':>10} {'Corr':>6}")
        for group_name, group in self.groups.items():
            for i, term in enumerate(group.term_names):
                grp = group_name if i == 0 else ""
                var = group.variance[term]
                sd = group.stddev[term]
                if i == 0 or group.corr is None:
                    lines.append(f" {grp:<11} {term:<12} {var:>10.4f} {sd:>10.4f}")
                else:
                    corr_vals = " ".join(f"{group.corr[i, j]:>6.2f}" for j in range(i))
                    lines.append(f" {grp:<11} {term:<12} {var:>10.4f} {sd:>10.4f} {corr_vals}")
        resid_sd = np.sqrt(self.residual)
        lines.append(f" {'Residual':<11} {'':<12} {self.residual:>10.4f} {resid_sd:>10.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_groups = len(self.groups)
        return f"VarCorr({n_groups} groups, residual={self.residual:.4f})"

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {name: group.variance for name, group in self.groups.items()}

    def get_cov(self, group: str) -> NDArray[np.floating]:
        return self.groups[group].cov

    def get_corr(self, group: str) -> NDArray[np.floating] | None:
        return self.groups[group].corr


@dataclass
class LmerResult:
    formula: Formula
    matrices: ModelMatrices
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    sigma: float
    u: NDArray[np.floating]
    deviance: float
    REML: bool
    converged: bool
    n_iter: int

    def fixef(self) -> dict[str, float]:
        return dict(zip(self.matrices.fixed_names, self.beta, strict=False))

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            u_block = self.u[u_idx : u_idx + n_u].reshape(n_levels, n_terms)
            u_idx += n_u

            term_ranefs: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_ranefs[term_name] = u_block[:, j]

            result[struct.grouping_factor] = term_ranefs

        if not condVar:
            return result

        cond_var = self._compute_condVar()
        return RanefResult(values=result, condVar=cond_var)

    def _compute_condVar(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        q = self.matrices.n_random
        if q == 0:
            return {}

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)

        Zt = self.matrices.Zt
        ZtZ = Zt @ Zt.T
        LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V = LambdatZtZLambda + I_q

        V_dense = V.toarray()

        Lambda_dense = Lambda.toarray() if sparse.issparse(Lambda) else Lambda
        V_inv_Lambda_t = linalg.solve(V_dense, Lambda_dense.T, assume_a="pos")
        cond_cov = self.sigma**2 * Lambda_dense @ V_inv_Lambda_t

        cond_var_result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            block_diag = np.diag(cond_cov[u_idx : u_idx + n_u, u_idx : u_idx + n_u])
            var_block = block_diag.reshape(n_levels, n_terms)

            u_idx += n_u

            term_vars: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_vars[term_name] = var_block[:, j]

            cond_var_result[struct.grouping_factor] = term_vars

        return cond_var_result

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

    def get_sigma(self) -> float:
        return self.sigma

    def df_residual(self) -> int:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        return n - p

    def weights(self) -> NDArray[np.floating]:
        """Get the model weights.

        Returns the prior weights used in model fitting.
        If no weights were specified, returns an array of ones.

        Returns
        -------
        NDArray
            Array of weights with length equal to number of observations.
        """
        return self.matrices.weights.copy()

    def offset(self) -> NDArray[np.floating]:
        """Get the model offset.

        Returns the offset used in model fitting.
        If no offset was specified, returns an array of zeros.

        Returns
        -------
        NDArray
            Array of offsets with length equal to number of observations.
        """
        return self.matrices.offset.copy()

    def model_matrix(
        self, type: str = "fixed"
    ) -> NDArray[np.floating] | sparse.csc_matrix:
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
        elif type in ("random", "Z"):
            return self.matrices.Z
        elif type == "both":
            return (self.matrices.X, self.matrices.Z)
        else:
            raise ValueError(
                f"Unknown type '{type}'. Use 'fixed', 'random', 'X', 'Z', or 'both'."
            )

    def terms(self) -> ModelTerms:
        """Get information about the model terms.

        Returns a ModelTerms object containing information about the
        response variable, fixed effect terms, random effect terms,
        and grouping factors.

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
        from mixedlm.formula.terms import InteractionTerm, VariableTerm

        fixed_terms = list(self.matrices.fixed_names)

        random_terms: dict[str, list[str]] = {}
        for struct in self.matrices.random_structures:
            random_terms[struct.grouping_factor] = list(struct.term_names)

        fixed_variables: set[str] = set()
        for term in self.formula.fixed.terms:
            if isinstance(term, VariableTerm):
                fixed_variables.add(term.name)
            elif isinstance(term, InteractionTerm):
                fixed_variables.update(term.variables)

        random_variables: set[str] = set()
        for rterm in self.formula.random:
            for term in rterm.expr:
                if isinstance(term, VariableTerm):
                    random_variables.add(term.name)
                elif isinstance(term, InteractionTerm):
                    random_variables.update(term.variables)

        grouping_factors = {struct.grouping_factor for struct in self.matrices.random_structures}

        return ModelTerms(
            response=self.formula.response,
            fixed_terms=fixed_terms,
            random_terms=random_terms,
            fixed_variables=fixed_variables,
            random_variables=random_variables,
            grouping_factors=grouping_factors,
            has_intercept=self.formula.fixed.has_intercept,
        )

    def get_formula(self) -> Formula:
        """Get the model formula.

        Returns
        -------
        Formula
            The formula object used to specify the model.
        """
        return self.formula

    def model_frame(self) -> pd.DataFrame:
        """Get the model frame.

        Returns the data frame containing only the variables used
        in the model formula, after any NA handling.

        Returns
        -------
        pd.DataFrame
            Data frame with the response variable, fixed effect
            variables, and grouping factors.

        Examples
        --------
        >>> result = lmer("y ~ x + (1 | group)", data)
        >>> mf = result.model_frame()
        >>> print(mf.columns.tolist())  # ['y', 'x', 'group']
        """
        import pandas as pd

        if self.matrices.frame is not None:
            return self.matrices.frame.copy()

        return pd.DataFrame({"y": self.matrices.y})

    @cached_property
    def _fitted_values(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    def fitted(self, na_expand: bool = True) -> NDArray[np.floating]:
        """Get fitted values.

        Parameters
        ----------
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Fitted values.
        """
        values = self._fitted_values
        if na_expand and self._should_expand_na():
            return self.matrices.na_info.expand_to_original(values)
        return values

    def _should_expand_na(self) -> bool:
        from mixedlm.utils.na_action import NAAction

        return (
            self.matrices.na_info is not None
            and self.matrices.na_info.action == NAAction.EXCLUDE
            and self.matrices.na_info.n_omitted > 0
        )

    def residuals(self, type: str = "response", na_expand: bool = True) -> NDArray[np.floating]:
        """Get residuals.

        Parameters
        ----------
        type : str, default "response"
            Type of residuals: "response" or "pearson".
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Residuals.
        """
        fitted = self._fitted_values
        if type == "response":
            resid = self.matrices.y - fitted
        elif type == "pearson":
            resid = (self.matrices.y - fitted) / self.sigma
        else:
            raise ValueError(f"Unknown residual type: {type}")

        if na_expand and self._should_expand_na():
            return self.matrices.na_info.expand_to_original(resid)
        return resid

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        re_form: str | None = None,
        allow_new_levels: bool = False,
        se_fit: bool = False,
        interval: str = "none",
        level: float = 0.95,
    ) -> NDArray[np.floating] | PredictResult:
        """Generate predictions from the fitted model.

        Parameters
        ----------
        newdata : DataFrame, optional
            New data for prediction. If None, returns fitted values.
        re_form : str, optional
            Formula for random effects. Use "NA" or "~0" for fixed effects only.
        allow_new_levels : bool, default False
            Allow new levels in grouping factors (predicts with RE=0).
        se_fit : bool, default False
            If True, return standard errors of predictions.
        interval : str, default "none"
            Type of interval: "none", "confidence", or "prediction".
        level : float, default 0.95
            Confidence level for intervals.

        Returns
        -------
        NDArray or PredictResult
            Predictions. Returns PredictResult if se_fit=True or interval!="none".
        """
        include_re = re_form != "NA" and re_form != "~0"

        if newdata is None:
            pred = self._fitted_values.copy()
            if not se_fit and interval == "none":
                return pred
            X = self.matrices.X
            new_matrices = self.matrices
        else:
            new_matrices = build_model_matrices(self.formula, newdata)
            X = new_matrices.X
            pred = X @ self.beta

            if new_matrices.offset is not None:
                pred = pred + new_matrices.offset

            if include_re:
                pred = self._add_random_effects_to_pred(
                    pred, newdata, new_matrices, allow_new_levels
                )

        if not se_fit and interval == "none":
            return pred

        vcov_beta = self.vcov()
        var_fixed = np.sum((X @ vcov_beta) * X, axis=1)

        if include_re and newdata is not None:
            var_re = self._compute_re_prediction_variance(newdata, new_matrices, allow_new_levels)
            var_fit = var_fixed + var_re
        else:
            var_fit = var_fixed

        se = np.sqrt(var_fit)

        if interval == "none":
            return PredictResult(fit=pred, se_fit=se, interval="none", level=level)

        z_crit = stats.norm.ppf(1 - (1 - level) / 2)

        if interval == "confidence":
            lower = pred - z_crit * se
            upper = pred + z_crit * se
            return PredictResult(
                fit=pred, se_fit=se, lower=lower, upper=upper, interval="confidence", level=level
            )
        elif interval == "prediction":
            var_pred = var_fit + self.sigma**2
            se_pred = np.sqrt(var_pred)
            lower = pred - z_crit * se_pred
            upper = pred + z_crit * se_pred
            return PredictResult(
                fit=pred,
                se_fit=se,
                lower=lower,
                upper=upper,
                interval="prediction",
                level=level,
            )
        else:
            raise ValueError(
                f"Unknown interval type: {interval}. Use 'none', 'confidence', or 'prediction'."
            )

    def _add_random_effects_to_pred(
        self,
        pred: NDArray[np.floating],
        newdata: pd.DataFrame,
        new_matrices: ModelMatrices,
        allow_new_levels: bool,
    ) -> NDArray[np.floating]:
        """Add random effects contribution to predictions."""
        u_idx = 0

        for struct in self.matrices.random_structures:
            group_col = struct.grouping_factor
            n_terms = struct.n_terms
            n_levels_orig = struct.n_levels

            if group_col not in newdata.columns:
                u_idx += n_levels_orig * n_terms
                continue

            new_groups = newdata[group_col].astype(str).values
            u_block = self.u[u_idx : u_idx + n_levels_orig * n_terms].reshape(
                n_levels_orig, n_terms
            )
            u_idx += n_levels_orig * n_terms

            for i, term_name in enumerate(struct.term_names):
                if term_name == "(Intercept)":
                    term_values = np.ones(len(newdata))
                elif term_name in newdata.columns:
                    term_values = newdata[term_name].values.astype(np.float64)
                else:
                    continue

                for j, group_level in enumerate(new_groups):
                    if group_level in struct.level_map:
                        level_idx = struct.level_map[group_level]
                        pred[j] += u_block[level_idx, i] * term_values[j]
                    elif not allow_new_levels:
                        raise ValueError(
                            f"New level '{group_level}' in grouping factor '{group_col}'. "
                            "Set allow_new_levels=True to predict with random effects = 0."
                        )

        return pred

    def _compute_re_prediction_variance(
        self,
        newdata: pd.DataFrame,
        new_matrices: ModelMatrices,
        allow_new_levels: bool,
    ) -> NDArray[np.floating]:
        """Compute variance contribution from random effects for predictions."""
        n = len(newdata)
        var_re = np.zeros(n, dtype=np.float64)

        cond_var = self._compute_condVar()
        u_idx = 0

        for struct in self.matrices.random_structures:
            group_col = struct.grouping_factor
            n_terms = struct.n_terms
            n_levels_orig = struct.n_levels

            if group_col not in newdata.columns:
                u_idx += n_levels_orig * n_terms
                continue

            if group_col not in cond_var:
                u_idx += n_levels_orig * n_terms
                continue

            new_groups = newdata[group_col].astype(str).values
            group_cond_var = cond_var[group_col]

            for i, term_name in enumerate(struct.term_names):
                if term_name == "(Intercept)":
                    term_values = np.ones(n)
                elif term_name in newdata.columns:
                    term_values = newdata[term_name].values.astype(np.float64)
                else:
                    continue

                if term_name not in group_cond_var:
                    continue

                term_var = group_cond_var[term_name]

                for j, group_level in enumerate(new_groups):
                    if group_level in struct.level_map:
                        level_idx = struct.level_map[group_level]
                        var_re[j] += term_var[level_idx] * term_values[j] ** 2
                    elif allow_new_levels:
                        theta_idx = 0
                        for s in self.matrices.random_structures:
                            if s.grouping_factor == group_col:
                                break
                            theta_idx += (
                                s.n_terms * (s.n_terms + 1) // 2 if s.correlated else s.n_terms
                            )
                        re_var = self.theta[theta_idx + i] ** 2 * self.sigma**2
                        var_re[j] += re_var * term_values[j] ** 2

            u_idx += n_levels_orig * n_terms

        return var_re

    def vcov(self) -> NDArray[np.floating]:
        q = self.matrices.n_random

        if q == 0:
            XtX = self.matrices.X.T @ self.matrices.X
            p = XtX.shape[0]
            try:
                L = linalg.cholesky(XtX, lower=True)
                XtX_inv = linalg.cho_solve((L, True), np.eye(p))
            except linalg.LinAlgError:
                XtX_inv = linalg.solve(XtX, np.eye(p))
            return self.sigma**2 * XtX_inv

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)

        Zt = self.matrices.Zt
        ZtZ = Zt @ Zt.T
        LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V_factor = LambdatZtZLambda + I_q
        V_factor_dense = V_factor.toarray()
        L_V = linalg.cholesky(V_factor_dense, lower=True)

        ZtX = Zt @ self.matrices.X
        Lambdat_ZtX = Lambda.T @ ZtX
        RZX = linalg.solve_triangular(L_V, Lambdat_ZtX, lower=True)

        XtX = self.matrices.X.T @ self.matrices.X
        RZX_tRZX = RZX.T @ RZX
        XtVinvX = XtX - RZX_tRZX

        p = XtVinvX.shape[0]
        try:
            L = linalg.cholesky(XtVinvX, lower=True)
            XtVinvX_inv = linalg.cho_solve((L, True), np.eye(p))
        except linalg.LinAlgError:
            XtVinvX_inv = linalg.solve(XtVinvX, np.eye(p))
        return self.sigma**2 * XtVinvX_inv

    def hatvalues(self) -> NDArray[np.floating]:
        """Compute leverage values (diagonal of the hat matrix).

        For mixed models, the hat matrix projects the response onto fitted values.
        The leverage h_i measures how much observation i influences its own
        fitted value.

        Returns
        -------
        NDArray
            Leverage values for each observation, between 0 and 1.
            Values close to 1 indicate high-leverage observations.

        Notes
        -----
        For linear mixed models, the hat matrix incorporates both fixed and
        random effects. High leverage observations can have undue influence
        on model estimates.
        """
        q = self.matrices.n_random
        X = self.matrices.X

        if q == 0:
            XtX = X.T @ X
            try:
                L = linalg.cholesky(XtX, lower=True)
                XtX_inv = linalg.cho_solve((L, True), np.eye(X.shape[1]))
            except linalg.LinAlgError:
                XtX_inv = linalg.pinv(XtX)
            return np.sum((X @ XtX_inv) * X, axis=1)

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        Z = self.matrices.Z
        Zt = self.matrices.Zt

        ZtZ = Zt @ Zt.T
        LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda
        I_q = sparse.eye(q, format="csc")
        V_factor = LambdatZtZLambda + I_q
        V_factor_dense = V_factor.toarray()

        try:
            L_V = linalg.cholesky(V_factor_dense, lower=True)
        except linalg.LinAlgError:
            V_factor_dense += 1e-6 * np.eye(q)
            L_V = linalg.cholesky(V_factor_dense, lower=True)

        ZtX = Zt @ X
        Lambdat_ZtX = Lambda.T @ ZtX
        RZX = linalg.solve_triangular(L_V, Lambdat_ZtX, lower=True)

        XtX = X.T @ X
        XtVinvX = XtX - RZX.T @ RZX

        try:
            L_XVX = linalg.cholesky(XtVinvX, lower=True)
            XtVinvX_inv = linalg.cho_solve((L_XVX, True), np.eye(X.shape[1]))
        except linalg.LinAlgError:
            XtVinvX_inv = linalg.pinv(XtVinvX)

        h_fixed = np.sum((X @ XtVinvX_inv) * X, axis=1)

        Lambda_dense = Lambda.toarray() if sparse.issparse(Lambda) else Lambda
        Z_dense = Z.toarray() if sparse.issparse(Z) else Z
        ZLambda = Z_dense @ Lambda_dense

        V_inv = linalg.solve(V_factor_dense, np.eye(q), assume_a="pos")
        ZLambda_Vinv_LambdatZt = ZLambda @ V_inv @ ZLambda.T
        h_random = np.diag(ZLambda_Vinv_LambdatZt)

        h = h_fixed + h_random

        h = np.clip(h, 0, 1 - 1e-10)

        return h

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        Cook's distance measures the influence of each observation on the
        fitted values. Large values indicate observations that have substantial
        influence on the model fit.

        Returns
        -------
        NDArray
            Cook's distance for each observation.

        Notes
        -----
        Cook's distance is computed as:
            D_i = (e_i^2 / (p * sigma^2)) * (h_i / (1 - h_i)^2)

        where e_i is the residual, h_i is the leverage, p is the number of
        fixed effect parameters, and sigma^2 is the residual variance.

        A common rule of thumb is that observations with D_i > 4/n or D_i > 1
        may be influential and warrant further investigation.
        """
        h = self.hatvalues()
        resid = self.residuals(type="response")
        p = self.matrices.n_fixed

        h = np.clip(h, 0, 1 - 1e-10)

        cooks_d = (resid**2 / (p * self.sigma**2)) * (h / (1 - h) ** 2)

        return cooks_d

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns a dictionary containing various influence measures for
        identifying influential observations.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'hat': Leverage values (hatvalues)
            - 'cooks_d': Cook's distance
            - 'std_resid': Standardized residuals
            - 'student_resid': Studentized residuals

        Notes
        -----
        Influential observations are those that have a large effect on
        the model estimates. Use these diagnostics to identify:
        - High leverage points (large hat values)
        - Outliers (large standardized residuals)
        - Influential points (large Cook's distance)
        """
        h = self.hatvalues()
        resid = self.residuals(type="response")

        h_safe = np.clip(h, 0, 1 - 1e-10)
        std_resid = resid / (self.sigma * np.sqrt(1 - h_safe))

        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        df = n - p

        mse = np.sum(resid**2) / df
        loo_var = ((df * mse) - (resid**2 / (1 - h_safe))) / (df - 1)
        loo_var = np.maximum(loo_var, 1e-10)
        student_resid = resid / np.sqrt(loo_var * (1 - h_safe))

        return {
            "hat": h,
            "cooks_d": self.cooks_distance(),
            "std_resid": std_resid,
            "student_resid": student_resid,
        }

    def VarCorr(self) -> VarCorr:
        groups: dict[str, VarCorrGroup] = {}
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
                    for j in range(i + 1):
                        L_block[i, j] = theta_block[idx]
                        idx += 1

                cov_scaled = L_block @ L_block.T
                cov = cov_scaled * self.sigma**2

                stddevs = np.sqrt(np.diag(cov))
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = cov / np.outer(stddevs, stddevs)
                    corr = np.where(np.isfinite(corr), corr, 0.0)
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q
                cov = np.diag(theta_block**2) * self.sigma**2
                corr = None

            variance = {term: cov[i, i] for i, term in enumerate(struct.term_names)}
            stddev = {term: np.sqrt(cov[i, i]) for i, term in enumerate(struct.term_names)}

            groups[struct.grouping_factor] = VarCorrGroup(
                name=struct.grouping_factor,
                term_names=list(struct.term_names),
                variance=variance,
                stddev=stddev,
                cov=cov,
                corr=corr,
            )

        return VarCorr(groups=groups, residual=self.sigma**2)

    def rePCA(self) -> RePCA:
        """Perform PCA on the random effects covariance matrix.

        This function computes principal component analysis on the covariance
        matrix of each random effect grouping factor. It's useful for diagnosing
        overparameterization in the random effects structure.

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
        groups: dict[str, RePCAGroup] = {}
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
                    for j in range(i + 1):
                        L_block[i, j] = theta_block[idx]
                        idx += 1

                cov_scaled = L_block @ L_block.T
                cov = cov_scaled * self.sigma**2
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q
                cov = np.diag(theta_block**2) * self.sigma**2

            eigenvalues = linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)

            sdev = np.sqrt(eigenvalues)
            total_var = np.sum(eigenvalues)

            proportion = eigenvalues / total_var if total_var > 0 else np.zeros(q)

            cumulative = np.cumsum(proportion)

            groups[struct.grouping_factor] = RePCAGroup(
                name=struct.grouping_factor,
                n_terms=q,
                sdev=sdev,
                proportion=proportion,
                cumulative=cumulative,
            )

        return RePCA(groups=groups)

    def dotplot(
        self,
        group: str | None = None,
        term: str | None = None,
        condVar: bool = True,
        order: bool = True,
        figsize: tuple[float, float] | None = None,
    ):
        """Create a caterpillar plot of random effects.

        Dotplots (also called caterpillar plots) show the estimated random
        effects with confidence intervals, ordered by magnitude. They are
        useful for visualizing the distribution of random effects across
        groups and identifying outlier groups.

        Parameters
        ----------
        group : str, optional
            Name of grouping factor to plot. If None, uses the first
            random effect grouping factor.
        term : str, optional
            Name of random effect term to plot. If None, plots all terms
            for the selected group in separate panels.
        condVar : bool, default True
            Whether to show 95% confidence intervals based on the
            conditional variance of the random effects.
        order : bool, default True
            Whether to order groups by random effect magnitude.
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with the dotplot(s).

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> fig = result.dotplot()  # Plot all random effects
        >>> fig = result.dotplot(term="(Intercept)")  # Plot only intercepts
        """
        from mixedlm.diagnostics.plots import _check_matplotlib, plot_ranef

        _check_matplotlib()
        import matplotlib.pyplot as plt

        if group is None:
            if not self.matrices.random_structures:
                raise ValueError("No random effects in model")
            group = self.matrices.random_structures[0].grouping_factor

        struct = None
        for s in self.matrices.random_structures:
            if s.grouping_factor == group:
                struct = s
                break

        if struct is None:
            raise ValueError(f"Grouping factor '{group}' not found")

        if term is not None:
            if figsize is None:
                figsize = (8, max(6, struct.n_levels * 0.3))
            fig, ax = plt.subplots(figsize=figsize)
            plot_ranef(self, group=group, term=term, ax=ax, condVar=condVar, order=order)
            fig.tight_layout()
            return fig

        n_terms = struct.n_terms
        if n_terms == 1:
            if figsize is None:
                figsize = (8, max(6, struct.n_levels * 0.3))
            fig, ax = plt.subplots(figsize=figsize)
            plot_ranef(
                self, group=group, term=struct.term_names[0], ax=ax, condVar=condVar, order=order
            )
            fig.tight_layout()
            return fig

        ncols = min(2, n_terms)
        nrows = (n_terms + ncols - 1) // ncols

        if figsize is None:
            figsize = (6 * ncols, max(6, struct.n_levels * 0.25) * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_terms > 1 else [axes]

        for i, term_name in enumerate(struct.term_names):
            if i < len(axes):
                plot_ranef(
                    self, group=group, term=term_name, ax=axes[i], condVar=condVar, order=order
                )

        for i in range(n_terms, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        return fig

    def qqmath(
        self,
        group: str | None = None,
        term: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Create QQ plots of random effects against normal distribution.

        QQ (quantile-quantile) plots compare the distribution of random
        effects to a theoretical normal distribution. Points falling along
        a diagonal line indicate normality. Deviations suggest the random
        effects may not be normally distributed.

        Parameters
        ----------
        group : str, optional
            Name of grouping factor to plot. If None, uses the first
            random effect grouping factor.
        term : str, optional
            Name of random effect term to plot. If None, plots all terms
            for the selected group in separate panels.
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with the QQ plot(s).

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> fig = result.qqmath()  # QQ plots for all random effects
        >>> fig = result.qqmath(term="(Intercept)")  # Only intercepts
        """
        from mixedlm.diagnostics.plots import _check_matplotlib

        _check_matplotlib()
        import matplotlib.pyplot as plt
        from scipy import stats

        if group is None:
            if not self.matrices.random_structures:
                raise ValueError("No random effects in model")
            group = self.matrices.random_structures[0].grouping_factor

        struct = None
        for s in self.matrices.random_structures:
            if s.grouping_factor == group:
                struct = s
                break

        if struct is None:
            raise ValueError(f"Grouping factor '{group}' not found")

        ranefs = self.ranef()
        group_ranefs = ranefs[group]

        def plot_qq(ax, values, title):
            values = np.asarray(values)
            values_sorted = np.sort(values)
            n = len(values_sorted)

            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

            ax.scatter(theoretical, values_sorted, alpha=0.7, edgecolors="black", linewidths=0.5)

            slope, intercept = np.polyfit(theoretical, values_sorted, 1)
            line_x = np.array([theoretical.min(), theoretical.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, "r--", linewidth=1.5, label="Reference line")

            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title(title)
            ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

        if term is not None:
            if term not in group_ranefs:
                raise ValueError(f"Term '{term}' not found in group '{group}'")
            if figsize is None:
                figsize = (6, 5)
            fig, ax = plt.subplots(figsize=figsize)
            plot_qq(ax, group_ranefs[term], f"QQ Plot: {group} / {term}")
            fig.tight_layout()
            return fig

        n_terms = len(group_ranefs)
        if n_terms == 1:
            if figsize is None:
                figsize = (6, 5)
            fig, ax = plt.subplots(figsize=figsize)
            term_name = list(group_ranefs.keys())[0]
            plot_qq(ax, group_ranefs[term_name], f"QQ Plot: {group} / {term_name}")
            fig.tight_layout()
            return fig

        ncols = min(2, n_terms)
        nrows = (n_terms + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_terms > 1 else [axes]

        for i, (term_name, values) in enumerate(group_ranefs.items()):
            if i < len(axes):
                plot_qq(axes[i], values, f"QQ Plot: {group} / {term_name}")

        for i in range(n_terms, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        return fig

    def plot(
        self,
        which: list[int] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Create diagnostic plots for the fitted model.

        Creates a panel of residual diagnostic plots similar to R's plot() method
        for lmer objects. This is useful for assessing model assumptions such as
        homoscedasticity, normality of residuals, and detecting outliers.

        Parameters
        ----------
        which : list of int, optional
            Which plots to include. Default is [1, 2, 3, 4].
            1 = Residuals vs Fitted values
            2 = Normal Q-Q plot of residuals
            3 = Scale-Location plot (sqrt of standardized residuals vs fitted)
            4 = Residuals by Group (boxplot, only if random effects exist)
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is calculated based
            on number of plots.

        Returns
        -------
        Figure
            Matplotlib figure containing the diagnostic plots.

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> result = lmer("y ~ x + (1 | group)", data)
        >>> fig = result.plot()  # All 4 diagnostic plots
        >>> fig = result.plot(which=[1, 2])  # Only residuals vs fitted and Q-Q

        See Also
        --------
        qqmath : QQ plots of random effects (normality assessment)
        residuals : Get residuals from the fitted model
        fitted : Get fitted values from the model
        """
        from mixedlm.diagnostics.plots import plot_diagnostics

        return plot_diagnostics(self, which=which, figsize=figsize)

    def isSingular(self, tol: float = 1e-4) -> bool:
        theta_idx = 0

        for struct in self.matrices.random_structures:
            q = struct.n_terms

            if struct.correlated:
                n_theta = q * (q + 1) // 2
                theta_block = self.theta[theta_idx : theta_idx + n_theta]
                theta_idx += n_theta

                diag_idx = 0
                for i in range(q):
                    if abs(theta_block[diag_idx]) < tol:
                        return True
                    diag_idx += i + 2
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q

                if np.any(np.abs(theta_block) < tol):
                    return True

        return False

    def getME(self, name: str):
        """Extract model components by name.

        This method provides access to internal model components, similar to
        R's getME() function in lme4. It's useful for advanced users who need
        access to specific matrices or parameters.

        Parameters
        ----------
        name : str
            Name of the component to extract. Valid names are:
            - "X" : Fixed effects design matrix (n x p)
            - "Z" : Random effects design matrix (n x q)
            - "Zt" : Transpose of Z (q x n)
            - "y" : Response vector
            - "beta" : Fixed effects coefficients
            - "theta" : Variance component parameters (relative covariance factors)
            - "Lambda" : Relative covariance factor (sparse, q x q)
            - "Lambdat" : Transpose of Lambda
            - "u" : Spherical random effects
            - "b" : Conditional modes of random effects (same as u for LMM)
            - "sigma" : Residual standard deviation
            - "n" or "n_obs" : Number of observations
            - "p" or "n_fixed" : Number of fixed effects
            - "q" or "n_random" : Number of random effects
            - "lower" : Lower bounds for theta
            - "weights" : Prior weights
            - "offset" : Offset term
            - "REML" : Whether REML estimation was used
            - "deviance" : Deviance (or REML criterion)
            - "flist" : List of grouping factors
            - "cnms" : Component names for random effects
            - "Gp" : Group pointers (cumulative number of random effects per group)

        Returns
        -------
        The requested component. Type depends on the component name.

        Raises
        ------
        ValueError
            If an unknown component name is requested.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> X = result.getME("X")  # Fixed effects design matrix
        >>> theta = result.getME("theta")  # Variance parameters
        >>> Lambda = result.getME("Lambda")  # Relative covariance factor
        """
        if name == "X":
            return self.matrices.X
        elif name == "Z":
            return self.matrices.Z
        elif name == "Zt":
            return self.matrices.Zt
        elif name == "y":
            return self.matrices.y
        elif name == "beta":
            return self.beta.copy()
        elif name == "theta":
            return self.theta.copy()
        elif name == "Lambda":
            return _build_lambda(self.theta, self.matrices.random_structures)
        elif name == "Lambdat":
            Lambda = _build_lambda(self.theta, self.matrices.random_structures)
            return Lambda.T
        elif name == "u" or name == "b":
            return self.u.copy()
        elif name == "sigma":
            return self.sigma
        elif name in ("n", "n_obs"):
            return self.matrices.n_obs
        elif name in ("p", "n_fixed"):
            return self.matrices.n_fixed
        elif name in ("q", "n_random"):
            return self.matrices.n_random
        elif name == "lower":
            bounds = []
            for struct in self.matrices.random_structures:
                q = struct.n_terms
                if struct.correlated:
                    for i in range(q):
                        for j in range(i + 1):
                            if i == j:
                                bounds.append(0.0)
                            else:
                                bounds.append(-np.inf)
                else:
                    bounds.extend([0.0] * q)
            return np.array(bounds)
        elif name == "weights":
            return self.matrices.weights.copy()
        elif name == "offset":
            return self.matrices.offset.copy()
        elif name == "REML":
            return self.REML
        elif name == "deviance":
            return self.deviance
        elif name == "flist":
            return [s.grouping_factor for s in self.matrices.random_structures]
        elif name == "cnms":
            return {s.grouping_factor: s.term_names for s in self.matrices.random_structures}
        elif name == "Gp":
            gp = [0]
            for s in self.matrices.random_structures:
                gp.append(gp[-1] + s.n_levels * s.n_terms)
            return np.array(gp)
        else:
            valid_names = [
                "X", "Z", "Zt", "y", "beta", "theta", "Lambda", "Lambdat",
                "u", "b", "sigma", "n", "n_obs", "p", "n_fixed", "q", "n_random",
                "lower", "weights", "offset", "REML", "deviance", "flist", "cnms", "Gp"
            ]
            raise ValueError(
                f"Unknown component name: '{name}'. Valid names are: {valid_names}"
            )

    def update(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        REML: bool | None = None,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        **kwargs,
    ) -> LmerResult:
        """Update and re-fit the model with modified arguments.

        This method allows updating the model formula, data, or other arguments
        and refitting. It's similar to R's update() function.

        Parameters
        ----------
        formula : str, optional
            New formula. If None, uses the original formula.
            Use "." to refer to the original formula components:
            - ". ~ . + newvar" adds a fixed effect
            - ". ~ . - oldvar" removes a fixed effect
        data : DataFrame, optional
            New data. If None, uses the original data (must be stored).
        REML : bool, optional
            Whether to use REML. If None, uses the original setting.
        weights : array-like, optional
            New weights. If None, uses the original weights.
        offset : array-like, optional
            New offset. If None, uses the original offset.
        **kwargs
            Additional arguments passed to lmer().

        Returns
        -------
        LmerResult
            New fitted model result.

        Raises
        ------
        ValueError
            If data is needed but not available.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> # Add another fixed effect
        >>> result2 = result.update(". ~ . + z")
        >>> # Change to ML estimation
        >>> result3 = result.update(REML=False)
        >>> # Fit with new data
        >>> result4 = result.update(data=new_data)
        """

        if data is None:
            if self.matrices.frame is not None:
                data = self.matrices.frame
            else:
                raise ValueError(
                    "No data available. Either provide data or ensure model_frame was stored."
                )

        new_formula = str(self.formula) if formula is None else self._update_formula(formula)

        if REML is None:
            REML = self.REML

        data_size_changed = len(data) != self.matrices.n_obs

        if weights is None and not data_size_changed:
            weights = self.matrices.weights
        if offset is None and not data_size_changed:
            offset = self.matrices.offset

        return lmer(
            new_formula,
            data,
            REML=REML,
            weights=weights,
            offset=offset,
            **kwargs
        )

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

    def refit(
        self,
        newresp: NDArray[np.floating],
        **kwargs,
    ) -> LmerResult:
        """Refit the model with a new response vector.

        This method refits the model using the same formula, random effects
        structure, and model matrices, but with a different response vector.
        This is useful for simulation studies, bootstrapping, and permutation
        tests where you want to fit the same model structure repeatedly with
        different response data.

        Parameters
        ----------
        newresp : array-like
            New response vector. Must have the same length as the original
            response (after NA removal if applicable).
        **kwargs
            Additional arguments passed to the optimizer (start, method, maxiter).

        Returns
        -------
        LmerResult
            New fitted model result with the new response.

        Raises
        ------
        ValueError
            If newresp has incorrect length.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> # Simulate new responses
        >>> new_y = result.simulate()
        >>> # Refit with simulated response
        >>> result2 = result.refit(new_y)

        Notes
        -----
        The refit uses the same model matrices (X, Z) and random effects
        structure as the original fit. Only the response vector y is replaced.
        This is more efficient than calling lmer() again because the design
        matrices don't need to be reconstructed.

        See Also
        --------
        simulate : Simulate responses from the fitted model.
        update : Update and refit with modified formula or data.
        refitML : Refit a REML model with ML estimation.
        """
        newresp = np.asarray(newresp, dtype=np.float64)

        if len(newresp) != self.matrices.n_obs:
            raise ValueError(
                f"newresp has length {len(newresp)}, but model has "
                f"{self.matrices.n_obs} observations"
            )

        new_matrices = ModelMatrices(
            y=newresp,
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

        optimizer = LMMOptimizer(
            new_matrices,
            REML=self.REML,
            verbose=0,
        )

        start = kwargs.pop("start", self.theta)

        opt_result = optimizer.optimize(start=start, **kwargs)

        return LmerResult(
            formula=self.formula,
            matrices=new_matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=self.REML,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
        )

    def logLik(self) -> LogLik:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        n_theta = _count_theta(self.matrices.random_structures)
        df = p + n_theta + 1

        if self.REML:
            value = -0.5 * (self.deviance + (n - p) * np.log(2 * np.pi))
        else:
            value = -0.5 * (self.deviance + n * np.log(2 * np.pi))

        return LogLik(value=value, df=df, nobs=n, REML=self.REML)

    def get_deviance(self) -> float:
        """Get the deviance of the fitted model.

        For linear mixed models, this returns the profiled deviance
        (or REML criterion if REML=True) which is minimized during fitting.

        Returns
        -------
        float
            The deviance value.

        See Also
        --------
        REMLcrit : Get the REML criterion value.
        logLik : Get the log-likelihood.
        """
        return self.deviance

    def REMLcrit(self) -> float:
        """Get the REML criterion value.

        Returns the REML criterion if the model was fit with REML=True,
        otherwise returns the ML deviance. This is the objective function
        value that was minimized during model fitting.

        Returns
        -------
        float
            The REML criterion (if REML=True) or ML deviance (if REML=False).

        Notes
        -----
        The REML criterion is related to the restricted log-likelihood by:
            REML_crit = -2 * log(L_REML) + constant

        For ML estimation, this returns the same value as `get_deviance()`.

        See Also
        --------
        get_deviance : Get the deviance value.
        logLik : Get the log-likelihood.
        isREML : Check if the model was fit with REML.
        """
        return self.deviance

    def AIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + 2 * ll.df

    def BIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + ll.df * np.log(ll.nobs)

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "Wald",
        n_boot: int = 1000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        from mixedlm.inference.bootstrap import bootstrap_lmer
        from mixedlm.inference.profile import profile_lmer

        if parm is None:
            parm = self.matrices.fixed_names
        elif isinstance(parm, str):
            parm = [parm]

        if method == "Wald":
            vcov = self.vcov()
            alpha = 1 - level
            z_crit = stats.norm.ppf(1 - alpha / 2)

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

        elif method == "profile":
            profiles = profile_lmer(self, which=parm, level=level)
            return {p: (profiles[p].ci_lower, profiles[p].ci_upper) for p in parm if p in profiles}

        elif method == "boot":
            boot_result = bootstrap_lmer(self, n_boot=n_boot, seed=seed)
            return boot_result.ci(level=level)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'Wald', 'profile', or 'boot'.")

    def simulate(
        self,
        nsim: int = 1,
        seed: int | None = None,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        if seed is not None:
            np.random.seed(seed)

        n = self.matrices.n_obs

        if nsim == 1:
            return self._simulate_once(use_re, re_form)

        result = np.zeros((n, nsim), dtype=np.float64)
        for i in range(nsim):
            result[:, i] = self._simulate_once(use_re, re_form)

        return result

    def _simulate_once(
        self,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        n = self.matrices.n_obs
        q = self.matrices.n_random

        fixed_part = self.matrices.X @ self.beta

        if re_form == "~0" or re_form == "NA" or not use_re or q == 0:
            random_part = np.zeros(n)
        else:
            u_new = np.zeros(q, dtype=np.float64)
            u_idx = 0
            theta_start = 0

            for struct in self.matrices.random_structures:
                n_levels = struct.n_levels
                n_terms = struct.n_terms

                n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms
                theta_block = self.theta[theta_start : theta_start + n_theta]

                if struct.correlated:
                    L = np.zeros((n_terms, n_terms))
                    idx = 0
                    for i in range(n_terms):
                        for j in range(i + 1):
                            L[i, j] = theta_block[idx]
                            idx += 1
                    cov = L @ L.T * self.sigma**2
                else:
                    cov = np.diag(theta_block**2) * self.sigma**2

                for g in range(n_levels):
                    b_g = np.random.multivariate_normal(np.zeros(n_terms), cov)
                    for j in range(n_terms):
                        u_new[u_idx + g * n_terms + j] = b_g[j]

                u_idx += n_levels * n_terms
                theta_start += n_theta

            random_part = self.matrices.Z @ u_new

        noise = np.random.randn(n) * self.sigma

        return fixed_part + random_part + noise

    def refitML(self, **kwargs) -> LmerResult:
        """Refit the model using ML instead of REML.

        This method refits a REML model using maximum likelihood estimation.
        This is useful for likelihood ratio tests comparing models with
        different fixed effects, where REML-based comparisons are not valid.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the optimizer (start, method, maxiter).

        Returns
        -------
        LmerResult
            New fitted model result with REML=False.
            If the model was already fit with ML (REML=False), returns self.

        Notes
        -----
        Likelihood ratio tests for comparing models with different fixed
        effects should use ML estimation, not REML. This is because REML
        estimates the variance components after profiling out the fixed
        effects, making the REML likelihoods not comparable when fixed
        effects differ.

        Examples
        --------
        >>> # Fit two models with REML
        >>> m1 = lmer("y ~ x1 + (1|group)", data)
        >>> m2 = lmer("y ~ x1 + x2 + (1|group)", data)
        >>> # Refit with ML for valid LRT comparison
        >>> m1_ml = m1.refitML()
        >>> m2_ml = m2.refitML()
        >>> # Now can compare likelihoods
        >>> from scipy import stats
        >>> lr_stat = -2 * (m1_ml.logLik().value - m2_ml.logLik().value)
        >>> p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        See Also
        --------
        refit : Refit with a new response vector.
        update : Update and refit with modified arguments.
        isREML : Check if the model was fit with REML.
        """
        if not self.REML:
            return self

        new_matrices = ModelMatrices(
            y=self.matrices.y,
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

        optimizer = LMMOptimizer(
            new_matrices,
            REML=False,
            verbose=0,
        )

        start = kwargs.pop("start", self.theta)

        opt_result = optimizer.optimize(start=start, **kwargs)

        return LmerResult(
            formula=self.formula,
            matrices=new_matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=False,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
        )

    def isREML(self) -> bool:
        """Check if the model was fit using REML."""
        return self.REML

    def isGLMM(self) -> bool:
        """Check if this is a generalized linear mixed model.

        Always returns False for LmerResult.
        """
        return False

    def isLMM(self) -> bool:
        """Check if this is a linear mixed model.

        Always returns True for LmerResult.
        """
        return True

    def isNLMM(self) -> bool:
        """Check if this is a nonlinear mixed model.

        Always returns False for LmerResult.
        """
        return False

    def npar(self) -> int:
        """Get the number of parameters in the model.

        Returns the total number of estimated parameters:
        - Fixed effects (beta)
        - Variance-covariance parameters (theta)
        - Residual standard deviation (sigma)

        Returns
        -------
        int
            Total number of parameters.
        """
        n_fixed = len(self.beta)
        n_theta = len(self.theta)
        n_sigma = 1
        return n_fixed + n_theta + n_sigma

    def drop1(self, data: pd.DataFrame, test: str = "Chisq"):
        from mixedlm.inference.drop1 import drop1_lmer

        return drop1_lmer(self, data, test=test)

    def allFit(
        self,
        data: pd.DataFrame,
        optimizers: list[str] | None = None,
        verbose: bool = False,
    ):
        from mixedlm.inference.allfit import allfit_lmer

        return allfit_lmer(self, data, optimizers=optimizers, verbose=verbose)

    def summary(self) -> str:
        lines = []
        lines.append("Linear mixed model fit by " + ("REML" if self.REML else "ML"))
        lines.append(f"Formula: {self.formula}")
        lines.append("")

        lines.append(str(self.VarCorr()))
        lines.append(f"Number of obs: {self.matrices.n_obs}")
        for struct in self.matrices.random_structures:
            lines.append(f"  groups:  {struct.grouping_factor}, {struct.n_levels}")
        lines.append("")

        lines.append("Fixed effects:")
        vcov = self.vcov()
        se = np.sqrt(np.diag(vcov))

        lines.append("             Estimate  Std. Error  t value")
        for i, name in enumerate(self.matrices.fixed_names):
            t_val = self.beta[i] / se[i] if se[i] > 0 else np.nan
            lines.append(f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {t_val:7.3f}")

        lines.append("")
        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"LmerResult(formula={self.formula}, deviance={self.deviance:.4f})"


class LmerMod:
    def __init__(
        self,
        formula: str | Formula,
        data: pd.DataFrame,
        REML: bool = True,
        verbose: int = 0,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        na_action: str | None = "omit",
        contrasts: dict[str, str | NDArray[np.floating]] | None = None,
        control: LmerControl | None = None,
    ) -> None:
        from mixedlm.models.control import LmerControl

        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.REML = REML
        self.verbose = verbose
        self.na_action = na_action
        self.contrasts = contrasts
        self.control = control if control is not None else LmerControl()

        self.matrices = build_model_matrices(
            self.formula, self.data, weights=weights, offset=offset,
            na_action=na_action, contrasts=contrasts
        )

    def fit(
        self,
        start: NDArray[np.floating] | None = None,
        method: str | None = None,
        maxiter: int | None = None,
    ) -> LmerResult:
        import warnings

        ctrl = self.control
        opt_method = method if method is not None else ctrl.optimizer
        opt_maxiter = maxiter if maxiter is not None else ctrl.maxiter

        optimizer = LMMOptimizer(
            self.matrices,
            REML=self.REML,
            verbose=self.verbose,
            use_rust=ctrl.use_rust,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=opt_method,
            maxiter=opt_maxiter,
        )

        result = LmerResult(
            formula=self.formula,
            matrices=self.matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=self.REML,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
        )

        if ctrl.check_conv and not result.converged:
            warnings.warn(
                "Model failed to converge. Consider increasing maxiter or "
                "trying a different optimizer.",
                category=UserWarning,
                stacklevel=2,
            )

        if ctrl.check_singular and result.isSingular(tol=ctrl.boundary_tol):
            warnings.warn(
                "Model is singular (boundary fit). Some variance components "
                "are estimated as zero or near-zero.",
                category=UserWarning,
                stacklevel=2,
            )

        return result


def lmer(
    formula: str,
    data: pd.DataFrame,
    REML: bool = True,
    verbose: int = 0,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    control: LmerControl | None = None,
    **kwargs,
) -> LmerResult:
    """Fit a linear mixed-effects model.

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    REML : bool, default True
        Use REML estimation. Set to False for ML estimation.
    verbose : int, default 0
        Verbosity level for optimization output.
    weights : array-like, optional
        Prior weights for observations.
    offset : array-like, optional
        Offset term for the linear predictor.
    na_action : str, optional
        How to handle missing values. Options:
        - "omit" (default): Remove rows with any NA values
        - "exclude": Like omit, but fitted/residuals return NA for removed rows
        - "fail": Raise an error if any NA values are present
    contrasts : dict, optional
        Contrast coding for categorical variables. Keys are variable names,
        values can be:
        - "treatment" (default): Treatment/dummy contrasts
        - "sum": Sum/deviation contrasts
        - "helmert": Helmert contrasts
        - "poly": Polynomial contrasts for ordered factors
        - A custom contrast matrix (NDArray)
    control : LmerControl, optional
        Control parameters for the optimizer. Use lmerControl() to create.
        If not provided, default settings are used.
    **kwargs
        Additional arguments passed to the optimizer (start, method, maxiter).

    Returns
    -------
    LmerResult
        Fitted model result.

    Examples
    --------
    >>> result = lmer("y ~ x + (1|group)", data)

    >>> from mixedlm import lmerControl
    >>> ctrl = lmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = lmer("y ~ x + (1|group)", data, control=ctrl)
    """
    model = LmerMod(
        formula, data, REML=REML, verbose=verbose,
        weights=weights, offset=offset, na_action=na_action,
        contrasts=contrasts, control=control
    )
    return model.fit(**kwargs)

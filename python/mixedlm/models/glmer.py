from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

    from mixedlm.models.control import GlmerControl

from mixedlm.estimation.laplace import GLMMOptimizer, _build_lambda, _count_theta
from mixedlm.families.base import Family
from mixedlm.families.binomial import Binomial
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices
from mixedlm.models.lmer import (
    LogLik,
    MerResultMixin,
    PredictResult,
    RanefResult,
    VarCorrGroup,
    _resolve_optional_vector,
)
from mixedlm.utils import _get_signif_code


@dataclass
class GlmerVarCorr:
    groups: dict[str, VarCorrGroup]

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
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_groups = len(self.groups)
        return f"GlmerVarCorr({n_groups} groups)"

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {name: group.variance for name, group in self.groups.items()}

    def get_cov(self, group: str) -> NDArray[np.floating]:
        return self.groups[group].cov

    def get_corr(self, group: str) -> NDArray[np.floating] | None:
        return self.groups[group].corr


@dataclass
class GlmerResult(MerResultMixin):
    formula: Formula
    matrices: ModelMatrices
    family: Family
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    u: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int
    nAGQ: int

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

        eta = self.linear_predictor()
        mu = self.family.link.inverse(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = self.family.weights(mu)
        W = np.maximum(W, 1e-10)
        W_diag = sparse.diags(W, format="csc")

        Zt = self.matrices.Zt
        ZtWZ = Zt @ W_diag @ self.matrices.Z
        LambdatZtWZLambda = Lambda.T @ ZtWZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V = LambdatZtWZLambda + I_q

        V_dense = V.toarray() if sparse.issparse(V) else V
        Lambda_dense = Lambda.toarray() if sparse.issparse(Lambda) else Lambda

        try:
            V_inv_Lambda_t = linalg.solve(V_dense, Lambda_dense.T, assume_a="pos")
        except linalg.LinAlgError:
            V_inv_Lambda_t = linalg.lstsq(V_dense, Lambda_dense.T)[0]

        cond_cov = Lambda_dense @ V_inv_Lambda_t

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

    def get_sigma(self) -> float:
        return 1.0

    @property
    def sigma(self) -> float:
        return 1.0

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

    def get_family(self) -> Family:
        """Get the GLM family.

        Returns the family object used for the generalized
        linear mixed model, including the link function.

        Returns
        -------
        Family
            The GLM family (e.g., Binomial, Poisson, Gaussian).

        Examples
        --------
        >>> result = glmer("y ~ x + (1|g)", data, family=Binomial())
        >>> result.get_family()
        Binomial(link=logit)
        >>> result.get_family().link.name
        'logit'
        """
        return self.family

    def model_matrix(self, type: str = "fixed") -> NDArray[np.floating] | sparse.csc_matrix:
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
        """
        if type in ("fixed", "X"):
            return self.matrices.X
        elif type in ("random", "Z"):
            return self.matrices.Z
        elif type == "both":
            return (self.matrices.X, self.matrices.Z)
        else:
            raise ValueError(f"Unknown type '{type}'. Use 'fixed', 'random', 'X', 'Z', or 'both'.")

    def terms(self):
        """Get information about the model terms.

        Returns a ModelTerms object containing information about the
        response variable, fixed effect terms, random effect terms,
        and grouping factors.

        Returns
        -------
        ModelTerms
            Object containing term information.
        """
        from mixedlm.formula.terms import InteractionTerm, VariableTerm
        from mixedlm.models.lmer import ModelTerms

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
        >>> result = glmer("y ~ x + (1 | group)", data, family=Binomial())
        >>> mf = result.model_frame()
        >>> print(mf.columns.tolist())  # ['y', 'x', 'group']
        """
        if self.matrices.frame is not None:
            return self.matrices.frame.copy()

        return pd.DataFrame({"y": self.matrices.y})

    @cached_property
    def _linear_predictor(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    def _should_expand_na(self) -> bool:
        from mixedlm.utils.na_action import NAAction

        return (
            self.matrices.na_info is not None
            and self.matrices.na_info.action == NAAction.EXCLUDE
            and self.matrices.na_info.n_omitted > 0
        )

    def linear_predictor(self, na_expand: bool = True) -> NDArray[np.floating]:
        values = self._linear_predictor
        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(values)
        return values

    def fitted(self, type: str = "response", na_expand: bool = True) -> NDArray[np.floating]:
        """Get fitted values.

        Parameters
        ----------
        type : str, default "response"
            Type of fitted values: "response" (mean) or "link" (linear predictor).
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Fitted values.
        """
        eta = self._linear_predictor
        values = eta if type == "link" else self.family.link.inverse(eta)

        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(values)
        return values

    def residuals(self, type: str = "deviance", na_expand: bool = True) -> NDArray[np.floating]:
        """Get residuals.

        Parameters
        ----------
        type : str, default "deviance"
            Type of residuals: "response", "pearson", or "deviance".
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Residuals.
        """
        mu = self.fitted(type="response", na_expand=False)

        if type == "response":
            resid = self.matrices.y - mu
        elif type == "pearson":
            var = self.family.variance(mu)
            resid = (self.matrices.y - mu) / np.sqrt(var)
        elif type == "deviance":
            dev_resids = self.family.deviance_resids(self.matrices.y, mu, self.matrices.weights)
            signs = np.sign(self.matrices.y - mu)
            resid = signs * np.sqrt(np.abs(dev_resids))
        else:
            raise ValueError(f"Unknown residual type: {type}")

        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(resid)
        return resid

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        type: str = "response",
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
        type : str, default "response"
            Type of prediction: "response" (mean) or "link" (linear predictor).
        re_form : str, optional
            Formula for random effects. Use "NA" or "~0" for fixed effects only.
        allow_new_levels : bool, default False
            Allow new levels in grouping factors (predicts with RE=0).
        se_fit : bool, default False
            If True, return standard errors of predictions.
        interval : str, default "none"
            Type of interval: "none" or "confidence".
            Note: prediction intervals not available for GLMMs.
        level : float, default 0.95
            Confidence level for intervals.

        Returns
        -------
        NDArray or PredictResult
            Predictions. Returns PredictResult if se_fit=True or interval!="none".
        """
        include_re = re_form != "NA" and re_form != "~0"

        if newdata is None:
            if not se_fit and interval == "none":
                return self.fitted(type=type)
            eta = self._linear_predictor.copy()
            X = self.matrices.X
            new_matrices = self.matrices
        else:
            new_matrices = build_model_matrices(self.formula, newdata)
            X = new_matrices.X
            eta = X @ self.beta

            if new_matrices.offset is not None:
                eta = eta + new_matrices.offset

            if include_re:
                eta = self._add_random_effects_to_eta(eta, newdata, new_matrices, allow_new_levels)

        if not se_fit and interval == "none":
            if type == "link":
                return eta
            else:
                return self.family.link.inverse(eta)

        vcov_beta = self.vcov()
        var_eta = np.sum((X @ vcov_beta) * X, axis=1)
        se_eta = np.sqrt(var_eta)

        if interval == "none":
            if type == "link":
                return PredictResult(fit=eta, se_fit=se_eta, interval="none", level=level)
            else:
                mu = self.family.link.inverse(eta)
                deriv = self.family.link.deriv(mu)
                se_mu = se_eta / np.abs(deriv)
                return PredictResult(fit=mu, se_fit=se_mu, interval="none", level=level)

        if interval == "prediction":
            raise ValueError(
                "Prediction intervals not available for GLMMs. Use interval='confidence'."
            )

        z_crit = stats.norm.ppf(1 - (1 - level) / 2)

        if interval == "confidence":
            if type == "link":
                lower = eta - z_crit * se_eta
                upper = eta + z_crit * se_eta
                return PredictResult(
                    fit=eta,
                    se_fit=se_eta,
                    lower=lower,
                    upper=upper,
                    interval="confidence",
                    level=level,
                )
            else:
                eta_lower = eta - z_crit * se_eta
                eta_upper = eta + z_crit * se_eta
                mu = self.family.link.inverse(eta)
                lower = self.family.link.inverse(eta_lower)
                upper = self.family.link.inverse(eta_upper)
                deriv = self.family.link.deriv(mu)
                se_mu = se_eta / np.abs(deriv)
                return PredictResult(
                    fit=mu,
                    se_fit=se_mu,
                    lower=lower,
                    upper=upper,
                    interval="confidence",
                    level=level,
                )
        else:
            raise ValueError(f"Unknown interval type: {interval}. Use 'none' or 'confidence'.")

    def _add_random_effects_to_eta(
        self,
        eta: NDArray[np.floating],
        newdata: pd.DataFrame,
        new_matrices: ModelMatrices,
        allow_new_levels: bool,
    ) -> NDArray[np.floating]:
        """Add random effects contribution to linear predictor."""
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
                        eta[j] += u_block[level_idx, i] * term_values[j]
                    elif not allow_new_levels:
                        raise ValueError(
                            f"New level '{group_level}' in grouping factor '{group_col}'. "
                            "Set allow_new_levels=True to predict with random effects = 0."
                        )

        return eta

    def vcov(self) -> NDArray[np.floating]:
        q = self.matrices.n_random
        Lambda = _build_lambda(self.theta, self.matrices.random_structures)

        eta = self.linear_predictor()
        mu = self.family.link.inverse(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = self.family.weights(mu)
        W = np.maximum(W, 1e-10)
        W_diag = sparse.diags(W, format="csc")

        XtWX = self.matrices.X.T @ W_diag @ self.matrices.X

        if q > 0:
            Zt = self.matrices.Zt
            XtWZ = self.matrices.X.T @ W_diag @ self.matrices.Z
            ZtWZ = Zt @ W_diag @ self.matrices.Z

            LambdatLambda = Lambda.T @ Lambda
            if sparse.issparse(LambdatLambda):
                LambdatLambda = LambdatLambda.toarray()
            if sparse.issparse(ZtWZ):
                ZtWZ = ZtWZ.toarray()

            C = ZtWZ + LambdatLambda
            try:
                L_C = linalg.cholesky(C, lower=True)
            except linalg.LinAlgError:
                C += 1e-6 * np.eye(q)
                L_C = linalg.cholesky(C, lower=True)

            if sparse.issparse(XtWZ):
                XtWZ = XtWZ.toarray()

            RZX = linalg.solve_triangular(L_C, XtWZ.T, lower=True)
            XtVinvX = XtWX - RZX.T @ RZX
        else:
            XtVinvX = XtWX

        p = XtVinvX.shape[0]
        try:
            L = linalg.cholesky(XtVinvX, lower=True)
            return linalg.cho_solve((L, True), np.eye(p))
        except linalg.LinAlgError:
            try:
                return linalg.solve(XtVinvX, np.eye(p))
            except linalg.LinAlgError:
                return linalg.pinv(XtVinvX)

    def hatvalues(self) -> NDArray[np.floating]:
        """Compute leverage values (diagonal of the hat matrix).

        For GLMMs, the hat matrix is computed using the working weights
        from the iteratively reweighted least squares algorithm.

        Returns
        -------
        NDArray
            Leverage values for each observation, between 0 and 1.
            Values close to 1 indicate high-leverage observations.

        Notes
        -----
        For generalized linear mixed models, the hat matrix incorporates
        both fixed and random effects, weighted by the variance function.
        """
        q = self.matrices.n_random
        X = self.matrices.X

        eta = self.linear_predictor()
        mu = self.family.link.inverse(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = self.family.weights(mu)
        W = np.maximum(W, 1e-10)
        sqrt_W = np.sqrt(W)

        if q == 0:
            XW = X * sqrt_W[:, np.newaxis]
            XtWX = XW.T @ XW
            try:
                L = linalg.cholesky(XtWX, lower=True)
                XtWX_inv = linalg.cho_solve((L, True), np.eye(X.shape[1]))
            except linalg.LinAlgError:
                XtWX_inv = linalg.pinv(XtWX)
            h = W * np.sum((X @ XtWX_inv) * X, axis=1)
            return np.clip(h, 0, 1 - 1e-10)

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        Z = self.matrices.Z
        Zt = self.matrices.Zt
        W_diag = sparse.diags(W, format="csc")

        ZtWZ = Zt @ W_diag @ self.matrices.Z
        LambdatLambda = Lambda.T @ Lambda

        if sparse.issparse(LambdatLambda):
            LambdatLambda = LambdatLambda.toarray()
        if sparse.issparse(ZtWZ):
            ZtWZ = ZtWZ.toarray()

        C = ZtWZ + LambdatLambda

        try:
            L_C = linalg.cholesky(C, lower=True)
        except linalg.LinAlgError:
            C += 1e-6 * np.eye(q)
            L_C = linalg.cholesky(C, lower=True)

        XtWX = X.T @ W_diag @ X
        XtWZ = X.T @ W_diag @ self.matrices.Z

        if sparse.issparse(XtWZ):
            XtWZ = XtWZ.toarray()

        RZX = linalg.solve_triangular(L_C, XtWZ.T, lower=True)
        XtVinvX = XtWX - RZX.T @ RZX

        try:
            L_XVX = linalg.cholesky(XtVinvX, lower=True)
            XtVinvX_inv = linalg.cho_solve((L_XVX, True), np.eye(X.shape[1]))
        except linalg.LinAlgError:
            XtVinvX_inv = linalg.pinv(XtVinvX)

        h_fixed = W * np.sum((X @ XtVinvX_inv) * X, axis=1)

        Lambda_dense = Lambda.toarray() if sparse.issparse(Lambda) else Lambda
        Z_dense = Z.toarray() if sparse.issparse(Z) else Z
        ZLambda = Z_dense @ Lambda_dense

        C_inv = linalg.solve(C, np.eye(q), assume_a="pos")
        ZLambda_Cinv_LambdatZt = ZLambda @ C_inv @ ZLambda.T
        h_random = W * np.diag(ZLambda_Cinv_LambdatZt)

        h = h_fixed + h_random
        h = np.clip(h, 0, 1 - 1e-10)

        return h

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        For GLMMs, Cook's distance measures the influence of each observation
        on the fitted values, using Pearson residuals.

        Returns
        -------
        NDArray
            Cook's distance for each observation.

        Notes
        -----
        For GLMMs, Cook's distance is computed using Pearson residuals
        and the working weights from the IRLS algorithm.
        """
        h = self.hatvalues()
        resid = self.residuals(type="pearson")
        p = self.matrices.n_fixed

        h = np.clip(h, 0, 1 - 1e-10)

        cooks_d = (resid**2 / p) * (h / (1 - h) ** 2)

        return cooks_d

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns a dictionary containing various influence measures for
        identifying influential observations in GLMMs.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'hat': Leverage values (hatvalues)
            - 'cooks_d': Cook's distance
            - 'pearson_resid': Pearson residuals
            - 'deviance_resid': Deviance residuals

        Notes
        -----
        Influential observations are those that have a large effect on
        the model estimates. For GLMMs, deviance residuals are often
        preferred over Pearson residuals for identifying outliers.
        """
        return {
            "hat": self.hatvalues(),
            "cooks_d": self.cooks_distance(),
            "pearson_resid": self.residuals(type="pearson"),
            "deviance_resid": self.residuals(type="deviance"),
        }

    def VarCorr(self) -> GlmerVarCorr:
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

                cov = L_block @ L_block.T

                stddevs = np.sqrt(np.diag(cov))
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = cov / np.outer(stddevs, stddevs)
                    corr = np.where(np.isfinite(corr), corr, 0.0)
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q
                cov = np.diag(theta_block**2)
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

        return GlmerVarCorr(groups=groups)

    def rePCA(self):
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
        """
        from mixedlm.models.lmer import RePCA, RePCAGroup

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

                cov = L_block @ L_block.T
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q
                cov = np.diag(theta_block**2)

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
        >>> result = glmer("y ~ x + (x | group)", data, family=Binomial())
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
        for glmer objects. This is useful for assessing model assumptions and
        detecting patterns in the residuals.

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
        >>> result = glmer("y ~ x + (1 | group)", data, family=Binomial())
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
        R's getME() function in lme4.

        Parameters
        ----------
        name : str
            Name of the component to extract. Valid names are:
            - "X" : Fixed effects design matrix (n x p)
            - "Z" : Random effects design matrix (n x q)
            - "Zt" : Transpose of Z (q x n)
            - "y" : Response vector
            - "beta" : Fixed effects coefficients
            - "theta" : Variance component parameters
            - "Lambda" : Relative covariance factor (sparse, q x q)
            - "Lambdat" : Transpose of Lambda
            - "u" : Spherical random effects
            - "b" : Conditional modes of random effects
            - "n" or "n_obs" : Number of observations
            - "p" or "n_fixed" : Number of fixed effects
            - "q" or "n_random" : Number of random effects
            - "lower" : Lower bounds for theta
            - "weights" : Prior weights
            - "offset" : Offset term
            - "deviance" : Deviance
            - "flist" : List of grouping factors
            - "cnms" : Component names for random effects
            - "Gp" : Group pointers
            - "family" : GLM family

        Returns
        -------
        The requested component.

        Raises
        ------
        ValueError
            If an unknown component name is requested.

        Examples
        --------
        >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())
        >>> X = result.getME("X")
        >>> family = result.getME("family")
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
        elif name == "family":
            return self.family
        elif name == "nAGQ":
            return self.nAGQ
        elif name == "RX":
            _, R = linalg.qr(self.matrices.X, mode="economic")
            return R
        elif name == "RZX":
            return self._compute_RZX()
        elif name == "Lind":
            return self._build_Lind()
        elif name == "devcomp":
            return self._get_devcomp()
        else:
            valid_names = [
                "X",
                "Z",
                "Zt",
                "y",
                "beta",
                "theta",
                "Lambda",
                "Lambdat",
                "u",
                "b",
                "n",
                "n_obs",
                "p",
                "n_fixed",
                "q",
                "n_random",
                "lower",
                "weights",
                "offset",
                "deviance",
                "flist",
                "cnms",
                "Gp",
                "family",
                "nAGQ",
                "RX",
                "RZX",
                "Lind",
                "devcomp",
            ]
            raise ValueError(f"Unknown component name: '{name}'. Valid names are: {valid_names}")

    def _compute_RZX(self) -> NDArray[np.floating]:
        """Compute RZX, the cross-term in the mixed model equations."""
        Z = self.matrices.Z
        X = self.matrices.X
        Lambda = _build_lambda(self.theta, self.matrices.random_structures)

        if sparse.issparse(Z):
            Zt = Z.T
            ZtZ = Zt @ Z
        else:
            Zt = Z.T
            ZtZ = Zt @ Z

        LambdatLambda = Lambda.T @ Lambda
        C = ZtZ + LambdatLambda

        if sparse.issparse(C):
            C = C.toarray()

        try:
            L_C = linalg.cholesky(C, lower=True)
        except linalg.LinAlgError:
            C = C + 1e-6 * np.eye(C.shape[0])
            L_C = linalg.cholesky(C, lower=True)

        XtZ = X.T @ Z
        if sparse.issparse(XtZ):
            XtZ = XtZ.toarray()

        RZX = linalg.solve_triangular(L_C, XtZ.T, lower=True)
        return RZX

    def _build_Lind(self) -> NDArray[np.int64]:
        """Build Lind, the index mapping from theta to Lambda entries."""
        indices = []
        theta_idx = 0

        for struct in self.matrices.random_structures:
            n_terms = struct.n_terms

            if struct.correlated:
                n_theta = n_terms * (n_terms + 1) // 2
                template_indices = []
                idx = 0
                for i in range(n_terms):
                    for _j in range(i + 1):
                        template_indices.append(theta_idx + idx)
                        idx += 1
            else:
                n_theta = n_terms
                template_indices = list(range(theta_idx, theta_idx + n_terms))

            for _ in range(struct.n_levels):
                indices.extend(template_indices)

            theta_idx += n_theta

        return np.array(indices, dtype=np.int64)

    def _get_devcomp(self) -> dict[str, Any]:
        """Get deviance components and model dimensions."""
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        q = self.matrices.n_random
        n_theta = len(self.theta)

        cmp = {
            "ldL2": 0.0,
            "ldRX2": 0.0,
            "pwrss": 0.0,
            "drsum": 0.0,
            "dev": float(self.deviance),
            "ussq": float(np.sum(self.u**2)) if self.u is not None else 0.0,
        }

        dims = {
            "n": n,
            "p": p,
            "q": q,
            "nmp": n - p,
            "nth": n_theta,
            "REML": 0,
            "useSc": 0,
            "nAGQ": self.nAGQ,
            "q0": q,
            "q1": 0,
            "qrx": p,
            "ngrps": len(self.matrices.random_structures),
        }

        return {"cmp": cmp, "dims": dims}

    def update(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        family: Family | None = None,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        nAGQ: int | None = None,
        **kwargs,
    ) -> GlmerResult:
        """Update and re-fit the model with modified arguments.

        This method allows updating the model formula, data, or other arguments
        and refitting.

        Parameters
        ----------
        formula : str, optional
            New formula. If None, uses the original formula.
            Use "." to refer to the original formula components.
        data : DataFrame, optional
            New data. If None, uses the original data (must be stored).
        family : Family, optional
            New GLM family. If None, uses the original family.
        weights : array-like, optional
            New weights. If None, uses the original weights.
        offset : array-like, optional
            New offset. If None, uses the original offset.
        nAGQ : int, optional
            Number of quadrature points. If None, uses original.
        **kwargs
            Additional arguments passed to glmer().

        Returns
        -------
        GlmerResult
            New fitted model result.

        Raises
        ------
        ValueError
            If data is needed but not available.

        Examples
        --------
        >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())
        >>> # Change family
        >>> result2 = result.update(family=Poisson())
        >>> # Add a term
        >>> result3 = result.update(". ~ . + z")
        """

        if data is None:
            if self.matrices.frame is not None:
                data = self.matrices.frame
            else:
                raise ValueError(
                    "No data available. Either provide data or ensure model_frame was stored."
                )

        new_formula = str(self.formula) if formula is None else self._update_formula(formula)

        if family is None:
            family = self.family

        data_size_changed = len(data) != self.matrices.n_obs

        if weights is None and not data_size_changed:
            weights = self.matrices.weights
        if offset is None and not data_size_changed:
            offset = self.matrices.offset
        if nAGQ is None:
            nAGQ = self.nAGQ

        return glmer(
            new_formula, data, family=family, weights=weights, offset=offset, nAGQ=nAGQ, **kwargs
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

    def logLik(self) -> LogLik:
        n = self.matrices.n_obs
        n_theta = _count_theta(self.matrices.random_structures)
        df = self.matrices.n_fixed + n_theta
        value = -0.5 * self.deviance

        return LogLik(value=value, df=df, nobs=n, REML=False)

    def get_deviance(self) -> float:
        """Get the deviance of the fitted model.

        For generalized linear mixed models, this returns the Laplace
        approximation to the deviance.

        Returns
        -------
        float
            The deviance value.

        See Also
        --------
        logLik : Get the log-likelihood.
        """
        return self.deviance

    def REMLcrit(self) -> float:
        """Get the ML deviance (GLMMs do not use REML).

        For generalized linear mixed models, REML estimation is not used.
        This method returns the ML deviance for API compatibility with
        LmerResult.

        Returns
        -------
        float
            The ML deviance.

        Notes
        -----
        Unlike linear mixed models, GLMMs are always fit using maximum
        likelihood (via Laplace approximation or adaptive Gauss-Hermite
        quadrature). This method exists for API consistency with LmerResult.

        See Also
        --------
        get_deviance : Get the deviance value.
        logLik : Get the log-likelihood.
        isREML : Check if the model was fit with REML (always False for GLMMs).
        """
        return self.deviance

    def AIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + 2 * ll.df

    def BIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + ll.df * np.log(ll.nobs)

    def extractAIC(self) -> tuple[float, float]:
        """Extract AIC with effective degrees of freedom.

        Returns the effective degrees of freedom and AIC value,
        matching the interface of R's extractAIC function.

        Returns
        -------
        tuple of (float, float)
            (edf, AIC) where edf is the effective degrees of freedom.
        """
        ll = self.logLik()
        edf = float(ll.df)
        aic = float(-2 * ll.value + 2 * ll.df)
        return (edf, aic)

    def get_formula(
        self,
        random_only: bool = False,
        fixed_only: bool = False,
    ) -> Formula | str:
        """Get the model formula.

        When called without arguments, returns the Formula object.
        When random_only or fixed_only is specified, returns a string.

        Parameters
        ----------
        random_only : bool, default False
            If True, return only the random effects part as a string.
        fixed_only : bool, default False
            If True, return only the fixed effects part as a string.

        Returns
        -------
        Formula or str
            The Formula object (default), or a string if random_only
            or fixed_only is specified.
        """
        from mixedlm.formula.parser import (
            getFixedFormulaStr,
            getRandomFormulaStr,
        )

        if random_only and fixed_only:
            raise ValueError("Cannot specify both random_only and fixed_only")

        if random_only:
            return getRandomFormulaStr(str(self.formula))
        elif fixed_only:
            return getFixedFormulaStr(str(self.formula))
        else:
            return self.formula

    def as_function(
        self,
        type: str = "deviance",
    ) -> object:
        """Return the model's objective function.

        Parameters
        ----------
        type : str, default "deviance"
            Type of function to return:
            - "deviance": returns the deviance function
            - "predict": returns a prediction function (linear predictor)

        Returns
        -------
        callable
            The requested function.
        """
        from mixedlm.estimation.laplace import GLMMOptimizer

        if type == "deviance":
            optimizer = GLMMOptimizer(
                self.matrices,
                self.family,
                verbose=0,
                nAGQ=self.nAGQ,
            )
            return optimizer.objective
        elif type == "predict":

            def predict_fn(X: NDArray[np.floating]) -> NDArray[np.floating]:
                return X @ self.beta

            return predict_fn
        else:
            raise ValueError(f"Unknown type: {type}. Use 'deviance' or 'predict'.")

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "Wald",
        n_boot: int = 1000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        from scipy import stats

        from mixedlm.inference.bootstrap import bootstrap_glmer
        from mixedlm.inference.profile import profile_glmer

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
            profiles = profile_glmer(self, which=parm, level=level)
            return {p: (profiles[p].ci_lower, profiles[p].ci_upper) for p in parm if p in profiles}

        elif method == "boot":
            boot_result = bootstrap_glmer(self, n_boot=n_boot, seed=seed)
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

        if re_form == "~0" or re_form == "NA" or not use_re or q == 0:
            eta = self.matrices.X @ self.beta
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
                    cov = L @ L.T
                else:
                    cov = np.diag(theta_block**2)

                for g in range(n_levels):
                    b_g = np.random.multivariate_normal(
                        np.zeros(n_terms), cov + 1e-8 * np.eye(n_terms)
                    )
                    for j in range(n_terms):
                        u_new[u_idx + g * n_terms + j] = b_g[j]

                u_idx += n_levels * n_terms
                theta_start += n_theta

            eta = self.matrices.X @ self.beta + self.matrices.Z @ u_new

        mu = self.family.link.inverse(eta)

        family_name = self.family.__class__.__name__

        if family_name == "Binomial":
            mu = np.clip(mu, 1e-6, 1 - 1e-6)
            y_sim = np.random.binomial(1, mu).astype(np.float64)
        elif family_name == "Poisson":
            mu = np.clip(mu, 1e-6, 1e15)
            y_sim = np.random.poisson(mu).astype(np.float64)
        elif family_name == "NegativeBinomial":
            mu = np.clip(mu, 1e-6, 1e10)
            theta = self.family.theta  # type: ignore[attr-defined]
            y_sim = np.random.negative_binomial(theta, theta / (mu + theta)).astype(np.float64)
        elif family_name == "Gamma":
            mu = np.clip(mu, 1e-6, 1e10)
            shape = 1.0
            y_sim = np.random.gamma(shape, mu / shape, n)
        elif family_name == "InverseGaussian":
            mu = np.clip(mu, 1e-6, 1e10)
            y_sim = np.random.wald(mu, 1.0, n)
        elif family_name == "Gaussian":
            y_sim = np.random.normal(mu, 1.0)
        else:
            y_sim = mu + np.random.randn(n) * 0.1

        return y_sim

    def refitML(self) -> GlmerResult:
        """Return self since GLMMs are always fit with ML.

        GLMMs do not use REML estimation, so this method simply returns
        the current result unchanged. It exists for API consistency with
        LmerResult.

        Returns
        -------
        GlmerResult
            Returns self (GLMMs are already fit with ML).
        """
        return self

    def refit(
        self,
        newresp: NDArray[np.floating] | None = None,
        **kwargs,
    ) -> GlmerResult:
        """Refit the model with a new response vector.

        This method refits the model using the same formula and design matrices
        but with a different response vector. This is useful for simulation
        studies, bootstrap, and permutation tests.

        Parameters
        ----------
        newresp : array-like, optional
            New response values. Must have the same length as the original
            response. If None, refits with the original response.
        **kwargs
            Additional arguments passed to the optimizer (start, method, maxiter).

        Returns
        -------
        GlmerResult
            New fitted model result with the updated response.

        Examples
        --------
        >>> result = glmer("y ~ x + (1|group)", data, family=families.Binomial())
        >>> # Refit with simulated response
        >>> y_sim = result.simulate()
        >>> result_sim = result.refit(newresp=y_sim)

        See Also
        --------
        simulate : Simulate response from the fitted model.
        """
        if newresp is None:
            newresp = self.matrices.y
        else:
            newresp = np.asarray(newresp, dtype=np.float64)
            if len(newresp) != self.matrices.n_obs:
                raise ValueError(
                    f"newresp has length {len(newresp)}, expected {self.matrices.n_obs}"
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

        optimizer = GLMMOptimizer(
            new_matrices,
            self.family,
            verbose=0,
            nAGQ=self.nAGQ,
        )

        start = kwargs.pop("start", self.theta)
        opt_result = optimizer.optimize(start=start, **kwargs)

        return GlmerResult(
            formula=self.formula,
            matrices=new_matrices,
            family=self.family,
            theta=opt_result.theta,
            beta=opt_result.beta,
            u=opt_result.u,
            deviance=opt_result.deviance,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            nAGQ=self.nAGQ,
        )

    def isREML(self) -> bool:
        """Check if the model was fit using REML.

        Always returns False for GLMMs since they use ML estimation.
        """
        return False

    def isGLMM(self) -> bool:
        """Check if this is a generalized linear mixed model.

        Always returns True for GlmerResult.
        """
        return True

    def isLMM(self) -> bool:
        """Check if this is a linear mixed model.

        Always returns False for GlmerResult.
        """
        return False

    def isNLMM(self) -> bool:
        """Check if this is a nonlinear mixed model.

        Always returns False for GlmerResult.
        """
        return False

    def npar(self) -> int:
        """Get the number of parameters in the model.

        Returns the total number of estimated parameters:
        - Fixed effects (beta)
        - Variance-covariance parameters (theta)

        Returns
        -------
        int
            Total number of parameters.
        """
        n_fixed = len(self.beta)
        n_theta = len(self.theta)
        return n_fixed + n_theta

    def drop1(self, data: pd.DataFrame, test: str = "Chisq"):
        from mixedlm.inference.drop1 import drop1_glmer

        return drop1_glmer(self, data, test=test)

    def allFit(
        self,
        data: pd.DataFrame,
        optimizers: list[str] | None = None,
        verbose: bool = False,
    ):
        from mixedlm.inference.allfit import allfit_glmer

        return allfit_glmer(self, data, optimizers=optimizers, verbose=verbose)

    def summary(self) -> str:
        lines = []
        lines.append("Generalized linear mixed model fit by maximum likelihood (Laplace)")
        lines.append(
            f" Family: {self.family.__class__.__name__} ({self.family.link.__class__.__name__})"
        )
        lines.append(f"Formula: {self.formula}")
        lines.append("")

        lines.append("     AIC      BIC   logLik deviance")
        lines.append(
            f"{self.AIC():8.1f} {self.BIC():8.1f} {self.logLik().value:8.1f} {self.deviance:8.1f}"
        )
        lines.append("")

        lines.append(str(self.VarCorr()))
        lines.append(f"Number of obs: {self.matrices.n_obs}")
        for struct in self.matrices.random_structures:
            lines.append(f"  groups:  {struct.grouping_factor}, {struct.n_levels}")
        lines.append("")

        lines.append("Fixed effects:")
        vcov = self.vcov()
        se = np.sqrt(np.diag(vcov))

        lines.append("             Estimate  Std. Error  z value  Pr(>|z|)")
        for i, name in enumerate(self.matrices.fixed_names):
            z_val = self.beta[i] / se[i] if se[i] > 0 else np.nan
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_val)))
            sig = _get_signif_code(p_val)
            lines.append(
                f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {z_val:7.3f}  {p_val:.4f} {sig}"
            )

        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")
            lines.append("  optimizer did not converge; try allFit() to compare optimizers")
        if self.isSingular():
            lines.append("  boundary (singular) fit: some random-effect variances are near zero")
            lines.append("  consider simplifying the random-effects structure")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"GlmerResult(formula={self.formula}, "
            f"family={self.family.__class__.__name__}, deviance={self.deviance:.4f})"
        )


class GlmerMod:
    def __init__(
        self,
        formula: str | Formula,
        data: pd.DataFrame,
        family: Family | None = None,
        verbose: int = 0,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        na_action: str | None = "omit",
        contrasts: dict[str, str | NDArray[np.floating]] | None = None,
        control: GlmerControl | None = None,
    ) -> None:
        from mixedlm.models.control import GlmerControl

        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.family = family if family is not None else Binomial()
        self.verbose = verbose
        self.na_action = na_action
        self.contrasts = contrasts
        self.control = control if control is not None else GlmerControl()

        self.matrices = build_model_matrices(
            self.formula,
            self.data,
            weights=weights,
            offset=offset,
            na_action=na_action,
            contrasts=contrasts,
        )

    def fit(
        self,
        start: NDArray[np.floating] | None = None,
        method: str | None = None,
        maxiter: int | None = None,
        nAGQ: int = 1,
    ) -> GlmerResult:
        import warnings

        from mixedlm.models.checks import run_model_checks

        ctrl = self.control
        opt_method = method if method is not None else ctrl.optimizer
        opt_maxiter = maxiter if maxiter is not None else ctrl.maxiter

        self.matrices, self._dropped_cols = run_model_checks(self.matrices, ctrl)

        if ctrl.em_init and start is None:
            from mixedlm.estimation.em_reml import em_reml_simple

            try:
                em_result = em_reml_simple(
                    self.matrices,
                    max_iter=ctrl.em_maxiter,
                    verbose=max(self.verbose - 1, 0),
                )
                start = em_result.theta
            except (NotImplementedError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"EM initialization failed ({type(exc).__name__}: {exc}). "
                    "Falling back to optimizer defaults.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        optimizer = GLMMOptimizer(
            self.matrices,
            self.family,
            verbose=self.verbose,
            nAGQ=nAGQ,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=opt_method,
            maxiter=opt_maxiter,
            options=ctrl.optCtrl,
        )

        result = GlmerResult(
            formula=self.formula,
            matrices=self.matrices,
            family=self.family,
            theta=opt_result.theta,
            beta=opt_result.beta,
            u=opt_result.u,
            deviance=opt_result.deviance,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            nAGQ=nAGQ,
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


def glmer_nb(
    formula: str,
    data: pd.DataFrame,
    verbose: int = 0,
    nAGQ: int = 1,
    weights: NDArray[np.floating] | str | None = None,
    offset: NDArray[np.floating] | str | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    control: GlmerControl | None = None,
    theta: float = 1.0,
    **kwargs,
) -> GlmerResult:
    """Fit a negative binomial generalized linear mixed-effects model.

    This is a convenience wrapper around glmer() that uses the negative
    binomial family. It's equivalent to calling glmer() with
    family=NegativeBinomial(theta).

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    verbose : int, default 0
        Verbosity level for optimization output.
    nAGQ : int, default 1
        Number of adaptive Gauss-Hermite quadrature points.
    weights : array-like, optional
        Prior weights for observations.
    offset : array-like, optional
        Offset term for the linear predictor.
    na_action : str, optional
        How to handle missing values ("omit", "exclude", "fail").
    contrasts : dict, optional
        Contrast specifications for categorical variables.
    control : GlmerControl, optional
        Control parameters for the optimizer.
    theta : float, default 1.0
        The theta (dispersion) parameter for the negative binomial
        distribution. Larger values indicate less overdispersion.
    **kwargs
        Additional arguments passed to the optimizer.

    Returns
    -------
    GlmerResult
        Fitted model result.

    Examples
    --------
    >>> result = glmer_nb("count ~ treatment + (1|subject)", data)

    >>> result = glmer_nb("count ~ x + (1|group)", data, theta=2.0)

    Notes
    -----
    The negative binomial distribution is useful for count data that
    exhibits overdispersion (variance > mean). The theta parameter
    controls the degree of overdispersion: as theta -> infinity, the
    negative binomial approaches the Poisson distribution.

    See Also
    --------
    glmer : General GLMM fitting function.
    NegativeBinomial : The negative binomial family class.
    """
    from mixedlm.families.negative_binomial import NegativeBinomial

    family = NegativeBinomial(theta=theta)
    return glmer(
        formula=formula,
        data=data,
        family=family,
        verbose=verbose,
        nAGQ=nAGQ,
        weights=weights,
        offset=offset,
        na_action=na_action,
        contrasts=contrasts,
        control=control,
        **kwargs,
    )


def glmer(
    formula: str,
    data: pd.DataFrame,
    family: Family | None = None,
    verbose: int = 0,
    nAGQ: int = 1,
    weights: NDArray[np.floating] | str | None = None,
    offset: NDArray[np.floating] | str | None = None,
    na_action: str | None = "omit",
    contrasts: dict[str, str | NDArray[np.floating]] | None = None,
    control: GlmerControl | None = None,
    **kwargs,
) -> GlmerResult:
    """Fit a generalized linear mixed-effects model.

    Parameters
    ----------
    formula : str
        Model formula in lme4 syntax (e.g., "y ~ x + (1|group)").
    data : DataFrame
        Data containing the variables in the formula.
    family : Family, optional
        GLM family (default: Binomial).
    verbose : int, default 0
        Verbosity level for optimization output.
    nAGQ : int, default 1
        Number of adaptive Gauss-Hermite quadrature points.
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
        Dictionary mapping variable names to contrast specifications.
        Values can be:
        - "treatment" (default): Treatment/dummy contrasts
        - "sum": Sum/deviation contrasts
        - "helmert": Helmert contrasts
        - "poly": Polynomial contrasts for ordered factors
        - A custom contrast matrix (NDArray of shape (n_levels, n_levels-1))
    control : GlmerControl, optional
        Control parameters for the optimizer. Use glmerControl() to create.
        If not provided, default settings are used.
    **kwargs
        Additional arguments passed to the optimizer (start, method, maxiter).

    Returns
    -------
    GlmerResult
        Fitted model result.

    Examples
    --------
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())

    >>> from mixedlm import glmerControl
    >>> ctrl = glmerControl(optimizer="Nelder-Mead", maxiter=2000)
    >>> result = glmer("y ~ x + (1|group)", data, family=Binomial(), control=ctrl)
    """
    weights_arr = _resolve_optional_vector(data, weights, "weights")
    offset_arr = _resolve_optional_vector(data, offset, "offset")

    model = GlmerMod(
        formula,
        data,
        family=family,
        verbose=verbose,
        weights=weights_arr,
        offset=offset_arr,
        na_action=na_action,
        contrasts=contrasts,
        control=control,
    )
    return model.fit(nAGQ=nAGQ, **kwargs)

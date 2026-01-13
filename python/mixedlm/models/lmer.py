from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.reml import LMMOptimizer, _build_lambda, _count_theta
from mixedlm.formula.parser import parse_formula, update_formula
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

    def getME(
        self, name: str
    ) -> NDArray[np.floating] | sparse.csc_matrix | float | int | list[str] | dict[str, int]:
        components = {
            "X": self.matrices.X,
            "Z": self.matrices.Z,
            "Zt": self.matrices.Zt,
            "y": self.matrices.y,
            "beta": self.beta,
            "theta": self.theta,
            "u": self.u,
            "b": self.u,
            "sigma": self.sigma,
            "deviance": self.deviance,
            "REML": self.REML,
            "n_obs": self.matrices.n_obs,
            "n_fixed": self.matrices.n_fixed,
            "n_random": self.matrices.n_random,
            "fixef_names": self.matrices.fixed_names,
            "weights": self.matrices.weights,
            "offset": self.matrices.offset,
        }

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        components["Lambda"] = Lambda
        components["Lambdat"] = Lambda.T

        if name not in components:
            valid = ", ".join(sorted(components.keys()))
            raise ValueError(f"Unknown component '{name}'. Valid components: {valid}")

        return components[name]

    @cached_property
    def _fitted_values(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    def fitted(self) -> NDArray[np.floating]:
        return self._fitted_values

    def residuals(self, type: str = "response") -> NDArray[np.floating]:
        if type == "response":
            return self.matrices.y - self.fitted()
        elif type == "pearson":
            return (self.matrices.y - self.fitted()) / self.sigma
        else:
            raise ValueError(f"Unknown residual type: {type}")

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

    def refit(self, newresp: NDArray[np.floating]) -> LmerResult:
        if len(newresp) != self.matrices.n_obs:
            raise ValueError(f"newresp has length {len(newresp)}, expected {self.matrices.n_obs}")

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
        )

        optimizer = LMMOptimizer(new_matrices, REML=self.REML, verbose=0)
        opt_result = optimizer.optimize(start=self.theta)

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

    def update(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        REML: bool | None = None,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
    ) -> LmerResult:
        if data is None:
            raise ValueError(
                "data must be provided to update(). "
                "The original data is not stored in the result object."
            )

        new_formula = update_formula(self.formula, formula) if formula is not None else self.formula

        new_REML = REML if REML is not None else self.REML

        model = LmerMod(
            new_formula,
            data,
            REML=new_REML,
            weights=weights,
            offset=offset,
        )
        return model.fit()

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

        if self.REML:
            self.matrices.n_obs - self.matrices.n_fixed
        else:
            self.matrices.n_obs - self.matrices.n_fixed

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

    def plot(
        self,
        which: list[int] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Create diagnostic plots for the model.

        Parameters
        ----------
        which : list of int, optional
            Which plots to include. Default is [1, 2, 3, 4].
            1 = Residuals vs Fitted
            2 = Normal Q-Q
            3 = Scale-Location
            4 = Residuals by Group
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with diagnostic plots.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        from mixedlm.diagnostics.plots import plot_diagnostics

        return plot_diagnostics(self, which=which, figsize=figsize)

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
    ) -> None:
        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.REML = REML
        self.verbose = verbose

        self.matrices = build_model_matrices(
            self.formula, self.data, weights=weights, offset=offset
        )

    def fit(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
    ) -> LmerResult:
        optimizer = LMMOptimizer(
            self.matrices,
            REML=self.REML,
            verbose=self.verbose,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=method,
            maxiter=maxiter,
        )

        return LmerResult(
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


def lmer(
    formula: str,
    data: pd.DataFrame,
    REML: bool = True,
    verbose: int = 0,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    **kwargs,
) -> LmerResult:
    model = LmerMod(formula, data, REML=REML, verbose=verbose, weights=weights, offset=offset)
    return model.fit(**kwargs)

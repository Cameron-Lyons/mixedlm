from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.laplace import GLMMOptimizer, _build_lambda, _count_theta
from mixedlm.families.base import Family
from mixedlm.families.binomial import Binomial
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices


@dataclass
class GlmerVarCorr:
    groups: dict[str, dict[str, float]]

    def __str__(self) -> str:
        lines = ["Random effects:"]
        lines.append(" Groups      Name         Variance  Std.Dev.")
        for group, terms in self.groups.items():
            for i, (name, var) in enumerate(terms.items()):
                grp_name = group if i == 0 else ""
                lines.append(f" {grp_name:11} {name:12} {var:9.4f}  {np.sqrt(var):.4f}")
        return "\n".join(lines)


@dataclass
class GlmerResult:
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

    def ranef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
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

        return result

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

    def linear_predictor(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    def fitted(self, type: str = "response") -> NDArray[np.floating]:
        eta = self.linear_predictor()
        if type == "link":
            return eta
        else:
            return self.family.link.inverse(eta)

    def residuals(self, type: str = "deviance") -> NDArray[np.floating]:
        mu = self.fitted(type="response")

        if type == "response":
            return self.matrices.y - mu
        elif type == "pearson":
            var = self.family.variance(mu)
            return (self.matrices.y - mu) / np.sqrt(var)
        elif type == "deviance":
            dev_resids = self.family.deviance_resids(self.matrices.y, mu, self.matrices.weights)
            signs = np.sign(self.matrices.y - mu)
            return signs * np.sqrt(np.abs(dev_resids))
        else:
            raise ValueError(f"Unknown residual type: {type}")

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        type: str = "response",
        re_form: str | None = None,
        offset: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        if newdata is None:
            return self.fitted(type=type)

        new_matrices = build_model_matrices(self.formula, newdata)
        eta = new_matrices.X @ self.beta

        if offset is None:
            offset = np.zeros(len(newdata), dtype=np.float64)

        eta = eta + offset

        if re_form != "NA" and re_form != "~0":
            pass

        if type == "link":
            return eta
        else:
            return self.family.link.inverse(eta)

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
            XtWZ = self.matrices.X.T @ W_diag @ self.matrices.Z
            ZtWZ = self.matrices.Z.T @ W_diag @ self.matrices.Z

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

        try:
            return linalg.inv(XtVinvX)
        except linalg.LinAlgError:
            return linalg.pinv(XtVinvX)

    def VarCorr(self) -> GlmerVarCorr:
        groups: dict[str, dict[str, float]] = {}
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

            term_vars: dict[str, float] = {}
            for i, term_name in enumerate(struct.term_names):
                term_vars[term_name] = cov[i, i]

            groups[struct.grouping_factor] = term_vars

        return GlmerVarCorr(groups=groups)

    def logLik(self) -> float:
        return -0.5 * self.deviance

    def AIC(self) -> float:
        n_theta = _count_theta(self.matrices.random_structures)
        n_params = self.matrices.n_fixed + n_theta
        return -2 * self.logLik() + 2 * n_params

    def BIC(self) -> float:
        n_theta = _count_theta(self.matrices.random_structures)
        n_params = self.matrices.n_fixed + n_theta
        n = self.matrices.n_obs
        return -2 * self.logLik() + n_params * np.log(n)

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

            for struct in self.matrices.random_structures:
                n_levels = struct.n_levels
                n_terms = struct.n_terms

                theta_start = sum(
                    s.n_terms * (s.n_terms + 1) // 2 if s.correlated else s.n_terms
                    for s in self.matrices.random_structures[
                        : self.matrices.random_structures.index(struct)
                    ]
                )
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
            theta = self.family.theta
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

    def refit(self, newresp: NDArray[np.floating]) -> GlmerResult:
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

        optimizer = GLMMOptimizer(new_matrices, self.family, verbose=0)
        opt_result = optimizer.optimize(start=self.theta)

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
            f"{self.AIC():8.1f} {self.BIC():8.1f} {self.logLik():8.1f} {self.deviance:8.1f}"
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
            from scipy import stats

            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_val)))
            sig = ""
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            elif p_val < 0.1:
                sig = "."
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
    ) -> None:
        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.family = family if family is not None else Binomial()
        self.verbose = verbose

        self.matrices = build_model_matrices(
            self.formula, self.data, weights=weights, offset=offset
        )

    def fit(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        nAGQ: int = 1,
    ) -> GlmerResult:
        if nAGQ != 1:
            pass

        optimizer = GLMMOptimizer(
            self.matrices,
            self.family,
            verbose=self.verbose,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=method,
            maxiter=maxiter,
        )

        return GlmerResult(
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


def glmer(
    formula: str,
    data: pd.DataFrame,
    family: Family | None = None,
    verbose: int = 0,
    nAGQ: int = 1,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    **kwargs,
) -> GlmerResult:
    model = GlmerMod(formula, data, family=family, verbose=verbose, weights=weights, offset=offset)
    return model.fit(nAGQ=nAGQ, **kwargs)

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
        return dict(zip(self.matrices.fixed_names, self.beta))

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

    def linear_predictor(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part

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
            dev_resids = self.family.deviance_resids(
                self.matrices.y, mu, np.ones(self.matrices.n_obs)
            )
            signs = np.sign(self.matrices.y - mu)
            return signs * np.sqrt(np.abs(dev_resids))
        else:
            raise ValueError(f"Unknown residual type: {type}")

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        type: str = "response",
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        if newdata is None:
            return self.fitted(type=type)

        new_matrices = build_model_matrices(self.formula, newdata)
        eta = new_matrices.X @ self.beta

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

    def summary(self) -> str:
        lines = []
        lines.append(f"Generalized linear mixed model fit by maximum likelihood (Laplace)")
        lines.append(f" Family: {self.family.__class__.__name__} ({self.family.link.__class__.__name__})")
        lines.append(f"Formula: {self.formula}")
        lines.append("")

        lines.append(f"     AIC      BIC   logLik deviance")
        lines.append(f"{self.AIC():8.1f} {self.BIC():8.1f} {self.logLik():8.1f} {self.deviance:8.1f}")
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
            lines.append(f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {z_val:7.3f}  {p_val:.4f} {sig}")

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
        return f"GlmerResult(formula={self.formula}, family={self.family.__class__.__name__}, deviance={self.deviance:.4f})"


class GlmerMod:
    def __init__(
        self,
        formula: str | Formula,
        data: pd.DataFrame,
        family: Family | None = None,
        verbose: int = 0,
    ) -> None:
        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.family = family if family is not None else Binomial()
        self.verbose = verbose

        self.matrices = build_model_matrices(self.formula, self.data)

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
    **kwargs,
) -> GlmerResult:
    model = GlmerMod(formula, data, family=family, verbose=verbose)
    return model.fit(nAGQ=nAGQ, **kwargs)

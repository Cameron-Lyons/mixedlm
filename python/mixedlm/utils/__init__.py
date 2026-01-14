from mixedlm.utils.lme4_compat import (
    ConvergenceInfo,
    DevComp,
    GHrule,
    REMLcrit,
    VarCorr,
    checkConv,
    coef,
    convergence_ok,
    devcomp,
    dummy,
    factorize,
    fixef,
    fortify,
    getME,
    isNested,
    lmList,
    mkMerMod,
    ngrps,
    pvalues,
    quickSimulate,
    ranef,
    scale_vcov,
    sigma,
    vcconv,
)
from mixedlm.utils.variance import (
    Cv_to_Sv,
    Cv_to_Vv,
    Sv_to_Cv,
    Vv_to_Cv,
    condVar,
    cov2sdcor,
    getL,
    mlist2vec,
    safe_chol,
    sdcor2cov,
    vec2mlist,
    vec2STlist,
)


def _get_signif_code(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


def _format_pvalue(p: float) -> str:
    if p < 2.2e-16:
        return "< 2e-16"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"


__all__ = [
    "sigma",
    "ngrps",
    "fixef",
    "ranef",
    "coef",
    "VarCorr",
    "getME",
    "lmList",
    "pvalues",
    "checkConv",
    "ConvergenceInfo",
    "convergence_ok",
    "fortify",
    "devcomp",
    "DevComp",
    "vcconv",
    "GHrule",
    "factorize",
    "mkMerMod",
    "isNested",
    "dummy",
    "REMLcrit",
    "scale_vcov",
    "quickSimulate",
    "sdcor2cov",
    "cov2sdcor",
    "Vv_to_Cv",
    "Cv_to_Vv",
    "Sv_to_Cv",
    "Cv_to_Sv",
    "mlist2vec",
    "vec2mlist",
    "vec2STlist",
    "condVar",
    "getL",
    "safe_chol",
]

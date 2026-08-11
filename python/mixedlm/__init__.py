"""Public package interface with demand-loaded exports.

The root module intentionally avoids importing the numerical, tabular, plotting,
and fitting stacks until a public object that needs them is accessed.  This keeps
lightweight uses such as version checks and formula parsing fast while preserving
the established ``from mixedlm import ...`` interface.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mixedlm import datasets, diagnostics, families, inference, nlme, power, utils
    from mixedlm._rust import SparseCholeskyNumeric, SparseCholeskySymbolic
    from mixedlm.datasets import (
        load_arabidopsis,
        load_cake,
        load_cbpp,
        load_dyestuff,
        load_dyestuff2,
        load_grouseticks,
        load_insteval,
        load_pastes,
        load_penicillin,
        load_sleepstudy,
        load_verbagg,
    )
    from mixedlm.diagnostics.collinearity import CollinearityResult, check_collinearity
    from mixedlm.diagnostics.fit_metrics import ICCResult, R2NakagawaResult, icc, r2_nakagawa
    from mixedlm.estimation.em_reml import EMResult, em_reml_simple
    from mixedlm.estimation.laplace import GQN, GQdk
    from mixedlm.estimation.optimizers import NelderMead, NelderMeadState, golden, nlminbwrap
    from mixedlm.estimation.reml import DevianceComponents
    from mixedlm.families.custom import CustomFamily, QuasiFamily, validate_family
    from mixedlm.formula.parser import (
        dropOffset,
        expandDoubleVerts,
        findbars,
        getFixedFormulaStr,
        getNGroups,
        getRandomFormulaStr,
        getResponseName,
        is_mixed_formula,
        nobars,
        parse_formula,
        set_cov_type,
        subbars,
    )
    from mixedlm.inference.anova import AnovaResult, AnovaType3Result, anova, anova_type3
    from mixedlm.inference.boot_ci import bootCI
    from mixedlm.inference.bootstrap import bootMer, bootstrap_nlmer
    from mixedlm.inference.cross_validation import (
        CrossValidationFold,
        CrossValidationResult,
        cross_validate,
        make_folds,
        weighted_mae,
        weighted_mse,
        weighted_r2,
        weighted_rmse,
    )
    from mixedlm.inference.ddf import (
        DenomDFResult,
        kenward_roger_df,
        pvalues_with_ddf,
        satterthwaite_df,
    )
    from mixedlm.inference.effects import allEffects, ggpredict
    from mixedlm.inference.emmeans import Emmeans, emmeans
    from mixedlm.inference.model_selection import ModelSelectionResult, model_selection
    from mixedlm.inference.profile import (
        Profile2DResult,
        logProf,
        plot_profiles,
        slice2D,
        splom_profiles,
        varianceProf,
    )
    from mixedlm.models.control import GlmerControl, LmerControl, glmerControl, lmerControl
    from mixedlm.models.glmer import GlmerMod, GlmerVarCorr, glmer, glmer_nb
    from mixedlm.models.lmer import (
        LmerMod,
        LogLik,
        ModelTerms,
        PredictResult,
        RanefResult,
        RePCA,
        RePCAGroup,
        VarCorr,
        VarCorrGroup,
        lmer,
    )
    from mixedlm.models.modular import (
        GlmerDevfun,
        GlmerParsedFormula,
        LmerDevfun,
        LmerParsedFormula,
        OptimizeResult,
        ReTrms,
        devfun2,
        glFormula,
        lFormula,
        mkDataTemplate,
        mkGlmerDevfun,
        mkGlmerMod,
        mkLmerDevfun,
        mkLmerMod,
        mkMinimalData,
        mkNewReTrms,
        mkParsTemplate,
        mkReTrms,
        optimizeGlmer,
        optimizeLmer,
        simulate_formula,
    )
    from mixedlm.models.nlmer import NlmerMod, NlmerResult, nlmer
    from mixedlm.power import PowerCurveResult, PowerResult, extend, powerCurve, powerSim
    from mixedlm.utils.allfit import AllFitResult, allFit
    from mixedlm.utils.lme4_compat import (
        ConvergenceInfo,
        DevComp,
        GHrule,
        REMLcrit,
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

__version__ = "1.1.0"

__all__ = [
    "lmer",
    "LmerMod",
    "lmerControl",
    "LmerControl",
    "allFit",
    "AllFitResult",
    "glmer",
    "glmer_nb",
    "GlmerMod",
    "glmerControl",
    "GlmerControl",
    "nlmer",
    "NlmerMod",
    "NlmerResult",
    "anova",
    "anova_type3",
    "AnovaResult",
    "AnovaType3Result",
    "emmeans",
    "Emmeans",
    "model_selection",
    "ModelSelectionResult",
    "ggpredict",
    "allEffects",
    "bootMer",
    "bootCI",
    "bootstrap_nlmer",
    "cross_validate",
    "make_folds",
    "weighted_mse",
    "weighted_rmse",
    "weighted_mae",
    "weighted_r2",
    "CrossValidationFold",
    "CrossValidationResult",
    "satterthwaite_df",
    "kenward_roger_df",
    "pvalues_with_ddf",
    "DenomDFResult",
    "r2_nakagawa",
    "R2NakagawaResult",
    "icc",
    "ICCResult",
    "check_collinearity",
    "CollinearityResult",
    "powerSim",
    "powerCurve",
    "extend",
    "PowerResult",
    "PowerCurveResult",
    "power",
    "sigma",
    "ngrps",
    "lmList",
    "pvalues",
    "checkConv",
    "fixef",
    "ranef",
    "coef",
    "getME",
    "ConvergenceInfo",
    "convergence_ok",
    "plot_profiles",
    "splom_profiles",
    "slice2D",
    "Profile2DResult",
    "logProf",
    "varianceProf",
    "parse_formula",
    "nobars",
    "findbars",
    "subbars",
    "is_mixed_formula",
    "set_cov_type",
    "expandDoubleVerts",
    "dropOffset",
    "getResponseName",
    "getFixedFormulaStr",
    "getRandomFormulaStr",
    "getNGroups",
    "families",
    "CustomFamily",
    "QuasiFamily",
    "validate_family",
    "DevianceComponents",
    "nlme",
    "inference",
    "diagnostics",
    "datasets",
    "utils",
    "load_sleepstudy",
    "load_cbpp",
    "load_dyestuff",
    "load_dyestuff2",
    "load_penicillin",
    "load_cake",
    "load_pastes",
    "load_insteval",
    "load_arabidopsis",
    "load_grouseticks",
    "load_verbagg",
    "fortify",
    "devcomp",
    "DevComp",
    "vcconv",
    "GHrule",
    "factorize",
    "mkMerMod",
    "LogLik",
    "ModelTerms",
    "PredictResult",
    "RanefResult",
    "VarCorr",
    "VarCorrGroup",
    "GlmerVarCorr",
    "RePCA",
    "RePCAGroup",
    "lFormula",
    "glFormula",
    "mkLmerDevfun",
    "mkGlmerDevfun",
    "optimizeLmer",
    "optimizeGlmer",
    "mkLmerMod",
    "mkGlmerMod",
    "LmerParsedFormula",
    "GlmerParsedFormula",
    "LmerDevfun",
    "GlmerDevfun",
    "OptimizeResult",
    "ReTrms",
    "mkReTrms",
    "simulate_formula",
    "devfun2",
    "isNested",
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
    "dummy",
    "REMLcrit",
    "scale_vcov",
    "quickSimulate",
    "GQdk",
    "GQN",
    "NelderMead",
    "NelderMeadState",
    "golden",
    "nlminbwrap",
    "mkDataTemplate",
    "mkParsTemplate",
    "mkMinimalData",
    "mkNewReTrms",
    "EMResult",
    "em_reml_simple",
    "SparseCholeskySymbolic",
    "SparseCholeskyNumeric",
]


def _module_exports(module: str, *names: str) -> dict[str, tuple[str, str | None]]:
    return {name: (module, name) for name in names}


_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "datasets": ("mixedlm.datasets", None),
    "diagnostics": ("mixedlm.diagnostics", None),
    "families": ("mixedlm.families", None),
    "inference": ("mixedlm.inference", None),
    "nlme": ("mixedlm.nlme", None),
    "power": ("mixedlm.power", None),
    "utils": ("mixedlm.utils", None),
    **_module_exports(
        "mixedlm.datasets",
        "load_arabidopsis",
        "load_cake",
        "load_cbpp",
        "load_dyestuff",
        "load_dyestuff2",
        "load_grouseticks",
        "load_insteval",
        "load_pastes",
        "load_penicillin",
        "load_sleepstudy",
        "load_verbagg",
    ),
    **_module_exports(
        "mixedlm.diagnostics.collinearity", "CollinearityResult", "check_collinearity"
    ),
    **_module_exports(
        "mixedlm.diagnostics.fit_metrics", "ICCResult", "R2NakagawaResult", "icc", "r2_nakagawa"
    ),
    **_module_exports("mixedlm.estimation.em_reml", "EMResult", "em_reml_simple"),
    **_module_exports("mixedlm.estimation.laplace", "GQN", "GQdk"),
    **_module_exports(
        "mixedlm.estimation.optimizers",
        "NelderMead",
        "NelderMeadState",
        "golden",
        "nlminbwrap",
    ),
    **_module_exports("mixedlm.estimation.reml", "DevianceComponents"),
    **_module_exports("mixedlm._rust", "SparseCholeskyNumeric", "SparseCholeskySymbolic"),
    **_module_exports("mixedlm.families.custom", "CustomFamily", "QuasiFamily", "validate_family"),
    **_module_exports(
        "mixedlm.formula.parser",
        "dropOffset",
        "expandDoubleVerts",
        "findbars",
        "getFixedFormulaStr",
        "getNGroups",
        "getRandomFormulaStr",
        "getResponseName",
        "is_mixed_formula",
        "nobars",
        "parse_formula",
        "set_cov_type",
        "subbars",
    ),
    **_module_exports(
        "mixedlm.inference.anova", "AnovaResult", "AnovaType3Result", "anova", "anova_type3"
    ),
    **_module_exports("mixedlm.inference.boot_ci", "bootCI"),
    **_module_exports("mixedlm.inference.bootstrap", "bootMer", "bootstrap_nlmer"),
    **_module_exports(
        "mixedlm.inference.cross_validation",
        "CrossValidationFold",
        "CrossValidationResult",
        "cross_validate",
        "make_folds",
        "weighted_mae",
        "weighted_mse",
        "weighted_r2",
        "weighted_rmse",
    ),
    **_module_exports(
        "mixedlm.inference.ddf",
        "DenomDFResult",
        "kenward_roger_df",
        "pvalues_with_ddf",
        "satterthwaite_df",
    ),
    **_module_exports("mixedlm.inference.effects", "allEffects", "ggpredict"),
    **_module_exports("mixedlm.inference.emmeans", "Emmeans", "emmeans"),
    **_module_exports(
        "mixedlm.inference.model_selection", "ModelSelectionResult", "model_selection"
    ),
    **_module_exports(
        "mixedlm.inference.profile",
        "Profile2DResult",
        "logProf",
        "plot_profiles",
        "slice2D",
        "splom_profiles",
        "varianceProf",
    ),
    **_module_exports(
        "mixedlm.models.control", "GlmerControl", "LmerControl", "glmerControl", "lmerControl"
    ),
    **_module_exports("mixedlm.models.glmer", "GlmerMod", "GlmerVarCorr", "glmer", "glmer_nb"),
    **_module_exports(
        "mixedlm.models.lmer",
        "LmerMod",
        "LogLik",
        "ModelTerms",
        "PredictResult",
        "RanefResult",
        "RePCA",
        "RePCAGroup",
        "VarCorr",
        "VarCorrGroup",
        "lmer",
    ),
    **_module_exports(
        "mixedlm.models.modular",
        "GlmerDevfun",
        "GlmerParsedFormula",
        "LmerDevfun",
        "LmerParsedFormula",
        "OptimizeResult",
        "ReTrms",
        "devfun2",
        "glFormula",
        "lFormula",
        "mkDataTemplate",
        "mkGlmerDevfun",
        "mkGlmerMod",
        "mkLmerDevfun",
        "mkLmerMod",
        "mkMinimalData",
        "mkNewReTrms",
        "mkParsTemplate",
        "mkReTrms",
        "optimizeGlmer",
        "optimizeLmer",
        "simulate_formula",
    ),
    **_module_exports("mixedlm.models.nlmer", "NlmerMod", "NlmerResult", "nlmer"),
    **_module_exports(
        "mixedlm.power", "PowerCurveResult", "PowerResult", "extend", "powerCurve", "powerSim"
    ),
    **_module_exports("mixedlm.utils.allfit", "AllFitResult", "allFit"),
    **_module_exports(
        "mixedlm.utils.lme4_compat",
        "ConvergenceInfo",
        "DevComp",
        "GHrule",
        "REMLcrit",
        "checkConv",
        "coef",
        "convergence_ok",
        "devcomp",
        "dummy",
        "factorize",
        "fixef",
        "fortify",
        "getME",
        "isNested",
        "lmList",
        "mkMerMod",
        "ngrps",
        "pvalues",
        "quickSimulate",
        "ranef",
        "scale_vcov",
        "sigma",
        "vcconv",
    ),
    **_module_exports(
        "mixedlm.utils.variance",
        "Cv_to_Sv",
        "Cv_to_Vv",
        "Sv_to_Cv",
        "Vv_to_Cv",
        "condVar",
        "cov2sdcor",
        "getL",
        "mlist2vec",
        "safe_chol",
        "sdcor2cov",
        "vec2mlist",
        "vec2STlist",
    ),
}

_OPTIONAL_EXPORTS = {"SparseCholeskyNumeric", "SparseCholeskySymbolic"}


def __getattr__(name: str) -> Any:
    """Load and cache a public export on first access."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    try:
        module = import_module(module_name)
        value = module if attribute_name is None else getattr(module, attribute_name)
    except (ImportError, AttributeError):
        if name not in _OPTIONAL_EXPORTS:
            raise
        value = None

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include demand-loaded exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))

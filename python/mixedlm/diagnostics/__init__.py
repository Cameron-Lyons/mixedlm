from __future__ import annotations

from mixedlm.diagnostics.collinearity import CollinearityResult, check_collinearity
from mixedlm.diagnostics.dispersion import (
    DispersionResult,
    ZeroInflationResult,
    check_overdispersion,
    check_zero_inflation,
    check_zeroinflation,
)
from mixedlm.diagnostics.fit_metrics import ICCResult, R2NakagawaResult, icc, r2_nakagawa
from mixedlm.diagnostics.influence import (
    InfluenceResult,
    cooks_distance,
    dfbeta,
    dfbetas,
    dffits,
    influence,
    influence_plot,
    influence_summary,
    influential_obs,
    leverage,
)
from mixedlm.diagnostics.plots import (
    plot_diagnostics,
    plot_qq,
    plot_ranef,
    plot_resid_fitted,
    plot_resid_group,
    plot_scale_location,
)

__all__ = [
    "DispersionResult",
    "ZeroInflationResult",
    "check_overdispersion",
    "check_zero_inflation",
    "check_zeroinflation",
    "plot_diagnostics",
    "plot_qq",
    "plot_ranef",
    "plot_resid_fitted",
    "plot_resid_group",
    "plot_scale_location",
    "influence",
    "InfluenceResult",
    "dfbeta",
    "dfbetas",
    "cooks_distance",
    "dffits",
    "leverage",
    "influence_plot",
    "influence_summary",
    "influential_obs",
    "r2_nakagawa",
    "R2NakagawaResult",
    "icc",
    "ICCResult",
    "check_collinearity",
    "CollinearityResult",
]

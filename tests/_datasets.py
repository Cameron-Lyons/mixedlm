"""Canonical lme4 example data and seeded grouped simulations shared by tests.

The frames are module-level so parametrize lists and module fixtures can use
them at import time. Treat them as read-only: copy before changing columns.
"""

import numpy as np
import pandas as pd
from mixedlm import load_cbpp, load_sleepstudy
from scipy.special import expit

# The bundled lme4 datasets, whose contents test_datasets.py hash-verifies.
SLEEPSTUDY = load_sleepstudy()
CBPP = load_cbpp()

# Binomial trials make this fit non-singular; lme4 publishes its estimates.
CBPP_FORMULA = "incidence / size ~ period + (1 | herd)"


def grouped_data(family="gaussian", *, seed=42, n_groups=10, n_per_group=20):
    """Simulate ``y ~ x + (1 | group)`` data with known fixed effects.

    Gaussian: intercept 2.0, slope 1.5, group SD 0.5, residual SD 0.5.
    Binomial: logit intercept -0.5, slope 0.5, group SD 0.3, one trial per row.
    Poisson: log intercept 0.5, slope 0.3, group SD 0.5.
    """
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(group.size)
    if family == "gaussian":
        effects = rng.normal(0.0, 0.5, n_groups)
        y = 2.0 + 1.5 * x + effects[group] + rng.normal(0.0, 0.5, group.size)
    elif family == "binomial":
        effects = rng.normal(0.0, 0.3, n_groups)
        y = rng.binomial(1, expit(-0.5 + 0.5 * x + effects[group])).astype(float)
    elif family == "poisson":
        effects = rng.normal(0.0, 0.5, n_groups)
        y = rng.poisson(np.exp(0.5 + 0.3 * x + effects[group])).astype(float)
    else:
        raise ValueError(f"unknown family {family!r}")
    return pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})

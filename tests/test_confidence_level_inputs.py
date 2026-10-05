"""Inference APIs share one confidence-level check with the same exception types."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import bootCI, emmeans, lmer, tidy
from mixedlm.inference.bootstrap import BootstrapResult
from mixedlm.inference.hypothesis import linear_hypothesis


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(17)
    group = np.repeat(np.arange(8), 6)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=48),
            "arm": np.tile(["a", "b", "c"], 16),
            "group": group.astype(str),
        }
    )
    data["y"] = 1.0 + 0.5 * data["x"] + rng.normal(size=8)[group] + rng.normal(size=48)
    return lmer("y ~ x + arm + (1 | group)", data)


@pytest.fixture(scope="module")
def bootstrap_result() -> BootstrapResult:
    samples = np.array([[0.8, 1.9], [1.1, 2.2], [1.3, 1.7], [0.9, 2.4]])
    return BootstrapResult(
        n_boot=4,
        beta_samples=samples,
        theta_samples=np.empty((4, 0)),
        sigma_samples=None,
        fixed_names=["(Intercept)", "x"],
        original_beta=np.array([1.0, 2.0]),
        original_theta=np.empty(0),
        original_sigma=None,
        n_failed=0,
    )


CALLS = {
    "BootstrapResult.ci": lambda model, boot, level: boot.ci(level=level),
    "bootCI": lambda model, boot, level: bootCI(boot, level=level),
    "linear_hypothesis": lambda model, boot, level: linear_hypothesis(
        model, {"x": 1.0}, level=level
    ),
    "tidy": lambda model, boot, level: tidy(model, conf_int=True, conf_level=level),
    "emmeans": lambda model, boot, level: emmeans(model, "arm", level=level),
    "Emmeans.pairs": lambda model, boot, level: emmeans(model, "arm").pairs(level=level),
}


@pytest.mark.parametrize("api", CALLS)
@pytest.mark.parametrize(
    ("level", "error"),
    [("0.9", TypeError), (True, TypeError), (np.nan, ValueError), (1.0, ValueError)],
)
def test_invalid_confidence_levels_raise_the_same_errors(
    model, bootstrap_result, api, level, error
) -> None:
    with pytest.raises(error, match="level must be a finite number strictly between 0 and 1"):
        CALLS[api](model, bootstrap_result, level)


@pytest.mark.parametrize("api", CALLS)
def test_valid_numpy_levels_are_accepted(model, bootstrap_result, api) -> None:
    CALLS[api](model, bootstrap_result, np.float32(0.9))

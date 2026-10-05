"""Model checks validate their control options and act on the fitted design."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.models.checks import ModelCheckError, check_rankX
from mixedlm.models.control import GlmerControl, LmerControl
from numpy.testing import assert_allclose


@pytest.mark.parametrize("control", [LmerControl, GlmerControl])
def test_check_defaults_follow_lme4(control):
    ctrl = control()

    assert ctrl.check_nobs_vs_rankZ == "ignore"
    assert ctrl.check_nobs_vs_nlev == "ignore"
    assert ctrl.check_nlev_gtreq_5 == "warning"
    assert ctrl.check_nlev_gtr_1 == "stop"
    assert ctrl.check_rankX == "message+drop.cols"
    assert ctrl.check_scaleX == "warning"


def test_invalid_check_actions_are_rejected():
    with pytest.raises(ValueError, match="check_nobs_vs_rankZ"):
        LmerControl(check_nobs_vs_rankZ="invalid")
    with pytest.raises(ValueError, match="check_rankX"):
        LmerControl(check_rankX="invalid+option")


@pytest.mark.parametrize("action", ["ignore", "warning+drop.cols"])
def test_rank_check_accepts_its_documented_actions(action):
    assert LmerControl(check_rankX=action).check_rankX == action


def test_rank_deficient_fixed_effects_keep_model_metadata_consistent():
    x = np.linspace(-1.0, 1.0, 30)
    data = pd.DataFrame(
        {
            "y": 1.5 + 0.75 * x,
            "x1": x,
            "x2": 2.0 * x,
            "group": np.repeat(np.arange(10), 3),
        }
    )
    control = LmerControl(
        check_rankX="warning+drop.cols",
        check_nlev_gtreq_5="ignore",
        check_conv=False,
        check_singular=False,
    )

    with pytest.warns(UserWarning, match="Dropping columns"):
        result = lmer("y ~ x1 + x2 + (1 | group)", data, control=control)

    assert result.matrices.X.shape[1] == 2
    assert result.matrices.n_fixed == 2
    assert len(result.matrices.fixed_names) == 2
    assert len(result.beta) == 2
    assert list(result.fixef()) == result.matrices.fixed_names

    predictions = result.predict(data, re_form="NA")
    assert np.allclose(predictions, result.matrices.X @ result.beta)


def test_rank_deficiency_drop_supports_more_columns_than_rows():
    matrices = SimpleNamespace(X=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))

    with pytest.warns(UserWarning, match=r"Dropping columns: \[1\]"):
        reduced, dropped = check_rankX(matrices, "warning+drop.cols")

    assert dropped == [1]
    np.testing.assert_array_equal(reduced, matrices.X[:, [0, 2]])


def test_single_level_grouping_stops_by_default_and_reduces_to_ols_when_ignored():
    data = pd.DataFrame({"y": [1.0, 2.0, 4.0, 3.5], "x": [1.0, 2.0, 3.0, 4.0], "group": ["A"] * 4})
    with pytest.raises(ModelCheckError, match="only 1 level"):
        lmer("y ~ x + (1 | group)", data)

    with pytest.warns(UserWarning, match="'group' has only 1 levels"):
        result = lmer("y ~ x + (1 | group)", data, control=LmerControl(check_nlev_gtr_1="ignore"))

    # One group shifts every row equally, so the intercept absorbs it and the
    # REML criterion is the ordinary least-squares one for every variance.
    X = np.column_stack((np.ones(4), data["x"]))
    beta, rss = np.linalg.lstsq(X, data["y"], rcond=None)[:2]
    df = 4 - 2
    assert_allclose(result.beta, beta, rtol=1e-10)
    assert result.sigma == pytest.approx(np.sqrt(rss[0] / df), rel=1e-10)
    expected = df * (1 + np.log(2 * np.pi * rss[0] / df)) + np.linalg.slogdet(X.T @ X)[1]
    assert result.deviance == pytest.approx(expected, rel=1e-10)


def test_few_grouping_levels_warn():
    rng = np.random.default_rng(12)
    data = pd.DataFrame(
        {"y": rng.normal(size=12), "x": rng.normal(size=12), "group": ["A", "B", "C"] * 4}
    )

    with pytest.warns(UserWarning, match="'group' has only 3 levels"):
        lmer(
            "y ~ x + (1 | group)",
            data,
            control=LmerControl(check_nlev_gtreq_5="warning", check_singular=False),
        )


def test_predictors_on_very_different_scales_warn():
    rng = np.random.default_rng(13)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x1": rng.normal(size=100),
            "x2": rng.normal(size=100) * 10000,
            "group": np.repeat(range(10), 10),
        }
    )

    with pytest.warns(UserWarning, match="very different scales"):
        lmer(
            "y ~ x1 + x2 + (1 | group)",
            data,
            control=LmerControl(
                check_scaleX="warning", check_nlev_gtreq_5="ignore", check_singular=False
            ),
        )

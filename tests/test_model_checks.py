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

    # As R's qr(): the later of two aliased columns is the one dropped.
    with pytest.warns(UserWarning, match=r"Dropping columns: \[2\]"):
        reduced, dropped = check_rankX(matrices, "warning+drop.cols")

    assert dropped == [2]
    np.testing.assert_array_equal(reduced, matrices.X[:, [0, 1]])


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


@pytest.mark.parametrize(
    ("formula", "kept"),
    [
        ("y ~ c + (1 | g)", ["(Intercept)"]),
        ("y ~ x + x2 + (1 | g)", ["(Intercept)", "x"]),
        ("y ~ x2 + x + (1 | g)", ["(Intercept)", "x2"]),
        ("y ~ c + x + (1 | g)", ["(Intercept)", "x"]),
    ],
)
def test_rank_deficient_designs_drop_the_later_columns_as_lme4_does(formula, kept):
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "c": np.full(60, np.log(10)),
            "x": x,
            "x2": 2 * x,
            "g": np.repeat(np.arange(10), 6).astype(str),
        }
    )
    control = LmerControl(check_rankX="silent.drop.cols", check_singular=False)
    fit = lmer(formula, data, control=control)
    assert list(fit.fixef()) == kept


def test_rank_checks_use_r_tolerance_and_order():
    from mixedlm.models.checks import kept_columns

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    near = X[:, 2] + 1e-9 * rng.normal(size=50)
    far = X[:, 2] + 1e-5 * rng.normal(size=50)
    np.testing.assert_array_equal(kept_columns(np.column_stack([X, near])), [0, 1, 2])
    np.testing.assert_array_equal(kept_columns(np.column_stack([X, far])), [0, 1, 2, 3])
    np.testing.assert_array_equal(kept_columns(np.column_stack([X[:, :1] * 0, X])), [1, 2, 3])
    np.testing.assert_array_equal(kept_columns(np.ones((2, 4))), [0])


@pytest.mark.parametrize(
    ("action", "expected"),
    [("warn+drop.cols", UserWarning), ("stop.deficient", ModelCheckError)],
)
def test_rank_checks_accept_lme4_spellings(action, expected):
    matrices = SimpleNamespace(X=np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))
    if expected is UserWarning:
        with pytest.warns(UserWarning, match=r"Dropping columns: \[1\]"):
            assert check_rankX(matrices, action)[1] == [1]
    else:
        with pytest.raises(ModelCheckError, match="rank deficient"):
            check_rankX(matrices, action)

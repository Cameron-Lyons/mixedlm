"""Influence summaries, flags and plots apply the conventional cutoffs to each measure."""

import numpy as np
import pytest
from mixedlm import lmer, load_sleepstudy
from mixedlm.diagnostics import influence, influence_plot, influence_summary, influential_obs
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module")
def diagnostics():
    return influence(lmer("Reaction ~ Days + (1 | Subject)", load_sleepstudy()))


def measures(diagnostics):
    """Each measure and its cutoff: 4/n, 2/sqrt(n), 2p/n and 2 sqrt(p/n)."""
    n, p = len(diagnostics.residuals), len(diagnostics.beta)
    return {
        "cooks": (diagnostics.cooks_distance, 4 / n),
        "dfbetas": (np.max(np.abs(diagnostics.dfbetas), axis=1), 2 / np.sqrt(n)),
        "leverage": (diagnostics.hat_values, 2 * p / n),
        "dffits": (np.abs(diagnostics.dffits), 2 * np.sqrt(p / n)),
    }


def test_summary_reports_every_measure_ordered_by_cooks_distance(diagnostics):
    summary = influence_summary(diagnostics)
    expected = measures(diagnostics)
    rows = summary["observation"].to_numpy()

    assert_array_equal(np.sort(rows), np.arange(len(diagnostics.residuals)))
    assert np.all(np.diff(summary["cooks_distance"]) <= 0)
    for column, flag, name in [
        ("cooks_distance", "influential_cooks", "cooks"),
        ("max_abs_dfbetas", "influential_dfbetas", "dfbetas"),
        ("leverage", "high_leverage", "leverage"),
        ("abs_dffits", "influential_dffits", "dffits"),
    ]:
        values, cutoff = expected[name]
        assert_allclose(summary[column], values[rows], rtol=1e-14)
        assert_array_equal(summary[flag], values[rows] > cutoff)


@pytest.mark.parametrize("threshold", ["cooks", "dfbetas", "leverage", "dffits"])
def test_influential_obs_returns_rows_above_the_cutoff(diagnostics, threshold):
    values, cutoff = measures(diagnostics)[threshold]

    flagged = influential_obs(diagnostics, threshold=threshold)

    assert flagged.size
    assert_array_equal(flagged, np.flatnonzero(values > cutoff))


@pytest.mark.parametrize("which", ["cooks", "dfbetas", "leverage", "dffits"])
def test_influence_plot_draws_each_measure_with_its_cutoff(diagnostics, which):
    figure = pytest.importorskip("matplotlib.figure")
    ax = figure.Figure().subplots()
    values, cutoff = measures(diagnostics)[which]

    influence_plot(diagnostics, which=which, ax=ax)

    stems = ax.containers[0]
    assert_allclose(stems.markerline.get_ydata(), values, rtol=1e-14)
    assert_array_equal(stems.markerline.get_xdata(), np.arange(len(values)))
    (line,) = [line for line in ax.get_lines() if line.get_label() == "threshold"]
    assert_allclose(line.get_ydata(), [cutoff, cutoff])
    assert ax.get_title() == f"Influence Diagnostics ({which})"


def test_unknown_measures_are_rejected(diagnostics):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="Unknown threshold type: bogus"):
        influential_obs(diagnostics, threshold="bogus")
    with pytest.raises(ValueError, match="Unknown plot type: bogus"):
        influence_plot(diagnostics, which="bogus", ax=object())

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glFormula, glmer, glmerControl, load_cbpp, parse_formula
from mixedlm.formula.parser import getFixedFormulaStr, nobars, update_formula
from mixedlm.inference.allfit import allfit_glmer
from mixedlm.inference.drop1 import drop1_glmer
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2


def test_grouped_response_parser_round_trips() -> None:
    formula = parse_formula("incidence / size ~ period + (1 | herd)")

    assert formula.response == "incidence"
    assert formula.response_denominator == "size"
    assert {"incidence", "size"}.issubset(formula.all_variables)
    assert str(formula) == "incidence / size ~ period + (1 | herd)"
    assert str(nobars(formula)) == "incidence / size ~ period"
    assert getFixedFormulaStr(formula) == "incidence / size ~ period"
    assert str(update_formula(formula, ". ~ . + size")) == (
        "incidence / size ~ period + size + (1 | herd)"
    )


def test_grouped_response_builds_proportions_and_trial_weights() -> None:
    data = load_cbpp()

    parsed = glFormula("incidence / size ~ period + (1 | herd)", data)

    expected_trials = data["size"].to_numpy(dtype=np.float64)
    assert np.allclose(parsed.matrices.y, data["incidence"] / data["size"])
    assert np.array_equal(parsed.matrices.trials, expected_trials)
    assert np.array_equal(parsed.matrices.weights, expected_trials)
    assert set(parsed.matrices.frame.columns) == {"incidence", "size", "period", "herd"}


def test_grouped_response_multiplies_explicit_prior_weights() -> None:
    data = load_cbpp()
    prior_weights = np.linspace(0.5, 1.5, len(data))

    parsed = glFormula(
        "incidence / size ~ period + (1 | herd)",
        data,
        weights=prior_weights,
    )

    assert np.allclose(parsed.matrices.weights, prior_weights * data["size"].to_numpy())


def test_grouped_response_omits_missing_trial_counts() -> None:
    data = load_cbpp()
    data.loc[0, "size"] = np.nan

    parsed = glFormula("incidence / size ~ period + (1 | herd)", data)

    assert parsed.matrices.n_obs == len(data) - 1
    assert parsed.matrices.na_info is not None
    assert np.array_equal(parsed.matrices.na_info.omitted_indices, np.array([0]))


def test_grouped_response_supports_polars() -> None:
    pl = pytest.importorskip("polars")
    data = pl.DataFrame(load_cbpp().to_dict(orient="list"))

    parsed = glFormula("incidence / size ~ period + (1 | herd)", data)

    assert np.allclose(
        parsed.matrices.y,
        data["incidence"].to_numpy() / data["size"].to_numpy(),
    )
    assert np.array_equal(parsed.matrices.weights, data["size"].to_numpy())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("size", 0, "trials must be greater than zero"),
        ("incidence", -1, "successes must be between zero and trials"),
        ("incidence", 100, "successes must be between zero and trials"),
        ("size", 14.5, "successes and trials must be whole numbers"),
    ],
)
def test_grouped_response_rejects_invalid_counts(column: str, value: float, message: str) -> None:
    data = load_cbpp()
    if isinstance(value, float) and not value.is_integer():
        data[column] = data[column].astype(np.float64)
    data.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        glFormula("incidence / size ~ period + (1 | herd)", data)


def test_grouped_response_requires_binomial_family() -> None:
    data = load_cbpp()

    with pytest.raises(ValueError, match="only supported for binomial GLMMs"):
        glFormula(
            "incidence / size ~ period + (1 | herd)",
            data,
            family=families.Poisson(),
        )


def test_grouped_response_fit_matches_manual_proportion_and_weights() -> None:
    data = load_cbpp()
    manual_data = data.assign(y=data["incidence"] / data["size"])

    grouped = glmer("incidence / size ~ period + (1 | herd)", data)
    manual = glmer("y ~ period + (1 | herd)", manual_data, weights="size")

    # Period 4 has no successes; both encodings must report the same status.
    assert grouped.converged == manual.converged
    assert grouped.pirls_converged == manual.pirls_converged
    assert np.allclose(grouped.beta, manual.beta)
    assert np.allclose(grouped.theta, manual.theta)
    assert grouped.deviance == pytest.approx(manual.deviance)

    simulated = grouped.simulate(nsim=3, seed=42)
    trials = data["size"].to_numpy()[:, None]
    assert np.all(simulated >= 0)
    assert np.all(simulated <= trials)
    assert np.all(simulated == np.floor(simulated))

    refitted = grouped.refit(simulated[:, 0])
    assert np.allclose(refitted.matrices.y, simulated[:, 0] / data["size"].to_numpy())
    assert np.array_equal(refitted.matrices.trials, grouped.matrices.trials)


@pytest.mark.parametrize("change_trials", [False, True])
def test_grouped_update_preserves_prior_weights_and_offsets(change_trials):
    rng = np.random.default_rng(526)
    group = np.repeat(np.arange(10), 8)
    x = rng.normal(size=len(group))
    trials = rng.integers(8, 30, size=len(group))
    prior = rng.uniform(0.7, 1.6, size=len(group))
    offset = rng.normal(scale=0.2, size=len(group))
    eta = -0.3 + 0.5 * x + offset + rng.normal(scale=0.5, size=10)[group]
    data = pd.DataFrame(
        {
            "successes": rng.binomial(trials, 1 / (1 + np.exp(-eta))),
            "trials": trials,
            "x": x,
            "group": group,
        }
    )
    formula = "successes / trials ~ x + (1 | group)"
    control = glmerControl(check_conv=False, check_singular=False)
    model = glmer(formula, data, weights=prior, offset=offset, control=control)
    updated_data = data.assign(trials=data["trials"] + 3) if change_trials else data
    updated = model.update(data=updated_data, control=control)
    expected = glmer(formula, updated_data, weights=prior, offset=offset, control=control)

    assert_allclose(updated.matrices.weights, prior * updated_data["trials"], rtol=1e-15)
    assert_array_equal(updated.matrices.offset, offset)
    assert_array_equal(updated.matrices.trials, updated_data["trials"])
    assert_allclose(updated.beta, expected.beta, rtol=1e-7, atol=1e-7)
    assert_allclose(updated.theta, expected.theta, rtol=1e-7, atol=1e-7)
    assert updated.deviance == pytest.approx(expected.deviance, rel=1e-10)
    assert updated.converged == expected.converged
    assert updated.pirls_converged == expected.pirls_converged
    assert_array_equal(model.matrices.trials, trials)


def test_grouped_update_explicit_weights_apply_trial_counts_once():
    data = load_cbpp()
    original = glmer("incidence / size ~ period + (1 | herd)", data)
    prior = np.linspace(0.7, 1.3, len(data))
    updated = original.update(weights=prior)
    expected = glmer(str(original.formula), data, weights=prior)

    assert_allclose(updated.matrices.weights, prior * data["size"])
    assert_allclose(updated.beta, expected.beta)
    assert_allclose(updated.theta, expected.theta)


@pytest.fixture
def grouped_comparison_data():
    rng = np.random.default_rng(984)
    group = np.repeat(np.arange(8), 10)
    x = rng.normal(size=len(group))
    z = rng.normal(size=len(group))
    trials = rng.integers(10, 30, size=len(group))
    offset = rng.normal(scale=0.2, size=len(group))
    eta = -0.5 + 0.7 * x - 0.4 * z + offset + rng.normal(scale=0.6, size=8)[group]
    data = pd.DataFrame(
        {
            "successes": rng.binomial(trials, 1 / (1 + np.exp(-eta))),
            "trials": trials,
            "x": x,
            "z": z,
            "group": group,
        }
    )
    return data, rng.uniform(0.6, 1.5, size=len(group)), offset


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_grouped_drop1_matches_independent_reduced_fits(grouped_comparison_data, weighted, n_jobs):
    data, prior, offset = grouped_comparison_data
    prior = prior if weighted else np.ones(len(data))
    control = glmerControl(check_conv=False, check_singular=False)
    full = glmer(
        "successes / trials ~ x + z + (1 | group)",
        data,
        weights=prior,
        offset=offset,
        control=control,
    )
    result = drop1_glmer(full, data, n_jobs=n_jobs)

    assert full.converged and full.pirls_converged
    assert result.terms == ["x", "z"]
    assert result.full_model_aic == pytest.approx(full.AIC())
    for index, retained in enumerate(["z", "x"]):
        reduced = glmer(
            f"successes / trials ~ {retained} + (1 | group)",
            data,
            weights=prior,
            offset=offset,
            start=full.theta,
            control=control,
        )
        assert reduced.converged and reduced.pirls_converged
        assert result.df[index] == reduced.logLik().df
        assert result.aic[index] == pytest.approx(reduced.AIC(), rel=1e-9)
        reported_loglik = (2 * result.df[index] - result.aic[index]) / 2
        assert reported_loglik == pytest.approx(reduced.logLik().value, rel=1e-9)
        lrt = 2 * (full.logLik().value - reduced.logLik().value)
        assert lrt > 0
        assert result.lrt[index] == pytest.approx(lrt, rel=1e-9)
        assert result.p_value[index] == pytest.approx(chi2.sf(lrt, 1), rel=1e-9)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_grouped_allfit_preserves_original_priors_and_offsets(
    grouped_comparison_data, weighted, n_jobs
):
    data, prior, offset = grouped_comparison_data
    prior = prior if weighted else np.ones(len(data))
    formula = "successes / trials ~ x + z + (1 | group)"
    control = glmerControl(check_conv=False, check_singular=False)
    full = glmer(formula, data, weights=prior, offset=offset, control=control)
    result = allfit_glmer(full, data, optimizers=["COBYQA", "L-BFGS-B"], n_jobs=n_jobs)

    assert not result.errors
    for optimizer, fit in result.fits.items():
        expected = glmer(
            formula, data, weights=prior, offset=offset, control=control, method=optimizer
        )
        assert fit is not None
        assert_array_equal(fit.matrices.trials, data["trials"])
        assert_allclose(fit.matrices.weights, prior * data["trials"], rtol=1e-15)
        assert_array_equal(fit.matrices.offset, offset)
        assert_allclose(fit.beta, expected.beta, rtol=1e-7, atol=1e-7)
        assert_allclose(fit.theta, expected.theta, rtol=1e-7, atol=1e-7)
        assert fit.deviance == pytest.approx(expected.deviance, rel=1e-10)
        assert fit.logLik().value == pytest.approx(expected.logLik().value, rel=1e-10)
        assert fit.AIC() == pytest.approx(expected.AIC(), rel=1e-10)
        assert fit.converged == expected.converged
        assert fit.pirls_converged == expected.pirls_converged

import numpy as np
import pytest
from mixedlm import families, glFormula, glmer, load_cbpp, parse_formula
from mixedlm.formula.parser import getFixedFormulaStr, nobars, update_formula


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
    data = pl.from_pandas(load_cbpp())

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

    assert grouped.converged
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

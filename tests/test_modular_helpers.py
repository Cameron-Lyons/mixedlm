"""Template, random-term, simulation and modular-fit helpers exported at the root."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    checkConv,
    families,
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
    quickSimulate,
    set_cov_type,
    simulate_formula,
)
from mixedlm.models.modular import ReTrms
from numpy.testing import assert_array_equal
from scipy import sparse

from tests._lmer_data import CBPP, SLEEPSTUDY

PREDICTORS = pd.DataFrame(
    {
        "y": np.linspace(-1.0, 1.0, 12),
        "x": [0.3, -1.2, 0.8, 1.5, -0.4, 0.1, -0.9, 2.0, 0.6, -1.7, 1.1, -0.2],
        "f": list("abc") * 4,
        "g": np.repeat(["A", "B", "C", "D"], 3),
        "h": list("pq") * 6,
    }
)


class TestDataTemplates:
    def test_balanced_template_crosses_every_grouping_level(self) -> None:
        data = mkDataTemplate("y ~ x + (1|subject) + (1|item)", nlevs={"subject": 10, "item": 5})

        assert list(data.columns) == ["y", "x", "subject", "item"]
        assert data.shape == (50, 4)
        assert set(data["subject"]) == {f"subject{i}" for i in range(1, 11)}
        assert set(data["item"]) == {f"item{i}" for i in range(1, 6)}
        assert data.groupby(["subject", "item"]).size().eq(1).all()

    def test_single_factor_template_has_one_row_per_level(self) -> None:
        data = mkDataTemplate("y ~ x + (1|subject)", nlevs={"subject": 20})

        assert data.shape == (20, 3)
        assert data["subject"].nunique() == 20

    def test_unbalanced_template_keeps_every_level(self) -> None:
        np.random.seed(4)
        data = mkDataTemplate("y ~ x + (1|a) + (1|b)", nlevs={"a": 10, "b": 5}, balanced=False)

        assert data.shape == (30, 4)
        assert data["a"].nunique() == 10
        assert data["b"].nunique() == 5
        assert data["a"].value_counts().nunique() > 1
        assert lFormula("y ~ x + (1|a) + (1|b)", data).n_theta == 2

    def test_template_variables_come_from_the_parsed_formula(self) -> None:
        data = mkDataTemplate("y ~ x * z + (w || g/h)", nlevs={"g": 3, "h": 2})

        assert list(data.columns) == ["y", "x", "z", "w", "g", "h"]
        assert len(data) == 6
        assert lFormula("y ~ x * z + (w || g/h)", data).n_fixed == 4

    def test_minimal_data_matches_the_documented_columns(self) -> None:
        data = mkMinimalData("y ~ x + z + (1|group)")

        assert list(data.columns) == ["y", "x", "z", "group"]
        assert len(data) == 10
        assert data["group"].nunique() == 5
        assert lFormula("y ~ x + z + (1|group)", data).n_fixed == 3

    def test_minimal_data_includes_interaction_and_slope_variables(self) -> None:
        data = mkMinimalData("y ~ x * z + (w | g) + (1 | h)", n=12)

        assert list(data.columns) == ["y", "x", "z", "w", "g", "h"]
        assert lFormula("y ~ x * z + (w | g) + (1 | h)", data).n_theta == 4


class TestParameterTemplate:
    def test_documented_template(self) -> None:
        data = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3], "g": ["A", "B", "A"]})
        template = mkParsTemplate("y ~ x + (1|g)", data)

        assert template["beta"] == {"(Intercept)": None, "x": None}
        assert template["theta"] == ["sd_(Intercept)|g"]
        assert template["n_theta"] == 1

    def test_correlated_slopes_label_the_cholesky_entries_in_theta_order(self) -> None:
        template = mkParsTemplate("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        assert template["theta"] == [
            "sd_(Intercept)|Subject",
            "cor_(Intercept)_Days|Subject",
            "sd_Days|Subject",
        ]

    @pytest.mark.parametrize("cov_type", ["cs", "ar1"])
    def test_structured_covariances_have_one_label_per_parameter(self, cov_type) -> None:
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", cov_type)
        template = mkParsTemplate(formula, SLEEPSTUDY)

        assert template["theta"] == ["sd|Subject", "rho|Subject"]
        assert template["n_theta"] == lFormula(formula, SLEEPSTUDY).n_theta


class TestNewRandomTerms:
    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ x + (f | g)",
            "y ~ x + (1 | g/h)",
            "y ~ (1 | g) + (0 + x | g)",
            "y ~ x + (x || g) + (1 | g:h)",
        ],
    )
    @pytest.mark.parametrize("rows", [[5, 0, 7], [0, 3]])
    def test_new_rows_reproduce_the_training_design(self, formula, rows) -> None:
        terms = mkReTrms(formula, PREDICTORS)
        new_terms = mkNewReTrms(terms, PREDICTORS.iloc[rows])

        assert_array_equal(new_terms.Zt.toarray(), terms.Zt.toarray()[:, rows])
        assert new_terms.nl == terms.nl
        assert new_terms.Gp == terms.Gp

    def test_unseen_levels_have_no_random_effect_columns(self) -> None:
        terms = mkReTrms("y ~ x + (1 | g) + (1 | h)", PREDICTORS)
        new_data = PREDICTORS.iloc[[0, 1]].assign(g=["A", "unseen"])

        Zt = mkNewReTrms(terms, new_data).Zt.toarray()

        assert_array_equal(Zt[:, 0], terms.Zt.toarray()[:, 0])
        # The unseen subject contributes nothing; its h level is still known.
        assert_array_equal(Zt[:4, 1], np.zeros(4))
        assert_array_equal(Zt[4:, 1], terms.Zt.toarray()[4:, 1])

    def test_requires_terms_from_mkretrms(self) -> None:
        terms = ReTrms(
            Zt=sparse.csc_matrix((1, 2)),
            theta=np.ones(1),
            Lind=np.zeros(1, dtype=int),
            Gp=[0, 1],
            flist={"g": np.array(["A"])},
            cnms={"g": ["(Intercept)"]},
            nl=[1],
        )
        with pytest.raises(ValueError, match="mkReTrms"):
            mkNewReTrms(terms, PREDICTORS)


class TestSimulateFormula:
    def test_documented_examples_do_not_need_a_response_column(self) -> None:
        rng = np.random.default_rng(2)
        data = pd.DataFrame(
            {"x": rng.normal(size=100), "group": np.repeat(["A", "B", "C", "D", "E"], 20)}
        )

        simulated = simulate_formula(
            "y ~ x + (1|group)", data, beta={"(Intercept)": 5.0, "x": 2.0}, theta=[0.5], seed=1
        )
        quick = quickSimulate(
            "y ~ x + (1|group)", data, beta={"(Intercept)": 5.0, "x": 2.0}, sigma=1.0, seed=1
        )

        for frame in (simulated, quick):
            assert list(frame.columns) == ["x", "group", "y"]
            slope = np.polyfit(frame["x"], frame["y"], 1)[0]
            assert slope == pytest.approx(2.0, abs=0.3)
        assert "y" not in data

    @pytest.mark.parametrize(
        ("name", "check"),
        [
            ("poisson", lambda y: np.all(y == np.round(y)) and np.all(y >= 0)),
            ("binomial", lambda y: set(np.unique(y)) <= {0.0, 1.0}),
            ("Gamma", lambda y: np.all(y > 0)),
            ("inverse.gaussian", lambda y: np.all(y > 0)),
        ],
    )
    def test_family_names_select_the_response_distribution(self, name, check) -> None:
        simulated = simulate_formula(
            "Reaction ~ Days + (1 | Subject)",
            SLEEPSTUDY,
            beta=np.array([0.1, 0.0]),
            theta=np.array([0.0]),
            family=name,
            seed=3,
        )

        assert check(simulated["Reaction"].to_numpy())

    def test_unknown_family_names_raise(self) -> None:
        with pytest.raises(ValueError, match="Unknown family 'bogus'"):
            simulate_formula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, family="bogus")
        with pytest.raises(ValueError, match="Unknown family 'bogus'"):
            quickSimulate("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, family="bogus")

    @pytest.mark.parametrize("name", ["gamma", "poisson", "binomial"])
    def test_quick_simulate_dispatches_family_names(self, name) -> None:
        arguments = {"beta": np.array([0.1, 0.0]), "theta": np.array([0.0]), "seed": 3}
        formula = "Reaction ~ Days + (1 | Subject)"

        quick = quickSimulate(formula, SLEEPSTUDY, family=name, **arguments)
        direct = simulate_formula(formula, SLEEPSTUDY, family=name, **arguments)

        assert_array_equal(quick["Reaction"], direct["Reaction"])

    def test_grouped_binomial_draws_success_counts_from_the_trials(self) -> None:
        data = pd.DataFrame({"n": np.full(300, 20), "g": np.repeat(np.arange(10), 30)})

        counts = simulate_formula(
            "s / n ~ 1 + (1 | g)",
            data,
            beta=np.array([0.0]),
            theta=np.array([0.0]),
            family=families.Binomial(),
            seed=5,
        )["s"].to_numpy()

        assert np.all(counts == np.round(counts))
        assert counts.min() >= 0 and counts.max() <= 20
        assert counts.mean() == pytest.approx(10.0, abs=0.5)
        assert counts.var() == pytest.approx(5.0, rel=0.25)

    @pytest.mark.parametrize(
        ("family", "variance"),
        [
            (families.Gaussian(), lambda mu, sigma: sigma**2),
            (families.Gamma(link="log"), lambda mu, sigma: sigma**2 * mu**2),
            (families.InverseGaussian(link="log"), lambda mu, sigma: sigma**2 * mu**3),
        ],
    )
    def test_sigma_sets_the_dispersion(self, family, variance) -> None:
        data = pd.DataFrame({"g": np.repeat(np.arange(20), 200)})
        mu, sigma = 2.0, 0.5
        intercept = mu if isinstance(family, families.Gaussian) else np.log(mu)

        y = simulate_formula(
            "y ~ 1 + (1 | g)",
            data,
            beta=np.array([intercept]),
            theta=np.array([0.0]),
            sigma=sigma,
            family=family,
            seed=8,
        )["y"].to_numpy()

        assert y.mean() == pytest.approx(mu, rel=0.02)
        assert y.var() == pytest.approx(variance(mu, sigma), rel=0.1)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"beta": {"Day": 1.0}}, r"unknown coefficient names \['Day'\]"),
            ({"sigma": 0.0}, "sigma must be finite and positive"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs, message) -> None:
        with pytest.raises(ValueError, match=message):
            simulate_formula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, **kwargs)


class TestModularFitMetadata:
    def test_linear_fit_records_the_optimizer_for_checkconv(self) -> None:
        devfun = mkLmerDevfun(lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY))
        fit = mkLmerMod(devfun, optimizeLmer(devfun, method="Nelder-Mead"))

        assert fit.optimizer == "Nelder-Mead"
        assert checkConv(fit).optimizer == "Nelder-Mead"

    def test_unconverged_linear_fit_reports_the_optimizer_message(self) -> None:
        devfun = mkLmerDevfun(lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY))
        opt = optimizeLmer(devfun, maxiter=1)
        fit = mkLmerMod(devfun, opt)

        assert not fit.converged
        assert fit.message == opt.message != ""
        assert f"Optimizer did not report convergence: {opt.message}" in checkConv(fit).messages

    def test_generalized_fit_records_the_optimizer_for_checkconv(self) -> None:
        parsed = glFormula(
            "incidence / size ~ period + (1 | herd)", CBPP, family=families.Binomial()
        )
        devfun = mkGlmerDevfun(parsed)
        opt = optimizeGlmer(devfun)
        fit = mkGlmerMod(devfun, opt)

        assert fit.optimizer == "L-BFGS-B"
        assert fit.message == opt.message
        assert checkConv(fit).optimizer == "L-BFGS-B"

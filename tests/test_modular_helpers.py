"""Template, random-term, simulation and modular-fit helpers exported at the root."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    OptimizeResult,
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
from mixedlm.models.modular import ReTrms, devfun2
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.optimize import minimize

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY, grouped_data

SLEEP_FORMULA = "Reaction ~ Days + (1 | Subject)"

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


class TestMkReTrms:
    def test_scalar_terms_match_the_fitted_design(self, sleepstudy_lmm) -> None:
        terms = mkReTrms(SLEEP_FORMULA, SLEEPSTUDY)

        assert_array_equal(terms.Zt.toarray(), sleepstudy_lmm.getME("Z").toarray().T)
        assert_array_equal(terms.theta, [1.0])
        assert_array_equal(terms.Lind, np.zeros(18))
        assert terms.Gp == [0, 18]
        assert terms.nl == [18]
        assert list(terms.flist) == ["Subject"]
        assert terms.cnms == {"Subject": ["(Intercept)"]}

    def test_correlated_slopes_start_at_the_identity_factor(self) -> None:
        terms = mkReTrms("Reaction ~ Days + (Days|Subject)", SLEEPSTUDY)

        assert terms.Zt.shape == (36, 180)
        assert_array_equal(terms.theta, [1.0, 0.0, 1.0])
        assert_array_equal(np.bincount(terms.Lind), [18, 18, 18])
        assert terms.Gp == [0, 36]

    def test_crossed_grouping_factors(self) -> None:
        data = pd.DataFrame(
            {
                "y": np.zeros(100),
                "x": np.linspace(-1.0, 1.0, 100),
                "g1": np.repeat(np.arange(10), 10).astype(str),
                "g2": np.tile(np.arange(5), 20).astype(str),
            }
        )

        terms = mkReTrms("y ~ x + (1|g1) + (1|g2)", data)

        assert list(terms.flist) == ["g1", "g2"]
        assert terms.nl == [10, 5]
        assert terms.Gp == [0, 10, 15]


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
    def test_seeded_draws_are_reproducible_and_accept_named_coefficients(self) -> None:
        arguments = {"theta": np.array([1.0]), "sigma": 25.0, "seed": 123}

        positional = simulate_formula(
            SLEEP_FORMULA, SLEEPSTUDY, beta=np.array([250.0, 10.0]), **arguments
        )
        named = simulate_formula(
            SLEEP_FORMULA, SLEEPSTUDY, beta={"(Intercept)": 250.0, "Days": 10.0}, **arguments
        )
        several = simulate_formula(
            SLEEP_FORMULA, SLEEPSTUDY, beta=np.array([250.0, 10.0]), nsim=5, **arguments
        )

        assert_array_equal(named["Reaction"], positional["Reaction"])
        pd.testing.assert_frame_equal(
            positional.drop(columns="Reaction"), SLEEPSTUDY.drop(columns="Reaction")
        )
        assert len(several) == 5
        assert_array_equal(several[0]["Reaction"], positional["Reaction"])
        assert not np.allclose(several[0]["Reaction"], several[1]["Reaction"])

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
            ({"nsim": 0}, "nsim must be at least 1"),
            ({"sigma": -1.0}, "sigma must be finite and positive"),
            ({"sigma": 0.0}, "sigma must be finite and positive"),
            ({"sigma": np.inf}, "sigma must be finite and positive"),
            ({"beta": np.array([1.0])}, "beta has length 1; expected 2"),
            ({"theta": np.array([1.0, 2.0])}, "theta must contain 1 parameters, got 2"),
            ({"theta": np.array([np.nan])}, "theta must contain only finite values"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs, message) -> None:
        with pytest.raises(ValueError, match=message):
            simulate_formula(SLEEP_FORMULA, SLEEPSTUDY, seed=42, **kwargs)

    def test_validates_theta_length_for_correlated_terms(self) -> None:
        with pytest.raises(ValueError, match="theta must contain 3 parameters, got 2"):
            simulate_formula(
                "Reaction ~ Days + (Days | Subject)",
                SLEEPSTUDY,
                theta=np.ones(2),
            )

    @pytest.mark.parametrize(
        ("family", "response_kind"),
        [
            (families.Gaussian(), "continuous"),
            (families.Binomial(), "binary"),
            (families.Poisson(), "count"),
            (families.NegativeBinomial(theta=2.0), "count"),
            (families.Gamma(), "positive"),
            (families.GammaInverse(), "positive"),
            (families.InverseGaussian(), "positive"),
            (families.InverseGaussianCanonical(), "positive"),
        ],
    )
    def test_supports_builtin_families(self, family, response_kind) -> None:
        simulations = simulate_formula(
            SLEEP_FORMULA,
            SLEEPSTUDY,
            beta=np.array([0.2, 0.0]),
            theta=np.array([0.0]),
            sigma=0.5,
            family=family,
            nsim=3,
            seed=42,
        )
        values = np.concatenate([simulation["Reaction"].to_numpy() for simulation in simulations])

        assert np.all(np.isfinite(values))
        if response_kind == "binary":
            assert np.all((values == 0) | (values == 1))
        elif response_kind == "count":
            assert np.all(values >= 0)
            assert np.all(values == values.astype(int))
        elif response_kind == "positive":
            assert np.all(values > 0)

    def test_preserves_global_random_state(self) -> None:
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)

        simulate_formula(SLEEP_FORMULA, SLEEPSTUDY, theta=np.array([0.0]), seed=999)
        observed = np.random.random(5)

        np.testing.assert_array_equal(observed, expected)

    @staticmethod
    def stub_rng(monkeypatch, draws):
        """Make simulations deterministic: fixed normal draws and noiseless responses."""

        class StubRNG:
            def standard_normal(self, size):
                assert size == len(draws)
                return np.asarray(draws, dtype=float)

            def normal(self, loc, scale):
                return np.asarray(loc)

        monkeypatch.setattr(np.random, "default_rng", lambda seed: StubRNG())

    def test_orders_uncorrelated_effects_by_level(self, monkeypatch) -> None:
        self.stub_rng(monkeypatch, [1.0, 10.0, 2.0, 20.0])
        data = pd.DataFrame(
            {
                "y": np.zeros(4),
                "x": [0.0, 1.0, 0.0, 1.0],
                "group": ["a", "a", "b", "b"],
            }
        )

        simulated = simulate_formula(
            "y ~ 1 + (1 + x || group)",
            data,
            beta=np.array([0.0]),
            theta=np.array([2.0, 3.0]),
            sigma=1.0,
            family=families.Gaussian(),
            seed=42,
        )

        np.testing.assert_allclose(simulated["y"], [2.0, 32.0, 4.0, 64.0])

    def test_uses_lower_triangular_theta_order(self, monkeypatch) -> None:
        self.stub_rng(monkeypatch, [1.0, 10.0, 100.0])
        data = pd.DataFrame(
            {
                "y": np.zeros(3),
                "x": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
                "group": ["a", "a", "a"],
            }
        )

        simulated = simulate_formula(
            "y ~ 1 + (1 + x + z | group)",
            data,
            beta=np.array([0.0]),
            theta=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            sigma=1.0,
            family=families.Gaussian(),
            seed=42,
        )

        np.testing.assert_allclose(simulated["y"], [1.0, 33.0, 655.0])

    def test_correlated_theta_order(self, monkeypatch) -> None:
        self.stub_rng(monkeypatch, np.ones(6))
        data = pd.DataFrame(
            {
                "y": np.zeros(6),
                "x": [1, 0, 0, 1, 0, 0],
                "z": [0, 1, 0, 0, 1, 0],
                "w": [0, 0, 1, 0, 0, 1],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )

        result = simulate_formula(
            "y ~ 0 + x + z + w + (0 + x + z + w | group)",
            data,
            beta=np.zeros(3),
            theta=np.array([1, 2, 3, 4, 5, 6]),
        )

        np.testing.assert_allclose(result["y"], [1, 5, 15, 1, 5, 15])

    def test_uncorrelated_level_order(self, monkeypatch) -> None:
        self.stub_rng(monkeypatch, [1, 2, 3, 4])
        data = pd.DataFrame(
            {
                "y": np.zeros(4),
                "x": [1, 0, 1, 0],
                "z": [0, 1, 0, 1],
                "group": ["A", "A", "B", "B"],
            }
        )

        result = simulate_formula(
            "y ~ 0 + x + z + (0 + x + z || group)",
            data,
            beta=np.zeros(2),
            theta=np.array([2, 3]),
        )

        np.testing.assert_allclose(result["y"], [2, 6, 6, 12])

    def test_structured_covariance(self, monkeypatch) -> None:
        self.stub_rng(monkeypatch, [1, 0])
        data = pd.DataFrame(
            {
                "y": np.zeros(2),
                "x": [1, 0],
                "z": [0, 1],
                "group": ["A", "A"],
            }
        )
        formula = set_cov_type("y ~ 0 + x + z + (0 + x + z | group)", "cs")

        result = simulate_formula(formula, data, beta=np.zeros(2), theta=np.array([2, 0.5]))

        np.testing.assert_allclose(result["y"], [2, 1])


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


class TestDevfun2:
    def test_varies_only_the_selected_parameters(self, sleepstudy_slopes_lmm) -> None:
        devfun = mkLmerDevfun(lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY))
        theta = sleepstudy_slopes_lmm.theta
        moved = theta.copy()
        moved[0] *= 1.5

        full = devfun2(devfun, theta)
        first = devfun2(devfun, theta, which=[0])

        assert full(theta) == pytest.approx(sleepstudy_slopes_lmm.deviance)
        assert full(moved) == pytest.approx(devfun(moved))
        assert first(np.array([moved[0]])) == pytest.approx(devfun(moved))
        assert first(np.array([moved[0]])) > first(np.array([theta[0]]))


class TestModularWorkflow:
    def test_lformula_describes_the_model(self) -> None:
        parsed = lFormula(SLEEP_FORMULA, SLEEPSTUDY)

        assert parsed.n_obs == 180
        assert parsed.n_fixed == 2
        assert parsed.n_random == 18
        assert parsed.n_theta == 1
        assert parsed.X.shape == (180, 2)
        assert parsed.y.shape == (180,)
        assert parsed.REML is True
        assert lFormula(SLEEP_FORMULA, SLEEPSTUDY, REML=False).REML is False

    def test_lformula_correlated_terms(self) -> None:
        parsed = lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        assert parsed.n_theta == 3
        assert parsed.Z.shape == (180, 36)
        assert mkLmerDevfun(parsed).get_bounds() == [(0.0, None), (None, None), (0.0, None)]

    def test_lformula_accepts_weights_and_offset(self) -> None:
        weights = np.r_[np.full(90, 2.0), np.ones(90)]
        offset = np.r_[np.full(90, 10.0), np.zeros(90)]

        parsed = lFormula(SLEEP_FORMULA, SLEEPSTUDY, weights=weights, offset=offset)

        assert_array_equal(parsed.matrices.weights, weights)
        assert_array_equal(parsed.matrices.offset, offset)

    def test_lformula_accepts_composed_formula(self) -> None:
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", "cs")
        parsed = lFormula(formula, SLEEPSTUDY)

        assert parsed.formula is formula
        assert parsed.n_theta == 2
        assert parsed.matrices.random_structures[0].cov_type == "cs"

        devfun = mkLmerDevfun(parsed)
        optimized = optimizeLmer(devfun)
        result = mkLmerMod(devfun, optimized)
        assert result.converged
        assert len(result.theta) == 2

    def test_lmer_steps_reproduce_lmer(self, sleepstudy_lmm) -> None:
        devfun = mkLmerDevfun(lFormula(SLEEP_FORMULA, SLEEPSTUDY))

        start = devfun.get_start()
        opt = optimizeLmer(devfun)
        result = mkLmerMod(devfun, opt)

        assert start.shape == (1,)
        assert 0.0 < start[0] < 10.0
        assert devfun(sleepstudy_lmm.theta) == pytest.approx(sleepstudy_lmm.deviance)
        assert opt.converged
        assert opt.deviance == pytest.approx(sleepstudy_lmm.deviance, abs=1e-6)
        assert_allclose(result.beta, sleepstudy_lmm.beta, rtol=1e-5)
        assert_allclose(result.theta, sleepstudy_lmm.theta, rtol=1e-4)
        assert result.sigma == pytest.approx(sleepstudy_lmm.sigma, rel=1e-4)
        assert_allclose(
            result.ranef()["Subject"]["(Intercept)"], sleepstudy_lmm.getME("b"), atol=1e-3
        )

    def test_custom_optimizer_result_builds_the_fit(self, sleepstudy_lmm) -> None:
        devfun = mkLmerDevfun(lFormula(SLEEP_FORMULA, SLEEPSTUDY))
        optimum = minimize(devfun, devfun.get_start(), method="Nelder-Mead")
        opt = OptimizeResult(
            theta=optimum.x,
            deviance=optimum.fun,
            converged=optimum.success,
            n_iter=optimum.nit,
            message="Custom optimizer",
        )

        result = mkLmerMod(devfun, opt)

        assert_allclose(list(result.fixef().values()), sleepstudy_lmm.beta, rtol=1e-4)

    def test_glformula_describes_the_model(self) -> None:
        parsed = glFormula(CBPP_FORMULA, CBPP, family=families.Binomial())

        assert parsed.n_obs == 56
        assert parsed.n_fixed == 4
        assert isinstance(parsed.family, families.Binomial)
        assert parsed.n_theta == 1
        assert_array_equal(parsed.matrices.weights, CBPP["size"])

    def test_glformula_accepts_composed_formula(self) -> None:
        formula = set_cov_type("incidence / size ~ period + (period | herd)", "cs")

        parsed = glFormula(formula, CBPP, family=families.Binomial())

        assert parsed.formula is formula
        assert parsed.n_theta == 2
        assert parsed.matrices.random_structures[0].cov_type == "cs"

    def test_glmer_steps_reproduce_glmer(self, cbpp_glmm) -> None:
        devfun = mkGlmerDevfun(glFormula(CBPP_FORMULA, CBPP, family=families.Binomial()))

        assert_array_equal(devfun.get_start(), [1.0])
        opt = optimizeGlmer(devfun)
        result = mkGlmerMod(devfun, opt)

        assert opt.converged
        assert opt.deviance == pytest.approx(cbpp_glmm.deviance, abs=1e-6)
        assert_allclose(result.beta, cbpp_glmm.beta, atol=1e-4)
        assert_allclose(result.theta, cbpp_glmm.theta, atol=1e-4)
        assert_allclose(result.ranef()["herd"]["(Intercept)"], cbpp_glmm.getME("b"), atol=1e-4)

    def test_modular_fits_match_one_step_fits_on_grouped_data(self, grouped_lmm, grouped_glmm):
        lmm_devfun = mkLmerDevfun(lFormula("y ~ x + (1 | group)", grouped_data()))
        lmm = mkLmerMod(lmm_devfun, optimizeLmer(lmm_devfun))
        parsed = glFormula(
            "y ~ x + (1 | group)", grouped_data("binomial"), family=families.Binomial()
        )
        glmm_devfun = mkGlmerDevfun(parsed)
        glmm = mkGlmerMod(glmm_devfun, optimizeGlmer(glmm_devfun))

        assert_allclose(lmm.beta, grouped_lmm.beta, rtol=1e-5)
        assert_allclose(glmm.beta, grouped_glmm.beta, atol=1e-4)
        assert_allclose(glmm.theta, grouped_glmm.theta, atol=1e-4)

"""Prediction, missing-data handling, influence, rePCA and contrast codings."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    families,
    glmer,
    lmer,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit

from tests._datasets import CBPP, SLEEPSTUDY, grouped_data


class TestPredict:
    def test_lmer_predictions_without_or_with_the_fitted_data(self, sleepstudy_lmm):
        assert_allclose(sleepstudy_lmm.predict(), sleepstudy_lmm.fitted())
        assert_allclose(sleepstudy_lmm.predict(newdata=SLEEPSTUDY), sleepstudy_lmm.fitted())
        new_data = SLEEPSTUDY.drop(columns="Reaction")
        assert_allclose(sleepstudy_lmm.predict(newdata=new_data), sleepstudy_lmm.fitted())

    def test_lmer_predict_fixed_only(self, sleepstudy_lmm):
        pred_fixed = sleepstudy_lmm.predict(newdata=SLEEPSTUDY, re_form="NA")

        fixef = sleepstudy_lmm.fixef()
        expected_fixed = fixef["(Intercept)"] + fixef["Days"] * SLEEPSTUDY["Days"].values
        assert np.allclose(pred_fixed, expected_fixed)

    def test_lmer_predict_fixed_only_without_newdata(self):
        rng = np.random.default_rng(42)
        n_groups = 8
        n_per_group = 15
        groups = np.repeat(np.arange(n_groups), n_per_group)
        x = rng.normal(size=len(groups))
        group_effects = np.repeat(rng.normal(0, 3, n_groups), n_per_group)
        y = 2 + 0.5 * x + group_effects + rng.normal(0, 0.3, len(groups))
        data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
        result = lmer("y ~ x + (1 | group)", data)
        expected = result.matrices.X @ result.beta + result.matrices.offset

        assert not np.allclose(result.fitted(), expected)
        assert np.allclose(result.predict(re_form="NA"), expected)
        assert np.allclose(result.predict(re_form="~0"), expected)
        with_se = result.predict(re_form="NA", se_fit=True)
        with_interval = result.predict(re_form="~0", interval="confidence")
        assert np.allclose(with_se.fit, expected)
        assert np.allclose(with_interval.fit, expected)

    def test_lmer_predict_new_levels_error(self, sleepstudy_lmm):
        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": ["999"]})

        with pytest.raises(ValueError, match="New level"):
            sleepstudy_lmm.predict(newdata=new_data)

    def test_lmer_predict_new_levels_allowed(self, sleepstudy_lmm):
        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": ["999"]})

        pred = sleepstudy_lmm.predict(newdata=new_data, allow_new_levels=True)
        fixef = sleepstudy_lmm.fixef()
        expected = fixef["(Intercept)"] + fixef["Days"] * 5.0

        assert np.isclose(pred[0], expected)

    def test_lmer_predict_adds_the_subject_intercept_and_slope(self, sleepstudy_slopes_lmm):
        new_data = pd.DataFrame({"Days": [5.0], "Subject": ["308"]})
        ranef = sleepstudy_slopes_lmm.ranef()["Subject"]
        fixef = sleepstudy_slopes_lmm.fixef()

        pred = sleepstudy_slopes_lmm.predict(newdata=new_data)

        expected = (
            fixef["(Intercept)"]
            + ranef["(Intercept)"][0]
            + 5.0 * (fixef["Days"] + ranef["Days"][0])
        )
        assert pred[0] == pytest.approx(expected)

    def test_new_level_standard_errors_include_the_random_effect_covariance(
        self, sleepstudy_slopes_lmm
    ):
        new_data = pd.DataFrame(
            {
                "Reaction": [300.0, 320.0],
                "Days": [2.0, 7.0],
                "Subject": ["new_subject_a", "new_subject_b"],
            }
        )
        X = np.column_stack((np.ones(2), new_data["Days"]))
        covariance = sleepstudy_slopes_lmm.vcov() + sleepstudy_slopes_lmm.VarCorr().get_cov(
            "Subject"
        )

        pred = sleepstudy_slopes_lmm.predict(newdata=new_data, allow_new_levels=True, se_fit=True)
        pred_fixed = sleepstudy_slopes_lmm.predict(newdata=new_data, re_form="NA")

        assert_allclose(pred.fit, pred_fixed)
        assert_allclose(pred.se_fit, np.sqrt(np.einsum("ij,jk,ik->i", X, covariance, X)))

    def test_glmer_predictions_on_link_and_response_scales(self, cbpp_glmm):
        X, Z, b = cbpp_glmm.getME("X"), cbpp_glmm.getME("Z"), cbpp_glmm.getME("b")

        assert_allclose(cbpp_glmm.predict(), cbpp_glmm.fitted())
        assert_allclose(cbpp_glmm.predict(newdata=CBPP), expit(X @ cbpp_glmm.beta + Z @ b))
        assert_allclose(cbpp_glmm.predict(newdata=CBPP, type="link"), X @ cbpp_glmm.beta + Z @ b)
        assert_allclose(cbpp_glmm.predict(newdata=CBPP, re_form="NA"), expit(X @ cbpp_glmm.beta))
        new_data = CBPP.drop(columns="incidence")
        assert_allclose(cbpp_glmm.predict(newdata=new_data), cbpp_glmm.fitted())

    def test_glmer_predict_fixed_only_without_newdata(self, cbpp_glmm):
        expected_link = cbpp_glmm.matrices.X @ cbpp_glmm.beta + cbpp_glmm.matrices.offset
        expected_response = cbpp_glmm.family.link.inverse(expected_link)

        assert not np.allclose(cbpp_glmm.fitted(type="link"), expected_link)
        for re_form in ("NA", "~0"):
            assert np.allclose(cbpp_glmm.predict(type="link", re_form=re_form), expected_link)
            assert np.allclose(
                cbpp_glmm.predict(type="response", re_form=re_form),
                expected_response,
            )
        with_se = cbpp_glmm.predict(type="link", re_form="NA", se_fit=True)
        with_interval = cbpp_glmm.predict(type="response", re_form="~0", interval="confidence")
        assert np.allclose(with_se.fit, expected_link)
        assert np.allclose(with_interval.fit, expected_response)

    def test_glmer_predict_new_levels_allowed(self, grouped_glmm):
        new_data = pd.DataFrame({"y": [0], "x": [0.5], "group": ["999"]})

        pred = grouped_glmm.predict(newdata=new_data, allow_new_levels=True)

        assert_allclose(pred, [expit(grouped_glmm.beta[0] + 0.5 * grouped_glmm.beta[1])])

    def test_lmer_predict_accepts_array_and_scalar_offsets(self, sleepstudy_lmm):
        new_data = SLEEPSTUDY.loc[:4, ["Days"]]
        baseline = sleepstudy_lmm.predict(new_data, re_form="NA")
        offset = np.linspace(-0.5, 0.5, len(new_data))

        predicted = sleepstudy_lmm.predict(new_data, re_form="NA", offset=offset)
        scalar_predicted = sleepstudy_lmm.predict(new_data, re_form="NA", offset=1.25)

        assert np.allclose(predicted, baseline + offset)
        assert np.allclose(scalar_predicted, baseline + 1.25)

    def test_glmer_predict_accepts_offset_column_on_link_scale(self, cbpp_glmm):
        new_data = CBPP.loc[:4, ["period"]].copy()
        offset = np.linspace(-0.4, 0.6, len(new_data))
        new_data["log_exposure"] = offset

        baseline = cbpp_glmm.predict(
            new_data,
            type="link",
            re_form="NA",
            interval="confidence",
        )
        shifted = cbpp_glmm.predict(
            new_data,
            type="link",
            re_form="NA",
            interval="confidence",
            offset="log_exposure",
        )
        response = cbpp_glmm.predict(
            new_data,
            type="response",
            re_form="NA",
            offset="log_exposure",
        )

        assert baseline.lower is not None and baseline.upper is not None
        assert shifted.lower is not None and shifted.upper is not None
        assert np.allclose(shifted.fit, baseline.fit + offset)
        assert np.allclose(shifted.se_fit, baseline.se_fit)
        assert np.allclose(shifted.lower, baseline.lower + offset)
        assert np.allclose(shifted.upper, baseline.upper + offset)
        assert np.allclose(response, cbpp_glmm.family.link.inverse(shifted.fit))

    @pytest.mark.parametrize(
        ("offset", "message"),
        [
            ([1.0, 2.0], "length 2; expected 3"),
            ([[1.0], [2.0], [3.0]], "scalar or one-dimensional"),
            ([0.0, np.nan, 1.0], "only finite"),
            (["low", "medium", "high"], "numeric values"),
        ],
    )
    def test_predict_rejects_invalid_offsets(self, sleepstudy_lmm, offset, message):
        new_data = SLEEPSTUDY.loc[:2, ["Days"]]

        with pytest.raises(ValueError, match=message):
            sleepstudy_lmm.predict(new_data, re_form="NA", offset=offset)

    def test_predict_rejects_missing_offset_column(self, sleepstudy_lmm):
        new_data = SLEEPSTUDY.loc[:2, ["Days"]]

        with pytest.raises(ValueError, match="missing offset column 'exposure'"):
            sleepstudy_lmm.predict(new_data, re_form="NA", offset="exposure")

    def test_predict_rejects_offset_without_newdata(self, sleepstudy_lmm):
        with pytest.raises(ValueError, match="only be supplied with newdata"):
            sleepstudy_lmm.predict(offset=1.0)

    def test_predict_categorical_subset_uses_fitted_levels(self):
        data = pd.DataFrame(
            {
                "y": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0],
                "treatment": ["A", "B"] * 6,
                "subject": np.repeat([f"s{i}" for i in range(6)], 2),
            }
        )
        result = lmer(
            "y ~ treatment + (1 | subject)",
            data,
            contrasts={"treatment": "sum"},
        )
        expected = result.predict(data, re_form="NA")[data["treatment"] == "B"]
        new_data = data.loc[data["treatment"] == "B", ["treatment", "subject"]]

        predicted = result.predict(new_data, re_form="NA")

        assert np.allclose(predicted, expected)

    def test_predict_custom_contrast_schema_is_immutable(self):
        data = pd.DataFrame(
            {
                "y": np.arange(18.0),
                "treatment": ["A", "B", "C"] * 6,
                "subject": np.repeat([f"s{i}" for i in range(6)], 3),
            }
        )
        custom = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        result = lmer(
            "y ~ treatment + (1 | subject)",
            data,
            contrasts={"treatment": custom},
        )
        expected = result.predict(data, re_form="NA")[data["treatment"] == "C"]
        custom[:] = 0.0
        new_data = data.loc[data["treatment"] == "C", ["treatment"]]

        predicted = result.predict(new_data, re_form="NA")

        assert np.allclose(predicted, expected)

    def test_predict_interaction_subset_uses_fitted_levels(self):
        data = pd.DataFrame(
            {
                "y": np.arange(1.0, 17.0),
                "a": ["A", "A", "B", "B"] * 4,
                "b": ["X", "Y", "X", "Y"] * 4,
                "subject": np.repeat([f"s{i}" for i in range(8)], 2),
            }
        )
        result = lmer("y ~ a * b + (1 | subject)", data)
        mask = (data["a"] == "B") & (data["b"] == "Y")
        expected = result.predict(data, re_form="NA")[mask]
        new_data = data.loc[mask, ["a", "b", "subject"]]

        predicted = result.predict(new_data, re_form="NA")

        assert np.allclose(predicted, expected)

    def test_predict_rejects_unseen_fixed_effect_level(self):
        data = pd.DataFrame(
            {
                "y": np.arange(12.0),
                "treatment": ["A", "B"] * 6,
                "subject": np.repeat([f"s{i}" for i in range(6)], 2),
            }
        )
        result = lmer("y ~ treatment + (1 | subject)", data)
        new_data = pd.DataFrame({"treatment": ["C"], "subject": ["s0"]})

        with pytest.raises(ValueError, match="New level.*'C'.*treatment"):
            result.predict(new_data)

    def test_predict_reports_missing_fixed_effect_variables(self, sleepstudy_lmm):
        new_data = pd.DataFrame({"Subject": ["308"]})

        with pytest.raises(ValueError, match="missing fixed-effect variable.*'Days'"):
            sleepstudy_lmm.predict(new_data)


class TestNAAction:
    @staticmethod
    def data_with_missing_values(family="gaussian"):
        data = grouped_data(family)
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan
        return data

    def test_omit_matches_a_fit_to_the_complete_rows(self) -> None:
        data = self.data_with_missing_values()
        data.loc[10, "group"] = np.nan

        result = lmer("y ~ x + (1 | group)", data, na_action="omit")

        complete = lmer("y ~ x + (1 | group)", data.dropna())
        assert result.matrices.n_obs == 197
        assert len(result.fitted()) == len(result.residuals()) == 197
        assert_allclose(result.beta, complete.beta)
        assert_allclose(result.fitted(), complete.fitted())

    def test_exclude_pads_fitted_values_and_residuals_with_nan(self) -> None:
        data = self.data_with_missing_values()

        result = lmer("y ~ x + (1 | group)", data, na_action="exclude")

        omitted = lmer("y ~ x + (1 | group)", data, na_action="omit")
        missing = np.isin(np.arange(200), [0, 5])
        assert result.matrices.n_obs == 198
        for values, complete in (
            (result.fitted(), omitted.fitted()),
            (result.residuals(), omitted.residuals()),
        ):
            assert len(values) == 200
            assert_array_equal(np.isnan(values), missing)
            assert_allclose(values[~missing], complete)

    def test_fail_rejects_missing_values(self) -> None:
        with pytest.raises(ValueError, match="Missing values"):
            lmer("y ~ x + (1 | group)", self.data_with_missing_values(), na_action="fail")

    def test_complete_data_is_unchanged(self, grouped_lmm) -> None:
        result = lmer("y ~ x + (1 | group)", grouped_data(), na_action="omit")

        assert result.matrices.n_obs == 200
        assert_allclose(result.beta, grouped_lmm.beta)

    def test_glmer_omit_and_exclude(self) -> None:
        data = self.data_with_missing_values("binomial")
        missing = np.isin(np.arange(200), [0, 5])

        omitted = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), na_action="omit")
        excluded = glmer(
            "y ~ x + (1 | group)", data, family=families.Binomial(), na_action="exclude"
        )

        complete = glmer("y ~ x + (1 | group)", data.dropna(), family=families.Binomial())
        assert omitted.matrices.n_obs == excluded.matrices.n_obs == 198
        assert_allclose(omitted.beta, complete.beta)
        fitted = excluded.fitted()
        assert len(fitted) == 200
        assert_array_equal(np.isnan(fitted), missing)
        assert_allclose(fitted[~missing], omitted.fitted())


class TestInfluenceDiagnostics:
    def test_lmm_hat_values_are_the_diagonal_of_the_smoother(self, grouped_lmm) -> None:
        X = grouped_lmm.getME("X")
        ZL = (grouped_lmm.getME("Z") @ grouped_lmm.getME("Lambda")).toarray()
        design = np.hstack((X, ZL))
        penalized = design.T @ design
        penalized[2:, 2:] += np.eye(ZL.shape[1])
        hat = design @ np.linalg.solve(penalized, design.T)

        assert_allclose(hat @ grouped_lmm.getME("y"), grouped_lmm.fitted())
        assert_allclose(grouped_lmm.hatvalues(), np.diag(hat))

    def test_lmm_influence_measures_follow_their_definitions(self, grouped_lmm) -> None:
        h = grouped_lmm.hatvalues()
        residuals = grouped_lmm.residuals()
        sigma = grouped_lmm.sigma
        rss = np.sum(residuals**2)
        loo_variance = (rss - residuals**2 / (1 - h)) / (200 - 2 - 1)

        influence = grouped_lmm.influence()

        cooks = residuals**2 / (2 * sigma**2) * h / (1 - h) ** 2
        assert_allclose(grouped_lmm.cooks_distance(), cooks)
        assert_allclose(influence["hat"], h)
        assert_allclose(influence["cooks_d"], cooks)
        assert_allclose(influence["std_resid"], residuals / (sigma * np.sqrt(1 - h)))
        assert_allclose(influence["student_resid"], residuals / np.sqrt(loo_variance * (1 - h)))

    def test_glmm_influence_uses_pearson_residuals_and_unit_scale(self, cbpp_glmm) -> None:
        h = cbpp_glmm.hatvalues()
        pearson = cbpp_glmm.residuals(type="pearson")

        influence = cbpp_glmm.influence()

        cooks = pearson**2 / 4 * h / (1 - h) ** 2
        assert np.all((h > 0) & (h < 1))
        assert_allclose(influence["hat"], h)
        assert_allclose(influence["cooks_d"], cooks)
        assert_allclose(cbpp_glmm.cooks_distance(), cooks)
        assert_allclose(influence["pearson_resid"], pearson)
        assert_allclose(influence["deviance_resid"], cbpp_glmm.residuals(type="deviance"))


class TestRePCA:
    def test_correlated_terms_are_the_singular_values_of_the_scaled_factor(
        self, sleepstudy_slopes_lmm
    ) -> None:
        theta, sigma = sleepstudy_slopes_lmm.theta, sleepstudy_slopes_lmm.sigma
        block = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
        sdev = sigma * np.linalg.svd(block, compute_uv=False)

        pca = sleepstudy_slopes_lmm.rePCA()

        subject = pca["Subject"]
        assert list(pca.groups) == ["Subject"]
        assert subject.n_terms == 2
        assert_allclose(subject.sdev, sdev)
        assert_allclose(subject.proportion, sdev**2 / np.sum(sdev**2))
        assert_allclose(subject.cumulative, np.cumsum(sdev**2) / np.sum(sdev**2))
        assert pca.is_singular() == {"Subject": False}
        output = str(pca)
        for text in ("Random effect PCA", "Subject", "PC1", "PC2"):
            assert text in output

    def test_single_terms_report_the_random_effect_sd(self, sleepstudy_lmm, cbpp_glmm) -> None:
        assert_allclose(
            sleepstudy_lmm.rePCA()["Subject"].sdev, sleepstudy_lmm.sigma * sleepstudy_lmm.theta
        )
        assert_allclose(cbpp_glmm.rePCA()["herd"].sdev, cbpp_glmm.theta)
        assert cbpp_glmm.rePCA()["herd"].n_terms == 1

    def test_boundary_fits_are_singular(self, singular_cbpp_glmm) -> None:
        assert singular_cbpp_glmm.rePCA().is_singular() == {"herd": True}


@pytest.fixture(scope="module")
def four_level_data():
    rng = np.random.default_rng(42)
    group = np.repeat(np.array(list("ABCD")), 30)
    subject = np.repeat(np.arange(12), 10)
    y = np.repeat([0.0, 1.0, 4.0, 9.0], 30) + rng.normal(0.0, 1.0, 12)[subject]
    y += rng.standard_normal(120)
    return pd.DataFrame({"y": y, "group": group, "subject": subject.astype(str)})


def level_codes(result, data, factor, levels):
    """Return the fixed-effect columns for the first row of each factor level."""
    X = result.getME("X")
    return np.array([X[np.flatnonzero(data[factor] == level)[0], 1:] for level in levels])


class TestContrasts:
    @pytest.mark.parametrize(
        ("contrast", "codes"),
        [
            (None, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ("sum", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]),
            # R's contr.helmert with unit-length columns.
            (
                "helmert",
                np.array([[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]])
                / np.sqrt([2.0, 6.0, 12.0]),
            ),
            # R's contr.poly(4): orthonormal linear, quadratic and cubic trends.
            (
                "poly",
                np.column_stack(
                    (
                        np.array([-3, -1, 1, 3]) / np.sqrt(20),
                        np.array([1, -1, -1, 1]) / 2,
                        np.array([-1, 3, -3, 1]) / np.sqrt(20),
                    )
                ),
            ),
        ],
        ids=["treatment", "sum", "helmert", "poly"],
    )
    def test_codings_match_r_and_leave_fitted_values_unchanged(
        self, four_level_data, contrast, codes
    ) -> None:
        contrasts = None if contrast is None else {"group": contrast}

        result = lmer("y ~ group + (1 | subject)", four_level_data, contrasts=contrasts)

        treatment = lmer("y ~ group + (1 | subject)", four_level_data)
        assert result.matrices.fixed_names == ["(Intercept)", "group.1", "group.2", "group.3"]
        assert_allclose(level_codes(result, four_level_data, "group", "ABCD"), codes, atol=1e-12)
        assert_allclose(result.fitted(), treatment.fitted(), atol=1e-6)

    def test_custom_contrast_matrix(self, four_level_data) -> None:
        data = four_level_data[four_level_data["group"] != "D"]
        custom = np.array([[-1, -1], [1, 0], [0, 1]], dtype=np.float64)

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": custom})

        assert result.matrices.fixed_names == ["(Intercept)", "group.1", "group.2"]
        assert_allclose(level_codes(result, data, "group", "ABC"), custom)

    def test_glmer_sum_contrasts(self) -> None:
        rng = np.random.default_rng(42)
        group = np.repeat(np.array(list("ABC")), 40)
        subject = np.repeat(np.arange(12), 10)
        eta = np.repeat([-1.0, 0.0, 1.0], 40) + rng.normal(0.0, 1.0, 12)[subject]
        data = pd.DataFrame(
            {
                "y": rng.binomial(1, expit(eta)).astype(float),
                "group": group,
                "subject": subject.astype(str),
            }
        )

        result = glmer(
            "y ~ group + (1 | subject)",
            data,
            family=families.Binomial(),
            contrasts={"group": "sum"},
        )

        assert result.matrices.fixed_names == ["(Intercept)", "group.1", "group.2"]
        assert_allclose(level_codes(result, data, "group", "ABC"), [[1, 0], [0, 1], [-1, -1]])

    def test_contrast_helpers(self) -> None:
        from mixedlm.utils.contrasts import contr_sum, contr_treatment

        assert_array_equal(contr_treatment(3), [[0, 0], [1, 0], [0, 1]])
        assert_array_equal(contr_sum(3), [[1, 0], [0, 1], [-1, -1]])

    def test_interactions_multiply_the_main_effect_codes(self) -> None:
        rng = np.random.default_rng(42)
        group1 = np.tile(np.repeat(["A", "B"], 60), 2)
        group2 = np.repeat(["X", "Y"], 120)
        subject = np.repeat(np.arange(24), 10)
        y = rng.normal(0.0, 1.0, 24)[subject] + rng.standard_normal(240)
        data = pd.DataFrame({"y": y, "g1": group1, "g2": group2, "subject": subject.astype(str)})

        result = lmer(
            "y ~ g1 * g2 + (1 | subject)",
            data,
            contrasts={"g1": "sum", "g2": "treatment"},
        )

        X = result.getME("X")
        g1_code = np.where(group1 == "A", 1.0, -1.0)
        g2_code = (group2 == "Y").astype(float)
        assert_allclose(X, np.column_stack((np.ones(240), g1_code, g2_code, g1_code * g2_code)))

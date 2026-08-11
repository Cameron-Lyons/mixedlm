"""Golden statistical checks for canonical mixed-model datasets."""

from __future__ import annotations

import mixedlm as mlm
import pytest
from mixedlm import families, pvalues
from numpy.testing import assert_allclose

from tests._lmer_data import CBPP

_BOUNDARY_FLOAT_ATOL = 1e-8
_PENICILLIN_FLOAT_ATOL = 2e-5
_CBPP_FLOAT_ATOL = 1e-6


@pytest.fixture(scope="class")
def model() -> mlm.LmerResult:
    data = mlm.load_sleepstudy()
    return mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=True)


@pytest.mark.filterwarnings("ignore:Model is singular")
class TestSleepstudyGolden:
    def test_likelihood_and_variance_components(self, model: mlm.LmerResult) -> None:
        assert model.converged
        assert_allclose(model.beta, [251.19101636363638, 10.529083771043759], rtol=0, atol=1e-10)
        assert_allclose(model.theta, 0.0, rtol=0, atol=_BOUNDARY_FLOAT_ATOL)
        assert model.sigma == pytest.approx(47.83369729516479, abs=1e-10)
        assert model.deviance == pytest.approx(1894.5502512670237, abs=1e-10)

        loglik = model.logLik()
        assert loglik.value == pytest.approx(-1110.8461845439435, abs=1e-10)
        assert loglik.df == 6
        assert loglik.nobs == 180
        assert model.AIC() == pytest.approx(2233.692369087887, abs=1e-10)
        assert model.BIC() == pytest.approx(2252.8501101932284, abs=1e-10)

        varcorr = model.VarCorr()
        subject = varcorr.groups["Subject"]
        assert subject.variance["(Intercept)"] == pytest.approx(0.0, abs=1e-12)
        assert subject.variance["Days"] == pytest.approx(0.0, abs=1e-12)
        assert varcorr.residual == pytest.approx(2288.062596925455, abs=1e-9)

    def test_vcov_residuals_fitted_and_pvalues(self, model: mlm.LmerResult) -> None:
        assert_allclose(
            model.vcov(),
            [
                [43.912312466246, -6.933523020986],
                [-6.933523020986, 1.540782893552],
            ],
            rtol=0,
            atol=1e-12,
        )
        assert_allclose(
            model.fitted()[:8],
            [
                251.191016363636,
                261.72010013468,
                272.249183905724,
                282.778267676768,
                293.307351447811,
                303.836435218855,
                314.365518989899,
                324.894602760943,
            ],
            rtol=0,
            atol=1e-12,
        )
        assert_allclose(
            model.residuals()[:8],
            [
                -1.631016363636,
                -3.01540013468,
                -21.448583905724,
                38.661532323232,
                63.544548552189,
                110.853664781145,
                67.838281010101,
                -34.746002760943,
            ],
            rtol=0,
            atol=1e-12,
        )

        normal = pvalues(model, method="normal")
        satterthwaite = pvalues(model, method="Satterthwaite")
        kenward_roger = pvalues(model, method="Kenward-Roger")

        assert normal["(Intercept)"] == 0.0
        assert normal["Days"] == pytest.approx(2.2055461574803517e-17, rel=1e-12)
        assert satterthwaite["(Intercept)"] == pytest.approx(3.6641075162632043e-87, rel=1e-12)
        assert satterthwaite["Days"] == pytest.approx(8.29796477881002e-15, rel=1e-12)
        assert kenward_roger["(Intercept)"] == pytest.approx(0.0011866042229741927, rel=2e-5)
        assert kenward_roger["Days"] == pytest.approx(0.017879205051023632, rel=2e-5)


@pytest.mark.filterwarnings("ignore:Model is singular")
def test_penicillin_crossed_random_effects_golden() -> None:
    data = mlm.load_penicillin()
    model = mlm.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data, REML=True)

    assert model.converged
    assert_allclose(model.beta, [22.81944444444444], rtol=0, atol=1e-12)
    assert model.theta[0] == pytest.approx(0.0, abs=_BOUNDARY_FLOAT_ATOL)
    assert model.theta[1] == pytest.approx(0.5594653125073, abs=_PENICILLIN_FLOAT_ATOL)
    assert model.sigma == pytest.approx(2.6562451541083103, abs=_PENICILLIN_FLOAT_ATOL)
    assert model.deviance == pytest.approx(702.922932400389, abs=_PENICILLIN_FLOAT_ATOL)

    loglik = model.logLik()
    assert loglik.value == pytest.approx(-482.86967644846266, abs=_PENICILLIN_FLOAT_ATOL)
    assert loglik.df == 4
    assert loglik.nobs == 144
    assert_allclose(model.vcov(), [[0.417068309149]], rtol=0, atol=_PENICILLIN_FLOAT_ATOL)
    assert_allclose(
        model.residuals()[:8],
        [
            1.557512161916,
            0.278613252505,
            1.587117979977,
            1.528848888722,
            1.08758925241,
            0.043651797752,
            5.557512161896,
            4.278613252485,
        ],
        rtol=0,
        atol=_PENICILLIN_FLOAT_ATOL,
    )

    varcorr = model.VarCorr()
    assert varcorr.groups["plate"].variance["(Intercept)"] == pytest.approx(
        0.0, abs=_BOUNDARY_FLOAT_ATOL
    )
    assert varcorr.groups["sample"].variance["(Intercept)"] == pytest.approx(
        2.208424924943698, abs=_PENICILLIN_FLOAT_ATOL
    )
    assert varcorr.residual == pytest.approx(7.055636597373891, abs=_PENICILLIN_FLOAT_ATOL)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply")
def test_cbpp_binomial_glmer_golden() -> None:
    data = CBPP.copy()
    model = mlm.glmer(
        "y ~ period + (1 | herd)",
        data,
        family=families.Binomial(),
        weights=data["size"].to_numpy(dtype=float),
    )

    assert model.converged
    assert_allclose(
        model.beta,
        [-1.861164732564236, -0.208553975355595, -0.078281733042295, -0.618291994021517],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(model.theta, [0.48750557774242], rtol=0, atol=_CBPP_FLOAT_ATOL)
    assert model.sigma == pytest.approx(1.0, abs=0.0)
    assert model.deviance == pytest.approx(74.03136198466316, abs=_CBPP_FLOAT_ATOL)

    loglik = model.logLik()
    assert loglik.value == pytest.approx(-37.01568099233158, abs=_CBPP_FLOAT_ATOL)
    assert loglik.df == 5
    assert loglik.nobs == 56
    assert model.AIC() == pytest.approx(84.03136198466316, abs=_CBPP_FLOAT_ATOL)
    assert model.BIC() == pytest.approx(94.15812043833891, abs=_CBPP_FLOAT_ATOL)
    assert_allclose(
        model.vcov(),
        [
            [0.086506090158, -0.065213683451, -0.067123707411, -0.067040141073],
            [-0.065213683451, 0.160280508337, 0.065151552713, 0.064818269431],
            [-0.067123707411, 0.065151552713, 0.144882987863, 0.067106011166],
            [-0.067040141073, 0.064818269431, 0.067106011166, 0.189605383319],
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(
        model.fitted()[:8],
        [
            0.198309632363,
            0.167221722142,
            0.186157376781,
            0.117617807046,
            0.131593373719,
            0.109535223114,
            0.122902679701,
            0.075491973286,
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(
        model.residuals()[:8],
        [
            -0.541510359941,
            0.726873965828,
            1.773098171001,
            -1.118615177302,
            0.065853413303,
            -0.802052054265,
            0.272457372704,
            0.265837027814,
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
    )

    herd = model.VarCorr().groups["herd"]
    assert herd.variance["(Intercept)"] == pytest.approx(0.23766168832997042, abs=_CBPP_FLOAT_ATOL)
    assert herd.stddev["(Intercept)"] == pytest.approx(0.4875055777424197, abs=_CBPP_FLOAT_ATOL)

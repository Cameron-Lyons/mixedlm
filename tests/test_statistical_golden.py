"""Golden statistical checks for canonical mixed-model datasets."""

from __future__ import annotations

import mixedlm as mlm
import pytest
from mixedlm import families, pvalues
from numpy.testing import assert_allclose

from tests._lmer_data import CBPP

# These fixtures use canonical lme4 benchmark datasets and lock the current
# numerical contract across outputs that users commonly compare between
# mixed-model implementations. The sleepstudy and penicillin LMM variance
# components currently land on singular fits; if optimizer parity changes, these
# values should be updated deliberately with external lme4/statsmodels evidence.


@pytest.mark.filterwarnings("ignore:Model is singular")
class TestSleepstudyGolden:
    @pytest.fixture(scope="class")
    def model(self) -> mlm.LmerResult:
        data = mlm.load_sleepstudy()
        return mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=True)

    def test_likelihood_and_variance_components(self, model: mlm.LmerResult) -> None:
        assert model.converged
        assert_allclose(model.beta, [251.19101636363638, 10.529083771043759], rtol=0, atol=1e-10)
        assert_allclose(model.theta, [0.0, -1.247341918772e-09, 0.0], rtol=0, atol=1e-12)
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
        assert subject.variance["(Intercept)"] == pytest.approx(0.0, abs=1e-14)
        assert subject.variance["Days"] == pytest.approx(3.55990933317144e-15, abs=1e-18)
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
        assert normal == {"(Intercept)": 0.0, "Days": 0.0}
        assert satterthwaite["(Intercept)"] == pytest.approx(0.0, abs=0.0)
        assert satterthwaite["Days"] == pytest.approx(1.354472090042691e-14, rel=1e-12)


@pytest.mark.filterwarnings("ignore:Model is singular")
def test_penicillin_crossed_random_effects_golden() -> None:
    data = mlm.load_penicillin()
    model = mlm.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data, REML=True)

    assert model.converged
    assert_allclose(model.beta, [22.81944444444444], rtol=0, atol=1e-12)
    assert_allclose(model.theta, [1.187215173162e-06, 0.5594653125073], rtol=0, atol=2e-11)
    assert model.sigma == pytest.approx(2.6562451541083103, abs=3e-12)
    assert model.deviance == pytest.approx(702.922932400389, abs=1e-12)

    loglik = model.logLik()
    assert loglik.value == pytest.approx(-482.86967644846266, abs=1e-12)
    assert loglik.df == 4
    assert loglik.nobs == 144
    assert_allclose(model.vcov(), [[0.417068309149]], rtol=0, atol=1e-12)
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
        atol=1e-12,
    )

    varcorr = model.VarCorr()
    assert varcorr.groups["plate"].variance["(Intercept)"] == pytest.approx(9.944780161800345e-12)
    assert varcorr.groups["sample"].variance["(Intercept)"] == pytest.approx(2.208424924943698)
    assert varcorr.residual == pytest.approx(7.055636597373891)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply")
def test_cbpp_binomial_glmer_golden() -> None:
    data = CBPP.copy()
    model = mlm.glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

    assert model.converged
    assert_allclose(
        model.beta,
        [-1.7557176513849455, -0.4488534827602638, -0.1186586925007642, -0.6261372107626156],
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(model.theta, [0.6372400408998262], rtol=0, atol=1e-12)
    assert model.sigma == pytest.approx(1.0, abs=0.0)
    assert model.deviance == pytest.approx(11.189698024186672, abs=1e-12)

    loglik = model.logLik()
    assert loglik.value == pytest.approx(-5.594849012093336, abs=1e-12)
    assert loglik.df == 5
    assert loglik.nobs == 56
    assert model.AIC() == pytest.approx(21.189698024186672, abs=1e-12)
    assert model.BIC() == pytest.approx(31.31645647786242, abs=1e-12)
    assert_allclose(
        model.vcov(),
        [
            [0.743029787332, -0.561571672073, -0.562166624516, -0.561302529527],
            [-0.561571672073, 1.341075534935, 0.562461023519, 0.562638393569],
            [-0.562166624516, 0.562461023519, 1.172570250194, 0.562477114363],
            [-0.561302529527, 0.562638393569, 0.562477114363, 1.458009117359],
        ],
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(
        model.fitted()[:8],
        [
            0.208594813036,
            0.144023075002,
            0.189682411694,
            0.123515364543,
            0.1414187027,
            0.095141799419,
            0.127614762741,
            0.080936258204,
        ],
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(
        model.residuals()[:8],
        [
            -0.169215458298,
            0.278367338822,
            0.580920231827,
            -0.513490222305,
            -0.014580538283,
            -0.145270485489,
            0.044944384645,
            0.035914772383,
        ],
        rtol=0,
        atol=1e-12,
    )

    herd = model.VarCorr().groups["herd"]
    assert herd.variance["(Intercept)"] == pytest.approx(0.4060748697260122, abs=1e-12)
    assert herd.stddev["(Intercept)"] == pytest.approx(0.6372400408998262, abs=1e-12)

"""Golden statistical checks for canonical mixed-model datasets."""

from __future__ import annotations

import mixedlm as mlm
import pytest
from mixedlm import families, pvalues
from numpy.testing import assert_allclose

from tests._lmer_data import CBPP

_LMM_FLOAT_ATOL = 2e-5
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
        assert_allclose(
            model.theta,
            [0.9467987315840892, 0.018801466252854657, 0.22768771787157],
            rtol=0,
            atol=_LMM_FLOAT_ATOL,
        )
        assert model.sigma == pytest.approx(25.828313444164642, abs=_LMM_FLOAT_ATOL)
        assert model.deviance == pytest.approx(1746.069424187031, abs=_LMM_FLOAT_ATOL)

        loglik = model.logLik()
        assert loglik.value == pytest.approx(-873.0347120935155, abs=_LMM_FLOAT_ATOL)
        assert loglik.df == 6
        assert loglik.nobs == 180
        assert model.AIC() == pytest.approx(1758.069424187031, abs=_LMM_FLOAT_ATOL)
        assert model.BIC() == pytest.approx(1777.2271652923723, abs=_LMM_FLOAT_ATOL)

        varcorr = model.VarCorr()
        subject = varcorr.groups["Subject"]
        assert subject.variance["(Intercept)"] == pytest.approx(
            598.0086023071216, abs=_LMM_FLOAT_ATOL
        )
        assert subject.variance["Days"] == pytest.approx(34.81950525086068, abs=_LMM_FLOAT_ATOL)
        assert varcorr.residual == pytest.approx(667.1017683274486, abs=_LMM_FLOAT_ATOL)

    def test_vcov_residuals_fitted_and_pvalues(self, model: mlm.LmerResult) -> None:
        assert_allclose(
            model.vcov(),
            [
                [46.025663493861, -1.361786361477],
                [-1.361786361477, 2.383643743142],
            ],
            rtol=0,
            atol=_LMM_FLOAT_ATOL,
        )
        assert_allclose(
            model.fitted()[:8],
            [
                253.870523528942,
                273.495137452821,
                293.1197513767,
                312.744365300579,
                332.368979224458,
                351.993593148337,
                371.618207072216,
                391.242820996095,
            ],
            rtol=0,
            atol=_LMM_FLOAT_ATOL,
        )
        assert_allclose(
            model.residuals()[:8],
            [
                -4.310523528942,
                -14.790437452821,
                -42.3191513767,
                8.695434699421,
                24.482920775542,
                62.696506851663,
                10.585592927784,
                -101.094220996095,
            ],
            rtol=0,
            atol=_LMM_FLOAT_ATOL,
        )

        normal = pvalues(model, method="normal")
        satterthwaite = pvalues(model, method="Satterthwaite")
        assert normal["(Intercept)"] == pytest.approx(0.0, abs=0.0)
        assert normal["Days"] == pytest.approx(9.118483745851336e-12, rel=1e-10)
        assert satterthwaite["(Intercept)"] == pytest.approx(0.0, abs=0.0)
        assert satterthwaite["Days"] == pytest.approx(1.744924205127063e-10, rel=1e-10)


@pytest.mark.filterwarnings("ignore:Model is singular")
def test_penicillin_crossed_random_effects_golden() -> None:
    data = mlm.load_penicillin()
    model = mlm.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data, REML=True)

    assert model.converged
    assert_allclose(model.beta, [22.81944444444444], rtol=0, atol=2e-12)
    assert_allclose(
        model.theta,
        [1.720313360433691, 2.12067267084168],
        rtol=0,
        atol=_PENICILLIN_FLOAT_ATOL,
    )
    assert model.sigma == pytest.approx(0.9376710791427862, abs=_PENICILLIN_FLOAT_ATOL)
    assert model.deviance == pytest.approx(483.25999628008196, abs=_PENICILLIN_FLOAT_ATOL)

    loglik = model.logLik()
    assert loglik.value == pytest.approx(-241.62999814004098, abs=_PENICILLIN_FLOAT_ATOL)
    assert loglik.df == 4
    assert loglik.nobs == 144
    assert_allclose(model.vcov(), [[0.773542313559]], rtol=0, atol=_PENICILLIN_FLOAT_ATOL)
    assert_allclose(
        model.residuals()[:8],
        [
            0.275783317045,
            -0.669187933892,
            0.431740141015,
            0.734473923786,
            0.239063856371,
            -0.687547664229,
            2.066848859365,
            1.121877608428,
        ],
        rtol=0,
        atol=_PENICILLIN_FLOAT_ATOL,
    )

    varcorr = model.VarCorr()
    assert varcorr.groups["plate"].variance["(Intercept)"] == pytest.approx(
        2.6020531704258327, abs=_PENICILLIN_FLOAT_ATOL
    )
    assert varcorr.groups["sample"].variance["(Intercept)"] == pytest.approx(
        3.954106128219207, abs=_PENICILLIN_FLOAT_ATOL
    )
    assert varcorr.residual == pytest.approx(0.8792270612569914, abs=_PENICILLIN_FLOAT_ATOL)


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
        atol=_CBPP_FLOAT_ATOL,
    )
    assert_allclose(model.theta, [0.6372400408998262], rtol=0, atol=_CBPP_FLOAT_ATOL)
    assert model.sigma == pytest.approx(1.0, abs=0.0)
    assert model.deviance == pytest.approx(11.189698024186672, abs=_CBPP_FLOAT_ATOL)

    loglik = model.logLik()
    assert loglik.value == pytest.approx(-5.594849012093336, abs=_CBPP_FLOAT_ATOL)
    assert loglik.df == 5
    assert loglik.nobs == 56
    assert model.AIC() == pytest.approx(21.189698024186672, abs=_CBPP_FLOAT_ATOL)
    assert model.BIC() == pytest.approx(31.31645647786242, abs=_CBPP_FLOAT_ATOL)
    assert_allclose(
        model.vcov(),
        [
            [0.743029787332, -0.561571672073, -0.562166624516, -0.561302529527],
            [-0.561571672073, 1.341075534935, 0.562461023519, 0.562638393569],
            [-0.562166624516, 0.562461023519, 1.172570250194, 0.562477114363],
            [-0.561302529527, 0.562638393569, 0.562477114363, 1.458009117359],
        ],
        rtol=0,
        atol=_CBPP_FLOAT_ATOL,
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
        atol=_CBPP_FLOAT_ATOL,
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
        atol=_CBPP_FLOAT_ATOL,
    )

    herd = model.VarCorr().groups["herd"]
    assert herd.variance["(Intercept)"] == pytest.approx(0.4060748697260122, abs=_CBPP_FLOAT_ATOL)
    assert herd.stddev["(Intercept)"] == pytest.approx(0.6372400408998262, abs=_CBPP_FLOAT_ATOL)

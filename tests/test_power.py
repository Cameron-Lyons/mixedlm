from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mixedlm import lmer
from mixedlm.power import (
    PowerCurveResult,
    PowerResult,
    extend,
    powerCurve,
    powerSim,
)


def create_power_data(
    n_groups: int = 20, n_per_group: int = 5, effect: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    data_rows = []
    for g in range(n_groups):
        group_effect = np.random.randn() * 0.5
        for _ in range(n_per_group):
            x = np.random.randn()
            y = 1.0 + effect * x + group_effect + np.random.randn() * 0.5
            data_rows.append({"group": f"G{g + 1}", "x": x, "y": y})
    return pd.DataFrame(data_rows)


POWER_DATA = create_power_data()


class TestPowerResult:
    def test_power_result_str(self) -> None:
        result = PowerResult(
            power=0.8,
            ci_lower=0.75,
            ci_upper=0.85,
            n_successes=80,
            n_simulations=100,
            effect_size=0.5,
            n_obs=100,
            n_groups=20,
        )
        s = str(result)
        assert "Power: 0.800" in s
        assert "0.750" in s
        assert "0.850" in s
        assert "Simulations: 100" in s
        assert "80 significant" in s
        assert "Observations: 100" in s
        assert "Groups: 20" in s
        assert "Effect size: 0.5000" in s

    def test_power_result_str_no_groups(self) -> None:
        result = PowerResult(
            power=0.8,
            ci_lower=0.75,
            ci_upper=0.85,
            n_successes=80,
            n_simulations=100,
            effect_size=None,
            n_obs=100,
            n_groups=None,
        )
        s = str(result)
        assert "Groups" not in s
        assert "Effect size" not in s


class TestPowerCurveResult:
    def test_power_curve_result_str(self) -> None:
        result = PowerCurveResult(
            values=[10, 20, 30],
            powers=[0.5, 0.7, 0.9],
            ci_lowers=[0.45, 0.65, 0.85],
            ci_uppers=[0.55, 0.75, 0.95],
            along="n_groups",
            n_simulations=100,
        )
        s = str(result)
        assert "Power curve along 'n_groups'" in s
        assert "Value" in s
        assert "Power" in s
        assert "95% CI" in s

    def test_power_curve_result_plot(self) -> None:
        pytest.importorskip("matplotlib")
        result = PowerCurveResult(
            values=[10, 20, 30],
            powers=[0.5, 0.7, 0.9],
            ci_lowers=[0.45, 0.65, 0.85],
            ci_uppers=[0.55, 0.75, 0.95],
            along="n_groups",
            n_simulations=100,
        )
        fig = result.plot()
        assert fig is not None

    def test_power_curve_result_plot_no_ci(self) -> None:
        pytest.importorskip("matplotlib")
        result = PowerCurveResult(
            values=[10, 20, 30],
            powers=[0.5, 0.7, 0.9],
            ci_lowers=[0.45, 0.65, 0.85],
            ci_uppers=[0.55, 0.75, 0.95],
            along="n_groups",
            n_simulations=100,
        )
        fig = result.plot(show_ci=False)
        assert fig is not None

    def test_power_curve_result_plot_with_ax(self) -> None:
        plt = pytest.importorskip("matplotlib.pyplot")
        result = PowerCurveResult(
            values=[10, 20, 30],
            powers=[0.5, 0.7, 0.9],
            ci_lowers=[0.45, 0.65, 0.85],
            ci_uppers=[0.55, 0.75, 0.95],
            along="n_groups",
            n_simulations=100,
        )
        fig, ax = plt.subplots()
        returned_fig = result.plot(ax=ax)
        assert returned_fig is fig


class TestPowerSim:
    def test_powersim_basic(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerSim(model, test="x", nsim=20, seed=42)

        assert isinstance(result, PowerResult)
        assert 0.0 <= result.power <= 1.0
        assert result.ci_lower <= result.power <= result.ci_upper
        assert result.n_simulations <= 20
        assert result.n_successes <= result.n_simulations
        assert result.n_obs == len(POWER_DATA)

    def test_powersim_default_test(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerSim(model, nsim=10, seed=42)

        assert isinstance(result, PowerResult)

    def test_powersim_custom_test_function(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)

        def custom_test(m) -> bool:
            return m.beta[1] > 0.3

        result = powerSim(model, test=custom_test, nsim=10, seed=42)
        assert isinstance(result, PowerResult)

    def test_powersim_reproducible(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)

        result1 = powerSim(model, test="x", nsim=10, seed=123)
        result2 = powerSim(model, test="x", nsim=10, seed=123)

        assert result1.power == result2.power
        assert result1.n_successes == result2.n_successes

    def test_powersim_returns_effect_size(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerSim(model, test="x", nsim=10, seed=42)

        assert result.effect_size is not None
        assert np.isclose(result.effect_size, model.beta[1])

    def test_powersim_returns_n_groups(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerSim(model, test="x", nsim=10, seed=42)

        assert result.n_groups == 20

    def test_powersim_invalid_param_returns_nan(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)

        result = powerSim(model, test="nonexistent", nsim=10, seed=42)
        assert np.isnan(result.power)
        assert result.n_successes == 0


class TestExtend:
    def test_extend_groups(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        extended = extend(model, along="group", n=30)

        assert len(extended["group"].unique()) == 30
        assert len(extended) > len(POWER_DATA)

    def test_extend_groups_no_change_if_fewer(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        extended = extend(model, along="group", n=10)

        assert len(extended) == len(POWER_DATA)

    def test_extend_within(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        extended = extend(model, along="within", n=20)

        assert len(extended) > len(POWER_DATA)

    def test_extend_within_no_change_if_fewer(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        extended = extend(model, along="within", n=3)

        assert len(extended) == len(POWER_DATA)

    def test_extend_invalid_along_raises(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)

        with pytest.raises(ValueError, match="Unknown 'along' value"):
            extend(model, along="invalid", n=30)

    def test_extend_with_custom_data(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        custom_data = POWER_DATA.copy()
        extended = extend(model, along="group", n=30, data=custom_data)

        assert len(extended["group"].unique()) == 30


class TestPowerCurve:
    def test_powercurve_n_groups(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerCurve(
            model, test="x", along="n_groups", values=[10, 20], nsim=10, seed=42
        )

        assert isinstance(result, PowerCurveResult)
        assert result.values == [10, 20]
        assert len(result.powers) == 2
        assert len(result.ci_lowers) == 2
        assert len(result.ci_uppers) == 2
        assert result.along == "n_groups"
        assert result.n_simulations == 10

    def test_powercurve_effect_size(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerCurve(
            model, test="x", along="effect_size", values=[0.5, 1.0], nsim=10, seed=42
        )

        assert isinstance(result, PowerCurveResult)
        assert result.values == [0.5, 1.0]
        assert result.along == "effect_size"

    def test_powercurve_default_values_n_groups(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerCurve(model, test="x", along="n_groups", nsim=5, seed=42)

        assert isinstance(result, PowerCurveResult)
        assert len(result.values) > 0

    def test_powercurve_default_values_effect_size(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)
        result = powerCurve(model, test="x", along="effect_size", nsim=5, seed=42)

        assert isinstance(result, PowerCurveResult)
        assert len(result.values) > 0

    def test_powercurve_reproducible(self) -> None:
        model = lmer("y ~ x + (1|group)", POWER_DATA)

        result1 = powerCurve(
            model, test="x", along="effect_size", values=[0.5, 1.0], nsim=5, seed=42
        )
        result2 = powerCurve(
            model, test="x", along="effect_size", values=[0.5, 1.0], nsim=5, seed=42
        )

        assert result1.powers == result2.powers

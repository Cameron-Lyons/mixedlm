"""Tests for polars DataFrame support."""

import mixedlm as mlm
import numpy as np
import pytest

pl = pytest.importorskip("polars")


@pytest.fixture
def sleepstudy_polars():
    """Load sleepstudy as polars DataFrame."""
    pandas_df = mlm.load_sleepstudy()
    return pl.from_pandas(pandas_df)


@pytest.fixture
def cbpp_polars():
    """Load cbpp as polars DataFrame."""
    pandas_df = mlm.load_cbpp()
    return pl.from_pandas(pandas_df)


class TestPolarsLmer:
    def test_lmer_basic(self, sleepstudy_polars):
        """Test basic lmer with polars DataFrame."""
        result = mlm.lmer("Reaction ~ Days + (1 | Subject)", sleepstudy_polars)

        assert result.converged
        assert len(result.fixef()) == 2
        assert "(Intercept)" in result.fixef()
        assert "Days" in result.fixef()

    def test_lmer_random_slope(self, sleepstudy_polars):
        """Test lmer with random slope using polars."""
        result = mlm.lmer("Reaction ~ Days + (Days | Subject)", sleepstudy_polars)

        assert result.converged
        ranef = result.ranef()
        assert "Subject" in ranef
        ranef_group = ranef["Subject"]
        if hasattr(ranef_group, "columns"):
            assert "(Intercept)" in ranef_group.columns
            assert "Days" in ranef_group.columns
        else:
            assert "(Intercept)" in ranef_group
            assert "Days" in ranef_group

    def test_lmer_uncorrelated(self, sleepstudy_polars):
        """Test lmer with uncorrelated random effects."""
        result = mlm.lmer("Reaction ~ Days + (Days || Subject)", sleepstudy_polars)

        assert result.converged
        assert len(result.theta) == 2

    def test_lmer_predictions(self, sleepstudy_polars):
        """Test predictions work with polars."""
        result = mlm.lmer("Reaction ~ Days + (1 | Subject)", sleepstudy_polars)

        fitted = result.fitted()
        assert len(fitted) == len(sleepstudy_polars)

        residuals = result.residuals()
        assert len(residuals) == len(sleepstudy_polars)

    def test_response_free_categorical_subset_prediction(self):
        data = pl.DataFrame(
            {
                "y": np.arange(12.0),
                "treatment": ["A", "B"] * 6,
                "group": np.repeat([f"G{i}" for i in range(6)], 2),
            }
        )
        result = mlm.lmer("y ~ treatment + (1 | group)", data)
        new_data = pl.DataFrame({"treatment": ["B", "B"]})

        predicted = result.predict(new_data, re_form="NA")

        assert predicted.shape == (2,)
        assert np.all(np.isfinite(predicted))

    def test_lmer_summary(self, sleepstudy_polars):
        """Test summary output with polars."""
        result = mlm.lmer("Reaction ~ Days + (1 | Subject)", sleepstudy_polars)

        summary = result.summary()
        assert "Linear mixed model" in summary
        assert "(Intercept)" in summary
        assert "Days" in summary


class TestPolarsGlmer:
    def test_glmer_binomial(self):
        """Test glmer with binomial family using polars."""
        np.random.seed(42)
        n = 60
        groups = np.repeat(np.arange(10), 6)
        x = np.random.randn(n)
        eta = -0.5 + 0.5 * x + np.random.randn(10)[groups] * 0.3
        prob = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, prob)

        data = pl.DataFrame({"y": y, "x": x, "group": groups.astype(str)})

        result = mlm.glmer("y ~ x + (1 | group)", data, family=mlm.families.Binomial())

        assert result.converged
        fixef = result.fixef()
        assert "(Intercept)" in fixef

    def test_glmer_poisson(self):
        """Test glmer with Poisson family using polars."""
        np.random.seed(42)
        n = 100
        groups = np.repeat(np.arange(10), 10)
        x = np.random.randn(n)
        eta = 0.5 + 0.3 * x + np.random.randn(10)[groups] * 0.5
        y = np.random.poisson(np.exp(eta))

        data = pl.DataFrame({"y": y, "x": x, "group": groups.astype(str)})

        result = mlm.glmer("y ~ x + (1 | group)", data, family=mlm.families.Poisson())

        assert result.converged
        assert len(result.fixef()) == 2


class TestPolarsDataTypes:
    def test_numeric_columns(self):
        """Test with various numeric dtypes."""
        data = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "B", "B", "C", "C"],
            }
        )

        result = mlm.lmer("y ~ x + (1 | group)", data)
        assert result.converged

    def test_categorical_column(self):
        """Test with polars Categorical dtype."""
        data = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": pl.Series(["A", "A", "B", "B", "C", "C"]).cast(pl.Categorical),
            }
        )

        result = mlm.lmer("y ~ x + (1 | group)", data)
        assert result.converged

    def test_string_grouping(self):
        """Test with string grouping variable."""
        data = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": ["A", "A", "B", "B", "C", "C"],
            }
        )

        result = mlm.lmer("y ~ x + (1 | group)", data)
        assert result.converged


class TestPolarsNAHandling:
    @pytest.fixture(params=[pl.Float32, pl.Float64], ids=["float32", "float64"])
    def nan_data(self, request):
        return pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x": pl.Series(
                    [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0],
                    dtype=request.param,
                ),
                "group": ["A", "A", "B", "B", "C", "C"],
            }
        )

    def test_na_omit(self):
        """Test NA handling with polars."""
        data = pl.DataFrame(
            {
                "y": [1.0, 2.0, None, 4.0, 5.0, 6.0],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": ["A", "A", "B", "B", "C", "C"],
            }
        )

        result = mlm.lmer("y ~ x + (1 | group)", data, na_action="omit")
        assert result.converged
        assert result.matrices.n_obs == 5

    def test_na_fail(self):
        """Test NA fail action with polars."""
        data = pl.DataFrame(
            {
                "y": [1.0, 2.0, None, 4.0, 5.0, 6.0],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "group": ["A", "A", "B", "B", "C", "C"],
            }
        )

        with pytest.raises(ValueError, match="Missing values"):
            mlm.lmer("y ~ x + (1 | group)", data, na_action="fail")

    def test_nan_omit(self, nan_data):
        """Floating-point NaN values are omitted like pandas missing values."""
        result = mlm.lmer("y ~ x + (1 | group)", nan_data, na_action="omit")

        assert result.converged
        assert result.matrices.n_obs == 5

    def test_nan_exclude(self, nan_data):
        """Excluded NaN rows are restored in observation-aligned outputs."""
        result = mlm.lmer("y ~ x + (1 | group)", nan_data, na_action="exclude")

        fitted = result.fitted()
        residuals = result.residuals()
        assert result.matrices.n_obs == 5
        assert len(fitted) == len(nan_data)
        assert len(residuals) == len(nan_data)
        assert np.isnan(fitted[2])
        assert np.isnan(residuals[2])

    def test_nan_fail(self, nan_data):
        """The fail action reports floating-point NaN values."""
        with pytest.raises(ValueError, match=r"Variables with NA: \['x'\]"):
            mlm.lmer("y ~ x + (1 | group)", nan_data, na_action="fail")


class TestPolarsEquivalence:
    def test_results_match_pandas(self, sleepstudy_polars):
        """Test that polars and pandas give equivalent results."""
        pandas_df = mlm.load_sleepstudy()

        result_polars = mlm.lmer("Reaction ~ Days + (1 | Subject)", sleepstudy_polars)
        result_pandas = mlm.lmer("Reaction ~ Days + (1 | Subject)", pandas_df)

        np.testing.assert_allclose(
            result_polars.fixef()["(Intercept)"],
            result_pandas.fixef()["(Intercept)"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result_polars.fixef()["Days"], result_pandas.fixef()["Days"], rtol=1e-5
        )
        np.testing.assert_allclose(result_polars.sigma, result_pandas.sigma, rtol=1e-5)

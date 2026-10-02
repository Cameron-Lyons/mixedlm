# Power Analysis

Simulation-based power analysis uses a fitted LMM or GLMM as the generating model.
The pilot coefficients and variance components define the alternative hypothesis.

## powerSim

```python
import mixedlm as mlm

model = mlm.lmer("Reaction ~ Days + (Days | Subject)", mlm.load_sleepstudy())
power = mlm.powerSim(model, test="Days", nsim=200, seed=42)
print(power)
```

`powerSim(model, test=None, nsim=1000, alpha=0.05, seed=None, verbose=False)`
simulates a new response, refits the model, and tests each valid fit.

- `test`: A coefficient name or a callable returning a Boolean significance decision.
  The default tests the first non-intercept coefficient, or the intercept in an
  intercept-only model. Named tests use a two-sided normal Wald test. A callable
  can specify another inferential procedure or test a model without fixed effects.
- `nsim`: Positive integer number of attempted simulations.
- `alpha`: Significance level strictly between zero and one for the named Wald test.
  Callable tests choose their own significance threshold.
- `seed`: Integer seed for reproducibility. NumPy's global random state is preserved.
- `verbose`: Print progress during long runs.

`PowerResult` exposes:

| Attribute | Meaning |
|-----------|---------|
| `power` | Significant tests divided by valid completed simulations |
| `ci_lower`, `ci_upper` | 95% Wilson score confidence limits for that proportion |
| `n_successes` | Number of significant tests |
| `n_simulations` | Number of valid completed simulations |
| `n_failed` | Number of excluded simulations |
| `effect_size` | Generating coefficient for a named test; `None` for a callable |
| `n_obs`, `n_groups` | Study size; group count for the first grouping factor |

Unconverged fits, invalid estimates, exceptions, and non-Boolean test results are
excluded and reported in a warning. If every simulation fails, power and its
confidence limits are `NaN`. A high failure rate makes the conditional-on-success
power estimate less representative of the intended study.

## powerCurve

```python
curve = mlm.powerCurve(
    model, test="Days", along="Subject", values=[10, 18, 24, 30], nsim=100, seed=42
)
print(curve)
curve.plot()  # requires the plots extra
```

`powerCurve(model, test=None, along="n_groups", values=None, nsim=500,
alpha=0.05, seed=None, verbose=False)` changes the generating study for each point.

| `along` | Values control |
|---------|----------------|
| `"n_groups"` | Group count for the first grouping factor |
| A grouping factor name | Group count for that factor |
| `"within"` | Observations per group for the first grouping factor |
| `"effect_size"` | Multipliers of the tested coefficient, or first non-intercept coefficient |
| A fixed coefficient name | Absolute values of that coefficient; by default tests that coefficient |

Sample sizes must be positive integers; effect values must be finite real numbers.
Smaller designs retain the first observed groups or rows within groups. Larger
designs cycle the pilot group or row templates. Group-size differences and the
other grouping factors in a crossed design are retained when varying one factor.
This deterministic template scheme does not generate new covariate distributions.

Each study keeps the pilot coefficients, covariance parameters, residual scale,
prior weights, offsets, contrast coding, and grouped-binomial trial counts.
Sample-size curves require a stored model frame. The input model remains unchanged. Extending a group that is also a fixed
categorical predictor requires coefficients for new levels; such unknown levels
are rejected rather than assigned arbitrary effects.

`PowerCurveResult` exposes `values`, `powers`, `ci_lowers`, `ci_uppers`, `along`,
and `n_simulations` (attempts per point). Its `results` tuple contains a
`PowerResult` for each point, including the actual observation count, the varied
factor's group count, and successful/failed simulation counts. `plot(ax=None,
show_ci=True)` returns a Matplotlib figure.

## extend

```python
extended = mlm.extend(model, along="Subject", n=30)
within = mlm.extend(model, along="within", n=15, data=extended)
```

`extend(model, along, n, data=None)` returns a pandas DataFrame with additional
groups or observations per group. Supply a grouping factor name or `"within"`.
The original model frame is used unless `data` is provided. Targets smaller than
the existing design leave its rows intact. Numeric and categorical group labels
are retained when adding groups. Polars categorical and Enum columns retain their
category order in the returned pandas frame, including the fitted binomial
response's success level.

`extend` only returns data. Use `powerCurve` to calculate power for the resized
study while retaining the pilot parameters.

## GLMM example

```python
cbpp = mlm.load_cbpp()
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)", cbpp, family=mlm.families.Binomial()
)
curve = mlm.powerCurve(
    model, test=model.matrices.fixed_names[1], along="herd", values=[10, 15, 20], nsim=100
)
```

Use `model.matrices.fixed_names` to inspect the encoded coefficient names before
choosing a named test. Grouped-binomial simulation draws success counts using
retained trial counts; refits use those same trials.

# Power Analysis

Power analysis uses a fitted pilot model to simulate new studies, refit them, and
count significant tests. It can estimate power at the current study size or show
how power changes with group counts, observations per group, and effect sizes.

## Fit a pilot model

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(model.fixef())
```

The fitted coefficients, covariance parameters, and residual scale define the
simulation alternative. The examples use small `nsim` values to run quickly;
use several hundred simulations or more for reported estimates. Pilot estimates
can be optimistic, so examine smaller plausible effects as well as the fitted
effect.

## Estimate current power

```python
power = mlm.powerSim(model, test="Days", nsim=100, seed=42)
print(f"Power: {power.power:.1%}")
print(f"95% CI: [{power.ci_lower:.1%}, {power.ci_upper:.1%}]")
print(f"Completed fits: {power.n_simulations}; failed: {power.n_failed}")
```

Named coefficient tests use a two-sided normal Wald test. To use a different
procedure, provide a callable that returns `True` when its test is significant.
For example, a joint linear-hypothesis test can select multiple coefficients:

```python
from mixedlm.inference import linear_hypothesis


def days_test(fitted):
    hypothesis = linear_hypothesis(fitted, {"Days": 1.0}, test="F")
    return bool(hypothesis.p_value < 0.05)

power = mlm.powerSim(model, test=days_test, nsim=50, seed=42)
```

For this LMM, the F test above uses residual denominator degrees of freedom.
The simulation confidence interval always has 95% coverage, independently of the
significance threshold chosen for each test.

## Vary the number of subjects

```python
curve = mlm.powerCurve(
    model,
    test="Days",
    along="Subject",
    values=[10, 15, 18, 24, 30],
    nsim=20,
    seed=42,
)
for size, result in zip(curve.values, curve.results):
    print(f"Subjects={size}: power={result.power:.1%}, failed={result.n_failed}")
curve.plot()
```

Every point changes the actual simulated design while retaining the pilot
parameters. Smaller studies select the first observed subjects; larger studies
cycle the pilot subjects' covariate templates under new group labels. This changes
the number of independent random-effect draws, rather than re-estimating the
alternative from repeated pilot responses.

In crossed designs, specify the factor to vary; the other factor labels remain
as supplied by the pilot templates.

## Vary observations per subject

```python
within = mlm.powerCurve(
    model, test="Days", along="within", values=[5, 10, 15, 20], nsim=20, seed=42
)
```

Each subject has exactly the requested number of observations. Smaller designs
retain each subject's first rows; larger designs cycle their rows. The existing
weights, offsets, and covariate values accompany those rows. If new measurement
times or covariate distributions are part of the proposed study, prepare an
appropriate pilot design instead of treating replicated rows as new covariates.

## Examine smaller effects

Use multipliers of the pilot effect:

```python
effects = mlm.powerCurve(
    model, test="Days", along="effect_size", values=[0.25, 0.5, 0.75, 1.0], nsim=20, seed=42
)
```

Or set absolute coefficient values:

```python
effects = mlm.powerCurve(model, along="Days", values=[2, 4, 6, 8, 10], nsim=20, seed=42)
```

When `along` names a coefficient and `test` is omitted, that coefficient is tested.
The original model remains unchanged.

## Inspect and extend data

```python
extended = mlm.extend(model, along="Subject", n=30)
print(extended.groupby("Subject", observed=True).size())
```

`extend` returns a pandas DataFrame and leaves the fitted model unchanged.
`powerCurve` handles study resizing and power calculation together, retaining the
pilot parameters. See the [API reference](../api/power.md) for result fields and
GLMM examples.

## Interpret uncertainty and failures

The confidence interval describes Monte Carlo uncertainty conditional on valid
refits. More simulations narrow it: use smaller runs while exploring designs and
increase `nsim` for a final estimate. Unconverged or invalid fits are excluded and
reported, so inspect `n_failed` at every curve point. If every fit fails, the power
estimate and interval are `NaN`.

The confidence interval does not include uncertainty in the pilot effect sizes or
variance components. Sensitivity curves help assess that uncertainty. Power need
not increase at every sampled point because both the simulation outcomes and the
pilot templates vary; compare the estimates together with their intervals.

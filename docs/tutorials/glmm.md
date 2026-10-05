# Generalized Linear Mixed Models

This tutorial covers generalized linear mixed models (GLMMs) for non-Gaussian outcomes like binary, count, and proportional data.

## When to Use GLMMs

Use GLMMs when:

- Your outcome is binary (yes/no), count, or proportional
- Data has a grouped/hierarchical structure
- You need both fixed effects and random effects

## Binary Outcomes

### Example: CBPP Data

The cbpp dataset contains counts of bovine pleuropneumonia cases in cattle herds:

```python
import mixedlm as mlm

cbpp = mlm.load_cbpp()
print(cbpp.head())
```

### Fitting a Binomial GLMM

```python
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial()
)
print(model.summary())
```

The formula syntax `incidence / size` specifies:

- `incidence`: number of successes
- `size`: number of trials

This is equivalent to R's `cbind(incidence, size - incidence)`.

### Link Functions

The binomial family uses logit link by default:

```python
# Logit link (default)
mlm.families.Binomial()

# Probit link
mlm.families.Binomial(link="probit")

# Complementary log-log
mlm.families.Binomial(link="cloglog")
```

### Interpreting Coefficients

Fixed effects are on the log-odds scale:

```python
# Log-odds coefficients
model.fixef()

# Convert to odds ratios
import numpy as np
odds_ratios = {k: np.exp(v) for k, v in model.fixef().items()}
```

## Count Outcomes

### Poisson GLMM

For count data, such as the number of ticks on red grouse chicks, grouped by brood:

```python
grouseticks = mlm.load_grouseticks()

pois_model = mlm.glmer(
    "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)",
    grouseticks,
    family=mlm.families.Poisson()
)
```

Coefficients are on the log scale. Exponentiate for rate ratios:

```python
rate_ratios = {k: np.exp(v) for k, v in pois_model.fixef().items()}
```

### Negative Binomial GLMM

For overdispersed count data:

```python
# With a known dispersion parameter
nb_model = mlm.glmer(
    "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)",
    grouseticks,
    family=mlm.families.NegativeBinomial(theta=2.0)
)
```

`mlm.glmer_nb(formula, data, theta=2.0)` is shorthand for the same fit. Unlike
lme4's `glmer.nb()`, it keeps `theta` fixed (default 1.0) rather than estimating it.

### Checking for Overdispersion

```python
# Weighted Pearson chi-squared check
dispersion = mlm.diagnostics.check_overdispersion(pois_model)
print(dispersion)

# Ratios above 1 indicate extra variation. If the result is significant,
# consider a negative-binomial model or a missing model component.
if dispersion.is_overdispersed:
    print("Consider a negative-binomial model")
```

The Pearson check is an approximation conditional on the fitted random effects.
It is most informative for Poisson and grouped-binomial models with moderate
expected counts. Simulation-based residual diagnostics are preferable for small
means, Bernoulli data, or complex variance structures.

### Checking for Excess Zeros

```python
zeros = mlm.diagnostics.check_zero_inflation(pois_model)
print(zeros)

if zeros.is_zero_inflated:
    print("The fitted count distribution underpredicts zeros")
```

The check compares the observed zero count with the sum of fitted zero
probabilities. It supports Poisson, negative-binomial, and binomial models; for
grouped binomial responses, integer model weights are interpreted as trial
counts.

## Estimation Methods

### Laplace Approximation

The default method (`nAGQ=1`) uses Laplace approximation:

```python
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial(),
    nAGQ=1  # default
)
```

This jointly optimizes fixed coefficients and covariance parameters while
approximating the random-effect integral. The approximation can be biased for
small cluster sizes or large random effects.

For a faster preliminary fit, use `nAGQ=0`. It estimates fixed and random effects
together with PIRLS while optimizing only covariance parameters externally,
reproducing the previous fitting algorithm. Its estimates may differ from the
joint Laplace optimum. The default fit uses this approximation to initialize
joint optimization; `GlmerControl(nAGQ0initStep=False)` skips that preliminary
covariance optimization.

### Adaptive Gauss-Hermite Quadrature

For more accurate estimates, use adaptive quadrature:

```python
agq_model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial(),
    nAGQ=10  # 10 quadrature points
)
```

!!! note
    AGQ requires one random-effect term with one coefficient per group, either a random intercept or a scalar random slope. Models with multiple terms or intercept/slope blocks support `nAGQ=0` and `nAGQ=1`.

### When to Use AGQ

- Small cluster sizes (< 5 observations per group)
- Large random effects variance
- Binary outcomes (more sensitive than counts)
- When accuracy is more important than speed

## Distribution Families

### Available Families

```python
from mixedlm import families

# Continuous (rarely used with glmer, use lmer instead)
families.Gaussian()

# Binary/binomial
families.Binomial()

# Counts
families.Poisson()
families.NegativeBinomial(theta=2.0)

# Positive continuous
families.Gamma()
families.InverseGaussian()
```

### Custom Families

Wrap a family to scale its variance by a dispersion factor, or subclass
`mlm.families.CustomFamily` for a new distribution (see the
[Families API](../api/families.md)):

```python
# Quasi-binomial for overdispersed proportions
quasi_binom = families.QuasiFamily(families.Binomial(), phi=1.5)
```

## Random Effects in GLMMs

### Random Intercepts

Most common for GLMMs:

```python
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)", cbpp, family=mlm.families.Binomial()
)
```

### Random Slopes

Random slopes in GLMMs can be difficult to estimate:

```py
# May have convergence issues
model = mlm.glmer(
    "y ~ time + (time | subject)",
    data,
    family=mlm.families.Binomial()
)
```

!!! warning
    Random slopes in GLMMs often cause convergence problems. Start with random intercepts and add complexity gradually.

### Uncorrelated Random Effects

If the full model doesn't converge:

```py
model = mlm.glmer(
    "y ~ time + (time || subject)",
    data,
    family=mlm.families.Binomial()
)
```

## Likelihood Confidence Intervals

Use likelihood profiles when the shape of the likelihood matters:

```python
intervals = model.confint(parm="period.1", method="profile")
profiles = model.profile(which="period.1", n_points=20)
profiles["period.1"].plot()
```

Profiling re-optimizes the other coefficients and covariance parameters and
can produce asymmetric intervals. It refines the joint likelihood optimum
before tracing each curve; a warning reports a material shift from the original
coefficient estimate, especially for a preliminary `nAGQ=0` fit. Default joint
fits usually retain their center within optimization tolerance. The fitted model remains unchanged. Profiling retains
`nAGQ`, weights, offsets, and PIRLS controls and requires converged solves.
`n_points` controls the curve resolution; confidence limits use root finding.
The default `model.confint()` continues to provide faster Wald intervals.

## Model Comparison

### Likelihood Ratio Tests

```python
# Nested models
m1 = mlm.glmer("incidence / size ~ 1 + (1 | herd)", cbpp, family=mlm.families.Binomial())
m2 = mlm.glmer(
    "incidence / size ~ period + (1 | herd)", cbpp, family=mlm.families.Binomial()
)

# Compare
mlm.anova(m1, m2)
```

### Single Term Deletions

```python
model.drop1(cbpp)
```

## Predictions

### On the Link Scale

```python
# Linear predictor (log-odds for binomial)
model.predict(type="link")
```

### On the Response Scale

```python
# Predicted probabilities (for binomial)
model.predict(type="response")
```

### Marginal vs Conditional

```python
# Conditional: includes random effects for known groups
model.predict(newdata=cbpp)

# Marginal: random effects set to zero
model.predict(newdata=cbpp, re_form="~0")
```

## Convergence Issues

GLMMs are more prone to convergence issues than LMMs.

### Strategies

1. **Start simple**: Random intercepts before random slopes
2. **Use uncorrelated random effects**: `||` instead of `|`
3. **Increase iterations**:
   ```py
   control = mlm.GlmerControl(maxiter=2000, pirls_maxiter=100)
   model = mlm.glmer(..., control=control)
   ```
   `maxiter` controls the outer optimizer; `pirls_maxiter` controls the inner
   solve. Inspect `model.pirls_converged` to distinguish an inner failure.
4. **Try different optimizers**:
   ```py
   model.allFit(data)
   ```
5. **Scale predictors**: Center and scale continuous variables
6. **EM-REML initialization**: Use EM-REML to find better starting values:
   ```py
   control = mlm.GlmerControl(em_init=True, em_maxiter=50)
   model = mlm.glmer(..., control=control)
   ```
   This runs a linear EM-REML algorithm to estimate starting theta values before the GLMM optimizer takes over. It is silently skipped for unsupported covariance types.

### Singular Fits

A singular fit often means:

- Random effects variance is near zero
- Too complex random effects structure for the data

Consider simplifying the model.

## Complete Example

```python
import mixedlm as mlm
import numpy as np

# Load data
cbpp = mlm.load_cbpp()

# Fit model with AGQ for accuracy
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial(),
    nAGQ=10
)

# Summary
print(model.summary())

# Odds ratios for fixed effects
fixef = model.fixef()
print("\nOdds ratios:")
for name, coef in fixef.items():
    if name != "(Intercept)":
        print(f"  {name}: {np.exp(coef):.3f}")

# Predicted probabilities
probs = model.predict(type="response")

# Confidence intervals
ci = model.confint()
print("\n95% CI:")
print(ci)

# Check convergence
conv = mlm.checkConv(model)
print(f"\nConverged: {conv.converged}")
```

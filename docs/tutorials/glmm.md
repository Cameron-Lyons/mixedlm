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

For count data:

```python
model = mlm.glmer(
    "count ~ treatment + (1 | subject)",
    data,
    family=mlm.families.Poisson()
)
```

Coefficients are on the log scale. Exponentiate for rate ratios:

```python
import numpy as np
rate_ratios = {k: np.exp(v) for k, v in model.fixef().items()}
```

### Negative Binomial GLMM

For overdispersed count data:

```python
# With known dispersion parameter
model = mlm.glmer(
    "count ~ treatment + (1 | subject)",
    data,
    family=mlm.families.NegativeBinomial(theta=2.0)
)

# Or use glmer.nb to estimate theta
model = mlm.glmer_nb(
    "count ~ treatment + (1 | subject)",
    data
)
```

### Checking for Overdispersion

```python
# Fit Poisson model
pois_model = mlm.glmer("count ~ x + (1 | g)", data, family=mlm.families.Poisson())

# Check residual deviance vs residual df
# If ratio >> 1, consider negative binomial
```

## Estimation Methods

### Laplace Approximation

The default method (`nAGQ=1`) uses Laplace approximation:

```python
model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial(),
    nAGQ=1  # default
)
```

This is fast but may be biased for small cluster sizes or large random effects.

### Adaptive Gauss-Hermite Quadrature

For more accurate estimates, use adaptive quadrature:

```python
model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial(),
    nAGQ=10  # 10 quadrature points
)
```

!!! note
    AGQ is only available for models with a single random effect (one grouping factor with random intercept only). For models with multiple random effects or random slopes, use `nAGQ=1`.

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

Create custom families for special cases:

```python
from mixedlm.families import CustomFamily

# Quasi-binomial for overdispersed proportions
quasi_binom = families.QuasiFamily(
    variance_func="mu*(1-mu)",
    link="logit"
)
```

## Random Effects in GLMMs

### Random Intercepts

Most common for GLMMs:

```python
model = mlm.glmer("y ~ x + (1 | group)", data, family=mlm.families.Binomial())
```

### Random Slopes

Random slopes in GLMMs can be difficult to estimate:

```python
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

```python
model = mlm.glmer(
    "y ~ time + (time || subject)",
    data,
    family=mlm.families.Binomial()
)
```

## Model Comparison

### Likelihood Ratio Tests

```python
# Nested models
m1 = mlm.glmer("y ~ x + (1 | g)", data, family=mlm.families.Binomial())
m2 = mlm.glmer("y ~ x + z + (1 | g)", data, family=mlm.families.Binomial())

# Compare
mlm.anova(m1, m2)
```

### Single Term Deletions

```python
model.drop1(data)
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
model.predict(newdata=data)

# Marginal: random effects set to zero
model.predict(newdata=data, re_form="~0")
```

## Convergence Issues

GLMMs are more prone to convergence issues than LMMs.

### Strategies

1. **Start simple**: Random intercepts before random slopes
2. **Use uncorrelated random effects**: `||` instead of `|`
3. **Increase iterations**:
   ```python
   control = mlm.GlmerControl(maxfun=50000)
   model = mlm.glmer(..., control=control)
   ```
4. **Try different optimizers**:
   ```python
   model.allFit(data)
   ```
5. **Scale predictors**: Center and scale continuous variables
6. **EM-REML initialization**: Use EM-REML to find better starting values:
   ```python
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
print(f"\nConverged: {conv.ok}")
```

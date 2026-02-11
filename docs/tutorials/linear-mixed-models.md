# Linear Mixed Models

This tutorial covers linear mixed models (LMMs) in depth, from simple random intercepts to complex nested and crossed designs.

## When to Use LMMs

Use linear mixed models when:

- Data has a hierarchical/grouped structure (students within schools, measurements within subjects)
- Observations within groups are correlated
- You want to estimate both population-level effects and group-specific deviations
- You have random sampling of groups from a larger population

## Random Intercept Model

The simplest mixed model allows intercepts to vary by group.

### Example: Sleep Study

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
print(data.head())
```

Fit a random intercept model:

```python
model = mlm.lmer("Reaction ~ Days + (1 | Subject)", data)
print(model.summary())
```

The formula `(1 | Subject)` specifies a random intercept for each Subject. The model is:

\[
y_{ij} = \beta_0 + \beta_1 \text{Days}_{ij} + u_j + \epsilon_{ij}
\]

where \(u_j \sim N(0, \sigma^2_u)\) is the random intercept for subject \(j\).

### Interpreting Results

```python
# Fixed effects: population-level estimates
model.fixef()

# Random effects: subject-specific deviations from population mean
model.ranef()

# Variance components
model.VarCorr()
```

The variance components show how much variability is due to between-subject differences vs. within-subject residual variation.

## Random Intercept and Slope Model

Allow both intercepts and slopes to vary by group:

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(model.summary())
```

The formula `(Days | Subject)` includes:

- Random intercept (baseline reaction time varies by subject)
- Random slope (effect of sleep deprivation varies by subject)
- Correlation between intercept and slope

### Interpreting Correlation

```python
vc = model.VarCorr()
print(vc)
```

The correlation tells you if subjects with higher baseline reaction times show larger or smaller effects of sleep deprivation.

## Uncorrelated Random Effects

Force random effects to be uncorrelated with `||`:

```python
model = mlm.lmer("Reaction ~ Days + (Days || Subject)", data)
```

This fits:
- Random intercept variance
- Random slope variance
- No correlation parameter

Use this when:
- You want a simpler model
- Correlation is not of interest
- The full model has convergence issues

## Nested Random Effects

When groups are nested (e.g., students within schools within districts):

```python
# Classrooms nested within schools
model = mlm.lmer("score ~ treatment + (1 | school/classroom)", data)

# Equivalent to:
model = mlm.lmer("score ~ treatment + (1 | school) + (1 | school:classroom)", data)
```

### Example: Pastes Data

```python
pastes = mlm.load_pastes()

# Samples nested within batches
model = mlm.lmer("strength ~ (1 | batch/sample)", pastes)
print(model.summary())
```

## Crossed Random Effects

When groups are crossed (not nested):

```python
# Items crossed with subjects
model = mlm.lmer("rating ~ (1 | subject) + (1 | item)", data)
```

### Example: Instructor Evaluations

```python
insteval = mlm.load_insteval()

# Students crossed with instructors, nested in departments
model = mlm.lmer(
    "y ~ service + lectage + studage + (1 | s) + (1 | d) + (1 | dept:service)",
    insteval
)
```

## Multiple Random Slopes

Include multiple random slopes:

```python
model = mlm.lmer("y ~ x1 + x2 + (x1 + x2 | group)", data)
```

This estimates:

- Random intercept
- Random slopes for x1 and x2
- Covariance matrix for all three random effects

## REML vs ML Estimation

By default, mixedlm uses REML (Restricted Maximum Likelihood):

```python
# REML (default) - better variance estimates
model_reml = mlm.lmer("y ~ x + (1 | g)", data, REML=True)

# ML - needed for likelihood ratio tests of fixed effects
model_ml = mlm.lmer("y ~ x + (1 | g)", data, REML=False)
```

Use ML when comparing models with different fixed effects. Use REML for final variance estimates.

## Model Comparison

### Comparing Random Effects Structures

Compare models with different random effects using likelihood ratio tests:

```python
# Random intercept only
m1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)

# Random intercept and slope
m2 = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)

# Likelihood ratio test
mlm.anova(m1, m2)
```

!!! note
    Use `REML=False` when comparing models with different random effects structures.

### Comparing Fixed Effects

For fixed effects comparisons, use Type III ANOVA or drop1:

```python
model = mlm.lmer("y ~ a * b + (1 | g)", data)

# Type III ANOVA (marginal tests)
mlm.anova_type3(model)

# Single term deletions
model.drop1(data)
```

## Predictions

### Conditional Predictions

Include random effects in predictions (default):

```python
# Predictions for observed subjects
model.predict()

# Predictions for new data with known subjects
model.predict(newdata=new_data)
```

### Marginal Predictions

Exclude random effects (set to zero):

```python
model.predict(newdata=new_data, re_form="~0")
```

## Handling Convergence Issues

### Check Convergence

```python
conv = mlm.checkConv(model)
if not conv.ok:
    print(conv.messages)
```

### Try Different Optimizers

```python
# Use allFit to try multiple optimizers
all_results = model.allFit(data)
print(all_results.summary())
```

### EM-REML Initialization

For models that struggle to converge from default starting values, use EM-REML initialization to find better starting points:

```python
ctrl = mlm.LmerControl(em_init=True, em_maxiter=50)
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, control=ctrl)
```

### Simplify Random Effects

If the full model won't converge:

1. Remove correlation: `(Days || Subject)` instead of `(Days | Subject)`
2. Remove random slopes: `(1 | Subject)` instead of `(Days | Subject)`
3. Check for near-zero variance components

### Singular Fits

A singular fit means some variance component is estimated at zero:

```python
if model.is_singular():
    print("Model has singular fit - consider simplifying random effects")
```

## Diagnostics

### Residual Plots

```python
from mixedlm import plot_diagnostics

plot_diagnostics(model)
```

This creates:

- Residuals vs fitted values
- Q-Q plot of residuals
- Scale-location plot
- Random effects Q-Q plots

### Influence Measures

```python
from mixedlm import influence, cooks_distance

# Full influence diagnostics
inf = influence(model, data)

# Cook's distance
cd = cooks_distance(model, data)
```

## Complete Example

```python
import mixedlm as mlm

# Load data
data = mlm.load_sleepstudy()

# Fit model
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Summary with p-values
print(model.summary())

# Variance components
print(model.VarCorr())

# Profile confidence intervals
ci = model.confint(method="profile")
print(ci)

# Check for influential observations
from mixedlm import cooks_distance
cd = cooks_distance(model, data)
print(f"Max Cook's distance: {cd.max():.3f}")

# Predictions
fitted = model.fitted()
```

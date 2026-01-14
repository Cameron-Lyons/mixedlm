# Statistical Inference

This tutorial covers hypothesis testing, confidence intervals, and model comparison for mixed models.

## P-values for Fixed Effects

### Satterthwaite Degrees of Freedom

By default, `summary()` reports p-values using Satterthwaite degrees of freedom:

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

print(model.summary())
```

The output includes degrees of freedom and p-values:

```
Fixed effects:
              Estimate  Std. Error    df  t value  Pr(>|t|)
(Intercept)    251.405       6.825  17.0   36.838    <0.001
Days            10.467       1.546  17.0    6.771    <0.001
```

### Kenward-Roger Degrees of Freedom

For small samples, Kenward-Roger provides better approximation:

```python
print(model.summary(ddf_method="Kenward-Roger"))
```

### Direct Access to DDF

```python
from mixedlm import satterthwaite_df, kenward_roger_df, pvalues_with_ddf

# Denominator degrees of freedom
sat_df = satterthwaite_df(model)
kr_df = kenward_roger_df(model)

# P-values with specific method
pvals = pvalues_with_ddf(model, method="Satterthwaite")
```

### When to Use Each Method

| Method | Use when |
|--------|----------|
| Satterthwaite | Default choice, fast, good for most cases |
| Kenward-Roger | Small samples, complex random effects, more accurate but slower |

## Confidence Intervals

### Wald Intervals

Fast but can be inaccurate for variance components:

```python
ci = model.confint(method="Wald")
print(ci)
```

### Profile Likelihood Intervals

More accurate, especially for variance components:

```python
ci = model.confint(method="profile")
print(ci)
```

Profile CIs are based on the likelihood function shape and don't assume symmetry.

### Bootstrap Intervals

Most robust but computationally intensive:

```python
ci = model.confint(method="boot", nsim=1000)
print(ci)
```

### Comparison

| Method | Speed | Fixed effects | Variance components |
|--------|-------|---------------|---------------------|
| Wald | Fast | Good | Poor (can go negative) |
| Profile | Medium | Excellent | Excellent |
| Bootstrap | Slow | Excellent | Excellent |

## Model Comparison

### Likelihood Ratio Tests

Compare nested models:

```python
# Simpler model
m1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)

# More complex model
m2 = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)

# Likelihood ratio test
result = mlm.anova(m1, m2)
print(result)
```

!!! important
    Use `REML=False` for likelihood ratio tests comparing random effects structures.

### Type III ANOVA

Test fixed effects in a single model:

```python
model = mlm.lmer("y ~ a * b + (1 | group)", data)
result = mlm.anova_type3(model)
print(result)
```

Type III tests are marginal: each effect is tested controlling for all others.

### Single Term Deletions (drop1)

Assess each term's contribution:

```python
result = model.drop1(data)
print(result)
```

This fits the model without each term and reports the change in fit.

## Estimated Marginal Means (emmeans)

### Computing Marginal Means

```python
model = mlm.lmer("yield ~ treatment + block + (1 | field)", data)

# Marginal means for treatment
em = mlm.emmeans(model, "treatment", data)
print(em.emmeans())
```

### Pairwise Contrasts

```python
# All pairwise comparisons
contrasts = em.contrasts("pairwise")
print(contrasts)
```

### Custom Contrasts

```python
# Compare specific levels
contrasts = em.contrasts("trt.vs.ctrl", ref="control")
print(contrasts)
```

### Multiple Comparison Adjustment

```python
contrasts = em.contrasts("pairwise", adjust="bonferroni")
# Options: "none", "bonferroni", "holm", "tukey"
```

## Profile Likelihood

### Computing Profiles

Profile likelihood for all parameters:

```python
profiles = model.profile(data)
```

### Visualizing Profiles

```python
from mixedlm import plot_profiles

plot_profiles(profiles)
```

Well-behaved profiles should be approximately parabolic (quadratic).

### 2D Profile Slices

Examine the relationship between two parameters:

```python
from mixedlm import slice2D

profile_2d = slice2D(model, "sigma", "theta1", n_points=20)
profile_2d.plot()
```

### Profile-Based CIs

```python
ci = profiles.confint()
print(ci)
```

## Parametric Bootstrap

### Basic Bootstrap

```python
from mixedlm import bootMer

# Bootstrap the model
boot = bootMer(model, data, nsim=500)

# Access bootstrap samples
boot.samples  # Array of parameter estimates

# Bootstrap confidence intervals
boot.confint()
```

### Bootstrap for Specific Statistics

```python
# Define a function to extract the statistic of interest
def my_stat(model):
    return model.fixef()['Days']

boot = bootMer(model, data, nsim=500, FUN=my_stat)
```

### Bootstrap for Predictions

```python
def predict_at_day_10(model):
    import pandas as pd
    new_data = pd.DataFrame({'Days': [10], 'Subject': ['308']})
    return model.predict(newdata=new_data)[0]

boot = bootMer(model, data, nsim=500, FUN=predict_at_day_10)
ci = boot.confint()
```

## Testing Random Effects

### Is the Random Effect Needed?

Compare models with and without the random effect:

```python
# Without random effect (regular linear model)
import scipy.stats as stats
from scipy import optimize

# With random effect
m1 = mlm.lmer("y ~ x + (1 | group)", data, REML=False)

# Compare to fixed-effect only model using LRT
# Note: test is on the boundary, so p-value should be halved
```

### Testing Variance Components

Likelihood ratio tests for variance components are conservative because the null hypothesis is on the boundary of the parameter space. The p-value from a chi-square test should typically be halved.

## Multiple Optimizers (allFit)

Check if results are sensitive to optimizer choice:

```python
all_results = model.allFit(data)
print(all_results.summary())
```

If different optimizers give very different results, the model may be problematic.

## Checking Convergence

```python
conv = mlm.checkConv(model)

if not conv.ok:
    print("Convergence issues detected:")
    for msg in conv.messages:
        print(f"  - {msg}")
```

## Complete Example

```python
import mixedlm as mlm

# Load data
data = mlm.load_sleepstudy()

# Fit model
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# 1. Summary with p-values
print("=== Model Summary ===")
print(model.summary())

# 2. Profile confidence intervals
print("\n=== Profile CIs ===")
profiles = model.profile(data)
print(profiles.confint())

# 3. Compare to simpler model
print("\n=== Model Comparison ===")
m_simple = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)
m_full = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)
print(mlm.anova(m_simple, m_full))

# 4. Bootstrap CI for the Days effect
print("\n=== Bootstrap CI for Days Effect ===")
boot = mlm.bootMer(model, data, nsim=200)
boot_ci = boot.confint()
print(f"Days: [{boot_ci['Days'][0]:.2f}, {boot_ci['Days'][1]:.2f}]")

# 5. Check convergence
conv = mlm.checkConv(model)
print(f"\n=== Convergence: {conv.ok} ===")
```

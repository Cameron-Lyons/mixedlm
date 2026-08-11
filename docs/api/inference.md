# Inference

This page documents functions for statistical inference, hypothesis testing, and confidence intervals.

## Model Comparison

### anova

Likelihood ratio tests between nested models.

```python
import mixedlm as mlm

result = mlm.anova(model1, model2, ...)
```

**Parameters:**

- `*models`: Two or more fitted models to compare

**Returns:** AnovaResult with chi-squared test statistics and p-values

**Example:**

```python
m1 = mlm.lmer("y ~ x + (1 | g)", data, REML=False)
m2 = mlm.lmer("y ~ x + z + (1 | g)", data, REML=False)
print(mlm.anova(m1, m2))
```

### anova_type3

Type III ANOVA for a single model.

```python
result = mlm.anova_type3(model)
```

**Returns:** AnovaType3Result with F-statistics and p-values for each fixed effect

**Example:**

```python
model = mlm.lmer("y ~ a * b + (1 | g)", data)
print(mlm.anova_type3(model))
```

## Degrees of Freedom

### satterthwaite_df

Compute Satterthwaite denominator degrees of freedom.

```python
df = mlm.satterthwaite_df(model)
```

**Returns:** Dictionary mapping coefficient names to degrees of freedom

### kenward_roger_df

Compute Kenward-Roger denominator degrees of freedom.

```python
df = mlm.kenward_roger_df(model)
```

**Returns:** Dictionary mapping coefficient names to degrees of freedom

### pvalues_with_ddf

Compute p-values using denominator degrees of freedom.

```python
pvals = mlm.pvalues_with_ddf(model, method="Satterthwaite")
```

**Parameters:**

- `model`: Fitted model
- `method`: `"Satterthwaite"` or `"Kenward-Roger"`

**Returns:** Dictionary mapping coefficient names to p-values

## Estimated Marginal Means

### ggpredict

Compute adjusted fixed-effect predictions for continuous variables, factors,
or their Cartesian product. Numeric variables outside the requested grid are
held at their mean and factors at their reference level.

```python
predictions = mlm.ggpredict(
    model,
    ["Days", "treatment"],
    at={"Days": [0, 5, 10]},
)
```

The returned data frame contains the requested grid columns plus `predicted`,
`std.error`, `conf.low`, and `conf.high`. For GLMMs, `type="response"` builds
the confidence interval on the link scale before transforming both endpoints.
Use `type="link"` to keep results on the linear-predictor scale.

### allEffects

Compute a separate adjusted prediction grid for every fixed-effect variable.

```python
effects = mlm.allEffects(model, n_points=25)
days_effect = effects["Days"]
```

Both functions are batched, use the fitted fixed-effect covariance matrix, and
require no plotting package. If a model was fit with non-default categorical
contrasts, pass the same mapping with `contrasts=` so the prediction grid uses
the fitted parameterization.

### emmeans

Compute estimated marginal means.

```python
em = mlm.emmeans(model, "treatment", type="response")
```

**Parameters:**

- `model`: Fitted model
- `specs`: Factor name or names to compute marginal means for
- `at`: Optional values at which to evaluate other predictors
- `cov_reduce`: Function used to reduce numeric covariates (default: mean)
- `type`: `"response"` (default) or `"link"` for generalized models
- `level`: Confidence level (default: `0.95`)

**Returns:** Emmeans object

**Methods on Emmeans object:**

- `pairs(adjust="tukey")`: Compute all pairwise comparisons
- `contrast(method, adjust="none")`: Compute pairwise, treatment-vs-control, or custom contrasts

**Example:**

```python
model = mlm.lmer("yield ~ treatment + (1 | block)", data)
em = mlm.emmeans(model, "treatment")
print(em)
print(em.pairs())
```

## Bootstrap

### bootMer

Parametric bootstrap for mixed models.

```python
boot = mlm.bootMer(model, nsim=500, seed=42)
```

**Parameters:**

- `model`: Fitted model
- `nsim`: Number of bootstrap simulations
- `seed`: Optional reproducibility seed

**Returns:** BootstrapResult object

**Methods:**

- `ci(level=0.95, method="percentile")`: Fixed-effect confidence intervals
- `se()`: Fixed-effect bootstrap standard errors
- `beta_samples`, `theta_samples`, `sigma_samples`: Bootstrap sample arrays

**Example:**

```python
boot = mlm.bootMer(model, nsim=500, seed=42)
ci = boot.ci()
print(ci)
```

### bootCI

Create tidy confidence intervals for fixed effects, variance parameters, and
the residual scale. Multiple interval methods can be computed in one pass.

```python
intervals = mlm.bootCI(
    boot,
    component="all",
    method=["percentile", "basic", "normal"],
)
```

The returned data frame includes each parameter's original estimate, bootstrap
mean, bias, sample standard error, confidence bounds, and successful replicate
count. `component="sigma"` is available for linear and nonlinear models; GLMM
results do not have a separately estimated residual scale.

## Profile Likelihood

### plot_profiles

Plot 1D profile likelihood curves.

```python
profiles = model.profile(data)
mlm.plot_profiles(profiles)
```

### slice2D

Compute 2D profile likelihood slice.

```python
profile_2d = mlm.slice2D(model, param1, param2, n_points=20)
```

**Parameters:**

- `model`: Fitted model
- `param1`, `param2`: Parameter names to profile
- `n_points`: Number of grid points per dimension

**Returns:** Profile2DResult with `plot()` method

## Convergence Checking

### checkConv

Check model convergence.

```python
conv = mlm.checkConv(model)
```

**Returns:** ConvergenceInfo object with:

- `ok`: Boolean indicating successful convergence
- `messages`: List of warning/error messages

### convergence_ok

Quick check if model converged successfully.

```python
if mlm.convergence_ok(model):
    print("Model converged")
```

**Returns:** Boolean

## Usage Examples

### Likelihood Ratio Test

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()

# Fit nested models (use REML=False for LRT)
m1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)
m2 = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)

# Compare
result = mlm.anova(m1, m2)
print(result)
```

### Type III ANOVA

```python
model = mlm.lmer("y ~ a * b + (1 | group)", data)
result = mlm.anova_type3(model)
print(result)
```

### P-values with Degrees of Freedom

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Satterthwaite (default in summary)
print(model.summary())

# Kenward-Roger
print(model.summary(ddf_method="Kenward-Roger"))

# Direct access
df_sat = mlm.satterthwaite_df(model)
df_kr = mlm.kenward_roger_df(model)
pvals = mlm.pvalues_with_ddf(model)
```

### Estimated Marginal Means

```python
model = mlm.lmer("yield ~ treatment + (1 | block)", data)

# Marginal means for treatment
em = mlm.emmeans(model, "treatment")
print(em)

# Pairwise contrasts
print(em.pairs())
```

### Bootstrap Confidence Intervals

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Parametric bootstrap
boot = mlm.bootMer(model, nsim=500, seed=42)

# Get CIs
ci = mlm.bootCI(boot, component="all")
print(ci)
```

### Profile Likelihood

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Compute profiles
profiles = model.profile(data)

# Plot
mlm.plot_profiles(profiles)

# Profile-based CIs
ci = profiles.confint()
print(ci)
```

### 2D Profile

```python
# Examine relationship between two parameters
profile_2d = mlm.slice2D(model, "sigma", "theta1", n_points=20)
profile_2d.plot()
```

### Check Convergence

```python
model = mlm.lmer("y ~ x + (x | g)", data)

conv = mlm.checkConv(model)
if not conv.ok:
    print("Convergence issues:")
    for msg in conv.messages:
        print(f"  - {msg}")
else:
    print("Model converged successfully")
```

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

### emmeans

Compute estimated marginal means.

```python
em = mlm.emmeans(model, variable, data)
```

**Parameters:**

- `model`: Fitted model
- `variable`: Variable name to compute marginal means for
- `data`: Original data frame

**Returns:** Emmeans object

**Methods on Emmeans object:**

- `emmeans()`: Get the marginal means
- `contrasts(type)`: Compute contrasts (`"pairwise"`, `"trt.vs.ctrl"`, etc.)

**Example:**

```python
model = mlm.lmer("yield ~ treatment + (1 | block)", data)
em = mlm.emmeans(model, "treatment", data)
print(em.emmeans())
print(em.contrasts("pairwise"))
```

## Bootstrap

### bootMer

Parametric bootstrap for mixed models.

```python
boot = mlm.bootMer(model, data, nsim=500, FUN=None)
```

**Parameters:**

- `model`: Fitted model
- `data`: Original data frame
- `nsim`: Number of bootstrap simulations
- `FUN`: Optional function to extract statistics (default: all parameters)

**Returns:** BootstrapResult object

**Methods:**

- `confint(level=0.95)`: Bootstrap confidence intervals
- `samples`: Array of bootstrap samples

**Example:**

```python
boot = mlm.bootMer(model, data, nsim=500)
ci = boot.confint()
print(ci)
```

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
em = mlm.emmeans(model, "treatment", data)
print(em.emmeans())

# Pairwise contrasts
print(em.contrasts("pairwise"))
```

### Bootstrap Confidence Intervals

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Parametric bootstrap
boot = mlm.bootMer(model, data, nsim=500)

# Get CIs
ci = boot.confint()
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

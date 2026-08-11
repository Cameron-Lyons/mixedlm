# Diagnostics

This page documents diagnostic functions for assessing model fit and identifying influential observations.

Diagnostic functions are available via `mixedlm.diagnostics`:

```python
import mixedlm as mlm
from mixedlm import diagnostics

# Or access directly
mlm.diagnostics.plot_diagnostics(model)
```

## Generalized-Model Diagnostics

### check_overdispersion

Compute a weighted Pearson chi-squared dispersion diagnostic for a fitted GLMM.

```python
result = diagnostics.check_overdispersion(
    model,
    alpha=0.05,
    alternative="two-sided",
)

print(result.dispersion_ratio)
print(result.p_value)
print(result.ci_low, result.ci_high)
```

The statistic is

\[
X^2 = \sum_i w_i \frac{(y_i - \hat{\mu}_i)^2}{V(\hat{\mu}_i)},
\]

and the dispersion ratio is \(X^2\) divided by the residual degrees of
freedom. Residual degrees of freedom subtract both fixed-effect and estimated
random-effect covariance parameters. The result exposes:

- `pearson_chi_square`: weighted Pearson statistic
- `residual_df`: residual degrees of freedom
- `dispersion_ratio`: estimated conditional dispersion
- `ci_low`, `ci_high`: chi-squared confidence interval for the ratio
- `p_value`: p-value for the requested alternative
- `status`: `"overdispersed"`, `"underdispersed"`, or `"ok"`
- `is_overdispersed`, `is_underdispersed`: convenience flags

Set `alternative="greater"` to test only for overdispersion or `"less"` to
test only for underdispersion. This is an analytic approximation conditional on
the fitted random effects. It is most useful for Poisson and grouped-binomial
models with moderate expected counts.

### check_zero_inflation

Compare the observed number of zeros with the fitted distribution's expected
number of zeros.

```python
result = diagnostics.check_zero_inflation(model)

print(result.observed_zeros)
print(result.expected_zeros)
print(result.observed_to_expected)
print(result.p_value)
```

The function computes each observation's probability of zero directly for
Poisson, negative-binomial, or binomial families. For grouped-binomial models,
integer model weights are interpreted as trial counts. It then uses the exact
Poisson-binomial mean and variance with a continuity-corrected normal
approximation for the zero-count p-value.

The default `alternative="greater"` tests for excess zeros. Use
`"two-sided"` to detect either excess or fewer-than-expected zeros. The result
exposes:

- `observed_zeros`, `expected_zeros`, and `observed_to_expected`
- `variance` and `z_score` for the zero-count approximation
- `p_value` and `status` (`"excess"`, `"deficit"`, or `"ok"`)
- `is_zero_inflated`: convenience flag for a significant excess

`check_zeroinflation()` is an alias. Quasi-families are unsupported because
they specify a mean-variance relationship without a complete probability
distribution.

## Diagnostic Plots

### plot_diagnostics

Create a panel of diagnostic plots including residuals vs fitted, Q-Q plot, scale-location, and random effects.

```python
diagnostics.plot_diagnostics(model, data=None)
```

**Parameters:**

- `model`: Fitted mixed model
- `data`: Optional data frame (required for some diagnostics)

### plot_resid_fitted

Residuals vs. fitted values plot.

```python
diagnostics.plot_resid_fitted(model, ax=None)
```

### plot_qq

Q-Q plot of residuals.

```python
diagnostics.plot_qq(model, ax=None)
```

### plot_scale_location

Scale-location plot for heteroscedasticity.

```python
diagnostics.plot_scale_location(model, ax=None)
```

### plot_ranef

Plot random effects with confidence intervals.

```python
diagnostics.plot_ranef(model, ax=None)
```

### plot_resid_group

Residuals by group.

```python
diagnostics.plot_resid_group(model, group, ax=None)
```

## Influence Diagnostics

### influence

Compute influence diagnostics for all observations.

```python
inf = diagnostics.influence(model, data)
```

**Returns:** InfluenceResult object with:

- `cooks_distance`: Cook's D values
- `dfbeta`: DFBETA values
- `dfbetas`: Standardized DFBETAS
- `dffits`: DFFITS values
- `leverage`: Leverage (hat) values

### cooks_distance

Compute Cook's distance for each observation.

```python
cd = diagnostics.cooks_distance(model, data)
```

**Returns:** Array of Cook's distance values.

**Interpretation:**

- Measures overall influence on all fitted values
- Common thresholds: > 4/n or > 1

### dfbeta

Compute DFBETA for each observation.

```python
dfb = diagnostics.dfbeta(model, data)
```

**Returns:** DataFrame with DFBETA for each coefficient.

### dfbetas

Compute standardized DFBETAS.

```python
dfbs = diagnostics.dfbetas(model, data)
```

**Returns:** DataFrame with standardized DFBETAS.

**Interpretation:**

- Common threshold: |DFBETAS| > 2/√n

### dffits

Compute DFFITS for each observation.

```python
dff = diagnostics.dffits(model, data)
```

**Returns:** Array of DFFITS values.

### leverage

Compute leverage (hat values) for each observation.

```python
lev = diagnostics.leverage(model, data)
```

**Returns:** Array of leverage values.

**Interpretation:**

- High leverage = unusual predictor values
- Common threshold: > 2p/n

### influence_plot

Create an influence plot showing leverage vs residuals, sized by Cook's distance.

```python
diagnostics.influence_plot(model, data, ax=None)
```

### influence_summary

Print a summary of influential observations.

```python
diagnostics.influence_summary(model, data)
```

### influential_obs

Identify influential observations based on multiple criteria.

```python
idx = diagnostics.influential_obs(model, data, threshold="default")
```

**Returns:** Indices of influential observations.

## Understanding Diagnostics

### Residuals vs Fitted

**What to look for:**

- Random scatter around zero: Good
- Funnel shape: Heteroscedasticity
- Curved pattern: Missing nonlinearity
- Outliers: Observations not well-fit

### Q-Q Plot

**What to look for:**

- Points on diagonal line: Normality satisfied
- Heavy tails (S-shape): Heavy-tailed distribution
- Light tails: Light-tailed distribution
- Skewness: Asymmetric deviation from line

### Scale-Location

**What to look for:**

- Horizontal line with random scatter: Constant variance
- Increasing trend: Variance increases with fitted values
- Decreasing trend: Variance decreases with fitted values

## Usage Examples

### Basic Diagnostics

```python
import mixedlm as mlm
from mixedlm import diagnostics

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Panel of diagnostic plots
diagnostics.plot_diagnostics(model)
```

### Individual Plots

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

diagnostics.plot_resid_fitted(model, ax=axes[0, 0])
diagnostics.plot_qq(model, ax=axes[0, 1])
diagnostics.plot_scale_location(model, ax=axes[1, 0])
diagnostics.plot_ranef(model, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

### Influence Analysis

```python
from mixedlm import diagnostics

# Compute influence measures
inf = diagnostics.influence(model, data)

# Cook's distance
cd = diagnostics.cooks_distance(model, data)
print(f"Max Cook's D: {cd.max():.4f}")

# Identify influential observations
influential = diagnostics.influential_obs(model, data)
print(f"Influential observations: {influential}")

# Summary of influential points
diagnostics.influence_summary(model, data)
```

### Influence Plot

```python
# Plot leverage vs residuals, sized by Cook's D
diagnostics.influence_plot(model, data)
```

### Checking Specific Observations

```python
# DFBETAS for effect on each coefficient
dfb = diagnostics.dfbetas(model, data)
print(dfb)

# Observations with large influence on Days coefficient
import numpy as np
large_influence = np.abs(dfb['Days']) > 2 / np.sqrt(len(data))
print(f"High influence on Days: {data.index[large_influence].tolist()}")
```

### Random Effects Diagnostics

```python
# Q-Q plot of random effects
diagnostics.plot_ranef(model)

# Check normality of random effects
ranef = model.ranef()
for group, effects in ranef.items():
    print(f"\n{group}:")
    for col in effects.columns:
        from scipy import stats
        stat, pval = stats.shapiro(effects[col])
        print(f"  {col}: Shapiro-Wilk p = {pval:.4f}")
```

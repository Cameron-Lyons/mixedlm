# Diagnostics

This page documents diagnostic functions for assessing model fit and identifying influential observations.

Diagnostic functions are available via `mixedlm.diagnostics`. The examples use a
sleepstudy fit:

```python
import mixedlm as mlm
from mixedlm import diagnostics

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Or access through the package
mlm.diagnostics.r2_nakagawa(model)
```

## Model Fit Metrics

### r2_nakagawa

Compute marginal R² (fixed effects) and conditional R² (fixed plus random effects):

```python
r2 = diagnostics.r2_nakagawa(model)
print(r2.marginal, r2.conditional)
print(r2.random_by_group)
```

Random-slope variance is evaluated at every observation, retaining covariance and predictor
values. Linear and nonlinear Gaussian models average the observation-specific residual
variance `sigma**2 / weights`; fixed and random contributions also give each fitted row
equal representation. Rescaling all precision weights by a common factor leaves the
diagnostic unchanged. Generalized models use a link-scale residual approximation:
lognormal for log links, link-specific theoretical variance for binomial models, or the
delta method when requested. Known offsets contribute to fixed prediction variance for
every model type. Built-in log-link families evaluate the residual approximation in log
space, so extreme fitted means do not overflow. Nonfinite fixed predictions raise an
error instead of being omitted.

### icc

Compute adjusted and unadjusted intraclass correlation coefficients:

```python
correlation = diagnostics.icc(model)
print(correlation.adjusted, correlation.unadjusted)
print(correlation.by_group)
```

The adjusted ICC excludes fixed-effect variance from its denominator. The unadjusted ICC
uses total fixed, random, and residual variance. Group-specific dictionaries partition each
overall coefficient for crossed or nested random effects.

## Collinearity Diagnostics

### check_collinearity

Compute weighted VIF, generalized VIF, tolerance, severity, and global condition indices:

```python
result = diagnostics.check_collinearity(model)
print(result)
print(result.to_dataframe())
print(result.problematic(threshold=5.0))
```

One-column terms report the conventional VIF. Multi-column categorical or interaction terms
also report raw GVIF, `GVIF^(1/(2*df))`, and a VIF-equivalent `GVIF^(1/df)` value so the same
thresholds can be used across terms. By default, values from 5 to 10 are marked moderate and
values of 10 or more are marked high; both thresholds are configurable.

Linear fits use prior weights, generalized fits use prior times local working weights, and
nonlinear fits use the locally weighted, uncentered parameter-gradient design so constant
information directions are preserved. The calculation uses one correlation eigendecomposition
and does not form the random-effect covariance matrix.

## Generalized-Model Diagnostics

### check_overdispersion

Compute a weighted Pearson chi-squared dispersion diagnostic for a fitted GLMM.

```python
grouseticks = mlm.load_grouseticks()
count_model = mlm.glmer(
    "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)", grouseticks, family=mlm.families.Poisson()
)

result = diagnostics.check_overdispersion(
    count_model,
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
result = diagnostics.check_zero_inflation(count_model)

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

Create a panel of diagnostic plots: residuals vs fitted, normal Q-Q, scale-location, and
residuals by group.

```python
fig = diagnostics.plot_diagnostics(model, which=None, figsize=None)
```

**Parameters:**

- `result`: Fitted linear or generalized mixed model
- `which`: Panels to draw, from 1 (residuals vs fitted), 2 (Q-Q), 3 (scale-location),
  and 4 (residuals by group); defaults to all four
- `figsize`: Figure size, default `(12, 10)`

**Returns:** The Matplotlib figure.

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

Plot random effects with conditional-variance intervals.

```python
diagnostics.plot_ranef(model, group="Subject", ax=None)
```

### plot_resid_group

Residuals by group.

```python
diagnostics.plot_resid_group(model, group="Subject", ax=None)
```

## Influence Diagnostics

The convenience functions below accept either a fitted model directly or an
`InfluenceResult` returned by `influence()`. Calculations use the final mixed-model
projection, including random effects, offsets, prior weights, and GLMM working weights.
Coefficient-deletion diagnostics hold variance components fixed. GLMM diagnostics
also hold the final working weights fixed, giving a local approximation to a full
GLMM deletion refit. They account for the change in random effects when calculating
the change in fixed coefficients, including the sign of decreasing links.

### influence

Compute influence diagnostics for all observations.

```python
inf = diagnostics.influence(model)
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
cd = diagnostics.cooks_distance(model)
```

**Returns:** Array of Cook's distance values. Models without fixed effects
return NaN, because Cook's distance measures changes in the fixed coefficients.
For models fitted with `na_action="exclude"`, influence values cover the fitted
observations.

**Interpretation:**

- Measures overall influence on all fitted values
- Common thresholds: > 4/n or > 1

### dfbeta

Compute DFBETA for each observation.

```python
dfb = diagnostics.dfbeta(model)
```

**Returns:** Array with one row per observation and one column per fixed-effect coefficient.

### dfbetas

Compute standardized DFBETAS.

```python
dfbs = diagnostics.dfbetas(model)
```

**Returns:** Array with standardized DFBETAS.

**Interpretation:**

- Common threshold: |DFBETAS| > 2/√n

### dffits

Compute DFFITS for each observation.

```python
dff = diagnostics.dffits(model)
```

**Returns:** Array of DFFITS values.

### leverage

Compute leverage (hat values) for each observation.

```python
lev = diagnostics.leverage(model)
```

**Returns:** Array of leverage values.

**Interpretation:**

- High leverage = unusual predictor values
- Common threshold: > 2p/n

### influence_plot

Plot Cook's distance, maximum absolute DFBETAS, leverage, or DFFITS.

```python
diagnostics.influence_plot(model, which="cooks", ax=None)
```

### influence_summary

Summarize the influence measures in a DataFrame.

```python
summary = diagnostics.influence_summary(model)
```

### influential_obs

Identify influential observations based on multiple criteria.

```python
idx = diagnostics.influential_obs(model, threshold="cooks")
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
```

### Influence Analysis

```python
from mixedlm import diagnostics

# Compute influence measures
inf = diagnostics.influence(model)

# Cook's distance
cd = diagnostics.cooks_distance(model)
print(f"Max Cook's D: {cd.max():.4f}")

# Identify influential observations
influential = diagnostics.influential_obs(model)
print(f"Influential observations: {influential}")

# Summary of influential points
print(diagnostics.influence_summary(model))
```

### Influence Plot

```python
# Plot Cook's distance
diagnostics.influence_plot(model, which="cooks")
```

### Checking Specific Observations

```python
# DFBETAS for effect on each coefficient
dfb = diagnostics.dfbetas(model)
print(dfb)

# Observations with large influence on Days coefficient
import numpy as np
days_index = model.matrices.fixed_names.index("Days")
large_influence = np.abs(dfb[:, days_index]) > 2 / np.sqrt(len(data))
print(f"High influence on Days: {data.index[large_influence].tolist()}")
```

### Random Effects Diagnostics

```python
# Q-Q plot of random effects
diagnostics.plot_ranef(model)

# Check normality of random effects
from scipy import stats

ranef = model.ranef()
for group, effects in ranef.items():
    print(f"\n{group}:")
    for term, values in effects.items():
        stat, pval = stats.shapiro(values)
        print(f"  {term}: Shapiro-Wilk p = {pval:.4f}")
```

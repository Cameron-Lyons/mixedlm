# Utilities

This page documents utility functions for working with mixed models.

## lme4 Compatibility

Functions for compatibility with R's lme4 package.

### sigma

Extract residual standard deviation.

```python
import mixedlm as mlm

s = mlm.sigma(model)
```

**Returns:** Float, residual standard deviation

### ngrps

Get number of groups for each random effect.

```python
ng = mlm.ngrps(model)
```

**Returns:** Dictionary mapping grouping factors to number of groups

### fixef

Extract fixed effects (standalone function).

```python
fe = mlm.fixef(model)
# Or: model.fixef()
```

**Returns:** Dictionary of fixed effect coefficients

### ranef

Extract random effects (standalone function).

```python
re = mlm.ranef(model)
# Or: model.ranef()
```

**Returns:** Dictionary with group-level random effects

### coef

Extract combined coefficients (standalone function).

```python
c = mlm.coef(model)
# Or: model.coef()
```

**Returns:** Dictionary with combined fixed + random effects per group

### getME

Extract model components.

```python
X = mlm.getME(model, "X")
# Or: model.getME("X")
```

**Common components:**

- `"X"`: Fixed effects design matrix
- `"Z"`: Random effects design matrix
- `"theta"`: Variance parameters
- `"Lambda"`: Relative covariance factor
- `"beta"`: Fixed effects
- `"b"`: Random effects (spherical)
- `"u"`: Random effects (conditional modes)

### fortify

Add model diagnostics to data.

```python
augmented = mlm.fortify(model, data)
```

**Returns:** DataFrame with added columns:

- `.fitted`: Fitted values
- `.resid`: Residuals
- `.hat`: Leverage values
- `.cooksd`: Cook's distance

### devcomp

Get deviance components.

```python
dc = mlm.devcomp(model)
```

**Returns:** DevComp object with deviance breakdown

### lmList

Fit separate linear models for each group.

```python
lm_dict = mlm.lmList("y ~ x | group", data)
```

**Returns:** Dictionary mapping group names to fitted OLS models

### isNested

Check if random effects are nested.

```python
nested = mlm.isNested(data['classroom'], data['school'])
```

**Returns:** Boolean indicating if first factor is nested in second

## Variance Transformations

Functions for converting between variance parameterizations.

### sdcor2cov

Convert standard deviations and correlations to covariance matrix.

```python
import numpy as np

sd = np.array([2.0, 1.5])
corr = np.array([[1.0, 0.3], [0.3, 1.0]])
cov = mlm.sdcor2cov(sd, corr)
```

### cov2sdcor

Convert covariance matrix to standard deviations and correlations.

```python
sd, corr = mlm.cov2sdcor(cov)
```

### Vv_to_Cv / Cv_to_Vv

Convert between variance vector and Cholesky factor.

```python
cv = mlm.Vv_to_Cv(variance_vector)
vv = mlm.Cv_to_Vv(cholesky_vector)
```

## EM-REML Initialization

### em_reml_simple

Fit a linear mixed model using the EM-REML algorithm. Useful as a standalone estimator or as an initialization step before direct optimization.

```python
from mixedlm import lFormula
from mixedlm.estimation.em_reml import em_reml_simple

parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)
result = em_reml_simple(parsed.matrices, max_iter=100, tol=1e-5)
```

**Parameters:**

- `matrices`: `ModelMatrices` — design matrices from `lFormula()`
- `max_iter`: int — maximum EM iterations (default 100)
- `tol`: float — convergence tolerance for relative log-likelihood change (default 1e-4)
- `verbose`: int — verbosity level (default 0)
- `variance_floor`: float — minimum variance to prevent numerical issues (default 1e-8)

**Returns:** `EMResult` with fields `theta`, `beta`, `sigma`, `converged`, `n_iter`, `final_loglik`

**Supported models:** Random intercepts, correlated and uncorrelated random slopes, multiple random effect terms (cov_type='us' only).

!!! tip
    For most users, `em_init=True` in `LmerControl` or `GlmerControl` is the easier way to use EM-REML. The standalone function is for advanced workflows.

## Formula Utilities

### parse_formula

Parse a model formula.

```python
parsed = mlm.parse_formula("y ~ x + (x | g)")
```

### findbars

Find random effects terms in a formula.

```python
bars = mlm.findbars("y ~ x + (x | g1) + (1 | g2)")
# ['(x | g1)', '(1 | g2)']
```

### nobars

Remove random effects from a formula.

```python
fixed = mlm.nobars("y ~ x + (x | g)")
# 'y ~ x'
```

### is_mixed_formula

Check if formula contains random effects.

```python
mlm.is_mixed_formula("y ~ x + (1 | g)")  # True
mlm.is_mixed_formula("y ~ x")            # False
```

## Usage Examples

### Extracting Model Information

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Residual SD
print(f"Sigma: {mlm.sigma(model)}")

# Number of groups
print(f"Groups: {mlm.ngrps(model)}")

# Fixed and random effects
print(f"Fixed: {mlm.fixef(model)}")
print(f"Random: {mlm.ranef(model)}")
```

### Model Matrices

```python
# Design matrices
X = mlm.getME(model, "X")  # Fixed effects
Z = mlm.getME(model, "Z")  # Random effects
print(f"X shape: {X.shape}")
print(f"Z shape: {Z.shape}")

# Variance parameters
theta = mlm.getME(model, "theta")
Lambda = mlm.getME(model, "Lambda")
```

### Adding Diagnostics to Data

```python
# Fortify adds residuals, fitted values, etc.
augmented = mlm.fortify(model, data)
print(augmented.columns.tolist())
# [..., '.fitted', '.resid', '.hat', '.cooksd', ...]
```

### Variance Conversions

```python
import numpy as np

# Standard deviations and correlation
sd = np.array([2.0, 1.5])
corr = np.array([[1.0, 0.3], [0.3, 1.0]])

# Convert to covariance
cov = mlm.sdcor2cov(sd, corr)
print(cov)

# Convert back
sd_back, corr_back = mlm.cov2sdcor(cov)
```

### Formula Parsing

```python
formula = "y ~ x + (x | g1) + (1 | g2)"

# Check if mixed
print(mlm.is_mixed_formula(formula))  # True

# Extract random effects
bars = mlm.findbars(formula)
print(bars)  # ['(x | g1)', '(1 | g2)']

# Get fixed part only
fixed = mlm.nobars(formula)
print(fixed)  # 'y ~ x'
```

### Per-Group Models

```python
# Fit separate models for each subject (no pooling)
lm_list = mlm.lmList("Reaction ~ Days | Subject", data)

# Compare to mixed model
for subj, lm in lm_list.items():
    print(f"{subj}: intercept={lm.params[0]:.1f}, slope={lm.params[1]:.2f}")
```

### Checking Nesting

```python
# Check if group2 is nested within group1
nested = mlm.isNested(data['classroom'], data['school'])
print(f"Classrooms nested in schools: {nested}")
```

### Deviance Components

```python
dc = mlm.devcomp(model)
print(f"Deviance: {dc.deviance}")
print(f"REML: {dc.REML}")
```

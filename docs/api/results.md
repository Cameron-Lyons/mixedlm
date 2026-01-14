# Results

This page documents the result objects returned by model fitting functions and their methods.

## LmerResult

The result object returned by `lmer()`.

### Methods

#### summary

```python
result.summary(ddf_method="Satterthwaite")
```

Print a summary of the fitted model including fixed effects with p-values.

**Parameters:**

- `ddf_method`: Method for denominator degrees of freedom. Options: `"Satterthwaite"` (default), `"Kenward-Roger"`.

#### fixef

```python
result.fixef()
```

Extract fixed effects coefficients.

**Returns:** Dictionary mapping coefficient names to values.

#### ranef

```python
result.ranef(condVar=False)
```

Extract random effects (BLUPs).

**Parameters:**

- `condVar`: If True, include conditional variances.

**Returns:** Dictionary with group names as keys and DataFrames of random effects as values.

#### VarCorr

```python
result.VarCorr()
```

Extract variance-covariance components of random effects.

**Returns:** VarCorr object with variance, standard deviation, and correlation information.

#### coef

```python
result.coef()
```

Extract combined coefficients (fixed + random) for each group.

**Returns:** Dictionary mapping group names to coefficient DataFrames.

#### fitted

```python
result.fitted()
```

Extract fitted values.

**Returns:** Array of fitted values.

#### residuals

```python
result.residuals(type="response")
```

Extract residuals.

**Parameters:**

- `type`: Type of residuals. Options: `"response"`, `"pearson"`, `"deviance"`.

**Returns:** Array of residuals.

#### predict

```python
result.predict(newdata=None, re_form=None, type="response")
```

Generate predictions.

**Parameters:**

- `newdata`: New data for prediction. If None, uses original data.
- `re_form`: Formula for random effects. Use `"~0"` to exclude random effects.
- `type`: Type of prediction. Options: `"response"`, `"link"`.

**Returns:** Array of predictions.

#### simulate

```python
result.simulate(nsim=1)
```

Simulate responses from the fitted model.

**Parameters:**

- `nsim`: Number of simulations.

**Returns:** Array of shape (n_obs, nsim).

#### confint

```python
result.confint(method="Wald", level=0.95)
```

Compute confidence intervals.

**Parameters:**

- `method`: CI method. Options: `"Wald"`, `"profile"`, `"boot"`.
- `level`: Confidence level.

**Returns:** DataFrame with lower and upper bounds.

#### logLik

```python
result.logLik()
```

Extract log-likelihood.

**Returns:** LogLik object with value and degrees of freedom.

#### AIC / BIC

```python
result.AIC()
result.BIC()
```

Compute information criteria.

**Returns:** Float value.

#### profile

```python
result.profile(data)
```

Compute profile likelihood.

**Returns:** ProfileResult object.

#### drop1

```python
result.drop1(data)
```

Test single term deletions.

**Returns:** Drop1Result with test statistics.

#### allFit

```python
result.allFit(data)
```

Fit model with multiple optimizers.

**Returns:** AllFitResult comparing optimizer results.

#### getME

```python
result.getME(name)
```

Extract model components.

**Parameters:**

- `name`: Component name. Options include `"X"`, `"Z"`, `"theta"`, `"Lambda"`, `"Zt"`, `"beta"`, `"b"`, `"u"`, etc.

**Returns:** The requested component.

#### is_singular

```python
result.is_singular()
```

Check if fit is singular (variance at boundary).

**Returns:** Boolean.

## GlmerResult

The result object returned by `glmer()`. Has the same methods as LmerResult plus:

#### family

```python
result.family
```

The distribution family used.

## NlmerResult

The result object returned by `nlmer()`. Has similar methods to LmerResult.

## VarCorr

Variance-covariance structure of random effects.

### Attributes

- `groups`: List of grouping factors
- `variance`: Variance estimates
- `stddev`: Standard deviation estimates
- `corr`: Correlation matrices

### String Representation

```python
print(result.VarCorr())
```

```
Groups   Name        Variance  Std.Dev.  Corr
Subject  (Intercept)  612.10    24.74
         Days          35.07     5.92    0.07
Residual              654.94    25.59
```

## LogLik

Log-likelihood with degrees of freedom.

### Attributes

- `value`: Log-likelihood value
- `df`: Degrees of freedom (number of parameters)
- `nobs`: Number of observations

## Usage Examples

### Extracting Components

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Fixed effects
print(result.fixef())
# {'(Intercept)': 251.405, 'Days': 10.467}

# Random effects for first subject
ranef = result.ranef()
print(ranef['Subject'].head())

# Variance components
print(result.VarCorr())

# Model matrices
X = result.getME("X")  # Fixed effects design matrix
Z = result.getME("Z")  # Random effects design matrix
```

### Predictions

```python
import pandas as pd

# Predictions on original data
fitted = result.predict()

# Predictions for new subjects
new_data = pd.DataFrame({
    'Days': [0, 5, 10],
    'Subject': ['new_subj', 'new_subj', 'new_subj']
})

# Include random effects (will be 0 for new subjects)
pred_cond = result.predict(newdata=new_data)

# Exclude random effects (population average)
pred_marg = result.predict(newdata=new_data, re_form="~0")
```

### Model Diagnostics

```python
# Check for singular fit
if result.is_singular():
    print("Warning: Singular fit detected")

# Check convergence
conv = mlm.checkConv(result)
print(f"Converged: {conv.ok}")
```

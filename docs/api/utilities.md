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

**Returns:** Nested dictionary mapping each grouping factor to all fixed and
random-only coefficient names, with one combined value per grouping level.

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

This uses the built-in formula encoder and NumPy least squares, including
categorical predictors, interactions, and multiple numeric predictors without
an additional dependency.

**Returns:** A dictionary containing per-group fits, a coefficient table, and
an optional pooled fit

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

Convert between a relative Cholesky factor (`theta`) and a covariance matrix,
both stored as lower-triangular vectors in row order. Supply the number of random
coefficients as `q` and the fitted residual scale as `sigma`.

```python
theta = np.array([1.0, 0.2, 0.8])  # L[0, 0], L[1, 0], L[1, 1]
cv = mlm.Vv_to_Cv(theta, q=2, sigma=1.5)
theta_back = mlm.Cv_to_Vv(cv, q=2, sigma=1.5)
```

### vcconv

Report fitted variance parameters as standard deviations and correlations
(`to="sdcorr"`), variances and covariances (`to="varcov"`), or the original
parameters (`to="theta"`). This supports unstructured, independent,
compound-symmetry, and AR(1) random effects.

```python
components = mlm.vcconv(
    result.theta,
    result.matrices.random_structures,
    sigma=result.sigma,  # Use 1.0 for a GLMM.
    to="varcov",
)
```

The result maps covariance block names to dictionaries containing coefficient
names (`terms`), the original `grouping_factor`, and the requested values.
Repeated grouping factors receive unique names such as `group` and `group.1`,
so each random-effect term is retained. Existing grouping-factor names are
reserved: if the data also contains a group named `group.1`, the second block
for `group` is named `group.2`.

Covariance and correlation lists use
upper-triangular row order: `(0, 1), (0, 2), ..., (1, 2), ...`. Independent terms
have empty off-diagonal lists. Returning `theta` preserves the fitted parameter
layout and does not apply `sigma`.

## Sparse Cholesky

The native `SparseCholeskySymbolic` class reuses symbolic analysis when a
positive definite matrix changes values while keeping its CSC sparsity pattern.
It uses approximate minimum degree (`ordering="amd"`) to reduce factor fill.
Choose `ordering="natural"` to retain the original variable order during
factorization. Solutions always follow the original row order.

```python
from mixedlm import SparseCholeskySymbolic

# A is a square scipy.sparse CSC matrix; rhs has shape (A.shape[0], n_rhs).
symbolic = SparseCholeskySymbolic(
    A.indices.astype("int64"), A.indptr.astype("int64"), A.shape[0],
    ordering="amd",
)
numeric = symbolic.factor(A.data.astype("float64"))
solution = numeric.solve(rhs)
logdet = numeric.logdet()
factor_entries = symbolic.factor_nonzeros()  # Includes diagonal and fill.
```

The matrix's lower triangle defines the symmetric system. Full symmetric
matrices and stored lower triangles are both accepted, including valid CSC
columns with unsorted or duplicate entries. Numeric factorization reports a
`ValueError` when the matrix is not positive definite.

Symbolic analysis, factorization, solves, and determinant calculation release
the Python interpreter lock. Factors can be reused across threads. Each call
copies its array inputs before releasing the lock, so subsequent changes to
those arrays do not affect work already in progress. `solve` accepts strided
right-hand sides and returns an independent C-contiguous `float64` array.

`python benchmarks/benchmark_sparse_ordering.py` compares both orderings on a
hub system, reports factor storage and median run times, and checks solutions
and determinants against its closed-form Schur complement.

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

**Supported models:** Random intercepts, correlated and uncorrelated random slopes, multiple random effect terms, and compound-symmetry covariance (`cov_type='cs'`).

!!! tip
    For most users, `em_init=True` in `LmerControl` or `GlmerControl` is the easier way to use EM-REML. The standalone function is for advanced workflows.

## Formula Utilities

### parse_formula

Parse a model formula.

```python
parsed = mlm.parse_formula("y ~ x + (x | g)")
```

### simulate_formula

Simulate responses before fitting, using the same variance-parameter ordering
and covariance structures as the model optimizers.

```python
formula = mlm.set_cov_type("y ~ x + (x | g)", "cs")
simulated = mlm.simulate_formula(
    formula,
    data,
    beta={"(Intercept)": 1.0, "x": 0.5},
    theta=[0.8, 0.25],
    seed=42,
)
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

# Inspect the per-subject coefficient table
print(lm_list["coef"])

# Access one fitted model's arrays and metadata
subject_fit = lm_list["fits"]["308"]
print(subject_fit["params"])
print(subject_fit["residuals"])
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

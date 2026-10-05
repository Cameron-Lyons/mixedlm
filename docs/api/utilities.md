# Utilities

This page documents utility functions for working with mixed models. The examples
use a sleepstudy fit:

```python
import mixedlm as mlm
import numpy as np

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
```

## lme4 Compatibility

Functions for compatibility with R's lme4 package.

### sigma

Extract residual standard deviation.

```python
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
- `"b"`: Conditional modes of the random effects
- `"u"`: Spherical random effects, `b = Lambda @ u`

### fortify

Add fitted values and residuals to data.

```python
augmented = mlm.fortify(model, data)
```

**Returns:** A copy of the data with added columns:

- `.fitted`: Fitted values on the response scale. With `include_re=False`, these
  are population-level predictions from the fixed effects and offset.
- `.resid`: Residuals conditional on the random effects
- `.fixed`: Fixed-effects linear predictor, including the offset
- `.mu`: Conditional response-scale fitted values (GLMMs only)

`data` can be the fitted observations or the original data including rows
dropped for missing values; dropped rows receive NaN. Data of any other length
raises `ValueError`. Without `data`, the stored model frame is used.

### devcomp

Get deviance components.

```python
dc = mlm.devcomp(model)
dc.cmp["pwrss"]  # Penalized weighted residual sum of squares
```

**Returns:** `DevComp` with the lme4 components, the same values as
`model.getME("devcomp")`:

- `cmp`: `ldL2` and `ldRX2` (log determinants of the random- and fixed-effect
  factors), `wrss` (prior-weighted residual sum of squares, or the Pearson sum of
  squares for GLMMs), `ussq` (squared length of the spherical random effects),
  `pwrss` (`wrss + ussq`), `drsum` (GLMM deviance residual sum), `REML` (REML fits),
  `dev` (ML fits), and `sigmaML` and `sigmaREML` (LMMs). Components a model does
  not define are NaN.
- `dims`: `n`, `p`, `q`, `nmp`, `nth`, `REML`, `useSc`, `nAGQ`, `q0`, `q1`, `qrx`,
  and `ngrps`, the number of grouping factors.

Nonlinear models raise `TypeError`.

### lmList

Fit separate linear models for each group.

```python
lm_dict = mlm.lmList("Reaction ~ Days | Subject", data)
```

This uses the built-in formula encoder and NumPy least squares, including
categorical predictors, interactions, and multiple numeric predictors without
an additional dependency.

**Returns:** A dictionary containing per-group fits, a coefficient table, and
an optional pooled fit

### isNested

Check if random effects are nested.

```python
pastes = mlm.load_pastes()
nested = mlm.isNested(pastes["sample"], pastes["batch"])
```

**Returns:** Boolean indicating if first factor is nested in second

### dummy

Build a coded matrix for one categorical variable.

```python
codes = mlm.dummy(["b", "a", "c", "a"], base="b")
```

Pandas categoricals keep their category order; other inputs use sorted unique
values as levels. `contrasts` accepts `"treatment"` (the default), `"sum"`,
`"helmert"`, and `"poly"`, matching the `contr_*` functions in
`mixedlm.utils.contrasts`. `base` names the treatment-coding reference level or
gives its index; negative indices count from the end, and unknown levels or
indices raise `ValueError`.

**Returns:** Array with one row per observation and one column per non-reference
level.

## Variance Transformations

Functions for converting between variance parameterizations.

### sdcor2cov

Convert standard deviations and correlations to covariance matrix.

```python
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
    model.theta,
    model.matrices.random_structures,
    sigma=model.sigma,  # Use 1.0 for a GLMM.
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
import scipy.sparse as sp

from mixedlm import SparseCholeskySymbolic

# A is a square scipy.sparse CSC matrix; rhs has shape (A.shape[0], n_rhs).
A = sp.csc_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]))
rhs = np.ones((3, 2))

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

Analysis, factorization, solves, and determinants release the GIL, so factors
can be shared across threads. Inputs are copied first; later changes to the
caller's arrays do not affect work in progress. `solve` accepts strided
right-hand sides and returns a new C-contiguous `float64` array.

The one-shot `mixedlm._rust.sparse_cholesky_solve()` and
`sparse_cholesky_logdet()` functions accept the same keyword-only `ordering`.
`"natural"` skips AMD analysis, which helps for banded or block systems that are
already well ordered:

```python
from mixedlm._rust import sparse_cholesky_logdet, sparse_cholesky_solve

parts = (A.data, A.indices.astype("int64"), A.indptr.astype("int64"), A.shape)
solution = sparse_cholesky_solve(*parts, rhs, ordering="natural")
logdet = sparse_cholesky_logdet(*parts, ordering="natural")
```

`pytest tests/test_benchmark.py -k sparse_hub_ordering --benchmark-only` compares
both orderings on a hub system whose fill depends on the ordering.

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
formula = mlm.set_cov_type("Reaction ~ Days + (Days | Subject)", "cs")
simulated = mlm.simulate_formula(
    formula,
    data,
    beta={"(Intercept)": 250.0, "Days": 10.0},
    theta=[0.8, 0.25],
    seed=42,
)
```

The data needs the predictor and grouping columns; the response column may be
absent. `family` accepts a `Family` instance or a name: `"gaussian"` (the
default), `"binomial"`, `"poisson"`, `"gamma"`, or `"inverse_gaussian"`, with R
spellings such as `"Gamma"` also accepted. Unknown names raise `ValueError`.
`sigma` is the Gaussian residual standard deviation and the gamma and inverse
Gaussian dispersion. Grouped `successes / trials` formulas draw success counts
using the trials column. `quickSimulate()` accepts the same arguments in a
shorter form:

```python
counts = mlm.quickSimulate(
    "count ~ Days + (1 | Subject)",
    data[["Days", "Subject"]],
    beta={"(Intercept)": 1.0, "Days": 0.1},
    theta=[0.5],
    family="poisson",
    seed=1,
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
# Fortify adds fitted values and residuals
augmented = mlm.fortify(model, data)
print(augmented.columns.tolist())
# ['Reaction', 'Days', 'Subject', '.fitted', '.resid', '.fixed']
```

### Variance Conversions

```python
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
# Check whether samples are nested within batches
nested = mlm.isNested(pastes["sample"], pastes["batch"])
print(f"Samples nested in batches: {nested}")
```

### Deviance Components

```python
dc = mlm.devcomp(model)
print(f"REML criterion: {dc.cmp['REML']}")
print(f"Observations: {dc.dims['n']}")
```

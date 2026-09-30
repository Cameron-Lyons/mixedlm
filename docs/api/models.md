# Models

This page documents the main model fitting functions and control classes.

Linear and generalized fits validate their final variance parameters, fixed effects,
random effects, and deviance before returning a result. Linear fits also require a
positive finite residual scale. Invalid final evaluations raise `RuntimeError` with
the underlying numerical reason; this applies to refits and the modular result
constructors as well. Bootstrap counts these refit errors as failed samples.

A finite fit that reaches an iteration limit can still be returned with
`converged=False`. Final deviance is recomputed with the reported estimates,
including the `nAGQ` requested when constructing a modular GLMM result.

## Model Fitting Functions

### lmer

Fit a linear mixed model.

```python
import mixedlm as mlm

result = mlm.lmer(formula, data, REML=True, control=None)
```

**Parameters:**

- `formula`: Model formula string (e.g., `"y ~ x + (x | group)"`)
- `data`: DataFrame (pandas or polars)
- `REML`: Use REML estimation (default True). Set to False for ML.
- `control`: Optional LmerControl object for optimization settings

**Returns:** LmerMod result object

**Example:**

```python
data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(model.summary())
```

### glmer

Fit a generalized linear mixed model.

```python
result = mlm.glmer(formula, data, family, nAGQ=1, control=None)
```

**Parameters:**

- `formula`: Model formula string
- `data`: DataFrame
- `family`: Distribution family (e.g., `mlm.families.Binomial()`)
- `nAGQ`: Positive integer number of quadrature points. 1 = Laplace approximation. Values above one require a single random-effect term with one coefficient per group. Models with no random effects are also supported.
- `control`: Optional GlmerControl object

**Returns:** GlmerMod result object

**Example:**

```python
cbpp = mlm.load_cbpp()
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial()
)
```

### glmer_nb

Fit a negative binomial GLMM with estimated dispersion.

```python
result = mlm.glmer_nb(formula, data, control=None)
```

**Parameters:**

- `formula`: Model formula string
- `data`: DataFrame
- `control`: Optional GlmerControl object

**Returns:** GlmerMod result object with estimated theta

**Example:**

```python
model = mlm.glmer_nb("count ~ treatment + (1 | subject)", data)
print(f"Estimated theta: {model.family.theta}")
```

### nlmer

Fit a nonlinear mixed model.

```python
result = mlm.nlmer(
    model, data, x_var, y_var, group_var,
    random_params=None, start=None, weights=None, offset=None,
)
```

**Parameters:**

- `model`: A nonlinear model instance, such as `SSasymp()` or `SSlogis()`
- `data`: DataFrame
- `x_var`, `y_var`, `group_var`: Predictor, response, and grouping column names
- `random_params`: Parameter names or indexes with random effects; defaults to all parameters
- `start`: Optional dictionary of starting parameter values; otherwise initialized automatically
- `weights`: Positive prior observation weights
- `offset`: Known observation offsets added to the nonlinear response mean
- `**kwargs`: Optimizer settings such as `method` and `maxiter`

**Returns:** NlmerResult object

**Raises:** `RuntimeError` if optimization ends at an invalid nonlinear
evaluation, including the underlying numerical or model error when available.
A valid result retains the optimizer's convergence status in `converged`.

**Example:**

```python
from mixedlm.nlme import SSlogis

model = mlm.nlmer(
    SSlogis(),
    data,
    x_var="time",
    y_var="response",
    group_var="subject",
    random_params=["Asym"],
)
```

## Control Classes

Control objects configure optimization and convergence settings.

### LmerControl

```python
control = mlm.LmerControl(
    optimizer="COBYQA",
    maxiter=10000,
    optCtrl={"final_tr_radius": 1e-6},
)
```

**Parameters:**

- `optimizer`: Optimization algorithm. Options include `"COBYQA"` (default), `"L-BFGS-B"`, `"BFGS"`, `"Nelder-Mead"`, and `"Powell"`
- `maxiter`: Maximum number of iterations
- `optCtrl`: Optimizer-specific options, such as COBYQA's `final_tr_radius`

### GlmerControl

```python
control = mlm.GlmerControl(
    optimizer="COBYQA",
    maxiter=10000,
    tolPwrss=1e-8,
    pirls_maxiter=100,
    optCtrl={"final_tr_radius": 1e-6},
)
```

**Parameters:**

- The outer optimizer settings are the same as `LmerControl`.
- `tolPwrss`: Positive finite tolerance for the maximum absolute changes in fixed
  effects and spherical random effects during PIRLS. The default is `1e-7`.
- `pirls_maxiter`: Positive integer limit on inner PIRLS iterations per likelihood
  evaluation. The default, `None`, retains each backend's existing limit: 100 for
  native fitting and 25 for Python fitting. Set an integer to use the same limit
  for both. This is separate from the outer optimizer's `maxiter`.

These inner controls apply to Laplace and adaptive quadrature, including final
coefficient extraction and modular fitting. Results retain `pirls_maxiter` and
`pirls_tol`; `result.refit()` inherits them and accepts overrides with those names.
Reconstructed objectives, model updates, bootstrap, optimizer comparisons,
term deletion, and cross-validation also preserve the fitted inner settings.
An explicit `control` supplied to an update or cross-validation fit overrides them.

A fit that reaches its inner limit without converging reports
`pirls_converged=False` even when the outer optimizer succeeds.

Pass quadrature order to `glmer(..., nAGQ=7)`.

## Modular Interface

For advanced users who need fine-grained control over the fitting process.

### lFormula

Parse formula and prepare data structures for LMM.

```python
parsed = mlm.lFormula(formula, data, REML=True)
```

**Returns:** LmerParsedFormula with design matrices, random effects terms, etc.

### glFormula

Parse formula and prepare data structures for GLMM.

```python
parsed = mlm.glFormula(formula, data, family)
```

**Returns:** GlmerParsedFormula

Both modular preparers accept strings and composed `Formula` objects, including
objects returned by utilities such as `set_cov_type()`.

### mkLmerDevfun

Create the deviance function for optimization.

```python
devfun = mlm.mkLmerDevfun(parsed_formula)
```

### optimizeLmer

Run the optimizer on the deviance function.

```python
opt_result = mlm.optimizeLmer(devfun)
```

### mkLmerMod

Create the final model object from optimization results.

```python
model = mlm.mkLmerMod(devfun, opt_result)
```


### Modular generalized models and quadrature

Choose the quadrature setting when creating the generalized deviance function:

```python
parsed = mlm.glFormula(
    "incidence / size ~ period + (1 | herd)",
    mlm.load_cbpp(),
    family=mlm.families.Binomial(),
)
devfun = mlm.mkGlmerDevfun(parsed, nAGQ=5)
opt_result = mlm.optimizeGlmer(devfun)
model = mlm.mkGlmerMod(devfun, opt_result)
assert model.nAGQ == 5
```

`mkGlmerDevfun()` accepts `nAGQ` as a keyword argument and defaults to 1.
`optimizeGlmer()` records the setting in `opt_result.nAGQ`. `mkGlmerMod()` inherits
that recorded value; for a custom `OptimizeResult` without quadrature metadata,
it uses the deviance function's setting. An explicit `mkGlmerMod(..., nAGQ=...)`
must agree with the setting used for optimization. To change it, create a new
deviance function and optimize again.

Quadrature requests must use a positive integer. Values above one require a
single random-effect term with one coefficient per group (for example, `(1 | g)`
or `(0 + x | g)`). Multiple random-effect terms and random-intercept/slope blocks
require `nAGQ=1`. Unsupported requests raise `ValueError` before numerical
optimization. `nAGQ=0` is not implemented. These checks also apply to direct GLMM
fitting and the Python quadrature evaluators.

## Usage Examples

### Basic Linear Mixed Model

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(model.summary())
```

### GLMM with Control Settings

```python
control = mlm.GlmerControl(
    optimizer="COBYQA",
    maxiter=50000,
    optCtrl={"final_tr_radius": 1e-8},
)

model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    data,
    family=mlm.families.Binomial(),
    control=control
)
```

### Modular Fitting

```python
# Step-by-step fitting for custom workflows
parsed = mlm.lFormula("y ~ x + (1 | g)", data)
devfun = mlm.mkLmerDevfun(parsed)
opt_result = mlm.optimizeLmer(devfun)
model = mlm.mkLmerMod(parsed, opt_result)
```

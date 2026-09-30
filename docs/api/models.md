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
- `nAGQ`: Nonnegative integer. 0 selects the faster joint-PIRLS approximation; 1 uses Laplace approximation with joint optimization of fixed coefficients and covariance parameters. Values above one use adaptive quadrature and require a single random-effect term with one coefficient per group. Models with no random effects are also supported.
- `control`: Optional GlmerControl object

**Returns:** GlmerMod result object

The default fit optimizes the integrated likelihood over both `theta` and `beta`.
This can change estimates and increase fitting time relative to the previous
theta-only algorithm, which is available through `nAGQ=0`. Zero selects that
fitting algorithm; it does not request an integral with zero quadrature nodes.
`result.joint_fit` records whether the result uses the joint likelihood objective.

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
- `**kwargs`: Outer optimizer settings `method` and `maxiter`, plus inner PNLS
  controls `pnls_maxiter` (positive integer, default 50) and `pnls_tol` (positive,
  finite tolerance for the full proposed parameter update, default `1e-6`).

Fixed and random parameters are updated jointly. A line search checks the
penalized residual error before accepting a step. The tolerance applies to the
full proposed step before any shortening.

Results expose `pnls_converged` separately; overall `converged` requires both
inner and outer convergence. `refit()` and `update()` retain the inner controls
and accept overrides. Bootstrap intervals exclude unconverged refits.

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
- `restart_edge`: Boolean, default `True`. Before accepting a zero variance
  scale, check nearby positive scales for a better likelihood. Restart the
  requested optimizer if a probe improves the objective. Iteration and explicit
  evaluation limits are shared across the original fit and its restarts; a fit
  that exhausts the budget before resolving an improvement reports nonconvergence.
  Set `False` to disable these checks. Interior solutions need no extra evaluations.

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
- `nAGQ0initStep`: Boolean, default `True`. Initialize a joint fit with the
  `nAGQ=0` covariance optimization. `False` starts joint optimization after one
  PIRLS evaluation at the starting covariance parameters. This setting has no
  effect when fitting with `nAGQ=0`.
- `maxiter` applies to each outer optimization stage; `result.n_iter` sums the
  iterations from both stages. Models with no random effects, and Laplace models
  with no fixed effects or a Gaussian identity link, avoid a redundant joint stage.
- `tolPwrss`: Positive finite tolerance for the maximum absolute changes in fixed
  effects and spherical random effects during PIRLS. The default is `1e-7`.
- `pirls_maxiter`: Positive integer limit on inner PIRLS iterations per likelihood
  evaluation. The default, `None`, retains each backend's existing limit: 100 for
  native fitting and 25 for Python fitting. Set an integer to use the same limit
  for both. This is separate from the outer optimizer's `maxiter`.

These inner controls apply to Laplace and adaptive quadrature, including final
coefficient extraction and modular fitting. Results retain `pirls_maxiter` and
`pirls_tol`; `result.refit()` inherits them and accepts overrides with those names.
Refits also retain `nAGQ` and accept `nAGQ0initStep=False` to skip initialization.
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

The modular optimizer uses the same solver dispatch and variance-boundary checks
as `lmer()`. Pass `restart_edge=False` to disable those checks.

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

Fitting requests must use a nonnegative integer. Values above one require a
single random-effect term with one coefficient per group (for example, `(1 | g)`
or `(0 + x | g)`). Multiple random-effect terms and random-intercept/slope blocks
support `nAGQ=0` or `nAGQ=1`. Unsupported requests raise `ValueError` before numerical
optimization. These checks also apply to direct GLMM
fitting and the Python quadrature evaluators.

`optimizeGlmer()` performs joint optimization for `nAGQ>=1` and stores the fixed
coefficients in `opt_result.beta`. `mkGlmerMod()` preserves them when evaluating
the final likelihood. A custom `OptimizeResult` with `beta=None` retains the
theta-only PIRLS extraction behavior.

For custom optimizers, `devfun.get_start(joint=True)` and
`devfun.get_bounds(joint=True)` describe a vector ordered as `[theta, beta]`.
Passing this full vector to `devfun` evaluates the joint likelihood, solving only
for random effects internally. The default start and bounds contain only `theta`;
passing that shorter vector retains the PIRLS objective at the chosen quadrature
order. In particular, `devfun(opt_result.theta)` generally differs from the final
joint deviance; evaluate `devfun(np.r_[opt_result.theta, opt_result.beta])` instead.

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

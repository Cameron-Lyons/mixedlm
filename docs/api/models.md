# Models

This page documents the main model fitting functions and control classes.

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
- `nAGQ`: Number of adaptive Gauss-Hermite quadrature points. 1 = Laplace approximation.
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
result = mlm.nlmer(formula, data, start, control=None)
```

**Parameters:**

- `formula`: Model formula with nonlinear function (e.g., `"y ~ SSasymp(x, Asym, R0, lrc) + (Asym | group)"`)
- `data`: DataFrame
- `start`: Dictionary of starting values for parameters
- `control`: Optional control object

**Returns:** NlmerResult object

**Example:**

```python
from mixedlm.nlme import SSlogis

start = SSlogis.get_start(data, 'y', 'x')
model = mlm.nlmer(
    "y ~ SSlogis(x, Asym, xmid, scal) + (Asym | group)",
    data,
    start=start
)
```

## Control Classes

Control objects configure optimization and convergence settings.

### LmerControl

```python
control = mlm.LmerControl(
    optimizer="L-BFGS-B",
    maxfun=10000,
    tol=1e-6
)
```

**Parameters:**

- `optimizer`: Optimization algorithm. Options: `"L-BFGS-B"` (default), `"BFGS"`, `"Nelder-Mead"`, `"Powell"`, `"bobyqa"`, `"newuoa"`
- `maxfun`: Maximum number of function evaluations
- `tol`: Convergence tolerance

### GlmerControl

```python
control = mlm.GlmerControl(
    optimizer="L-BFGS-B",
    maxfun=10000,
    tol=1e-6,
    nAGQ=1
)
```

**Parameters:**

- Same as LmerControl, plus:
- `nAGQ`: Number of quadrature points (can be overridden in glmer call)

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

### mkLmerDevfun

Create the deviance function for optimization.

```python
devfun = mlm.mkLmerDevfun(parsed_formula)
```

### optimizeLmer

Run the optimizer on the deviance function.

```python
opt_result = mlm.optimizeLmer(devfun, control=None)
```

### mkLmerMod

Create the final model object from optimization results.

```python
model = mlm.mkLmerMod(parsed_formula, opt_result)
```

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
    optimizer="bobyqa",
    maxfun=50000,
    tol=1e-8
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

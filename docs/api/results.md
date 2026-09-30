# Results

This page documents the result objects returned by model fitting functions and their methods.

## Tidy reporting

Linear, generalized, and nonlinear result objects share two reporting methods. The same
functions are also available as `mixedlm.tidy(model, ...)` and `mixedlm.glance(model)`.

### tidy

```python
fixed = result.tidy(conf_int=True)
everything = result.tidy(effects="all", conf_int=True)
```

Return a row-oriented pandas table. `effects` accepts:

- `"fixed"` for estimates, standard errors, test statistics, p-values, and optional
  confidence intervals
- `"ran_pars"` for random-effect standard deviations and correlations
- `"ran_vals"` for group-level conditional modes and conditional standard errors when available
- `"all"` for all three components in one stable schema

For linear mixed models, `ddf_method` accepts `"Satterthwaite"` (the default),
`"Kenward-Roger"`, or `"normal"`. Generalized models use Wald z tests, while nonlinear
models use residual degrees of freedom.

Fixed-effect rows follow the fitted coefficient order, including repeated `term`
labels. For example, a generated categorical contrast and a quoted numeric variable
can both be named `a.1`. Use row positions to distinguish these coefficients;
`tidy(conf_int=True)` preserves both estimates and their individual intervals.

### glance

```python
fit_stats = result.glance()
```

Return one row containing the model type, family, observation and group counts, parameter
counts, residual scale, log likelihood, deviance, AIC, BIC, fit method, convergence state,
singularity state, and iteration count. The schema is common across all model families, so
rows from several fits can be concatenated directly.

## LmerResult

The result object returned by `lmer()`.

Fixed-effect covariance (`vcov()`), prediction standard errors, and leverage
(`hatvalues()`) reuse a factored random-effect precision system. Large systems
use sparse solves, and pointwise variances use bounded batches instead of a
full dense observation-by-random-effect matrix. GLMM covariance and leverage
use the same approach with the final working weights.

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

Dictionary results require unique coefficient names. If labels repeat, use
`result.tidy()` or pair `result.beta` with `result.matrices.fixed_names` by position
(`result.phi` and `result.model.param_names` for nonlinear models). Named interval,
profile, and hypothesis selections reject a requested name that identifies multiple
columns. Unambiguous selections remain available even if other labels repeat.
Rename colliding formula variables before requesting profiles for those coefficients.

#### ranef

```python
result.ranef(condVar=False)
```

Extract random effects (BLUPs).

**Parameters:**

- `condVar`: If True, return a `RanefResult` containing the random effects and
  their per-level conditional variances. The calculation uses sparse block
  extraction, so it does not materialize the full random-effect covariance
  matrix. For GLMMs, it uses the final working weights (including prior weights)
  and sparse random-effect information without computing the dense fixed-effect
  projection. This sparse setup is reused if coefficient covariance or leverage
  is requested later.

**Returns:** A nested dictionary of random-effect arrays, or a `RanefResult`
when `condVar=True`.

#### VarCorr

```python
result.VarCorr()
```

Extract variance-covariance components of random effects.

**Returns:** VarCorr object with variance, standard deviation, and correlation information.

Every random-effect term has its own covariance block. When terms share a
grouping factor, entries receive unique names such as `group`, `group.1`, and
`group.2`; generated names skip any existing grouping-factor names. Each entry's
`grouping_factor` attribute retains the original factor name.

For example, `(1 | group) + (0 + x | group)` produces two covariance entries.
`rePCA()` instead returns one entry for `group`, including the principal
components from both independent blocks. This retains zero-variance components
when checking PCA singularity.

Compound-symmetry and AR(1) structures are reported on their exact fitted covariance scale.
The same structured covariance is used by `rePCA()`, `isSingular()`, and the parameter bounds
returned by `getME("lower")`:

```python
from mixedlm import lmer, set_cov_type

formula = set_cov_type("y ~ time + (time | subject)", "ar1")
result = lmer(formula, data)

print(result.VarCorr())
print(result.rePCA())
print(result.isSingular())
```

#### coef

```python
result.coef()
```

Extract combined coefficients (fixed + random) for each grouping factor.

Every fixed-effect coefficient is repeated across the grouping levels, then the
matching conditional random effect is added. Random-only terms are included with
a zero fixed baseline.

**Returns:** Nested dictionary mapping grouping factors to coefficient names and
their per-level NumPy arrays.

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
lmm_result.predict(
    newdata=None,
    re_form=None,
    allow_new_levels=False,
    se_fit=False,
    interval="none",
    level=0.95,
    offset=None,
    weights=None,
)
glmm_result.predict(
    newdata=None,
    type="response",
    re_form=None,
    allow_new_levels=False,
    se_fit=False,
    interval="none",
    level=0.95,
    offset=None,
)
```

Generate predictions.

Conditional predictions use the fitted coding for random-effect terms, including
interactions, powers, categorical slopes, and custom contrasts. Supply all random-effect
predictors and grouping columns in `newdata`, or use `re_form="NA"` for fixed effects only.
`allow_new_levels=True` accepts unseen grouping levels with zero random effects; unseen
categories of a random-effect predictor still require a fitted encoding and are rejected.

**Parameters:**

- `newdata`: New data for prediction. If None, uses original data.
  Accepts pandas or Polars DataFrames and Polars LazyFrames. A lazy query is
  collected once per call, selecting fixed-effect predictors, requested
  random-effect predictors/grouping columns, and named offset columns. Unused response
  columns and unrelated output expressions are not selected. Fixed-effect-only
  predictions do not select random-effect columns.
- `re_form`: Formula for random effects. Use `"~0"` to exclude random effects.
- `type`: For GLMMs, `"response"` or `"link"`.
- `offset`: Numeric offset for new rows, a scalar, or the name of an offset
  column in `newdata`. GLMM offsets are applied on the link scale. Values must
  be finite and real; complex and masked values are rejected. Arrays and columns
  supply exactly one value per row and use positional order, including pandas
  Series with custom indexes. New-data offsets default to zero. Without
  `newdata`, fitted offsets are already included and an override is not accepted.
- `weights`: LMM residual precision weights for new-data prediction intervals.
  Accepts a positive finite scalar, an array in row order, or a column name.
  Requires `newdata` and `interval="prediction"`; defaults to one.
- `allow_new_levels`: Allow unseen grouping levels and center their random effects at zero.
- `se_fit`: Return pointwise standard errors for the predicted mean.
- `interval`: For LMMs, `"none"`, `"confidence"`, or `"prediction"`; for GLMMs,
  `"none"` or `"confidence"`.
- `level`: Interval coverage strictly between zero and one.

**Returns:** An array, or a `PredictResult` when standard errors or intervals are requested.

GLMM predictions validate `type` and `interval` before building prediction
matrices or calculating covariance. Confidence intervals use the link's
`inverse_interval()` transformation: decreasing inverse links have ordered
response bounds, and a square-root inverse includes zero when the link-scale
interval crosses zero. GLMM standard errors still use fixed-coefficient
uncertainty and the delta method on the response scale.

Fixed-coefficient covariance projections are evaluated in batches for both model
types, bounding each temporary projection to one million elements (or one row
when the fitted coefficient count exceeds that limit).

Lazy query filters and ordering are preserved, so array offsets follow the
resulting row order. The collected frame is reused for all prediction work in
that call and is not cached on the fitted model. Intercept-only predictions with
no required columns collect a row-count query, preserving empty and nonempty
grids; expressions needed to determine that count may still be evaluated.
The selected frame and returned predictions remain in memory.

New-data predictions use the fitted fixed-effect column order and omit columns
removed by rank checks. Distinct formula columns can share a display name (for example,
a categorical contrast `a.1` and a quoted numeric variable named `a.1`); their fitted
positions distinguish them during prediction and refitting. Older or manually constructed
results without that position information raise an error when a reduced schema is
ambiguous; refit those models before predicting new data.

For conditional LMM predictions, uncertainty is evaluated from the joint fixed- and
random-effect covariance. This includes covariance between fixed and random estimates,
covariance among correlated random slopes, and covariance across crossed structures. For an
unseen group accepted with `allow_new_levels=True`, the fitted prior covariance is added while
the predicted random effect remains zero. Prediction intervals add residual variance to the
mean-prediction variance; `se_fit` continues to report the standard error of the mean.
The covariance calculation uses the fitted prior weights. In-sample prediction intervals
add residual variance `sigma**2 / weight`. New-data prediction intervals add
`sigma**2 / weights`, using the supplied prediction weights or one by default.
Prediction weights must use the same scale as the fitted prior weights. They change
the residual variance in prediction intervals; predicted means and their `se_fit`
values are unaffected. Repeated uncertainty calculations reuse the fitted weighted
factorization.

```python
mean_ci = result.predict(newdata, interval="confidence", level=0.95)
future_pi = result.predict(newdata, interval="prediction", level=0.95)

# Allow different residual variances for future observations.
# newdata["precision"] contains positive weights on the training-weight scale.
weighted_pi = result.predict(newdata, interval="prediction", weights="precision")

new_groups = result.predict(
    new_group_data,
    allow_new_levels=True,
    interval="prediction",
)
```

#### simulate

```python
result.simulate(nsim=1, seed=None, use_re=True, re_form=None)
```

Simulate responses from the fitted model, including its offsets and random-effect
covariance structure (unstructured, diagonal, compound symmetry, or AR(1)). LMM
residuals have standard deviation `sigma / sqrt(weight)` for each observation.
These weights do not rescale the random effects.

**Parameters:**

- `nsim`: Positive integer number of simulations.
- `seed`: Integer seed, NumPy `RandomState` or `Generator`, or `None` for a fresh stream.
- `use_re`: Include newly sampled random effects (default `True`).
- `re_form`: `"~0"` or `"NA"` excludes random effects.

**Returns:** A vector of shape `(n_obs,)` for one simulation, or an array of shape
`(n_obs, nsim)` for multiple simulations.

For linear and generalized models, simulation leaves NumPy's global random state
unchanged. Pass an integer seed for repeatable calls, or reuse a stream to continue
drawing new samples:

```python
import numpy as np

rng = np.random.default_rng(42)
first = result.simulate(nsim=10, seed=rng)
next_batch = result.simulate(nsim=10, seed=rng)
```

An integer seed preserves the previous draw sequence for the same backend and
call shape. Batch sizes and native-backend availability can affect the sequence.
Calling `np.random.seed()` separately no longer controls these simulations; pass
`seed` explicitly instead. Custom family `simulate(mu, rng=...)` methods should
use the supplied stream for their response draws.

Grouped-binomial GLMM simulations return success counts.

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

**Returns:** Numeric `LogLik` value with `value`, `df`, `nobs`, and `REML`
metadata. It can be used directly in arithmetic and NumPy operations.

#### AIC / BIC

```python
result.AIC()
result.BIC()
```

Compute information criteria.

**Returns:** Float value.

#### profile

```python
result.profile(which=None, n_points=20, level=0.95)
```

Compute fixed-effect likelihood profiles. LMM profiles re-optimize covariance
parameters, other fixed coefficients, and residual scale using ML, including for
REML inputs. The returned center records the ML estimate. GLMM profiles re-optimize nuisance
fixed coefficients and covariance parameters using the fitted quadrature and
inner solver controls. The refined profile center can differ from the fitted
coefficient, particularly for `nAGQ=0` fits, which are profiled using the joint
Laplace likelihood. Default joint fits usually retain their center within
optimization tolerance; the original result is unchanged. `n_points` must be at least
3 and affects the plotted curve, not the interval endpoint accuracy. See
[profile likelihood](inference.md#profile-likelihood) for convergence behavior.

**Returns:** Dictionary mapping parameter names to `ProfileResult` objects.

#### as_function

`result.as_function(type="deviance")` reconstructs the GLMM objective with its
quadrature and inner solver controls. For `result.joint_fit=True`, a full vector
`np.r_[result.theta, result.beta]` evaluates the joint likelihood. A theta-only
vector holds the fitted beta fixed; it does not re-optimize nuisance coefficients.
For `joint_fit=False`, the callable retains the theta-only PIRLS objective.

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

Requesting `"RZX"` materializes a dense random-effect Cholesky factor on demand.
It retains the original coefficient order and is cached for subsequent calls.
LMM fixed-effect profiling reuses the precision solver without requesting this
dense factor.

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

### predict

```python
nlmm_result.predict(newdata=None, x_var=None, group_var=None, offset=None)
```

With no `newdata`, return fitted responses including the fitted offsets.
For new observations, `x_var` defaults to the predictor used for fitting.
Supply `group_var` to add fitted random effects for known groups; unknown groups
receive population-level predictions. Omitting `group_var` requests
population-level predictions.

`offset` accepts a finite real scalar, a one-dimensional array with one value per
new row, or the name of a column in `newdata`. It is added to the nonlinear
response mean after applying any group effects. New-data offsets default to
zero and do not reuse the fitted observation offsets. Explicit offsets require
`newdata`; complex, masked, missing, and infinite offsets are rejected.

```python
predictions = nlmm_result.predict(
    newdata,
    group_var="subject",
    offset="known_shift",
)
```

### confint

`result.confint(method="boot", n_boot=1000, seed=42)` returns percentile
intervals using the same samples and failure handling as `bootstrap_nlmer()`.
`n_boot` must be a positive integer. `parm` selects one name or a list of names;
unknown names are omitted.

Pass `n_jobs=2` to refit bootstrap samples in two worker processes, or `-1` for
available CPUs. The same integer seed produces the same samples across worker
counts. `seed` also accepts a reusable NumPy `RandomState` or `Generator`.

Failed simulations or refits and refits with nonfinite or incorrectly shaped
estimates are excluded from all bootstrap components. They are never replaced
with the original fitted estimates. When all replicates fail, confidence bounds
are `NaN`. Use `bootstrap_nlmer(result, n_boot=1000, seed=42)` or
`bootMer(result, nsim=1000, seed=42)` to inspect `n_failed` and sample arrays.

### simulate

```python
import numpy as np

draws = nlmm_result.simulate(nsim=100, seed=42)

stream = np.random.default_rng(42)
first = nlmm_result.simulate(seed=stream)
more = nlmm_result.simulate(nsim=10, seed=stream)
```

Nonlinear simulation accepts an integer seed, NumPy `RandomState`, or NumPy
`Generator`. Integer seeds retain the previous draw sequence. A supplied stream
advances across calls; omitting `seed` creates an independent stream. These
calls do not reset or consume NumPy's global random state. Use `seed` or an
explicit stream for reproducibility instead of calling `np.random.seed()`.

`nsim` must be a nonnegative integer. One draw returns shape `(n_obs,)`, multiple
draws return `(n_obs, nsim)`, and zero draws return `(n_obs, 0)`. Simulation
preserves fitted offsets and inverse-weight residual variances. `use_re=False`,
`re_form="NA"`, and `re_form="~0"` exclude random effects.

For many groups, simulation prepares group rows with one stable ordering
instead of repeatedly scanning all observations for every group. Multi-draw
calls reuse these rows, the random-effect covariance transform, offsets, and residual scales.
Fixed-only calls evaluate the nonlinear mean once. This setup is local to each
call, so changes to a result are reflected in the next simulation.

## VarCorr

Variance-covariance structure of random effects.

### Attributes

- `groups`: Dictionary mapping unique report names to `VarCorrGroup` entries
- `residual`: Residual variance for LMMs

Each `VarCorrGroup` entry contains:

- `name`: Unique report name
- `grouping_factor`: Original grouping factor, before any report-name suffix
- `term_names`: Ordered random-effect coefficient names
- `variance`: Dictionary of coefficient variances
- `stddev`: Dictionary of coefficient standard deviations
- `cov`: Covariance matrix for this term
- `corr`: Correlation matrix, or `None` for independent coefficients

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

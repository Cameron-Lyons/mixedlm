# Results

This page documents the result objects returned by model fitting functions and their methods.
The examples use a sleepstudy fit:

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
```

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
(`hatvalues()`) share one factored random-effect precision system, so requesting
several of them does not refactor the model. GLMMs use the final working weights.

Besides the estimates (`beta`, `theta`, and `sigma`), a result records
`optimizer`, the method whose estimates were kept, and `control`, the
`LmerControl` used for fitting. `update()`, `drop1()`, and `allFit()` refit with
that control; `refit()` and `refitML()` keep its `use_rust` setting and use the
`"auto"` optimizer unless `method=` is given. The statsmodels-style aliases
`fe_params`, `re_params`, `fittedvalues`, and `resid` are deprecated; use `beta`,
`theta`, `fitted()`, and `residuals()`.

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
  their per-level conditional variances, without forming the full random-effect
  covariance matrix. GLMMs use the final working weights, including prior weights.

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

```py
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

```py
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
Conditional predictions require nonmissing grouping values, including when new levels are allowed.

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
interval crosses zero. GLMM response-scale standard errors use the delta method.

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

For conditional LMM and GLMM predictions, uncertainty is evaluated from the joint fixed- and
random-effect covariance. This includes covariance between fixed and random estimates,
covariance among correlated random slopes, and covariance across crossed structures. For an
unseen group accepted with `allow_new_levels=True`, the fitted prior covariance is added while
the predicted random effect remains zero. GLMMs use the final PIRLS working approximation
and hold fitted covariance parameters fixed. LMM prediction intervals add residual variance to the
mean-prediction variance; `se_fit` continues to report the standard error of the mean.
The covariance calculation uses the fitted prior weights. In-sample prediction intervals
add residual variance `sigma**2 / weight`. New-data prediction intervals add
`sigma**2 / weights`, using the supplied prediction weights or one by default.
Prediction weights must use the same scale as the fitted prior weights. They change
the residual variance in prediction intervals; predicted means and their `se_fit`
values are unaffected. Repeated uncertainty calculations reuse the fitted weighted
factorization.

```python
import pandas as pd

newdata = pd.DataFrame({"Days": [0.0, 5.0], "Subject": ["308", "309"], "precision": [1.0, 0.5]})
new_group_data = pd.DataFrame({"Days": [0.0, 5.0], "Subject": ["new", "new"]})

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
`seed` explicitly instead.

GLMM responses are drawn by the family's
`simulate(mu, rng=None, *, weights=None, trials=None)` method, which should use
the supplied stream. Families without a response distribution, such as quasi
families and custom families that do not implement `simulate()`, raise
`NotImplementedError`; see [custom families](families.md#customfamily).

Grouped-binomial GLMM simulations return success counts.

#### confint

```python
result.confint(method="Wald", level=0.95)
```

Compute confidence intervals.

**Parameters:**

- `method`: CI method. Options: `"Wald"`, `"profile"`, `"boot"`.
- `level`: Confidence level.

**Returns:** Dictionary mapping parameter names to `(lower, upper)` tuples.

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

```py
result.profile(which=None, n_points=20, level=0.95, n_jobs=1)
```

Compute fixed-effect likelihood profiles; `which=None` profiles every fixed
coefficient. LMM profiles re-optimize covariance
parameters, other fixed coefficients, and residual scale using ML, including for
REML inputs. The returned center records the ML estimate. GLMM profiles re-optimize nuisance
fixed coefficients and covariance parameters using the fitted quadrature and
inner solver controls. The refined profile center can differ from the fitted
coefficient, particularly for `nAGQ=0` fits, which are profiled using the joint
Laplace likelihood. Default joint fits usually retain their center within
optimization tolerance; the original result is unchanged. `n_points` must be at least
3 and affects the plotted curve, not the interval endpoint accuracy. LMM profiles
accept `n_jobs` to profile coefficients in worker processes; GLMM profiles run
serially. See [profile likelihood](inference.md#profile-likelihood) for
convergence behavior.

**Returns:** Dictionary mapping parameter names to `ProfileResult` objects.

#### as_function

`result.as_function(type="deviance")` reconstructs the GLMM objective with its
quadrature and inner solver controls. For `result.joint_fit=True`, a full vector
`np.r_[result.theta, result.beta]` evaluates the joint likelihood. A theta-only
vector holds the fitted beta fixed; it does not re-optimize nuisance coefficients.
For `joint_fit=False`, the callable retains the theta-only PIRLS objective.

#### drop1

```python
result.drop1(data, test="Chisq", n_jobs=1)
```

Test single term deletions. LMM refits reuse the fitted control.
`n_jobs` refits the reduced models in worker processes; see
[parallel execution](inference.md#parallel-execution).

**Returns:** Drop1Result with test statistics.

#### allFit

```python
result.allFit(data, optimizers=None, verbose=False, n_jobs=1)
```

Refit the model with each optimizer, keeping the other control settings. The
default list is every solver from `mixedlm.estimation.available_optimizers()`.
`n_jobs` runs the refits in worker processes.

**Returns:** AllFitResult comparing optimizer results. `summary` tabulates
each fit, `best_fit()` returns the lowest-deviance fit, and `is_consistent()`
checks whether the converged fits reach the same deviance.

#### getME

```python
result.getME("theta")
```

Extract model components.

**Parameters:**

- `name`: Component name. Options include `"X"`, `"Z"`, `"theta"`, `"Lambda"`, `"Zt"`, `"beta"`, `"b"`, `"u"`, `"devcomp"`, etc.

**Returns:** The requested component. As in lme4, `"b"` holds the conditional
modes of the random effects and `"u"` the spherical random effects, with
`b = Lambda @ u`; for singular fits `u` is the minimum-norm solution.

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

```py
result.family
```

The distribution family used.

## NlmerResult

The result object returned by `nlmer()`. Has similar methods to LmerResult.

### predict

```py
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

```py
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

```py
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

Each call prepares its own simulation setup, so changes to a result are
reflected in the next simulation.

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

```text
Random effects:
 Groups      Name           Variance   Std.Dev.   Corr
 Subject     (Intercept)    612.0901    24.7405
             Days            35.0717     5.9221   0.07
 Residual                   654.9410    25.5918
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

# Random effects for the first subjects
import pandas as pd

ranef = result.ranef()
print(pd.DataFrame(ranef["Subject"]).head())

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

# Include random effects; new subjects need allow_new_levels and get zero
pred_cond = result.predict(newdata=new_data, allow_new_levels=True)

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
print(f"Converged: {conv.converged}")
```

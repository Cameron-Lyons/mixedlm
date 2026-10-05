# Inference

This page documents functions for statistical inference, hypothesis testing, and confidence intervals.

Confidence levels must lie strictly between zero and one. Marginal means,
adjusted effects, parameter intervals, prediction intervals, and profile builders
validate finite real levels before model calculations or resampling. Normal and
Student's t interval cutoffs use the tail probability directly, preserving finite
cutoffs for valid levels close to one. A finite cutoff does not guarantee finite
endpoints when the model itself has undefined uncertainty or a response
transformation overflows.

## Parallel Execution

Bootstrap (`bootMer()`, `bootstrap_lmer()`, `bootstrap_glmer()`,
`bootstrap_nlmer()`, and nonlinear `confint(method="boot")`), term deletion
(`drop1()`), optimizer comparison (`allFit()`), LMM likelihood profiles
(`profile()`), `slice2D()`, and `cross_validate()` accept `n_jobs`:

- `n_jobs=1`, the default, runs everything in the calling process.
- A positive integer sets the number of worker processes, and `-1` uses every
  available CPU. The count is capped at the number of tasks, and a single task,
  such as one profiled coefficient, runs in the calling process.
- `0`, other negative values, booleans, and non-integers raise `TypeError` or
  `ValueError`.

Results match a serial run, and seeded bootstrap samples are the same for every
worker count. Worker processes never fork the calling process: they start
through forkserver on Linux and spawn on macOS and Windows. Scripts must
therefore start parallel work under an `if __name__ == "__main__":` guard, and
custom families, nonlinear models, and other arguments sent to workers must be
importable and picklable:

```py
import mixedlm as mlm


def main():
    data = mlm.load_sleepstudy()
    model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
    boot = mlm.bootMer(model, nsim=1000, seed=42, n_jobs=-1)
    print(mlm.bootCI(boot))


if __name__ == "__main__":
    main()
```

Each worker starts with one BLAS and OpenMP thread, so parallelism comes from the
number of workers rather than oversubscribed threads. Values you set yourself in
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, or `RAYON_NUM_THREADS` are kept. On Linux, mixedlm sets
the interpreter's forkserver preload list to `["mixedlm.inference"]`, replacing a
list set with `multiprocessing.set_forkserver_preload()`. A forkserver started by
mixedlm keeps these thread limits and preloaded modules for later forkserver
pools in the same interpreter, including your own.

Starting workers has a fixed cost that can outweigh the gain for small jobs.
A conditional `slice2D()` stays serial unless its remaining rows would take
about a second or more. GLMM profiles run serially and reject `n_jobs` other
than 1.

Native code is also safe in processes you fork yourself, for example with
`os.fork()` or a multiprocessing `"fork"` pool: a forked child runs native
kernels on a single thread instead of hanging.

## Linear Hypotheses

### linear_hypothesis

Test arbitrary linear restrictions on fixed-effect coefficients. The null
hypothesis is expressed as $C\beta = r$, where $C$ contains one or more
constraint rows and $r$ is supplied with `rhs`.

```py
from mixedlm.inference import linear_hypothesis

# H0: the x and z slopes are equal
test = linear_hypothesis(model, {"x": 1, "z": -1})
print(test)
```

Named rows test several restrictions jointly while retaining readable labels:

```py
test = linear_hypothesis(
    model,
    {
        "equal slopes": {"x": 1, "z": -1},
        "target sum": {"x": 1, "z": 1},
    },
    rhs=[0, 2],
)

test.statistic    # Joint Wald statistic
test.p_value      # Joint p-value
test.table        # Row-level estimates, SEs, intervals, and p-values
```

Numeric arrays are interpreted in `model.matrices.fixed_names` order. A pandas
DataFrame can instead use coefficient names as columns and row labels as its
index. Constraint rows must be finite, real, nonzero, linearly independent, and
estimable from the fitted covariance matrix.

The calculation scales restriction equations and coefficient uncertainty
internally, preserving the test when equivalent equations use different units.
Unused coefficient columns are omitted from the calculation, and a single
restriction avoids matrix rank checks and factorization. Complex-valued inputs
and masked elements are rejected before covariance calculation.

Returned estimates, differences, standard errors, confidence limits, and covariance
use the original restriction units. At extreme scales, an output value can
underflow to zero or overflow to infinity while the row and joint tests remain
finite. Scaling cannot recover precision already lost in the supplied arrays.

The default joint test is an F test for linear mixed models and a chi-square
test for generalized linear mixed models. An F test uses residual denominator
degrees of freedom unless `denominator_df` is provided explicitly.

**Parameters:**

- `model`: Fitted linear or generalized linear mixed model
- `hypothesis`: Constraint matrix, named weights, named rows, or DataFrame
- `rhs`: Scalar null value or one value per row; defaults to zero
- `labels`: Optional replacement row labels
- `test`: `"auto"`, `"F"`, or `"Chisq"`
- `denominator_df`: Optional positive denominator DF for an F test
- `level`: Confidence level for row-level intervals

**Returns:** `LinearHypothesisResult`
## Cross-Validation

### cross_validate

Refit an LMM or GLMM across exhaustive folds and score aligned out-of-fold
predictions.

```py
import mixedlm as mlm

model = mlm.lmer("y ~ x + (1 | subject)", data)

# Case-level validation: held-out rows can use random effects estimated from
# other training rows for the same subject.
case_cv = mlm.cross_validate(model, cv=5, random_state=42)

# Cluster-level validation: each subject appears in exactly one test fold.
# Predictions use fixed effects because held-out subjects are unseen.
group_cv = mlm.cross_validate(
    model,
    cv=5,
    group="subject",
    metrics=["rmse", "mae", "r2"],
    random_state=42,
)

print(group_cv)
print(group_cv.scores)
print(group_cv.fold_scores)
```

**Parameters:**

- `model`: Fitted `LmerResult` or `GlmerResult`
- `data`: Optional aligned data; the stored clean model frame is used by default.
  Modeled predictor and grouping values, row order, and categorical encoding must
  match the fit. Additional columns can define external holdout groups.
- `cv`: Number of folds (default 5), or an iterable of `(train_indices, test_indices)`
  pairs or `CrossValidationFold` objects
- `group`: Optional column defining whole clusters to hold out
- `metrics`: Metric name, callable, or sequence; defaults are selected by model type
- `shuffle`, `random_state`: Reproducible fold assignment controls
- `re_form`: Random-effect prediction mode; `"auto"` uses fixed effects for grouped folds
- `n_jobs`: Worker processes for the fold refits, or `-1` for all CPUs (default 1).
  Results match a serial run. See [parallel execution](#parallel-execution) for
  the `__main__` guard; custom families and `fit_kwargs` values must be importable
  and picklable. Warnings from refits in workers are not repeated in the calling
  process; the `converged` and `singular` fold columns record each refit.
- `fit_kwargs`: Additional refit options such as optimizer controls

Built-in metrics are weighted `"mse"`, `"rmse"`, `"mae"`, `"r2"`, and GLMM
`"deviance"`. A custom metric receives `(y_true, y_pred, weights)` and returns
one finite scalar. Original model weights and offsets are automatically subset
and preserved in every fold.

Explicit folds use zero-based row positions, independent of dataframe index labels.
Their test sets must cover every fitted observation exactly once. Train and test
sets must be nonempty, contain unique integer positions, and be disjoint. Training
sets can omit additional observations to create buffers around held-out blocks.
With `group`, each whole group must occur in one test fold, and no train/test pair
can share a group. Invalid partitions are rejected before any model is refitted.
`shuffle` and `random_state` affect generated folds only.

Reuse a partition to compare model specifications on identical held-out observations:

```py
folds = mlm.make_folds(len(data), cv=5, groups=data["subject"], random_state=42)
first_cv = mlm.cross_validate(first_model, cv=folds, group="subject")
second_cv = mlm.cross_validate(second_model, cv=folds, group="subject")
```

Train/test iterables produced by external splitters work directly when their test
sets form an exhaustive partition. Splits with overlapping or incomplete test
coverage are rejected: this API returns one out-of-fold prediction per fitted row.

The result exposes:

- `scores`: Overall metrics computed from every out-of-fold prediction
- `fold_scores`: Per-fold sizes, convergence and singularity flags, and metrics
- `predictions`: Finite predictions aligned to the fitted observations
- `fold_ids` and `folds`: Fold membership and explicit train/test positions
- `summary()`: Overall, fold-mean, fold-SD, minimum, and maximum scores
- `all_converged`: Whether every refit reported convergence
- `any_singular`: Whether any refit is on a random-effects boundary

### make_folds

Construct folds without fitting a model:

```py
folds = mlm.make_folds(
    len(data),
    cv=5,
    groups=data["subject"],
    random_state=42,
)
```

Grouped folds never split a cluster. Groups are assigned by decreasing size to
the currently smallest fold, keeping test observation counts approximately
balanced without adding a machine-learning dependency.

### Weighted scoring helpers

The vectorized scoring functions are also public:

```py
rmse = mlm.weighted_rmse(y_true, y_pred, weights)
mse = mlm.weighted_mse(y_true, y_pred, weights)
mae = mlm.weighted_mae(y_true, y_pred, weights)
r2 = mlm.weighted_r2(y_true, y_pred, weights)
```

Inputs must be aligned, finite, unmasked real vectors with strictly positive weights.
Scores remain stable across extreme response and weight units by scaling intermediate
products. RMSE can remain finite even when its squared value exceeds floating-point range;
MSE returns infinity when the final squared score is unrepresentable. R² retains small
weights attached to large observations, and constant responses score one for exact
predictions and zero otherwise. Fold means and sample standard deviations also avoid
overflow from intermediate sums and squares.

## Model Comparison

### model_selection

Rank candidate mixed models with AIC, small-sample corrected AIC, or BIC:

```py
ranking = mlm.model_selection(
    model1,
    model2,
    model3,
    names=["baseline", "linear", "interaction"],
    criterion="AICc",
)

print(ranking.to_dataframe())
best = ranking.best_model
supported = ranking.evidence_set(threshold=0.95)
```

Models are returned from strongest to weakest support. The result includes log-likelihood,
parameter count, AIC, AICc, BIC, criterion deltas, relative likelihoods, normalized weights,
and cumulative weights. Candidate models must use the same observations, response, likelihood
class, and generalized family. Linear models should be fit with `REML=False`; set
`allow_reml=True` only when every candidate has the same fixed-effects specification.

### anova

Likelihood ratio tests between nested models.

```py
import mixedlm as mlm

result = mlm.anova(model1, model2, ...)
```

**Parameters:**

- `*models`: Two or more fitted models to compare

**Returns:** AnovaResult with chi-squared test statistics and p-values

**Example:**

```py
m1 = mlm.lmer("y ~ x + (1 | g)", data, REML=False)
m2 = mlm.lmer("y ~ x + z + (1 | g)", data, REML=False)
print(mlm.anova(m1, m2))
```

### anova_type3

Type III ANOVA for a single model.

```py
result = mlm.anova_type3(model)
```

**Returns:** AnovaType3Result with F-statistics and p-values for each fixed effect

**Example:**

```py
model = mlm.lmer("y ~ a * b + (1 | g)", data)
print(mlm.anova_type3(model))
```

### allFit

Fit a linear mixed model with several optimizers and compare the results.

```py
comparison = mlm.allFit(
    "Reaction ~ Days + (Days | Subject)",
    data,
    optimizers=["COBYQA", "L-BFGS-B", "Nelder-Mead"],
    control=mlm.LmerControl(maxiter=5000),
    n_jobs=1,
)
print(comparison.summary)
```

`optimizers` defaults to `["COBYQA", "Nelder-Mead", "L-BFGS-B"]`. `control`
applies to every fit, with only the optimizer replaced, and `n_jobs` runs the
fits in worker processes. The `allFit()` method of a fitted LMM or GLMM refits
that model with every installed solver instead; see
[results](results.md#allfit). `AllFitResult.is_consistent()` checks whether the
converged fits reach the same deviance, and `best_fit()` returns the fit with the
lowest deviance.

## Degrees of Freedom

### satterthwaite_df

Compute Satterthwaite denominator degrees of freedom.

The variance calculation uses relative uncertainty so that changing response
units (for example, milliseconds to seconds) preserves the degrees of freedom
and p-values, apart from numerical fitting tolerance.

```py
df = mlm.satterthwaite_df(model)
by_coefficient = df.as_dict()
```

**Returns:** `DenomDFResult` containing `df`, `method`, and `param_names`.
Use `df["coefficient_name"]` to retrieve one value or `df.as_dict()` to obtain a
dictionary.

### kenward_roger_df

Compute Kenward-Roger denominator degrees of freedom.

```py
df = mlm.kenward_roger_df(model)
```

**Returns:** `DenomDFResult`, with the same accessors as `satterthwaite_df`.

### pvalues_with_ddf

Compute p-values using denominator degrees of freedom.

```py
pvals = mlm.pvalues_with_ddf(model, method="Satterthwaite")
```

**Parameters:**

- `model`: Fitted model
- `method`: `"Satterthwaite"` or `"Kenward-Roger"`

**Returns:** Dictionary mapping coefficient names to `(estimate, t_value, p_value)`
tuples.

## Estimated Marginal Means

### ggpredict

Compute adjusted fixed-effect predictions for continuous variables, factors,
or their Cartesian product. Numeric variables outside the requested grid are
held at their mean and factors at their reference level.

```py
predictions = mlm.ggpredict(
    model,
    ["Days", "treatment"],
    at={"Days": [0, 5, 10]},
)
```

The returned data frame contains the requested grid columns plus `predicted`,
`std.error`, `conf.low`, and `conf.high`. For GLMMs, `type="response"` builds
the confidence interval on the link scale before transforming its bounds with
`Link.inverse_interval()`. Decreasing inverse links return ordered response
bounds, and square-root links include zero when the interval crosses zero.
Use `type="link"` to keep results on the linear-predictor scale.

Effect grids follow the Cartesian-product order with the last requested variable
changing fastest. Predictions are computed in bounded batches, but the returned
data frame grows with the number of combinations. The same applies to each grid
returned by `allEffects()`.

Both adjusted-effect functions default to `offset=None`, using the unweighted
mean of the model's fitted link-scale offsets after missing-value omission.
Set `offset=0` to request per-unit rates for a count model with a log-exposure
offset, or pass another finite scalar to choose a reference exposure. This
changes the previous default of zero for models fitted with nonzero offsets.
Offsets are treated as known and add no coefficient uncertainty.

`ggpredict()` also accepts one finite offset per returned grid row, in the
Cartesian order of `terms` with the last term varying fastest. Values are
positional, including when passed as a pandas Series; they are not recycled.
For example, `offset=np.log([10, 20, 30])` requests three different exposures
for a three-row grid. The result's `attrs["offset"]` records the resolved scalar
or an immutable tuple of row offsets.

### allEffects

Compute a separate adjusted prediction grid for every fixed-effect variable.

```py
effects = mlm.allEffects(model, n_points=25)
days_effect = effects["Days"]
```

When a model has multiple fixed-effect variables, each variable also conditions
the other grids. Therefore, `at` must supply just one value per variable, either
as a scalar or a one-element iterable. Use `ggpredict()` with all relevant terms
for a joint grid with several values per variable. Invalid options and unknown
`at` variables raise an error even when the model has no fixed-effect predictors
and `allEffects()` would otherwise return an empty dictionary.

`allEffects()` uses the same scalar offset for every grid. Call `ggpredict()`
separately when different grids require different row offsets.

Both functions are batched, use the fitted fixed-effect covariance matrix, and
require no plotting package. Prediction grids automatically reuse fitted
categorical contrasts, category order, and retained fixed-effect columns. Sum,
Helmert, polynomial, and custom contrasts need no repeated configuration. Factors
outside the requested grid are held at the first fitted category, including when
their source data uses a different category order.

The optional `contrasts=` mapping remains available as an explicit override; it
should match the fitted coefficient parameterization. For example:

```py
model = mlm.lmer("yield ~ treatment * dose + (1 | block)", data,
                 contrasts={"treatment": "sum"})
predictions = mlm.ggpredict(model, "treatment")
effects = mlm.allEffects(model)
```

Grid calculations read the fitted pandas frame without copying or modifying it.
Returned prediction tables are independent of that frame.

Polars models are supported as well; category order and missing values are
preserved, and returned prediction tables can be edited independently of the
fitted data.

### emmeans

Compute estimated marginal means.

Factors outside `specs` are averaged equally over their reference-grid levels;
`at` can restrict those levels. Grid reduction preserves the order of `specs`
and each factor's levels.

```py
em = mlm.emmeans(model, "treatment", type="response")
```

**Parameters:**

- `model`: Fitted model
- `specs`: Fixed-effect predictor name or names to compute marginal means for;
  numeric predictors are supported, and `[]` requests a grand mean
- `by`: Optional predictor name or names defining separate comparison groups
- `offset`: A finite scalar or one-dimensional sequence with one value per result
  row; `None` (default) uses the unweighted mean of the fitted link-scale offsets
  after missing-value omission
- `at`: Reference values for fixed-effect predictors, given as scalars or nonempty
  one-dimensional sequences of distinct values; unknown names and missing or
  nonfinite numeric values raise an error
- `cov_reduce`: Function used to reduce numeric covariates (default: mean)
- `type`: `"response"` (default) or `"link"` for generalized models
- `level`: Confidence level (default: `0.95`)

**Returns:** Emmeans object

GLMM intervals and comparisons use an asymptotic normal reference, reported as
`df=inf`. Their comparison tables label the statistic `z.ratio`; the result's
`t_ratio` array retains its existing name. LMM comparisons use Student's t
reference with residual degrees of freedom. GLMM contrasts remain on the link
scale even when `type="response"` displays back-transformed marginal means.

Undefined comparison p-values print as `nan`. Small p-values use scientific
notation or `< 2e-16`, as in model summaries.

All combinations of the averaged factors contribute with equal weight, using the
fitted categorical encoding.

Pass `specs=[]` to return a single overall mean, averaged over all factor levels.

**Methods on Emmeans object:**

- `pairs(adjust="tukey")`: Compute all pairwise comparisons
- `contrast(method, adjust=None)`: Compute pairwise, treatment-vs-control, or custom contrasts

Contrast tests retain very small p-values when computing two-sided Student's
t probabilities. Holm and FDR adjustments preserve the input comparison order
and keep undefined p-values as `NaN`; the adjustment count includes all
comparisons in the supplied family.

Contrast results provide `confint()` for a table containing the contrast label,
estimate, standard error, degrees of freedom, and `lower`/`upper` confidence bounds:

```py
comparisons = em.pairs(adjust="tukey", level=0.90)
intervals = comparisons.confint()  # uses the requested 90% confidence level
pointwise = comparisons.confint(level=0.95, adjust="none")
```

The table preserves comparison order and reports its confidence level and actual
interval adjustment in `intervals.attrs["level"]` and `intervals.attrs["adjust"]`.
Overrides affect only the returned intervals. The estimates, p-values, and stored
defaults remain unchanged. Interval calculations are performed on request and
reuse scalar critical values for families with the same settings.

Unadjusted intervals use the result's Student's t reference, or the normal
reference when `df=inf`. Tukey intervals use the studentized range with the
number of means in the pairwise family; treatment-versus-control differences
can also use this adjustment. Custom rows with exactly two nonzero, opposite
coefficients are recognized as scaled pairwise differences and support Tukey
tests and intervals. The family includes all marginal means represented by
the matrix columns, even when only a subset of pairs is requested. General
custom linear combinations reject Tukey and can use other adjustments instead.
Holm, FDR/BH, and the current Dunnett approximation
use Bonferroni intervals; both the requested and actual methods are recorded
in the table's attributes. The Holm/FDR fallback follows the
[emmeans interval convention](https://rvlenth.github.io/emmeans/reference/summary.emmGrid.html#p-value-adjustments).

Intervals remain on the linear predictor scale, including when marginal means
were displayed with `type="response"`. Tables can be edited independently of
the contrast result, and empty contrast sets return empty tables.

Custom coefficients accept rectangular two-dimensional arrays, nested lists,
or data frames, with one row per comparison and one column per marginal mean
in the comparison family. Malformed shapes, complex values, and nonfinite or
masked coefficients raise clear errors before covariance calculations. The
legacy `"dunnett"` option remains a Bonferroni approximation and counts the
comparisons actually requested; it does not compute the exact Dunnett
distribution.

Multiplying a custom contrast row by a nonzero constant does not change its test.
Estimates and standard errors are returned in the requested units.

**Example:**

```py
model = mlm.lmer("yield ~ treatment + (1 | block)", data)
em = mlm.emmeans(model, "treatment")
print(em)
print(em.pairs())
```

Omitting `adjust` or passing `None` to `contrast()` uses Tukey for `"pairwise"`
and no adjustment for the other contrast methods. An explicit `adjust="none"`
always requests unadjusted p-values. `pairs()` continues to default to Tukey.

Adjustment names ignore case and surrounding whitespace. Supported names are
`"none"`, `"bonferroni"`, `"holm"`, `"fdr"`, `"tukey"`, and `"dunnett"`;
`"BH"` is an alias for `"fdr"`. Results report the canonical name. The existing
`"dunnett"` option uses a Bonferroni approximation. Unknown names raise an error.


Numeric predictors use `cov_reduce` unless `at` overrides their values. Every
requested value enters the reference grid. Predictors in `specs` or `by` identify
separate result rows; other grid dimensions are averaged with equal weights.
Categorical levels follow the fitted order unless `at` supplies an explicit order.

```py
# Compare treatments separately at each requested dose
model = mlm.lmer("yield ~ treatment * dose + (1 | block)", data)
em = mlm.emmeans(model, "treatment", by="dose", at={"dose": [0, 5, 10]})
print(em)
comparisons = em.pairs(adjust="holm")
print(comparisons)
print(comparisons.grid)  # One row of grouping values per comparison
```

Pairwise, treatment-vs-control, and custom contrasts operate separately within each
`by` group. P-value adjustments apply to each group's comparison family. Custom
contrast matrices need one column per mean **within a group**, in the displayed
order, and the same matrix is applied to every group. Contrast labels include the
group values, also available in `ContrastResult.grid`; this field is `None` for
ungrouped comparisons. For generalized models, comparisons remain on the link
scale even when the displayed means use `type="response"`.

Offsets enter means and contrasts on the link scale. They are treated as known,
so they do not change link-scale standard errors. Response-scale means, standard
errors, and confidence limits incorporate the offset through the inverse link.
The default reference offset is shared by all means; it does not depend on `by`,
`at`, prior weights, or `cov_reduce`. For a log-exposure offset this uses the mean
of the log exposures, rather than the log of the mean exposure. Supply an override
when a different exposure or group-specific offsets are wanted.

```py
# Per-unit rates from a count model fitted with a log-exposure offset
rates = mlm.emmeans(count_model, "treatment", offset=0)
# Expected counts at an exposure of 10
counts = mlm.emmeans(count_model, "treatment", offset=np.log(10))
```

An offset sequence must match the rows of `em.result.grid` exactly, including
their order when `by` is used. It replaces the fitted reference offset and is
neither recycled nor added to it. A scalar applies to every row. Equal offsets
cancel in ordinary pairwise comparisons; differing offsets enter the comparison
estimate. Custom contrasts also apply their coefficients to the offsets.

The third positional argument now behaves as `by`. The former `_by=` keyword is
retained as an alias; passing both names raises an error.

## Bootstrap

### bootMer

Parametric bootstrap for mixed models.

```py
boot = mlm.bootMer(model, nsim=500, seed=42)
```

**Parameters:**

- `model`: Fitted model
- `nsim`: Number of bootstrap simulations
- `seed`: Optional reproducibility seed
- `n_jobs`: Worker processes for the refits, or `-1` for all CPUs, for all
  model types; default 1. See [parallel execution](#parallel-execution).

**Returns:** `BootstrapResult`, or `NlmerBootstrapResult` for nonlinear models

Linear and generalized parametric bootstrap use local random streams for each
replicate and leave NumPy's global random state unchanged, including in worker
processes. A fixed integer seed gives the same samples with serial and parallel
execution. The direct `bootstrap_lmer()` and `bootstrap_glmer()` functions also
accept a reusable NumPy `RandomState` or `Generator` as `seed`:

```py
import numpy as np

from mixedlm.inference import bootstrap_lmer

rng = np.random.default_rng(42)
first = bootstrap_lmer(model, n_boot=100, seed=rng)
next_batch = bootstrap_lmer(model, n_boot=100, seed=rng)
```

The sample count must be a positive integer.

Parallel runs return samples in replicate order. The worker count is capped at
the number of replicates, and invalid counts fail before consuming a supplied
random stream. Each refit starts from the original fitted covariance parameters.

**Methods:**

- `ci(level=0.95, method="percentile")`: Fixed-effect confidence intervals
- `se()`: Fixed-effect bootstrap standard errors
- `beta_samples`, `theta_samples`, `sigma_samples`: Bootstrap sample arrays
- `n_failed`: Number of unsuccessful replicates
- `failures`: Tuple of `BootstrapFailure` records, ordered by sample row
- `summary()`: Sample statistics and failure counts by stage

**Example:**

```py
boot = mlm.bootMer(model, nsim=500, seed=42)
ci = boot.ci()
print(ci)
```

For nonlinear fits, `bootMer()` returns `NlmerBootstrapResult`, with
`phi_samples` in place of `beta_samples`.

For all model types, each failed simulation or refit leaves an entire sample row
as `NaN` and increments `n_failed`. Refits must converge, including the inner
PIRLS or PNLS solve where applicable, and return finite real estimates with the
expected shapes. Residual scales must be positive. These checks apply to both
serial and parallel bootstrap execution. Converged fits with zero variance
components are retained.

Each `BootstrapFailure` contains:

| Attribute | Meaning |
| --- | --- |
| `index` | Zero-based row in the sample arrays |
| `stage` | `"simulation"`, `"refit"`, `"convergence"`, or `"validation"` |
| `exception_type` | Exception class name, such as `"ValueError"` |
| `message` | Exception message or reason a convergence/estimate check failed |

Simulation includes response generation and validation; responses must be finite
real vectors with one entry per observation. Refit includes per-replicate model
preparation and fitting. Convergence checks identify unsuccessful status flags,
including `pirls_converged` or `pnls_converged` when applicable. Validation checks
the returned estimates and residual scale. The first failure in each replicate
is recorded. Setup errors, worker-pool failures, and interruptions still propagate.

```py
print(boot.summary())
for failure in boot.failures:
    print(failure.index, failure.stage, failure.exception_type, failure.message)
```

Records are immutable and contain strings rather than exception objects or
tracebacks, so results remain serializable even when a custom exception is not.
They use the same format in serial and parallel runs. A successful bootstrap has
`failures == ()`. Results constructed without this optional field default to an
empty tuple; their historical failure details cannot be recovered.

Confidence intervals and standard errors exclude failed samples and are `NaN`
when fewer than two valid samples remain for a parameter. Check `n_failed`
before interpreting intervals. `NlmerResult.confint(method="boot")` uses this
same bootstrap path. Nonlinear bootstrap counts (`nsim` or `n_boot`) must be
positive integers.

Nonlinear bootstrap (`bootstrap_nlmer()`, `bootMer()` on an `NlmerResult`, and
`NlmerResult.confint(method="boot")`) uses a local random stream shared across
replicates. Integer seeds retain the previous simulation sequence without
changing NumPy's global random state. These interfaces also accept a reusable
NumPy `RandomState` or `Generator` as `seed`.

Set `n_jobs=2` on any nonlinear bootstrap interface to refit with two worker
processes. Simulation stays in the calling process and follows the same draw
sequence as serial execution; completed samples retain replicate order. Custom
overrides of `simulate()` are invoked for every replicate, and a failed draw does
not prevent later draws from being attempted.

Custom nonlinear model classes must be importable and picklable, with
deterministic prediction and gradient methods. Each worker refit receives a
separate copy of the model. Start parallel bootstrap inside an
`if __name__ == "__main__":` guard. Process startup can outweigh the gain for
small bootstrap jobs; `n_jobs=1` remains the default.

### bootCI

Create tidy confidence intervals for fixed effects, variance parameters, and
the residual scale. Multiple interval methods can be computed in one pass.

```py
intervals = mlm.bootCI(
    boot,
    component="all",
    method=["percentile", "basic", "normal"],
)
```

The returned data frame includes each parameter's original estimate, bootstrap
mean, bias, sample standard error, confidence bounds, and successful replicate
count. `component="sigma"` is available for linear and nonlinear models; GLMM
results do not have a separately estimated residual scale.

With fewer than two finite samples for a parameter, every interval method
returns `NaN` confidence bounds and standard errors. A single valid sample
still has a reported mean, bias, and `n.success` count.

Repeated coefficient labels remain separate rows in `bootCI`, in sample-column
order. Bootstrap dictionary methods (`ci()` and `se()`) reject repeated labels
because a dictionary cannot represent both coefficients under one key.

## Profile Likelihood

LMM fixed-effect profiles re-optimize covariance parameters, remaining fixed
coefficients, and residual scale at each constrained value. They use maximum
likelihood, including an ML refit when the input used REML. The original result
is unchanged. `ProfileResult.mle` records the ML center, which can differ from
the input coefficient; a warning reports shifts above 0.001 ML standard errors.

`n_jobs` profiles coefficients in parallel worker processes, falling back to
serial execution with a warning if workers cannot be created.
Confidence limits use likelihood-ratio cutoffs and adaptive bracketing. Nuisance
fits use L-BFGS-B with exact native gradients; compound-symmetry and AR(1)
covariances use finite differences of the Python likelihood. A failed
gradient optimization retries the same likelihood with COBYQA. Fits starting at
zero variance use COBYQA directly so constrained optima can leave that boundary.
Gradient fits that reach zero variance are also checked with COBYQA.
If optimization does not converge, or an interval cannot be bracketed, the
calculation raises an error. The calculation costs more than the former
covariance-fixed approximation; Wald intervals remain available for faster
inference.

GLMM fixed-effect profiles re-optimize every other fixed coefficient and the
covariance parameters at each constrained value, using the fitted model's
Laplace or adaptive-quadrature likelihood. Profiles check zero variance scales
for improving directions and restart the nuisance optimizer when needed, so
constrained optima can leave a zero-variance fit. `model.confint(method="profile")`
uses the same calculation. These intervals can be asymmetric; they are no
longer copies of the Wald intervals.

```py
profiles = model.profile(which="x", n_points=20, level=0.95)
interval = model.confint(parm="x", method="profile", level=0.95)
```

The profiler first refines the joint likelihood optimum over fixed coefficients
and covariance parameters. Default joint fits usually retain their center within
optimization tolerance. For a fit made with `nAGQ=0`, profiling uses the joint
Laplace likelihood and can shift the center from the preliminary PIRLS estimate.
`ProfileResult.mle` records the refined center, and
a warning identifies shifts greater than 0.001 fitted standard errors. The
original fitted result is unchanged.

The likelihood-ratio endpoints are solved independently of the plotting grid;
`n_points` must be an integer of at least 3. Nearby constrained solutions and
repeated evaluations are reused. Quadrature order, prior weights, offsets,
trial counts, and PIRLS controls are retained. The input fit and each inner and
outer solve must converge. Failed optimization or an interval that cannot be
bracketed raises an error, without substituting a Wald interval. GLMM profiling
currently runs serially and costs more than Wald inference.

### plot_profiles

Plot 1D profile likelihood curves.

```py
profiles = model.profile()
mlm.plot_profiles(profiles)
```

### slice2D

Compute a conditional slice or a full two-parameter likelihood profile.

```py
profile_2d = mlm.slice2D(model, param1, param2, n_points=20)

# Re-optimize covariance for a joint likelihood-ratio region:
joint_profile = mlm.slice2D(
    model, param1, param2, n_points=15, profile_covariance=True
)
```

**Parameters:**

- `model`: Fitted model
- `param1`, `param2`: Two distinct fixed-coefficient names
- `n_points`: Number of grid points per dimension
- `profile_covariance`: Default `False` retains the conditional slice at the
  fitted covariance parameters. `True` re-optimizes covariance, other fixed
  coefficients, and residual scale at every grid point using ML. Its grid spans
  the coordinate ranges of the requested joint likelihood-ratio region and
  includes the ML center. Use this mode for joint likelihood-ratio inference.
- `n_jobs`: Worker processes for grid rows, or `-1` for available CPUs. A
  conditional slice stays serial when its remaining rows would finish in about a
  second.

**Returns:** `Profile2DResult` with a `plot()` method and `profile_covariance`
metadata identifying the calculation used.

## Convergence Checking

### checkConv

Check model convergence.

```py
conv = mlm.checkConv(model)
```

**Returns:** ConvergenceInfo object with:

- `converged`: Boolean indicating successful convergence
- `messages`: List of warning/error messages
- `is_singular`, `gradient_norm`, `hessian_ok`, `iterations`, and `optimizer`

The optimizer name and iteration count come from the fitted result, including
modular fits. For a fit that did not converge, the message includes the
optimizer's own reason. The gradient check (`check_gradient=True`) applies to
gradient-based optimizers, which record a final gradient: it divides the gradient
norm by the number of observations, compares it with `grad_tol` (default `1e-3`),
and skips fits with a variance parameter on its boundary.

### convergence_ok

Quick check if model converged successfully.

```py
if mlm.convergence_ok(model):
    print("Model converged")
```

**Returns:** Boolean

## Usage Examples

### Likelihood Ratio Test

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()

# Fit nested models (use REML=False for LRT)
m1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)
m2 = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)

# Compare
result = mlm.anova(m1, m2)
print(result)
```

### Type III ANOVA

```python
cake = mlm.load_cake()
cake_model = mlm.lmer("angle ~ recipe * temperature + (1 | recipe:replicate)", cake)
result = mlm.anova_type3(cake_model)
print(result)
```

### P-values with Degrees of Freedom

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Satterthwaite (default in summary)
print(model.summary())

# Kenward-Roger
print(model.summary(ddf_method="Kenward-Roger"))

# Direct access
df_sat = mlm.satterthwaite_df(model)
df_kr = mlm.kenward_roger_df(model)
pvals = mlm.pvalues_with_ddf(model)
```

### Estimated Marginal Means

```python
# Marginal means for each recipe
em = mlm.emmeans(cake_model, "recipe")
print(em)

# Pairwise contrasts
print(em.pairs())
```

### Bootstrap Confidence Intervals

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Parametric bootstrap; use 1000 or more replicates for reported intervals
boot = mlm.bootMer(model, nsim=50, seed=42)

# Get CIs
ci = mlm.bootCI(boot, component="all")
print(ci)
```

### Profile Likelihood

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Compute profiles
profiles = model.profile(which="Days")

# Plot
mlm.plot_profiles(profiles)

# Profile-based CIs
from mixedlm.inference import confint_profile

ci = confint_profile(profiles)
print(ci)
```

### 2D Profile

```python
# Examine relationship between two parameters
profile_2d = mlm.slice2D(model, "(Intercept)", "Days", n_points=20)
profile_2d.plot()
```

### Check Convergence

```python
conv = mlm.checkConv(model)
if not conv.converged:
    print("Convergence issues:")
    for msg in conv.messages:
        print(f"  - {msg}")
else:
    print("Model converged successfully")
```

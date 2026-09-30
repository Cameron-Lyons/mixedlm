# Inference

This page documents functions for statistical inference, hypothesis testing, and confidence intervals.

## Linear Hypotheses

### linear_hypothesis

Test arbitrary linear restrictions on fixed-effect coefficients. The null
hypothesis is expressed as $C\beta = r$, where $C$ contains one or more
constraint rows and $r$ is supplied with `rhs`.

```python
from mixedlm.inference import linear_hypothesis

# H0: the x and z slopes are equal
test = linear_hypothesis(model, {"x": 1, "z": -1})
print(test)
```

Named rows test several restrictions jointly while retaining readable labels:

```python
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
index. Constraint rows must be finite, nonzero, linearly independent, and
estimable from the fitted covariance matrix.

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

```python
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
- `data`: Optional aligned data; the stored clean model frame is used by default
- `cv`: Number of folds, default 5
- `group`: Optional column defining whole clusters to hold out
- `metrics`: Metric name, callable, or sequence; defaults are selected by model type
- `shuffle`, `random_state`: Reproducible fold assignment controls
- `re_form`: Random-effect prediction mode; `"auto"` uses fixed effects for grouped folds
- `n_jobs`: Number of folds to fit concurrently, default 1
- `fit_kwargs`: Additional refit options such as optimizer controls

Built-in metrics are weighted `"mse"`, `"rmse"`, `"mae"`, `"r2"`, and GLMM
`"deviance"`. A custom metric receives `(y_true, y_pred, weights)` and returns
one finite scalar. Original model weights and offsets are automatically subset
and preserved in every fold.

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

```python
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

```python
rmse = mlm.weighted_rmse(y_true, y_pred, weights)
mse = mlm.weighted_mse(y_true, y_pred, weights)
mae = mlm.weighted_mae(y_true, y_pred, weights)
r2 = mlm.weighted_r2(y_true, y_pred, weights)
```

## Model Comparison

### model_selection

Rank candidate mixed models with AIC, small-sample corrected AIC, or BIC:

```python
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

```python
import mixedlm as mlm

result = mlm.anova(model1, model2, ...)
```

**Parameters:**

- `*models`: Two or more fitted models to compare

**Returns:** AnovaResult with chi-squared test statistics and p-values

**Example:**

```python
m1 = mlm.lmer("y ~ x + (1 | g)", data, REML=False)
m2 = mlm.lmer("y ~ x + z + (1 | g)", data, REML=False)
print(mlm.anova(m1, m2))
```

### anova_type3

Type III ANOVA for a single model.

```python
result = mlm.anova_type3(model)
```

**Returns:** AnovaType3Result with F-statistics and p-values for each fixed effect

**Example:**

```python
model = mlm.lmer("y ~ a * b + (1 | g)", data)
print(mlm.anova_type3(model))
```

## Degrees of Freedom

Fixed-effect information projections use sparse random-effect precision solves
for systems with at least 256 random coefficients. This avoids constructing a
dense random-effect precision matrix for each information perturbation. Smaller
systems retain dense Cholesky solves.

### satterthwaite_df

Compute Satterthwaite denominator degrees of freedom.

The variance calculation uses relative uncertainty so that changing response
units (for example, milliseconds to seconds) preserves the degrees of freedom
and p-values, apart from numerical fitting tolerance.

```python
df = mlm.satterthwaite_df(model)
by_coefficient = df.as_dict()
```

**Returns:** `DenomDFResult` containing `df`, `method`, and `param_names`.
Use `df["coefficient_name"]` to retrieve one value or `df.as_dict()` to obtain a
dictionary.

### kenward_roger_df

Compute Kenward-Roger denominator degrees of freedom.

```python
df = mlm.kenward_roger_df(model)
```

**Returns:** `DenomDFResult`, with the same accessors as `satterthwaite_df`.

### pvalues_with_ddf

Compute p-values using denominator degrees of freedom.

```python
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

```python
predictions = mlm.ggpredict(
    model,
    ["Days", "treatment"],
    at={"Days": [0, 5, 10]},
)
```

The returned data frame contains the requested grid columns plus `predicted`,
`std.error`, `conf.low`, and `conf.high`. For GLMMs, `type="response"` builds
the confidence interval on the link scale before transforming both endpoints.
Use `type="link"` to keep results on the linear-predictor scale.

### allEffects

Compute a separate adjusted prediction grid for every fixed-effect variable.

```python
effects = mlm.allEffects(model, n_points=25)
days_effect = effects["Days"]
```

`allEffects()` prepares the model frame, conditioning values, coefficient
covariance, and confidence cutoff once per call. It evaluates grids separately,
so temporary grid storage depends on the largest individual grid. Prepared
values are discarded after the call; later calls use current model values.

When a model has multiple fixed-effect variables, each variable also conditions
the other grids. Therefore, `at` must supply just one value per variable, either
as a scalar or a one-element iterable. Use `ggpredict()` with all relevant terms
for a joint grid with several values per variable. Invalid options and unknown
`at` variables raise an error even when the model has no fixed-effect predictors
and `allEffects()` would otherwise return an empty dictionary.

Both functions are batched, use the fitted fixed-effect covariance matrix, and
require no plotting package. Prediction grids automatically reuse fitted
categorical contrasts, category order, and retained fixed-effect columns. Sum,
Helmert, polynomial, and custom contrasts need no repeated configuration. Factors
outside the requested grid are held at the first fitted category, including when
their source data uses a different category order.

The optional `contrasts=` mapping remains available as an explicit override; it
should match the fitted coefficient parameterization. For example:

```python
model = mlm.lmer("yield ~ treatment * dose + (1 | block)", data,
                 contrasts={"treatment": "sum"})
predictions = mlm.ggpredict(model, "treatment")
effects = mlm.allEffects(model)
```

Grid calculations read the fitted pandas frame without copying or modifying it.
Returned prediction tables are independent of that frame.

For Polars models, adjusted prediction grids convert only fixed-effect predictor
columns through NumPy arrays. This avoids creating Python objects for the whole
model frame. Contiguous numeric columns without missing values can share their
underlying storage; floating-point reference reductions retain float64 precision.
Category order and missing values are preserved, and returned prediction tables
can be edited independently of the fitted data.
Extraction also handles Polars 0.20 releases that cannot export categorical
columns directly to NumPy or return nonnullable Booleans as object arrays.

### emmeans

Compute estimated marginal means.

Factors outside `specs` are averaged equally over their reference-grid levels;
`at` can restrict those levels. Grid reduction preserves the order of `specs`
and each factor's levels. Pairwise, treatment-versus-control, and custom
comparisons evaluate coefficient projections in bounded batches. The returned
comparison arrays and labels still grow with the number of comparisons.

```python
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

Reference grids are evaluated in batches so averaging over many combinations
of other factors does not require keeping the complete grid and design matrix
in memory. All combinations still contribute with equal weight, using the
fitted categorical encoding. Memory for the returned grid and its coefficient
matrix scales with the number of requested means. Splitting a large average
across batches can change floating-point rounding slightly.

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

```python
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
in the comparison family.
Malformed shapes, complex values, and nonfinite or masked coefficients raise
clear errors before covariance calculations. Numeric coefficient matrices are
validated in bounded batches. The legacy `"dunnett"` option remains a Bonferroni approximation and
counts the comparisons actually requested; it does not compute the exact
Dunnett distribution.

Custom contrasts are evaluated in batches, with each coefficient row normalized
by a power of two before its estimate and standard error are calculated. This
keeps two-sided tests stable when a row is multiplied by a very
small or large nonzero constant. Estimates and standard errors are returned in
the requested units. Calculation uses float64 precision; values outside its
representable range may round to zero or infinity even when the corresponding
test statistic is finite. Scaling cannot recover precision already lost in the
input coefficients.

**Example:**

```python
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

```python
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

```python
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

```python
boot = mlm.bootMer(model, nsim=500, seed=42)
```

**Parameters:**

- `model`: Fitted model
- `nsim`: Number of bootstrap simulations
- `seed`: Optional reproducibility seed

**Returns:** BootstrapResult object

**Methods:**

- `ci(level=0.95, method="percentile")`: Fixed-effect confidence intervals
- `se()`: Fixed-effect bootstrap standard errors
- `beta_samples`, `theta_samples`, `sigma_samples`: Bootstrap sample arrays

**Example:**

```python
boot = mlm.bootMer(model, nsim=500, seed=42)
ci = boot.ci()
print(ci)
```

### bootCI

Create tidy confidence intervals for fixed effects, variance parameters, and
the residual scale. Multiple interval methods can be computed in one pass.

```python
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

## Profile Likelihood

LMM fixed-effect profiles hold the fitted covariance parameters (`theta`) fixed
while recomputing the remaining fixed effects and residual scale. The
one-parameter curves and two-parameter slices reuse the fitted precision
solver. Large random-effect systems stay sparse, including calculations in
parallel workers.

### plot_profiles

Plot 1D profile likelihood curves.

```python
profiles = model.profile()
mlm.plot_profiles(profiles)
```

### slice2D

Compute 2D profile likelihood slice.

```python
profile_2d = mlm.slice2D(model, param1, param2, n_points=20)
```

**Parameters:**

- `model`: Fitted model
- `param1`, `param2`: Parameter names to profile
- `n_points`: Number of grid points per dimension

**Returns:** Profile2DResult with `plot()` method

## Convergence Checking

### checkConv

Check model convergence.

```python
conv = mlm.checkConv(model)
```

**Returns:** ConvergenceInfo object with:

- `ok`: Boolean indicating successful convergence
- `messages`: List of warning/error messages

### convergence_ok

Quick check if model converged successfully.

```python
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
model = mlm.lmer("y ~ a * b + (1 | group)", data)
result = mlm.anova_type3(model)
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
model = mlm.lmer("yield ~ treatment + (1 | block)", data)

# Marginal means for treatment
em = mlm.emmeans(model, "treatment")
print(em)

# Pairwise contrasts
print(em.pairs())
```

### Bootstrap Confidence Intervals

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Parametric bootstrap
boot = mlm.bootMer(model, nsim=500, seed=42)

# Get CIs
ci = mlm.bootCI(boot, component="all")
print(ci)
```

### Profile Likelihood

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Compute profiles
profiles = model.profile()

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
model = mlm.lmer("y ~ x + (x | g)", data)

conv = mlm.checkConv(model)
if not conv.ok:
    print("Convergence issues:")
    for msg in conv.messages:
        print(f"  - {msg}")
else:
    print("Model converged successfully")
```

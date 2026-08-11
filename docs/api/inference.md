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

### satterthwaite_df

Compute Satterthwaite denominator degrees of freedom.

```python
df = mlm.satterthwaite_df(model)
```

**Returns:** Dictionary mapping coefficient names to degrees of freedom

### kenward_roger_df

Compute Kenward-Roger denominator degrees of freedom.

```python
df = mlm.kenward_roger_df(model)
```

**Returns:** Dictionary mapping coefficient names to degrees of freedom

### pvalues_with_ddf

Compute p-values using denominator degrees of freedom.

```python
pvals = mlm.pvalues_with_ddf(model, method="Satterthwaite")
```

**Parameters:**

- `model`: Fitted model
- `method`: `"Satterthwaite"` or `"Kenward-Roger"`

**Returns:** Dictionary mapping coefficient names to p-values

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

Both functions are batched, use the fitted fixed-effect covariance matrix, and
require no plotting package. If a model was fit with non-default categorical
contrasts, pass the same mapping with `contrasts=` so the prediction grid uses
the fitted parameterization.

### emmeans

Compute estimated marginal means.

```python
em = mlm.emmeans(model, "treatment", type="response")
```

**Parameters:**

- `model`: Fitted model
- `specs`: Factor name or names to compute marginal means for
- `at`: Optional values at which to evaluate other predictors
- `cov_reduce`: Function used to reduce numeric covariates (default: mean)
- `type`: `"response"` (default) or `"link"` for generalized models
- `level`: Confidence level (default: `0.95`)

**Returns:** Emmeans object

**Methods on Emmeans object:**

- `pairs(adjust="tukey")`: Compute all pairwise comparisons
- `contrast(method, adjust="none")`: Compute pairwise, treatment-vs-control, or custom contrasts

**Example:**

```python
model = mlm.lmer("yield ~ treatment + (1 | block)", data)
em = mlm.emmeans(model, "treatment")
print(em)
print(em.pairs())
```

## Bootstrap

### bootMer

Parametric bootstrap for mixed models.

```python
boot = mlm.bootMer(model, data, nsim=500, FUN=None)
```

**Parameters:**

- `model`: Fitted model
- `data`: Original data frame
- `nsim`: Number of bootstrap simulations
- `FUN`: Optional function to extract statistics (default: all parameters)

**Returns:** BootstrapResult object

**Methods:**

- `confint(level=0.95)`: Bootstrap confidence intervals
- `samples`: Array of bootstrap samples

**Example:**

```python
boot = mlm.bootMer(model, data, nsim=500)
ci = boot.confint()
print(ci)
```

## Profile Likelihood

### plot_profiles

Plot 1D profile likelihood curves.

```python
profiles = model.profile(data)
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
boot = mlm.bootMer(model, data, nsim=500)

# Get CIs
ci = boot.confint()
print(ci)
```

### Profile Likelihood

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Compute profiles
profiles = model.profile(data)

# Plot
mlm.plot_profiles(profiles)

# Profile-based CIs
ci = profiles.confint()
print(ci)
```

### 2D Profile

```python
# Examine relationship between two parameters
profile_2d = mlm.slice2D(model, "sigma", "theta1", n_points=20)
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

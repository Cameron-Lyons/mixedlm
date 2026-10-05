# Statistical Inference

This tutorial covers hypothesis testing, confidence intervals, and model comparison for mixed models.

## Cross-Validation

Cross-validation asks how accurately a fitted model predicts observations that
were not used for estimation. Mixed models require choosing the level of
generalization explicitly.

### Predicting New Rows Within Known Groups

Case-level folds split individual observations. A held-out row can use a random
effect estimated from other training rows for the same group.

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

case_cv = mlm.cross_validate(
    model,
    cv=5,
    random_state=123,
)

print(case_cv.scores)
```

### Predicting Entirely New Groups

Set `group` to keep every cluster wholly inside one test fold. Since test groups
are absent from training, the default prediction uses the fixed-effects portion
of each fold model.

```python
subject_cv = mlm.cross_validate(
    model,
    cv=5,
    group="Subject",
    metrics=["rmse", "mae", "r2"],
    random_state=123,
)

print(subject_cv)
print(subject_cv.fold_scores)
```

Pass `n_jobs=-1` (or a worker count) to refit the folds in parallel worker
processes. Results match a serial run. Scripts that do this must call
`cross_validate()` under an `if __name__ == "__main__":` guard; see
[parallel execution](../api/inference.md#parallel-execution).

The grouped splitter assigns larger clusters first to the smallest available
fold. This preserves groups while balancing the number of held-out observations.
Use the same `random_state` when comparing models so they receive identical
fold assignments.

You can also construct the folds once and reuse them, or supply your own
`(train_indices, test_indices)` pairs. Positions refer to the rows used for
fitting, even if the dataframe has other index labels:

```python
folds = mlm.make_folds(len(data), cv=5, groups=data["Subject"], random_state=123)
subject_cv = mlm.cross_validate(model, cv=folds, group="Subject")
```

Explicit test sets must cover every fitted row exactly once, and each train/test
pair must be disjoint. Training sets can exclude additional rows for buffered
holdouts. With `group`, the partitions must hold out whole clusters and exclude
their observations from training. All partition checks happen before refitting.
If you provide `data`, keep its modeled values and categorical encoding in the
original row order so fitted weights and offsets remain aligned. Extra columns
can supply an external holdout grouping.

For GLMMs, the default metrics are weighted RMSE and mean unit deviance:

```python
cbpp = mlm.load_cbpp()
glmm_model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)", cbpp, family=mlm.families.Binomial()
)

glmm_cv = mlm.cross_validate(
    glmm_model,
    cv=5,
    group="herd",
    random_state=123,
)
```

Always inspect `all_converged`, `any_singular`, and their per-fold columns. A
validation score based on failed or boundary fold fits should not be interpreted
without revisiting the model or optimizer settings.

## P-values for Fixed Effects

### Satterthwaite Degrees of Freedom

By default, `summary()` reports p-values using Satterthwaite degrees of freedom:

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

print(model.summary())
```

The output includes degrees of freedom and p-values:

```
Fixed effects:
              Estimate  Std. Error    df  t value  Pr(>|t|)
(Intercept)    251.405       6.825  17.0   36.838    <0.001
Days            10.467       1.546  17.0    6.771    <0.001
```

### Kenward-Roger Degrees of Freedom

For small samples, Kenward-Roger provides better approximation:

```python
print(model.summary(ddf_method="Kenward-Roger"))
```

### Direct Access to DDF

```python
from mixedlm import satterthwaite_df, kenward_roger_df, pvalues_with_ddf

# Denominator degrees of freedom
sat_df = satterthwaite_df(model)
kr_df = kenward_roger_df(model)

# P-values with specific method
pvals = pvalues_with_ddf(model, method="Satterthwaite")
```

### When to Use Each Method

| Method | Use when |
|--------|----------|
| Satterthwaite | Default choice, fast, good for most cases |
| Kenward-Roger | Small samples, complex random effects, more accurate but slower |

## Custom Linear Hypotheses

Use `linear_hypothesis` when the null cannot be expressed as a single formula
term or nested-model comparison. It tests linear combinations of the fitted
fixed effects without refitting the model.

The examples use simulated data with two predictors:

```python
import numpy as np
import pandas as pd

from mixedlm.inference import linear_hypothesis

rng = np.random.default_rng(1)
group = np.repeat(np.arange(20), 10)
x = rng.normal(size=group.size)
z = rng.normal(size=group.size)
xz_data = pd.DataFrame({
    "y": 1 + 0.5 * x + 1.5 * z + rng.normal(0, 0.5, 20)[group] + rng.normal(size=group.size),
    "x": x,
    "z": z,
    "group": group,
})
xz_model = mlm.lmer("y ~ x + z + (1 | group)", xz_data)

# Test H0: beta_x - beta_z = 0
equal_slopes = linear_hypothesis(xz_model, {"x": 1, "z": -1})
print(equal_slopes)
```

Non-zero null values and joint tests are supported:

```python
joint = linear_hypothesis(
    xz_model,
    {
        "equal slopes": {"x": 1, "z": -1},
        "sum equals two": {"x": 1, "z": 1},
    },
    rhs=[0, 2],
)

print(joint.table)       # Individual restrictions
print(joint.statistic)   # Joint F statistic
print(joint.p_value)     # Joint p-value
```

For a numeric matrix, columns must follow `model.matrices.fixed_names`. Named
weights are safer when coefficient order may change. LMMs use F tests by
default; GLMMs use chi-square tests. Supply `denominator_df` when a specific
small-sample denominator DF is required.

## Confidence Intervals

### Wald Intervals

Fast but can be inaccurate for variance components:

```python
ci = model.confint(method="Wald")
print(ci)
```

### Profile Likelihood Intervals

More accurate, especially for variance components:

```python
ci = model.confint(method="profile")
print(ci)
```

Profile CIs are based on the likelihood function shape and don't assume symmetry.

### Bootstrap Intervals

Most robust but computationally intensive. This quick example uses 50
replicates; use 1000 or more for reported intervals:

```python
ci = model.confint(method="boot", n_boot=50, seed=42)
print(ci)
```

### Comparison

| Method | Speed | Fixed effects | Variance components |
|--------|-------|---------------|---------------------|
| Wald | Fast | Good | Poor (can go negative) |
| Profile | Medium | Excellent | Excellent |
| Bootstrap | Slow | Excellent | Excellent |

## Model Comparison

### Likelihood Ratio Tests

Compare nested models:

```python
# Simpler model
m1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)

# More complex model
m2 = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)

# Likelihood ratio test
result = mlm.anova(m1, m2)
print(result)
```

!!! important
    `anova()` automatically refits REML linear mixed models with ML before
    comparison. Pass `refit=False` only when you intentionally want to compare
    the supplied REML fits directly, such as models with identical fixed effects.

### Type III ANOVA

Test fixed effects in a single model. The cake data crosses two treatment
factors, with replicates nested in recipes:

```python
cake = mlm.load_cake()
cake_model = mlm.lmer("angle ~ recipe * temperature + (1 | recipe:replicate)", cake)
result = mlm.anova_type3(cake_model)
print(result)
```

Type III tests are marginal: each effect is tested controlling for all others.

### Single Term Deletions (drop1)

Assess each term's contribution:

```python
result = cake_model.drop1(cake)
print(result)
```

This fits the model without each marginal term and reports the change in fit.
Lower-order terms are retained when they belong to a higher-order interaction.
Linear mixed models originally fitted with REML are automatically refitted with
ML so that fixed-effect deletion likelihoods and AIC values are comparable.

## Estimated Marginal Means (emmeans)

### Computing Marginal Means

```python
# Marginal means for each recipe, averaged over temperatures
em = mlm.emmeans(cake_model, "recipe")
print(em)
```

### Pairwise Contrasts

```python
# All pairwise comparisons
contrasts = em.pairs()
print(contrasts)
```

### Custom Contrasts

```python
# Compare each recipe with the first (control) level
contrasts = em.contrast("trt.vs.ctrl")
print(contrasts)
```

### Multiple Comparison Adjustment

```python
contrasts = em.pairs(adjust="bonferroni")
# Options include "none", "bonferroni", "holm", "fdr" (or "BH"), and "tukey"

# Explicitly request unadjusted pairwise tests
unadjusted = em.contrast("pairwise", adjust="none")
```

`em.pairs()` and `em.contrast("pairwise")` default to Tukey adjustment.
Unknown adjustment names raise an error; names are case-insensitive.

## Profile Likelihood

### Computing Profiles

Compute likelihood profiles for the fixed-effect coefficients. For LMMs, each
constrained value re-optimizes the covariance parameters, the other fixed
effects, and the residual scale by maximum likelihood:

```python
profiles = model.profile()
```

### Visualizing Profiles

```python
from mixedlm import plot_profiles

plot_profiles(profiles)
```

Near the optimum, the default signed square-root deviance (`zeta`) plot is
approximately linear.

### 2D Profile Slices

Examine the relationship between two parameters:

```python
from mixedlm import slice2D

profile_2d = slice2D(model, "(Intercept)", "Days", n_points=20)
profile_2d.plot()
```

### Profile-Based CIs

```python
from mixedlm.inference import confint_profile

ci = confint_profile(profiles)
print(ci)
```

## Parametric Bootstrap

### Basic Bootstrap

```python
from mixedlm import bootCI, bootMer

# Bootstrap the model; use 1000 or more replicates for reported intervals
boot = bootMer(model, nsim=50, seed=42)

# Access bootstrap samples
boot.beta_samples   # Fixed-effect estimates
boot.theta_samples  # Variance-parameter estimates

# Bootstrap confidence intervals
bootCI(boot, component="all")
```

Large bootstraps can refit in parallel with `n_jobs`, for example
`bootMer(model, nsim=1000, seed=42, n_jobs=-1)`. A fixed seed gives the same
samples for every worker count. Run parallel work under an
`if __name__ == "__main__":` guard, as described in
[parallel execution](../api/inference.md#parallel-execution).

### Bootstrap for Specific Statistics

```python
# Select a named parameter from the tidy interval table
days_ci = bootCI(boot, parameters="Days")

# Or work directly with its bootstrap samples
days_index = boot.fixed_names.index("Days")
days_samples = boot.beta_samples[:, days_index]
```

### Bootstrap for Predictions

```python
import numpy as np

# Fixed-only linear prediction at Days=10 for every bootstrap replicate
intercept = boot.beta_samples[:, boot.fixed_names.index("(Intercept)")]
days = boot.beta_samples[:, boot.fixed_names.index("Days")]
prediction_samples = intercept + 10 * days
prediction_ci = np.quantile(prediction_samples, [0.025, 0.975])
```

## Testing Random Effects

### Is the Random Effect Needed?

Compare the ML fit with an ordinary least squares fit of the same fixed effects:

```python
from scipy import stats

m1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)

# Maximized log-likelihood without the random intercept
X = np.column_stack([np.ones(len(data)), data["Days"]])
beta, *_ = np.linalg.lstsq(X, data["Reaction"], rcond=None)
sigma2 = np.mean((data["Reaction"] - X @ beta) ** 2)
loglik_ols = -0.5 * len(data) * (np.log(2 * np.pi * sigma2) + 1)

lrt = 2 * (m1.logLik().value - loglik_ols)
# The null variance is on the boundary, so halve the chi-square p-value
p_value = 0.5 * stats.chi2.sf(lrt, df=1)
```

### Testing Variance Components

Likelihood ratio tests for variance components are conservative because the null hypothesis is on the boundary of the parameter space. The p-value from a chi-square test should typically be halved.

## Multiple Optimizers (allFit)

Check if results are sensitive to optimizer choice:

```python
all_results = model.allFit(data)
print(all_results.summary)
print(all_results.is_consistent())
```

The default list contains every installed solver from
`mixedlm.estimation.available_optimizers()`, and the refits keep the model's other
control settings. `is_consistent()` checks whether the converged fits reach the
same deviance. If different optimizers give very different results, the model may
be problematic. Pass `n_jobs` to run the refits in worker processes.

## Checking Convergence

```python
conv = mlm.checkConv(model)

if not conv.converged:
    print("Convergence issues detected:")
    for msg in conv.messages:
        print(f"  - {msg}")
```

## Complete Example

```python
import mixedlm as mlm
from mixedlm.inference import confint_profile

# Load data
data = mlm.load_sleepstudy()

# Fit model
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# 1. Summary with p-values
print("=== Model Summary ===")
print(model.summary())

# 2. Profile confidence interval for the Days effect
print("\n=== Profile CI ===")
profiles = model.profile(which="Days")
print(confint_profile(profiles))

# 3. Compare to simpler model
print("\n=== Model Comparison ===")
m_simple = mlm.lmer("Reaction ~ Days + (1 | Subject)", data, REML=False)
m_full = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)
print(mlm.anova(m_simple, m_full))

# 4. Bootstrap CI for the Days effect (use 1000 or more replicates in practice)
print("\n=== Bootstrap CI for Days Effect ===")
boot = mlm.bootMer(model, nsim=50, seed=42)
boot_ci = mlm.bootCI(boot, parameters="Days")
print(boot_ci[["parameter", "conf.low", "conf.high"]])

# 5. Check convergence
conv = mlm.checkConv(model)
print(f"\n=== Convergence: {conv.converged} ===")
```

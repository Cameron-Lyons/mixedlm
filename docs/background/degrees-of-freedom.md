# Degrees of Freedom

This page explains denominator degrees of freedom methods for inference in mixed models.

## The Problem

In ordinary linear regression, the t-statistic for testing \(\beta_j = 0\) follows a t-distribution with \(n - p\) degrees of freedom.

For mixed models, determining the appropriate degrees of freedom is more complex because:

1. Observations within groups are correlated
2. The "effective sample size" depends on the variance structure
3. Different fixed effects may have different effective degrees of freedom

## Why It Matters

Using incorrect degrees of freedom leads to:

- **Too few DF**: Conservative tests, inflated p-values, low power
- **Too many DF**: Liberal tests, false positives, invalid inference

The problem is most severe with:

- Small number of groups
- Complex random effects structures
- Unbalanced designs

## Available Methods

### Satterthwaite Approximation

The Satterthwaite method approximates the sampling distribution of \(\hat{\beta}_j / SE(\hat{\beta}_j)\) by matching moments to a t-distribution.

For a contrast \(\mathbf{c}^T\boldsymbol{\beta}\):

\[
\nu = \frac{2[\mathbf{c}^T\mathbf{V}_{\hat{\beta}}\mathbf{c}]^2}{\text{Var}[\mathbf{c}^T\mathbf{V}_{\hat{\beta}}\mathbf{c}]}
\]

where \(\mathbf{V}_{\hat{\beta}}\) is the covariance matrix of the fixed effects.

**Computation:**

The denominator requires derivatives of the covariance matrix with respect to variance parameters, computed via numerical differentiation.

```python
import mixedlm as mlm

model = mlm.lmer("y ~ x + (1 | g)", data)
print(model.summary())  # Uses Satterthwaite by default

# Direct access
df = mlm.satterthwaite_df(model)
```

**Properties:**

- Fast to compute
- Generally accurate for balanced designs
- May be slightly anti-conservative for very small samples

### Kenward-Roger Approximation

Kenward-Roger extends Satterthwaite by:

1. Adjusting the covariance matrix for small-sample bias
2. Using a more accurate approximation to the variance of \(\mathbf{c}^T\mathbf{V}_{\hat{\beta}}\mathbf{c}\)

The adjusted covariance is:

\[
\mathbf{V}^{KR}_{\hat{\beta}} = \mathbf{V}_{\hat{\beta}} + \text{bias adjustment}
\]

```python
print(model.summary(ddf_method="Kenward-Roger"))

# Direct access
df_kr = mlm.kenward_roger_df(model)
```

**Properties:**

- More accurate than Satterthwaite for small samples
- Computationally more expensive
- Recommended for studies with few groups (< 50)

### Comparison

| Aspect | Satterthwaite | Kenward-Roger |
|--------|---------------|---------------|
| Speed | Fast | Slower |
| Small sample bias | Some | Corrected |
| Implementation | Simpler | More complex |
| Recommendation | Default | Small samples |

## When Each Method is Appropriate

### Use Satterthwaite When:

- You have many groups (> 50)
- Computational speed matters
- Design is reasonably balanced
- This is the default for most analyses

### Use Kenward-Roger When:

- You have few groups (< 30)
- Design is highly unbalanced
- Maximum accuracy is needed
- You're using REML estimation

## Practical Examples

### Example: Sleep Study

```python
import mixedlm as mlm

data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Compare methods
print("=== Satterthwaite ===")
print(model.summary(ddf_method="Satterthwaite"))

print("\n=== Kenward-Roger ===")
print(model.summary(ddf_method="Kenward-Roger"))
```

With 18 subjects, you'll see slightly different DF values. For this moderately-sized dataset, the difference in p-values is usually small.

### Example: Small Study

With very few groups, the difference is more pronounced:

```python
# Subset to 6 subjects
small_data = data[data['Subject'].isin(data['Subject'].unique()[:6])]
model_small = mlm.lmer("Reaction ~ Days + (1 | Subject)", small_data)

print("Satterthwaite DF:", mlm.satterthwaite_df(model_small))
print("Kenward-Roger DF:", mlm.kenward_roger_df(model_small))
```

## Understanding the Output

### Degrees of Freedom per Effect

Different fixed effects can have different DF:

```python
model = mlm.lmer("y ~ x1 + x2 + (1 | group)", data)
df = mlm.satterthwaite_df(model)
print(df)
# {'(Intercept)': 25.3, 'x1': 150.2, 'x2': 148.7}
```

- **Intercept**: DF related to between-group variation
- **Level-1 predictors** (varying within groups): Higher DF
- **Level-2 predictors** (constant within groups): Lower DF

### Interpreting Fractional DF

The DF can be non-integer (e.g., 17.3). This reflects the approximation matching moments of a t-distribution. Use the fractional value directly when computing p-values.

## Effect on Confidence Intervals

Profile and bootstrap CIs don't use degrees of freedom approximations. Wald CIs do:

```python
# Wald CI uses DF for critical value
ci_wald = model.confint(method="Wald")

# Profile CI doesn't depend on DF approximation
ci_profile = model.confint(method="profile")
```

For small samples, profile or bootstrap CIs are preferred.

## Common Issues

### Very Small DF

If estimated DF is < 4, interpret results cautiously:

```python
df = mlm.satterthwaite_df(model)
for term, d in df.items():
    if d < 4:
        print(f"Warning: {term} has only {d:.1f} DF")
```

### DF Larger Than Expected

Very large DF (approaching n) might indicate:

- Random effects variance near zero
- Predictor varies mostly within groups
- Model may be over-parameterized

### Negative or Undefined DF

Can occur with:

- Singular fits (variance = 0)
- Numerical issues in covariance estimation

Check model convergence if this happens.

## ANOVA F-tests

For Type III ANOVA tests, both numerator and denominator DF are needed:

```python
result = mlm.anova_type3(model)
print(result)
# Shows Num DF, Den DF, F value, Pr(>F)
```

The denominator DF comes from Satterthwaite (or KR if specified).

## References

- Satterthwaite, F. E. (1946). An approximate distribution of estimates of variance components. *Biometrics Bulletin*, 2(6), 110-114.
- Kenward, M. G., & Roger, J. H. (1997). Small sample inference for fixed effects from restricted maximum likelihood. *Biometrics*, 53(3), 983-997.
- Luke, S. G. (2017). Evaluating significance in linear mixed-effects models in R. *Behavior Research Methods*, 49(4), 1494-1502.

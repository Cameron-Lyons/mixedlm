# Quickstart

This guide walks through fitting your first mixed-effects model in 5 minutes.

## Loading Data

mixedlm includes several built-in datasets from lme4:

```python
import mixedlm as mlm

# Sleep deprivation study
data = mlm.load_sleepstudy()
print(data.head())
```

```text
   Reaction  Days Subject
0  249.5600   0.0     308
1  258.7047   1.0     308
2  250.8006   2.0     308
3  321.4398   3.0     308
4  356.8519   4.0     308
```

This dataset contains reaction times measured over 10 days of sleep deprivation for 18 subjects.

## Fitting a Linear Mixed Model

Fit a model with random intercepts and slopes for each subject:

```python
result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
```

The formula syntax follows lme4:

- `Reaction ~ Days` - Fixed effect of Days on Reaction
- `(Days | Subject)` - Random intercept and slope for each Subject, with correlation

## Viewing Results

The `summary()` method shows fixed effects with p-values:

```python
print(result.summary())
```

```text
Linear mixed model fit by REML
Formula: Reaction ~ Days + (1 + Days | Subject)

Random effects:
 Groups      Name           Variance   Std.Dev.   Corr
 Subject     (Intercept)    612.0901    24.7405
             Days            35.0717     5.9221   0.07
 Residual                   654.9410    25.5918
Number of obs: 180
  groups:  Subject, 18

Fixed effects:
               Estimate   Std.Error        df   t value    Pr(>|t|)
(Intercept)    251.4051      6.8246     17.00    36.838     < 2e-16 ***
Days            10.4673      1.5458     17.00     6.771    3.26e-06 ***
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Extracting Components

```python
# Fixed effects coefficients
result.fixef()
# {'(Intercept)': 251.405, 'Days': 10.467}

# Random effects (BLUPs) by subject
result.ranef()
# Returns dict with Subject-level deviations

# Variance components
result.VarCorr()
# Shows variance-covariance of random effects

# Fitted values
result.fitted()

# Residuals
result.residuals()
```

## Inference

### Confidence Intervals

```python
# Wald intervals (fast)
result.confint(method="Wald")

# Profile likelihood intervals (more accurate)
result.confint(method="profile")

# Bootstrap intervals (most robust; use more replicates for reported results)
result.confint(method="boot", n_boot=200, seed=42)
```

### Model Comparison

Compare nested models with likelihood ratio tests:

```python
# Simpler model without random slopes
model1 = mlm.lmer("Reaction ~ Days + (1 | Subject)", data)

# Full model
model2 = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Compare
mlm.anova(model1, model2)
```

### Predictions

```python
# Predictions on original data
result.predict()

# Predictions on new data
import pandas as pd
new_data = pd.DataFrame({
    'Days': [0, 5, 10],
    'Subject': ['308', '308', '308']
})
result.predict(newdata=new_data)
```

## Using Polars

mixedlm works directly with eager and lazy polars frames. Lazy inputs are
projected to the columns required by the formula before collection:

```python
import polars as pl

data_pl = pl.DataFrame(data.to_dict(orient="list"))

result_pl = mlm.lmer("Reaction ~ Days + (Days | Subject)", data_pl)
print(result_pl.fixef())

lazy_result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data_pl.lazy())
```

## Next Steps

- [Linear Mixed Models Tutorial](../tutorials/linear-mixed-models.md) - Deeper dive into LMMs
- [Formula Syntax](../background/formula-syntax.md) - Complete reference for random effects notation
- [Coming from R](coming-from-r.md) - If you're familiar with lme4

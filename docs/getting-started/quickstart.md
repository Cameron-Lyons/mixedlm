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

```
   Reaction  Days  Subject
0  249.5600     0      308
1  258.7047     1      308
2  250.8006     2      308
3  321.4398     3      308
4  356.8519     4      308
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

```
Linear mixed model fit by REML

Formula: Reaction ~ Days + (Days | Subject)

Random effects:
 Groups   Name        Variance  Std.Dev.  Corr
 Subject  (Intercept)  612.10    24.74
          Days          35.07     5.92    0.07
 Residual              654.94    25.59

Number of obs: 180, groups: Subject, 18

Fixed effects:
              Estimate  Std. Error    df  t value  Pr(>|t|)
(Intercept)    251.405       6.825  17.0   36.838    <0.001
Days            10.467       1.546  17.0    6.771    <0.001
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

# Bootstrap intervals (most robust)
result.confint(method="boot", nsim=500)
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

mixedlm works directly with polars DataFrames:

```python
import polars as pl

data_pl = pl.DataFrame({
    "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "group": ["A", "A", "B", "B", "C", "C"],
})

result = mlm.lmer("y ~ x + (1 | group)", data_pl)
print(result.summary())
```

## Next Steps

- [Linear Mixed Models Tutorial](../tutorials/linear-mixed-models.md) - Deeper dive into LMMs
- [Formula Syntax](../background/formula-syntax.md) - Complete reference for random effects notation
- [Coming from R](coming-from-r.md) - If you're familiar with lme4

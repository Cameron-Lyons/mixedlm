# Coming from R

This guide helps lme4 users transition to mixedlm. The API is designed to be as similar as possible.

## Side-by-Side Comparison

### Basic Model Fitting

=== "R (lme4)"

    ```r
    library(lme4)

    # Linear mixed model
    m <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)

    # Generalized linear mixed model
    m <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               data = cbpp, family = binomial)
    ```

=== "Python (mixedlm)"

    ```python
    import mixedlm as mlm

    # Linear mixed model
    m = mlm.lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

    # Generalized linear mixed model
    m = mlm.glmer("incidence / size ~ period + (1 | herd)",
                  cbpp, family=mlm.families.Binomial())
    ```

### Extracting Results

=== "R (lme4)"

    ```r
    fixef(m)        # Fixed effects
    ranef(m)        # Random effects
    VarCorr(m)      # Variance components
    coef(m)         # Combined coefficients
    sigma(m)        # Residual SD
    logLik(m)       # Log-likelihood
    AIC(m)          # AIC
    fitted(m)       # Fitted values
    residuals(m)    # Residuals
    ```

=== "Python (mixedlm)"

    ```python
    m.fixef()       # Fixed effects
    m.ranef()       # Random effects
    m.VarCorr()     # Variance components
    m.coef()        # Combined coefficients
    mlm.sigma(m)    # Residual SD
    m.logLik()      # Log-likelihood
    m.AIC()         # AIC
    m.fitted()      # Fitted values
    m.residuals()   # Residuals
    ```

### Model Comparison

=== "R (lme4)"

    ```r
    m1 <- lmer(y ~ x + (1 | g), data)
    m2 <- lmer(y ~ x + z + (1 | g), data)
    anova(m1, m2)
    ```

=== "Python (mixedlm)"

    ```python
    m1 = mlm.lmer("y ~ x + (1 | g)", data)
    m2 = mlm.lmer("y ~ x + z + (1 | g)", data)
    mlm.anova(m1, m2)
    ```

### Confidence Intervals

=== "R (lme4)"

    ```r
    confint(m)                    # Profile (default)
    confint(m, method = "Wald")   # Wald
    confint(m, method = "boot")   # Bootstrap
    ```

=== "Python (mixedlm)"

    ```python
    m.confint(method="profile")   # Profile
    m.confint(method="Wald")      # Wald
    m.confint(method="boot")      # Bootstrap
    ```

### P-values with lmerTest

=== "R (lmerTest)"

    ```r
    library(lmerTest)
    m <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
    summary(m)  # Includes Satterthwaite p-values

    # Or use Kenward-Roger
    anova(m, ddf = "Kenward-Roger")
    ```

=== "Python (mixedlm)"

    ```python
    m = mlm.lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    m.summary()  # Includes Satterthwaite p-values by default

    # Or use Kenward-Roger
    m.summary(ddf_method="Kenward-Roger")
    ```

### Estimated Marginal Means

=== "R (emmeans)"

    ```r
    library(emmeans)
    emmeans(m, ~ treatment)
    pairs(emmeans(m, ~ treatment))
    ```

=== "Python (mixedlm)"

    ```python
    em = mlm.emmeans(m, "treatment", data)
    em.emmeans()
    em.contrasts("pairwise")
    ```

### Power Analysis

=== "R (simr)"

    ```r
    library(simr)
    powerSim(m, nsim = 100)
    powerCurve(m, along = "n")
    extend(m, along = "n", n = 100)
    ```

=== "Python (mixedlm)"

    ```python
    mlm.powerSim(m, data, nsim=100)
    mlm.powerCurve(m, data, along="n", breaks=[50, 100, 200])
    mlm.extend(m, data, along="n", n=100)
    ```

## Key Differences

### Formula Syntax

The formula syntax is identical:

| Pattern | Meaning |
|---------|---------|
| `(1 \| group)` | Random intercept |
| `(x \| group)` | Random intercept + slope (correlated) |
| `(x \|\| group)` | Random intercept + slope (uncorrelated) |
| `(1 \| a/b)` | Nested random effects |
| `(1 \| a) + (1 \| b)` | Crossed random effects |

### GLMM Binomial Syntax

R uses `cbind()` for binomial responses:

```r
# R
glmer(cbind(successes, failures) ~ x + (1 | g), family = binomial)
```

mixedlm uses division syntax:

```python
# Python
mlm.glmer("successes / total ~ x + (1 | g)", data, family=mlm.families.Binomial())
```

### Family Objects

R families are functions, mixedlm families are classes:

```r
# R
glmer(..., family = binomial(link = "logit"))
```

```python
# Python
mlm.glmer(..., family=mlm.families.Binomial(link="logit"))
```

### DataFrame Backend

R uses data.frame. mixedlm works with both pandas and polars:

```python
# Works with pandas
import pandas as pd
df = pd.DataFrame(...)
mlm.lmer("y ~ x + (1 | g)", df)

# Works with polars
import polars as pl
df = pl.DataFrame(...)
mlm.lmer("y ~ x + (1 | g)", df)
```

### Control Objects

Control arguments are similar:

=== "R"

    ```r
    lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10000))
    ```

=== "Python"

    ```python
    mlm.LmerControl(optimizer="bobyqa", maxfun=10000)
    ```

## Functions Not Yet Available

These lme4/related functions are not yet implemented:

- `allEffects` (effects package)
- `bootCI` (boot package style)
- `ggpredict` (ggeffects)
- `plot.merMod` (use `mlm.plot_diagnostics` instead)

## Getting Help

If you're stuck translating R code:

1. Check the [API Reference](../api/models.md) for function signatures
2. Most lme4 methods have direct equivalents as result methods
3. Check [Formula Syntax](../background/formula-syntax.md) for random effects notation

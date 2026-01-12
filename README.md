# mixedlm

A Python implementation of mixed-effects models inspired by R's [lme4](https://github.com/lme4/lme4) package. Features a Rust backend for performance-critical operations.

## Features

- **Linear Mixed Models (LMM)** via `lmer()` - REML and ML estimation
- **Generalized Linear Mixed Models (GLMM)** via `glmer()` - Laplace approximation and adaptive Gauss-Hermite quadrature
- **Nonlinear Mixed Models (NLMM)** via `nlmer()` - Self-starting models (SSasymp, SSlogis, SSmicmen)
- **Formula interface** - lme4-style formulas with random effects syntax
- **Inference tools** - Profile likelihood, parametric bootstrap, confidence intervals
- **Model comparison** - ANOVA, drop1, allFit

## Installation

```bash
pip install mixedlm
```

Or install from source:

```bash
git clone https://github.com/cameronlyons/mixedlm.git
cd mixedlm
pip install -e .
```

## Quick Start

### Linear Mixed Model

```python
import mixedlm as mlm
import pandas as pd

# Fit a random intercept model
result = mlm.lmer("Reaction ~ Days + (1 | Subject)", data)
print(result.summary())

# Extract components
result.fixef()      # Fixed effects
result.ranef()      # Random effects (BLUPs)
result.VarCorr()    # Variance components
result.coef()       # Combined coefficients

# Inference
result.confint(method="profile")  # Profile confidence intervals
result.confint(method="boot")     # Bootstrap confidence intervals
```

### Generalized Linear Mixed Model

```python
# Binomial GLMM
result = mlm.glmer(
    "y ~ treatment + (1 | subject)",
    data,
    family=mlm.families.Binomial()
)

# Poisson GLMM with adaptive quadrature
result = mlm.glmer(
    "count ~ x + (1 | group)",
    data,
    family=mlm.families.Poisson(),
    nAGQ=10
)
```

### Model Comparison

```python
# Fit nested models
model1 = mlm.lmer("y ~ x + (1 | group)", data)
model2 = mlm.lmer("y ~ x + z + (1 | group)", data)

# Likelihood ratio test
mlm.anova(model1, model2)

# Single term deletions
result.drop1(data)

# Try multiple optimizers
result.allFit(data)
```

## Formula Syntax

mixedlm supports lme4-style formula syntax for specifying random effects:

| Syntax | Description |
|--------|-------------|
| `(1 \| group)` | Random intercept |
| `(x \| group)` | Random intercept and slope (correlated) |
| `(x \|\| group)` | Random intercept and slope (uncorrelated) |
| `(1 \| group1/group2)` | Nested random effects |
| `(1 \| group1) + (1 \| group2)` | Crossed random effects |

## API Reference

### Model Fitting

- `lmer(formula, data, REML=True)` - Fit linear mixed model
- `glmer(formula, data, family, nAGQ=1)` - Fit generalized linear mixed model
- `nlmer(formula, data, start)` - Fit nonlinear mixed model

### Result Methods

| Method | Description |
|--------|-------------|
| `fixef()` | Extract fixed effects |
| `ranef(condVar=False)` | Extract random effects (BLUPs) |
| `coef()` | Combined fixed + random effects |
| `VarCorr()` | Variance-covariance of random effects |
| `fitted()` | Fitted values |
| `residuals(type)` | Residuals (response, pearson, deviance) |
| `predict(newdata)` | Predictions |
| `simulate(nsim)` | Simulate responses |
| `confint(method)` | Confidence intervals (Wald, profile, boot) |
| `logLik()` | Log-likelihood with df |
| `AIC()` / `BIC()` | Information criteria |
| `summary()` | Model summary |

### Inference Functions

- `anova(*models)` - Likelihood ratio tests
- `drop1(model, data)` - Single term deletions
- `profile(model, data)` - Likelihood profiles
- `bootstrap(model, data, nsim)` - Parametric bootstrap

### Families (for glmer)

- `Gaussian()` - Normal distribution (identity link)
- `Binomial()` - Binomial distribution (logit link)
- `Poisson()` - Poisson distribution (log link)
- `Gamma()` - Gamma distribution (inverse link)
- `InverseGaussian()` - Inverse Gaussian (1/mu^2 link)
- `NegativeBinomial(theta)` - Negative binomial (log link)

## Requirements

- Python >= 3.14
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This package is inspired by and aims to be compatible with R's lme4 package by Douglas Bates, Martin Maechler, Ben Bolker, and Steve Walker.

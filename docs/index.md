# mixedlm

A Python implementation of mixed-effects models inspired by R's [lme4](https://github.com/lme4/lme4) package. Features a Rust backend for performance-critical operations and native support for both **pandas** and **polars** DataFrames.

## Features

- **Linear Mixed Models (LMM)** via `lmer()` - REML and ML estimation
- **Generalized Linear Mixed Models (GLMM)** via `glmer()` - Laplace approximation and adaptive Gauss-Hermite quadrature
- **Nonlinear Mixed Models (NLMM)** via `nlmer()` - Self-starting models
- **lme4-style formula interface** - `(1 | group)`, `(x | group)`, `(x || group)`, nested and crossed effects
- **Inference tools** - Profile likelihood, parametric bootstrap, Satterthwaite/Kenward-Roger degrees of freedom
- **Model comparison** - ANOVA (including Type III), drop1, allFit
- **Power analysis** - powerSim, powerCurve for sample size planning
- **Diagnostics** - Influence measures, Cook's distance, leverage, residual plots

## Quick Example

```python
import mixedlm as mlm

# Load example data
data = mlm.load_sleepstudy()

# Fit a random intercept and slope model
result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# View results with p-values (Satterthwaite DF)
print(result.summary())

# Extract components
result.fixef()      # Fixed effects
result.ranef()      # Random effects (BLUPs)
result.VarCorr()    # Variance components
result.confint()    # Confidence intervals
```

## Why mixedlm?

| Feature | mixedlm | statsmodels | lme4 (R) |
|---------|---------|-------------|----------|
| Formula syntax | lme4-style | Different | Native |
| GLMM support | Yes (Laplace + AGQ) | Limited | Yes |
| NLMM support | Yes | No | Yes |
| P-values | Satterthwaite/KR | Wald only | lmerTest |
| Power analysis | Built-in | No | simr package |
| DataFrame support | pandas + polars | pandas only | data.frame |

## Installation

```bash
pip install mixedlm
```

See [Installation](getting-started/installation.md) for optional dependencies and building from source.

## Getting Started

- [Installation](getting-started/installation.md) - Install mixedlm and optional dependencies
- [Quickstart](getting-started/quickstart.md) - Fit your first mixed model in 5 minutes
- [Coming from R](getting-started/coming-from-r.md) - Guide for lme4 users

## Documentation

### Tutorials
Step-by-step guides for common workflows:

- [Linear Mixed Models](tutorials/linear-mixed-models.md) - Random intercepts, slopes, nested effects
- [Generalized Linear Mixed Models](tutorials/glmm.md) - Binomial, Poisson, negative binomial
- [Nonlinear Mixed Models](tutorials/nlmm.md) - Self-starting models, custom functions
- [Statistical Inference](tutorials/inference.md) - Testing, confidence intervals, emmeans
- [Power Analysis](tutorials/power-analysis.md) - Sample size planning with simulation

### Background
Mathematical and statistical foundations:

- [Estimation Methods](background/estimation.md) - REML, ML, Laplace, adaptive quadrature
- [Degrees of Freedom](background/degrees-of-freedom.md) - Satterthwaite vs Kenward-Roger
- [Formula Syntax](background/formula-syntax.md) - Complete reference for random effects notation

### API Reference
Complete documentation of all functions and classes:

- [Models](api/models.md) - `lmer`, `glmer`, `nlmer`, control classes
- [Results](api/results.md) - Result objects and their methods
- [Inference](api/inference.md) - `anova`, `emmeans`, `bootMer`, profile likelihood
- [Families](api/families.md) - Distribution families for GLMMs
- [Diagnostics](api/diagnostics.md) - Influence measures and plots
- [Power](api/power.md) - `powerSim`, `powerCurve`, `extend`
- [Datasets](api/datasets.md) - Built-in example datasets
- [Utilities](api/utilities.md) - Helper functions and lme4 compatibility

## License

MIT License - see [LICENSE](https://github.com/cameronlyons/mixedlm/blob/main/LICENSE) for details.

## Acknowledgments

This package is inspired by and aims to be compatible with R's lme4 package by Douglas Bates, Martin Maechler, Ben Bolker, and Steve Walker.

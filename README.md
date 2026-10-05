# mixedlm

[![CI](https://github.com/Cameron-Lyons/mixedlm/actions/workflows/ci.yml/badge.svg)](https://github.com/Cameron-Lyons/mixedlm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mixedlm.svg)](https://pypi.org/project/mixedlm/)
[![Python](https://img.shields.io/pypi/pyversions/mixedlm.svg)](https://pypi.org/project/mixedlm/)
[![codecov](https://codecov.io/gh/Cameron-Lyons/mixedlm/branch/main/graph/badge.svg)](https://codecov.io/gh/Cameron-Lyons/mixedlm)
[![Docs](https://readthedocs.org/projects/mixedlm/badge/?version=latest)](https://mixedlm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of mixed-effects models inspired by R's [lme4](https://github.com/lme4/lme4) package. Features a Rust backend for performance-critical operations and native support for both **pandas** and **polars** DataFrames.

**[Documentation](https://mixedlm.readthedocs.io/)** | **[PyPI](https://pypi.org/project/mixedlm/)** | **[Changelog](https://github.com/Cameron-Lyons/mixedlm/blob/main/CHANGELOG.md)**

## Features

<!-- --8<-- [start:features] -->
- **Linear Mixed Models (LMM)** via `lmer()` - REML and ML estimation
- **Generalized Linear Mixed Models (GLMM)** via `glmer()` - Laplace approximation and adaptive Gauss-Hermite quadrature
- **Nonlinear Mixed Models (NLMM)** via `nlmer()` - Self-starting models (SSasymp, SSlogis, SSmicmen, and more) and custom nonlinear functions
- **Formula interface** - lme4-style formulas with random effects syntax
- **Tidy reporting** - Analysis-ready coefficient, random-effect, and model-fit tables
- **Inference tools** - Linear hypotheses, profile likelihood, bootstrap, confidence intervals, Satterthwaite/Kenward-Roger degrees of freedom
- **Model comparison** - ANOVA (including Type III), drop1, allFit
- **Model selection** - AIC/AICc/BIC rankings, normalized weights, and evidence sets
- **Model validation** - Case-level, whole-group, and custom-partition cross-validation with weighted scoring and leakage checks
- **Prediction uncertainty** - Conditional LMM and GLMM mean intervals with joint fixed/random-effect covariance and prior variance for new groups
- **Power analysis** - powerSim, powerCurve for sample size planning
- **Diagnostics** - Dispersion and zero-inflation checks, influence measures, Cook's distance, leverage, VIF/GVIF, condition indices
- **Fit metrics** - Nakagawa marginal/conditional R² and adjusted/unadjusted ICC
- **Fast startup** - Public objects are loaded on demand, so lightweight imports avoid the modeling stack
<!-- --8<-- [end:features] -->

## Installation

```bash
pip install mixedlm
```

mixedlm requires Python >= 3.10, NumPy >= 1.23.5, SciPy >= 1.14, and pandas >= 1.4.
Optional extras add polars input (`mixedlm[polars]`), plotting with matplotlib
(`mixedlm[plots]`), and nlopt optimizers (`mixedlm[optimizers]`). Building from
source requires a Rust toolchain; see the
[installation guide](https://mixedlm.readthedocs.io/en/latest/getting-started/installation/).

## Quick Start

```python
import mixedlm as mlm

# Random intercept and slope for each subject
data = mlm.load_sleepstudy()
result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(result.summary())  # Includes p-values via Satterthwaite DF

result.fixef()               # Fixed effects
result.ranef()               # Random effects (BLUPs)
result.VarCorr()             # Variance components
result.tidy(conf_int=True)   # Analysis-ready pandas table

# Binomial GLMM with successes / trials responses
cbpp = mlm.load_cbpp()
gm = mlm.glmer(
    "incidence / size ~ period + (1 | herd)", cbpp, family=mlm.families.Binomial()
)
print(gm.summary())
```

## Documentation

- [Quickstart](https://mixedlm.readthedocs.io/en/latest/getting-started/quickstart/) and [Coming from R](https://mixedlm.readthedocs.io/en/latest/getting-started/coming-from-r/)
- [Tutorials](https://mixedlm.readthedocs.io/en/latest/tutorials/linear-mixed-models/) for LMMs, GLMMs, NLMMs, inference, and power analysis
- [API reference](https://mixedlm.readthedocs.io/en/latest/api/models/)
- [Contributing](https://mixedlm.readthedocs.io/en/latest/contributing/), including development setup and CI
- [Changelog](https://github.com/Cameron-Lyons/mixedlm/blob/main/CHANGELOG.md)

## License

The mixedlm code uses the MIT License; see [LICENSE](https://github.com/Cameron-Lyons/mixedlm/blob/main/LICENSE). The bundled lme4
datasets retain their upstream GPL (>= 2) license, with attribution and license
text in [datasets/data](https://github.com/Cameron-Lyons/mixedlm/blob/main/python/mixedlm/datasets/data/README.md).

## Acknowledgments

This package is inspired by and aims to be compatible with R's lme4 package by Douglas Bates, Martin Maechler, Ben Bolker, and Steve Walker.

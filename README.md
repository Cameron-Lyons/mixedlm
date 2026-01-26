# mixedlm

[![CI](https://github.com/cameronlyons/mixedlm/actions/workflows/ci.yml/badge.svg)](https://github.com/cameronlyons/mixedlm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mixedlm.svg)](https://pypi.org/project/mixedlm/)
[![Python](https://img.shields.io/pypi/pyversions/mixedlm.svg)](https://pypi.org/project/mixedlm/)
[![codecov](https://codecov.io/gh/cameronlyons/mixedlm/branch/main/graph/badge.svg)](https://codecov.io/gh/cameronlyons/mixedlm)
[![pre-commit.ci](https://results.pre-commit.ci/badge/github/cameronlyons/mixedlm/main.svg)](https://results.pre-commit.ci/latest/github/cameronlyons/mixedlm/main)
[![Docs](https://readthedocs.org/projects/mixedlm/badge/?version=latest)](https://mixedlm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of mixed-effects models inspired by R's [lme4](https://github.com/lme4/lme4) package. Features a Rust backend for performance-critical operations and native support for both **pandas** and **polars** DataFrames.

**[Documentation](https://mixedlm.readthedocs.io/)** | **[PyPI](https://pypi.org/project/mixedlm/)**

## Features

- **Linear Mixed Models (LMM)** via `lmer()` - REML and ML estimation
- **Generalized Linear Mixed Models (GLMM)** via `glmer()` - Laplace approximation and adaptive Gauss-Hermite quadrature
- **Nonlinear Mixed Models (NLMM)** via `nlmer()` - Self-starting models (SSasymp, SSlogis, SSmicmen)
- **Formula interface** - lme4-style formulas with random effects syntax
- **Inference tools** - Profile likelihood, parametric bootstrap, confidence intervals, Satterthwaite/Kenward-Roger degrees of freedom
- **Model comparison** - ANOVA (including Type III), drop1, allFit
- **Power analysis** - powerSim, powerCurve for sample size planning
- **Diagnostics** - Influence measures, Cook's distance, leverage

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

# Works with pandas
data = mlm.load_sleepstudy()  # Returns pandas DataFrame

# Fit a random intercept and slope model
result = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(result.summary())  # Includes p-values via Satterthwaite DF

# Extract components
result.fixef()      # Fixed effects
result.ranef()      # Random effects (BLUPs)
result.VarCorr()    # Variance components
result.coef()       # Combined coefficients

# Inference
result.confint(method="profile")  # Profile confidence intervals
result.confint(method="boot")     # Bootstrap confidence intervals
```

### Using Polars

```python
import polars as pl
import mixedlm as mlm

# Works directly with polars DataFrames
data = pl.DataFrame({
    "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "group": ["A", "A", "B", "B", "C", "C"],
})

result = mlm.lmer("y ~ x + (1 | group)", data)
print(result.summary())
```

### Generalized Linear Mixed Model

```python
# Load example data
cbpp = mlm.load_cbpp()

# Binomial GLMM
result = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial()
)

# Poisson GLMM with adaptive Gauss-Hermite quadrature
result = mlm.glmer(
    "count ~ x + (1 | group)",
    data,
    family=mlm.families.Poisson(),
    nAGQ=10  # 10-point adaptive quadrature
)
```

### lmerTest-style P-values

```python
# Summary with Satterthwaite degrees of freedom (default)
print(result.summary())

# Or use Kenward-Roger
print(result.summary(ddf_method="Kenward-Roger"))

# Direct access to denominator DF
from mixedlm import satterthwaite_df, pvalues_with_ddf
ddf = satterthwaite_df(result)
pvals = pvalues_with_ddf(result)
```

### Model Comparison

```python
# Fit nested models
model1 = mlm.lmer("y ~ x + (1 | group)", data)
model2 = mlm.lmer("y ~ x + z + (1 | group)", data)

# Likelihood ratio test
mlm.anova(model1, model2)

# Type III ANOVA for a single model
mlm.anova_type3(model1)

# Single term deletions
result.drop1(data)

# Try multiple optimizers
result.allFit(data)
```

### Power Analysis

```python
from mixedlm import powerSim, powerCurve, extend

# Simulate power for detecting an effect
power = powerSim(fitted_model, data, nsim=100, test="fixed")

# Power curve across sample sizes
curve = powerCurve(fitted_model, data, along="n", breaks=[50, 100, 200])

# Extend dataset for larger sample size simulations
extended_data = extend(fitted_model, data, along="n", n=200)
```

### Profile Likelihood

```python
from mixedlm import plot_profiles, slice2D

# 1D profile likelihood
profiles = result.profile(data)
plot_profiles(profiles)

# 2D profile likelihood slice
profile_2d = slice2D(result, "sigma", "theta1", n_points=20)
profile_2d.plot()
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
| `summary(ddf_method)` | Model summary with optional p-values |
| `getME(name)` | Extract model components (X, Z, theta, Lambda, etc.) |
| `get_deviance_components()` | Breakdown of deviance into components |

### Inference Functions

- `anova(*models)` - Likelihood ratio tests between models
- `anova_type3(model)` - Type III ANOVA for single model
- `drop1(model, data)` - Single term deletions
- `profile(model, data)` - 1D likelihood profiles
- `slice2D(model, param1, param2)` - 2D profile likelihood
- `bootMer(model, data, nsim)` - Parametric bootstrap
- `satterthwaite_df(model)` - Satterthwaite denominator DF
- `kenward_roger_df(model)` - Kenward-Roger denominator DF
- `pvalues_with_ddf(model)` - P-values using denominator DF

### Power Analysis

- `powerSim(model, data, nsim)` - Simulate power
- `powerCurve(model, data, along, breaks)` - Power across sample sizes
- `extend(model, data, along, n)` - Extend dataset for simulations

### Families (for glmer)

- `Gaussian()` - Normal distribution (identity link)
- `Binomial()` - Binomial distribution (logit link)
- `Poisson()` - Poisson distribution (log link)
- `Gamma()` - Gamma distribution (inverse link)
- `InverseGaussian()` - Inverse Gaussian (1/mu^2 link)
- `NegativeBinomial(theta)` - Negative binomial (log link)
- `CustomFamily` - Base class for user-defined families

### Datasets

Built-in datasets from lme4:

- `load_sleepstudy()` - Sleep deprivation study
- `load_cbpp()` - Contagious bovine pleuropneumonia
- `load_cake()` - Cake baking experiment
- `load_dyestuff()` / `load_dyestuff2()` - Dyestuff experiments
- `load_penicillin()` - Penicillin assay
- `load_pastes()` - Paste strength
- `load_insteval()` - Instructor evaluations
- `load_arabidopsis()` - Arabidopsis clipping experiment
- `load_grouseticks()` - Grouse tick counts
- `load_verbagg()` - Verbal aggression

## Requirements

- Python >= 3.10
- NumPy >= 1.21
- SciPy >= 1.8
- One of: **pandas** >= 1.4 or **polars** >= 0.20

The package works with either pandas or polars DataFrames. Install whichever you prefer:

```bash
pip install mixedlm pandas   # For pandas support
pip install mixedlm polars   # For polars support
pip install mixedlm pandas polars  # For both
```

### Optional Dependencies

```bash
# For plotting
pip install mixedlm[plots]  # matplotlib

# For additional optimizers
pip install mixedlm[optimizers]  # Py-BOBYQA, nlopt
```

## Development

### Setup

```bash
git clone https://github.com/cameronlyons/mixedlm.git
cd mixedlm
uv venv && uv pip install -e ".[dev]"
maturin develop --release
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=mixedlm

# Run benchmarks
pytest tests/ --benchmark-only

# Run property-based tests
pytest tests/test_property.py
```

### Code Quality

```bash
# Linting and formatting (Python)
ruff check python/ tests/
ruff format python/ tests/

# Type checking
mypy python/

# Rust
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### CI Pipeline

The project uses GitHub Actions for continuous integration:

- **Linting**: ruff, clippy, rustfmt
- **Type checking**: mypy
- **Testing**: pytest across Python 3.10-3.13 (including free-threaded 3.13t)
- **Coverage**: codecov for both Python and Rust
- **Security**: cargo-audit, pip-audit
- **Docs**: mkdocs build verification
- **Benchmarks**: pytest-benchmark
- **Property testing**: hypothesis
- **Memory safety**: Miri
- **Fuzz testing**: cargo-fuzz (weekly)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This package is inspired by and aims to be compatible with R's lme4 package by Douglas Bates, Martin Maechler, Ben Bolker, and Steve Walker.

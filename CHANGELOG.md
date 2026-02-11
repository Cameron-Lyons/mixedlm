# Changelog

All notable changes to mixedlm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- EM-REML now supports multiple random effects and random slopes (correlated and uncorrelated)
- Automatic convergence recommendations in `summary()` output for non-converged and singular fits
- `em_init` control parameter is now wired up in `lmer()` and `glmer()` model fitting
- `em_init` and `em_maxiter` parameters added to `GlmerControl` and `glmerControl()`
- EM-REML initialization section in estimation background documentation
- Root-level CHANGELOG.md

### Changed
- EM-REML algorithm generalized from single random intercept to arbitrary unstructured covariance models

## [1.1.0] - 2026-01-27

### Added
- Comprehensive CI pipeline with testing across Python 3.10-3.13 and free-threaded 3.13t
- Symbolic factorization caching for sparse Cholesky operations
- EM-REML initialization option for variance component estimation
- Miri, property-based testing, and benchmark jobs in CI
- Rust and Python code coverage reporting

### Changed
- Updated faer dependency from 0.23.2 to 0.24.0
- Performance optimizations and code cleanup
- BOBYQA is now the default optimizer
- Enhanced convergence diagnostics and adaptive starting values

### Fixed
- pandas 2.x StringDtype compatibility issues
- Polars compatibility fixes for dataset loaders

## [1.0.0] - 2026-01-14

### Added
- Comprehensive documentation site with MkDocs and Material theme
- Tutorials for LMM, GLMM, NLMM, inference, and power analysis
- Background sections on estimation methods and degrees of freedom
- Full API reference documentation
- ReadTheDocs integration
- Gradient support for linear mixed models
- PyArrayLike support for flexible array input types

### Changed
- Marked as Production/Stable (v1 stability milestone)

## [0.1.1] - 2026-01-13

### Added
- Type III ANOVA via `anova_type3()`
- 2D profile likelihood slices via `slice2D()`
- Satterthwaite and Kenward-Roger degrees of freedom methods
- Power analysis functions: `powerSim()`, `powerCurve()`, `extend()`
- Influence diagnostics: `cooks_distance()`, `dfbeta()`, `dfbetas()`, `dffits()`, `leverage()`
- Diagnostic plots: `plot_diagnostics()`, `plot_qq()`, `plot_ranef()`
- Estimated marginal means via `emmeans()`
- `drop1()` for single term deletions
- `allFit()` to try multiple optimizers
- Nonlinear mixed models via `nlmer()` with self-starting models
- Negative binomial GLMM via `glmer_nb()`
- Polars DataFrame support via narwhals
- Variance transformation utilities
- `checkConv()` and `convergence_ok()` for convergence diagnostics

### Changed
- Improved numerical stability in PIRLS algorithm for GLMMs
- Performance optimizations for profile likelihood
- Enhanced Laplace deviance computation

### Fixed
- Fixed mypy error in profile cache
- Numerical stability in boundary cases for GLMM

## [0.1.0] - 2026-01-12

### Added
- Initial release
- Linear mixed models via `lmer()`
- Generalized linear mixed models via `glmer()`
- REML and ML estimation
- Laplace approximation and adaptive Gauss-Hermite quadrature for GLMMs
- lme4-style formula syntax with random effects
- Distribution families: Gaussian, Binomial, Poisson, Gamma, InverseGaussian, NegativeBinomial
- Basic inference: `anova()`, `confint()`, `bootMer()`
- Profile likelihood confidence intervals
- Built-in datasets from lme4
- Rust backend for performance-critical operations
- pandas DataFrame support

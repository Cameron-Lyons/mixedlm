# Changelog

All notable changes to mixedlm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Formula-driven simulation now supports every built-in response family
- New-data LMM and GLMM predictions accept numeric, scalar, or column-based offsets
- Vectorized AIC/AICc/BIC model rankings with normalized weights and evidence sets
- Nakagawa marginal/conditional R² and adjusted/unadjusted ICC for all model families
- Weighted VIF/GVIF, tolerance, severity, and condition diagnostics for all model types
- Vectorized Pearson dispersion and observed-versus-expected zero diagnostics for GLMMs
- Dependency-free case-level and grouped cross-validation for LMMs and GLMMs
- EM-REML now supports multiple random effects and random slopes (correlated and uncorrelated)
- Automatic convergence recommendations in `summary()` output for non-converged and singular fits
- `em_init` control parameter is now wired up in `lmer()` and `glmer()` model fitting
- `em_init` and `em_maxiter` parameters added to `GlmerControl` and `glmerControl()`
- EM-REML initialization section in estimation background documentation
- Root-level CHANGELOG.md

### Changed
- Vectorized grouping-level factorization and nested-key construction for sparse designs
- EM-REML algorithm generalized from single random intercept to arbitrary unstructured covariance models
- Replaced the Py-BOBYQA dependency and default optimizer with SciPy COBYQA
- Reduced unused and redundant Python and Rust dependencies
- Vectorized pandas nested-group construction for faster large model setup
- Consolidated duplicate CI and security checks while preserving coverage

### Fixed
- Poisson and other unbounded GLMM families no longer clamp fitted means below one
- Unsupported family and link combinations no longer route through the native fast path

### Fixed
- Synthetic dataset loaders no longer reset NumPy's global random state

### Fixed
- Nonlinear mixed models now apply offsets consistently to responses, fitted values, simulations, refits, covariance estimates, and leverage diagnostics

### Fixed
- Formula simulation now preserves global random state and random-effect coefficient ordering
- LMM prediction uncertainty now includes conditional random-effect covariance, fixed/random cross-covariance, correlated slopes, and unseen-group prior variance
- Covariance tables, PCA diagnostics, singularity checks, and parameter bounds now honor compound-symmetry and AR(1) random-effect structures
- Gamma GLMMs now minimize the non-negative unit deviance instead of its negative
- Poisson and other unbounded GLMM families no longer clamp fitted means below one
- Unsupported family and link combinations no longer route through the native fast path

## [1.1.0] - 2026-01-27

### Added
- Comprehensive CI pipeline with testing across Python 3.10-3.13, plus free-threaded 3.14t
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

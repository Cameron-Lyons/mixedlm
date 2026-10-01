# Changelog

All notable changes to mixedlm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LMM fitting accepts `use_analytic_gradient=True` through `lmerControl()`, `LMMOptimizer.optimize()`, and `optimizeLmer()`. Supported native gradient-based solvers share value/gradient evaluations within each fit. The option is disabled by default because its benefit depends on the model and optimizer, and large coupled random-effect systems can make analytic gradients more expensive than numerical derivatives.

- Prepared native LMM responses expose `deviance_with_gradient(theta, reml=True)`, reusing validated design and response products for analytic covariance gradients. Calls snapshot parameters and release the interpreter lock, allowing concurrent evaluations and direct use with SciPy's `jac=True` interface.

- Bootstrap results expose ordered, immutable `BootstrapFailure` records with sample indices, failure stages, and error messages. Summaries include stage counts, and custom simulated responses are validated before refitting.

- Nonlinear bootstrap and bootstrap confidence intervals accept `n_jobs` for parallel refits. `bootMer()` honors the worker count for nonlinear models, preserving seeded samples and failure handling. Nonlinear bootstrap also accepts reusable NumPy random streams.

- Nonlinear fits accept `pnls_maxiter` and `pnls_tol` for inner iteration control on both backends. Results expose `pnls_converged` and retain the controls for refits and updates.

- `LmerControl` and `GlmerControl` accept `restart_edge` to control likelihood checks and optimizer restarts at zero variance; enabled by default.

- `slice2D(..., profile_covariance=True)` computes a full joint ML likelihood profile, with nuisance covariance and scale optimization, adaptive grid ranges, and parallel row evaluation.

- `nAGQ=0` exposes the previous fast joint-PIRLS GLMM approximation; `nAGQ0initStep` now controls preliminary covariance optimization for joint fits.

- GLMM fitting accepts `pirls_maxiter` to limit inner iterations independently of outer optimization; fitted results and refits retain the inner controls

- New-data LMM prediction intervals accept scalar, array, or column-based residual precision weights.
- `ggpredict()` accepts one known link-scale offset per prediction-grid row.
- Nonlinear new-data predictions accept scalar, array, or column-based response offsets.

### Changed

- Native LMM gradients use one shared adjoint solve for the conditional-mode contribution across covariance parameters, retaining the correction for numerical error at large variances. This removes per-parameter mode solves and covariance-factor transforms for both ML and REML.

- Native REML gradients share a fixed-effect solve and projected crossproduct across covariance parameters, then contract only the selected factor entries. This avoids an explicit fixed-effect information inverse and per-parameter random-by-fixed matrix products. Independent levels retain compact block products.

- Native LMM gradients use compact per-level inverses and crossproducts when the weighted design separates across all levels and grouping structures. Eligibility is cached from the design, including exact checks for tiny couplings. These gradient temporaries use per-level block sizes; the prepared design still stores the full crossproduct.

- Native LMM covariance assembly transforms complete grouping blocks in place, avoiding temporary matrices for each pair of levels. Independent-level blocks retain their compact representation and skip redundant copies.

- Native LMM gradients multiply crossproducts by covariance factors directly, avoiding an extra full-matrix transpose and allocation. The same column-oriented transform is shared with penalized covariance construction.

- Native LMM blocked solves reuse one owned workspace and optimized dense triangular solves. Covariance gradients build the inverse in its final buffer, reducing temporary matrix allocation for models with many random-effect coefficients.

- Native LMM analytic gradients compute covariance-derivative traces and products directly from the affected factor rows, avoiding a full square derivative matrix for each parameter. ML gradients also skip the fixed-effect correction solve used only by REML.

- Native LMM fits and response refits extract final estimates from their prepared design products, avoiding a second Python crossproduct cache. Final residual scale uses conditional residuals plus the spherical random-effect penalty, and extraction can run concurrently across Python threads. Fixed-only fits retain their existing solve and least-squares fallback.

- Native sparse input parsing reuses converted row-index and column-offset buffers for canonical CSC matrices and reserves conversion storage once, reducing temporary allocation during model preparation and sparse solves.

- Native LMM design preparation releases Python's interpreter lock after snapshotting its inputs, allowing weighted crossproduct preparation to overlap across independent fits.

- Prepared native LMM likelihoods release Python's interpreter lock during ML and REML evaluation. Calls snapshot covariance parameters and keep solve state local, allowing concurrent evaluations of shared designs and responses.

- Prepared native GLMM likelihoods release Python's interpreter lock during evaluation, allowing independent solves from Python threads. Per-call covariance parameters and offset overrides are copied before release, and scratch state remains local to each solve.

- Native GLMM iterations and final mode calculations reuse a linear-predictor buffer. Mode-only updates start directly from the fixed-effect offset, avoiding empty matrix multiplication, and adaptive quadrature uses the same predictor update.

- Native binomial/logit iterations use a specialized working-value loop that enables compiler vectorization, preserving the existing arithmetic and weight floors.

- Native GLMM iterations reuse working buffers and avoid intermediate derivative and variance vectors. Mode-only solves, including joint likelihood evaluations, skip empty fixed-effect matrix construction and factorization.

- Modular GLMM deviance callables reuse joint likelihood preparation across `[theta, beta]` evaluations, refreshing it when the optimizer or solver settings change. Covariance-only calls do not construct a joint objective.

- Native adaptive quadrature runs on the calling thread when only one worker is available. Parallel evaluations collect group contributions in a fixed order for compensated summation, preserving small contributions and making the quadrature reduction reproducible across worker counts.

- Native GLMM fitting and joint likelihood profiles reuse owned input preparation and starting coefficients across parameter evaluations. Each solve remains independent, including when its fixed-effect offset changes.

- Native GLMM likelihoods reuse final PIRLS means for Laplace and adaptive quadrature corrections. Laplace also reuses the covariance factor, and each iteration shares link derivatives between working weights and responses.

- Nonlinear prediction uses stable group row indexing, and nonlinear bootstrap reuses simulation preparation across responses. Seeded draw order, custom simulation overrides, and per-replicate failure handling are preserved.

- Native LMM likelihood evaluations reuse weighted design products across covariance steps. Linear bootstrap shares this preparation across responses, with an independent workspace per worker. `LMMOptimizer.with_response()` supports the same reuse on both backends.

- Parallel LMM and GLMM bootstrap reuse model data within each worker and bound queued tasks, reducing serialization and scheduling memory. Worker counts are validated before drawing seeds and capped at the number of replicates.

- Nonlinear fitting uses joint fixed/random parameter updates with a backtracking line search on both backends. Group systems keep the solves small, and Python consumes group results with bounded worker queues.

- Python nonlinear fits reuse group and weight preparation across covariance evaluations, share one worker pool per fit, and compute final residual variance once per evaluation.

- Python LMM likelihood evaluation uses scalar solves for diagonal random-effect systems, avoiding covariance-matrix assembly and factorization during fitting and likelihood profiling.

- Native GLMMs with diagonal random-effect precision use scalar solves and log determinants at every model size, reusing the prepared design across PIRLS iterations.

- `ggpredict()` and `allEffects()` now default to the unweighted mean of fitted
  link-scale offsets after missing-value omission. Pass `offset=0` for the previous
  behavior, including per-unit rates from models fitted with log-exposure offsets.

### Fixed

- COBYLA fits return results when SciPy supplies an evaluation count without an iteration count. The normalized iteration count uses evaluations, matching COBYLA's `maxiter` budget units. Boundary probes and restarts share that evaluation budget, retaining the best probe when too few evaluations remain to initialize a restart.

- `optimizeLmer()` applies the stored control's convergence tolerances and `optCtrl` options to the requested solver, including evaluation limits and normalized option aliases. Explicit `optCtrl` entries override generated options without changing the stored control.

- `optimizeLmer()` honors the `restart_edge` control supplied to `mkLmerDevfun()` when no override is given. Explicit booleans override the control for one fit; `None` inherits it, and manually constructed deviance callables without a control retain enabled checks.

- Variance-boundary restarts check small positive covariance scales as well as zero. This recovers better likelihoods when numerical derivatives stop just outside the old boundary threshold, while retaining genuine small estimates and the remaining optimization budget.

- `trust-constr` reports the objective gradient for convergence summaries and accepts the same one-argument iteration callbacks as other SciPy optimizers.

- Native LMM likelihoods, final estimates, and covariance gradients preserve weighted crossproducts between levels of the same random-effect structure. This corrects advanced designs with overlapping level columns; ordinary grouped designs retain their specialized block solves.

- GLMM optimizers, joint likelihood objectives, and modular deviance callables can be deep-copied and pickled with native preparation enabled. Restored objects rebuild the cache using the available backend and preserve model inputs and solver controls.

- Native GLMM entry points reject inconsistent array dimensions, covariance parameter counts, and random-effect metadata with `ValueError`, preventing Rust panics and silently ignored input. Dimension arithmetic and sparse column-pointer lengths are checked for overflow.

- Bootstrap refits require convergence and finite real estimates of the expected shapes in both serial and parallel execution. Failed samples stay entirely missing, and all confidence interval methods require at least two valid samples per parameter.

- Nonlinear fits no longer report convergence when the inner PNLS iteration limit is reached. Summaries identify unfinished inner solves, and nonlinear bootstrap intervals exclude unconverged refits.

- LMM and GLMM fitting, modular optimization, and profiles detect zero variance scales with a zero gradient but a better nearby likelihood. Restarts retain the selected optimizer, share its remaining budget, and report nonconvergence if an improvement cannot be resolved.

- LMM fixed-effect profile intervals now re-optimize nuisance covariance and residual scale using ML, including for REML inputs. Failed fits and unbracketed intervals raise instead of substituting Wald limits; interval extraction skips unnecessary plotting points.

- GLMM fits with `nAGQ>=1` now jointly optimize fixed coefficients and covariance parameters against the integrated likelihood; modular fits, refits, reconstructed objectives, and profiles use the same objective. Default estimates can change and fitting may take longer; `nAGQ=0` preserves the previous algorithm.

- GLMM profile confidence intervals now use constrained likelihood fits and likelihood-ratio cutoffs, with nuisance covariance and fixed coefficients re-optimized, instead of returning Wald intervals.

- `GlmerControl.tolPwrss` now governs PIRLS stopping in native and Python likelihoods, quadrature, modular fitting, and final extraction; invalid inner controls are rejected before solving

- Prediction offsets reject complex, masked, and non-finite values before evaluating
  a model. Scalar offsets use constant-size storage even for large prediction grids.

- LMM and GLMM predictions collect Polars lazy queries once, projecting to prediction
  columns and preserving row order and intercept-only grid sizes. Conditional lazy
  predictions no longer fail when checking the number of rows.

## [1.2.0] - 2026-08-18

### Added
- Modular GLMM fitting accepts `nAGQ` when creating the deviance function and carries the quadrature setting through optimization and result construction
- `tidy()` and `glance()` analysis-ready reports for linear, generalized, and nonlinear fits
- Arbitrary linear fixed-effect hypothesis tests with named or matrix constraints
- Grouped-binomial `successes / trials` responses with automatic trial weights and validation
- Configurable named links and documented family/link helper APIs
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
- Formula matrix construction now reuses base-factor encodings and streams fixed-effect interaction columns into a single output matrix

- Prediction reuses an already aligned fixed-effect matrix without copying every column

- LMM summaries reuse one denominator-DF calculation and evaluate coefficient p-values together

- Multi-draw nonlinear simulation reuses covariance, group, and residual setup and accepts reusable NumPy random streams

- Nonlinear objective evaluations reuse group row indices; threaded Python updates share the covariance inverse and group calculations

- Native sparse solves batch all right-hand sides and reuse the result buffer, eliminating per-column allocations and redundant solve setup
- Adaptive quadrature evaluates each integration point on its own group and computes scalar curvature directly, reducing repeated full-model work and parallel memory use
- Quadrature fitting and helpers reuse bounded, immutable rule caches; public rule arrays remain independently writable
- Multi-draw GLMM simulation now batches random effects and response generation
- Vectorized grouping-level factorization and nested-key construction for sparse designs
- EM-REML algorithm generalized from single random intercept to arbitrary unstructured covariance models
- Replaced the Py-BOBYQA dependency and default optimizer with SciPy COBYQA
- Reduced unused and redundant Python and Rust dependencies
- Top-level public exports now load on demand to reduce startup time and memory use
- Vectorized pandas nested-group construction for faster large model setup
- Consolidated duplicate CI and security checks while preserving coverage

### Fixed
- New-data LMM and GLMM predictions preserve distinct fixed-effect columns with colliding display names, including after rank reduction and refitting

- Fixed-effect tables and summaries preserve repeated coefficient names and keep inference aligned by position; ambiguous named selections and dictionary results raise clear errors

- Nonlinear bootstrap confidence intervals exclude failed refits instead of substituting original estimates, validate requests before refitting, and share sample handling with `bootstrap_nlmer()`

- Nonlinear simulation and bootstrap preserve NumPy's global random state while retaining integer-seeded draw sequences

- Python nonlinear fitting honors arbitrary integer group labels and matches native sorted-label ordering

- Python nonlinear least squares checks random-effect changes against the previous iteration, preventing premature convergence when fixed parameters are already stationary

- Nonlinear fits and refits now reject invalid final evaluations instead of reporting convergence with a failure penalty and starting estimates

- Nonlinear fitting and refitting preserve custom prediction and gradient methods instead of selecting a built-in native formula by display name

- Sparse-cache correctness tests no longer depend on single-run timing comparisons; paired cached and uncached measurements now use the benchmark suite

- Linear and generalized model simulation and parametric bootstrap preserve NumPy's global random state and accept reusable random streams without changing integer-seeded draws

- Linear and generalized fits, refits, and modular result constructors reject invalid final estimates instead of returning nonfinite values or fabricated coefficients; reported deviance comes from the validated final evaluation

- GLMM quadrature validates positive integer orders and rejects unsupported random-effect structures; modular results cannot relabel a fit with a different quadrature order
- Adaptive quadrature includes observations with zero random-effect design rows and rejects overlapping group designs that cannot be integrated independently
- Python quadrature uses stable high-order rules shared by `GHrule`, `GQN`, and `GQdk`; invalid orders and tensor dimensions raise clear errors
- Native GLMM starting coefficients use weighted response means on the link scale with offsets removed, preventing large Poisson counts from exhausting PIRLS iterations or overflowing; nonfinite PIRLS estimates no longer report convergence
- GLMM fitting, refitting, and modular assembly include inner PIRLS convergence in the reported status; `pirls_converged` identifies inner failures and summaries and warnings explain them
- Poisson and other unbounded GLMM families no longer clamp fitted means below one
- Unsupported family and link combinations no longer route through the native fast path

- LMM profiling now applies fitted random effects before computing the penalized residual sum of squares, preventing variance estimates from being biased toward zero
- LMM likelihood reporting and fixed-effect profiles now use the corrected profiled criterion consistently
- Synthetic dataset loaders no longer reset NumPy's global random state
- Nonlinear mixed models now apply offsets consistently to responses, fitted values, simulations, refits, covariance estimates, and leverage diagnostics

### Fixed
- Canonical Gamma and inverse-Gaussian variants now simulate from their family distributions
- LMM and GLMM simulations now preserve model offsets in generated responses
- Formula simulation now preserves global random state and random-effect coefficient ordering
- Nonlinear mixed-model optimization now uses a deterministic profiled Laplace deviance with consistent relative covariance scaling
- GLMM PIRLS, Laplace, and adaptive-quadrature calculations now use the covariance factor
  in spherical random-effect coordinates, with consistent likelihood normalization and
  deterministic outer optimization
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

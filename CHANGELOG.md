# Changelog

All notable changes to mixedlm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cross-validation accepts reusable train/test partitions and buffered holdouts, validating row coverage and group leakage before refitting.
- Native sparse Cholesky exposes `ordering="amd"` or `"natural"` and `factor_nonzeros()` to inspect fill-in. The one-shot `sparse_cholesky_solve()` and `sparse_cholesky_logdet()` accept the same keyword-only `ordering`; `"natural"` skips AMD analysis for systems that are already well ordered.
- GLMM conditional prediction standard errors include joint fixed/random-effect uncertainty. Predictions for allowed new grouping levels include the fitted prior random-effect variance; response-scale intervals transform the link-scale limits.
- Binomial GLMMs accept two-level factor responses, preserve the success level through refits, updates, and cross-validation, and return numeric simulations using the fitted encoding.
- Power curves now support actual group-count and within-group sample-size changes, absolute coefficient values, and per-point simulation diagnostics. Power results expose `n_failed` for excluded simulations.
- Nonlinear fits accept `pnls_maxiter` and `pnls_tol` for inner iteration control on both backends. Results expose `pnls_converged` and retain the controls for refits and updates.
- `LmerControl` and `GlmerControl` accept `restart_edge` to control likelihood checks and optimizer restarts at zero variance; enabled by default.
- `slice2D(..., profile_covariance=True)` computes a full joint ML likelihood profile, with nuisance covariance and scale optimization, adaptive grid ranges, and parallel row evaluation.
- `nAGQ=0` exposes the previous fast joint-PIRLS GLMM approximation; `nAGQ0initStep` now controls preliminary covariance optimization for joint fits.
- GLMM fitting accepts `pirls_maxiter` to limit inner iterations independently of outer optimization; fitted results and refits retain the inner controls.
- New-data LMM prediction intervals accept scalar, array, or column-based residual precision weights.
- `ggpredict()` accepts one known link-scale offset per prediction-grid row.
- Nonlinear new-data predictions accept scalar, array, or column-based response offsets.
- Modular GLMM fitting accepts `nAGQ` when creating the deviance function and carries the quadrature setting through optimization and result construction.
- LMM fitting accepts `use_analytic_gradient=True` through `lmerControl()`, `LMMOptimizer.optimize()`, and `optimizeLmer()` for an explicitly chosen L-BFGS-B, BFGS, TNC, SLSQP, or trust-constr optimizer. Supported native gradient-based solvers share value/gradient evaluations within each fit. The option is off by default; the default `"auto"` optimizer always uses exact native gradients where they are available.
- Prepared native LMM responses expose `deviance_with_gradient(theta, reml=True)`, reusing validated design and response products for analytic covariance gradients. Calls snapshot parameters and release the interpreter lock, allowing concurrent evaluations and direct use with SciPy's `jac=True` interface.
- Bootstrap results expose ordered, immutable `BootstrapFailure` records with sample indices, failure stages, and error messages. Summaries include stage counts, and custom simulated responses are validated before refitting.
- Nonlinear bootstrap and bootstrap confidence intervals accept `n_jobs` for parallel refits. `bootMer()` honors the worker count for nonlinear models, preserving seeded samples and failure handling. Nonlinear bootstrap also accepts reusable NumPy random streams.
- `allFit()` accepts `n_jobs` and `control`; the control settings apply to every optimizer. The `allFit()` and `drop1()` methods of linear and generalized results accept `n_jobs`.
- `LmerResult.control` holds the fitting controls. `drop1()`, `allFit()`, and `update()` refit with them, and `refit()` and `refitML()` keep their `use_rust` setting.
- `LmerResult.optimizer`, `GlmerResult.optimizer`, and `GlmerResult.message` record the optimizer used for each fit, including modular fits; `checkConv()` reports them.
- NLopt optimizers (`nloptwrap_*`) accept the `optCtrl` options `initial_step` (default 0.5, at most a quarter of any finite bound range), a relative `ftol`, `ftol_abs`, and `xtol_abs`.

### Changed

- `lmer()` defaults to `optimizer="auto"`: L-BFGS-B with exact native covariance gradients. When that fit does not converge, ends with a large gradient, or leaves a variance scale of a correlated term near zero (any scale with `restart_edge=False`), COBYQA refits from the same start and the lower deviance is kept. Fits without native gradients (`use_rust=False`, no native extension, or `"cs"`/`"ar1"` covariances) use COBYQA as before. Most fits are faster with estimates that agree with COBYQA to optimizer tolerance; fits that need the fallback run both optimizers. `result.optimizer` records the method whose estimates were kept, `optCtrl` applies to the COBYQA stage, and an evaluation limit also bounds the L-BFGS-B stage. Pass `optimizer="COBYQA"` for the previous behavior. `glmer()` keeps COBYQA, and `GlmerControl` rejects `"auto"`.
- Linear refits evaluate the likelihood natively like the original fit: `refit()`, `refitML()`, `as_function()`, `anova()`, `drop1()`, and `powerSim()` no longer use the Python likelihood for models with 50 or more random effects, and `refit()`, `refitML()`, and parametric bootstrap refits use the `"auto"` optimizer.
- Satterthwaite and Kenward-Roger degrees of freedom and LMM likelihood profiles (`confint(method="profile")`, `profile()`, and `slice2D(profile_covariance=True)`) use the native likelihood with exact gradients; `"cs"` and `"ar1"` covariances keep the Python likelihood. Profiling holds one prepared design at a time, so memory does not grow with the number of profiled coefficients.
- Fixed-effect covariance, summaries, predictions, `hatvalues()`, `ranef(condVar=True)`, denominator degrees of freedom, and the Python likelihood factor random-effect systems with at least 256 random effects through the native AMD sparse Cholesky instead of SciPy's SuperLU. `hatvalues()` and prediction standard errors and intervals read the needed inverse entries instead of solving once per row, which makes them much faster for large models.
- Parallel inference never forks the calling process. Worker processes start through forkserver on Linux and spawn on macOS and Windows, so scripts that pass `n_jobs` greater than one need an `if __name__ == "__main__":` guard, and custom families, models, and arguments sent to workers must be importable and picklable.
- `n_jobs` follows one rule in bootstrap, `drop1()`, `allFit()`, likelihood profiles, `slice2D()`, `cross_validate()`, and nonlinear and quadrature workers: `-1` uses every CPU, a positive integer is a worker count capped at the number of tasks, and 0, other negative values, booleans, and non-integers raise `TypeError` or `ValueError`. Previously `slice2D()` silently ran serially for 0 or negative values and `allFit()` accepted `True`.
- Worker processes start with one BLAS, OpenMP, and native thread unless `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, or `RAYON_NUM_THREADS` is set, avoiding oversubscription. On Linux, mixedlm sets the forkserver preload list to `["mixedlm.inference"]`, and a forkserver it starts keeps these limits and imports for later forkserver pools in the same interpreter, including user-created ones.
- A single parallel task, such as one bootstrap replicate or one profiled coefficient, runs in the calling process instead of starting a one-worker pool. A conditional `slice2D()` stays serial unless its remaining rows would take about a second or more.
- `cross_validate(n_jobs=...)` refits folds in worker processes instead of threads and accepts `-1`. Results match a serial run; warnings raised by worker refits are not repeated in the calling process.
- NLopt optimizers stop on an absolute change in deviance, as lme4's `nloptwrap` does: the control's `ftol` (default `1e-8`) is NLopt's `ftol_abs`, and `optCtrl={"ftol": ...}` adds a relative tolerance. The former relative tolerance stopped some fits up to 0.1 deviance short of the optimum; fits can now take more evaluations.
- `checkConv()` reports the optimizer and iteration count recorded on the fit and appends the optimizer's message for non-converged fits. The gradient check applies to gradient-based optimizers, uses the final gradient norm per observation (`grad_tol`), and skips fits on a variance boundary.
- `Family.simulate(mu, rng=None, *, weights=None, trials=None)` is the single response sampler used by `simulate()`, `bootMer()`, `powerSim()`, and `simulate_formula()`. Prior weights act as precisions for Gaussian, gamma, and inverse Gaussian families, `trials` gives grouped binomial counts, and Binomial subclasses keep their trial counts. Overrides with the older `simulate(mu, rng)` signature still work.
- Families without a response distribution, such as `QuasiFamily` and `CustomFamily` subclasses that do not implement `simulate()`, raise `NotImplementedError` from `simulate()`, `bootMer()` and `bootstrap_glmer()` (before any refit), and `powerSim()` instead of drawing `mu + N(0, 0.1**2)`.
- `powerSim()` and `powerCurve()` count only failed refits, including named Wald tests that a degenerate refit cannot support, in `n_failed`. Exceptions raised by a callable `test` and by response simulation now propagate, and a non-Boolean test result raises `TypeError`.
- `simulate_formula()` and `quickSimulate()` accept family names (`"gaussian"`, `"binomial"`, `"poisson"`, `"gamma"`, `"inverse_gaussian"`, and R spellings such as `"Gamma"`) and raise `ValueError` for unknown names. `simulate_formula()` no longer requires a response column, draws grouped binomial successes from the trials column, treats `sigma` as the gamma and inverse Gaussian dispersion, rejects unknown `beta` names, and requires `sigma > 0`.
- `logProf()`, `sdProf()`, and `varianceProf()` raise `ValueError` for profiles outside their domain instead of clamping values.
- `BootstrapResult.ci()`, `bootCI()`, `linear_hypothesis()`, `tidy(conf_level=...)`, `emmeans()`, and its contrasts validate confidence levels the same way: `TypeError` for non-numbers, including strings and booleans, and `ValueError` outside (0, 1).
- `mkNewReTrms()` encodes new data with the original design, so factor slopes, nested and interaction grouping factors, and repeated grouping factors produce correct columns; missing columns raise. `mkParsTemplate()` labels compound-symmetry and AR(1) parameters, and `mkDataTemplate()` and `mkMinimalData()` read variables with the formula parser and keep every level in unbalanced templates.
- `dummy()` keeps the category order of pandas categoricals, its polynomial and Helmert codings match `contr_poly()` and `contr_helmert()`, and `base` accepts negative indices and raises `ValueError` for out-of-range indices or unknown levels.
- The `allFit()` method of linear and generalized results tries every optimizer from `available_optimizers()`, which adds COBYLA and trust-constr to the previous defaults. `AllFitResult.is_consistent()` compares only converged fits.
- `devcomp()` no longer reports `logLik` and raises `TypeError` for nonlinear models.
- Native LMMs with nested or crossed random effects no longer build a dense random-effect crossproduct. They choose a sparse AMD-ordered or blocked factorization by estimated fill and compute gradients from a selected inverse, which makes nested and regularly crossed designs much faster and lowers peak memory for large crossings. Independent random intercepts and slopes use dedicated per-level kernels. Parameter, `VarCorr()`, `ranef()`, and `getME()` order is unchanged.
- The native nonlinear likelihood is several times faster per evaluation. Native GLMMs factor dense random-effect systems below 1,024 random effects on the calling thread instead of the global thread pool, which speeds up small nested, random-slope, and densely crossed fits on busy machines; prepared sparse GLMMs reuse their sparsity analysis between evaluations; and cached AMD sparse solves use a single permuted solve. Results agree to rounding.
- Native GLMM (`glmm_deviance`) and nonlinear (`nlmm_deviance_with_status`) evaluations release the GIL during the solve after copying their inputs.
- Accessing `mixedlm.lmer` or `mixedlm.glmer` no longer imports `scipy.stats`; it loads when summaries, intervals, or p-values need it.
- NumPy 1.23.5 is the minimum supported version, the effective floor of SciPy 1.14. Package metadata includes the README as the long description and project links.
- `ggpredict()` and `allEffects()` now default to the unweighted mean of fitted link-scale offsets after missing-value omission. Pass `offset=0` for the previous behavior, including per-unit rates from models fitted with log-exposure offsets.
- GLMM fits with `nAGQ>=1` now jointly optimize fixed coefficients and covariance parameters against the integrated likelihood; modular fits, refits, reconstructed objectives, and profiles use the same objective. Default estimates can change and fitting may take longer; `nAGQ=0` preserves the previous algorithm.
- Public sparse Cholesky uses AMD ordering by default and releases the GIL during analysis, factorization, solves, and log determinants. Detached operations snapshot Python inputs and reuse the final solve buffer.
- Random-slope R² and ICC evaluate their covariance projections in bounded chunks.
- The native RNG dependency uses the non-yanked chacha20 0.10.2 patch release.
- Native and Python random-effect simulations use compact scale vectors for independent coefficients. Native correlated draws reuse output storage, and native simulation releases the GIL while drawing batches.
- Blocked Cholesky solves skip zero contributions, and dense Schur updates accumulate into their destination without allocating full products.
- Native LMM factorization consumes its assembled random-effect blocks directly, avoiding duplicate working matrices during likelihood and gradient evaluations.
- Native weighted crossproducts reuse row layouts for wide, partially dense random-effect designs, reducing repeated GLMM weight-update work while preserving accumulation order.
- Native Gaussian GLMMs with the identity link reuse a constant working factor across PIRLS iterations and the final likelihood correction.
- Native LMM factorization reuses its working covariance buffers, removing extra matrix copies for both independent levels and coupled random effects.
- Parallel LMM and GLMM bootstrap reuse model data within each worker and bound queued tasks, reducing serialization and scheduling memory. Worker counts are validated before drawing seeds and capped at the number of replicates.
- Nonlinear fitting uses joint fixed/random parameter updates with a backtracking line search on both backends. Group systems keep the solves small, and Python consumes group results with bounded worker queues.
- Python nonlinear fits reuse group and weight preparation across covariance evaluations, share one worker pool per fit, and compute final residual variance once per evaluation.
- Python LMM likelihood evaluation uses scalar solves for diagonal random-effect systems, avoiding covariance-matrix assembly and factorization during fitting and likelihood profiling.
- Native GLMMs with diagonal random-effect precision use scalar solves and log determinants at every model size, reusing the prepared design across PIRLS iterations.
- Formula matrix construction now reuses base-factor encodings and streams fixed-effect interaction columns into a single output matrix.
- Prediction reuses an already aligned fixed-effect matrix without copying every column.
- LMM summaries reuse one denominator-DF calculation and evaluate coefficient p-values together.
- Multi-draw nonlinear simulation reuses covariance, group, and residual setup and accepts reusable NumPy random streams.
- Nonlinear objective evaluations reuse group row indices; threaded Python updates share the covariance inverse and group calculations.
- Native sparse solves batch all right-hand sides and reuse the result buffer, eliminating per-column allocations and redundant solve setup.
- Adaptive quadrature evaluates each integration point on its own group and computes scalar curvature directly, reducing repeated full-model work and parallel memory use.
- Quadrature fitting and helpers reuse bounded, immutable rule caches; public rule arrays remain independently writable.
- Native LMM covariance assembly reuses one intermediate transform per independent random-effect structure, reducing small-matrix allocations during likelihood and gradient evaluations.
- Prepared native LMM designs store independent level crossproducts in compact blocks, avoiding a full square random-effect matrix during preparation and subsequent likelihood and gradient evaluations. Designs with rows spanning levels or grouping structures retain the general path, including tiny couplings and stored zeros. Supplied independent cached products also use compact internal storage.
- Native LMM gradients share projected mode and adjoint vectors for random-effect structures wider than an intercept and slope. Each mode derivative then contracts only the selected factor entries, avoiding per-parameter derivative vectors and scans over all random-effect coefficients. Narrow structures retain direct contractions and their optimizer stopping behavior.
- Native LMM gradients use one shared adjoint solve for the conditional-mode contribution across covariance parameters, retaining the correction for numerical error at large variances. This removes per-parameter mode solves and covariance-factor transforms for both ML and REML.
- Native REML gradients share a fixed-effect solve and projected crossproduct across covariance parameters, then contract only the selected factor entries. This avoids an explicit fixed-effect information inverse and per-parameter random-by-fixed matrix products. Independent levels retain compact block products.
- Native LMM gradients use compact per-level inverses and crossproducts when the weighted design separates across all levels and grouping structures. Eligibility is cached from the design, including exact checks for tiny couplings.
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

### Deprecated

- `LmerResult.fe_params`, `re_params`, `fittedvalues`, and `resid` emit `DeprecationWarning`; use `beta`, `theta`, `fitted()`, and `residuals()`.
- `Family.clip_mu()` is deprecated; use `clamp_mu(mu, eps, out=mu)`.
- `mixedlm.estimation.profiled_reml()` is deprecated; use `profiled_deviance(theta, matrices, REML=True)`.

### Removed

- Undocumented private native functions: `mixedlm._rust.mm_reml`, `augmented_ai_reml`, `riemannian_reml`, `profiled_deviance`, `profiled_deviance_cached`, `profiled_deviance_with_gradient`, `compute_ztwz`, `pirls`, `laplace_deviance`, `adaptive_gh_deviance`, `pnls_step`, `nlmm_deviance`, `gauss_hermite`, `adaptive_gauss_hermite_1d`, `compute_zu`, and `update_cholesky_factor`. Use the prepared `LmmDesign(...).with_response(y)` likelihood, `glmm_deviance`, `nlmm_deviance_with_status`, or the Python estimation functions.
- `mixedlm.estimation.laplace.laplace_deviance_fast()` and `adaptive_gh_deviance_fast()`, which duplicated `glmm_deviance_with_status()`, and the private `mixedlm.estimation.reml.profiled_deviance_fast()`.
- `mixedlm.inference.allfit.SCIPY_OPTIMIZERS`; use `mixedlm.estimation.available_optimizers()`. Unused internal helpers `utils.dataframe.get_row_value`, `DataFrameLike`, `utils.contrasts.ContrastsSpec`, and `NAInfo.n_complete` are also removed.

### Fixed

- NLopt optimizers no longer report convergence when they stop at the evaluation or time limit, and they use true infinite bounds, so PRAXIS no longer reports convergence at enormous variance parameters. Non-finite objective values no longer make NLopt COBYLA loop forever, and failures return the best point evaluated.
- NLopt optimizers set an explicit initial step, so BOBYQA accepts bound ranges narrower than one and NEWUOA and Nelder-Mead no longer stop at a bound or at the start.
- `optimizer="COBYLA"` works for linear mixed models with more than one covariance parameter.
- Starting values for compound-symmetry, AR(1), and uncorrelated (`||`) covariances were in response units. They are now dimensionless, and NLopt Nelder-Mead and PRAXIS no longer converge to a correlation of one on structured covariances.
- Parallel inference with `n_jobs` greater than one no longer hangs on Linux after a native fit, such as a correlated-slope LMM or an `nAGQ > 1` GLMM.
- Native fits no longer hang in a process forked after a native fit, whether by `os.fork()`, a multiprocessing `"fork"` pool, or a pre-forking server; forked children run native kernels on one thread.
- On Python 3.14, a script that fits a model at import time and runs parallel inference under its `__main__` guard no longer hangs.
- `allFit(..., control=...)` no longer fails every fit, and failed optimizers record an empty warnings list in serial runs as in parallel runs.
- `getME("u")` returns the spherical random effects `u`, with `b = Lambda @ u` as in lme4, instead of a copy of `getME("b")`; singular fits give the minimum-norm solution.
- `devcomp()` and `getME("devcomp")` return the same lme4-style components. LMM `wrss` uses the prior weights and GLMM `wrss` is the Pearson sum of squares; `ussq` uses the spherical random effects; `ldRX2` is reported for ML fits; LMMs report `sigmaML` and `sigmaREML`; GLMM `ldL2`, `ldRX2`, `pwrss`, and `drsum` are computed instead of 0.0; components a model does not define are NaN; and `dims["ngrps"]` counts grouping factors.
- `fortify()` keeps fitted values on their rows when observations were dropped for missing values, giving dropped rows NaN, and data of any other length raises `ValueError` instead of being truncated. `include_re=False` gives population-level `.fitted`, and `.fixed` includes the offset.
- GLMM `cooks_distance()` and `influence()` work with `na_action="exclude"`. Cook's distance is NaN for models without fixed effects, and nonlinear Cook's distances are no longer floored or clipped at extreme leverage.
- `GlmerResult.weights()` and `offset()` accept `copy=`, and `GlmerResult.getME("fixef_names")` works.
- `pvalues()` rejects unknown `method` names for generalized and nonlinear models instead of silently returning z-tests.
- `NlmerResult.isSingular()` checks the eigenvalues of the random-effect covariance, so a correlated fit with a near-zero off-diagonal factor entry is no longer reported as singular.
- `dotplot()` draws every term of a grouping factor that appears in several random-effect terms, and an unknown `term` raises before a figure is created.
- `mkDataTemplate()` and `mkMinimalData()` no longer raise `NameError`. `quickSimulate()` works with data that has no response column, and family names other than `"gaussian"` no longer silently simulate Gaussian responses.
- Models built with `mkLmerMod()` and `mkGlmerMod()` record the optimizer and its message, so `checkConv()` no longer reports an unknown optimizer.
- Native nonlinear likelihood evaluation raises `ValueError` for inconsistent input shapes instead of panicking; a short grouping array previously dropped observations silently.
- Documentation and README examples match the current API, including a rewritten nonlinear mixed-model tutorial and corrected convergence, bootstrap, diagnostics, deviance-component, and optimizer examples. Repository links point to `Cameron-Lyons/mixedlm`.
- R² and ICC average Gaussian residual variance over fitted precision weights (unit weights still give exactly `sigma**2`), include nonlinear offsets in fixed-prediction variance, keep small fixed-effect variation on a large common baseline, and reject non-finite fixed predictions instead of dropping them. Built-in log-link residual approximations evaluate in log space, preserving extreme fitted means.
- Cross-validation checks supplied predictor, grouping, and categorical values against fitted observations, preventing tied responses from concealing row misalignment.
- Weighted cross-validation MSE, RMSE, MAE, and R² retain contributions across extreme response and weight scales without overflowing intermediate products. Scoring rejects complex and masked inputs.
- Native sparse random-effect products validate dimensions and CSC buffers before accessing them, returning informative errors for malformed inputs.
- Conditional prediction matrices preserve positional identity when random-effect column names coincide and are reused for means and uncertainty. New-level prior variances use bounded sparse projection buffers.
- Nested random effects expand to every parent factor, and explicit `:` grouping is supported. Joint levels escape separators within labels. Formula parsing rejects malformed or unsupported trailing expressions instead of silently fitting a truncated model. Nested fits can change because earlier versions omitted parent factors.
- Group display names and covariance-selection keys quote unusual identifiers, distinguishing a literal column such as `a:b` from the joint `a:b` factor. Ordinary names retain their existing keys; use the names shown by `ngrps()` for quoted or joint factors.
- Fixed-coefficient deletion diagnostics include changes in random effects and decreasing-link derivatives. Collinearity retains varying predictors under large translations, extreme units, and weight scales.
- Python GLMM iterations keep predictors in the family and link domains, recover feasible starting values, and backtrack invalid or worsening steps. Quadrature assigns zero likelihood outside the valid support instead of clamping invalid predictors into it.
- GLMM log-likelihoods, information criteria, and reported marginal deviance include the response-distribution constants for every quadrature setting. The stored `deviance` retains the optimization criterion. Built-in families expose normalized conditional likelihoods; custom distributions can implement the same hook, and quasi likelihoods report unavailable information criteria.
- Grouped binomial updates, reduced-model tests, and optimizer comparisons apply trial counts once while preserving original prior weights and offsets. Model selection rejects different binomial trial counts even when effective fitting weights coincide.
- Response refits synchronize the stored model frame, so subsequent updates, cross-validation, and reduced-model tests use the refitted response rather than the original observations.
- Gaussian, Gamma, and inverse-Gaussian GLMM simulations honor prior precision weights for single and batched draws. Power design extension preserves Polars factor order, and contrast coding and validation work with the supported minimum NumPy and pandas versions.
- Built-in datasets now contain the original lme4 observations from a pinned upstream revision, including full InstEval and VerbAgg tables, with offline bundled data and recorded provenance. Earlier altered, truncated, and synthetic observations are corrected; fitted results can change. Canonical column names are restored, with documented aliases for `total_fruits` and `cTICKS`.
- Cross-validation preserves held-out offsets, contrast coding, and grouped-binomial trials without multiplying trial weights twice. Weighted R² retains response and weight scale invariance, including large baselines, and custom metrics cannot overwrite fold metadata.
- Blocked Cholesky retains fill-in between structures coupled through earlier blocks, fixing affected likelihoods, estimates, and gradients. Nonfinite diagonal pivots are rejected.
- Power simulations exclude unconverged or invalid refits and require Boolean test decisions. Sample-size curves change their simulation designs while retaining pilot parameters and row metadata, and named coefficient curves test the varied coefficient by default. Extreme numeric group labels can be extended without rounding collisions or overflow.
- Bootstrap refits require convergence and finite real estimates of the expected shapes in both serial and parallel execution. Failed samples stay entirely missing, and all confidence interval methods require at least two valid samples per parameter.
- Nonlinear fits no longer report convergence when the inner PNLS iteration limit is reached. Summaries identify unfinished inner solves, and nonlinear bootstrap intervals exclude unconverged refits.
- LMM and GLMM fitting, modular optimization, and profiles detect zero variance scales with a zero gradient but a better nearby likelihood. Restarts retain the selected optimizer, share its remaining budget, and report nonconvergence if an improvement cannot be resolved.
- LMM fixed-effect profile intervals now re-optimize nuisance covariance and residual scale using ML, including for REML inputs. Failed fits and unbracketed intervals raise instead of substituting Wald limits; interval extraction skips unnecessary plotting points.
- GLMM profile confidence intervals now use constrained likelihood fits and likelihood-ratio cutoffs, with nuisance covariance and fixed coefficients re-optimized, instead of returning Wald intervals.
- `GlmerControl.tolPwrss` now governs PIRLS stopping in native and Python likelihoods, quadrature, modular fitting, and final extraction; invalid inner controls are rejected before solving.
- Prediction offsets reject complex, masked, and non-finite values before evaluating a model. Scalar offsets use constant-size storage even for large prediction grids.
- LMM and GLMM predictions collect Polars lazy queries once, projecting to prediction columns and preserving row order and intercept-only grid sizes. Conditional lazy predictions no longer fail when checking the number of rows.
- New-data LMM and GLMM predictions preserve distinct fixed-effect columns with colliding display names, including after rank reduction and refitting.
- Fixed-effect tables and summaries preserve repeated coefficient names and keep inference aligned by position; ambiguous named selections and dictionary results raise clear errors.
- Nonlinear bootstrap confidence intervals exclude failed refits instead of substituting original estimates, validate requests before refitting, and share sample handling with `bootstrap_nlmer()`.
- Nonlinear simulation and bootstrap preserve NumPy's global random state while retaining integer-seeded draw sequences.
- Python nonlinear fitting honors arbitrary integer group labels and matches native sorted-label ordering.
- Python nonlinear least squares checks random-effect changes against the previous iteration, preventing premature convergence when fixed parameters are already stationary.
- Nonlinear fits and refits now reject invalid final evaluations instead of reporting convergence with a failure penalty and starting estimates.
- Nonlinear fitting and refitting preserve custom prediction and gradient methods instead of selecting a built-in native formula by display name.
- Linear and generalized model simulation and parametric bootstrap preserve NumPy's global random state and accept reusable random streams without changing integer-seeded draws.
- Linear and generalized fits, refits, and modular result constructors reject invalid final estimates instead of returning nonfinite values or fabricated coefficients; reported deviance comes from the validated final evaluation.
- GLMM quadrature validates positive integer orders and rejects unsupported random-effect structures; modular results cannot relabel a fit with a different quadrature order.
- Adaptive quadrature includes observations with zero random-effect design rows and rejects overlapping group designs that cannot be integrated independently.
- Python quadrature uses stable high-order rules shared by `GHrule`, `GQN`, and `GQdk`; invalid orders and tensor dimensions raise clear errors.
- Native GLMM starting coefficients use weighted response means on the link scale with offsets removed, preventing large Poisson counts from exhausting PIRLS iterations or overflowing; nonfinite PIRLS estimates no longer report convergence.
- GLMM fitting, refitting, and modular assembly include inner PIRLS convergence in the reported status; `pirls_converged` identifies inner failures and summaries and warnings explain them.
- COBYLA fits return results when SciPy supplies an evaluation count without an iteration count. The normalized iteration count uses evaluations, matching COBYLA's `maxiter` budget units. Boundary probes and restarts share that evaluation budget, retaining the best probe when too few evaluations remain to initialize a restart.
- `optimizeLmer()` applies the stored control's convergence tolerances and `optCtrl` options to the requested solver, including evaluation limits and normalized option aliases. Explicit `optCtrl` entries override generated options without changing the stored control.
- `optimizeLmer()` honors the `restart_edge` control supplied to `mkLmerDevfun()` when no override is given. Explicit booleans override the control for one fit; `None` inherits it, and manually constructed deviance callables without a control retain enabled checks.
- Variance-boundary restarts check small positive covariance scales as well as zero. This recovers better likelihoods when numerical derivatives stop just outside the old boundary threshold, while retaining genuine small estimates and the remaining optimization budget.
- `trust-constr` reports the objective gradient for convergence summaries and accepts the same one-argument iteration callbacks as other SciPy optimizers.
- Native LMM likelihoods, final estimates, and covariance gradients preserve weighted crossproducts between levels of the same random-effect structure. This corrects advanced designs with overlapping level columns; ordinary grouped designs retain their specialized block solves.
- GLMM optimizers, joint likelihood objectives, and modular deviance callables can be deep-copied and pickled with native preparation enabled. Restored objects rebuild the cache using the available backend and preserve model inputs and solver controls.
- Native GLMM entry points reject inconsistent array dimensions, covariance parameter counts, and random-effect metadata with `ValueError`, preventing Rust panics and silently ignored input. Dimension arithmetic and sparse column-pointer lengths are checked for overflow.

### Internal

- Security scans are included in the single required CI gate, audit optional Python runtime dependencies and both Rust lockfiles, and use the complete hashed dependency graph.
- Every standard and free-threaded wheel target runs independent statistical, nonlinear, and native-threading regressions from its installed artifact before upload or publication.
- Standard Python 3.14 joins the test matrix. Property tests compare generated weighted and crossed designs against independent Gaussian likelihoods. Required CI runs locked, sanitized fuzz targets against the production sparse routines, with mathematical solve and input-validation oracles.
- CI uses locked test dependencies, checks optional plotting and optimizer features, enforces an 87% combined line/branch coverage floor in the complete feature job, validates workflows, and exercises installed wheels and rebuilt source distributions before publication. A single required-check gate aggregates all CI jobs.
- Reproducible independent/crossed likelihood benchmarks run in CI.
- Sparse-cache correctness tests no longer depend on single-run timing comparisons; paired cached and uncached measurements now use the benchmark suite.
- `LmerResult` and `GlmerResult` share one implementation of their common accessors, and Cook's distance has one formula shared with nonlinear results.
- The native extension is compiled with `#![forbid(unsafe_code)]`.
- Documentation code examples run in the test suite, and their mixedlm references are checked.
- Wheel and source-distribution builds are shared between CI and publishing through a reusable workflow, and installed-wheel tests are selected with the `installed_wheel` marker.
- Miri runs weekly and on demand instead of on every pull request. Pull requests check benchmark results without timing them, and timings are recorded on `main`.
- Fuzz smoke runs reuse the weekly fuzz workflow, and the fuzz crate's tests and lints run in the stable Rust jobs, replacing the separate Rust CI workflow.
- Security scanners are locked in `uv.lock`, Dependabot updates `uv.lock`, pre-commit hooks use the locked tools, and Codecov uploads use a repository token.
- Native GLMM and NLMM tests and benchmarks exercise the production entry points, and new benchmarks cover nested and crossed LMM and GLMM designs and the native nonlinear likelihood.

## [1.2.0] - 2026-08-18

### Added

- Arbitrary linear fixed-effect hypothesis tests with named or matrix constraints.
- Grouped-binomial `successes / trials` responses with automatic trial weights and validation.
- Configurable named links and documented family/link helper APIs.
- Formula-driven simulation now supports every built-in response family.
- New-data LMM and GLMM predictions accept numeric, scalar, or column-based offsets.
- Backtick quoting for column names with spaces or formula operators.
- EM-REML now supports multiple random effects and random slopes (correlated and uncorrelated).
- Automatic convergence recommendations in `summary()` output for non-converged and singular fits.
- `em_init` control parameter is now wired up in `lmer()` and `glmer()` model fitting.
- `em_init` and `em_maxiter` parameters added to `GlmerControl` and `glmerControl()`.
- EM-REML initialization section in estimation background documentation.
- `tidy()` and `glance()` analysis-ready reports for linear, generalized, and nonlinear fits.
- Vectorized AIC/AICc/BIC model rankings with normalized weights and evidence sets.
- Nakagawa marginal/conditional R² and adjusted/unadjusted ICC for all model families.
- Weighted VIF/GVIF, tolerance, severity, and condition diagnostics for all model types.
- Vectorized Pearson dispersion and observed-versus-expected zero diagnostics for GLMMs.
- Dependency-free case-level and grouped cross-validation for LMMs and GLMMs.

### Changed

- Multi-draw GLMM simulation now batches random effects and response generation.
- EM-REML algorithm generalized from single random intercept to arbitrary unstructured covariance models.
- Replaced the Py-BOBYQA dependency and default optimizer with SciPy COBYQA.
- Reduced unused and redundant Python and Rust dependencies.
- Top-level public exports now load on demand to reduce startup time and memory use.
- Vectorized pandas nested-group construction for faster large model setup.
- Vectorized grouping-level factorization and nested-key construction for sparse designs.

### Fixed

- Poisson and other unbounded GLMM families no longer clamp fitted means below one.
- Unsupported family and link combinations no longer route through the native fast path.
- LMM profiling now applies fitted random effects before computing the penalized residual sum of squares, preventing variance estimates from being biased toward zero.
- LMM likelihood reporting and fixed-effect profiles now use the corrected profiled criterion consistently.
- Exported influence diagnostics now use the fitted mixed-model projection, prior weights, random effects, and offsets.
- GLMM covariance, leverage, conditional variance, Pearson residuals, and influence diagnostics now honor prior weights.
- One- and two-parameter LMM profiles now honor prior weights and offsets, including weight-scale invariant REML normalization.
- Synthetic dataset loaders no longer reset NumPy's global random state.
- Nonlinear mixed models now apply offsets consistently to responses, fitted values, simulations, refits, covariance estimates, and leverage diagnostics.
- Canonical Gamma and inverse-Gaussian variants now simulate from their family distributions.
- LMM and GLMM simulations now preserve model offsets in generated responses.
- Formula simulation now preserves global random state and random-effect coefficient ordering.
- Nonlinear mixed-model optimization now uses a deterministic profiled Laplace deviance with consistent relative covariance scaling.
- GLMM PIRLS, Laplace, and adaptive-quadrature calculations now use the covariance factor in spherical random-effect coordinates, with consistent likelihood normalization and deterministic outer optimization.
- LMM prediction uncertainty now includes conditional random-effect covariance, fixed/random cross-covariance, correlated slopes, and unseen-group prior variance.
- Covariance tables, PCA diagnostics, singularity checks, and parameter bounds now honor compound-symmetry and AR(1) random-effect structures.
- Gamma GLMMs now minimize the non-negative unit deviance instead of its negative.

### Internal

- Root-level CHANGELOG.md.
- Consolidated duplicate CI and security checks while preserving coverage.

## [1.1.0] - 2026-01-27

### Added

- Symbolic factorization caching for sparse Cholesky operations.
- EM-REML initialization option for variance component estimation.

### Changed

- Updated faer dependency from 0.23.2 to 0.24.0.
- Performance optimizations and code cleanup.
- BOBYQA is now the default optimizer.
- Enhanced convergence diagnostics and adaptive starting values.

### Fixed

- pandas 2.x StringDtype compatibility issues.
- Polars compatibility fixes for dataset loaders.

### Internal

- Comprehensive CI pipeline with testing across Python 3.10-3.13, plus free-threaded 3.14t.
- Miri, property-based testing, and benchmark jobs in CI.
- Rust and Python code coverage reporting.

## [1.0.0] - 2026-01-14

### Added

- Comprehensive documentation site with MkDocs and Material theme.
- Tutorials for LMM, GLMM, NLMM, inference, and power analysis.
- Background sections on estimation methods and degrees of freedom.
- Full API reference documentation.
- ReadTheDocs integration.
- Gradient support for linear mixed models.
- PyArrayLike support for flexible array input types.

### Changed

- Marked as Production/Stable (v1 stability milestone).

## [0.1.1] - 2026-01-13

### Added

- Type III ANOVA via `anova_type3()`.
- 2D profile likelihood slices via `slice2D()`.
- Satterthwaite and Kenward-Roger degrees of freedom methods.
- Power analysis functions: `powerSim()`, `powerCurve()`, `extend()`.
- Influence diagnostics: `cooks_distance()`, `dfbeta()`, `dfbetas()`, `dffits()`, `leverage()`.
- Diagnostic plots: `plot_diagnostics()`, `plot_qq()`, `plot_ranef()`.
- Estimated marginal means via `emmeans()`.
- `drop1()` for single term deletions.
- `allFit()` to try multiple optimizers.
- Nonlinear mixed models via `nlmer()` with self-starting models.
- Negative binomial GLMM via `glmer_nb()`.
- Polars DataFrame support via narwhals.
- Variance transformation utilities.
- `checkConv()` and `convergence_ok()` for convergence diagnostics.

### Changed

- Improved numerical stability in PIRLS algorithm for GLMMs.
- Performance optimizations for profile likelihood.
- Enhanced Laplace deviance computation.

### Fixed

- Numerical stability in boundary cases for GLMM.

### Internal

- Fixed mypy error in profile cache.

## [0.1.0] - 2026-01-12

### Added

- Initial release.
- Linear mixed models via `lmer()`.
- Generalized linear mixed models via `glmer()`.
- REML and ML estimation.
- Laplace approximation and adaptive Gauss-Hermite quadrature for GLMMs.
- lme4-style formula syntax with random effects.
- Distribution families: Gaussian, Binomial, Poisson, Gamma, InverseGaussian, NegativeBinomial.
- Basic inference: `anova()`, `confint()`, `bootMer()`.
- Profile likelihood confidence intervals.
- Built-in datasets from lme4.
- Rust backend for performance-critical operations.
- pandas DataFrame support.

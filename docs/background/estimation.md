# Estimation Methods

This page explains the statistical methods used to estimate mixed model parameters.

## Linear Mixed Models

### The Model

A linear mixed model has the form:

\[
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\mathbf{b} + \boldsymbol{\epsilon}
\]

where:

- \(\mathbf{y}\) is the \(n \times 1\) response vector
- \(\mathbf{X}\) is the \(n \times p\) fixed effects design matrix
- \(\boldsymbol{\beta}\) is the \(p \times 1\) fixed effects vector
- \(\mathbf{Z}\) is the \(n \times q\) random effects design matrix
- \(\mathbf{b} \sim N(\mathbf{0}, \boldsymbol{\Sigma})\) is the \(q \times 1\) random effects vector
- \(\boldsymbol{\epsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})\) is the residual error

### Maximum Likelihood (ML)

ML estimation maximizes the marginal likelihood:

\[
L(\boldsymbol{\beta}, \boldsymbol{\theta}, \sigma^2 | \mathbf{y}) = \int p(\mathbf{y} | \boldsymbol{\beta}, \mathbf{b}, \sigma^2) p(\mathbf{b} | \boldsymbol{\theta}) d\mathbf{b}
\]

For linear mixed models, this integral has a closed form. The log-likelihood is:

\[
\ell = -\frac{1}{2}\left[ n \log(2\pi\sigma^2) + \log|\mathbf{V}| + \frac{(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T\mathbf{V}^{-1}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})}{\sigma^2} \right]
\]

where \(\mathbf{V} = \mathbf{Z}\boldsymbol{\Sigma}\mathbf{Z}^T + \sigma^2\mathbf{I}\).

**Pros:**

- Consistent estimates
- Allows comparison of models with different fixed effects

**Cons:**

- Variance components are biased downward, especially in small samples

### Restricted Maximum Likelihood (REML)

REML addresses the bias in variance estimation by maximizing a modified likelihood that doesn't depend on fixed effects:

\[
L_R(\boldsymbol{\theta}, \sigma^2 | \mathbf{y}) = \int L(\boldsymbol{\beta}, \boldsymbol{\theta}, \sigma^2 | \mathbf{y}) d\boldsymbol{\beta}
\]

This is equivalent to ML estimation on residuals after removing the fixed effects.

**Pros:**

- Unbiased variance estimates (analogous to dividing by \(n-p\) instead of \(n\))
- Default choice for most applications

**Cons:**

- Cannot compare models with different fixed effects using likelihood ratio tests

### When to Use Each

| Situation | Method |
|-----------|--------|
| Final variance estimates | REML |
| Comparing random effects structures | ML or REML |
| Comparing fixed effects | ML |
| AIC/BIC for fixed effects selection | ML |

```python
import mixedlm as mlm

# REML (default)
model_reml = mlm.lmer("y ~ x + (1 | g)", data, REML=True)

# ML
model_ml = mlm.lmer("y ~ x + (1 | g)", data, REML=False)
```

### Profiled Deviance

mixedlm uses a profiled deviance approach for optimization efficiency. Given variance parameters \(\boldsymbol{\theta}\), the optimal \(\boldsymbol{\beta}\) and \(\sigma^2\) have closed-form solutions:

\[
\hat{\boldsymbol{\beta}}(\boldsymbol{\theta}) = (\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}\mathbf{y}
\]

The optimization is then over just the variance parameters \(\boldsymbol{\theta}\), reducing dimensionality.

For a fixed design and prior weights, the native evaluator prepares
\(\mathbf{X}^T\mathbf{W}\mathbf{X}\),
\(\mathbf{Z}^T\mathbf{W}\mathbf{X}\), and
\(\mathbf{Z}^T\mathbf{W}\mathbf{Z}\) once per fit. Covariance evaluations
reuse these products. Linear bootstrap refits also share them across responses,
preparing a separate workspace in each parallel worker. Response-dependent
products are recomputed for each replicate, and each refit starts from the
original fitted covariance parameters.

The native solver checks whether the weighted design products separate by
grouping level and caches that pattern with the design. Standard grouped
designs use diagonal or small block solves. If an advanced design includes
overlapping level columns, the solver retains the full covariance within each
affected structure; covariance gradients retain those cross-level terms too.

Native mixed-model fits also use these products to extract final fixed and
random effects, scale, likelihood components, and fixed-effect information.
They avoid building a second Python crossproduct cache during final extraction
or response refits. The final scale is computed from weighted conditional
residuals plus the squared spherical random effects, avoiding cancellation
between marginal quadratic forms. Fixed-only fits retain the existing Python
solve and least-squares fallback.

At the estimation API level, `LMMOptimizer.with_response(y)` creates an
independent optimizer sharing the prepared design on either backend. It copies
the new response and retains the optimizer's ML/REML setting. Design matrices,
weights, and offsets must remain unchanged while the optimizers are in use;
construct a new optimizer when those inputs change. Large Python random-effect
systems retain sparse crossproducts.

Native design preparation releases Python's interpreter lock after copying its
inputs, allowing independent fits to prepare weighted crossproducts concurrently.
Prepared native ML and REML evaluations release the interpreter lock after
copying the covariance parameters. Each solve reads an immutable design and
response and uses its own scratch storage, so Python threads can evaluate a
shared response or separate responses concurrently. This applies to the native
backend, including final mixed-model estimate extraction; automatic backend
selection remains unchanged. Complete-fit throughput
also depends on the optimizer: SciPy's default COBYQA implementation serializes
optimizer calls with its own lock.

To fit with prepared analytic covariance gradients, enable them in the control:

```python
from mixedlm import lmer, lmerControl

fit = lmer(
    "y ~ x + (x | group)",
    data,
    control=lmerControl(optimizer="L-BFGS-B", use_analytic_gradient=True),
)
```

This option supports native LMM fitting with L-BFGS-B, BFGS, TNC, SLSQP, and
trust-constr. Value and gradient requests at the same parameters share an
evaluation within each fit. It is disabled by default because the benefit
depends on the model and optimizer. Python backends and structured covariance
types retain the solver's numerical derivatives; derivative-free solvers
continue to use the scalar objective.

When weighted design crossproducts separate across all grouping structures
and their levels, gradients use compact per-level inverses and transformed
crossproducts. Their storage grows with the sum of the squared level widths,
which reduces gradient costs for models with many independent levels. The
prepared design still stores the full random-effect crossproduct. Coupled
designs require a full inverse and can make analytic gradients more expensive
than numerical derivatives. Eligibility depends on exact zeros in the design
crossproducts, so zero variance parameters do not hide coupled levels.

REML gradients also share the fixed-effect information solve and projected
crossproduct across covariance parameters. Each derivative contracts only the
selected covariance-factor entries, avoiding a separate random-by-fixed matrix
product for every parameter. Independent levels retain compact block products.

`LMMOptimizer.optimize(use_analytic_gradient=True)` enables the same path for
prepared fits and response refits. `optimizeLmer()` inherits this setting from
the control passed to `mkLmerDevfun()` and accepts an explicit override.
Custom modular deviance callables retain numerical derivatives for their full objective.
Analytic derivatives can change the optimization path; convergence checks and
variance-boundary restarts still apply.

With `restart_edge=True`, fitting checks zero and near-zero covariance scales
for likelihood improvement before accepting convergence. The check includes
scales within `1e-4 * max(1, abs(start))` of zero, where numerical derivatives
can appear stationary. It retains the fitted value unless an inward probe
improves the likelihood, then restarts the requested optimizer within the
remaining budget. Genuine small positive estimates are not rounded to zero.
Set `restart_edge=False` to disable these checks.

Callers using `mixedlm._rust.LmmDesign` can reuse the same preparation for
analytic covariance gradients. After `response = design.with_response(y)`,
`response.deviance_with_gradient(theta, reml=True)` returns the profiled
deviance and a NumPy gradient in the same parameter order as the scalar
likelihood. It reuses design and response products across calls, checks the
parameter count and finite values, and supports both ML and REML. Responses
share the immutable design and own their response data; each returned gradient
owns its array. Gradient solves release the interpreter lock after copying
`theta`, allowing concurrent calls on shared or independent responses.

The value-and-gradient pair can be passed directly to SciPy with `jac=True`.
Given starting covariance parameters `theta0` and their corresponding `bounds`:

```python
from scipy.optimize import minimize

result = minimize(
    response.deviance_with_gradient,
    theta0,
    method="L-BFGS-B",
    jac=True,
    bounds=bounds,
)
```

Use `lambda theta: response.deviance_with_gradient(theta, reml=False)` for ML.
Fixed-effects-only responses return an empty gradient. Invalid parameters or
nonpositive REML residual degrees of freedom raise `ValueError`; numerical
factorization failures return the existing `1e10` penalty and a zero gradient.

## Generalized Linear Mixed Models

### The Model

GLMMs extend LMMs to non-Gaussian responses:

\[
g(E[y_{ij} | \mathbf{b}_j]) = \mathbf{x}_{ij}^T\boldsymbol{\beta} + \mathbf{z}_{ij}^T\mathbf{b}_j
\]

where \(g(\cdot)\) is the link function and \(y_{ij}\) follows an exponential family distribution.

### The Integration Problem

Unlike LMMs, the marginal likelihood for GLMMs doesn't have a closed form:

\[
L(\boldsymbol{\beta}, \boldsymbol{\theta} | \mathbf{y}) = \int \prod_i p(y_i | \boldsymbol{\beta}, \mathbf{b}) p(\mathbf{b} | \boldsymbol{\theta}) d\mathbf{b}
\]

This integral is typically high-dimensional and must be approximated.

### Laplace Approximation

The Laplace approximation replaces the integrand with a Gaussian approximation around its mode:

\[
\int e^{f(\mathbf{b})} d\mathbf{b} \approx (2\pi)^{q/2} |\mathbf{H}|^{-1/2} e^{f(\hat{\mathbf{b}})}
\]

where \(\hat{\mathbf{b}}\) is the mode and \(\mathbf{H}\) is the Hessian at the mode.

**In practice:**

1. Find the mode \(\hat{\mathbf{b}}\) that maximizes \(\log p(\mathbf{y}|\mathbf{b}) + \log p(\mathbf{b})\)
2. Compute the Hessian at the mode
3. Approximate the integral using the Gaussian formula

At `nAGQ>=1`, outer optimization varies both fixed coefficients and covariance
parameters. Each likelihood evaluation holds these parameters fixed and solves
only for the conditional random-effect mode. This includes the effect of the
curvature correction on the optimal fixed coefficients.

**Accuracy:**

- Works well for large cluster sizes (many observations per random effect)
- Can be biased for small clusters or binary data
- Faster than quadrature methods

```python
model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial(),
    nAGQ=1  # Laplace approximation
)
```

### Adaptive Gauss-Hermite Quadrature

For more accuracy, use numerical integration with Gauss-Hermite quadrature:

\[
\int e^{f(b)} db \approx \sum_{k=1}^{K} w_k e^{f(a_k)}
\]

**Adaptive** quadrature centers the quadrature points at the mode and scales by the curvature, improving accuracy.

**Trade-offs:**

| nAGQ | Accuracy | Speed | Use case |
|------|----------|-------|----------|
| 0 | Joint-PIRLS approximation | Fastest | Preliminary fits, previous fitting behavior |
| 1 | Moderate | Fast | Default, large clusters |
| 5-10 | High | Medium | Small clusters, binary data |
| 25+ | Very high | Slow | Research, validation |

```python
# More accurate for small clusters
model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial(),
    nAGQ=10
)
```

**Limitation:** AGQ is available for one grouping factor with one random-effect
coefficient per group: a random intercept or a scalar random slope.

#### Quadrature computation

For adaptive quadrature with a single scalar random effect per group, each
integration point evaluates only the observations affected by that group's
coefficient. Curvature is computed from the corresponding weighted design column.
This avoids rebuilding full-model predictors and dense curvature matrices at each
quadrature evaluation. Python parallel evaluation shares the fitted random-effect
modes without modifying them.

An observation with a zero random-effect design row still contributes its
fixed-effect response likelihood. This includes zero-valued scalar random-slope
predictors and observations in groups whose slope predictors are all zero.
Explicit and implicit sparse zeros produce the same likelihood. A supplied design
with multiple nonzero random-effect coefficients in one row cannot use independent
group quadrature and raises `ValueError`.

#### Quadrature rules

`GHrule(n)` and `GQN(n)` return rules normalized for the standard normal
distribution. `GQdk(d, k)` builds a tensor rule for `d` independent standard
normal variables with `k` points per dimension. Orders and dimensions must be
positive integers. Public rule arrays are writable and independent of later calls.

Python helpers and fitting share stable Hermite rule generation, including high
orders where polynomial-based construction can overflow. Python and native fitting
cache up to 32 rules with at most 1,024 points each. Larger rules remain supported
and bypass the cache. Cache entries are shared read-only across evaluations;
changing model data or parameters does not require clearing them.

### PIRLS Algorithm

For fitting GLMMs, mixedlm uses Penalized Iteratively Reweighted Least Squares (PIRLS):

1. Initialize fixed effects from a weighted regression of starting response means on the link scale, subtracting offsets; initialize random effects at zero
2. Given current \(\mathbf{b}\), compute working responses and weights
3. Solve a penalized weighted least squares problem
4. Update \(\mathbf{b}\)
5. Repeat until convergence

For `nAGQ=0`, PIRLS updates fixed and random effects together inside an outer
optimization over covariance parameters. This reproduces the previous fitting
algorithm. It need not maximize the integrated Laplace likelihood over fixed
coefficients.

For `nAGQ>=1`, this preliminary fit initializes joint optimization over covariance
and fixed-effect parameters. During the joint stage, `X @ beta` becomes a fixed
offset and PIRLS updates only random effects. Set
`GlmerControl(nAGQ0initStep=False)` to skip the preliminary covariance optimization;
a single PIRLS solve still supplies starting coefficients. The outer `maxiter`
limit applies separately to each stage, and `n_iter` totals both. Exact GLM and
Gaussian identity-link Laplace fits, and Laplace fits with no fixed effects,
avoid redundant optimization.

`GlmerControl(tolPwrss=1e-8, pirls_maxiter=100)` controls the inner solve.
`tolPwrss` bounds the maximum absolute coefficient update in both the fixed and
spherical random effects. `pirls_maxiter` limits inner iterations independently
of the outer optimizer's `maxiter`. Its default `None` retains the native limit
of 100 and the Python limit of 25. Fitting now honors the configured `tolPwrss`
default of `1e-7`; previously both backends used `1e-6` regardless of this control.

Direct likelihood functions accept keyword arguments `pirls_tol` and
`pirls_maxiter`. Their defaults preserve the previous `1e-6` tolerance and backend
iteration limits. `pirls()` uses its existing `tol` and `maxiter` arguments;
native entry points also accept these keywords. Refits inherit the fitted inner
settings and allow `result.refit(pirls_maxiter=200, pirls_tol=1e-9)` to override them.
Model-derived bootstrap and comparison fits, model updates, and cross-validation
also retain these settings, including their parallel worker paths.

The native `pirls`, `laplace_deviance`, `glmm_deviance`, and
`adaptive_gh_deviance` bindings require a nonempty response and matching design,
weight, and offset row counts. They also check covariance parameter counts and
random-effect structure dimensions. Mismatched row counts, vector lengths, or
metadata, and dimension overflows raise `ValueError` before fitting.

Native fitting prepares an owned response, design, prior weights, offsets, and
starting coefficients once per objective. `GLMMOptimizer` and
`JointGLMMObjective` reuse this preparation across parameter evaluations; joint
fits supply a new combined fixed-effect offset for each solve. Each evaluation
starts independently, so parameter order and earlier failed evaluations do not
change its result. Treat the model arrays and family as immutable while using
these estimation objects; construct a new object when the inputs change. Public
fits and refits prepare their current inputs automatically. Custom families,
links, and covariance structures retain the Python implementation.

Prepared native GLMM likelihood evaluations release Python's interpreter lock
during the solve. Covariance parameters and offset overrides are copied before
release, and each evaluation has its own scratch storage, allowing Python threads
to evaluate one immutable prepared problem concurrently. Complete-fit throughput
also depends on the optimizer: the tested SciPy 1.17.0 COBYQA wrapper serializes
optimizer calls with its own lock. Releasing the interpreter lock does not remove
that synchronization.

Native PIRLS reuses its linear-predictor, working-weight, and working-response
buffers across iterations and computes link derivatives and variances per
observation without retaining separate vectors. Predictor updates overwrite the
previous iteration's values before adding current random effects. Joint
likelihoods pass fixed coefficients through the offset, so their mode solves
start directly from that offset, skip the empty fixed-effect system, and use the
two triangular random-effect solves directly. The final mode calculation reuses
the predictor buffer. Working-weight floors, convergence checks, and the final
likelihood correction use the same formulas.

Binomial/logit iterations select a specialized working-value loop once per
iteration. Separate contiguous input and output slices let the compiler
vectorize this loop while retaining the existing derivative, variance, and
weight-floor arithmetic. Other family/link combinations use the general loop.

Modular `GlmerDevfun` calls with full `[theta, beta]` vectors prepare the joint
objective on first use and reuse it for later parameter values. Changing its
optimizer, quadrature order, or inner solver controls refreshes this preparation.
Covariance-only calls do not allocate a joint objective. The cached objective
keeps its native mode-solve inputs alive for the deviance callable's lifetime.

These objectives and modular `GlmerDevfun` callables support copying and Python
pickling when their model inputs and custom family are serializable. Deep copies and
unpickled objects rebuild native preparation from the retained Python inputs,
including when passed to a spawned worker process. Restoration uses the backend
available in the receiving process; native buffers are not included in the pickle.

A fitted GLMM reports `converged=True` only when both the outer optimizer and the
inner PIRLS solver converge. `result.pirls_converged` exposes the inner status,
including on refitted and modular results. For example, all-zero Poisson responses
or constant binary responses can leave the inner coefficients drifting even when
the outer objective stops changing. Such fits retain their finite estimates but
report nonconvergence, with an inner-solver warning and a summary note.

`check_conv=False` suppresses the fitting warning without changing these flags.
For direct objective evaluation,
`mixedlm.estimation.laplace.glmm_deviance_with_status(...)` returns
`(deviance, beta, u, pirls_converged)` from one solve. Existing deviance functions
continue to return their three-item tuples.

The native solver computes weighted random-effect crossproducts by observation,
using only columns that occur together in a row of the sparse design matrix.
It builds this row layout once per likelihood evaluation and reuses it as the
working weights change. Dense designs accumulate one column pair at a time,
using direct dot products for fully populated matrices. Linear and generalized
linear models share this implementation for dense systems. Larger sparse GLMM
systems use the sparse precision pattern described below.

When the covariance factor is diagonal and its active design columns do not
share observations, the penalized random-effect precision is diagonal. The
native solver then accumulates one precision value per coefficient and solves
by scalar division. The same diagonal supplies the Laplace log determinant.
The prepared design is reused as working weights change. This covers random
intercepts, scalar random slopes, and disjoint independent coefficients at any
model size, including models below the sparse-factorization cutoff. Empty levels
retain their unit prior precision. Coupled effects use the existing dense or
sparse factorization.

The native solver stores one small covariance factor per random-effect
structure and applies it across the grouping levels. For sufficiently sparse
models with at least 128 random-effect coefficients, it forms the scaled design
and penalized random-effect system in sparse storage. A fill-reducing ordering
keeps nested and crossed group structures sparse when possible. The row layout,
precision pattern, and symbolic factorization are reused as the PIRLS working
weights change, including the final Laplace determinant.

All contributions between levels and grouping factors are retained, including
correlated slopes and zero variance components. Small systems, dense designs,
and patterns with excessive factor fill use dense kernels. Dense covariance
transforms operate in place without constructing a full block-diagonal factor.
For independent random intercepts, the sparse precision and factor each store
one entry per group. Storage for the fixed-effect design and its crossproducts
still depends on the numbers of observations and fixed-effect coefficients.

The random effects are solved in spherical coordinates,

\[
\mathbf{b} = \boldsymbol{\Lambda}_{\theta}\mathbf{u},
\qquad \mathbf{u} \sim N(\mathbf{0}, \mathbf{I}),
\]

so the penalized random-effect system is

\[
\mathbf{C} = \mathbf{I} +
\boldsymbol{\Lambda}_{\theta}^{T}\mathbf{Z}^{T}\mathbf{W}\mathbf{Z}
\boldsymbol{\Lambda}_{\theta}.
\]

This parameterization makes the covariance scale explicit, keeps zero-variance boundaries
well-defined, and uses the same system for the PIRLS mode, Laplace determinant, post-fit
covariance, and leverage calculations. Adaptive quadrature uses the normalized standard-normal
prior in these coordinates and evaluates each grouping level's likelihood contribution once.
With one native worker, group integration runs on the calling thread. With multiple
workers, groups are evaluated in parallel and their scalar contributions are added
in group order with compensated summation. Compensation preserves small contributions
beside much larger group log likelihoods. This keeps the reduction independent of scheduling and
worker count, using one additional scalar per group for parallel collection.
Compared with earlier versions, fixing the addition order can change the last few
bits of the deviance. The model likelihood and quadrature rule are unchanged.

Starting means lie inside the family and link domains. For Poisson models with a
log link, positive counts are transformed with the logarithm before estimating
starting coefficients; zero counts use a small positive starting mean. This keeps
large counts from producing an excessively large initial linear predictor. The
native solver uses the same starting-mean convention as Python, including prior
weights and offsets. Its inner convergence flag remains false if an update or
final deviance is nonfinite.

The native PIRLS solver handles the fixed-effect and working-response columns in
one triangular solve, borrowing the Cholesky factor. It reuses the transformed
columns to recover random effects with a transpose triangular solve. This avoids
copying the full factor and repeating a forward solve on each iteration. The
random-effect factorization uses the sparse or dense path selected for the model.

## Nonlinear Mixed Models

### The Model

NLMMs use a nonlinear function of parameters:

\[
y_{ij} = f(\mathbf{x}_{ij}, \boldsymbol{\phi}_j) + \epsilon_{ij}
\]

where \(\boldsymbol{\phi}_j = \boldsymbol{\beta} + \mathbf{b}_j\) are group-specific parameters.

### Estimation Approach

mixedlm uses a first-order linearization approach:

1. Linearize \(f\) around current parameter estimates
2. Solve the resulting approximate LMM
3. Update parameters
4. Iterate until convergence

This is similar to the Lindstrom-Bates algorithm.

The low-level `pnls_step()` and `nlmm_deviance()` functions, and
`NLMMOptimizer`, accept integer grouping labels with gaps or negative values.
Rows of the random-effect matrix `b` correspond to sorted unique labels in
both the Python and native implementations; observations within each group
keep their input order.

Each objective evaluation builds the group row indices once and reuses them
through linearization, random-effect updates, residual calculations, and the
Laplace correction. Python's serial and threaded paths share the same group
calculations, and threaded updates reuse the covariance inverse. This avoids
repeated full-data group scans and retaining one full-length mask per group.

The inner penalized nonlinear least-squares loop checks changes in both the
fixed parameters and every group-specific random effect. Each change is
measured against the preceding iteration before deciding whether to stop;
stationary fixed parameters alone do not establish convergence. The iteration
limit still bounds the work for problems that do not converge.

## EM-REML Initialization

### The Algorithm

The EM-REML (Expectation-Maximization Restricted Maximum Likelihood) algorithm provides a robust alternative for initializing variance components before switching to direct optimization. It alternates between:

1. **E-step:** Compute conditional expectations of random effects given current variance parameters
2. **M-step:** Update each random effect structure's covariance matrix and the residual variance

The E-step solves a joint linear system for fixed effects \(\hat{\boldsymbol{\beta}}\) and BLUPs \(\hat{\mathbf{b}}\), and computes the posterior variance \(\text{Var}(\mathbf{b} | \mathbf{y})\) via the Schur complement. The M-step updates each structure's covariance:

\[
\hat{\boldsymbol{\Sigma}}_k = \frac{1}{n_k}\sum_{i=1}^{n_k}\left(\hat{\mathbf{b}}_{ki}\hat{\mathbf{b}}_{ki}^T + \text{Var}(\mathbf{b}_{ki} | \mathbf{y})\right)
\]

### When to Use EM-REML

EM-REML is useful when direct optimization converges to boundary solutions (zero variance components) or fails to converge entirely. It tends to find interior solutions because it updates variances as averages of squared effects plus uncertainty, which naturally stay positive.

Enable EM-REML initialization via the `em_init` control parameter:

```python
import mixedlm as mlm

# Use EM-REML initialization for LMMs
ctrl = mlm.LmerControl(em_init=True, em_maxiter=50)
model = mlm.lmer("y ~ x + (x | group)", data, control=ctrl)

# Also available for GLMMs (provides starting theta from a linear approximation)
ctrl = mlm.GlmerControl(em_init=True)
model = mlm.glmer("y ~ x + (1 | group)", data, family=mlm.families.Binomial(), control=ctrl)
```

### Supported Models

EM-REML supports:

- Random intercepts: `(1 | group)`
- Correlated random slopes: `(x | group)`
- Uncorrelated random slopes: `(x || group)`
- Multiple random effects: `(1 | group1) + (1 | group2)`
- Compound symmetry via `set_cov_type(formula, "cs")`

AR(1) covariance is not supported; EM initialization falls back to the direct optimizer for these models.

### Trade-offs

| Property | EM-REML | Direct Optimization |
|----------|---------|---------------------|
| Robustness to starting values | High | Moderate |
| Convergence speed | Slow (linear) | Fast (superlinear) |
| Boundary avoidance | Good | May converge to zero |
| Best use | Initialization | Final estimation |

## Optimization

### Available Optimizers

mixedlm supports multiple optimization algorithms:

**Always available (SciPy):**

- `COBYQA` - Derivative-free constrained optimization (default)
- `L-BFGS-B` - Quasi-Newton with bounds
- `BFGS` - Quasi-Newton
- `Nelder-Mead` - Simplex method
- `Powell` - Direction set method
- `trust-constr` - Trust region with constraints
- `SLSQP` - Sequential least squares
- `TNC` - Truncated Newton
- `COBYLA` - Constrained optimization by linear approximation

**Optional (requires additional packages):**

- `newuoa` - Derivative-free unconstrained (nlopt)
- `praxis` - Principal axis (nlopt)
- `sbplx` - Subplex algorithm (nlopt)

### Choosing an Optimizer

```python
# Use a specific optimizer
model = mlm.lmer(
    "y ~ x + (1 | g)",
    data,
    control=mlm.LmerControl(optimizer="COBYQA")
)

# Try all available optimizers
results = model.allFit(data)
print(results.summary())
```

### Convergence Criteria

The optimizer stops when:

1. Gradient is near zero (for gradient-based methods)
2. Change in parameters is below tolerance
3. Change in objective is below tolerance
4. Maximum iterations reached

```python
control = mlm.LmerControl(
    maxfun=50000,    # Maximum function evaluations
    tol=1e-8         # Convergence tolerance
)
```

## Numerical Stability

### Parameterization

mixedlm uses a relative covariance factor parameterization (\(\boldsymbol{\theta}\)) rather than variances directly:

\[
\boldsymbol{\Sigma} = \sigma^2 \mathbf{\Lambda}\mathbf{\Lambda}^T
\]

where \(\boldsymbol{\theta}\) contains the elements of \(\mathbf{\Lambda}\). This:

- Ensures positive semidefinite covariance matrices
- Improves optimization stability
- Allows variance to approach zero smoothly

### Sparse Matrix Methods

For models with many groups, mixedlm uses sparse matrix operations to efficiently compute:

- Weighted products \(\mathbf{Z}^T\mathbf{W}\mathbf{Z}\)
- Factorizations of the random-effect precision system
- Linear system solutions

This enables fitting models with thousands of groups.

The Python ML/REML evaluator keeps large random-effect systems sparse and reuses
one factorization for the fixed-effect and random-effect solves. Small systems
use dense Cholesky. Both paths support observation weights, offsets, and
unstructured, independent, compound-symmetry, and AR(1) random effects.

Native sparse solves process multiple right-hand sides together, reusing the
factor for every column. Cached and uncached solves share this implementation
and solve directly in the returned array's storage, leaving the input unchanged
even for strided or read-only arrays.

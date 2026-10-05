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

data = mlm.load_sleepstudy()

# REML (default)
model_reml = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=True)

# ML
model_ml = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, REML=False)
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

At the estimation API level, `LMMOptimizer.with_response(y)` creates an
independent optimizer sharing the prepared design on either backend. It copies
the new response and retains the optimizer's ML/REML setting. Design matrices,
weights, and offsets must remain unchanged while the optimizers are in use;
construct a new optimizer when those inputs change. Large Python random-effect
systems retain sparse crossproducts.

Native design preparation and prepared ML and REML evaluations release Python's
interpreter lock after copying their inputs, so Python threads can prepare
independent fits and evaluate shared or separate responses concurrently.
Complete-fit throughput also depends on the optimizer: SciPy's COBYQA wrapper
serializes optimizer calls with its own lock.

To fit with prepared analytic covariance gradients, enable them in the control:

```python
from mixedlm import lmer, lmerControl

fit = lmer(
    "Reaction ~ Days + (Days | Subject)",
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

Analytic gradients are cheapest when the random-effect levels are independent,
as in nested or single-factor designs. Crossed or otherwise coupled designs need
a full inverse and can make analytic gradients more expensive than numerical
derivatives.

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

```py
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
cbpp = mlm.load_cbpp()
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
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
    "incidence / size ~ period + (1 | herd)",
    cbpp,
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
orders where polynomial-based construction can overflow.

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

`GLMMOptimizer` and `JointGLMMObjective` prepare their inputs once and reuse
them across parameter evaluations. Each evaluation starts independently, so
parameter order and earlier failed evaluations do not change its result. Treat
the model arrays and family as immutable while using these estimation objects;
construct a new object when the inputs change. Custom families, links, and
covariance structures use the Python implementation.

Prepared native GLMM likelihood evaluations release Python's interpreter lock
after copying their parameters, so Python threads can evaluate one prepared
problem concurrently. As for LMMs, SciPy's COBYQA wrapper still serializes
optimizer calls.

These objectives and modular `GlmerDevfun` callables can be copied and pickled
when their model inputs and custom family are picklable, for example to send
them to a spawned worker process.

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

Independent random intercepts and scalar slopes give a diagonal penalized
precision, solved by scalar division. Larger sparse models use a sparse
penalized system with a fill-reducing ordering, so nested and crossed designs
with many levels stay sparse; small or dense systems use dense kernels.

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
Group contributions are added in a fixed order with compensated summation, so the
quadrature deviance does not depend on the number of native worker threads.

Starting means lie inside the family and link domains. For Poisson models with a
log link, positive counts are transformed with the logarithm before estimating
starting coefficients; zero counts use a small positive starting mean. This keeps
large counts from producing an excessively large initial linear predictor. The
native solver uses the same starting-mean convention as Python, including prior
weights and offsets. Its inner convergence flag remains false if an update or
final deviance is nonfinite.

For Python families and links with restricted predictor domains, PIRLS validates
each proposed predictor before computing its mean. If the initial coefficients
violate the domain, a sparse feasibility solve finds an interior starting point
subject to the observation offsets. Accepted updates remain feasible and reduce
the penalized deviance through step halving. An impossible domain or unfinished
inner solve cannot report convergence. Quadrature nodes outside the valid domain
contribute zero likelihood. Numerically saturated probabilities for unrestricted
logit and similar links retain the existing stable mean clamping.

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
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data, control=ctrl)

# Also available for GLMMs (provides starting theta from a linear approximation)
ctrl = mlm.GlmerControl(em_init=True)
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)", cbpp, family=mlm.families.Binomial(), control=ctrl
)
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

- `COBYQA` - Derivative-free constrained optimization
- `L-BFGS-B` - Quasi-Newton with bounds
- `BFGS` - Quasi-Newton
- `Nelder-Mead` - Simplex method
- `Powell` - Direction set method
- `trust-constr` - Trust region with constraints
- `SLSQP` - Sequential least squares
- `TNC` - Truncated Newton
- `COBYLA` - Constrained optimization by linear approximation

**Optional (requires the `optimizers` extra, which installs nlopt):**

- `nloptwrap_BOBYQA` - Bound-constrained quadratic approximation
- `nloptwrap_NEWUOA` - Derivative-free unconstrained
- `nloptwrap_PRAXIS` - Principal axis
- `nloptwrap_SBPLX` - Subplex algorithm
- `nloptwrap_COBYLA` and `nloptwrap_NELDERMEAD`

`mlm.LmerControl().optimizer` shows the default, and
`mixedlm.estimation.available_optimizers()` lists the installed choices.

### Choosing an Optimizer

```python
# Use a specific optimizer
model = mlm.lmer(
    "Reaction ~ Days + (Days | Subject)",
    data,
    control=mlm.LmerControl(optimizer="L-BFGS-B")
)

# Try all available optimizers
results = model.allFit(data)
print(results.summary)
```

### Convergence Criteria

The optimizer stops when:

1. Gradient is near zero (for gradient-based methods)
2. Change in parameters is below tolerance
3. Change in objective is below tolerance
4. Maximum iterations reached

```python
control = mlm.LmerControl(
    maxiter=50000,   # Maximum iterations (evaluations for TNC and COBYLA)
    ftol=1e-10,      # Objective-change tolerance for solvers that use one
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

The Python ML/REML evaluator also keeps large random-effect systems sparse and
supports observation weights, offsets, and unstructured, independent,
compound-symmetry, and AR(1) random effects.

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

**Limitation:** AGQ is only available for models with a single scalar random effect (random intercept only, one grouping factor).

### PIRLS Algorithm

For fitting GLMMs, mixedlm uses Penalized Iteratively Reweighted Least Squares (PIRLS):

1. Initialize random effects at zero
2. Given current \(\mathbf{b}\), compute working responses and weights
3. Solve a penalized weighted least squares problem
4. Update \(\mathbf{b}\)
5. Repeat until convergence

This is nested within the outer optimization over variance parameters.

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

## Optimization

### Available Optimizers

mixedlm supports multiple optimization algorithms:

**Always available (SciPy):**

- `L-BFGS-B` - Quasi-Newton with bounds (default)
- `BFGS` - Quasi-Newton
- `Nelder-Mead` - Simplex method
- `Powell` - Direction set method
- `trust-constr` - Trust region with constraints
- `SLSQP` - Sequential least squares
- `TNC` - Truncated Newton
- `COBYLA` - Constrained optimization by linear approximation

**Optional (requires additional packages):**

- `bobyqa` - Derivative-free bounded optimization (Py-BOBYQA)
- `newuoa` - Derivative-free unconstrained (nlopt)
- `praxis` - Principal axis (nlopt)
- `sbplx` - Subplex algorithm (nlopt)

### Choosing an Optimizer

```python
# Use a specific optimizer
model = mlm.lmer(
    "y ~ x + (1 | g)",
    data,
    control=mlm.LmerControl(optimizer="bobyqa")
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

- Ensures positive definite covariance matrices
- Improves optimization stability
- Allows variance to approach zero smoothly

### Sparse Matrix Methods

For models with many groups, mixedlm uses sparse matrix operations to efficiently compute:

- \(\mathbf{Z}^T\mathbf{Z}\) (block diagonal structure)
- Cholesky factorizations
- Linear system solutions

This enables fitting models with thousands of groups.

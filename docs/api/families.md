# Families

This page documents distribution families for generalized linear mixed models.

## Overview

Families define the distribution and link function for GLMMs. Access them via `mixedlm.families`:

```python
import mixedlm as mlm

cbpp = mlm.load_cbpp()
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial()
)
```

Family constructors accept either a documented link name or a custom `Link`
instance. Invalid names and family/link combinations raise `ValueError` when
the family is created.

`Link.inverse_interval(lower, upper)` transforms ordered link-scale bounds into
ordered response bounds. Its default handles increasing and decreasing inverse
links. The square-root link also includes zero when the interval crosses zero,
because squaring has an interior minimum there. Custom inverse links with
interior extrema should override this method. Prediction, adjusted-effect, and
marginal-mean confidence intervals all use it.

## Available Families

### Gaussian

For continuous responses (rarely used with `glmer`, use `lmer` instead).

```python
mlm.families.Gaussian(link="identity")
```

**Default link:** identity

**Other links:** log

### Binomial

For binary or proportion data.

```python
mlm.families.Binomial(link="logit")
```

**Default link:** logit

**Other links:** probit, cloglog, cauchit, log

**Usage:**

```py
# Binary outcome
model = mlm.glmer("success ~ x + (1 | g)", data, family=mlm.families.Binomial())

# Two-level factor outcome, such as the N/Y response in VerbAgg
model = mlm.glmer("r2 ~ Anger + (1 | id)", mlm.load_verbagg())

# Proportion (successes / trials)
model = mlm.glmer("successes / trials ~ x + (1 | g)", data, family=mlm.families.Binomial())
```

Binomial responses can be numeric 0/1 values, proportions with trial weights,
or factors with exactly two levels. The second factor level represents success.
Pandas categorical and Polars Enum responses use their declared category order;
ordinary string responses use sorted labels, so `N` maps to 0 and `Y` to 1.
Polars categorical responses use the order of their observed categories,
excluding labels shared with unrelated columns in the category pool.
To choose a different success label, declare the factor order explicitly:

```py
import pandas as pd

data["outcome"] = pd.Categorical(data["outcome"], categories=["Y", "N"])
model = mlm.glmer("outcome ~ x + (1 | g)", data)  # predicts the probability of N
```

The fitted level order is retained for updates, refits, and cross-validation,
including training subsets that contain one class. `simulate()` returns numeric
0/1 responses. `refit()` accepts either those encoded numeric values or factor
labels; numeric arrays keep their 0/1 meaning even when the factor order is
reversed. Unknown labels raise `ValueError`. Missing labels follow `na_action`
during fitting. Numeric and Boolean responses keep their original values.

### Poisson

For count data.

```python
mlm.families.Poisson(link="log")
```

**Default link:** log

**Other links:** identity, sqrt

### Gamma

For positive continuous data with constant coefficient of variation.

```python
mlm.families.Gamma(link="inverse")
```

**Default link:** log

**Other links:** inverse, identity

### InverseGaussian

For positive continuous data.

```python
mlm.families.InverseGaussian(link="1/mu^2")
```

**Default link:** log

**Other links:** 1/mu^2, inverse, identity

### NegativeBinomial

For overdispersed count data.

```python
mlm.families.NegativeBinomial(theta=1.0, link="log")
```

**Parameters:**

- `theta`: Dispersion parameter. Larger values = less overdispersion.

**Default link:** log

**Usage:**

```py
# With known theta
model = mlm.glmer(
    "count ~ x + (1 | g)",
    data,
    family=mlm.families.NegativeBinomial(theta=2.0)
)

# Fit with the supplied fixed theta (default 1.0)
model = mlm.glmer_nb("count ~ x + (1 | g)", data)
```

## Likelihood Reporting

`model.logLik()` includes the response-density constants for all built-in
families, using the fitted Laplace or adaptive Gauss-Hermite approximation.
`AIC()`, `BIC()`, `extractAIC()`, and `model_selection()` use this normalized
likelihood. `get_deviance()` and `REMLcrit()` return `-2 * logLik().value`.
The stored `model.deviance` and `as_function("deviance")` retain the fitting
criterion relative to the saturated conditional density; they remain suitable
for optimization and likelihood-ratio profiles. Continuous response simulation
uses the same precision weights as the conditional densities below.

These quantities differ by a response-dependent constant. For grouped binomial
data with successes \(k_i\), trials \(n_i\), and prior weights \(a_i\),

\[
\log L = -\tfrac12\,\text{model.deviance}
          + \sum_i a_i\log\Pr\{K_i=k_i\mid n_i,p_i=k_i/n_i\}.
\]

The trial counts are retained separately from the effective fitting weights
\(a_i n_i\). Numeric proportions supplied without explicit trials use their
weights as trial counts. Normalized binomial likelihood reporting requires
whole-number trials and successes; fitting a fractional-count criterion remains
possible, but its `logLik()` raises `ValueError`. Binary 0/1 responses have a zero
saturated constant and permit arbitrary positive prior weights.

| Family | Conditional density and weight meaning |
| --- | --- |
| Binomial | Bernoulli for binary responses; binomial counts for grouped responses. Prior weights multiply each log probability. |
| Poisson | Poisson mean \(\mu_i\); weights multiply each log probability. |
| NegativeBinomial | Negative binomial mean \(\mu_i\) and fixed `theta`; weights multiply each log probability. |
| Gaussian | Normal mean \(\mu_i\), variance \(1/w_i\). |
| Gamma | Gamma shape \(w_i\), scale \(\mu_i/w_i\). |
| InverseGaussian | Inverse Gaussian mean \(\mu_i\), shape \(w_i\). |

Continuous GLMM families fix dispersion at one: their prior weights specify
precision, and their densities have variance \(V(\mu_i)/w_i\). The dispersion
is not estimated or counted as a fitted parameter. Use `lmer()` for a Gaussian
model with estimated residual scale. Poisson and negative binomial weights
give ordinary replicated-observation likelihoods when integral, and weighted
power likelihoods otherwise. `glmer_nb()` fixes `theta` at the supplied value.

The [R family documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html)
describes the response encodings, and the
[lme4 deviance documentation](https://lme4.github.io/lme4/reference/merMod-class.html#deviance-and-log-likelihood-of-glmms)
distinguishes relative and absolute, conditional and marginal deviances. lme4
documents that its adaptive-quadrature likelihood may omit response constants;
mixedlm includes them for every supported approximation.

## Custom Families

### CustomFamily

Create a custom family with user-defined functions.

```python
from mixedlm.families import CustomFamily
import numpy as np

class MyFamily(CustomFamily):
    name = "my_family"

    def __init__(self):
        super().__init__(link="log")

    def variance(self, mu):
        return mu  # Poisson variance in this example

    def deviance_resids(self, y, mu, wt):
        # Custom deviance residuals
        return 2 * wt * (y * np.log(y / mu) - (y - mu))
```

Custom families can fit using their variance and deviance functions alone.
To enable normalized `logLik()`, AIC, and BIC, also implement the optional
`log_likelihood(y, mu, wt, *, trials=None)` hook. For the Poisson example above:

```python
from scipy import stats

def log_likelihood(self, y, mu, wt, *, trials=None):
    return float(np.sum(wt * stats.poisson.logpmf(y, mu)))

MyFamily.log_likelihood = log_likelihood
```

The hook must include all response-density constants and handle the saturated
mean `mu=y`, including boundary values. It must satisfy
`sum(deviance_resids(y, mu, wt)) == 2 * (log_likelihood(y, y, wt) - log_likelihood(y, mu, wt))`.
Without this hook, likelihood reporting raises `NotImplementedError`; model
summaries display `NA` for these quantities. A quasi family defines a mean and
variance relationship without a probability density, so its normalized
likelihood and information criteria raise `ValueError`.

To use `simulate()`, `bootMer()`, or `powerSim()`, also implement
`simulate(self, mu, rng=None, *, weights=None, trials=None)`, returning one draw
per mean from the supplied NumPy random stream. `weights` are prior weights
(precisions for families with a dispersion parameter) and `trials` are binomial
trial counts. The older `simulate(self, mu, rng=None)` signature still works.
Without this method, those functions raise `NotImplementedError` instead of
inventing a response distribution. For the Poisson example:

```python
def simulate(self, mu, rng=None, *, weights=None, trials=None):
    return rng.poisson(mu).astype(float)

MyFamily.simulate = simulate
```

### QuasiFamily

For quasi-likelihood models with custom variance functions.

```python
from mixedlm.families import Binomial, Poisson, QuasiFamily

# Quasi-Poisson for overdispersed counts
quasi_pois = QuasiFamily(Poisson(), phi=2.0)

# Quasi-binomial for overdispersed proportions
quasi_binom = QuasiFamily(Binomial(), phi=2.0)
```

**Parameters:**

- `base_family`: Family whose link, variance, and deviance are wrapped
- `phi`: Positive dispersion multiplier

Quasi families define no response distribution, so `simulate()`, `bootMer()`,
and `powerSim()` raise `NotImplementedError` for them.

## Family Components

Each family provides:

### link

The link function \(g(\mu)\).

```python
import numpy as np

fam = mlm.families.Binomial()
mu = np.array([0.2, 0.5, 0.9])
eta = fam.link(mu)  # log-odds
```

### linkinv

The inverse link function \(g^{-1}(\eta)\).

```python
mu = fam.linkinv(eta)  # probabilities
```

### variance

The variance function \(V(\mu)\).

```python
var = fam.variance(mu)
```

### deviance_residuals

Per-observation deviance contributions, like R's `dev.resids`; their sum is the
deviance. An optional third argument supplies prior weights.

```python
y = np.array([0.0, 1.0, 1.0])
dev_resid = fam.deviance_residuals(y, mu)
```

### Distribution Diagnostics

For fitted GLMMs, the diagnostics module can compare the observed conditional
variance and zero count with the selected family:

```python
dispersion = mlm.diagnostics.check_overdispersion(model)
zeros = mlm.diagnostics.check_zero_inflation(model)
```

The zero check supports Poisson, negative-binomial, and binomial families.
Quasi-families define a variance function but not the zero probability required
for that check.

## Link Functions

### Available Links

| Link | Function | Inverse | Typical Use |
|------|----------|---------|-------------|
| identity | \(\eta = \mu\) | \(\mu = \eta\) | Gaussian |
| log | \(\eta = \log(\mu)\) | \(\mu = e^\eta\) | Poisson, Gamma |
| logit | \(\eta = \log(\frac{\mu}{1-\mu})\) | \(\mu = \frac{e^\eta}{1+e^\eta}\) | Binomial |
| probit | \(\eta = \Phi^{-1}(\mu)\) | \(\mu = \Phi(\eta)\) | Binomial |
| cloglog | \(\eta = \log(-\log(1-\mu))\) | \(\mu = 1-e^{-e^\eta}\) | Binomial |
| inverse | \(\eta = 1/\mu\) | \(\mu = 1/\eta\) | Gamma |
| sqrt | \(\eta = \sqrt{\mu}\) | \(\mu = \eta^2\) | Poisson |

### Choosing a Link

- **logit**: Standard for binary/proportional data. Coefficients are log-odds ratios.
- **probit**: Similar to logit, assumes normal latent variable.
- **cloglog**: For rare events, asymmetric around 0.5.
- **log**: For counts. Coefficients are log rate ratios.
- **identity**: When you want coefficients on the original scale.

## Usage Examples

### Binomial with Different Links

```python
import mixedlm as mlm

# Logit (default)
model_logit = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial(link="logit")
)

# Probit
model_probit = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial(link="probit")
)

# Compare fits
print(f"Logit AIC: {model_logit.AIC()}")
print(f"Probit AIC: {model_probit.AIC()}")
```

### Overdispersed Counts

```py
# Check for overdispersion with Poisson
pois_model = mlm.glmer(
    "count ~ x + (1 | g)",
    data,
    family=mlm.families.Poisson()
)
print(mlm.diagnostics.check_overdispersion(pois_model))

# If overdispersed, use negative binomial with a chosen dispersion
nb_model = mlm.glmer_nb("count ~ x + (1 | g)", data, theta=2.0)
```

### Quasi-Likelihood Dispersion

```py
from mixedlm.families import QuasiFamily

# Double the variance of a Poisson model without changing its log link
overdispersed = QuasiFamily(mlm.families.Poisson(), phi=2.0)

model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=overdispersed
)
```

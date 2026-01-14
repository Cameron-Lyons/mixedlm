# Families

This page documents distribution families for generalized linear mixed models.

## Overview

Families define the distribution and link function for GLMMs. Access them via `mixedlm.families`:

```python
import mixedlm as mlm

model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial()
)
```

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

```python
# Binary outcome
model = mlm.glmer("success ~ x + (1 | g)", data, family=mlm.families.Binomial())

# Proportion (successes / trials)
model = mlm.glmer("successes / trials ~ x + (1 | g)", data, family=mlm.families.Binomial())
```

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

**Default link:** inverse

**Other links:** identity, log

### InverseGaussian

For positive continuous data.

```python
mlm.families.InverseGaussian(link="1/mu^2")
```

**Default link:** 1/mu^2

**Other links:** inverse, identity, log

### NegativeBinomial

For overdispersed count data.

```python
mlm.families.NegativeBinomial(theta=1.0, link="log")
```

**Parameters:**

- `theta`: Dispersion parameter. Larger values = less overdispersion.

**Default link:** log

**Usage:**

```python
# With known theta
model = mlm.glmer(
    "count ~ x + (1 | g)",
    data,
    family=mlm.families.NegativeBinomial(theta=2.0)
)

# Estimate theta automatically
model = mlm.glmer_nb("count ~ x + (1 | g)", data)
```

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
        return mu ** 1.5  # Custom variance function

    def deviance_residuals(self, y, mu, wt=1):
        # Custom deviance residuals
        return 2 * wt * (y * np.log(y / mu) - (y - mu))
```

### QuasiFamily

For quasi-likelihood models with custom variance functions.

```python
from mixedlm.families import QuasiFamily

# Quasi-Poisson for overdispersed counts
quasi_pois = QuasiFamily(variance_func="mu", link="log")

# Quasi-binomial for overdispersed proportions
quasi_binom = QuasiFamily(variance_func="mu*(1-mu)", link="logit")
```

**Parameters:**

- `variance_func`: String expression for variance as function of mu
- `link`: Link function name

## Family Components

Each family provides:

### link

The link function \(g(\mu)\).

```python
fam = mlm.families.Binomial()
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

Deviance residuals for model diagnostics.

```python
dev_resid = fam.deviance_residuals(y, mu)
```

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
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial(link="logit")
)

# Probit
model_probit = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=mlm.families.Binomial(link="probit")
)

# Compare fits
print(f"Logit AIC: {model_logit.AIC()}")
print(f"Probit AIC: {model_probit.AIC()}")
```

### Overdispersed Counts

```python
# Check for overdispersion with Poisson
pois_model = mlm.glmer(
    "count ~ x + (1 | g)",
    data,
    family=mlm.families.Poisson()
)

# If overdispersed, use negative binomial
nb_model = mlm.glmer_nb("count ~ x + (1 | g)", data)
print(f"Estimated theta: {nb_model.family.theta}")
```

### Custom Variance Function

```python
from mixedlm.families import QuasiFamily

# Power variance: Var(Y) = phi * mu^p
power_var = QuasiFamily(
    variance_func="mu**1.5",
    link="log"
)

model = mlm.glmer(
    "y ~ x + (1 | g)",
    data,
    family=power_var
)
```

# Nonlinear Mixed Models

This tutorial covers nonlinear mixed models (NLMMs) for data that follows a nonlinear relationship.

## When to Use NLMMs

Use nonlinear mixed models when:

- The relationship between predictors and response is inherently nonlinear
- A linear approximation would be inadequate
- You have repeated measurements and group-level variation
- The nonlinear function has meaningful parameters (e.g., asymptote, rate)

Common applications:

- Growth curves (asymptotic, logistic, Gompertz)
- Pharmacokinetics (drug concentration over time)
- Dose-response curves
- Enzyme kinetics (Michaelis-Menten)

## Self-Starting Models

mixedlm provides self-starting nonlinear models that automatically compute starting values.

### Available Models

| Model | Function | Description |
|-------|----------|-------------|
| `SSasymp` | \(y = A + (R_0 - A)e^{-e^{lrc} \cdot x}\) | Asymptotic regression |
| `SSlogis` | \(y = \frac{A}{1 + e^{(xmid - x)/scal}}\) | Logistic growth |
| `SSmicmen` | \(y = \frac{V_m \cdot x}{K + x}\) | Michaelis-Menten kinetics |
| `SSfpl` | \(y = A + \frac{B - A}{1 + e^{(xmid - x)/scal}}\) | Four-parameter logistic |
| `SSgompertz` | \(y = A \cdot e^{-b_2 \cdot b_3^x}\) | Gompertz growth |
| `SSbiexp` | \(y = A_1 e^{-e^{lrc_1} x} + A_2 e^{-e^{lrc_2} x}\) | Biexponential decay |

## Asymptotic Regression

### Example: Growth to Asymptote

```python
import mixedlm as mlm
from mixedlm.nlme import SSasymp

# Example: weight gain approaching maximum
model = mlm.nlmer(
    "weight ~ SSasymp(time, Asym, R0, lrc) + (Asym | subject)",
    data,
    start=SSasymp.get_start(data, "weight", "time")
)
```

Parameters:

- `Asym`: Horizontal asymptote (final value)
- `R0`: Response at time 0
- `lrc`: Log of the rate constant

### Interpreting Results

```python
# Fixed effects: population-level parameters
model.fixef()
# {'Asym': 100.5, 'R0': 20.3, 'lrc': -2.1}

# Random effects: subject deviations
model.ranef()
# Subjects with higher Asym have higher final weights
```

## Logistic Growth

For S-shaped growth curves:

```python
from mixedlm.nlme import SSlogis

model = mlm.nlmer(
    "size ~ SSlogis(time, Asym, xmid, scal) + (Asym | subject)",
    data,
    start=SSlogis.get_start(data, "size", "time")
)
```

Parameters:

- `Asym`: Upper asymptote
- `xmid`: x-value at inflection point (50% of Asym)
- `scal`: Scale parameter (steepness)

## Michaelis-Menten Kinetics

For enzyme kinetics and saturation curves:

```python
from mixedlm.nlme import SSmicmen

model = mlm.nlmer(
    "velocity ~ SSmicmen(conc, Vm, K) + (Vm | enzyme)",
    data,
    start=SSmicmen.get_start(data, "velocity", "conc")
)
```

Parameters:

- `Vm`: Maximum velocity (saturation level)
- `K`: Michaelis constant (concentration at half-max velocity)

## Specifying Random Effects

### Random Effect on One Parameter

Most common: random intercepts on the asymptote:

```python
model = mlm.nlmer(
    "y ~ SSasymp(x, Asym, R0, lrc) + (Asym | group)",
    data,
    start=...
)
```

### Random Effects on Multiple Parameters

```python
model = mlm.nlmer(
    "y ~ SSasymp(x, Asym, R0, lrc) + (Asym + lrc | group)",
    data,
    start=...
)
```

This allows both the asymptote and rate to vary by group.

### Uncorrelated Random Effects

```python
model = mlm.nlmer(
    "y ~ SSasymp(x, Asym, R0, lrc) + (Asym || group) + (lrc || group)",
    data,
    start=...
)
```

## Starting Values

Nonlinear optimization requires good starting values.

### Using Self-Starting Functions

The `get_start` method computes starting values from data:

```python
from mixedlm.nlme import SSlogis

start = SSlogis.get_start(data, y_col="response", x_col="time")
# {'Asym': 100.0, 'xmid': 5.0, 'scal': 2.0}
```

### Manual Starting Values

Provide your own starting values:

```python
model = mlm.nlmer(
    "y ~ SSasymp(x, Asym, R0, lrc) + (Asym | group)",
    data,
    start={'Asym': 100, 'R0': 10, 'lrc': -1}
)
```

### Getting Starting Values from Linear Fit

For some models, transform and fit linearly first:

```python
import numpy as np

# For asymptotic model, if asymptote is known
# log(Asym - y) is linear in x
```

## Model Results

### Fixed Effects

Population-level parameter estimates:

```python
model.fixef()
```

### Random Effects

Group-level deviations:

```python
model.ranef()
```

### Variance Components

Random effects variance:

```python
model.VarCorr()
```

### Predictions

```python
# Fitted values
model.fitted()

# Population-level predictions for new data. The predictor column used during
# fitting is remembered automatically.
model.predict(newdata=new_data)

# Add fitted random effects for known groups; unseen groups use population values.
model.predict(newdata=new_data, group_var="subject")
```

## Bootstrap Inference

For confidence intervals on nonlinear parameters:

```python
boot_result = mlm.bootstrap_nlmer(model, n_boot=500, seed=42)

# Bootstrap CIs
mlm.bootCI(boot_result, component="all")
```

## Custom Nonlinear Functions

Define a model with parameter names, predictions, gradients, and starting values:

```python
from mixedlm.nlme import NonlinearModel
import numpy as np

class MyModel(NonlinearModel):
    @property
    def name(self):
        return "exponential_decay"

    @property
    def param_names(self):
        return ["a", "b", "c"]

    def predict(self, params, x):
        a, b, c = params
        return a * np.exp(-b * x) + c

    def gradient(self, params, x):
        a, b, c = params
        exp_term = np.exp(-b * x)
        return np.column_stack([
            exp_term,           # d/da
            -a * x * exp_term,  # d/db
            np.ones_like(x)     # d/dc
        ])

    def get_start(self, x, y):
        return np.array([y.max() - y.min(), 0.1, y.min()])
```

Then use it:

```python
model = mlm.nlmer(
    MyModel(),
    data,
    x_var="x",
    y_var="y",
    group_var="group",
    random_params=["a"],
)
```

Custom models and subclasses use their Python prediction and gradient methods,
even when their display name matches a built-in model. The six built-in model
classes use the native fast path when available and their prediction and
gradient methods are unchanged. Replacing either method on a built-in instance
or class selects the Python path as well.

Python fits reuse group and weight preparation across covariance evaluations.
The low-level `NLMMOptimizer(..., n_jobs=2)` shares a worker pool across the fit
when there are at least two groups. Threads are released when fitting finishes,
including after a model error or interruption. The default `n_jobs=1` runs
serially; threading is most useful for expensive group prediction and gradient
functions and can add overhead for small models.

## Convergence Issues

Both backends update fixed and random parameters together using small group
systems. A line search checks the penalized residual error and shortens a step
when needed. This avoids the slow alternating updates that previously required
large inner iteration budgets for many models.

During optimization, a failed trial evaluation receives a finite penalty so the
optimizer can try other parameter values. If the final evaluation fails,
`nlmer()` and `refit()` raise `RuntimeError` with the evaluation's failure reason.
They also reject nonfinite or incorrectly shaped estimates and a nonpositive
residual scale. The failure penalty is never returned as a fitted deviance.

A valid final evaluation has `converged=True` only when both the outer optimizer
and inner parameter updates converge. `pnls_converged` records the inner status;
iteration limits and failed line searches leave it false. `nlmer()` warns
when the final inner solve is unfinished, and the summary identifies that case.

`pnls_maxiter` sets the inner iteration budget (default 50). `pnls_tol` sets the
largest allowed absolute proposed change in any fixed or random parameter
(default `1e-6`). The tolerance applies to the full proposal, so a heavily
shortened step cannot falsely establish convergence. Both controls apply to
Python and native fitting. The existing `maxiter` argument limits outer
covariance optimization independently. Results retain the
inner controls for `refit()` and `update()`, which accept overrides:

```python
refined = model.refit(pnls_maxiter=2000, pnls_tol=1e-6)
print(refined.converged, refined.pnls_converged)
```

Inspect convergence before using the estimates. `bootstrap_nlmer()`, `bootMer()`,
and bootstrap confidence intervals count refits that fail or do not converge as
failed replicates and exclude them from their intervals. If every replicate
fails, interval bounds are NaN.

NLMMs are particularly sensitive to:

### Starting Values

Poor starting values lead to convergence failure or local optima:

```python
# Try different starting values
for scale in [0.5, 1.0, 2.0]:
    start = {'Asym': 100 * scale, 'R0': 10, 'lrc': -1}
    try:
        model = mlm.nlmer(..., start=start)
        if mlm.convergence_ok(model):
            break
    except:
        continue
```

### Model Complexity

Start simple and add complexity:

1. First: random intercept on one parameter
2. Then: add random effects on other parameters
3. Finally: allow correlations if supported by data

### Optimizer Settings

```python
from mixedlm import NlmerControl

control = NlmerControl(maxiter=500, tol=1e-6)
model = mlm.nlmer(..., control=control)
```

## Complete Example

```python
import mixedlm as mlm
from mixedlm.nlme import SSlogis
import numpy as np

# Simulate growth data
np.random.seed(42)
n_subjects = 20
times = np.tile(np.linspace(0, 10, 15), n_subjects)
subjects = np.repeat(range(n_subjects), 15)

# True parameters with subject variation
true_asym = 100 + np.random.normal(0, 10, n_subjects)[subjects]
true_xmid = 5
true_scal = 1.5

y = true_asym / (1 + np.exp((true_xmid - times) / true_scal))
y += np.random.normal(0, 5, len(y))

import pandas as pd
data = pd.DataFrame({
    'growth': y,
    'time': times,
    'subject': [f'S{i}' for i in subjects]
})

# Get starting values
start = SSlogis.get_start(data, 'growth', 'time')
print(f"Starting values: {start}")

# Fit model
model = mlm.nlmer(
    "growth ~ SSlogis(time, Asym, xmid, scal) + (Asym | subject)",
    data,
    start=start
)

# Results
print("\nFixed effects (population parameters):")
print(model.fixef())

print("\nVariance components:")
print(model.VarCorr())

# Subject-specific asymptotes
ranef = model.ranef()
print("\nSubject deviations from population Asym:")
print(ranef)
```

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

Each model is a class. Pass an instance to `nlmer()` with the names of the
predictor, response, and grouping columns.

## Example Data

The examples below use simulated logistic growth curves for 20 subjects whose
asymptotes vary around 100:

```python
import numpy as np
import pandas as pd

import mixedlm as mlm
from mixedlm.nlme import SSlogis

rng = np.random.default_rng(42)
n_subjects, n_times = 20, 15
time = np.tile(np.linspace(0, 10, n_times), n_subjects)
subject = np.repeat(np.arange(n_subjects), n_times)
asym = 100 + rng.normal(0, 10, n_subjects)[subject]
growth = asym / (1 + np.exp((5 - time) / 1.5)) + rng.normal(0, 3, time.size)

data = pd.DataFrame({
    "growth": growth,
    "time": time,
    "subject": [f"S{i:02d}" for i in subject],
})
```

## Logistic Growth

For S-shaped growth curves:

```python
model = mlm.nlmer(
    SSlogis(),
    data,
    x_var="time",
    y_var="growth",
    group_var="subject",
    random_params=["Asym"],
)
print(model.summary())
```

Parameters:

- `Asym`: Upper asymptote
- `xmid`: x-value at inflection point (50% of Asym)
- `scal`: Scale parameter (steepness)

### Interpreting Results

```python
# Fixed effects: population-level parameters
model.fixef()
# {'Asym': 99.8, 'xmid': 5.01, 'scal': 1.48}

# Random effects: subject deviations, by grouping factor and parameter
model.ranef()["subject"]["Asym"]

# Variance components
model.VarCorr()
```

Subjects with a positive `Asym` deviation level off above the population asymptote.

## Asymptotic Regression

For growth or decay towards an asymptote:

```py
from mixedlm.nlme import SSasymp

model = mlm.nlmer(
    SSasymp(), data, x_var="time", y_var="weight", group_var="subject",
    random_params=["Asym"],
)
```

Parameters:

- `Asym`: Horizontal asymptote (final value)
- `R0`: Response at time 0
- `lrc`: Log of the rate constant

## Michaelis-Menten Kinetics

For enzyme kinetics and saturation curves:

```py
from mixedlm.nlme import SSmicmen

model = mlm.nlmer(
    SSmicmen(), data, x_var="conc", y_var="velocity", group_var="enzyme",
    random_params=["Vm"],
)
```

Parameters:

- `Vm`: Maximum velocity (saturation level)
- `K`: Michaelis constant (concentration at half-max velocity)

## Specifying Random Effects

`random_params` lists the parameters that vary by group, by name or index. It
defaults to all parameters.

### Random Effect on One Parameter

Most common: a random asymptote, as in the fit above.

### Random Effects on Multiple Parameters

```python
model_2re = mlm.nlmer(
    SSlogis(),
    data,
    x_var="time",
    y_var="growth",
    group_var="subject",
    random_params=["Asym", "xmid"],
)
print(model_2re.VarCorr())
```

This allows both the asymptote and the inflection point to vary by subject.

## Starting Values

Nonlinear optimization requires good starting values.

### Using Self-Starting Functions

Without `start`, `nlmer()` calls the model's `get_start(x, y)` method, which
estimates starting values from the data, in `param_names` order:

```python
SSlogis().get_start(data["time"].to_numpy(), data["growth"].to_numpy())
# array([107.5, 5., 2.5])
```

### Manual Starting Values

Provide your own starting values by parameter name:

```python
model = mlm.nlmer(
    SSlogis(),
    data,
    x_var="time",
    y_var="growth",
    group_var="subject",
    random_params=["Asym"],
    start={"Asym": 100.0, "xmid": 5.0, "scal": 1.5},
)
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
new_data = pd.DataFrame({"time": [0.0, 5.0, 10.0], "subject": ["S00", "S00", "new"]})
model.predict(newdata=new_data)

# Add fitted random effects for known groups; unseen groups use population values.
model.predict(newdata=new_data, group_var="subject")
```

## Bootstrap Inference

For confidence intervals on nonlinear parameters:

```python
boot_result = mlm.bootstrap_nlmer(model, n_boot=50, seed=42)

# Bootstrap CIs
mlm.bootCI(boot_result, component="all")
```

Use several hundred or more replicates for reported intervals.
`bootstrap_nlmer()`, `bootMer(model, nsim=500, seed=42, n_jobs=2)` and
`model.confint(n_boot=500, seed=42, n_jobs=2)` accept `n_jobs` for parallel refits.
Simulation preserves the serial draw sequence, and failed refits are excluded
in both modes. Use `n_jobs=1` for small jobs where process startup would dominate.
In scripts using process spawning, run parallel bootstrap inside an
`if __name__ == "__main__":` guard; custom model classes must be importable.

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

Then use it, here on simulated decay curves for 12 groups:

```python
x = np.tile(np.linspace(0, 5, 10), 12)
group = np.repeat(np.arange(12), 10)
a = 10 + rng.normal(0, 1, 12)[group]
decay = pd.DataFrame({
    "x": x,
    "y": a * np.exp(-0.8 * x) + 2 + rng.normal(0, 0.2, x.size),
    "group": group,
})

decay_model = mlm.nlmer(
    MyModel(),
    decay,
    x_var="x",
    y_var="y",
    group_var="group",
    random_params=["a"],
)
decay_model.fixef()
```

Custom models and subclasses use their Python prediction and gradient methods,
even when their display name matches a built-in model. The six built-in model
classes use the native fast path when available and their prediction and
gradient methods are unchanged. Replacing either method on a built-in instance
or class selects the Python path as well.

The low-level `NLMMOptimizer(..., n_jobs=2)` evaluates groups on a thread pool
during a Python fit. The default `n_jobs=1` runs serially; threading helps only
for expensive prediction and gradient functions and adds overhead for small models.

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
failed replicates and exclude them from their intervals. With fewer than two
successful replicates, interval bounds and standard errors are NaN.

NLMMs are particularly sensitive to:

### Starting Values

Poor starting values lead to convergence failure or local optima:

```python
# Try different starting values
for scale in [0.5, 1.0, 2.0]:
    start = {"Asym": 100 * scale, "xmid": 5.0, "scal": 1.5}
    try:
        model = mlm.nlmer(
            SSlogis(), data, x_var="time", y_var="growth", group_var="subject",
            random_params=["Asym"], start=start,
        )
    except RuntimeError:
        continue
    if mlm.convergence_ok(model):
        break
```

### Model Complexity

Start simple and add complexity:

1. First: a random effect on one parameter
2. Then: add random effects on other parameters

### Optimizer Settings

`nlmer()` takes optimizer settings as keyword arguments: `method` and `maxiter`
for the outer covariance optimization, and `pnls_maxiter` and `pnls_tol` for the
inner parameter updates:

```python
model = mlm.nlmer(
    SSlogis(),
    data,
    x_var="time",
    y_var="growth",
    group_var="subject",
    random_params=["Asym"],
    method="L-BFGS-B",
    maxiter=500,
    pnls_maxiter=100,
)
```

## Complete Example

```python
import numpy as np
import pandas as pd

import mixedlm as mlm
from mixedlm.nlme import SSlogis

# Simulate logistic growth with subject-specific asymptotes
rng = np.random.default_rng(42)
n_subjects, n_times = 20, 15
times = np.tile(np.linspace(0, 10, n_times), n_subjects)
subjects = np.repeat(np.arange(n_subjects), n_times)
true_asym = 100 + rng.normal(0, 10, n_subjects)[subjects]
growth = true_asym / (1 + np.exp((5 - times) / 1.5)) + rng.normal(0, 5, times.size)

data = pd.DataFrame({
    "growth": growth,
    "time": times,
    "subject": [f"S{i:02d}" for i in subjects],
})

# Fit with a random asymptote; starting values come from SSlogis().get_start()
model = mlm.nlmer(
    SSlogis(),
    data,
    x_var="time",
    y_var="growth",
    group_var="subject",
    random_params=["Asym"],
)
print("Converged:", model.converged)

print("\nFixed effects (population parameters):")
print(model.fixef())

print("\nVariance components:")
print(model.VarCorr())

# Subject-specific deviations from the population asymptote
print("\nSubject deviations from population Asym:")
print(model.ranef()["subject"]["Asym"])
```

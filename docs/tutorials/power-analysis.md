# Power Analysis

This tutorial covers power analysis for mixed models using simulation-based methods.

## Overview

Power analysis helps you determine:

- Whether your study has sufficient power to detect an effect
- What sample size you need to achieve target power
- How power changes with different design choices

mixedlm provides simulation-based power analysis, similar to R's simr package.

## Basic Workflow

1. Fit a model to pilot data (or specify parameters)
2. Use `powerSim()` to estimate power
3. Use `powerCurve()` to see how power changes with sample size
4. Use `extend()` to simulate larger datasets

## Power Simulation

### Example: Sleep Study

```python
import mixedlm as mlm

# Fit model to existing data
data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Simulate power for the Days effect
power = mlm.powerSim(model, data, nsim=100, test="fixed")
print(power)
```

Output:

```
Power for fixed effect 'Days': 99.0% (95% CI: 94.6%, 100.0%)
Based on 100 simulations
Effect size: 10.47
```

### What powerSim Does

1. Simulates new response data from the fitted model
2. Refits the model to each simulated dataset
3. Tests the specified effect
4. Reports the proportion of significant results (power)

### Testing Different Effects

```python
# Test a specific fixed effect
power = mlm.powerSim(model, data, nsim=100, test="fixed", term="Days")

# Test random effects structure
power = mlm.powerSim(model, data, nsim=100, test="random")
```

## Power Curves

See how power changes with sample size:

```python
# Power curve varying number of subjects
curve = mlm.powerCurve(
    model, data,
    along="Subject",  # Vary this grouping factor
    breaks=[10, 15, 20, 25, 30],  # Sample sizes to test
    nsim=100
)
print(curve)
curve.plot()
```

### Varying Observations per Subject

```python
# Power curve varying observations per subject
curve = mlm.powerCurve(
    model, data,
    along="n",  # Vary within-group sample size
    breaks=[5, 10, 15, 20],
    nsim=100
)
```

## Extending Datasets

Create larger simulated datasets:

```python
# Double the number of subjects
extended = mlm.extend(model, data, along="Subject", n=36)
print(f"Original: {data['Subject'].nunique()} subjects")
print(f"Extended: {extended['Subject'].nunique()} subjects")
```

### Extension Methods

```python
# Add more subjects
extended = mlm.extend(model, data, along="Subject", n=50)

# Add more observations per subject
extended = mlm.extend(model, data, along="n", n=20)

# Both: more subjects with more observations
extended = mlm.extend(model, data, along="Subject", n=50)
extended = mlm.extend(model, extended, along="n", n=20)
```

## Setting Effect Sizes

### Using Fitted Values

By default, power is calculated for the effect size in the fitted model:

```python
print(f"Effect size from model: {model.fixef()['Days']}")
power = mlm.powerSim(model, data, nsim=100)
```

### Specifying Target Effect Size

Modify the model to test a different effect size:

```python
import copy

# Create model with different effect size
model_modified = copy.deepcopy(model)
model_modified._fixef['Days'] = 5.0  # Smaller effect

power = mlm.powerSim(model_modified, data, nsim=100)
```

### Minimum Detectable Effect

Find the smallest effect detectable with 80% power:

```python
effect_sizes = [2, 4, 6, 8, 10]
powers = []

for effect in effect_sizes:
    model_mod = copy.deepcopy(model)
    model_mod._fixef['Days'] = effect
    p = mlm.powerSim(model_mod, data, nsim=100)
    powers.append(p.power)

# Find where power crosses 80%
import numpy as np
mde_idx = np.searchsorted(powers, 0.80)
print(f"MDE for 80% power: ~{effect_sizes[mde_idx]}")
```

## Designing a New Study

When you don't have pilot data:

```python
import pandas as pd
import numpy as np

# Create hypothetical data structure
n_subjects = 20
n_obs_per_subject = 10

data = pd.DataFrame({
    'Subject': np.repeat(range(n_subjects), n_obs_per_subject),
    'Days': np.tile(range(n_obs_per_subject), n_subjects),
    'Reaction': np.random.normal(250, 50, n_subjects * n_obs_per_subject)
})

# Fit model to get structure
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Modify parameters to expected values
model._fixef['(Intercept)'] = 250
model._fixef['Days'] = 10  # Expected effect
# Set variance components...

# Now run power analysis
power = mlm.powerSim(model, data, nsim=100)
```

## Sample Size Determination

### Target Power Approach

```python
target_power = 0.80

# Test different sample sizes
for n_subj in [10, 15, 20, 25, 30, 35, 40]:
    extended = mlm.extend(model, data, along="Subject", n=n_subj)
    power = mlm.powerSim(model, extended, nsim=100)
    print(f"n={n_subj}: power={power.power:.1%}")
    if power.power >= target_power:
        print(f"  -> Minimum n = {n_subj} for {target_power:.0%} power")
        break
```

### Using Power Curve

```python
curve = mlm.powerCurve(
    model, data,
    along="Subject",
    breaks=list(range(10, 50, 5)),
    nsim=100
)

# Find sample size for target power
target = 0.80
for n, p in zip(curve.breaks, curve.powers):
    if p >= target:
        print(f"Need n={n} subjects for {target:.0%} power")
        break
```

## Parallel Simulation

For faster computation with many simulations:

```python
power = mlm.powerSim(
    model, data,
    nsim=1000,
    n_jobs=-1  # Use all cores
)
```

## Interpreting Results

### Power Estimate

```python
power = mlm.powerSim(model, data, nsim=500)
print(f"Power: {power.power:.1%}")
print(f"95% CI: [{power.ci_lower:.1%}, {power.ci_upper:.1%}]")
```

The confidence interval reflects uncertainty from the simulation.

### Number of Simulations

More simulations = narrower CI:

| nsim | Typical CI width |
|------|-----------------|
| 100 | ±10% |
| 500 | ±4% |
| 1000 | ±3% |

Use 100-200 for exploration, 500-1000 for final estimates.

## Common Pitfalls

### Overly Optimistic Power

Power estimates based on pilot data can be optimistic if:

- Effect size in pilot is overestimated
- Variance is underestimated
- Model is too simple

### Convergence Failures

Some simulations may fail to converge:

```python
power = mlm.powerSim(model, data, nsim=100)
print(f"Successful fits: {power.n_success} / {power.nsim}")
```

High failure rates suggest model issues.

## Complete Example

```python
import mixedlm as mlm
import numpy as np

# Load pilot data
data = mlm.load_sleepstudy()

# Fit model
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
print(f"Effect of Days: {model.fixef()['Days']:.2f}")

# Current power
print("\n=== Current Power ===")
power = mlm.powerSim(model, data, nsim=200)
print(f"Power: {power.power:.1%} ({power.ci_lower:.1%}, {power.ci_upper:.1%})")

# Power curve
print("\n=== Power Curve ===")
curve = mlm.powerCurve(
    model, data,
    along="Subject",
    breaks=[10, 15, 18, 20, 25, 30],
    nsim=100
)
for n, p in zip(curve.breaks, curve.powers):
    marker = " <-- current" if n == 18 else ""
    print(f"  n={n:2d}: {p:.1%}{marker}")

# Sample size for smaller effect
print("\n=== Sample Size for Effect=5 ===")
import copy
model_small = copy.deepcopy(model)
model_small._fixef['Days'] = 5.0

for n in [20, 30, 40, 50, 60]:
    extended = mlm.extend(model_small, data, along="Subject", n=n)
    p = mlm.powerSim(model_small, extended, nsim=100)
    print(f"  n={n}: {p.power:.1%}")
    if p.power >= 0.80:
        break
```

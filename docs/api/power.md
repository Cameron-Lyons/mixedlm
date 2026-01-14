# Power Analysis

This page documents functions for simulation-based power analysis.

## Power Simulation

### powerSim

Simulate power for detecting an effect.

```python
import mixedlm as mlm

power = mlm.powerSim(model, data, nsim=100, test="fixed")
```

**Parameters:**

- `model`: Fitted mixed model
- `data`: Original data frame
- `nsim`: Number of simulations
- `test`: What to test. Options: `"fixed"` (fixed effects), `"random"` (random effects)

**Returns:** PowerResult object with:

- `power`: Estimated power (proportion significant)
- `ci_lower`, `ci_upper`: Confidence interval for power
- `nsim`: Number of simulations run
- `n_success`: Number of successful model fits

**Example:**

```python
data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

power = mlm.powerSim(model, data, nsim=200)
print(power)
# Power for fixed effect 'Days': 99.5% (95% CI: 97.2%, 100.0%)
```

### powerCurve

Compute power across a range of sample sizes.

```python
curve = mlm.powerCurve(model, data, along, breaks, nsim=100)
```

**Parameters:**

- `model`: Fitted model
- `data`: Original data frame
- `along`: What to vary. Options: grouping factor name (e.g., `"Subject"`) or `"n"` for observations per group
- `breaks`: List of sample sizes to test
- `nsim`: Simulations per sample size

**Returns:** PowerCurveResult with:

- `breaks`: Sample sizes tested
- `powers`: Power at each sample size
- `plot()`: Method to visualize the curve

**Example:**

```python
curve = mlm.powerCurve(
    model, data,
    along="Subject",
    breaks=[10, 15, 20, 25, 30],
    nsim=100
)
print(curve)
curve.plot()
```

## Data Extension

### extend

Extend a dataset for power analysis.

```python
extended = mlm.extend(model, data, along, n)
```

**Parameters:**

- `model`: Fitted model (used to generate new observations)
- `data`: Original data frame
- `along`: What to extend. Options: grouping factor name or `"n"`
- `n`: Target sample size

**Returns:** Extended DataFrame

**Example:**

```python
# Original: 18 subjects
print(f"Original: {data['Subject'].nunique()} subjects")

# Extend to 30 subjects
extended = mlm.extend(model, data, along="Subject", n=30)
print(f"Extended: {extended['Subject'].nunique()} subjects")
```

## Result Classes

### PowerResult

Result from `powerSim()`.

**Attributes:**

- `power`: Estimated power (0 to 1)
- `ci_lower`: Lower bound of 95% CI
- `ci_upper`: Upper bound of 95% CI
- `nsim`: Total simulations
- `n_success`: Successful model fits
- `effect_size`: Effect size tested

### PowerCurveResult

Result from `powerCurve()`.

**Attributes:**

- `breaks`: Sample sizes tested
- `powers`: Power at each size

**Methods:**

- `plot()`: Plot the power curve

## Usage Examples

### Basic Power Analysis

```python
import mixedlm as mlm

# Fit model to pilot data
data = mlm.load_sleepstudy()
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

# Simulate power
power = mlm.powerSim(model, data, nsim=200)
print(power)
```

### Power Curve

```python
# See how power changes with number of subjects
curve = mlm.powerCurve(
    model, data,
    along="Subject",
    breaks=[10, 15, 18, 20, 25, 30],
    nsim=100
)

print(curve)
curve.plot()
```

### Finding Required Sample Size

```python
# Find minimum sample size for 80% power
target = 0.80

for n in [10, 15, 20, 25, 30, 35, 40]:
    extended = mlm.extend(model, data, along="Subject", n=n)
    p = mlm.powerSim(model, extended, nsim=100)
    print(f"n={n}: power={p.power:.1%}")
    if p.power >= target:
        print(f"Minimum n = {n}")
        break
```

### Extending Data

```python
# Original: 18 subjects
print(f"Original subjects: {data['Subject'].nunique()}")

# Extend to 30 subjects
extended = mlm.extend(model, data, along="Subject", n=30)
print(f"Extended subjects: {extended['Subject'].nunique()}")

# Extend observations per subject
extended2 = mlm.extend(model, data, along="n", n=20)
```

### Testing Different Effect Sizes

```python
import copy

effect_sizes = [5, 7.5, 10, 12.5, 15]
results = []

for effect in effect_sizes:
    # Modify effect size
    model_mod = copy.deepcopy(model)
    model_mod._fixef['Days'] = effect

    # Simulate power
    p = mlm.powerSim(model_mod, data, nsim=100)
    results.append({'effect': effect, 'power': p.power})
    print(f"Effect={effect}: power={p.power:.1%}")
```

### Power for GLMM

```python
# Binomial GLMM
cbpp = mlm.load_cbpp()
glmm = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    cbpp,
    family=mlm.families.Binomial()
)

# Power analysis
power = mlm.powerSim(glmm, cbpp, nsim=100, test="fixed")
print(power)
```

## Interpreting Results

### Power Estimate Precision

The confidence interval depends on the number of simulations:

| nsim | Approximate CI width |
|------|---------------------|
| 100 | ±10% |
| 200 | ±7% |
| 500 | ±4% |
| 1000 | ±3% |

Use more simulations for final estimates.

### Convergence Failures

Some simulations may fail to converge:

```python
power = mlm.powerSim(model, data, nsim=100)
print(f"Successful fits: {power.n_success}/{power.nsim}")
```

High failure rates (>10%) suggest model issues.

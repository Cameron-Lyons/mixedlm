# Formula Syntax

This page provides a complete reference for the formula syntax used in mixedlm.

## Overview

mixedlm uses lme4-style formulas to specify mixed models:

```python
"response ~ fixed_effects + (random_effects | grouping_factor)"
```

The formula has two parts:

1. **Fixed effects**: Standard regression terms
2. **Random effects**: Enclosed in parentheses with `|` separating effects from grouping factor

## Fixed Effects

### Basic Terms

```python
# Intercept only
"y ~ 1"

# Single predictor
"y ~ x"

# Multiple predictors
"y ~ x1 + x2 + x3"
```

### Interactions

```python
# Interaction only
"y ~ x1:x2"

# Main effects and interaction
"y ~ x1 + x2 + x1:x2"

# Shorthand for above
"y ~ x1 * x2"
```

### Polynomial Terms

```python
# Quadratic
"y ~ x + I(x**2)"

# Using poly() equivalent
"y ~ x + I(x**2) + I(x**3)"
```

### Removing the Intercept

```python
# No intercept
"y ~ 0 + x"
"y ~ x - 1"
```

### Categorical Variables

Categorical variables are automatically dummy-coded:

```python
# Treatment with 3 levels -> 2 dummy variables
"y ~ treatment"
```

## Random Effects

### Random Intercept

Allow the intercept to vary by group:

```python
"y ~ x + (1 | group)"
```

Model: \(y_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \epsilon_{ij}\)

where \(u_j \sim N(0, \sigma^2_u)\)

### Random Slope

Allow a predictor's effect to vary by group:

```python
# Random slope only (rarely used alone)
"y ~ x + (0 + x | group)"
```

### Random Intercept and Slope (Correlated)

The most common specification:

```python
"y ~ x + (x | group)"
# Equivalent to:
"y ~ x + (1 + x | group)"
```

This estimates:

- Variance of random intercepts: \(\sigma^2_{u_0}\)
- Variance of random slopes: \(\sigma^2_{u_1}\)
- Correlation between intercepts and slopes: \(\rho\)

### Random Intercept and Slope (Uncorrelated)

Force random effects to be independent:

```python
"y ~ x + (x || group)"
```

Uses `||` instead of `|`. This estimates:

- Variance of random intercepts: \(\sigma^2_{u_0}\)
- Variance of random slopes: \(\sigma^2_{u_1}\)
- No correlation parameter

**When to use:**

- Simpler model (fewer parameters)
- Correlation not of interest
- Full model has convergence issues

### Multiple Random Slopes

```python
"y ~ x1 + x2 + (x1 + x2 | group)"
```

Estimates a 3×3 covariance matrix for (intercept, x1-slope, x2-slope).

## Nested Random Effects

When groups are nested (e.g., students within schools):

```python
# Explicit nesting
"y ~ x + (1 | school) + (1 | school:class)"

# Shorthand using /
"y ~ x + (1 | school/class)"
```

The shorthand `school/class` expands to `school + school:class`.

### Example

```python
import mixedlm as mlm

# Students nested in classrooms nested in schools
model = mlm.lmer(
    "score ~ treatment + (1 | school/classroom)",
    data
)
```

This fits:

- Random intercept for school
- Random intercept for classroom within school

## Crossed Random Effects

When grouping factors are crossed (not nested):

```python
# Items crossed with subjects
"y ~ x + (1 | subject) + (1 | item)"
```

### Example

```python
# Subjects rate multiple items
# Each item rated by multiple subjects
model = mlm.lmer(
    "rating ~ condition + (1 | subject) + (1 | item)",
    data
)
```

## Complex Random Effects

### Different Effects for Different Factors

```python
# Random intercept for subject, random slope for item
"y ~ x + (x | subject) + (1 | item)"
```

### Combining Nested and Crossed

```python
# Students in schools, crossed with items
"y ~ x + (1 | school/student) + (1 | item)"
```

## Summary Table

| Pattern | Description |
|---------|-------------|
| `(1 \| g)` | Random intercept for g |
| `(0 + x \| g)` | Random slope for x (no intercept) |
| `(x \| g)` | Random intercept and slope, correlated |
| `(x \|\| g)` | Random intercept and slope, uncorrelated |
| `(x1 + x2 \| g)` | Multiple random slopes, all correlated |
| `(x1 \|\| g) + (x2 \|\| g)` | Multiple random slopes, all uncorrelated |
| `(1 \| g1/g2)` | Nested: g2 within g1 |
| `(1 \| g1) + (1 \| g2)` | Crossed: g1 and g2 |
| `(1 \| g1) + (1 \| g1:g2)` | Explicit nesting |

## Practical Considerations

### Keep It Simple

Start with simple random effects and add complexity only if needed:

1. Random intercept only: `(1 | g)`
2. Add random slope: `(x | g)`
3. If convergence issues, try uncorrelated: `(x || g)`

### Maximal vs Parsimonious

**Maximal approach** (Barr et al., 2013): Include all random effects that the design supports.

**Parsimonious approach**: Start simple, add complexity if justified by data.

mixedlm recommendation: Start with the maximal model supported by your design. Simplify if you encounter convergence issues.

### Identifiability

Some random effects structures aren't identifiable:

```python
# Usually won't converge well
"y ~ (x | g1) + (x | g2)"  # Same slope varies by two factors
```

### Singular Fits

If variance is estimated at zero, the model is singular:

```python
model = mlm.lmer("y ~ x + (x | g)", data)
if model.is_singular():
    # Try simpler model
    model = mlm.lmer("y ~ x + (1 | g)", data)
```

## Formula Utilities

### Parsing Formulas

```python
from mixedlm.formula import parse_formula, findbars, nobars

formula = "y ~ x + (x | g)"

# Find random effects terms
bars = findbars(formula)
# ['(x | g)']

# Get fixed effects formula
fixed = nobars(formula)
# 'y ~ x'
```

### Checking Formula Type

```python
from mixedlm.formula import is_mixed_formula

is_mixed_formula("y ~ x + (1 | g)")  # True
is_mixed_formula("y ~ x")            # False
```

## Common Mistakes

### Forgetting the Intercept in Random Effects

```python
# This includes random intercept:
"y ~ x + (x | g)"

# This does NOT:
"y ~ x + (0 + x | g)"
```

### Confusing `|` and `||`

```python
# Correlated (estimates correlation parameter)
"y ~ x + (x | g)"

# Uncorrelated (no correlation parameter)
"y ~ x + (x || g)"
```

### Incorrect Nesting Syntax

```python
# Correct: classroom within school
"y ~ x + (1 | school/classroom)"

# Wrong: would cross them
"y ~ x + (1 | school) + (1 | classroom)"  # Only if truly crossed
```

### Too Complex for Data

If you have 5 groups, don't try to estimate:

```python
# 6 variance parameters with only 5 groups
"y ~ x + (x1 + x2 | g)"  # Won't work well
```

Rule of thumb: Need at least 5-10 groups per variance parameter.

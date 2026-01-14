# Datasets

This page documents the built-in datasets from lme4.

## Dataset Loaders

All loaders return pandas DataFrames by default.

### load_sleepstudy

Sleep deprivation study data.

```python
import mixedlm as mlm
data = mlm.load_sleepstudy()
```

**Description:** Reaction times in a sleep deprivation study. 18 subjects were restricted to 3 hours of sleep per night for 10 days. Reaction times were measured each day.

**Variables:**

| Variable | Description |
|----------|-------------|
| Reaction | Average reaction time (ms) |
| Days | Days of sleep deprivation (0-9) |
| Subject | Subject identifier |

**Size:** 180 observations, 18 subjects

**Example usage:**

```python
model = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)
```

### load_cbpp

Contagious bovine pleuropneumonia data.

```python
data = mlm.load_cbpp()
```

**Description:** Serological incidence of contagious bovine pleuropneumonia in Ethiopian herds.

**Variables:**

| Variable | Description |
|----------|-------------|
| herd | Herd identifier |
| incidence | Number of new cases |
| size | Herd size at beginning of period |
| period | Time period (4 levels) |

**Size:** 56 observations, 15 herds

**Example usage:**

```python
model = mlm.glmer(
    "incidence / size ~ period + (1 | herd)",
    data,
    family=mlm.families.Binomial()
)
```

### load_cake

Cake baking experiment data.

```python
data = mlm.load_cake()
```

**Description:** Data from a cake baking experiment. Three recipes and six baking temperatures.

**Variables:**

| Variable | Description |
|----------|-------------|
| replicate | Replicate number |
| recipe | Recipe (A, B, C) |
| temperature | Baking temperature |
| angle | Angle at which cake broke |

**Size:** 270 observations

**Example usage:**

```python
model = mlm.lmer("angle ~ recipe * temperature + (1 | replicate)", data)
```

### load_dyestuff

Dyestuff yield data.

```python
data = mlm.load_dyestuff()
```

**Description:** Yield of dyestuff from batches of an intermediate product.

**Variables:**

| Variable | Description |
|----------|-------------|
| Batch | Batch identifier (A-F) |
| Yield | Yield of dyestuff |

**Size:** 30 observations, 6 batches

**Example usage:**

```python
model = mlm.lmer("Yield ~ 1 + (1 | Batch)", data)
```

### load_dyestuff2

Second dyestuff data.

```python
data = mlm.load_dyestuff2()
```

**Description:** Similar to dyestuff but with lower between-batch variability.

**Size:** 30 observations, 6 batches

### load_penicillin

Penicillin assay data.

```python
data = mlm.load_penicillin()
```

**Description:** Penicillin potency assay using a plate microbiological assay.

**Variables:**

| Variable | Description |
|----------|-------------|
| diameter | Diameter of zone of inhibition |
| plate | Plate identifier |
| sample | Penicillin sample |

**Size:** 144 observations

**Example usage:**

```python
model = mlm.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data)
```

### load_pastes

Paste strength data.

```python
data = mlm.load_pastes()
```

**Description:** Strength of a chemical paste from a balanced incomplete block design.

**Variables:**

| Variable | Description |
|----------|-------------|
| strength | Paste strength |
| batch | Batch identifier |
| sample | Sample within batch |

**Size:** 60 observations

**Example usage:**

```python
model = mlm.lmer("strength ~ 1 + (1 | batch/sample)", data)
```

### load_insteval

Instructor evaluations data.

```python
data = mlm.load_insteval()
```

**Description:** University instructor evaluations by students.

**Variables:**

| Variable | Description |
|----------|-------------|
| s | Student identifier |
| d | Instructor identifier |
| dept | Department |
| service | Service course (0/1) |
| lectage | Lecturer age category |
| studage | Student age category |
| y | Evaluation score |

**Size:** 73,421 observations

**Example usage:**

```python
model = mlm.lmer(
    "y ~ service + lectage + studage + (1 | s) + (1 | d) + (1 | dept:service)",
    data
)
```

### load_arabidopsis

Arabidopsis clipping experiment data.

```python
data = mlm.load_arabidopsis()
```

**Description:** Data from an experiment on Arabidopsis plants with clipping treatments.

**Variables:**

| Variable | Description |
|----------|-------------|
| reg | Region |
| poession | Position |
| gen | Genotype |
| rack | Rack |
| nutrient | Nutrient treatment |
| aession | Assessment |
| status | Status |
| total.fruits | Total number of fruits |

**Example usage:**

```python
model = mlm.glmer(
    "total.fruits ~ nutrient * gen + (1 | reg) + (1 | rack)",
    data,
    family=mlm.families.Poisson()
)
```

### load_grouseticks

Grouse tick data.

```python
data = mlm.load_grouseticks()
```

**Description:** Tick counts on red grouse chicks.

**Variables:**

| Variable | Description |
|----------|-------------|
| TICKS | Number of ticks |
| BROOD | Brood identifier |
| ALTITUDE | Altitude |
| YEAR | Year |
| HEIGHT | Chick height |

**Example usage:**

```python
model = mlm.glmer(
    "TICKS ~ ALTITUDE + HEIGHT + (1 | BROOD) + (1 | YEAR)",
    data,
    family=mlm.families.Poisson()
)
```

### load_verbagg

Verbal aggression data.

```python
data = mlm.load_verbagg()
```

**Description:** Verbal aggression item responses.

**Variables:**

| Variable | Description |
|----------|-------------|
| r2 | Binary response |
| Anger | Anger score |
| Gender | Gender |
| btype | Behavior type |
| situ | Situation |
| mode | Mode |
| item | Item identifier |
| id | Subject identifier |

**Example usage:**

```python
model = mlm.glmer(
    "r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item)",
    data,
    family=mlm.families.Binomial()
)
```

## Common Patterns

### Loading Datasets

```python
import mixedlm as mlm

# All loaders work the same way
sleepstudy = mlm.load_sleepstudy()
cbpp = mlm.load_cbpp()
cake = mlm.load_cake()
```

### Dataset Information

```python
data = mlm.load_sleepstudy()

# Basic info
print(data.shape)
print(data.columns.tolist())
print(data.head())

# Summary statistics
print(data.describe())

# Grouping structure
print(f"Subjects: {data['Subject'].nunique()}")
print(f"Obs per subject: {data.groupby('Subject').size().mean()}")
```

# Datasets

These loaders return the original lme4 tables bundled with mixedlm. They work
offline and preserve every original observation, numeric value, factor label,
and row order. The immutable [source commit](https://github.com/lme4/lme4/tree/67d71b0e264bda95f22bdc3ec52261c5fc993d4a/data),
source file checksums, and CSV checksums are recorded in the bundled
`mixedlm/datasets/data/provenance.json`. Upstream dataset attribution and license
information accompany the assets in `data/README.md` and `data/LICENSE`.

## Dataset Loaders

All loaders return independent pandas DataFrames. Modifying a loaded table does
not affect later calls. R factor labels are represented as ordinary Python
strings; numeric columns retain their original values. The only supplemental
columns are the documented `total_fruits` and `cTICKS` compatibility aliases.

### load_sleepstudy

Sleep deprivation study data.

```python
import mixedlm as mlm
data = mlm.load_sleepstudy()
```

**Description:** Reaction times for 18 participants in the group restricted to
3 hours in bed. Days 0–1 are adaptation and training, day 2 is baseline, and sleep
restriction starts after day 2. The table contains the first ten study days.
[lme4 study documentation](https://lme4.github.io/lme4/reference/sleepstudy.html)

**Variables:**

| Variable | Description |
|----------|-------------|
| Reaction | Average reaction time (ms) |
| Days | Study day (0-9) |
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

**Description:** A split-plot experiment with three recipes and six baking
temperatures. Recipes are whole units, and temperatures are subunits within
replicates. The table includes both string temperature labels and numeric
temperatures.

**Variables:**

| Variable | Description |
|----------|-------------|
| replicate | Replicate number |
| recipe | Recipe (A, B, C) |
| temperature | Baking temperature |
| angle | Angle at which cake broke |
| temp | Numeric baking temperature in degrees Fahrenheit |

**Size:** 270 observations

**Example usage:**

```python
model = mlm.lmer("angle ~ recipe * temperature + (1 | recipe:replicate)", data)
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

**Description:** Two strength assays for each of three casks within ten delivery
batches. `sample` identifies each batch/cask combination uniquely.

**Variables:**

| Variable | Description |
|----------|-------------|
| strength | Paste strength |
| batch | Batch identifier |
| cask | Cask within batch (a, b, c) |
| sample | Unique batch/cask identifier (A:a through J:c) |

**Size:** 60 observations

**Example usage:**

```python
model = mlm.lmer("strength ~ 1 + (1 | batch/cask)", data)
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

**Size:** 73,421 original observations, 2,972 students and 1,128 instructors.
For a smaller example, select rows explicitly with `mlm.load_insteval().head(1000)`.
Student age labels are semesters 2, 4, 6, and 8; lecture age labels are semesters
1–6. [lme4 dataset documentation](https://lme4.github.io/lme4/reference/InstEval.html)

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
| popu | Population within region |
| gen | Genotype |
| rack | Rack |
| nutrient | Nutrient treatment |
| amd | Clipping treatment (clipped/unclipped) |
| status | Germination method (Normal/Petri.Plate/Transplant) |
| total.fruits | Total number of fruits |
| total_fruits | Identical compatibility alias of total.fruits |

**Size:** 625 original observations.

**Example usage:**

```python
model = mlm.glmer(
    "total_fruits ~ nutrient * amd + (1 | gen) + (1 | rack)",
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
| INDEX | Chick identifier |
| YEAR | Year |
| HEIGHT | Height above sea level in meters |
| LOCATION | Geographic location |
| cHEIGHT | Centered height |
| cTICKS | Identical compatibility alias of TICKS |

**Size:** 403 original observations in 118 broods.

**Example usage:**

```python
model = mlm.glmer(
    "TICKS ~ YEAR + HEIGHT + (1 | BROOD) + (1 | LOCATION)",
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
| r2 | Dichotomous response labels N/Y |
| resp | Response labels no/perhaps/yes |
| Anger | Anger score |
| Gender | Gender |
| btype | Behavior type |
| situ | Situation |
| mode | Mode |
| item | Item identifier |
| id | Subject identifier |

**Size:** 7,584 original responses, 316 subjects and 24 items. The binomial
response `r2` preserves its N/Y labels; Y is the success level.

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

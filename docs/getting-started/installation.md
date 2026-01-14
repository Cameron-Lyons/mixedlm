# Installation

## Requirements

- Python 3.10 or later
- NumPy >= 1.21
- SciPy >= 1.8
- One of: pandas >= 1.4 or polars >= 0.20

## Basic Installation

Install mixedlm from PyPI:

```bash
pip install mixedlm
```

This installs the core package. You'll also need either pandas or polars for DataFrame support:

=== "pandas"

    ```bash
    pip install mixedlm pandas
    ```

=== "polars"

    ```bash
    pip install mixedlm polars
    ```

=== "Both"

    ```bash
    pip install mixedlm pandas polars
    ```

## Optional Dependencies

### Plotting

For diagnostic plots and profile likelihood visualization:

```bash
pip install mixedlm[plots]
```

This installs matplotlib >= 3.5.

### Additional Optimizers

For access to BOBYQA, NEWUOA, and other optimization algorithms:

```bash
pip install mixedlm[optimizers]
```

This installs Py-BOBYQA and nlopt.

### All Optional Dependencies

```bash
pip install mixedlm[plots,optimizers]
```

## Installing from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/cameronlyons/mixedlm.git
cd mixedlm
pip install -e ".[dev]"
```

Building from source requires:

- Rust toolchain (for the Rust backend)
- maturin >= 1.4

The Rust components are automatically compiled during installation.

## Verifying Installation

```python
import mixedlm as mlm

# Check version
print(mlm.__version__)

# Quick test
data = mlm.load_sleepstudy()
result = mlm.lmer("Reaction ~ Days + (1 | Subject)", data)
print(result.fixef())
```

## Troubleshooting

### ImportError: No module named 'mixedlm._rust'

The Rust extension failed to build. Ensure you have the Rust toolchain installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then reinstall mixedlm.

### No DataFrame backend available

You need either pandas or polars installed:

```bash
pip install pandas  # or polars
```

### Optimizer not available

Some optimizers require optional dependencies:

```python
# Check available optimizers
from mixedlm.estimation.optimizers import AVAILABLE_OPTIMIZERS
print(AVAILABLE_OPTIMIZERS)
```

Install additional optimizers with `pip install mixedlm[optimizers]`.

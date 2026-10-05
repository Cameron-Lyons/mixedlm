# Installation

## Requirements

- Python 3.10 or later
- NumPy >= 1.23.5
- SciPy >= 1.14
- pandas >= 1.4

## Basic Installation

Install mixedlm from PyPI:

```bash
pip install mixedlm
```

This installs the core package with pandas support. Polars support is optional:

=== "Core"

    ```bash
    pip install mixedlm
    ```

=== "Polars"

    ```bash
    pip install mixedlm[polars]
    ```

## Optional Dependencies

### Plotting

For diagnostic plots and profile likelihood visualization:

```bash
pip install mixedlm[plots]
```

This installs matplotlib >= 3.5.

### Additional Optimizers

The core package includes SciPy's optimizers. Install the optimizer extra for
the NLopt algorithms, such as `nloptwrap_BOBYQA`, `nloptwrap_NEWUOA`, and
`nloptwrap_SBPLX`:

```bash
pip install mixedlm[optimizers]
```

This installs nlopt.

### All Optional Dependencies

```bash
pip install mixedlm[plots,optimizers]
```

## Installing from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/Cameron-Lyons/mixedlm.git
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

### Polars support not available

Pandas is installed with the core package. Install the polars extra if you want to pass polars
DataFrames directly:

```bash
pip install mixedlm[polars]
```

### Optimizer not available

Some optimizers require optional dependencies:

```python
# Check available optimizers
from mixedlm.estimation import available_optimizers
print(available_optimizers())
```

Install additional optimizers with `pip install mixedlm[optimizers]`. The list
contains solver names; `"auto"`, the default `lmer()` optimizer, is a fitting
policy and is always available.

### Parallel calls fail in a script

Functions that accept `n_jobs`, such as `bootMer()`, `allFit()`, and
`cross_validate()`, start worker processes without forking. Each worker imports
the calling script again, so a script that calls them at module level with
`n_jobs` greater than one fails: workers report `RuntimeError: An attempt has
been made to start a new process before the current process has finished its
bootstrapping phase`, and the call raises `BrokenProcessPool`. Move the work
under an `if __name__ == "__main__":` guard; see
[parallel execution](../api/inference.md#parallel-execution).

# Contributing

Thank you for your interest in contributing to mixedlm!

## Development Setup

### Prerequisites

- Python 3.10 or later
- Rust toolchain (for building the Rust backend)
- Git

### Setting Up the Development Environment

1. Clone the repository:

```bash
git clone https://github.com/cameronlyons/mixedlm.git
cd mixedlm
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode with test and optional-feature dependencies:

```bash
pip install -e ".[dev,plots,optimizers,docs]"
```

To reproduce the locked CI dependency versions with uv, use:

```bash
uv sync --locked --extra plots --extra optimizers --extra docs
```

The plotting and optimizer extras make their behavioral tests run instead of
skipping because an optional package is absent.

4. Install pre-commit hooks (optional but recommended):

```bash
pip install pre-commit
pre-commit install
```

## Running Tests

Run the test suite with pytest:

```bash
pytest --ignore=tests/test_benchmark.py --strict-config --strict-markers
```

Run with coverage:

```bash
pytest --ignore=tests/test_benchmark.py --cov=mixedlm --cov-branch --cov-report=html
```

Run specific tests:

```bash
pytest tests/test_lmer_fits.py
pytest tests/test_lmer_fits.py::TestLmer::test_random_intercept_model
```

### Performance Benchmarks

Run performance measurements with the benchmark fixture and save the results:

```bash
pytest tests/test_benchmark.py --benchmark-only \
  --benchmark-save=local --benchmark-storage=benchmark-results
```

The symbolic Cholesky cache has paired cached and uncached benchmarks for repeated
factorizations of banded and irregular sparse matrices, with one or sixteen
right-hand sides:

```bash
pytest tests/test_benchmark.py -k sparse_symbolic_repeated \
  --benchmark-only --benchmark-save=symbolic-cache \
  --benchmark-storage=benchmark-results
```

Both cases use the same matrix values and solve work; the uncached case also
repeats symbolic analysis. Input preparation and numerical validation happen
outside the timed region. Compare repeated timing statistics on the same machine
under similar load. Correctness tests check solutions and log-determinants
independently of elapsed time, including reuse of earlier numeric factors after
later factorizations.

Saved results contain summary statistics. Add `--benchmark-save-data` only when
individual timing samples are needed; collecting every sample across the full
benchmark suite can produce very large artifacts.

### Rebuilding the Native Backend

After editing Rust code, rebuild the backend before running tests:

```bash
python -m pip install -e ".[dev]"
python tools/native_build.py
```

The test suite checks that the loaded native library matches the checkout's
Rust sources. The check covers `src/**/*.rs`, `Cargo.toml`, `Cargo.lock`, and
`build.rs` using their contents, so it detects changes even when file timestamps
are preserved. Test collection and tests without an available native backend
keep their existing behavior. Normal package imports and model fitting do not
run this development check.

If the check reports an older or mismatched library, rebuild from the checkout
you intend to test, using the active virtual environment:

```bash
cargo clean --release --package mixedlm
python -m pip install -e ".[dev]"
python tools/native_build.py
```

Use the same `CARGO_TARGET_DIR` as the build if you configured one. Sharing a
target directory between worktrees can reuse an older library even when Cargo
reports a successful build. Cleaning just the `mixedlm` package keeps dependency
artifacts available. The embedded checksum identifies source contents; it is
not a cryptographic signature or a record of compiler flags.

### CI and Distribution Checks

CI uses `uv sync --locked --no-install-project` to install dependencies, builds
the native backend explicitly, and runs tools with `uv run --no-sync`. This
prevents a test command from quietly rebuilding or switching the backend under
test. The native-source check is required before the Python suites execute.
Python 3.12 exercises plotting and nlopt alongside the core suite, and standard
Python jobs cover Polars. Free-threaded 3.14t also checks plotting and verifies
that native imports keep the GIL disabled. It omits Polars and its runtime because
compatible free-threaded wheels are unavailable. This job builds a wheel and
installs it with `uv pip install --no-deps` to preserve the locked environment
without reinstalling the development group. Property tests and benchmarks run
in dedicated jobs.
The Python 3.12 job enforces 87% combined line/branch coverage, based on the
measured complete feature suite. Other Python jobs report coverage without this
floor because they exercise different optional-feature combinations.

A separate Python 3.10 job runs the core suite with NumPy 1.23.5, SciPy 1.14.0,
and pandas 1.4.0. NumPy 1.23.5 is SciPy 1.14's effective lower bound. This job
downloads the normal abi3 wheel and installs it in an isolated environment, so
it checks an artifact built for modern Python against older NumPy and pandas
without an editable installation hiding compatibility issues. Test tools and
Polars retain their locked versions; plotting and nlopt are checked separately
by the complete feature run. The minimum job sets `OPENBLAS_CORETYPE=Nehalem`
to avoid a [known CPU-dispatch bug in NumPy 1.23.5's bundled OpenBLAS](https://github.com/numpy/numpy/issues/24903)
on newer x86 CPUs while retaining a kernel supported by the wheel's CPU baseline.

Each wheel is installed and exercised on its target operating system and CPU,
including Linux ARM. The source distribution is rebuilt and installed in a
fresh environment as well. `tools/check_wheel.py` verifies installed-package locations,
metadata, packaged datasets, LMM and grouped-binomial fits, sparse solves and
log-determinants against NumPy, and concurrent use of a shared native factor.
Before those numerical checks, `tools/native_build.py` compares each installed
wheel against its build inputs. Source-built wheels use the checkout; rebuilt
source distributions use the extracted archive, including maturin's normalized
Cargo manifest.
Run the same check after installing a wheel into a fresh virtual environment:

```bash
python -I tools/native_build.py
python -I tools/check_wheel.py
# For a free-threaded interpreter and its matching wheel:
python -I tools/check_wheel.py --expect-free-threaded
```

The isolated interpreter excludes checkout imports, and the script rejects
editable installations. Release jobs require these checks before uploading
artifacts for publication. Actionlint validates workflow structure and
expressions on every pull request.
The `Required CI checks` job aggregates every CI job and fails if any failed,
was cancelled, or was skipped, so branch protection can require one stable check.

## Code Style

This project uses:

- **ruff** for linting and formatting
- **mypy** for type checking

Run the linters:

```bash
ruff check python/ tests/ tools/
ruff format python/ tests/ tools/
mypy python/ tools/ --ignore-missing-imports
```

### Style Guidelines

- Follow PEP 8
- Use type hints for all public functions
- Write docstrings in NumPy format
- Keep lines under 100 characters

## Building Documentation

Build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

Build for production:

```bash
mkdocs build --strict
```

## Making Changes

### Workflow

1. Create a new branch for your changes:

```bash
git checkout -b feature/my-feature
```

2. Make your changes and write tests

3. Run the test suite and linters:

```bash
pytest
ruff check python/
mypy python/mixedlm/
```

4. Commit your changes with a descriptive message

5. Push and create a pull request

### Pull Request Guidelines

- Include tests for new functionality
- Update documentation if needed
- Keep PRs focused on a single change
- Write clear commit messages
- Ensure all CI checks pass

## Project Structure

```
mixedlm/
├── python/
│   └── mixedlm/
│       ├── models/         # Model fitting (lmer, glmer, nlmer)
│       ├── estimation/     # Optimization and estimation
│       ├── inference/      # Hypothesis testing, CIs
│       ├── families/       # Distribution families
│       ├── formula/        # Formula parsing
│       ├── matrices/       # Design matrices
│       ├── diagnostics/    # Model diagnostics
│       ├── nlme/           # Nonlinear models
│       ├── power/          # Power analysis
│       ├── datasets/       # Built-in datasets
│       └── utils/          # Utilities
├── src/                    # Rust source code
├── tests/                  # Test suite
├── docs/                   # Documentation
└── pyproject.toml          # Project configuration
```

## Adding New Features

### New Model Methods

1. Implement in appropriate module under `python/mixedlm/`
2. Add to `__all__` in the module's `__init__.py`
3. Export from main `__init__.py` if user-facing
4. Write tests in `tests/`
5. Add documentation

### New Dataset

1. Add data loading function to `datasets/lme4.py`
2. Include in `datasets/__init__.py`
3. Export from main `__init__.py`
4. Document in `docs/api/datasets.md`

## Reporting Issues

When reporting bugs, please include:

- Python version
- mixedlm version
- Minimal reproducible example
- Full error traceback
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions about contributing.

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

3. Install in development mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks (optional but recommended):

```bash
pip install pre-commit
pre-commit install
```

## Running Tests

Run the test suite with pytest:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=mixedlm --cov-report=html
```

Run specific tests:

```bash
pytest tests/test_lmer.py
pytest tests/test_lmer.py::test_random_intercept
```

### Performance Benchmarks

Run performance measurements with the benchmark fixture and save the results:

```bash
pytest tests/test_benchmark.py --benchmark-only --benchmark-json=benchmark.json
```

The symbolic Cholesky cache has paired cached and uncached benchmarks for repeated
factorizations of banded and irregular sparse matrices, with one or sixteen
right-hand sides:

```bash
pytest tests/test_benchmark.py -k sparse_symbolic_repeated \
  --benchmark-only --benchmark-json=symbolic-cache-benchmark.json
```

Both cases use the same matrix values and solve work; the uncached case also
repeats symbolic analysis. Input preparation and numerical validation happen
outside the timed region. Compare repeated timing statistics on the same machine
under similar load. Correctness tests check solutions and log-determinants
independently of elapsed time, including reuse of earlier numeric factors after
later factorizations.

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

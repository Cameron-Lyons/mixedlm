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

## Code Style

This project uses:

- **ruff** for linting and formatting
- **mypy** for type checking

Run the linters:

```bash
ruff check python/
ruff format python/
mypy python/mixedlm/
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

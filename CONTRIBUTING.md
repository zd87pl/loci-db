# Contributing to loci-db

Thank you for your interest in contributing to loci-db!

## Getting started

1. Fork the repository and clone your fork.
2. Install the development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite to confirm everything passes:

   ```bash
   pytest
   ```

## Submitting changes

- Open an issue before starting non-trivial work so we can discuss the approach.
- Create a branch from `main` with a descriptive name (e.g. `fix/cors-config`).
- Keep commits focused; one logical change per commit.
- Add or update tests for any new behaviour.
- Run `ruff check .` and `mypy loci/` before pushing — CI will enforce both.
- Open a pull request against `main`. Fill in the PR template and link the relevant issue.

## Code style

- Formatting and linting: [ruff](https://docs.astral.sh/ruff/) (`ruff check .` and `ruff format .`).
- Type annotations: required for all public functions and classes; checked with [mypy](https://mypy.readthedocs.io/).
- Line length: 100 characters.

## Running tests

```bash
# Unit + integration tests
pytest

# With coverage report
pytest --cov=loci --cov-report=term-missing
```

## Reporting bugs

Please open a [GitHub issue](https://github.com/zd87pl/loci-db/issues) and include:

- Python version and OS.
- Steps to reproduce.
- Expected vs. actual behaviour.
- Relevant log output or tracebacks.

## Security issues

Do **not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

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

## Where to start

The highest-impact contributions right now are **integrations with the tools
physical-AI builders already use**. Each one is self-contained and a great
first project:

- **ROS 2**: a node that turns odometry + camera embeddings into LOCI memories
  and exposes recall/novelty as services.
- **LeRobot**: a dataset loader and a policy-memory example.
- **NVIDIA Isaac Sim / Habitat**: an end-to-end example with real embeddings.
- **World-model adapters**: new models alongside `loci/adapters/` (V-JEPA 2,
  DreamerV3 and a generic adapter exist today).
- **Benchmarks**: the world-model memory benchmark described in
  [RFC-0001](docs/RFC-0001-memory-for-world-models.md) (R3).

Open an issue with the **Integration proposal** template first so we can agree
on the shape. Smaller fixes (docs, examples, bugs) are always welcome without
prior discussion.

To see LOCI working end to end before diving in, run the demo:

```bash
pip install -e . fastapi "uvicorn[standard]"
uvicorn demo.app.main:app   # http://localhost:8000
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

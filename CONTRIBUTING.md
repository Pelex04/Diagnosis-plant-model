# Contributing to PlantDx

Thank you for your interest in contributing. This document covers everything
you need to get from zero to a merged pull request.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Making Changes](#making-changes)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Code of Conduct

Be respectful. Constructive criticism of code is welcome; personal attacks are not.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Diagnosis-plant-model.git
   cd Diagnosis-plant-model
   ```
3. **Add upstream** remote so you can keep your fork in sync:
   ```bash
   git remote add upstream https://github.com/Pelex04/Diagnosis-plant-model.git
   ```

---

## Development Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch (CPU is fine for development)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

After this, `from plantdx import ...` works anywhere — no `sys.path` hacks needed.

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, protected. All changes come in via PR. |
| `feature/<name>` | New features or enhancements |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation-only changes |
| `chore/<name>` | Tooling, CI, dependency updates |

Always branch off `main`:

```bash
git fetch upstream
git checkout -b feature/my-feature upstream/main
```

---

## Making Changes

- **One concern per PR.** Don't mix a bug fix with a feature and a refactor.
- **Write or update tests** for any code you touch.
- **Update `CHANGELOG.md`** under the `[Unreleased]` section.
- **Update docstrings** if you change function signatures or behaviour.
- Keep commits atomic and write meaningful commit messages:
  ```
  fix: correct class index off-by-one in predict()

  The softmax output was indexed before applying top_k, which caused
  incorrect class name lookup when confidence_threshold filtered results.
  Fixes #42.
  ```

---

## Running Tests

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=src/plantdx --cov-report=term-missing

# Single test file
pytest tests/test_model.py -v

# Single test
pytest tests/test_model.py::TestBuildModel::test_output_shape -v
```

The CI pipeline runs the same commands on Python 3.10 and 3.11. Your PR will
not be merged if tests fail.

---

## Code Style

This project uses **ruff** for linting and formatting. Run it before committing:

```bash
# Check
ruff check src/ scripts/ tests/

# Auto-fix
ruff check --fix src/ scripts/ tests/
```

Key style rules (see `pyproject.toml` for full config):
- Line length: 100 characters
- Type annotations required on all public functions
- Docstrings required on all public classes and methods (Google style)
- No bare `except:` clauses
- No `print()` in library code — use `logging`

---

## Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```
2. Open a PR against `Pelex04/Diagnosis-plant-model:main`.
3. Fill in the PR template completely.
4. Wait for CI to pass (tests on 3.10 & 3.11, ruff lint).
5. Address any review comments — push additional commits to the same branch.
6. Once approved and CI is green, a maintainer will squash-merge.

---

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml).
Include:
- Your OS, Python version, PyTorch version, GPU model
- Minimal reproduction steps
- Full traceback

---

## Requesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml).
Describe the problem you're solving, not just the solution you have in mind.

---

## Questions?

Open a [Discussion](https://github.com/Pelex04/Diagnosis-plant-model/discussions)
rather than an issue for general questions.

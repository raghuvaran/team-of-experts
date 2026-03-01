# AGENTS.md

Instructions for AI coding agents working on this repository.

## Project Overview

TOXP (Team Of eXPerts) is a CLI tool that spawns multiple parallel LLM reasoning agents via AWS Bedrock, then synthesizes their outputs through a coordinator agent.

**Tech stack:** Python 3.10+, boto3, pydantic, rich

## Setup Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run CLI
uvx toxp "your question"
```

## Code Style

- Python with type hints (mypy strict)
- Line length: 100 characters
- Formatter: black
- Linter: ruff
- Docstrings for all public functions

## Testing

- Run `pytest` before committing
- Tests are in `tests/` directory
- Use `pytest-asyncio` for async tests
- Property-based tests use `hypothesis`

## Release Workflow

After merging bug fixes or features, **always bump the version**:

- **Patch** (0.x.Y): Bug fixes, security patches
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

Version is defined in TWO places (keep them in sync):
1. `pyproject.toml` → `version = "x.y.z"`
2. `toxp/__init__.py` → `__version__ = "x.y.z"`

## Git Conventions

Commit messages use conventional commits:
- `fix:` bug fixes
- `feat:` new features
- `chore:` maintenance (version bumps, deps)
- `docs:` documentation

## Pre-Push Verification

Before pushing commits, run the full test suite including live Bedrock smoke tests:

1. Run unit/integration tests: `pytest -m "not live" -q`
2. Read the configured AWS profile: `toxp config get aws-profile`
3. Run live smoke tests: `TOXP_LIVE_PROFILE=<profile> pytest -m live -v`

If live tests fail due to expired credentials, **do not push**. Ask the user to refresh credentials first.

## PR Workflow

1. Create feature branch from main
2. Make changes with conventional commit messages
3. Push branch and create PR
4. Squash merge to main
5. Bump version in both files
6. Commit: `git commit -m "chore: bump version to x.y.z"`
7. Push to main

## Architecture

```
toxp/
├── cli.py           # CLI entry point
├── orchestrator.py  # Coordinates parallel agents
├── providers/       # LLM providers (Bedrock)
├── agents/          # Reasoning and coordinator agents
├── models/          # Pydantic data models
└── utils/           # Rate limiting, helpers
```

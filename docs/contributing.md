> **Internal Use Only:** This document is intended for internal contributors and is not for external distribution.

# Contributing

Thank you for your interest in improving BasicRL.  This document describes how to set up a development environment and the expectations for testing changes.

## Environment setup

1. Clone the repository and switch into its directory.
2. Create a Python 3.11 virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   An `environment.yml` is also provided for Conda users.

## Development workflow

- Keep changes focused and well documented.
- When adding code or content, update relevant documentation in `docs/` or the README.
- Follow the existing code style and reuse helper functions where possible.

## Testing expectations

Run the unit test suite before submitting changes:

```bash
pytest
```

Tests cover core systems such as pathfinding, effects, perception, and inventory handling.  Adding new features should include accompanying tests when feasible.

Passing tests gives confidence that the engine and configurations still behave as expected.

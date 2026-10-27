# Contributing to fips

Thank you for considering contributing to fips! We welcome contributions from the community.

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science. If you
contribute with an agent, [AGENTS.md](AGENTS.md) at the repository root is
the orientation file it should read.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fips.git
   cd fips
   ```
3. Install system tools:
   - [uv](https://docs.astral.sh/uv/getting-started/installation/) - Environment and dependency management
   - [just](https://just.systems/) (the task runner) comes with the dev tools: `uv run just ...`

4. Install dependencies:
   ```bash
   # Using uv (recommended - faster):
   uv sync  # fips, its viz extra, and the dev tools

   # OR using pip:
   pip install --group dev -e .
   ```

5. Install pre-commit hooks:
   ```bash
   # If using uv:
   uv run pre-commit install

   # Or with activated venv/without uv:
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow our coding standards:
   - Code is formatted with ruff
   - All tests pass
   - New features include tests
   - Documentation is updated if needed

3. Run quality checks:
   ```bash
   just quality-check
   ```

4. Run test suite:
   ```bash
   just test
   # Or directly: uv run pytest
   ```
   This also runs the `>>>` examples in the docstrings under `src/`.

5. Run pre-commit checks:
   ```bash
   just pre-commit
   # Or directly: uv run pre-commit run --all-files
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "fix(matrix): keep labels when slicing"  # Conventional Commits
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request on GitHub

## Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Write clear, descriptive commit messages
- Leave `CHANGELOG.md` alone: the release notes are drafted from the commit
  messages when the maintainer releases, so write each message as the line a
  user would read
- Ensure all tests pass
- Maintain or improve test coverage
- Update documentation as needed

## Reporting Bugs

When reporting bugs, please include:
- Your operating system and Python version
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any error messages or logs

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain why it would be useful
- Consider submitting a pull request to implement it

## Questions?

If you have questions, please:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to the maintainers

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Releasing

The version comes from git tags (setuptools-scm), so there is no version string
to bump.

1. Write the changelog: `just changelog` drafts the entries from the commit
   messages since the last release. Edit them into `CHANGELOG.md` under
   `## [Unreleased]`, then rename that heading to `## [X.Y.Z] - YYYY-MM-DD` and
   start a new empty `## [Unreleased]` above it. Commit (`chore(release): cut X.Y.Z`)
   and push to `main`.
2. Run `just release X.Y.Z`. It checks that the tree is clean, that `main` is in
   sync with GitHub, and that the version is newer than every existing tag,
   then pushes the tag `vX.Y.Z`.
3. The Publish workflow builds the tag, uploads it to PyPI and creates the
   GitHub Release from the CHANGELOG section; Zenodo archives it. The
   Documentation workflow publishes its docs as `X.Y.Z/` in the version
   dropdown.

## Dependency updates

Dependabot opens one pull request a month for each of GitHub Actions and
pre-commit hooks. Merge it when CI passes.

`uv.lock` (the dev tools and every dependency) is updated by the Upgrade lock
workflow alone, not by Dependabot, which would leave a package at its old
version when the newest release needs a newer Python than the oldest one this
project supports. On the 1st of each month the workflow runs `uv lock --upgrade`
(`uv.lock` then holds the newest release for each Python; it never raises the
minimum versions in `pyproject.toml`) and, when the lock changes, opens the pull
request `chore/lock-upgrade` (or updates it). GitHub does not start workflows on
a pull request that its own token opened, so the workflow starts Tests, Code
Quality and Documentation on the branch itself; their results show on the pull
request. Merge it when they pass. You can also run it by hand: Actions > Upgrade
lock > Run workflow.

## Template

The tooling (CI workflows, pre-commit, justfile, packaging configuration) comes
from [jmineau/python-template](https://github.com/jmineau/python-template).
`.copier-answers.yml` records the template version; `copier update` pulls in
later template changes. Improvements that would help every package are best
made in the template.

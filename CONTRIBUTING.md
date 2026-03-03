# Contributing to fips

Thank you for considering contributing to fips! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fips.git
   cd fips
   ```
3. Install system tools:
   - [just](https://just.systems/man/en/prerequisites.html) - Task runner
   - [Pandoc](https://pandoc.org/installing.html) - For building documentation
   - [uv](https://docs.astral.sh/uv/getting-started/installation/) - Recommended for faster dependency management

4. Install dependencies:
   ```bash
   # Using uv (recommended - faster):
   uv sync --all-extras

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

5. Run pre-commit checks:
   ```bash
   just pre-commit
   # Or directly: uv run pre-commit run --all-files
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request on GitHub

## Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Write clear, descriptive commit messages
- Update the changelog if applicable
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

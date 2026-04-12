.PHONY: setup lint typecheck format test clean

## Install all dependencies and set up pre-commit hooks
setup:
	uv sync --extra dev --extra all
	uv run pre-commit install

## Run ruff linter
lint:
	uv run ruff check src/ tests/

## Run mypy strict type checking
typecheck:
	uv run mypy --strict src/

## Apply ruff formatter
format:
	uv run ruff format src/ tests/

## Run the full test suite with coverage
test:
	uv run pytest

## Remove build artifacts, caches, and compiled files
clean:
	rm -rf dist/ build/ .eggs/ *.egg-info/ .venv/
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.json htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

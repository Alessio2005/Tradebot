.PHONY: help install install-dev lint format type-check
.PHONY: test test-fast test-slow test-regression benchmark coverage clean

PYTHON  ?= python
PYTEST  ?= pytest
PIP     ?= pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-25s %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────────────────

install:  ## Install package (editable)
	$(PIP) install -e .

install-dev:  ## Install package + dev extras
	$(PIP) install -e ".[dev]"
	pre-commit install

# ── Code quality ──────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format with ruff
	ruff format src/ tests/

type-check:  ## Run mypy on schema + utils modules
	mypy src/tradebot/schemas/ src/tradebot/utils/ --ignore-missing-imports

# ── Testing ───────────────────────────────────────────────────────────────────

test-fast:  ## Run unit + property tests (no slow/regression)
	$(PYTEST) -m "not slow and not regression" -q

test-slow:  ## Run all tests including slow integration tests
	$(PYTEST) -m "not regression" -q

test-regression:  ## Run regression (bit-equivalence) tests
	$(PYTEST) -m "regression" -q

test:  ## Run all tests
	$(PYTEST) -q

benchmark:  ## Run benchmark tests
	$(PYTEST) -m "slow" tests/benchmark/ -q

coverage:  ## Run fast tests + coverage report
	$(PYTEST) -m "not slow and not regression" --cov=src/tradebot --cov-report=term-missing -q

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info src/*.egg-info

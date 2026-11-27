.PHONY: help install install-dev lint format type-check
.PHONY: test test-fast test-slow test-regression benchmark coverage clean
.PHONY: options-replay options-smoke

PYTHON  ?= python
PYTEST  ?= pytest
PIP     ?= pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-25s %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────────────────

install:  ## Install package (editable) uit de lockfile
	$(PIP) install -r requirements.lock
	$(PIP) install -e . --no-deps

install-dev:  ## Install package + testketen en linters uit de lockfiles
	$(PIP) install -r requirements.lock -r requirements-dev.lock
	$(PIP) install -e . --no-deps
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

# ── Options ───────────────────────────────────────────────────────────────────

REPLAY_DIR ?= /tmp/tradebot-replay

DOLT_CACHE ?= $(REPLAY_DIR)/dolthub-cache
DOLT_START ?= 2019-02-11
DOLT_END   ?= 2026-11-26

options-replay:  ## DoltHub-markt bouwen + replay (fictief geld) naar $(REPLAY_DIR)
	$(PYTHON) -I -m tradebot.options dolthub --cache $(DOLT_CACHE) --out $(REPLAY_DIR)/markt --start $(DOLT_START) --end $(DOLT_END)
	$(PYTHON) -I -m tradebot.options replay --market $(REPLAY_DIR)/markt --out $(REPLAY_DIR)/run --overwrite

options-smoke:  ## Replay op de kleine DoltHub-testmarkt in tests/fixtures (geen netwerk)
	$(PYTHON) -I -m tradebot.options replay --market tests/fixtures/options/dolthub_mini --out $(REPLAY_DIR)/smoke --overwrite

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info src/*.egg-info

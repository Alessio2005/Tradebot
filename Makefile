.PHONY: help install install-dev lint type-check test test-fast test-slow test-regression
.PHONY: benchmark clean doctor sync-data build-features tune train backtest tearsheet

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
	ruff check src/ apps/ tests/

format:  ## Auto-format with ruff
	ruff format src/ apps/ tests/

type-check:  ## Run mypy on schema + utils modules
	mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports

loc-check:  ## Check no file exceeds 800 LOC (whitelist: tightly-coupled algorithms)
	@# LOC whitelist: files where splitting would break algorithm cohesion.
	@# Each exception is documented in the file header with # LOC-EXCEPTION.
	@LOC_WHITELIST="train/ensemble.py labeling/meta.py risk/portfolio.py tune/objective.py"; \
	find src/ -name "*.py" | while read f; do \
	  lines=$$(wc -l < "$$f"); \
	  basename=$$(echo "$$f" | sed 's|src/tradebot/||'); \
	  exempt=0; \
	  for w in $$LOC_WHITELIST; do [ "$$basename" = "$$w" ] && exempt=1 && break; done; \
	  if [ "$$lines" -gt 800 ] && [ "$$exempt" -eq 0 ]; then \
	    echo "FAIL: $$f ($$lines LOC > 800 limit)"; exit 1; \
	  fi; \
	done && echo "LOC check passed."

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

# ── Diagnostics ───────────────────────────────────────────────────────────────

doctor:  ## Run environment diagnostics
	$(PYTHON) apps/doctor.py

# ── Pipeline entry-points ─────────────────────────────────────────────────────

sync-data:  ## Stage 0: sync raw OHLCV data
	tb-data-sync

build-features:  ## Stage 1: build features + labels
	tb-build-features

tune:  ## Stage 2: hyperparameter optimisation
	tb-tune-hparams

train:  ## Stage 3: CPCV training
	tb-train-cpcv

backtest:  ## Stage 4: portfolio backtest
	tb-backtest-portfolio

tearsheet:  ## Stage 4.5: generate tearsheet
	tb-make-tearsheet

# ── Maintenance ───────────────────────────────────────────────────────────────

regenerate-baselines:  ## Regenerate frozen regression test baselines
	$(PYTHON) apps/regenerate_baseline.py

monitor-drift:  ## Run nightly drift monitoring check
	tb-monitor-drift

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info src/*.egg-info

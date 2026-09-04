.PHONY: help install install-dev lint format type-check loc-check inventory
.PHONY: test test-fast test-slow test-regression benchmark coverage
.PHONY: doctor sync-data build-features tune train baseline reproduce
.PHONY: regenerate-baselines monitor-drift clean

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

loc-check:  ## Check no file exceeds 800 LOC (ratchet: per-file caps)
	@# De logica staat in scripts/check_file_size.py en NIET meer hier.
	@#
	@# Waarom: deze regel stond hier als shell-lus met een whitelist, en die
	@# bewaakte niets. `make` bestaat niet in elke omgeving waarin dit project
	@# draait -- gemeten op 2026-09-04: `make: command not found` -- dus de poort
	@# is nooit uitgevoerd. De whitelist noemde bovendien risk/portfolio.py, dat
	@# sinds Phase 4 niet bestaat. Een poort die zijn eigen scope niet kent en
	@# nooit draait, is documentatie.
	@#
	@# De vervanger is een ratchet: elk bestand boven 800 regels heeft een CAP op
	@# zijn gemeten omvang en mag niet groeien. Hij draait in CI
	@# (.github/workflows/inventory.yml) en wordt getoetst door
	@# tests/unit/test_file_size_ratchet.py, inclusief het bewijs dat hij rood
	@# kan worden.
	$(PYTHON) scripts/check_file_size.py

inventory:  ## Run the inventory gates (same set as CI)
	$(PYTHON) scripts/reachability_map.py --strict
	$(PYTHON) scripts/check_file_size.py
	$(PYTHON) scripts/check_hardcoded_params.py --strict
	$(PYTHON) scripts/audit_fallbacks.py --strict
	$(PYTHON) scripts/check_banned_methods.py --strict

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

baseline:  ## Stage 4: authoritative event-driven baseline (Phase 5)
	python apps/run_phase5_baseline.py

reproduce:  ## E-1: het volledige reproductiepad, zie docs/REPRODUCTION.md
	dvc repro

# ── Maintenance ───────────────────────────────────────────────────────────────

regenerate-baselines:  ## Regenerate frozen regression test baselines
	$(PYTHON) apps/regenerate_baseline.py

monitor-drift:  ## Run nightly drift monitoring check
	tb-monitor-drift

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info src/*.egg-info

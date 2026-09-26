"""test_imports.py — Smoke tests: every tradebot.* module imports cleanly.

If any module raises ImportError at collection time, the rest of the test
suite is moot.  This file is the first line of defence.

Run:
    pytest tests/test_imports.py -v
"""
from __future__ import annotations

import importlib

import pytest

# ── Canonical tradebot.* package modules ─────────────────────────────────────
TRADEBOT_MODULES = [
    "tradebot.schemas.bars",
    "tradebot.schemas.events",
    "tradebot.schemas.features",
    "tradebot.schemas.folds",
    "tradebot.schemas.labels",
    "tradebot.utils.arrays",
    "tradebot.utils.parquet_io",
    "tradebot.cv.cpcv",
    "tradebot.cv.uniqueness",
    "tradebot.cv.purge",
    "tradebot.labeling.cusum",
    "tradebot.labeling.trend_scanning",
    "tradebot.labeling.triple_barrier",
    "tradebot.features.blocks",
    "tradebot.features.scaling",
    "tradebot.features.cfi",
    "tradebot.features.pipeline",
    "tradebot.execution.spread",
    "tradebot.data.ingestion",
    "tradebot.data.macro",
    "tradebot.backtest._kernels",
    "tradebot.tune.search_space",
    "tradebot.tune.pruning",
    "tradebot.tune.objective",
]


@pytest.mark.parametrize("modname", TRADEBOT_MODULES)
def test_tradebot_module_imports(modname: str) -> None:
    """Each tradebot.* module imports without errors."""
    importlib.import_module(modname)


def test_the_legacy_backtest_engines_are_gone() -> None:
    """Phase 5: exact een authoritative engine (audit sectie 16.1, sectie 24).

    `apps/backtest_portfolio.py` was de DVC Stage-4 entrypoint en dreef
    `bidirectional_backtest` plus `PortfolioBacktester`. Beide zijn verwijderd
    na het pariteitsbewijs in
    `tests/integration/test_engine_parity.py::TestExecutionTimingParity`; de
    forensische vergelijking staat in `reports/phase5_engine_diff.md`.

    Deze test vervangt de oude import-test. Hij bewaakt de verwijdering in
    plaats van de aanwezigheid: een module die terugkomt, komt terug met zijn
    eigen leverage-caps.
    """
    for name in (
        "tradebot.backtest.per_side",
        "tradebot.backtest.bidirectional",
        "tradebot.backtest.portfolio",
        "tradebot.backtest.tracks",
        "tradebot.schemas.tracks",
        "tradebot.schemas.portfolio",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_the_authoritative_engine_imports() -> None:
    """Stage 4 is vanaf Phase 5 `backtest.engine`."""
    module = importlib.import_module("tradebot.backtest.engine")
    assert hasattr(module, "EventDrivenEngine")
    importlib.import_module("tradebot.backtest.accounting")
    importlib.import_module("tradebot.backtest.vectorized")
    importlib.import_module("tradebot.execution.order_router")
    importlib.import_module("tradebot.execution.context")

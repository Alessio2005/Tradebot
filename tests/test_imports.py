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
    "tradebot.schemas.tracks",
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
    "tradebot.data.funding",
    "tradebot.data.ingestion",
    "tradebot.data.macro",
    "tradebot.backtest._kernels",
    "tradebot.backtest.per_side",
    "tradebot.backtest.bidirectional",
    "tradebot.tune.search_space",
    "tradebot.tune.pruning",
    "tradebot.tune.objective",
]


@pytest.mark.parametrize("modname", TRADEBOT_MODULES)
def test_tradebot_module_imports(modname: str) -> None:
    """Each tradebot.* module imports without errors."""
    importlib.import_module(modname)


def test_apps_build_features_imports() -> None:
    """Stage 1 entrypoint imports."""
    importlib.import_module("apps.build_features")


def test_apps_tune_hparams_imports() -> None:
    """Stage 2 entrypoint imports."""
    importlib.import_module("apps.tune_hparams")


def test_apps_train_cpcv_imports() -> None:
    """Stage 3 entrypoint imports."""
    importlib.import_module("apps.train_cpcv")


def test_apps_backtest_portfolio_imports() -> None:
    """Stage 4 entrypoint imports."""
    importlib.import_module("apps.backtest_portfolio")

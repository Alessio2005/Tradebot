# src/tradebot/backtest/__init__.py
"""Backtest sub-package — L10.

PHASE 5 CONSOLIDATIE (audit §16.1, §24). Er is vanaf deze fase exact EEN
authoritative backtester:

    `engine.EventDrivenEngine`   market -> signal -> sovereign risk -> order
                                 -> fill -> accounting

Daarnaast staat `vectorized.run_vectorized` als research-engine, en elke output
daarvan draagt `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`.

WAT HIER WEG IS, EN WAAROM
--------------------------
`per_side.py`, `bidirectional.py`, `portfolio.py` en hun `tracks.py` zijn
verwijderd na het pariteitsbewijs in `tests/integration/test_engine_parity.py`.
De forensische vergelijking staat in `reports/phase5_engine_diff.md`; de
samenvatting is dat geen van drieen een kasregister had, geen sluitende balans,
geen soevereine risicobeslissing en geen order-object, en dat ze elk hun eigen
leverage- en gross-caps meebrachten die ruimer stonden dan `conf/risk/`.

`reporting.py` en `reporting/tearsheet.py` gingen mee omdat zij uitsluitend
`PortfolioBacktestResult` consumeerden.

`metrics.py` BLIJFT. Audit §24 merkt `backtest/metrics.py::DSR` aan als RETAIN;
het is statistiek, geen engine. `evaluation.py` blijft eveneens: het is een
hulpbibliotheek (DSR, block bootstrap, MTM) en geen backtester, zoals
`reports/phase5_engine_diff.md` §0 vaststelt.
"""
from __future__ import annotations

from .accounting import AccountingError, Fill, Ledger, LedgerSnapshot, Position
from .engine import BacktestResult, EventDrivenEngine, build_slices, exposures_from_frame
from .evaluation import (
    ASSET_GAP_MULTIPLES,
    CollisionResolution,
    KellySizingResult,
    MTMResult,
    block_bootstrap_path_sharpes,
    compute_bar_by_bar_mtm,
    compute_timeout_fraction,
    count_git_commits,
    deflated_sharpe_penalty,
    gap_risk_kelly_size,
    get_gap_multiple_for_asset,
    institutional_evaluation_report,
    resolve_long_short_collision,
    stationary_block_bootstrap_indices,
)
from .metrics import bootstrap_ci, calmar_ratio, deflated_sharpe, max_drawdown, sharpe_ratio
from .vectorized import (
    NOT_ADMISSIBLE,
    VectorizedResult,
    is_vectorized_evidence,
    reject_vectorized_evidence,
    run_vectorized,
)

__all__ = [
    # de authoritative engine
    "BacktestResult",
    "EventDrivenEngine",
    "build_slices",
    "exposures_from_frame",
    # boekhouding
    "AccountingError",
    "Fill",
    "Ledger",
    "LedgerSnapshot",
    "Position",
    # research-only, nooit promotiebewijs
    "NOT_ADMISSIBLE",
    "VectorizedResult",
    "is_vectorized_evidence",
    "reject_vectorized_evidence",
    "run_vectorized",
    # statistiek
    "sharpe_ratio",
    "calmar_ratio",
    "max_drawdown",
    "deflated_sharpe",
    "bootstrap_ci",
    # evaluation-hulpbibliotheek
    "count_git_commits",
    "deflated_sharpe_penalty",
    "KellySizingResult",
    "gap_risk_kelly_size",
    "get_gap_multiple_for_asset",
    "ASSET_GAP_MULTIPLES",
    "stationary_block_bootstrap_indices",
    "block_bootstrap_path_sharpes",
    "compute_timeout_fraction",
    "CollisionResolution",
    "resolve_long_short_collision",
    "MTMResult",
    "compute_bar_by_bar_mtm",
    "institutional_evaluation_report",
]

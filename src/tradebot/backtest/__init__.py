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

from typing import TYPE_CHECKING, Any

from .accounting import AccountingError, Fill, Ledger, LedgerSnapshot, Position
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

if TYPE_CHECKING:                       # pragma: no cover - typing only
    from .engine import (
        BacktestResult,
        EventDrivenEngine,
        build_slices,
        exposures_from_frame,
    )

#: De vier namen uit `.engine` die LUI worden geladen. Zie `__getattr__`.
_LAZY_ENGINE_NAMES = frozenset({
    "BacktestResult", "EventDrivenEngine", "build_slices", "exposures_from_frame",
})


def __getattr__(name: str) -> Any:
    """Laad `.engine` pas bij eerste gebruik (PEP 562).

    WAAROM DIT GEEN STIJLKEUZE IS — gevonden in Phase 7/8 Stage B-2.
    ---------------------------------------------------------------
    Er stond een importcyclus in het soevereine executiepad, en hij sloeg alleen
    toe bij een bepaalde importvolgorde:

        tradebot.execution.order_router
          -> from ..backtest.accounting import Fill        (L9 importeert L10)
          -> initialiseert het PAKKET tradebot.backtest
          -> backtest/__init__.py: from .engine import ...
          -> engine.py: from ..execution.order_router import ExecutionReport
          -> order_router staat pas op regel 56 en heeft ExecutionReport nog niet
          -> ImportError

    `import tradebot.backtest` werkte; `import tradebot.execution.order_router`
    als EERSTE tradebot-import crashte. De volledige testsuite liep daardoor
    groen — daar importeert altijd wel iets `tradebot.backtest` eerder — en
    `pytest tests/lookahead` crashte bij collectie. Precies de deelverzameling
    die `research_gates.yml` moet draaien.

    Het pakket-`__init__` was de enige eager schakel in de cyclus, en dus de
    goedkoopste plek om hem te breken. De publieke API verandert niet:
    `from tradebot.backtest import EventDrivenEngine` blijft werken, alleen
    later.

    De onderliggende laagfout blijft staan en is als DI-18 geregistreerd: L9
    (`execution/`) hoort niet uit L10 (`backtest/`) te importeren. `Fill` is een
    executie-primitief dat in de verkeerde laag woont. Dat verplaatsen raakt
    ~40 aanroepen en is een refactor, geen bijvangst van deze fase.
    """
    if name in _LAZY_ENGINE_NAMES:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

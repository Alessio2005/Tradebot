# src/tradebot/backtest/__init__.py
"""Backtest sub-package — portfolio backtester, performance metrics, reporting."""
from __future__ import annotations

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
from .portfolio import PortfolioBacktester
from .reporting import build_tearsheet_dict, save_tearsheet_json
from .tracks import AssetTrack, PortfolioBacktestResult

__all__ = [
    "AssetTrack",
    "PortfolioBacktestResult",
    "PortfolioBacktester",
    "sharpe_ratio",
    "calmar_ratio",
    "max_drawdown",
    "deflated_sharpe",
    "bootstrap_ci",
    "build_tearsheet_dict",
    "save_tearsheet_json",
    # evaluation
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

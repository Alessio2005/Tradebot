# src/tradebot/reporting/tearsheet.py
"""AFML-style tearsheet generator.

Builds a text + JSON summary of a backtest run.  Rendering to HTML/PDF
is delegated to apps/make_tearsheet.py which may use matplotlib/jinja2.

This module remains import-safe with no optional heavy dependencies.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..backtest.metrics import (
    annualized_return,
    annualized_vol,
    deflated_sharpe,
    max_drawdown,
    sortino_ratio,
)
from ..backtest.tracks import PortfolioBacktestResult

logger = logging.getLogger(__name__)

__all__ = ["build_tearsheet", "print_tearsheet", "save_tearsheet"]


def build_tearsheet(
    result: PortfolioBacktestResult,
    study_name: str = "backtest",
    bars_per_year: int = 365 * 24,
    n_trials: int = 1,
) -> dict[str, Any]:
    """Build a flat dict summarising the backtest result for reporting.

    Parameters
    ----------
    result : PortfolioBacktestResult from PortfolioBacktester.run().
    study_name : name of the study / run identifier.
    bars_per_year : calibration for annualisation.
    n_trials : number of strategy variants evaluated (for DSR).

    Returns
    -------
    dict with all scalar performance metrics + metadata.
    """
    r = result.portfolio_returns.values.astype(np.float64)
    eq = result.equity_curve.values.astype(np.float64)
    mdd_val, _, _ = max_drawdown(eq)

    dsr = deflated_sharpe(
        sr_observed=result.sharpe,
        n_trials=max(n_trials, 1),
        n_obs=max(len(r), 5),
    )

    return {
        "study_name": study_name,
        # Returns
        "annualized_return": round(annualized_return(r, bars_per_year=bars_per_year), 4),
        "annualized_vol": round(annualized_vol(r, bars_per_year=bars_per_year), 4),
        "sharpe": round(result.sharpe, 4),
        "sortino": round(sortino_ratio(r, bars_per_year=bars_per_year), 4),
        "calmar": round(result.calmar, 4),
        "max_drawdown": round(mdd_val, 4),
        "total_return": round(result.total_return, 4),
        "deflated_sharpe": round(dsr, 4),
        # Realised risk
        "realized_vol": round(result.realized_vol, 4),
        "avg_gross_leverage": round(result.avg_gross_leverage, 4),
        "avg_net_leverage": round(result.avg_net_leverage, 4),
        "avg_corr": round(result.avg_corr, 4),
        # Costs
        "total_rebalance_cost": round(result.total_rebalance_cost, 6),
        "total_funding_cost": round(result.total_funding_cost, 6),
        "avg_turnover": round(result.avg_turnover, 6),
        # Diagnostics
        "n_bars": result.n_bars,
        "n_dd_breaker_bars": result.n_dd_breaker_bars,
        # Bootstrap
        "bootstrap_sharpe_lo": round(result.bootstrap.get("sharpe_lo", float("nan")), 4),
        "bootstrap_sharpe_hi": round(result.bootstrap.get("sharpe_hi", float("nan")), 4),
        # Per-asset
        "per_asset_contribution": {k: round(v, 4) for k, v in result.per_asset_contribution.items()},
    }


def print_tearsheet(result: PortfolioBacktestResult, study_name: str = "backtest") -> None:
    """Print a concise tearsheet to stdout."""
    d = build_tearsheet(result, study_name=study_name)

    print(f"\n{'='*60}")
    print(f"  TEARSHEET — {d['study_name']}")
    print(f"{'='*60}")
    print(f"  Total Return    : {d['total_return']:>+9.2%}")
    print(f"  Ann. Return     : {d['annualized_return']:>+9.2%}")
    print(f"  Ann. Vol        : {d['annualized_vol']:>9.2%}")
    print(f"  Sharpe          : {d['sharpe']:>9.4f}")
    print(f"  Sortino         : {d['sortino']:>9.4f}")
    print(f"  Calmar          : {d['calmar']:>9.4f}")
    print(f"  Max Drawdown    : {d['max_drawdown']:>9.2%}")
    print(f"  Deflated Sharpe : {d['deflated_sharpe']:>9.4f}")
    print(f"{'─'*60}")
    print(f"  Avg Gross Lev   : {d['avg_gross_leverage']:>9.4f}")
    print(f"  Avg Corr        : {d['avg_corr']:>9.4f}")
    print(f"  DD Breaker Bars : {d['n_dd_breaker_bars']:>9d}  / {d['n_bars']}")
    print(f"  Rebalance Cost  : {d['total_rebalance_cost']:>9.4%}")
    print(f"  Funding Cost    : {d['total_funding_cost']:>9.4%}")
    print(f"{'─'*60}")
    if d["per_asset_contribution"]:
        print("  Per-asset Sharpe contribution:")
        for sym, contrib in d["per_asset_contribution"].items():
            print(f"    {sym:>15}: {contrib:>+8.4f}")
    print(f"{'='*60}\n")


def save_tearsheet(
    result: PortfolioBacktestResult,
    output_path: str | Path,
    study_name: str = "backtest",
    bars_per_year: int = 365 * 24,
    n_trials: int = 1,
) -> Path:
    """Save tearsheet as JSON.  Parent dirs created automatically."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    d = build_tearsheet(result, study_name=study_name, bars_per_year=bars_per_year, n_trials=n_trials)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, default=str)
    logger.info("Tearsheet saved: %s", path)
    return path

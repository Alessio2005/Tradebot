# src/tradebot/backtest/reporting.py
"""Backtest reporting helpers — tearsheet generation entry-point.

This module is intentionally thin; heavy rendering lives in apps/make_tearsheet.py.
Here we provide pure-Python data preparation utilities that can run in any context
(no matplotlib dependency at import time).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    annualized_return,
    annualized_vol,
    deflated_sharpe,
    max_drawdown,
    sortino_ratio,
)
from .tracks import PortfolioBacktestResult

__all__ = ["build_tearsheet_dict", "save_tearsheet_json"]


def build_tearsheet_dict(
    result: PortfolioBacktestResult,
    bars_per_year: int = 365 * 24,
    n_trials: int = 1,
) -> dict[str, Any]:
    """Flatten a ``PortfolioBacktestResult`` into a serialisable dict for tearsheet rendering.

    Parameters
    ----------
    result : backtest result.
    bars_per_year : calibration for annualisation.
    n_trials : total number of strategy variants evaluated (for DSR correction).

    Returns
    -------
    dict with flat scalar values + list-valued equity_curve / returns keys.
    """
    r = result.portfolio_returns.values.astype(np.float64)
    eq = result.equity_curve.values.astype(np.float64)

    _mdd_val, _peak_idx, _trough_idx = max_drawdown(eq)

    dsr = deflated_sharpe(
        sr_observed=result.sharpe,
        n_trials=max(n_trials, 1),
        n_obs=max(len(r), 5),
    )

    d: dict[str, Any] = {
        # Core scalars
        "sharpe": round(result.sharpe, 4),
        "sortino": round(sortino_ratio(r, bars_per_year=bars_per_year), 4),
        "calmar": round(result.calmar, 4),
        "max_drawdown": round(result.max_dd, 4),
        "total_return": round(result.total_return, 4),
        "deflated_sharpe": round(dsr, 4),
        "annualized_return": round(annualized_return(r, bars_per_year=bars_per_year), 4),
        "annualized_vol": round(annualized_vol(r, bars_per_year=bars_per_year), 4),
        "realized_vol": round(result.realized_vol, 4),
        "avg_gross_leverage": round(result.avg_gross_leverage, 4),
        "avg_net_leverage": round(result.avg_net_leverage, 4),
        "avg_corr": round(result.avg_corr, 4),
        "n_bars": result.n_bars,
        "n_dd_breaker_bars": result.n_dd_breaker_bars,
        "total_rebalance_cost": round(result.total_rebalance_cost, 6),
        "total_funding_cost": round(result.total_funding_cost, 6),
        "avg_turnover": round(result.avg_turnover, 6),
        # Bootstrap CI if available
        "bootstrap_sharpe_lo": round(result.bootstrap.get("sharpe_lo", float("nan")), 4),
        "bootstrap_sharpe_hi": round(result.bootstrap.get("sharpe_hi", float("nan")), 4),
        # Per-asset Sharpe contribution
        "per_asset_contribution": {
            k: round(v, 4) for k, v in result.per_asset_contribution.items()
        },
        # Time-series (as lists for JSON serialisation)
        "equity_curve": result.equity_curve.rename("equity").to_json(orient="split"),
        "portfolio_returns": result.portfolio_returns.rename("returns").to_json(orient="split"),
    }
    return d


def save_tearsheet_json(
    result: PortfolioBacktestResult,
    output_path: str | Path,
    bars_per_year: int = 365 * 24,
    n_trials: int = 1,
) -> Path:
    """Save tearsheet data to a JSON file for downstream rendering.

    Parameters
    ----------
    result : backtest result.
    output_path : where to write (parent dirs created automatically).
    bars_per_year : calibration for annualisation.
    n_trials : total strategy variants (for DSR).

    Returns
    -------
    Resolved ``Path`` of the written file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    d = build_tearsheet_dict(result, bars_per_year=bars_per_year, n_trials=n_trials)

    # Replace embedded JSON strings with parsed objects for a cleaner output.
    import json as _json
    for key in ("equity_curve", "portfolio_returns"):
        if key in d and isinstance(d[key], str):
            d[key] = _json.loads(d[key])

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, default=str)

    return path

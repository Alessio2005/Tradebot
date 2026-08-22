"""Methodological guards for the market-neutral book (CHIEF).

Covers the two Silent Killers that matter for this strategy:
  1. look-ahead: weights at date t must be invariant to FUTURE data.
  2. neutrality: target weights are dollar-neutral (sum≈0) and gross-normalised.
Plus a correct-accounting check: portfolio P&L uses simple returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from tradebot.alpha.neutral_book import NeutralBook, NeutralBookConfig

CACHE = _ROOT / "artefacts" / "broad_perp_daily_close.parquet"


def _panel():
    if not CACHE.exists():
        pytest.skip("broad_perp_daily_close.parquet not present")
    return pd.read_parquet(CACHE).resample("1D").last()


def test_dollar_neutral_and_gross():
    nb = NeutralBook()
    w = nb.target_weights(_panel())
    assert abs(w.sum()) < 1e-6, f"not dollar-neutral: sum={w.sum()}"
    assert abs(w.abs().sum() - 1.0) < 1e-6, f"gross != 1: {w.abs().sum()}"


def test_no_lookahead():
    """Weights at date T must not change when future rows are appended."""
    nb = NeutralBook()
    P = _panel()
    cut = len(P) - 30
    full, _ = nb.weight_history(P)
    truncated, _ = nb.weight_history(P.iloc[:cut])
    common = truncated.index.intersection(full.index)
    # compare the last 5 dates of the truncated run against the full run
    tail = common[-5:]
    diff = (full.loc[tail] - truncated.loc[tail]).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9, f"look-ahead detected: max diff {np.nanmax(diff)}"


def test_market_neutral_beta():
    """Strategy P&L beta to the equal-weight market must be ~0."""
    nb = NeutralBook()
    P = _panel()
    combined, _ = nb.weight_history(P)
    R = P / P.shift(1) - 1.0
    R = R[R.notna().sum(axis=1) >= 8]
    mkt = R.mean(axis=1)
    we = combined.shift(1).reindex(R.index).fillna(0.0)
    pnl = (we * R).sum(axis=1).dropna()
    beta = np.polyfit(mkt.reindex(pnl.index).fillna(0.0), pnl, 1)[0]
    assert abs(beta) < 0.20, f"not market-neutral: beta={beta:.3f}"

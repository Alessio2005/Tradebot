"""Development panel for the trend/carry research programme.

Everything here is truncated at DEV_END (the weekly spec's holdout split, 2026-06-24). The 60
bars after it are the one-look holdout and are never loaded by research code, only by
`research/final_holdout.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
DEV_END = pd.Timestamp("2026-06-23T00:00:00+00:00")  # last dev bar (split is 06-24)
SPLIT = pd.Timestamp("2026-06-24T00:00:00+00:00")


@dataclass(frozen=True)
class Panel:
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    turnover: pd.DataFrame      # USD volume
    funding: pd.DataFrame       # daily sum of 8h settlements, fraction of notional; long pays +
    oi: pd.DataFrame            # open interest (contracts), daily
    sigma_ann: pd.DataFrame     # repo EWMA(0.94) annualised vol, causal
    ret: pd.DataFrame           # close-to-close simple return

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index


def load_panel(until: pd.Timestamp = SPLIT) -> Panel:
    from tradebot.data.weekly_market import load_weekly_market

    full = load_weekly_market(ROOT, SYMS)
    m = full.truncate(until)

    def f(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: m.ohlcv[s][col] for s in SYMS})

    close = f("close")
    return Panel(
        close=close, open=f("open"), high=f("high"), low=f("low"), turnover=f("turnover"),
        funding=m.funding[SYMS].fillna(0.0).where(close.notna(), 0.0),
        oi=m.open_interest[SYMS], sigma_ann=m.sigma_annual[SYMS], ret=close.pct_change(),
    )

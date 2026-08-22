# src/tradebot/data/orderbook.py
"""L2 order-book snapshot utilities and imbalance features.

Handles:
  - Parsing raw L2 snapshots (list of [price, qty] bids/asks).
  - Computing standard order-book imbalance metrics.
  - Rolling aggregation over a sequence of snapshots.

All computations are lookahead-free: only past and current snapshots
are used when computing features for bar t.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "OrderBookSnapshot",
    "compute_book_imbalance",
    "compute_depth_weighted_midprice",
    "rolling_book_features",
]


@dataclass
class OrderBookSnapshot:
    """A single L2 order-book snapshot.

    Parameters
    ----------
    timestamp :
        UTC time of the snapshot.
    bids :
        Array of shape (n, 2) — [[price, qty], ...] sorted descending.
    asks :
        Array of shape (m, 2) — [[price, qty], ...] sorted ascending.
    """

    timestamp: pd.Timestamp
    bids: np.ndarray  # shape (n, 2): [[price, qty], ...]
    asks: np.ndarray  # shape (m, 2): [[price, qty], ...]

    def best_bid(self) -> float:
        return float(self.bids[0, 0]) if len(self.bids) > 0 else np.nan

    def best_ask(self) -> float:
        return float(self.asks[0, 0]) if len(self.asks) > 0 else np.nan

    def mid_price(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if np.isnan(bb) or np.isnan(ba):
            return np.nan
        return (bb + ba) / 2.0

    def spread(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if np.isnan(bb) or np.isnan(ba):
            return np.nan
        return ba - bb


def compute_book_imbalance(
    snapshot: OrderBookSnapshot,
    depth_levels: int = 5,
) -> float:
    """Volume-weighted order-book imbalance over the top ``depth_levels``.

    OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)

    Returns a value in [-1, +1]:
      +1 = all volume on bid side (bullish pressure)
      -1 = all volume on ask side (bearish pressure)
    """
    bid_vol = float(snapshot.bids[:depth_levels, 1].sum()) if len(snapshot.bids) > 0 else 0.0
    ask_vol = float(snapshot.asks[:depth_levels, 1].sum()) if len(snapshot.asks) > 0 else 0.0
    total   = bid_vol + ask_vol + 1e-9
    return (bid_vol - ask_vol) / total


def compute_depth_weighted_midprice(
    snapshot: OrderBookSnapshot,
    depth_levels: int = 5,
) -> float:
    """Volume-weighted mid-price using top ``depth_levels`` on each side.

    More robust to single-level spoofing than best-bid/ask midprice.
    """
    bids = snapshot.bids[:depth_levels]
    asks = snapshot.asks[:depth_levels]

    if len(bids) == 0 or len(asks) == 0:
        return snapshot.mid_price()

    bid_vwap = float(np.average(bids[:, 0], weights=bids[:, 1])) if bids[:, 1].sum() > 0 else float(bids[0, 0])
    ask_vwap = float(np.average(asks[:, 0], weights=asks[:, 1])) if asks[:, 1].sum() > 0 else float(asks[0, 0])
    return (bid_vwap + ask_vwap) / 2.0


def rolling_book_features(
    snapshots: list[OrderBookSnapshot],
    window: int = 20,
    depth_levels: int = 5,
) -> pd.DataFrame:
    """Convert a sequence of snapshots to a rolling-feature DataFrame.

    Columns
    -------
    book_imbalance       : OBI in [-1, +1] (rolling mean)
    depth_mid_price      : depth-weighted midprice
    spread_bps           : bid-ask spread in basis points
    obi_momentum         : rolling mean of sign(OBI_t - OBI_{t-1})

    Index = snapshot timestamps.
    """
    records = []
    for snap in snapshots:
        mid = snap.mid_price()
        spread_bps = (snap.spread() / mid * 10_000) if mid and mid > 0 else np.nan
        records.append({
            "timestamp":       snap.timestamp,
            "obi_raw":         compute_book_imbalance(snap, depth_levels),
            "depth_mid_price": compute_depth_weighted_midprice(snap, depth_levels),
            "spread_bps":      spread_bps,
        })

    df = pd.DataFrame(records).set_index("timestamp")
    if df.empty:
        return df

    # CHIEF AUDIT 2026-05-23 (FIX 8 / partial-window leak):
    # ``min_periods=1`` makes the first ``window-1`` rolling values reflect
    # only 1..window-1 samples, but the downstream consumer treats them as
    # if they were full-window averages.  This skews the warm-up bars and
    # — because the strategy may trade on them — leaks an undefined signal
    # into the first few hours of each backtest.  Require at least a
    # quarter-window of samples (capped at a minimum of 5) so partial
    # windows are emitted as NaN and dropped/filled by downstream callers.
    min_periods = max(5, window // 4)
    df["book_imbalance"] = df["obi_raw"].rolling(window, min_periods=min_periods).mean()
    df["obi_momentum"]   = (
        df["obi_raw"].diff().apply(np.sign)
        .rolling(window, min_periods=min_periods).mean()
    )
    df = df.drop(columns=["obi_raw"])
    return df

# src/tradebot/alpha/eq_xsmom.py
"""Wave 22a — equity cross-sectional momentum 12-1 (Jegadeesh-Titman 1993).

Prior: "Returns to Buying Winners and Selling Losers" (JF 1993): stocks
ranked on trailing 12-month return SKIPPING the most recent month continue
to outperform over 3-12 months. The skip-month avoids contamination by
short-term reversal (Jegadeesh 1990 — that premium is unit 22b, eq_strev).

Construction (literature-conform, FIXED — no sweeps, F12):
  score(t) = P(t-21) / P(t-252) - 1     (12-1 trailing return)
  monthly rebalance, long top 30% / short bottom 30%, dollar-neutral.

NOTE F8: crypto XSMOM is falsified (crypto is XS-reversal). Equities XSMOM
is a DIFFERENT, documented premium — that is exactly why this unit exists
in the equities sleeve and not in crypto.
"""
from __future__ import annotations

import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, run_xs_unit

__all__ = ["UNIT", "PRIOR", "signal_panel", "run"]

UNIT = "eq_xsmom_12_1"
PRIOR = "Jegadeesh & Titman (1993), J. Finance 48(1), 65-91"

_LOOKBACK = 252   # ~12 months of trading days
_SKIP = 21        # ~1 month skip


def signal_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum score per (date, symbol); row t uses closes <= t.

    Requires at least _LOOKBACK observations of history; earlier rows are
    NaN and therefore excluded from the cross-section by the harness.
    """
    return prices.shift(_SKIP) / prices.shift(_LOOKBACK) - 1.0


def run(
    prices: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    cost: CostModel = CostModel(),
) -> XSUnitResult:
    return run_xs_unit(
        prices=prices,
        signal_panel=signal_panel(prices),
        unit=UNIT,
        membership=membership,
        rebalance="ME",
        cost=cost,
        top_frac=0.3,
        config={"lookback": _LOOKBACK, "skip": _SKIP, "prior": PRIOR},
    )

# src/tradebot/alpha/eq_overnight.py
"""Wave 23c — overnight-return persistence (Lou-Polk-Skouras 2019).

Prior: "A tug of war: Overnight versus intraday expected returns" (JFE
2019): the overnight component of returns exhibits strong cross-sectional
PERSISTENCE (firms with high past overnight returns keep earning their
premium overnight), while intraday components tend to reverse — documented
explicitly INCLUDING large caps, which is why this unit follows the W22
finding that plain price premia don't survive in the S&P universe.

Construction (literature-conform, FIXED):
  overnight_ret(t) = open(t) / close(t-1) - 1          (known at close t)
  score(t) = sum of log overnight returns over the past 21 trading days
  monthly rebalance, long top 30% / short bottom 30%, dollar-neutral.

Causality: row t uses open(t) (09:30, hours before the close) and closes
<= t-1 — strictly <= t information. Guarded in tests/lookahead/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, run_xs_unit

__all__ = ["UNIT", "PRIOR", "signal_panel", "run"]

UNIT = "eq_overnight_1m"
PRIOR = "Lou, Polk & Skouras (2019), J. Fin. Econ. 134(1), 192-213"

_LOOKBACK = 21


def signal_panel(close: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
    """Trailing 21d cumulative overnight log-return; row t uses data <= t."""
    if not close.index.equals(open_.index):
        raise ValueError("close and open panels must share the same index")
    overnight = np.log(open_ / close.shift(1))
    # a glitched/missing open must not silently zero the whole window
    return overnight.rolling(_LOOKBACK, min_periods=_LOOKBACK).sum()


def run(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    cost: CostModel = CostModel(),
) -> XSUnitResult:
    return run_xs_unit(
        prices=close,
        signal_panel=signal_panel(close, open_),
        unit=UNIT,
        membership=membership,
        rebalance="ME",
        cost=cost,
        top_frac=0.3,
        config={"lookback": _LOOKBACK, "component": "overnight", "prior": PRIOR},
    )

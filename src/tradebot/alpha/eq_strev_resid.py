# src/tradebot/alpha/eq_strev_resid.py
"""Wave 24 — residual short-term reversal (Da-Liu-Schaumburg 2014).

Prior: "A Closer Look at the Short-Term Return Reversal" (Mgmt Sci 2014):
reversal profits concentrate in the RESIDUAL component of returns; the
factor-driven component does not revert (it drags noise into the raw
construct). Activated via the registered reopening clause on the archived
``eq_strev_1m`` — whose formal G4 showed the premium exists (alpha +3.1%/yr,
t=2.47 vs FF5+MOM) but is drowned by factor noise in the raw harvest.

Construction (literature-conform, FIXED):
  beta_i(t)  = rolling 252d market beta (EW universe return), data <= t
  resid_i(t) = r_i(t) − beta_i(t) · mkt(t)
  score(t)   = −sum(resid_i over the last 21 trading days)
  monthly rebalance, long top 30% / short bottom 30%, dollar-neutral.
"""
from __future__ import annotations

import pandas as pd

from tradebot.alpha.eq_lowvol import beta_panel
from tradebot.alpha.xs_unit import CostModel, XSUnitResult, run_xs_unit

__all__ = ["UNIT", "PRIOR", "signal_panel", "run"]

UNIT = "eq_strev_resid_1m"
PRIOR = "Da, Liu & Schaumburg (2014), Mgmt Science 60(3), 658-674"

_LOOKBACK = 21


def signal_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Negative trailing 21d cumulative RESIDUAL return; data <= t only."""
    rets = prices.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    beta = beta_panel(prices)
    resid = rets.sub(beta.mul(mkt, axis=0))
    return -resid.rolling(_LOOKBACK, min_periods=_LOOKBACK).sum()


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
        config={"lookback": _LOOKBACK, "residual": "mkt-beta-252d", "prior": PRIOR},
    )

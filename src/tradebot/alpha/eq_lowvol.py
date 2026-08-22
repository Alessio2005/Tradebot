# src/tradebot/alpha/eq_lowvol.py
"""Wave 22c — equity low-volatility / BAB-lite (Frazzini-Pedersen 2014).

Prior: "Betting Against Beta" (JFE 2014): low-beta/low-vol assets deliver
higher risk-adjusted returns than high-beta assets. Implemented here in the
volatility form (rank on trailing 252d vol) — the simple OHLCV-only variant;
the beta-ranked variant needs the same data and is NOT a separate trial
unless the simple form lives OOS first (§5.6).

G4 warning recorded up-front (Mandate §1.2 / stappenplan W22): this unit
must retain residual alpha AFTER the Ken French set INCLUDING the BAB
factor — otherwise it is factor loading, not alpha, and gets archived.

Construction v2 (literature-conform, FIXED):
  score(t) = -realised_vol_252(t)        (long low vol, short high vol)
  legs BETA-LEVERED to |beta| = 1 (the actual FP2014 construction) —
  v1 (dollar-neutral, unscaled) was structurally short-beta and bled
  -0.82 Sharpe of pure beta drag (W22 run 3, archived in the ledger).
  monthly rebalance, long top 30% / short bottom 30%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, run_xs_unit

__all__ = ["UNIT", "PRIOR", "signal_panel", "beta_panel", "run"]

UNIT = "eq_lowvol"
PRIOR = "Frazzini & Pedersen (2014), J. Fin. Econ. 111(1), 1-25"

_VOL_WINDOW = 252
_MIN_OBS = 126
_BETA_WINDOW = 252


def signal_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Negative trailing realised vol; row t uses closes <= t."""
    rets = prices.pct_change(fill_method=None)
    vol = rets.rolling(_VOL_WINDOW, min_periods=_MIN_OBS).std() * np.sqrt(252)
    return -vol


def beta_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Rolling 252d beta vs the equal-weight universe return; data <= t."""
    rets = prices.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    cov = rets.rolling(_BETA_WINDOW, min_periods=_MIN_OBS).cov(mkt)
    var = mkt.rolling(_BETA_WINDOW, min_periods=_MIN_OBS).var()
    return cov.div(var, axis=0)


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
        beta_panel=beta_panel(prices),
        config={"vol_window": _VOL_WINDOW, "min_obs": _MIN_OBS,
                "beta_window": _BETA_WINDOW, "construction": "BAB-v2-beta-levered",
                "prior": PRIOR},
    )

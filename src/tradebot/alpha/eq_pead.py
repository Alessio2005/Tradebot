# src/tradebot/alpha/eq_pead.py
"""Wave 23a — post-earnings-announcement drift via announcement returns.

Prior: Bernard & Thomas (1989); Chan, Jegadeesh & Lakonishok (1996) show
the drift is equally well captured by the ABNORMAL RETURN AROUND THE
ANNOUNCEMENT (no analyst estimates needed — estimates are not free data).

Anchor (PIT-conservative): the EDGAR acceptance timestamp of the 10-Q/10-K.
The 8-K press release usually precedes it, so part of the drift is already
consumed by our anchor — this BIASES THE UNIT AGAINST US (never toward us),
which is the acceptable direction (Mandate §1.3).

Construction (literature-conform, FIXED):
  ar3(e)   = sum over the 3 trading days FOLLOWING acceptance of
             (r_i − r_EW-market)            [days d+1..d+3, all <= t]
  score(t) = latest ar3 whose window is complete, valid 63 trading days
  monthly rebalance, long top 30% / short bottom 30%, dollar-neutral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, event_to_panel, run_xs_unit

__all__ = ["UNIT", "PRIOR", "announcement_events", "signal_panel", "run"]

UNIT = "eq_pead_ar3"
PRIOR = "Bernard-Thomas (1989) JAR 27; Chan-Jegadeesh-Lakonishok (1996) JF 51(5)"

_AR_DAYS = 3
_EXPIRY_BDAYS = 63


def announcement_events(
    prices: pd.DataFrame, filings: pd.DataFrame
) -> pd.DataFrame:
    """[symbol, asof_ts, value=ar3] — asof = close of day d+3 (window done).

    Causality: the score becomes available only AFTER the last return in
    its own window has printed; the harness then trades it at t+1.
    """
    fil = filings[filings["form"].isin(["10-Q", "10-K"])].copy()
    fil["asof_ts"] = pd.to_datetime(fil["asof_ts"], utc=True)
    rets = prices.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    abret = rets.sub(mkt, axis=0)
    idx = prices.index
    rows: list[dict] = []
    for sym, grp in fil.groupby("ticker"):
        if sym not in abret.columns:
            continue
        col = abret[sym].to_numpy()
        for acc in grp["asof_ts"]:
            # d = first trading day on/after acceptance; its close-to-close
            # return contains the announcement reaction (realised at close d)
            d = int(np.searchsorted(idx.values, np.datetime64(acc), side="left"))
            if d + _AR_DAYS - 1 >= len(idx) or d == 0:
                continue
            window = col[d : d + _AR_DAYS]
            if np.isnan(window).any():
                continue
            rows.append(
                {"symbol": sym, "asof_ts": idx[d + _AR_DAYS - 1],
                 "value": float(window.sum())}
            )
    return (
        pd.DataFrame(rows)
        .sort_values("asof_ts", kind="stable")
        .reset_index(drop=True)
    )


def signal_panel(prices: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    ev = announcement_events(prices, filings)
    return event_to_panel(ev, prices.index, prices.columns, _EXPIRY_BDAYS)


def run(
    prices: pd.DataFrame,
    filings: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    cost: CostModel = CostModel(),
) -> XSUnitResult:
    return run_xs_unit(
        prices=prices,
        signal_panel=signal_panel(prices, filings),
        unit=UNIT,
        membership=membership,
        rebalance="ME",
        cost=cost,
        top_frac=0.3,
        config={"anchor": "edgar-acceptance", "ar_days": _AR_DAYS,
                "expiry_bdays": _EXPIRY_BDAYS, "prior": PRIOR},
    )

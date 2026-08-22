# src/tradebot/alpha/eq_quality.py
"""Wave 23b — quality / gross profitability (Novy-Marx 2013), PIT EDGAR.

Prior: "The Other Side of Value: The Gross Profitability Premium" (JFE
2013): GP/A (gross profit / total assets) predicts returns with power
comparable to value, INCLUDING in large caps — the explicit reason this
unit still gets its shot after the price-premia archive (W22/24): it is a
genuinely different information source (F6-conform).

PIT discipline (the classic fundamentals lookahead, Mandate §7): a fiscal
year's GP/A becomes tradeable at the FIRST filing date that disclosed it
(asof_ts = filed + 1d from EDGAR), never at the fiscal period end. Values
expire after 400 trading days (a company that stops filing drops out).

Construction (literature-conform, FIXED):
  score = GP/A of the latest KNOWN annual (10-K) figures
  monthly rebalance, long top 30% / short bottom 30%, dollar-neutral.
Coverage note (G8): financials rarely report GrossProfit -> they drop out;
the coverage count is printed by the eval and recorded in the wave log.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, event_to_panel, run_xs_unit

__all__ = ["UNIT", "PRIOR", "gpa_events", "signal_panel", "run"]

UNIT = "eq_quality_gpa"
PRIOR = "Novy-Marx (2013), J. Fin. Econ. 108(1), 1-28"

_EXPIRY_BDAYS = 400


def gpa_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """PIT GP/A events: [symbol, asof_ts, value] from annual 10-K facts.

    First-filed value per (ticker, concept, period end) — amendments and
    re-disclosures in later filings never move the availability backwards.
    """
    f = fundamentals[fundamentals["form"].isin(["10-K", "10-K/A"])].copy()
    f["asof_ts"] = pd.to_datetime(f["asof_ts"], utc=True)
    f["event_ts"] = pd.to_datetime(f["event_ts"], utc=True)
    f = (
        f.sort_values("asof_ts", kind="stable")
        .groupby(["ticker", "concept", "event_ts"], as_index=False)
        .first()
    )
    gp = f[f["concept"] == "GrossProfit"][["ticker", "event_ts", "asof_ts", "value"]]
    at = f[f["concept"] == "Assets"][["ticker", "event_ts", "value"]]
    m = gp.merge(at, on=["ticker", "event_ts"], suffixes=("_gp", "_at"))
    m = m[m["value_at"] > 0]
    m["value"] = m["value_gp"] / m["value_at"]
    out = m[["ticker", "asof_ts", "value"]].rename(columns={"ticker": "symbol"})
    return out.sort_values("asof_ts", kind="stable").reset_index(drop=True)


def signal_panel(
    prices: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    ev = gpa_events(fundamentals)
    return event_to_panel(ev, prices.index, prices.columns, _EXPIRY_BDAYS)


def run(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    cost: CostModel = CostModel(),
) -> XSUnitResult:
    return run_xs_unit(
        prices=prices,
        signal_panel=signal_panel(prices, fundamentals),
        unit=UNIT,
        membership=membership,
        rebalance="ME",
        cost=cost,
        top_frac=0.3,
        config={"metric": "GrossProfit/Assets", "form": "10-K",
                "expiry_bdays": _EXPIRY_BDAYS, "prior": PRIOR},
    )

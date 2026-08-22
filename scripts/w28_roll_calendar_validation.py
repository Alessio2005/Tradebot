#!/usr/bin/env python
"""W28 — validate CALENDAR-derived NYMEX roll dates against the EIA slot data.

WHY THIS EXISTS
---------------
EIA publishes settlements by SLOT ("Contract 1..4"), not by contract. To turn
that into a return you must know which days the slots shift. Detecting the
shift from prices does not work: adjacent-slot spreads are small next to daily
noise, and a per-day proximity test fired 51-78 times a year against a true
rate of ~12, with a diffuse day-of-month distribution (i.e. it was finding
noise, not rolls).

Getting this wrong is not a rounding error. Natural gas averages -27.5%/yr of
carry, so mis-assigning its 12 rolls a year injects ~25%/yr of return that
nobody earned — which is exactly the +25.1%/yr contamination this program
already measured on the free continuous ``NG=F`` series.

So roll dates come from the published contract specifications, which are
calendar facts, and this script tests that rule against the data.

THE RULES (CME contract specs)
------------------------------
CL  WTI      : terminates 3 business days before the 25th calendar day of the
               month preceding delivery; if the 25th is not a business day,
               3 business days before the business day preceding the 25th.
NG  Henry Hub: terminates 3 business days before the 1st of the delivery month.
HO  ULSD     : terminates on the last business day of the month preceding delivery.
RB  RBOB     : terminates on the last business day of the month preceding delivery.

The panel's own index is used as the business-day calendar — self-consistent,
and it needs no external holiday file.

THE TEST
--------
On a true roll day slot k holds what slot k+1 held yesterday, so
``|F_k(t) - F_{k+1}(t-1)|`` should beat ``|F_k(t) - F_k(t-1)|``. That test only
has power when the curve is steep enough to see past daily noise, so it is run
on the steepest decile of months and compared against non-roll days.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.alpha.cm_carry import build_slot_panel  # noqa: E402

PANEL = "market_data_parquet/commodities/eia_term_structure.parquet"


def expiry_dates(idx: pd.DatetimeIndex, product: str) -> pd.DatetimeIndex:
    """Contract expiry dates implied by the CME spec, on this trading calendar."""
    idx = pd.DatetimeIndex(idx).sort_values()
    naive = idx.tz_convert(None)
    out: list[pd.Timestamp] = []

    months = pd.period_range(naive.min().to_period("M"), naive.max().to_period("M"), freq="M")
    for m in months:
        if product == "WTI":
            ref = pd.Timestamp(year=m.year, month=m.month, day=25)
            prior = naive[naive <= ref]
            if len(prior) < 4:
                continue
            b = len(prior) - 1                       # index of the 25th-or-before
            if naive[b] != ref:                      # 25th not a trading day
                pass                                 # already the preceding one
            j = b - 3
        elif product == "NG":
            first_next = (m + 1).to_timestamp()
            prior = naive[naive < first_next]
            if len(prior) < 3:
                continue
            j = len(prior) - 3
        elif product in ("HO", "RBOB"):
            in_month = naive[(naive.year == m.year) & (naive.month == m.month)]
            if len(in_month) == 0:
                continue
            j = int(np.searchsorted(naive.values, in_month[-1].to_datetime64()))
        else:
            raise ValueError(product)
        if 0 <= j < len(idx):
            out.append(idx[j])
    return pd.DatetimeIndex(sorted(set(out)))


def roll_days(idx: pd.DatetimeIndex, product: str) -> pd.DatetimeIndex:
    """The slots shift on the first trading day strictly AFTER expiry."""
    idx = pd.DatetimeIndex(idx).sort_values()
    exp = expiry_dates(idx, product)
    pos = np.searchsorted(idx.values, exp.values, side="right")
    pos = pos[pos < len(idx)]
    return idx[np.unique(pos)]


def evidence(slots: pd.DataFrame) -> pd.Series:
    """same_score - shift_score, in price units. Positive => the slots shifted.

    Absolute differences, not logs: WTI settled NEGATIVE in April 2020 and a
    log transform silently produces NaN there.
    """
    same = sum((slots[k] - slots[k].shift(1)).abs() for k in (1, 2, 3))
    shift = sum((slots[k] - slots[k + 1].shift(1)).abs() for k in (1, 2, 3))
    return (same - shift).rename("evidence")


def main() -> int:
    ts = pd.read_parquet(PANEL)
    print(f"{'prod':5s} {'rolls':>6s} {'/yr':>6s} {'domMed':>7s} {'dom15_25':>9s} "
          f"{'ev@roll':>9s} {'ev@other':>9s} {'AUC':>6s}")
    all_ok = True
    for p in ("WTI", "NG", "HO", "RBOB"):
        slots = build_slot_panel(ts, p)
        rolls = roll_days(slots.index, p)
        ev = evidence(slots)

        # power subset: steepest decile of |F1-F2| relative to price level
        steep = ((slots[1] - slots[2]).abs() / slots[1].abs().clip(lower=1e-6))
        thresh = steep.quantile(0.90)
        mask = steep >= thresh

        is_roll = pd.Series(False, index=slots.index)
        is_roll.loc[is_roll.index.isin(rolls)] = True

        a = ev[mask & is_roll].dropna()
        b = ev[mask & ~is_roll].dropna()
        # rank AUC: P(evidence at a roll > evidence at a non-roll)
        if len(a) and len(b):
            comb = np.concatenate([a.values, b.values])
            ranks = pd.Series(comb).rank().values
            auc = (ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        else:
            auc = float("nan")

        years = (slots.index[-1] - slots.index[0]).days / 365.25
        dom = pd.DatetimeIndex(rolls).day
        in_window = float(np.mean((dom >= 15) & (dom <= 25))) if p == "WTI" else float("nan")
        print(f"{p:5s} {len(rolls):>6d} {len(rolls)/years:>6.2f} {np.median(dom):>7.0f} "
              f"{in_window:>9.2f} {a.mean():>9.4f} {b.mean():>9.4f} {auc:>6.3f}")
        if not (10.0 <= len(rolls) / years <= 14.0):
            all_ok = False
            print(f"      !! {p}: {len(rolls)/years:.2f} rolls/yr is not ~12")
        if not (auc > 0.75):
            all_ok = False
            print(f"      !! {p}: AUC {auc:.3f} — calendar rule not confirmed by prices")

    print("\nVERDICT:", "calendar roll rule CONFIRMED" if all_ok else "NOT CONFIRMED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

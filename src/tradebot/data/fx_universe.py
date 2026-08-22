# src/tradebot/data/fx_universe.py
"""Wave 25 — G10 FX total-return panel + carry signal (PIT, FRED-based).

Construction:
  spot panel  : USD per 1 unit foreign (direction-normalised, G10_SPOT_SERIES)
  rate panel  : 3m rate per currency + USD (percent, PIT via asof_join with
                the per-series release lag from G10_RATE_SERIES)
  TR index    : tr_ret(t) = dlog(spot) + (i_fx − i_usd)/100/252
                — long-foreign total return: spot move + interest accrual
  carry signal: i_fx − i_usd, as KNOWN at t (asof-joined, lag respected)

The XS harness consumes the TR index as a price panel; signal row t is
acted on at t+1 (harness shift). Costs: majors ~0.5–2 bps half-spread.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.data.sources.fred import G10_RATE_SERIES, G10_SPOT_SERIES
from tradebot.utils.time import asof_join

__all__ = ["load_fx_panels", "build_fx_factors"]


def _panel_from_long(
    long: pd.DataFrame, ids: dict[str, str], bidx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Long PIT frame [series_id, event_ts, value, asof_ts] -> wide panel on
    ``bidx`` where each cell is the latest value KNOWN at that date."""
    out = {}
    base = pd.DataFrame(index=bidx)
    for ccy, sid in ids.items():
        sub = long[long["series_id"] == sid][["value", "asof_ts"]]
        if sub.empty:
            continue
        joined = asof_join(base, sub)
        out[ccy] = joined["value"]
    return pd.DataFrame(out, index=bidx)


def load_fx_panels(
    root: Path | str = "market_data_parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tr_index, carry_signal) panels, columns = G10 ex-USD.

    tr_index     : total-return price index per currency (start = 1.0)
    carry_signal : i_fx − i_usd in percent, PIT (NaN until both legs known)
    """
    root = Path(root)
    spots = pd.read_parquet(root / "fx" / "spots.parquet")
    rates = pd.read_parquet(root / "fx" / "rates.parquet")
    for df in (spots, rates):
        df["asof_ts"] = pd.to_datetime(df["asof_ts"], utc=True)

    start = max(spots["event_ts"].min(), rates["event_ts"].min())
    end = spots["event_ts"].max()
    bidx = pd.date_range(start, end, freq="B", tz="UTC")

    spot = _panel_from_long(
        spots, {c: s for c, (s, _) in G10_SPOT_SERIES.items()}, bidx
    )
    # direction-normalise to USD per 1 unit foreign
    for ccy, (_, direction) in G10_SPOT_SERIES.items():
        if ccy in spot.columns and direction == -1:
            spot[ccy] = 1.0 / spot[ccy]

    rate = _panel_from_long(
        rates, {c: s for c, (s, _) in G10_RATE_SERIES.items()}, bidx
    )
    if "USD" not in rate.columns:
        raise ValueError("USD rate series missing from rates.parquet")
    usd = rate.pop("USD")
    carry = rate.sub(usd, axis=0)  # percent points, PIT by construction

    common = [c for c in carry.columns if c in spot.columns]
    spot, carry = spot[common], carry[common]

    tr_ret = np.log(spot / spot.shift(1)) + carry.shift(1) / 100.0 / 252.0
    tr_index = np.exp(tr_ret.fillna(0.0).cumsum())
    tr_index = tr_index.where(spot.notna())
    return tr_index, carry


def build_fx_factors(
    tr_index: pd.DataFrame,
    carry_signal: pd.DataFrame,
    trend_lookback: int = 252,
    min_names: int = 4,
) -> pd.DataFrame:
    """G4 FX factorset (DOLLAR, CARRY, TREND) as daily GROSS factor returns.

    Canonical constructions, weights strictly from data <= t-1 (R-1):
      DOLLAR : equal-weight long all foreign vs USD (Lustig-Roussanov-
               Verdelhan 2011) — the level factor.
      CARRY  : rank-weighted long high-carry / short low-carry, dollar-
               neutral, gross 1 (Koijen et al. 2018 rank weights — a
               DIFFERENT canonical construction than the tercile unit, so
               the unit-vs-factor regression is not the degenerate y~y).
      TREND  : sign of trailing 12m TR per currency, equal-weight scaled
               to gross 1 (Moskowitz-Ooi-Pedersen 2012).
    Factors are gross of costs — they are G4 benchmarks, not tradables.
    Rows with fewer than ``min_names`` valid names are NaN, never a
    2-currency "factor".
    """
    tr_ret = tr_index.pct_change(fill_method=None)
    dollar = tr_ret.mean(axis=1)

    sig = carry_signal.shift(1)
    rank = sig.rank(axis=1)
    w_c = rank.sub(rank.mean(axis=1), axis=0)
    w_c = w_c.div(w_c.abs().sum(axis=1), axis=0)
    carry_f = (w_c * tr_ret).sum(axis=1, min_count=1)
    carry_f[sig.notna().sum(axis=1) < min_names] = np.nan

    mom = tr_index.shift(1) / tr_index.shift(trend_lookback + 1) - 1.0
    w_t = np.sign(mom)
    w_t = w_t.div(w_t.abs().sum(axis=1), axis=0)
    trend_f = (w_t * tr_ret).sum(axis=1, min_count=1)
    trend_f[mom.notna().sum(axis=1) < min_names] = np.nan

    return pd.DataFrame({"DOLLAR": dollar, "CARRY": carry_f, "TREND": trend_f})

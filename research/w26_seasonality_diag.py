"""w26_seasonality_diag.py — Wave 26 step 1: MEASURE crypto intraday seasonality.

Mandate §5.1: hour-of-day / day-of-week effects on info-bars, horizon >=
minutes, NO HFT. This script is the diagnostic that precedes any unit build
(wave protocol: a unit is only constructed if the gross effect clears the
honest cost bar BY CONSTRUCTION — 10 bps/side taker, F2 lesson).

Priors (cited in wave log): Eross, McGroarty, Urquhart & Wood (2019) —
intraday bitcoin seasonality; Baur, Cahill, Godfrey & Liu (2019) — crypto
trading around the clock; practitioner-documented drift around the 8h
funding timestamps (00/08/16 UTC) — clock-time mechanism, distinct from
the F1-falsified funding-VALUE signal.

Outputs artefacts/crypto_hourly.parquet (close panel, UTC hourly) for the
unit build, and prints per-hour / per-DOW / funding-window diagnostics
with per-year stability.

Run:  python research/w26_seasonality_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]
CACHE = ROOT / "artefacts" / "crypto_hourly.parquet"


def build_hourly() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    cols = {}
    for pair in PAIRS:
        pdir = ROOT / "market_data_parquet" / pair
        frames = []
        for f in sorted(pdir.rglob(f"{pair}_*.parquet")):
            df = pd.read_parquet(f, columns=["timestamp", "close"])
            s = df.set_index("timestamp")["close"].resample("1h").last()
            frames.append(s)
        if frames:
            cols[pair] = pd.concat(frames).sort_index()
            print(f"  {pair}: {len(cols[pair])} hourly bars "
                  f"({cols[pair].index[0]} -> {cols[pair].index[-1]})")
    panel = pd.DataFrame(cols)
    panel.to_parquet(CACHE)
    return panel


def main() -> int:
    print("=== building/loading hourly close panel ===")
    close = build_hourly()
    rets = np.log(close / close.shift(1))
    # equal-weight market hourly return (the seasonal effect is market-wide)
    mkt = rets.mean(axis=1)
    mkt = mkt.dropna()
    print(f"panel: {close.shape}, mkt hours {len(mkt)}")

    print("\n=== HOUR-OF-DAY (UTC) — mean bps/hour, t-stat, per-year sign ===")
    years = sorted(mkt.index.year.unique())
    hod = mkt.groupby(mkt.index.hour)
    rows = []
    for h, grp in hod:
        t = grp.mean() / grp.std() * np.sqrt(len(grp))
        per_year = [
            np.sign(mkt[(mkt.index.hour == h) & (mkt.index.year == y)].mean())
            for y in years
        ]
        sign_str = "".join("+" if s > 0 else "-" for s in per_year)
        rows.append((h, grp.mean() * 1e4, t, sign_str))
        print(f"  h{h:02d}  {grp.mean() * 1e4:+6.2f} bps  t={t:+5.2f}  years[{sign_str}]")

    print("\n=== DAY-OF-WEEK — mean bps/day (close-to-close), t, per-year sign ===")
    daily = mkt.resample("1D").sum()
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        grp = daily[daily.index.dayofweek == d].dropna()
        t = grp.mean() / grp.std() * np.sqrt(len(grp))
        per_year = [
            np.sign(grp[grp.index.year == y].mean()) for y in years
        ]
        sign_str = "".join("+" if s > 0 else "-" for s in per_year)
        print(f"  {name}  {grp.mean() * 1e4:+7.2f} bps  t={t:+5.2f}  years[{sign_str}]")

    print("\n=== FUNDING WINDOW — hour before vs after 00/08/16 UTC ===")
    pre = mkt[mkt.index.hour.isin([23, 7, 15])]
    post = mkt[mkt.index.hour.isin([0, 8, 16])]
    for label, grp in (("pre-funding ", pre), ("post-funding", post)):
        t = grp.mean() / grp.std() * np.sqrt(len(grp))
        print(f"  {label}  {grp.mean() * 1e4:+6.2f} bps/h  t={t:+5.2f}  n={len(grp)}")

    # honest cost bar: one round-trip/day at 10 bps/side = 20 bps/day.
    # report the best contiguous long-window drift vs that bar
    print("\n=== COST BAR CHECK — best contiguous hour-window drift ===")
    hour_mu = {h: grp.mean() * 1e4 for h, grp in hod}
    best = None
    for start in range(24):
        run = 0.0
        for ln in range(1, 13):
            run += hour_mu[(start + ln - 1) % 24]
            if best is None or run > best[2]:
                best = (start, ln, run)
    s, ln, bps = best
    print(f"  best window: h{s:02d}+{ln}h = {bps:+.2f} bps/day gross "
          f"vs 20 bps/day round-trip taker cost -> "
          f"{'CLEARS' if bps > 20 else 'BELOW'} bar")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""fetch_funding_universe.py — free, full-history funding rates (Binance) for the
broad perp universe. Funding = price of leverage / crowded positioning = a genuine
non-price information source. 8h funding, paginated to full history.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artefacts" / "funding_universe.parquet"
# names from the WIDE daily panel (already established, multi-regime history)
PANEL = pd.read_parquet(ROOT / "artefacts/broad_perp_daily_close_WIDE.parquet")
SYMS = list(PANEL.columns)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=25))


def fetch_funding(sym):
    rows, start = [], int(pd.Timestamp("2021-06-01", tz="UTC").timestamp() * 1000)
    for _ in range(30):
        u = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}"
             f"&startTime={start}&limit=1000")
        d = _get(u)
        if not d:
            break
        rows += [(x["fundingTime"], float(x["fundingRate"])) for x in d]
        if len(d) < 1000:
            break
        start = d[-1]["fundingTime"] + 1
        time.sleep(0.02)
    if not rows:
        return None
    s = pd.Series({pd.to_datetime(t, unit="ms", utc=True): r for t, r in rows}).sort_index()
    return s[~s.index.duplicated()]


def main():
    print(f"Fetching funding for {len(SYMS)} symbols...", flush=True)
    cols = {}
    for i, sym in enumerate(SYMS):
        try:
            s = fetch_funding(sym)
            if s is not None and len(s) > 1000:
                # daily sum of 8h funding
                cols[sym] = s.resample("1D").sum()
        except Exception as e:
            print(f"  {sym} fail {str(e)[:40]}", flush=True)
        if i % 20 == 0:
            print(f"  {i}/{len(SYMS)} ({len(cols)} kept)", flush=True)
        time.sleep(0.02)
    F = pd.DataFrame(cols).sort_index()
    F.to_parquet(CACHE)
    print(f"SAVED {CACHE.name}: {F.shape}  range {F.index.min().date()}..{F.index.max().date()}", flush=True)


if __name__ == "__main__":
    main()

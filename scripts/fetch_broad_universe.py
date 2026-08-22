"""fetch_broad_universe.py — wider perp panel for the breadth test (Wave 2).

Fetches daily closes for ALL currently-trading USDT perps onboarded before
2022-07-01 (>=~2yr multi-regime history), to test whether cross-sectional
Sharpe scales with breadth beyond the 75-name panel. Survivorship caveat stands
(delisted symbols unavailable from the live klines endpoint).
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artefacts" / "broad_perp_daily_close_WIDE.parquet"
START_MS = int(pd.Timestamp("2021-06-01", tz="UTC").timestamp() * 1000)
ONBOARD_CUT = pd.Timestamp("2022-07-01").timestamp()


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def universe():
    info = _get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    out = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("onboardDate", 0) / 1000 < ONBOARD_CUT and s["symbol"] != "BTCDOMUSDT":
            out.append(s["symbol"])
    return out


def fetch_daily_close(sym):
    rows, start = [], START_MS
    for _ in range(12):
        d = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d&startTime={start}&limit=1000")
        if not d:
            break
        rows += [(k[0], float(k[4])) for k in d]
        if len(d) < 1000:
            break
        start = d[-1][0] + 86400000
    if not rows:
        return None
    s = pd.Series({pd.to_datetime(t, unit="ms", utc=True): c for t, c in rows}).sort_index()
    return s[~s.index.duplicated()]


def main():
    syms = universe()
    print(f"Universe (onboard<2022-07, trading): {len(syms)} symbols", flush=True)
    cols = {}
    for i, sym in enumerate(syms):
        try:
            s = fetch_daily_close(sym)
            if s is not None and len(s) > 700:
                cols[sym] = s
        except Exception as e:
            print(f"  {sym} fail: {str(e)[:40]}", flush=True)
        if i % 25 == 0:
            print(f"  {i}/{len(syms)} ({len(cols)} kept)", flush=True)
        time.sleep(0.04)
    P = pd.DataFrame(cols).sort_index()
    P.to_parquet(CACHE)
    print(f"SAVED {CACHE.name}: {P.shape}", flush=True)


if __name__ == "__main__":
    main()

"""fetch_bybit_spot.py — actual Bybit spot daily closes (the live-tradeable venue).

Bybit v5 market data. Fetches daily closes for all trading USDT spot pairs with
>=600 days history, paginating backward via the `end` cursor.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artefacts" / "bybit_spot_daily_close.parquet"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=25))


def universe():
    d = _get("https://api.bybit.com/v5/market/instruments-info?category=spot")
    return [x["symbol"] for x in d["result"]["list"]
            if x["symbol"].endswith("USDT") and x.get("status") == "Trading"]


def fetch_daily(sym):
    out = {}; end = int(pd.Timestamp.utcnow().timestamp() * 1000)
    for _ in range(8):
        u = (f"https://api.bybit.com/v5/market/kline?category=spot&symbol={sym}"
             f"&interval=D&limit=1000&end={end}")
        rows = _get(u)["result"]["list"]
        if not rows:
            break
        for r in rows:
            out[pd.to_datetime(int(r[0]), unit="ms", utc=True)] = float(r[4])
        end = int(rows[-1][0]) - 86400000
        if len(rows) < 1000:
            break
        time.sleep(0.03)
    if not out:
        return None
    return pd.Series(out).sort_index()


def main():
    syms = universe()
    print(f"Bybit spot USDT trading: {len(syms)}", flush=True)
    cols = {}
    for i, s in enumerate(syms):
        try:
            ser = fetch_daily(s)
            if ser is not None and len(ser) > 600:
                cols[s] = ser
        except Exception as e:
            print(f"  {s} fail {str(e)[:40]}", flush=True)
        if i % 30 == 0:
            print(f"  {i}/{len(syms)} ({len(cols)} kept)", flush=True)
        time.sleep(0.03)
    P = pd.DataFrame(cols).sort_index()
    P.to_parquet(CACHE)
    print(f"SAVED {CACHE.name}: {P.shape}  range {P.index.min().date()}..{P.index.max().date()}", flush=True)


if __name__ == "__main__":
    main()

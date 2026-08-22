"""fetch_broad_ohlcv.py — daily OHLCV panel for the 70x2 breadth rework (Wave 14).

Unlike fetch_broad_universe.py (close-only), this fetches full daily OHLCV for all
currently-trading USDT perps onboarded before 2022-07-01 (>=~4yr multi-regime
history), so the directional ML book can use range/vol features across breadth.
Survivorship caveat stands (delisted symbols unavailable from live klines).

Output: artefacts/broad_perp_ohlcv.parquet — tidy long frame
[date, symbol, open, high, low, close, volume].
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
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


def fetch_ohlcv(sym):
    rows, start = [], START_MS
    for _ in range(12):
        d = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d&startTime={start}&limit=1000")
        if not d:
            break
        rows += [(k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in d]
        if len(d) < 1000:
            break
        start = d[-1][0] + 86400000
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.drop(columns="t").drop_duplicates("date").set_index("date").sort_index()
    return df


def main():
    syms = universe()
    print(f"Universe (onboard<2022-07, trading): {len(syms)} symbols", flush=True)
    frames = []
    for i, sym in enumerate(syms):
        try:
            df = fetch_ohlcv(sym)
            if df is not None and len(df) > 700:
                df = df.reset_index()
                df["symbol"] = sym
                frames.append(df)
        except Exception as e:
            print(f"  {sym} fail: {str(e)[:40]}", flush=True)
        if i % 25 == 0:
            print(f"  {i}/{len(syms)} ({len(frames)} kept)", flush=True)
        time.sleep(0.04)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[["date", "symbol", "open", "high", "low", "close", "volume"]]
    panel.to_parquet(CACHE)
    print(f"SAVED {CACHE.name}: {panel.shape}, {panel.symbol.nunique()} symbols, "
          f"{panel.date.min().date()}..{panel.date.max().date()}", flush=True)


if __name__ == "__main__":
    main()

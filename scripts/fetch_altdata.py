"""fetch_altdata.py — free market-wide alt-data panel (Wave 16).

Fetches genuinely NEW free data the prior waves never used, to raise IC via better
features (the user's point: the feature set was NOT maxed). All sources free, no key:
  - Fear & Greed sentiment      (alternative.me)
  - DeFi TVL + stablecoin supply (DefiLlama)            -> capital-flow regime
  - BTC on-chain: active addrs, tx count, hashrate (blockchain.info)
  - Deribit DVOL implied vol     (BTC, ETH)             -> options/vol-risk regime
  - FRED macro: broad USD, 10y, 2y (best-effort)        -> macro liquidity regime

Output: artefacts/altdata_macro.parquet (daily UTC panel; lag in feature build).
"""
from __future__ import annotations
import json, time, io, csv, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artefacts" / "altdata_macro.parquet"


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _daily(ts_val_pairs, name):
    s = pd.Series({pd.to_datetime(int(t), unit="s", utc=True).normalize(): float(v)
                   for t, v in ts_val_pairs if v is not None})
    return s[~s.index.duplicated()].sort_index().rename(name)


def fear_greed():
    d = json.loads(_get("https://api.alternative.me/fng/?limit=0&format=json"))["data"]
    return _daily([(x["timestamp"], x["value"]) for x in d], "fng")


def defi_tvl():
    d = json.loads(_get("https://api.llama.fi/v2/historicalChainTvl"))
    return _daily([(x["date"], x["tvl"]) for x in d], "defi_tvl")


def stablecoins():
    d = json.loads(_get("https://stablecoins.llama.fi/stablecoincharts/all"))
    return _daily([(x["date"], x["totalCirculating"]["peggedUSD"]) for x in d], "stbl_supply")


def bc_chart(slug, name):
    d = json.loads(_get(f"https://api.blockchain.info/charts/{slug}?timespan=all&format=json"))
    return _daily([(p["x"], p["y"]) for p in d["values"]], name)


def deribit_dvol(ccy):
    rows, end = [], int(time.time() * 1000)
    start0 = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
    cur = start0
    while cur < end:
        nxt = min(cur + 86400000 * 900, end)
        u = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={ccy}"
             f"&start_timestamp={cur}&end_timestamp={nxt}&resolution=86400")
        try:
            d = json.loads(_get(u))["result"]["data"]
            rows += [(int(r[0] / 1000), r[4]) for r in d]   # close
        except Exception:
            pass
        cur = nxt
        time.sleep(0.05)
    return _daily(rows, f"dvol_{ccy.lower()}")


def fred(series, name):
    raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=45).decode()
    rd = list(csv.reader(io.StringIO(raw)))
    out = {}
    for row in rd[1:]:
        if len(row) == 2 and row[1] not in (".", ""):
            out[pd.Timestamp(row[0], tz="UTC")] = float(row[1])
    return pd.Series(out, name=name).sort_index()


def main():
    cols = []
    for fn, label in [(fear_greed, "fng"), (defi_tvl, "defi_tvl"), (stablecoins, "stbl"),
                      (lambda: bc_chart("n-unique-addresses", "btc_active_addr"), "active_addr"),
                      (lambda: bc_chart("n-transactions", "btc_txcount"), "txcount"),
                      (lambda: bc_chart("hash-rate", "btc_hashrate"), "hashrate"),
                      (lambda: deribit_dvol("BTC"), "dvol_btc"),
                      (lambda: deribit_dvol("ETH"), "dvol_eth")]:
        try:
            s = fn(); cols.append(s); print(f"  OK {label:14s} n={len(s)} {s.index.min().date()}..{s.index.max().date()}", flush=True)
        except Exception as e:
            print(f"  FAIL {label}: {str(e)[:50]}", flush=True)
    for sid, nm in [("DTWEXBGS", "usd_broad"), ("DGS10", "ust10y"), ("DGS2", "ust2y")]:
        try:
            s = fred(sid, nm); cols.append(s); print(f"  OK {nm:14s} (FRED) n={len(s)}", flush=True)
        except Exception as e:
            print(f"  FAIL FRED {nm}: {str(e)[:40]}", flush=True)
    panel = pd.concat(cols, axis=1).sort_index()
    panel = panel[panel.index >= "2020-12-31"].ffill()
    panel.to_parquet(OUT)
    print(f"\nSAVED {OUT.name}: {panel.shape}  cols={list(panel.columns)}", flush=True)


if __name__ == "__main__":
    main()

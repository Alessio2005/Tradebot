"""combined_alpha_book.py — CHIEF step 1: combine the genuine, correctly-measured
edges into one book and measure combined Sharpe vs buy&hold.

Sleeves (all SIMPLE-return P&L, 6 bps costs — no log bug):
  1. TREND  : BTC+ETH multi-MA{50,100,200} trend-following, long/flat (momentum).
  2. DVOL   : Deribit BTC implied-vol z-score timer, long/flat (contrarian fear).
  3. CARRY  : 5-asset cross-sectional funding carry, dollar-neutral (risk premium).
Combined by causal inverse-vol risk-parity (no tuning). Goal: robust Sharpe
~1.0-1.3 that beats buy&hold with roughly halved drawdown.
"""
from __future__ import annotations
import json, urllib.request, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
COST = 6.0


def _get(u):
    r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(r, timeout=25))


def sh(n):
    return n.mean() / n.std() * np.sqrt(365) if n.std() > 0 else 0.0


def dd(n):
    e = (1 + n).cumprod()
    return (e / e.cummax() - 1).min()


def trend_sleeve(perp):
    out = []
    for s in ["BTCUSDT", "ETHUSDT"]:
        px = perp[s].dropna(); r = px / px.shift(1) - 1
        for N in (50, 100, 200):
            w = (px > px.rolling(N).mean()).astype(float).shift(1)
            out.append((w * r - (w - w.shift(1)).abs() * COST / 1e4))
    return pd.concat(out, axis=1).mean(axis=1).rename("trend")


def dvol_sleeve(perp):
    e = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp() * 1000)
    cs = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000); rows = []
    while cs < e:
        ce = min(cs + 200 * 86400000, e)
        d = _get(f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={cs}&end_timestamp={ce}&resolution=86400")["result"]["data"]
        if not d:
            break
        rows += [(x[0], x[4]) for x in d]; cs = ce + 86400000
    dv = pd.Series({pd.to_datetime(t, unit="ms", utc=True): v for t, v in rows}).sort_index().resample("1D").last()
    z = ((dv - dv.rolling(90).mean()) / dv.rolling(90).std()).shift(1)
    px = perp["BTCUSDT"].dropna(); r = px / px.shift(1) - 1
    w = z.reindex(r.index).clip(0, 1.5)
    return (w * r - (w - w.shift(1)).abs() * COST / 1e4).dropna().rename("dvol")


def carry_sleeve():
    assets = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
    rets, fund = {}, {}
    for a in assets:
        df = pd.read_parquet(ROOT / "artefacts/features" / f"{a}.parquet", columns=["close", "fundingRate"]).sort_index()
        c = df["close"].resample("1D").last()
        rets[a] = c / c.shift(1) - 1                      # SIMPLE
        fund[a] = df["fundingRate"].resample("8h").last().resample("1D").sum()
    R = pd.DataFrame(rets).dropna(how="all"); F = pd.DataFrame(fund).reindex(R.index).fillna(0.0)
    car = -F.rolling(3).mean()
    z = car.sub(car.mean(axis=1), axis=0).div(car.std(axis=1).replace(0, np.nan), axis=0)
    w = z.sub(z.mean(axis=1), axis=0); w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    we = w.shift(1).fillna(0.0)
    return ((we * R).sum(axis=1) - (we * F).sum(axis=1) - (we - we.shift(1).fillna(0)).abs().sum(axis=1) * COST / 1e4).rename("carry")


def main():
    perp = pd.read_parquet(ROOT / "artefacts/broad_perp_daily_close.parquet").resample("1D").last()
    sleeves = pd.concat([trend_sleeve(perp), dvol_sleeve(perp), carry_sleeve()], axis=1).dropna()
    print(f"window: {sleeves.index.min().date()}..{sleeves.index.max().date()} | {len(sleeves)}d\n")
    print("Per-sleeve (simple, 6bps):")
    for c in sleeves.columns:
        print(f"  {c:7s} Sharpe={sh(sleeves[c]):+.2f} ann={((1+sleeves[c]).prod()**(365/len(sleeves))-1)*100:+.0f}% DD={dd(sleeves[c])*100:.0f}%")
    print("\nCorrelation matrix:")
    print(sleeves.corr().round(3).to_string())

    # causal inverse-vol risk-parity
    vol = sleeves.rolling(252, min_periods=60).std().shift(1)
    rp = (1 / vol).div((1 / vol).sum(axis=1), axis=0).fillna(1 / sleeves.shape[1])
    combo = (rp * sleeves).sum(axis=1).dropna()

    btc = (perp["BTCUSDT"] / perp["BTCUSDT"].shift(1) - 1).reindex(combo.index)
    beta = np.polyfit(btc.fillna(0), combo, 1)[0]
    py = [(1 + g).prod() - 1 for _, g in combo.groupby(combo.index.year)]
    line = " ".join(f"{y}:{p*100:+.0f}%" for (y, _), p in zip(combo.groupby(combo.index.year), py))
    print("\n=== COMBINED BOOK (risk-parity) ===")
    print(f"  Sharpe={sh(combo):+.2f}  ann={((1+combo).prod()**(365/len(combo))-1)*100:+.0f}%  MaxDD={dd(combo)*100:.0f}%  "
          f"beta_BTC={beta:+.2f}  t-stat={sh(combo)*np.sqrt(len(combo)/365):.2f}  neg={sum(p<0 for p in py)}/{len(py)}")
    print(f"  per-jaar: {line}")
    print(f"  vs BTC buy&hold: Sharpe={sh(btc):+.2f} MaxDD={dd(btc)*100:.0f}%")


if __name__ == "__main__":
    main()
